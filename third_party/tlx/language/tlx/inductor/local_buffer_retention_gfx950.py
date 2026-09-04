"""gfx950 TLX local-buffer retention integration for TorchInductor."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import sympy
import torch
from torch._inductor import config
from torch._inductor.codegen.common import CSEVariable, StoreMode
from torch._inductor.codegen.simd_kernel_features import (
    DisableReduction,
    EnableReduction,
    NodeScheduleMarker,
    SIMDKernelFeatures,
)
from torch._inductor.codegen.triton import (
    FixedTritonConfig,
    TritonCSEVariable,
    TritonKernel,
    triton_type,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer
from torch._inductor.scheduler import BaseSchedulerNode, SchedulerNode
from torch._inductor.utils import get_dtype_size, IndentedBuffer
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


@dataclasses.dataclass(frozen=True)
class LocalBufferRetentionSpec:
    """One global buffer access interval that can be backed by CTA-local LDS."""

    name: str
    dtype: torch.dtype
    element_count: int
    padded_element_count: int
    store_phase: int
    load_phases: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class LocalBufferRetentionPlan:
    """Structured scheduler-to-codegen contract for local buffer retention."""

    buffers: tuple[LocalBufferRetentionSpec, ...]
    reduction_numel: int
    reduction_block: int
    num_warps: int
    waves_per_eu: int

    @property
    def total_bytes(self) -> int:
        return sum(
            spec.padded_element_count * get_dtype_size(spec.dtype)
            for spec in self.buffers
        )


class LocalBufferRetention:
    """Find cross-phase values that can stay in LDS instead of round-tripping HBM."""

    _MAX_LOCAL_BYTES = 32 * 1024

    @staticmethod
    def _is_enabled() -> bool:
        if config.triton.tlx_mode != "allow" or torch.version.hip is None:
            return False
        try:
            properties = torch.cuda.get_device_properties(0)
        except (AssertionError, RuntimeError):
            return False
        return "gfx950" in getattr(properties, "gcnArchName", "")

    @staticmethod
    def _next_power_of_2(value: int) -> int:
        return 1 << (value - 1).bit_length()

    @staticmethod
    def _phase_accesses(
        node_schedule: Sequence[object],
    ) -> tuple[
        dict[str, dict[int, list[MemoryDep]]],
        dict[str, dict[int, list[MemoryDep]]],
    ]:
        reads: dict[str, dict[int, list[MemoryDep]]] = defaultdict(
            lambda: defaultdict(list)
        )
        writes: dict[str, dict[int, list[MemoryDep]]] = defaultdict(
            lambda: defaultdict(list)
        )
        phase = 0
        for item in node_schedule:
            if item is DisableReduction or item is EnableReduction:
                phase += 1
                continue
            if not isinstance(item, BaseSchedulerNode):
                continue
            for dep in item.read_writes.reads:
                if isinstance(dep, MemoryDep):
                    reads[dep.name][phase].append(dep.simplify_with_ranges())
            for dep in item.read_writes.writes:
                if isinstance(dep, MemoryDep) and dep.mode is None:
                    writes[dep.name][phase].append(dep.simplify_with_ranges())
        return reads, writes

    @staticmethod
    def _matching_contiguous_accesses(
        stores: Sequence[MemoryDep], loads: Sequence[MemoryDep]
    ) -> bool:
        if not stores or not loads:
            return False
        normalized_stores = [dep.normalize() for dep in stores]
        normalized_loads = [dep.normalize() for dep in loads]
        reference = normalized_stores[0]
        return all(
            dep.mode is None
            and dep.is_contiguous()
            and dep.index == reference.index
            and dep.size == reference.size
            for dep in (*normalized_stores, *normalized_loads)
        )

    @staticmethod
    def _can_elide_global_store(
        name: str,
        store_phase: int,
        load_phases: Sequence[int],
        writes: dict[str, dict[int, list[MemoryDep]]],
        fused_node_names: OrderedSet[str],
    ) -> bool:
        if any(
            phase >= max(load_phases) for phase in writes[name] if phase > store_phase
        ):
            return True

        scheduler = V.graph.scheduler
        return bool(
            scheduler
            and scheduler.can_buffer_be_removed_through_fusion(name, fused_node_names)
        )

    @classmethod
    def plan_for(
        cls, node_schedule: Sequence[object]
    ) -> LocalBufferRetentionPlan | None:
        if not cls._is_enabled():
            return None

        scheduled_nodes = list(NodeScheduleMarker.only_nodes(node_schedule))
        if not scheduled_nodes or DisableReduction not in node_schedule:
            return None
        if any(
            node.get_device() is None or node.get_device().type != "cuda"
            for node in scheduled_nodes
        ):
            return None

        reductions: list[SchedulerNode] = []
        for scheduled_node in scheduled_nodes:
            for node in scheduled_node.get_nodes():
                if not node.is_reduction():
                    continue
                if not isinstance(node, SchedulerNode) or not isinstance(
                    node.node, ComputedBuffer
                ):
                    return None
                if node.has_strict_reduction():
                    return None
                reductions.append(node)

        if not reductions:
            return None

        first_numel, first_rnumel = reductions[0].group[1]
        for reduction in reductions[1:]:
            _, (numel, rnumel) = reduction.group
            if not (
                V.graph.sizevars.statically_known_equals(first_numel, numel)
                and V.graph.sizevars.statically_known_equals(first_rnumel, rnumel)
            ):
                return None

        reduction_numel = V.graph.sizevars.simplify(first_rnumel)
        if not isinstance(reduction_numel, (int, sympy.Integer)):
            return None
        reduction_numel = int(reduction_numel)
        if reduction_numel <= 1:
            return None

        reads, writes = cls._phase_accesses(node_schedule)
        fused_node_names = OrderedSet(
            name for node in scheduled_nodes for name in node.get_operation_names()
        )
        padded_numel = cls._next_power_of_2(reduction_numel)
        specs: list[LocalBufferRetentionSpec] = []
        used_bytes = 0

        for name in sorted(writes.keys() & reads.keys()):
            write_phases = sorted(writes[name])
            read_phases = sorted(reads[name])
            for store_phase in write_phases:
                if store_phase % 2 != 0:
                    continue
                later_reads = tuple(
                    phase for phase in read_phases if phase > store_phase
                )
                if not later_reads:
                    continue
                next_store = next(
                    (phase for phase in write_phases if phase > store_phase), None
                )
                load_phases = tuple(
                    phase
                    for phase in later_reads
                    if phase % 2 == 0 and (next_store is None or phase <= next_store)
                )
                if not load_phases or not cls._matching_contiguous_accesses(
                    writes[name][store_phase],
                    [dep for phase in load_phases for dep in reads[name][phase]],
                ):
                    continue
                if not cls._can_elide_global_store(
                    name,
                    store_phase,
                    load_phases,
                    writes,
                    fused_node_names,
                ):
                    continue

                try:
                    dtype = V.graph.get_dtype(name)
                    buffer_numel = V.graph.sizevars.simplify(V.graph.get_numel(name))
                except (KeyError, NotImplementedError, RuntimeError):
                    continue
                if dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    continue
                if not V.graph.sizevars.statically_known_multiple_of(
                    buffer_numel, reduction_numel
                ):
                    continue

                spec_bytes = padded_numel * get_dtype_size(dtype)
                if used_bytes + spec_bytes > cls._MAX_LOCAL_BYTES:
                    continue
                specs.append(
                    LocalBufferRetentionSpec(
                        name=name,
                        dtype=dtype,
                        element_count=reduction_numel,
                        padded_element_count=padded_numel,
                        store_phase=store_phase,
                        load_phases=load_phases,
                    )
                )
                used_bytes += spec_bytes
                break

        if not specs:
            return None

        return LocalBufferRetentionPlan(
            buffers=tuple(specs),
            reduction_numel=reduction_numel,
            reduction_block=min(2048, 1 << (reduction_numel.bit_length() - 1)),
            num_warps=4,
            waves_per_eu=4,
        )


class LocalBufferRetentionKernel(TritonKernel):
    """Triton kernel candidate that retains cross-phase values in TLX LDS."""

    def __init__(
        self,
        *args: Any,
        local_buffer_retention_plan: LocalBufferRetentionPlan,
        **kwargs: Any,
    ) -> None:
        self.local_buffer_retention_plan = local_buffer_retention_plan
        self.local_buffer_retention_phase = 0
        self.local_buffer_retention_barriers: set[int] = set()
        self.local_buffer_retention_names = {
            spec.name: f"tlx_local_{index}"
            for index, spec in enumerate(local_buffer_retention_plan.buffers)
        }
        super().__init__(*args, **kwargs)

    def _local_buffer_spec(self, name: str) -> LocalBufferRetentionSpec | None:
        return next(
            (
                spec
                for spec in self.local_buffer_retention_plan.buffers
                if spec.name == name
            ),
            None,
        )

    def is_buffer_retained_locally(self, name: str, *, store: bool) -> bool:
        spec = self._local_buffer_spec(name)
        if spec is None:
            return False
        if store:
            return self.local_buffer_retention_phase == spec.store_phase
        return self.local_buffer_retention_phase in spec.load_phases

    def set_codegen_phase(self, phase: int) -> None:
        self.local_buffer_retention_phase = phase
        if phase in self.local_buffer_retention_barriers:
            return
        if any(
            phase in spec.load_phases
            for spec in self.local_buffer_retention_plan.buffers
        ):
            self.body.writeline("tl.debug_barrier()")
            self.local_buffer_retention_barriers.add(phase)

    def _local_buffer_shape_and_slice(self, name: str) -> tuple[str, str]:
        spec = self._local_buffer_spec(name)
        if spec is None:
            raise AssertionError(f"no local-buffer retention spec for {name}")
        reduction_trees = [tree for tree in self.range_trees if tree.is_reduction]
        if len(reduction_trees) != 1:
            raise AssertionError("local-buffer retention requires one reduction axis")
        reduction_tree = reduction_trees[0]
        if reduction_tree.tensor_dim is None:
            raise AssertionError("local-buffer retention requires a tensor dimension")

        allocation_shape = ["1"] * self.triton_tensor_ndim()
        allocation_shape[reduction_tree.tensor_dim] = str(spec.padded_element_count)
        offsets = ["0"] * self.triton_tensor_ndim()
        offsets[reduction_tree.tensor_dim] = self.index_to_str(
            reduction_tree.block_offset()
        )
        access_shape = ["1"] * self.triton_tensor_ndim()
        access_shape[reduction_tree.tensor_dim] = reduction_tree.block_size_str()
        return (
            f"({', '.join(allocation_shape)},)",
            f"tlx.local_slice({self.local_buffer_retention_names[name]}, "
            f"[{', '.join(offsets)}], "
            f"[{', '.join(access_shape)}])",
        )

    def _load_from_local_buffer(
        self, name: str, index: sympy.Expr
    ) -> TritonCSEVariable:
        spec = self._local_buffer_spec(name)
        if spec is None:
            raise AssertionError(f"no local-buffer retention spec for {name}")
        self.args.input(name)
        self.must_keep_buffers.add(name)
        _, local_slice = self._local_buffer_shape_and_slice(name)
        line = f"tlx.local_load({local_slice}, relaxed=True)"
        dtype = spec.dtype
        if (
            dtype in (torch.float16, torch.bfloat16)
            and config.triton.codegen_upcast_to_fp32
        ):
            line += ".to(tl.float32)"
            dtype = torch.float32
        result = self.cse.generate(
            self.loads,
            line,
            dtype=dtype,
            shape=tuple(self.dense_size_list()),
        )
        if not isinstance(result, TritonCSEVariable):
            raise AssertionError(f"expected TritonCSEVariable, got {type(result)}")
        return result

    def _store_to_local_buffer(
        self, name: str, value: CSEVariable, mode: StoreMode
    ) -> None:
        if mode is not None:
            raise AssertionError("local-buffer retention only supports plain stores")
        spec = self._local_buffer_spec(name)
        if spec is None:
            raise AssertionError(f"no local-buffer retention spec for {name}")
        self.args.output(name)
        self.must_keep_buffers.add(name)
        _, local_slice = self._local_buffer_shape_and_slice(name)
        self.stores.writeline(
            f"tlx.local_store({local_slice}, {value}.to({triton_type(spec.dtype)}))"
        )

    def load(self, name: str, index: sympy.Expr):
        if self.is_buffer_retained_locally(name, store=False):
            return self._load_from_local_buffer(name, index)
        return super().load(name, index)

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        mode: StoreMode = None,
    ) -> None:
        if self.is_buffer_retained_locally(name, store=True):
            self._store_to_local_buffer(name, value, mode)
            return
        super().store(name, index, value, mode)

    def reduction_loop_range(
        self, prefix: str, loop_start: str, loop_end: str
    ) -> str:
        if self.cooperative_reduction:
            raise AssertionError(
                "local-buffer retention does not support cooperative reduction"
            )
        return (
            f"tl.static_range(0, {self.local_buffer_retention_plan.reduction_numel}, "
            f"{prefix.upper()}BLOCK)"
        )

    def codegen_static_numels(self, code: IndentedBuffer) -> None:
        super().codegen_static_numels(code)
        code.writeline("tl.static_assert(XBLOCK == 1)")
        for spec in self.local_buffer_retention_plan.buffers:
            allocation_shape, _ = self._local_buffer_shape_and_slice(spec.name)
            local_name = self.local_buffer_retention_names[spec.name]
            code.writeline(
                f"{local_name}_storage = tlx.local_alloc("
                f"{allocation_shape}, {triton_type(spec.dtype)}, 1)"
            )
            code.writeline(f"{local_name} = tlx.local_view({local_name}_storage, 0)")

    def inductor_meta_per_kernel(self) -> dict[str, Any]:
        metadata = super().inductor_meta_per_kernel()
        metadata["tlx_local_buffer_retention"] = {
            "buffers": tuple(
                spec.name for spec in self.local_buffer_retention_plan.buffers
            ),
            "bytes": self.local_buffer_retention_plan.total_bytes,
        }
        return metadata


def get_extra_kernel_choices(
    kernel_cls: type[TritonKernel],
    features: SIMDKernelFeatures,
    kernel_args: list[Any],
    kernel_kwargs: dict[str, Any],
) -> list[TritonKernel]:
    """Return the opt-in retained-LDS candidate for a compatible schedule."""
    if kernel_cls is not TritonKernel or "fixed_config" in kernel_kwargs:
        return []
    plan = LocalBufferRetention.plan_for(features.node_schedule)
    if plan is None:
        return []

    retained_kwargs = {
        **kernel_kwargs,
        "local_buffer_retention_plan": plan,
        "override_persistent_reduction": False,
        "override_cooperative_reduction": False,
        "fixed_config": FixedTritonConfig(
            {
                "XBLOCK": 1,
                "R0_BLOCK": plan.reduction_block,
                "num_warps": plan.num_warps,
                "num_stages": 1,
                "waves_per_eu": plan.waves_per_eu,
            }
        ),
    }
    return [LocalBufferRetentionKernel(*kernel_args, **retained_kwargs)]
