"""
TLXInductorChoices: InductorChoices subclass that consolidates all TLX
template selection, filtering, and layout logic.

Activated via config.inductor_choices_class set in template_heuristics/tlx.py.
All methods check config.triton.tlx_mode at runtime and fall through to
super() when TLX is disabled (mode is None).
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.scheduler import BaseSchedulerNode, Scheduler, WhyNoFuse
from torch._inductor.select_algorithm import ExternKernelChoice
from torch.utils._ordered_set import OrderedSet

from . import tlx_config

if TYPE_CHECKING:
    from collections.abc import Generator

    from torch._inductor.codegen.common import KernelTemplate
    from torch._inductor.ir import ChoiceCaller
    from torch._inductor.kernel_inputs import KernelInputs
    from torch._inductor.kernel_template_choice import KernelTemplateChoice

log = logging.getLogger(__name__)


# NVIDIA warp-specialized GEMM templates whose compute_epilogue/TMA path
# redirects intermediate stores into variable assignments and materializes
# only the final result through a single TMA descriptor store.
_TLX_WS_GEMM_TEMPLATE_NAMES = frozenset({
    "tlx_blackwell_gemm_ws",
    "tlx_hopper_gemm_ws",
})


def _has_tlx_ws_gemm_candidate(
    node1: BaseSchedulerNode,
    node2: BaseSchedulerNode,
) -> bool:
    """Whether the proposed fused group contains an affected TLX GEMM candidate.

    Inspects every scheduler node in both groups so an already-fused producer
    (GEMM plus earlier epilogue pointwise ops) is still detected. During
    fusion search template nodes still wrap MultiTemplateBuffer (finalization
    happens after fuse_nodes), so candidates are found in
    ``template_node.choices``; a finalized single-choice TritonTemplateBuffer
    is detected via its template name.
    """
    from torch._inductor import ir

    for snode in (*node1.get_nodes(), *node2.get_nodes()):
        if not snode.is_template():
            continue
        template_node = snode.get_template_node()
        if template_node is None:
            continue
        if _is_tlx_ws_gemm_template(template_node):
            return True
    return False


def _is_tlx_ws_gemm_template(template_node: object) -> bool:
    """Whether an IR template buffer is backed by an affected TLX WS GEMM template.

    Covers both scheduler phases: a MultiTemplateBuffer (max-autotune choice
    not yet finalized; candidates in ``choices`` whose ChoiceCaller names embed
    the template uid with affixes, e.g. "triton_tlx_blackwell_gemm_ws_0") and
    a single-choice TritonTemplateBuffer (carries the originating
    KernelTemplateChoice in ``annotations["ktc"]``, whose ``template.name`` is
    the exact template uid).
    """
    from torch._inductor import ir

    if isinstance(template_node, ir.MultiTemplateBuffer):
        return any(
            isinstance(getattr(choice, "name", ""), str)
            and any(
                template_name in choice.name
                for template_name in _TLX_WS_GEMM_TEMPLATE_NAMES
            )
            for choice in template_node.choices
        )
    ktc = getattr(template_node, "annotations", {}).get("ktc")
    template = getattr(ktc, "template", None)
    if isinstance(getattr(template, "name", None), str):
        return template.name in _TLX_WS_GEMM_TEMPLATE_NAMES
    template = getattr(template_node, "template", None)
    return getattr(template, "name", None) in _TLX_WS_GEMM_TEMPLATE_NAMES


class TLXInductorChoices(InductorChoices):
    """
    InductorChoices subclass that handles TLX template injection,
    filtering, and layout fixing.

    When disabled (mode is None): delegates entirely to the parent class.
    In "allow" mode: TLX templates compete with other backends via autotuning.
    In "force" mode: only TLX templates are used.
    """

    def uuid(self) -> str:
        return "TLXInductorChoices"

    def can_fuse_vertical(
        self,
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """Veto fusing consumers into a TLX WS GEMM with a live intermediate.

        The TLX compute_epilogue/TMA path redirects intermediate stores into
        variable assignments and materializes only the final result, so every
        producer write that would cease being materialized must be removable
        through fusion (no users outside the proposed fused group, e.g. a
        buffer saved for backward). Otherwise the GEMM output stays
        materialized and the pointwise work runs separately.

        Only node1 (producer side) writes are examined: node2's final outputs
        become the fused kernel's TMA store target and are correctly
        materialized even when they are graph outputs.

        This hook is graph-level, not per-candidate: a proposed group
        containing an affected TLX GEMM candidate is conservatively rejected
        even if another candidate (extern/cuBLAS or the ordinary
        pointer-store variant) could preserve the intermediate. That is a
        deliberate performance tradeoff (a missed fusion, never wrong code);
        per-choice filtering and multi-output TMA support are follow-ups.

        Applies in both "allow" and "force" modes: should_force_fusion only
        gates benchmark-time forcing, not this scheduler legality check.
        """
        if not super().can_fuse_vertical(scheduler, node1, node2, shared_data_score):
            return False
        if config.triton.tlx_mode is None:
            return True
        if not _has_tlx_ws_gemm_candidate(node1, node2):
            return True
        from torch._inductor.dependencies import Dep

        fused_node_names = OrderedSet(
            node.get_name() for node in (*node1.get_nodes(), *node2.get_nodes())
        )
        for dep in node1.read_writes.writes:
            if not isinstance(dep, Dep):
                continue
            write_name = scheduler.mutation_renames.get(dep.name, dep.name)
            if write_name not in scheduler.name_to_buf:
                WhyNoFuse(node1, node2)(
                    "TLX WS GEMM epilogue would drop unknown buffer %s",
                    write_name,
                )
                return False
            if not scheduler.can_buffer_be_removed_through_fusion(
                write_name, fused_node_names
            ):
                WhyNoFuse(node1, node2)(
                    "TLX WS GEMM epilogue would drop live intermediate %s",
                    write_name,
                )
                return False
        return True

    def _finalize_template_configs(
        self,
        template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        templates: list[KernelTemplate | ExternKernelChoice],
        op_name: str,
        kwarg_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> list[KernelTemplateChoice]:
        if config.triton.tlx_mode == "force":
            # In force mode, keep only TLX templates
            return [
                c for uid, gen in template_choices.items() if "tlx_" in uid for c in gen
            ]

        return super()._finalize_template_configs(
            template_choices, kernel_inputs, templates, op_name, kwarg_overrides
        )

    def customize_fused_kernel_name(self, fused_name: str, src_code: str) -> str:
        from .fusion import maybe_add_tlx_prefix

        return maybe_add_tlx_prefix(fused_name, src_code)

    def override_best_choice(
        self,
        best_choice: ChoiceCaller,
        timings: dict[ChoiceCaller, float],
    ) -> ChoiceCaller:
        return maybe_override_best_choice(best_choice, timings)

    def _need_to_fix_layout(
        self,
        adjusted_choices: list[KernelTemplateChoice],
        op_name: str,
    ) -> bool:
        if config.triton.tlx_mode == "force":
            return True
        return super()._need_to_fix_layout(adjusted_choices, op_name)

    # The Blackwell template accepts only the two plain-mm inputs. baddbmm has a
    # bias prefix argument, while scaled_mm has two additional scale arguments.
    # AMD addmm remains supported by its warp-pipe template.
    _UNSUPPORTED_OPS = frozenset({"baddbmm", "scaled_mm"})

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        templates: list[KernelTemplate | ExternKernelChoice],
        op_name: str,
        kwarg_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> list[ChoiceCaller]:
        tlx_mode = config.triton.tlx_mode
        if tlx_mode is None or op_name in self._UNSUPPORTED_OPS:
            return super().get_template_configs(
                kernel_inputs, templates, op_name, kwarg_overrides
            )

        from .mm_templates import append_tlx, tlx_kernel_inputs

        templates = list(templates)
        kwarg_overrides = kwarg_overrides or {}
        append_tlx(templates, op_name, kernel_inputs)

        input_tensors = kernel_inputs.nodes()
        if len(input_tensors) < 2:
            raise ValueError(f"Need at least 2 input tensors, got {len(input_tensors)}")

        template_choices = {
            t.uid: self.get_ktc(
                tlx_kernel_inputs(t, kernel_inputs, op_name), t, op_name,
                kwarg_overrides.get(t.uid, {})
            )
            for t in templates
        }

        # Second pass: Adjust the template choices
        adjusted_choices = self._finalize_template_configs(
            template_choices,
            kernel_inputs,
            templates,
            op_name,
            kwarg_overrides,
        )

        # Fix layouts: force mode fixes all, allow mode fixes non-extern only
        fix_all = self._need_to_fix_layout(adjusted_choices, op_name)
        if fix_all or tlx_mode == "allow":
            fixed_layout = kernel_inputs.output_layout(flexible=False)
            for ktc in adjusted_choices:
                needs_fix = fix_all or not isinstance(ktc.template, ExternKernelChoice)
                if needs_fix:
                    ktc.layout = fixed_layout
                    if hasattr(ktc, "_choice"):
                        del ktc._choice

        # Third pass: Convert to ChoiceCaller objects
        return [ktc.choice for ktc in adjusted_choices if ktc.choice is not None]

    def append_flex_attention_choices(
        self,
        choices: list[Any],
        configs: list[Any],
        input_nodes: list[Any],
        subgraphs: list[Any],
        layout: Any,
        kernel_options: dict[str, Any],
        sparse_q_block_size: int,
        sparse_kv_block_size: int,
    ) -> list[Any]:
        from .flex_attention_templates import append_tlx_flex

        return append_tlx_flex(
            choices,
            configs,
            input_nodes,
            subgraphs,
            layout,
            kernel_options,
            sparse_q_block_size,
            sparse_kv_block_size,
        )

    def append_flex_attention_backward_choices(
        self,
        choices: list[Any],
        configs: list[Any],
        input_nodes: list[Any],
        subgraphs: list[Any],
        layout: Any,
        kernel_options: dict[str, Any],
        sparse_q_block_size: int,
        sparse_kv_block_size: int,
        *,
        mutated_inputs: list[Any],
    ) -> list[Any]:
        from .flex_attention_templates import append_tlx_flex_backward

        return append_tlx_flex_backward(
            choices,
            configs,
            input_nodes,
            subgraphs,
            layout,
            kernel_options,
            sparse_q_block_size,
            sparse_kv_block_size,
            mutated_inputs=mutated_inputs,
        )


def maybe_override_best_choice(
    best_choice: ChoiceCaller,
    timings: dict[ChoiceCaller, float],
) -> ChoiceCaller:
    """
    In TLX "allow" mode, prefer extern kernels (cublas) unless a TLX template
    beats them by a meaningful margin. Short autotuning benchmarks are noisy
    and TLX can narrowly "win" but run slower in steady-state.

    Returns the (possibly overridden) best choice.
    """
    from .fusion import _is_tlx_choice

    if config.triton.tlx_mode != "allow" or not _is_tlx_choice(best_choice):
        return best_choice

    from torch._inductor.select_algorithm import ExternKernelCaller

    extern_choices = [c for c in timings if isinstance(c, ExternKernelCaller)]
    if not extern_choices:
        return best_choice

    tlx_threshold = tlx_config.allow_min_speedup

    best_extern_time = min(timings[c] for c in extern_choices)
    best_time = timings[best_choice]
    speedup = best_extern_time / best_time if best_time > 0 else 0
    if speedup < tlx_threshold:
        log.debug(
            "TLX choice %s speedup %.2fx < threshold %.2fx over extern, using extern",
            best_choice.name,
            speedup,
            tlx_threshold,
        )
        return min(extern_choices, key=lambda c: timings[c])

    return best_choice
