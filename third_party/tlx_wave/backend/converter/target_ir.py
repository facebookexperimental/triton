"""Closed target-program schema for the TLX Wave converter.

Target programs inherit Triton's layout-address contract: backend-synthesized
index, stride, coordinate, LDS-offset, and pointer-offset expressions are only
defined when their signed i32 layout arithmetic does not overflow.  Overflowing
executions are outside the target IR semantics rather than cases the emitter
must preserve with wrapping arithmetic.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

from .diagnostics import fail

STAGE = "target_ir"

# Target values named by this attribute were consumed while proving or
# constructing a structural replacement, but are not runtime operands of the
# target operation.  Keeping the IDs as provenance lets verification and
# diagnostics retain the source graph while target DCE can remove that graph
# when it has no other semantic users.
PROVENANCE_ONLY_TARGET_IDS_ATTR = "provenance_only_target_ids"

# Token representations all lower to ``!wave.mem.token``, but protocol
# verification must not confuse completion, issue ordering, LDS release, and
# publication.  These domains are deliberately target-IR data rather than
# emitter-side producer inspection.
EVENT_DOMAIN_DMA_COMPLETION = "dma_completion"
EVENT_DOMAIN_DMA_GROUP = "dma_group"
EVENT_DOMAIN_DMA_ISSUE = "dma_issue"
EVENT_DOMAIN_MEMORY_COMPLETION = "memory_completion"
EVENT_DOMAIN_MEMORY_ISSUE = "memory_issue"
EVENT_DOMAIN_FULL_BARRIER = "full_barrier"
EVENT_DOMAIN_BARRIER_ISSUE = "barrier_issue"
EVENT_DOMAIN_LDS_COMPLETION = "lds_completion"
EVENT_DOMAIN_LDS_FRONTIER = "lds_frontier"
EVENT_DOMAIN_LDS_RELEASED = "lds_released"
EVENT_DOMAIN_WORKGROUP_READY = "workgroup_ready"
EVENT_DOMAIN_WAVE_LOCAL_READY = "wave_local_ready"
EVENT_DOMAIN_EMPTY = "empty"
EVENT_DOMAINS = frozenset({
    EVENT_DOMAIN_DMA_COMPLETION,
    EVENT_DOMAIN_DMA_GROUP,
    EVENT_DOMAIN_DMA_ISSUE,
    EVENT_DOMAIN_MEMORY_COMPLETION,
    EVENT_DOMAIN_MEMORY_ISSUE,
    EVENT_DOMAIN_FULL_BARRIER,
    EVENT_DOMAIN_BARRIER_ISSUE,
    EVENT_DOMAIN_LDS_COMPLETION,
    EVENT_DOMAIN_LDS_FRONTIER,
    EVENT_DOMAIN_LDS_RELEASED,
    EVENT_DOMAIN_WORKGROUP_READY,
    EVENT_DOMAIN_WAVE_LOCAL_READY,
    EVENT_DOMAIN_EMPTY,
})

# Target operations that issue real memory instructions and therefore
# participate in the completion-free ordering frontier around an explicit
# full-memory barrier.  High-level value transforms such as layout_convert and
# reduction are deliberately absent even when their eventual implementation
# may use private scratch memory.
MEMORY_ISSUER_OP_KINDS = frozenset({
    "buffer_load",
    "buffer_load_to_local",
    "buffer_store",
    "load",
    "local_load",
    "local_load_mma_payload",
    "local_store",
    "store",
})


@dataclass(frozen=True)
class TargetType:
    kind: str
    representation: str
    element_type: str | None = None
    lane_width: int | None = None
    component_count: int = 1


@dataclass(frozen=True)
class TargetValue:
    target_value_id: int
    type: TargetType
    source_value_id: int | None = None
    debug_name: str | None = None
    event_domain: str | None = None


@dataclass(frozen=True)
class TargetAttr:
    name: str
    value: object


@dataclass(frozen=True)
class TargetOp:
    target_op_id: int
    kind: str
    operands: tuple[int, ...] = ()
    results: tuple[int, ...] = ()
    attrs: tuple[TargetAttr, ...] = ()
    fact_ids: tuple[int, ...] = ()
    fact_target_ids: tuple[int, ...] = ()
    layout_map_ids: tuple[int, ...] = ()
    region_ids: tuple[int, ...] = ()
    source_op_index: int | None = None


@dataclass(frozen=True)
class TargetRegion:
    target_region_id: int
    op_ids: tuple[int, ...] = ()
    block_arg_ids: tuple[int, ...] = ()
    yield_value_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TargetKernel:
    name: str = "kernel"
    target: str | None = None
    num_ctas: int | None = None
    num_warps: int | None = None
    threads_per_warp: int | None = None
    noinline: bool | None = None
    arg_target_ids: tuple[int, ...] = ()
    enable_split_barriers: bool = False
    enable_multi_wave_specialization: bool = False


@dataclass(frozen=True)
class TargetProgram:
    values: tuple[TargetValue, ...]
    ops: tuple[TargetOp, ...]
    regions: tuple[TargetRegion, ...]
    source_value_targets: dict[int, tuple[int, ...]]
    erased_source_values: dict[int, str]
    kernel: TargetKernel = field(default_factory=TargetKernel)


class TargetBuilder:

    def __init__(self, kernel=None):
        self.values = []
        self.ops = []
        self.regions = [TargetRegion(0)]
        self._region_stack = [0]
        self.source_value_targets = {}
        self.erased_source_values = {}
        self.kernel = kernel or TargetKernel()
        # Conversion-only protocol state.  The final dependency graph is
        # serialized as ordinary target SSA; these maps merely make that graph
        # convenient to construct while walking structured source regions.
        self.protocol_frontiers = {}

    @property
    def current_region_id(self):
        return self._region_stack[-1]

    def add_value(
        self,
        target_type,
        *,
        source_value_id=None,
        debug_name=None,
        event_domain=None,
    ):
        value_id = len(self.values)
        self.values.append(TargetValue(
            value_id,
            target_type,
            source_value_id,
            debug_name,
            event_domain,
        ))
        if source_value_id is not None:
            self.source_value_targets.setdefault(source_value_id, tuple())
            self.source_value_targets[source_value_id] = (
                *self.source_value_targets[source_value_id],
                value_id,
            )
        return value_id

    def set_value_event_domain(self, target_value_id, event_domain):
        target_value_id = int(target_value_id)
        value = self.values[target_value_id]
        self.values[target_value_id] = TargetValue(
            value.target_value_id,
            value.type,
            value.source_value_id,
            value.debug_name,
            None if event_domain is None else str(event_domain),
        )

    def snapshot_protocol_state(self):
        return {
            int(source_value_id): tuple(int(target_id) for target_id in target_ids)
            for source_value_id, target_ids in self.protocol_frontiers.items()
        }

    def restore_protocol_state(self, snapshot):
        self.protocol_frontiers = {
            int(source_value_id): tuple(int(target_id) for target_id in target_ids)
            for source_value_id, target_ids in snapshot.items()
        }

    def set_protocol_frontier(self, source_value_id, target_ids):
        self.protocol_frontiers[int(source_value_id)] = tuple(dict.fromkeys(
            int(target_id) for target_id in target_ids
        ))

    def append_protocol_frontier(self, source_value_ids, target_id):
        target_id = int(target_id)
        for source_value_id in source_value_ids:
            source_value_id = int(source_value_id)
            current = self.protocol_frontiers.get(source_value_id, ())
            if target_id not in current:
                self.protocol_frontiers[source_value_id] = (*current, target_id)

    def protocol_frontier_target_ids(self):
        return tuple(dict.fromkeys(
            target_id
            for source_value_id in sorted(self.protocol_frontiers)
            for target_id in self.protocol_frontiers[source_value_id]
        ))

    def erase_source_value(self, source_value_id, reason):
        self.erased_source_values[source_value_id] = str(reason)

    def set_kernel_arg_targets(self, target_value_ids):
        self.kernel = TargetKernel(
            self.kernel.name,
            self.kernel.target,
            self.kernel.num_ctas,
            self.kernel.num_warps,
            self.kernel.threads_per_warp,
            self.kernel.noinline,
            tuple(int(value_id) for value_id in target_value_ids),
            self.kernel.enable_split_barriers,
            self.kernel.enable_multi_wave_specialization,
        )

    def add_region(self, *, block_arg_ids=()):
        region_id = len(self.regions)
        self.regions.append(TargetRegion(
            region_id,
            (),
            tuple(int(value_id) for value_id in block_arg_ids),
            (),
        ))
        return region_id

    @contextmanager
    def insertion_region(self, region_id):
        if region_id < 0 or region_id >= len(self.regions):
            fail(
                "TLXW_TARGET_UNKNOWN_REGION",
                STAGE,
                f"unknown target region {region_id}",
            )
        self._region_stack.append(int(region_id))
        try:
            yield
        finally:
            self._region_stack.pop()

    def set_region_yields(self, region_id, yield_value_ids):
        region = self.regions[region_id]
        self.regions[region_id] = TargetRegion(
            region.target_region_id,
            region.op_ids,
            region.block_arg_ids,
            tuple(int(value_id) for value_id in yield_value_ids),
        )

    def add_op(
            self,
            kind,
            *,
            operands=(),
            results=(),
            attrs=None,
            fact_ids=(),
            fact_target_ids=(),
            layout_map_ids=(),
            region_ids=(),
            source_op_index=None,
    ):
        op_id = len(self.ops)
        self.ops.append(
            TargetOp(
                op_id,
                str(kind),
                tuple(int(operand) for operand in operands),
                tuple(int(result) for result in results),
                _attrs_tuple(attrs or {}, op_id),
                tuple(int(fact_id) for fact_id in fact_ids),
                tuple(int(target_id) for target_id in fact_target_ids),
                tuple(int(layout_map_id) for layout_map_id in layout_map_ids),
                tuple(int(region_id) for region_id in region_ids),
                source_op_index,
            ))
        region_id = self.current_region_id
        region = self.regions[region_id]
        self.regions[region_id] = TargetRegion(
            region.target_region_id,
            (*region.op_ids, op_id),
            region.block_arg_ids,
            region.yield_value_ids,
        )
        return op_id

    def build(self):
        return TargetProgram(
            tuple(self.values),
            tuple(self.ops),
            tuple(self.regions),
            dict(self.source_value_targets),
            dict(self.erased_source_values),
            self.kernel,
        )


def target_type_from_converted(converted_type):
    return TargetType(
        converted_type.kind,
        converted_type.representation,
        converted_type.element_type,
        converted_type.lane_width,
        converted_type.component_count,
    )


def attrs_dict(op):
    return {attr.name: attr.value for attr in op.attrs}


def _attrs_tuple(attrs, target_op_id):
    result = []
    for name, value in sorted(attrs.items()):
        if not _is_attr_value(value):
            fail(
                "TLXW_TARGET_NON_SCHEMA_ATTR",
                STAGE,
                f"target attr {name} has unsupported value {value!r}",
                target_op_id=target_op_id,
            )
        result.append(TargetAttr(str(name), value))
    return tuple(result)


def _is_attr_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, tuple):
        return all(_is_attr_value(item) for item in value)
    if isinstance(value, frozenset):
        return all(_is_attr_value(item) for item in value)
    return False
