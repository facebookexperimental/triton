"""Closed target-program schema for the TLX Wave converter."""

from contextlib import contextmanager
from dataclasses import dataclass, field

from .diagnostics import fail

STAGE = "target_ir"

TARGET_SCHEMA_VERSION = 5
ADDRESS_ARITHMETIC_NO_OVERFLOW = "no_overflow"

# Token representations all lower to ``!wave.mem.token``, but protocol
# verification must not confuse completion, issue ordering, LDS release, and
# publication.  These domains are deliberately target-IR data rather than
# emitter-side producer inspection.
EVENT_DOMAIN_DMA_COMPLETION = "dma_completion"
EVENT_DOMAIN_DMA_GROUP = "dma_group"
EVENT_DOMAIN_DMA_ISSUE = "dma_issue"
EVENT_DOMAIN_MEMORY_COMPLETION = "memory_completion"
EVENT_DOMAIN_MEMORY_ISSUE = "memory_issue"
EVENT_DOMAIN_MEMORY_BARRIER = "memory_barrier"
EVENT_DOMAIN_BARRIER_ISSUE = "barrier_issue"
EVENT_DOMAIN_WAVE_LOCAL_READY = "wave_local_ready"
EVENT_DOMAIN_EMPTY = "empty"
EVENT_DOMAINS = frozenset({
    EVENT_DOMAIN_DMA_COMPLETION,
    EVENT_DOMAIN_DMA_GROUP,
    EVENT_DOMAIN_DMA_ISSUE,
    EVENT_DOMAIN_MEMORY_COMPLETION,
    EVENT_DOMAIN_MEMORY_ISSUE,
    EVENT_DOMAIN_MEMORY_BARRIER,
    EVENT_DOMAIN_BARRIER_ISSUE,
    EVENT_DOMAIN_WAVE_LOCAL_READY,
    EVENT_DOMAIN_EMPTY,
})

# Target operations that issue real memory instructions and therefore
# participate in the explicit ordering frontier around a barrier. LDS reads
# contribute completion; global memory, LDS writes, and direct-to-LDS DMA
# contribute issue only. High-level transforms and reductions are deliberately
# absent even when their eventual implementation may use private scratch memory.
MEMORY_ISSUER_OP_KINDS = frozenset({
    "buffer_load",
    "buffer_load_to_local",
    "buffer_store",
    "load",
    "local_load",
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
class TargetAttr:
    name: str
    value: object


@dataclass(frozen=True)
class TargetLinearLayout:
    in_dims: tuple[str, ...]
    out_dims: tuple[tuple[str, int], ...]
    bases: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]


@dataclass(frozen=True)
class TargetLayout:
    layout_map_id: int
    kind: str
    shape: tuple[int, ...]
    element_type: str | None
    component_count: int
    lane_width: int
    properties: tuple[TargetAttr, ...] = ()
    linear_layout: TargetLinearLayout | None = None
    component_grid: tuple[int, int] | None = None
    component_relation: tuple[int, ...] | None = None
    active_relation: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TargetAssumption:
    assumption_id: int
    kind: str
    predicate: str
    subject_target_ids: tuple[int, ...]
    lower: int | None = None
    upper: int | None = None
    width: int | None = None
    signedness: str | None = None
    divisor: int | None = None
    mask_scope: int | None = None
    provenance: str = ""
    source_op_index: int | None = None


@dataclass(frozen=True)
class TargetContract:
    schema_version: int = TARGET_SCHEMA_VERSION
    address_arithmetic: str = ADDRESS_ARITHMETIC_NO_OVERFLOW
    enable_fp_fusion: bool = False


@dataclass(frozen=True)
class TargetValue:
    target_value_id: int
    type: TargetType
    source_value_id: int | None = None
    debug_name: str | None = None
    event_domain: str | None = None
    layout_map_id: int | None = None
    resource_target_ids: tuple[int, ...] = ()


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
    enable_multi_wave_specialization: bool = False


@dataclass(frozen=True)
class TargetProgram:
    values: tuple[TargetValue, ...]
    ops: tuple[TargetOp, ...]
    regions: tuple[TargetRegion, ...]
    source_value_targets: dict[int, tuple[int, ...]]
    erased_source_values: dict[int, str]
    kernel: TargetKernel = field(default_factory=TargetKernel)
    layouts: tuple[TargetLayout, ...] = ()
    assumptions: tuple[TargetAssumption, ...] = ()
    contract: TargetContract = field(default_factory=TargetContract)


class TargetBuilder:

    def __init__(self, kernel=None, *, layouts=(), contract=None):
        self.values = []
        self.ops = []
        self.regions = [TargetRegion(0)]
        self._region_stack = [0]
        self.source_value_targets = {}
        self.erased_source_values = {}
        self.kernel = kernel or TargetKernel()
        self.layouts = tuple(layouts)
        self.assumptions = ()
        self.contract = contract or TargetContract()
        # Conversion-only exact integer relations. They are consumed while
        # building memory operations and are not part of the target schema.
        self.value_relations = {}
        self.pointer_relations = {}

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
            layout_map_id=None,
            resource_target_ids=(),
    ):
        value_id = len(self.values)
        self.values.append(
            TargetValue(
                value_id,
                target_type,
                source_value_id,
                debug_name,
                event_domain,
                None if layout_map_id is None else int(layout_map_id),
                tuple(dict.fromkeys(int(target_id) for target_id in resource_target_ids)),
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
            value.layout_map_id,
            value.resource_target_ids,
        )

    def set_value_resource_targets(self, target_value_id, resource_target_ids):
        target_value_id = int(target_value_id)
        value = self.values[target_value_id]
        self.values[target_value_id] = TargetValue(
            value.target_value_id,
            value.type,
            value.source_value_id,
            value.debug_name,
            value.event_domain,
            value.layout_map_id,
            tuple(dict.fromkeys(int(target_id) for target_id in resource_target_ids)),
        )

    def set_assumptions(self, assumptions):
        self.assumptions = tuple(assumptions)

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
            self.layouts,
            self.assumptions,
            self.contract,
        )


def target_type_from_converted(converted_type):
    return TargetType(
        converted_type.kind,
        converted_type.representation,
        converted_type.element_type,
        converted_type.lane_width,
        converted_type.component_count,
    )


def target_layout_from_converted(layout):
    return TargetLayout(
        int(layout.layout_map_id),
        str(layout.kind),
        tuple(int(dim) for dim in layout.shape),
        None if layout.element_type is None else str(layout.element_type),
        int(layout.component_count),
        int(layout.lane_width),
        tuple(TargetAttr(str(name), _target_layout_property_value(value)) for name, value in layout.properties.items()),
        (None if layout.linear_layout is None else _target_layout_property_value(layout.linear_layout)),
        (None if layout.component_grid is None else tuple(int(extent) for extent in layout.component_grid)),
        (None if layout.component_relation is None else tuple(int(byte) for byte in layout.component_relation)),
        (None if layout.active_relation is None else tuple(int(byte) for byte in layout.active_relation)),
    )


def _target_layout_property_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_target_layout_property_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(TargetAttr(str(name), _target_layout_property_value(item)) for name, item in value.items())
    get_in_dim_names = getattr(value, "get_in_dim_names", None)
    if (callable(get_in_dim_names) and hasattr(value, "out_dims") and hasattr(value, "bases")):
        return TargetLinearLayout(
            tuple(str(name) for name in get_in_dim_names()),
            tuple((str(name), int(size)) for name, size in value.out_dims),
            tuple((
                str(name),
                tuple(tuple(int(component) for component in basis) for basis in bases),
            ) for name, bases in value.bases),
        )
    fail(
        "TLXW_TARGET_LAYOUT_SCHEMA",
        STAGE,
        f"layout property has unsupported schema value {type(value).__name__}",
    )


def target_assumptions_from_facts(
    fact_program,
    source_value_targets,
    target_ops,
):
    assume_targets = {}
    for op in target_ops:
        if op.kind != "assume":
            continue
        result_ids = frozenset(int(result_id) for result_id in op.results)
        for fact_id, target_id in zip(
                op.fact_ids,
                op.fact_target_ids,
                strict=True,
        ):
            target_id = int(target_id)
            if target_id in result_ids:
                assume_targets.setdefault(int(fact_id), []).append(target_id)
    return tuple(
        TargetAssumption(
            int(fact.fact_id),
            str(fact.kind),
            str(fact.predicate),
            tuple(
                dict.fromkeys(
                    assume_targets.get(
                        int(fact.fact_id),
                        source_value_targets.get(
                            int(fact.subject_value_id),
                            (),
                        ),
                    ))),
            None if fact.lower is None else int(fact.lower),
            None if fact.upper is None else int(fact.upper),
            None if fact.width is None else int(fact.width),
            None if fact.signedness is None else str(fact.signedness),
            None if fact.divisor is None else int(fact.divisor),
            None if fact.mask_scope is None else int(fact.mask_scope),
            str(fact.provenance),
            None if fact.source_op_index is None else int(fact.source_op_index),
        ) for fact in fact_program.facts)


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
