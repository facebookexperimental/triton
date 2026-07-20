"""Structural Wave emission for verified target programs."""

from dataclasses import dataclass, field
from pathlib import Path
import sys
import warnings

from .diagnostics import fail
from . import coordinates
from . import domains
from . import target_ir

STAGE = "emission"

# TLX Wave target programs are only defined for executions where synthesized
# layout-address arithmetic fits signed i32: indexes, strides, coordinates,
# LDS offsets, and pointer offsets must not overflow.  This module-level flag
# encodes that target IR contract in emitted Wave ops; it is not a per-op proof
# computed in emission.  Keep it restricted to backend-synthesized layout math.
# Generic source arithmetic still flows through _emit_binary and only receives
# overflow flags that were present on the source arith op.
_LAYOUT_MATH_NSW = True
_MMA_PACKET_REPRESENTATIONS = frozenset({
    "simd_packet",
    "simd_packet_tuple",
})
_COMPARE_SELECT_MASK_BUDGET_DWORDS = 8
_FULL_MEMORY_BARRIER_ADDRESS_SPACE = 31


@dataclass(frozen=True)
class EmittedWaveModule:
    text: str
    lds_size: int = 0


@dataclass(frozen=True)
class _SharedPointerDwordBase:
    base: object
    dword_offset: object | None = None


@dataclass(frozen=True)
class _CircularLocalMemoryRegion:
    base_root: int
    index_base_target_id: int
    phase: int
    slot_count: int
    slot_stride_bytes: int


@dataclass(frozen=True)
class _I32MaskPayload:
    components: tuple[object, ...]
    predicates: tuple[object, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "components", tuple(self.components))
        if self.predicates is None:
            return
        predicates = tuple(self.predicates)
        if len(predicates) != len(self.components):
            raise ValueError("mask payload predicate count must match payload component count")
        object.__setattr__(self, "predicates", predicates)


@dataclass(frozen=True)
class _VectorPacketPayload:
    packets: tuple[object, ...]
    packet_width: int
    logical_component_count: int


@dataclass(frozen=True)
class _RawLayoutVectorPacketPayload:
    # Keep packed memory bits intact until the layout exchange has selected
    # its vector LDS packets; unpacking here scalarizes both memory operations.
    packets: tuple[object, ...]
    packet_width: int
    logical_component_count: int
    element_bit_width: int


@dataclass(frozen=True)
class _DeferredPacketLocalLoad:
    loaded: object
    element_count: int


@dataclass(frozen=True)
class _LoopValueShape:
    component_count: int
    is_mask_payload: bool = False
    mask_payload_component_count: int | None = None
    mask_predicate_component_count: int = 0
    packet_width: int | None = None
    logical_component_count: int | None = None
    preserved_vector_payload_key: tuple[int, str, int] | None = None
    preserved_vector_payload_type: object | None = field(default=None, compare=False)


@dataclass
class _EmissionState:
    dsl: object
    ir: object
    builder: object
    target_program: target_ir.TargetProgram
    fact_program: object | None
    values: dict[int, object]
    uniform_pointer_bases: dict[int, tuple[object, ...]] = field(default_factory=dict)
    shared_pointer_dword_bases: dict[int, _SharedPointerDwordBase] = field(default_factory=dict)
    shared_pointer_offset_cache: dict[tuple[object, ...], object] = field(default_factory=dict)
    wave_offset_i32_cache: dict[tuple[object, ...], object] = field(default_factory=dict)
    lane_mask_loop_phase: object | None = None
    scratch_token: object | None = None
    scratch_token_needs_write_barrier: bool = False
    local_memory_tokens: dict[int, object] = field(default_factory=dict)
    local_memory_pending_accesses: dict[int, str] = field(default_factory=dict)
    pending_mma_read_boundaries: dict[int, int] = field(default_factory=dict)
    materialized_mma_read_boundaries: dict[int, object] = field(default_factory=dict)
    next_mma_read_boundary_id: int = 0
    local_memory_root_sets: dict[int, frozenset[int]] = field(default_factory=dict)
    token_local_memory_root_sets: dict[int, frozenset[int]] = field(default_factory=dict)
    static_local_memory_roots: dict[tuple[int, int, int], int] = field(default_factory=dict)
    static_local_memory_root_intervals: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    circular_local_memory_roots: dict[tuple[int, int, int, int, int], int] = field(default_factory=dict)
    circular_local_memory_root_regions: dict[int, _CircularLocalMemoryRegion] = field(default_factory=dict)
    local_memory_allocations: dict[int, object] = field(default_factory=dict)
    released_local_memory_allocations: set[int] = field(default_factory=set)
    local_memory_release_unsafe_roots: set[int] = field(default_factory=set)
    local_memory_descendants: dict[int, frozenset[int]] = field(default_factory=dict)
    local_memory_access_tokens: dict[int, tuple[object, ...]] = field(default_factory=dict)
    local_memory_read_tokens: dict[int, tuple[object, ...]] = field(default_factory=dict)
    target_value_def_ops: dict[int, object] = field(default_factory=dict)
    next_local_memory_root: int = -1


def emit_wave_module(
    target_program,
    fact_program=None,
    *,
    enable_split_barriers=False,
    waves_per_eu=0,
):
    dsl, ir = _load_wave_dsl()
    kernel = target_program.kernel
    lds_size = _target_lds_size(target_program)
    with dsl.ModuleBuilder() as module_builder:
        _set_module_attrs(module_builder, dsl, ir, kernel)
        arg_types = [
            _wave_type(dsl, target_program.values[target_value_id].type) for target_value_id in kernel.arg_target_ids
        ]
        with module_builder.gpu_module("kernels") as gpu_module:
            with gpu_module.kernel(
                    kernel.name,
                    arg_types,
                    lds_size=lds_size or None,
                    workgroup_size=_kernel_workgroup_size(kernel),
                    attrs=_function_attrs(
                        dsl,
                        ir,
                        kernel,
                        enable_split_barriers=enable_split_barriers,
                        waves_per_eu=waves_per_eu,
                    ),
            ) as builder:
                state = _EmissionState(
                    dsl,
                    ir,
                    builder,
                    target_program,
                    fact_program,
                    {},
                )
                for target_value_id, arg in zip(kernel.arg_target_ids, builder.args):
                    state.values[target_value_id] = arg
                _emit_region(state, 0)
        return EmittedWaveModule(str(module_builder), lds_size)


def _emit_region(state, region_id):
    try:
        region = state.target_program.regions[region_id]
    except IndexError:
        fail(
            "TLXW_EMIT_UNKNOWN_REGION",
            STAGE,
            f"unknown target region {region_id}",
        )
    for target_op_id in region.op_ids:
        try:
            op = state.target_program.ops[target_op_id]
        except IndexError:
            fail(
                "TLXW_EMIT_UNKNOWN_REGION_OP",
                STAGE,
                f"target region {region_id} references missing op {target_op_id}",
                target_op_id=target_op_id,
            )
        _emit_target_op(state, op)
    return tuple(_require_value(state, target_value_id, None) for target_value_id in region.yield_value_ids)


def _emit_target_op(state, op):
    emitter = _TARGET_EMITTERS.get(op.kind)
    if emitter is not None:
        emitter(state, op)
        for target_value_id in op.results:
            value = state.values.get(target_value_id)
            if value is not None and _contains_waveamd_fragment(state, value):
                fail(
                    "TLXW_EMIT_FRAGMENT_BOUNDARY",
                    STAGE,
                    "WaveAMD fragments are MMA-local views and cannot be "
                    "stored as target-op results",
                    target_op_id=op.target_op_id,
                    target_value_id=target_value_id,
                )
        return
    fail(
        "TLXW_EMIT_UNSUPPORTED_TARGET_OP",
        STAGE,
        f"no structural emission for target op {op.kind}",
        target_op_id=op.target_op_id,
    )


def _contains_waveamd_fragment(state, value):
    if isinstance(value, tuple):
        return any(_contains_waveamd_fragment(state, component) for component in value)
    if isinstance(value, (_VectorPacketPayload, _RawLayoutVectorPacketPayload)):
        return any(_contains_waveamd_fragment(state, packet) for packet in value.packets)
    if isinstance(value, _I32MaskPayload):
        return any(_contains_waveamd_fragment(state, component) for component in value.components)
    value_type = getattr(value, "type", None)
    return value_type is not None and state.dsl.FragmentType.isinstance(value_type)


def _emit_return(state, op):
    del state
    if op.operands:
        fail(
            "TLXW_EMIT_RETURN_VALUES",
            STAGE,
            "empty-return emission is supported first; return values are not",
            target_op_id=op.target_op_id,
        )


def _emit_constant(state, op):
    attrs = target_ir.attrs_dict(op)
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    result_type = _wave_type(state.dsl, state.target_program.values[result_id].type)
    literal = attrs["value"]
    if target_type.representation == "scalar":
        state.values[result_id] = _scalar_constant(
            state,
            _scalar_type(state.dsl, target_type.element_type),
            target_type.element_type,
            literal,
            op,
        )
        return
    if target_type.representation in {"simd", "simd_tuple"}:
        value = _wave_constant(
            state,
            result_type,
            _scalar_type(state.dsl, target_type.element_type),
            target_type.element_type,
            literal,
            op,
        )
        state.values[result_id] = _pack_components(tuple(value for _ in range(_component_count(state, result_id))))
        return
    if target_type.representation in {"mask", "mask_tuple"}:
        value = state.dsl.wave.ConstantOp(
            result_type,
            state.ir.Attribute.parse("true" if _literal_bool(literal, op) else "false"),
        ).result
        state.values[result_id] = _pack_components(tuple(value for _ in range(_component_count(state, result_id))))
        return
    fail(
        "TLXW_EMIT_UNSUPPORTED_CONSTANT",
        STAGE,
        f"constant emission does not support {target_type.representation}",
        target_op_id=op.target_op_id,
        target_value_id=result_id,
    )


def _emit_binary(state, op):
    _materialize_fact_ids(state, op)
    attrs = target_ir.attrs_dict(op)
    lhs, rhs = _operand_values(state, op, 2)
    result_id = _single_result(op)
    result_type = state.target_program.values[result_id].type
    count = _component_count(state, result_id)
    if result_type.representation in {"mask", "mask_tuple"}:
        if attrs["operation"] != "andi":
            fail(
                "TLXW_EMIT_UNSUPPORTED_MASK_BINARY",
                STAGE,
                f"unsupported mask binary operation {attrs['operation']}",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        if not isinstance(lhs, _I32MaskPayload) and not isinstance(rhs, _I32MaskPayload):
            lhs_components, rhs_components = _broadcast_components(state, (lhs, rhs), count, op)
            false_i1 = None
            false_mask = None
            reused = []
            components = []
            for lhs_component, rhs_component in zip(lhs_components, rhs_components):
                if _is_scalar_i1_value(state, lhs_component) and _is_scalar_i1_value(
                        state,
                        rhs_component,
                ):
                    if false_i1 is None:
                        false_i1 = _scalar_constant(
                            state,
                            state.dsl.i1(),
                            "i1",
                            False,
                            op,
                        )

                    def emit_scalar_mask_and(
                        lhs_component=lhs_component,
                        rhs_component=rhs_component,
                        false_i1=false_i1,
                    ):
                        return state.builder.select(
                            lhs_component,
                            rhs_component,
                            false_i1,
                        )

                    components.append(
                        _reuse_component_result(
                            reused,
                            (lhs_component, rhs_component, false_i1),
                            emit_scalar_mask_and,
                        ))
                    continue
                if _is_scalar_i1_value(state, rhs_component):
                    lhs_component, rhs_component = rhs_component, lhs_component
                if false_mask is None:
                    false_mask = _wave_mask_constant(
                        state,
                        _wave_type(state.dsl, result_type),
                        False,
                    )
                components.append(
                    _reuse_component_result(
                        reused,
                        (lhs_component, rhs_component, false_mask),
                        lambda lhs_component=lhs_component, rhs_component=rhs_component: state.builder.select(
                            lhs_component,
                            rhs_component,
                            false_mask,
                        ),
                    ))
            state.values[result_id] = _pack_components(tuple(components))
            return
        lane_width = int(result_type.lane_width or 64)
        lhs_components, lhs_predicates = _as_mask_payload_components_with_predicates(state, lhs, count, lane_width, op)
        rhs_components, rhs_predicates = _as_mask_payload_components_with_predicates(state, rhs, count, lane_width, op)
        reused = []
        payload_components = tuple(
            _reuse_component_result(
                reused,
                (lhs_component, rhs_component),
                lambda lhs_component=lhs_component, rhs_component=rhs_component: state.builder.binary(
                    state.dsl.BinaryKind.AndI,
                    lhs_component,
                    rhs_component,
                ),
            ) for lhs_component, rhs_component in zip(lhs_components, rhs_components))
        predicates = None
        if lhs_predicates is not None or rhs_predicates is not None:
            lhs_predicates = lhs_predicates or tuple(
                _i32_payload_to_mask(state, component, lane_width) for component in lhs_components
            )
            rhs_predicates = rhs_predicates or tuple(
                _i32_payload_to_mask(state, component, lane_width) for component in rhs_components
            )
            predicates = tuple(
                _mask_and_predicate(
                    state,
                    lhs_predicate,
                    rhs_predicate,
                    lane_width,
                    op,
                ) for lhs_predicate, rhs_predicate in zip(lhs_predicates, rhs_predicates))
        state.values[result_id] = _I32MaskPayload(
            payload_components,
            predicates=predicates,
        )
        return
    lhs_components, rhs_components = _broadcast_components(state, (lhs, rhs), count, op)
    reused = []
    state.values[result_id] = _pack_components(
        tuple(
            _reuse_component_result(
                reused,
                (lhs_component, rhs_component),
                lambda lhs_component=lhs_component, rhs_component=rhs_component: state.builder.binary(
                    _binary_kind(state.dsl, attrs["operation"]),
                    lhs_component,
                    rhs_component,
                    nsw=bool(attrs.get("nsw", False)),
                    nuw=bool(attrs.get("nuw", False)),
                ),
            ) for lhs_component, rhs_component in zip(lhs_components, rhs_components)))


def _emit_float_binary(state, op):
    attrs = target_ir.attrs_dict(op)
    operation = attrs["operation"]
    fastmath = _fastmath_attr(state, attrs.get("fastmath"), op)
    lhs, rhs = _operand_values(state, op, 2)
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    count = _component_count(state, result_id)
    lhs_components, rhs_components = _broadcast_components(state, (lhs, rhs), count, op)
    if target_type.representation in _MMA_PACKET_REPRESENTATIONS:
        state.values[result_id] = _pack_components(
            tuple(
                _emit_mma_packet_float_binary_component(
                    state,
                    operation,
                    fastmath,
                    lhs_component,
                    rhs_component,
                    op,
                ) for lhs_component, rhs_component in zip(lhs_components, rhs_components)))
        return

    def emit_component(lhs_component, rhs_component):
        return _emit_wave_float_binary_component(
            state,
            operation,
            lhs_component,
            rhs_component,
            fastmath,
            op,
        )

    reused = []
    state.values[result_id] = _pack_components(
        tuple(
            _reuse_component_result(
                reused,
                (lhs_component, rhs_component),
                lambda lhs_component=lhs_component, rhs_component=rhs_component: emit_component(
                    lhs_component,
                    rhs_component,
                ),
            ) for lhs_component, rhs_component in zip(lhs_components, rhs_components)))


def _emit_float_unary(state, op):
    attrs = target_ir.attrs_dict(op)
    operation = attrs["operation"]
    fastmath = _fastmath_attr(state, attrs.get("fastmath"), op)
    operand = _operand_values(state, op, 1)[0]
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    components = _value_components(state, operand, op)
    if len(components) != _component_count(state, result_id):
        fail(
            "TLXW_EMIT_UNSUPPORTED_FLOAT_UNARY",
            STAGE,
            "float unary operand and result component counts must match",
            target_op_id=op.target_op_id,
        )
    emitted = []
    for component in components:
        payload = _simd_1d_vector_payload(state, component)
        if target_type.representation not in _MMA_PACKET_REPRESENTATIONS:
            if payload is not None:
                fail(
                    "TLXW_EMIT_UNSUPPORTED_FLOAT_UNARY",
                    STAGE,
                    "ordinary SIMD float unary values must have scalar lane payloads",
                    target_op_id=op.target_op_id,
                )
            emitted.append(
                _emit_wave_float_unary_component(
                    state,
                    operation,
                    component,
                    fastmath,
                    op,
                ))
            continue
        if payload is None or str(payload[1]) != "f32":
            fail(
                "TLXW_EMIT_UNSUPPORTED_FLOAT_UNARY",
                STAGE,
                "fragment float unary operands must contain f32 register vectors",
                target_op_id=op.target_op_id,
            )
        width, element_type, lane_width = payload
        scalar_type = state.dsl.simd_type(element_type, int(lane_width))
        scalars = []
        for index in range(int(width)):
            scalar = state.dsl.wave.ExtractOp(
                scalar_type,
                component,
                index,
            ).result
            scalars.append(
                _emit_wave_float_unary_component(
                    state,
                    operation,
                    scalar,
                    fastmath,
                    op,
                ))
        emitted.append(state.dsl.wave.PackOp(component.type, scalars).result)
    state.values[result_id] = _pack_components(tuple(emitted))


def _emit_wave_float_unary_component(state, operation, value, fastmath, op):
    builders = {
        "exp2": state.dsl.wave.FExp2Op,
    }
    builder = builders.get(operation)
    if builder is None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_FLOAT_UNARY",
            STAGE,
            f"unsupported float unary operation {operation}",
            target_op_id=op.target_op_id,
        )
    return builder(
        value.type,
        value,
        fastmath=fastmath,
    ).result


def _fastmath_attr(state, flags, op):
    if not flags:
        return None
    flag_text = ",".join(str(flag) for flag in flags)
    try:
        return state.ir.Attribute.parse(f"#arith.fastmath<{flag_text}>")
    except Exception as exc:
        fail(
            "TLXW_EMIT_UNSUPPORTED_FASTMATH",
            STAGE,
            f"cannot build arith fastmath attribute for {flag_text}: {type(exc).__name__}: {exc}",
            target_op_id=op.target_op_id,
        )


def _emit_wave_float_binary_component(
        state, operation, lhs_component, rhs_component, fastmath, op):
    if operation == "divf":
        reciprocal = state.dsl.wave.FRcpOp(
            rhs_component.type,
            rhs_component,
            fastmath=fastmath,
        ).result
        return state.dsl.wave.FMulOp(
            lhs_component.type,
            lhs_component,
            reciprocal,
            fastmath=fastmath,
        ).result
    builders = {
        "addf": state.dsl.wave.FAddOp,
        "maximumf": state.dsl.wave.FMaxOp,
        "maxnumf": state.dsl.wave.FMaxOp,
        "subf": state.dsl.wave.FSubOp,
        "mulf": state.dsl.wave.FMulOp,
    }
    builder = builders.get(operation)
    if builder is None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_FLOAT_BINARY",
            STAGE,
            f"unsupported float binary operation {operation}",
            target_op_id=op.target_op_id,
        )
    return builder(
        lhs_component.type,
        lhs_component,
        rhs_component,
        fastmath=fastmath,
    ).result


def _emit_mma_packet_float_binary_component(
        state, operation, fastmath, lhs_component, rhs_component, op):
    lhs_payload = _simd_1d_vector_payload(state, lhs_component)
    rhs_payload = _simd_1d_vector_payload(state, rhs_component)
    if lhs_payload is None or rhs_payload is None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_FLOAT_BINARY",
            STAGE,
            "MMA packet float binary operands must be SIMD vector payloads",
            target_op_id=op.target_op_id,
        )
    lhs_width, lhs_element_type, lhs_lane_width = lhs_payload
    rhs_width, rhs_element_type, rhs_lane_width = rhs_payload
    if (int(lhs_width) != int(rhs_width) or int(lhs_lane_width) != int(rhs_lane_width)
            or str(lhs_element_type) != str(rhs_element_type)):
        fail(
            "TLXW_EMIT_UNSUPPORTED_FLOAT_BINARY",
            STAGE,
            "MMA packet float binary operands must have matching vector payload types",
            target_op_id=op.target_op_id,
        )
    scalar_type = state.dsl.simd_type(lhs_element_type, int(lhs_lane_width))
    lhs_elements = _packed_vector_payload_elements(
        lhs_component,
        scalar_type,
        int(lhs_width),
    )
    rhs_elements = _packed_vector_payload_elements(
        rhs_component,
        scalar_type,
        int(rhs_width),
    )
    packed_splat_operand = (
        _packed_vector_payload_is_splat(lhs_elements)
        or _packed_vector_payload_is_splat(rhs_elements)
    )
    if (operation in {"addf", "mulf"} and str(lhs_element_type) == "f32"
            and int(lhs_width) >= 2 and int(lhs_width) % 2 == 0
            and not packed_splat_operand):
        return _emit_wave_float_binary_component(
            state,
            operation,
            lhs_component,
            rhs_component,
            fastmath,
            op,
        )
    result_scalars = []
    for element in range(int(lhs_width)):
        lhs_scalar = (
            lhs_elements[element] if lhs_elements is not None
            else state.dsl.wave.ExtractOp(
                scalar_type,
                lhs_component,
                element,
            ).result
        )
        rhs_scalar = (
            rhs_elements[element] if rhs_elements is not None
            else state.dsl.wave.ExtractOp(
                scalar_type,
                rhs_component,
                element,
            ).result
        )
        result_scalars.append(_emit_wave_float_binary_component(
            state,
            operation,
            lhs_scalar,
            rhs_scalar,
            fastmath,
            op,
        ))
    return state.dsl.wave.PackOp(
        lhs_component.type,
        result_scalars,
    ).result


def _emit_float_cast(state, op):
    attrs = target_ir.attrs_dict(op)
    if attrs["operation"] != "fp_convert":
        fail(
            "TLXW_EMIT_UNSUPPORTED_FLOAT_CAST",
            STAGE,
            f"unsupported float cast operation {attrs['operation']}",
            target_op_id=op.target_op_id,
        )
    (source, ) = _operand_values(state, op, 1)
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    result_value_mode = attrs.get("result_value_mode")
    if result_value_mode in {"mma_packet_payload", "register_vector_payload"}:
        result_type = _mma_packet_payload_type(
            state,
            attrs,
            target_type.element_type,
            target_type.lane_width,
            op,
        )
    elif target_type.representation in _MMA_PACKET_REPRESENTATIONS:
        if result_value_mode != "mma_packet_payload":
            fail(
                "TLXW_EMIT_UNSUPPORTED_FLOAT_CAST",
                STAGE,
                "MMA packet float cast requires packet payload attrs",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
    else:
        result_type = _wave_type(state.dsl, target_type)
    count = _component_count(state, result_id)
    source_components = _broadcast_component(state, source, count, op)
    reused = []
    state.values[result_id] = _pack_components(
        tuple(
            _reuse_component_result(
                reused,
                (source_component, ),
                lambda source_component=source_component: state.builder.fpconvert(
                    source_component,
                    result_type,
                ),
            ) for source_component in source_components))


def _emit_cmpi(state, op):
    attrs = target_ir.attrs_dict(op)
    lhs, rhs = _operand_values(state, op, 2)
    result_id = _single_result(op)
    count = _component_count(state, result_id)
    lhs_components, rhs_components = _broadcast_components(state, (lhs, rhs), count, op)
    reused = []
    components = tuple(
        _reuse_component_result(
            reused,
            (lhs_component, rhs_component),
            lambda lhs_component=lhs_component, rhs_component=rhs_component: _cmpi(
                state,
                attrs["predicate"],
                lhs_component,
                rhs_component,
            ),
        ) for lhs_component, rhs_component in zip(lhs_components, rhs_components))
    state.values[result_id] = _pack_components(components)


def _emit_cmpi_select(state, op):
    attrs = target_ir.attrs_dict(op)
    lhs, rhs, true_value, false_value = _operand_values(state, op, 4)
    result_id = _single_result(op)
    count = _component_count(state, result_id)
    lhs_components, rhs_components, true_components, false_components = (
        _broadcast_components(
            state,
            (lhs, rhs, true_value, false_value),
            count,
            op,
        )
    )
    # Keep the transient tuple of materialized scalar masks bounded without
    # forcing every compare/select pair through VCC back-to-back.  The latter
    # creates a long fixed-register hazard chain that the machine scheduler
    # fills with otherwise-late data operations, extending their VGPR live
    # ranges.  Budget in SGPR dwords so the policy is independent of wave32 vs
    # wave64 mask width.
    lane_width = int(
        state.target_program.values[result_id].type.lane_width or 64
    )
    mask_dwords = max(1, lane_width // 32)
    batch_components = max(
        1,
        _COMPARE_SELECT_MASK_BUDGET_DWORDS // mask_dwords,
    )
    component_operands = tuple(zip(
        lhs_components,
        rhs_components,
        true_components,
        false_components,
    ))
    components = [None] * count
    reused = []
    for batch_start in range(0, count, batch_components):
        batch = component_operands[
            batch_start:batch_start + batch_components
        ]
        pending = []
        for component_index, operands in enumerate(batch, start=batch_start):
            existing = _find_reused_component_result(reused, operands)
            if existing is not None:
                components[component_index] = existing
                continue
            matching = next(
                (
                    indices
                    for pending_operands, indices in pending
                    if _same_component_operands(pending_operands, operands)
                ),
                None,
            )
            if matching is not None:
                matching.append(component_index)
                continue
            pending.append((operands, [component_index]))
        masks = tuple(
            _cmpi(
                state,
                attrs["predicate"],
                operands[0],
                operands[1],
            )
            for operands, _ in pending
        )
        for (operands, indices), mask in zip(pending, masks):
            selected = state.builder.select(
                mask,
                operands[2],
                operands[3],
            )
            reused.append((operands, selected))
            for component_index in indices:
                components[component_index] = selected
    state.values[result_id] = _pack_components(tuple(components))


def _emit_affine_materialize(state, op):
    attrs = target_ir.attrs_dict(op)
    result_id = _single_result(op)
    result_type = state.target_program.values[result_id].type
    if (
        result_type.element_type not in {"i32", "index"}
        or result_type.representation not in {"simd", "simd_tuple"}
    ):
        fail(
            "TLXW_EMIT_AFFINE_EDGE",
            STAGE,
            "affine_materialize requires an i32 or index SIMD target result",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    scalar_count = int(attrs.get("scalar_count", 0))
    scalar_values = _operand_values(state, op, scalar_count)
    component_count = int(result_type.component_count)
    lane_width = int(result_type.lane_width or 64)
    mode = attrs.get("mode", "layout_coordinates")
    if mode == "packet_coordinates":
        shape = tuple(int(dim) for dim in attrs.get("coordinate_shape", ()))
        packet_order = _physical_order_from_attrs(
            attrs,
            "coordinate_order",
            shape,
            op,
            "TLXW_EMIT_AFFINE_EDGE",
        )
        component_thread_count = int(attrs["component_thread_count"])
        packet_elements = int(attrs["packet_elements"])
        workitem = state.builder.workitem_id(
            0,
            state.dsl.i32(),
            lane_width,
        )
        state.values[result_id] = _pack_components(tuple(
            _bounded_index_edge(
                state,
                _packet_source_offset_index_expr(
                    state,
                    component,
                    component_count,
                    workitem,
                    component_thread_count,
                    packet_elements,
                    shape,
                    packet_order,
                    attrs.get("terms", ()),
                    scalar_values,
                    op,
                    attrs.get("value_range"),
                    attrs.get("scalar_component_sources", ()),
                    coordinate_mode=attrs.get(
                        "coordinate_mode",
                        "ordered_linear",
                    ),
                    linear_component_bases=attrs.get(
                        "linear_component_bases",
                        (),
                    ),
                ),
                attrs.get("value_range"),
                op,
            )
            for component in range(component_count)
        ))
        return
    if mode != "layout_coordinates":
        fail(
            "TLXW_EMIT_AFFINE_EDGE",
            STAGE,
            f"unsupported affine materialization mode {mode!r}",
            target_op_id=op.target_op_id,
        )
    shape = tuple(int(dim) for dim in attrs.get("coordinate_shape", ()))
    component_bases = tuple(
        tuple(int(component) for component in bases)
        for bases in attrs.get("component_coordinate_bases", ())
    )
    workitem_coefficients = tuple(
        tuple(int(component) for component in coefficients)
        for coefficients in attrs.get("workitem_coordinate_coefficients", ())
    )
    if len(component_bases) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "affine_materialize component coordinate count does not match "
            "its result type",
            target_op_id=op.target_op_id,
        )
    if any(len(bases) != len(shape) for bases in component_bases) or any(
        len(coefficients) != len(shape)
        for coefficients in workitem_coefficients
    ):
        fail(
            "TLXW_EMIT_BAD_COORDINATES",
            STAGE,
            "affine_materialize coordinate ranks do not match its shape",
            target_op_id=op.target_op_id,
        )
    workitem = state.builder.workitem_id(0, state.dsl.i32(), lane_width)
    components = []
    if result_type.element_type == "index":
        workitem_symbol = state.dsl.sym("wi")
        scalar_symbols = tuple(
            state.dsl.sym(f"s{index}")
            for index in range(len(scalar_values))
        )
        scalar_component_sources = attrs.get("scalar_component_sources", ())
        for component, component_base in enumerate(component_bases):
            coords = tuple(
                _bit_linear_thread_coordinate_expr(
                    state,
                    workitem_symbol,
                    int(base),
                    tuple(
                        coefficients[dim]
                        for coefficients in workitem_coefficients
                    ),
                )
                for dim, base in enumerate(component_base)
            )
            expr = _affine_offset_expr(
                state,
                attrs.get("terms", ()),
                coords,
                scalar_symbols,
                op,
            )
            bindings = {workitem_symbol: workitem}
            bindings.update({
                symbol: _mapped_affine_component_binding_value(
                    state,
                    value,
                    scalar_component_sources,
                    scalar_index,
                    component_count,
                    component,
                    op,
                )
                for scalar_index, (symbol, value) in enumerate(
                    zip(scalar_symbols, scalar_values)
                )
            })
            index_value = state.builder.index_expr(
                expr,
                bindings=bindings,
                assumptions=_index_expr_range_assumptions(
                    expr,
                    attrs.get("value_range"),
                ),
            )
            components.append(_bounded_index_edge(
                state,
                index_value,
                attrs.get("value_range"),
                op,
            ))
        state.values[result_id] = _pack_components(tuple(components))
        return
    for component_base in component_bases:
        coords = tuple(
            _bit_linear_thread_coordinate(
                state,
                workitem,
                int(base),
                tuple(
                    coefficients[dim]
                    for coefficients in workitem_coefficients
                ),
                lane_width,
            )
            for dim, base in enumerate(component_base)
        )
        components.append(
            _affine_offset_value(
                state,
                attrs.get("terms", ()),
                coords,
                scalar_values,
                op,
                no_signed_wrap=bool(attrs.get("no_signed_wrap", False)),
            )
        )
    state.values[result_id] = _pack_components(tuple(components))


def _emit_type_convert(state, op):
    attrs = target_ir.attrs_dict(op)
    mode = attrs.get("mode")
    (source, ) = _operand_values(state, op, 1)
    result_id = _single_result(op)
    if mode == "index_cast":
        result_type = state.target_program.values[result_id].type
        result_element_type = _scalar_type(
            state.dsl,
            result_type.element_type,
        )
        components = []
        for component in _as_components(source):
            if _is_simd_value(state.dsl, component):
                physical_type = state.dsl.SimdType(component.type)
                cast_type = state.dsl.simd_type(
                    result_element_type,
                    int(physical_type.width),
                )
            else:
                cast_type = result_element_type
            components.append(
                component
                if str(component.type) == str(cast_type)
                else state.builder.index_cast(component, cast_type)
            )
        state.values[result_id] = _pack_components(tuple(components))
        return
    if mode == "bounded_i32_to_index":
        source_components = _as_components(source)
        result_type = state.target_program.values[result_id].type
        if len(source_components) != int(result_type.component_count):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "bounded index conversion does not match its result type",
                target_op_id=op.target_op_id,
            )
        symbol = state.dsl.sym("x")
        assumptions = _index_expr_range_assumptions(
            symbol,
            attrs.get("value_range"),
        )
        state.values[result_id] = _pack_components(tuple(
            _bounded_index_edge(
                state,
                state.builder.index_expr(
                    symbol,
                    bindings={symbol: component},
                    assumptions=assumptions,
                ),
                attrs.get("value_range"),
                op,
            )
            for component in source_components
        ))
        return
    if mode == "component_remap":
        mask_payload = source if isinstance(source, _I32MaskPayload) else None
        source_components = (
            tuple(mask_payload.components)
            if mask_payload is not None
            else _as_components(source)
        )
        component_sources = tuple(
            int(component)
            for component in attrs.get("component_sources", ())
        )
        if not component_sources or any(
            component < 0 or component >= len(source_components)
            for component in component_sources
        ):
            fail(
                "TLXW_EMIT_TYPE_CONVERT",
                STAGE,
                "component remap references an invalid source component",
                target_op_id=op.target_op_id,
            )
        result_type = state.target_program.values[result_id].type
        if len(component_sources) != int(result_type.component_count):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "component remap does not match its result type",
                target_op_id=op.target_op_id,
            )
        remapped_components = tuple(
            source_components[component]
            for component in component_sources
        )
        if mask_payload is not None:
            remapped_predicates = (
                None
                if mask_payload.predicates is None
                else tuple(
                    mask_payload.predicates[component]
                    for component in component_sources
                )
            )
            state.values[result_id] = _I32MaskPayload(
                remapped_components,
                predicates=remapped_predicates,
            )
        else:
            state.values[result_id] = _pack_components(remapped_components)
        return
    packet_count = int(attrs.get("packet_component_count", 0))
    packet_width = int(attrs.get("packet_width", 0))
    if packet_count <= 0 or packet_width <= 0:
        fail(
            "TLXW_EMIT_TYPE_CONVERT",
            STAGE,
            "packet edge conversion requires positive packet dimensions",
            target_op_id=op.target_op_id,
        )
    if mode == "packet_to_scalar_components":
        packets = _as_components(source)
        if len(packets) != packet_count:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "packet edge source count does not match its conversion attrs",
                target_op_id=op.target_op_id,
            )
        result_type = _wave_type(
            state.dsl,
            state.target_program.values[result_id].type,
        )
        components = []
        for packet in packets:
            payload = _simd_1d_vector_payload(state, packet)
            if payload is None or int(payload[0]) != packet_width:
                fail(
                    "TLXW_EMIT_TYPE_CONVERT",
                    STAGE,
                    "packet edge source has the wrong physical vector width",
                    target_op_id=op.target_op_id,
                )
            known_elements = _packed_vector_payload_elements(
                packet,
                result_type,
                packet_width,
            )
            if known_elements is None:
                known_elements = tuple(
                    state.dsl.wave.ExtractOp(
                        result_type,
                        packet,
                        element,
                    ).result
                    for element in range(packet_width)
                )
            components.extend(known_elements)
        state.values[result_id] = _pack_components(tuple(components))
        return
    if mode == "scalar_components_to_packet":
        components = _as_components(source)
        if len(components) != packet_count * packet_width:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "scalar edge source count does not match packet dimensions",
                target_op_id=op.target_op_id,
            )
        target_type = state.target_program.values[result_id].type
        packet_type = _mma_packet_payload_type(
            state,
            {"registers": packet_width},
            target_type.element_type,
            target_type.lane_width,
            op,
        )
        packets = tuple(
            state.dsl.wave.PackOp(
                packet_type,
                components[
                    packet * packet_width:(packet + 1) * packet_width
                ],
            ).result
            for packet in range(packet_count)
        )
        state.values[result_id] = _pack_components(packets)
        return
    fail(
        "TLXW_EMIT_TYPE_CONVERT",
        STAGE,
        f"unsupported structural type conversion mode {mode!r}",
        target_op_id=op.target_op_id,
    )


def _emit_signed_extremum(state, op, predicate):
    lhs, rhs = _operand_values(state, op, 2)
    result_id = _single_result(op)
    count = _component_count(state, result_id)
    lhs_components, rhs_components = _broadcast_components(state, (lhs, rhs), count, op)
    reused = []
    components = []
    for lhs_component, rhs_component in zip(lhs_components, rhs_components):
        components.append(
            _reuse_component_result(
                reused,
                (lhs_component, rhs_component),
                lambda lhs_component=lhs_component, rhs_component=rhs_component: state.builder.select(
                    _cmpi(state, predicate, lhs_component, rhs_component),
                    lhs_component,
                    rhs_component,
                ),
            ))
    state.values[result_id] = _pack_components(tuple(components))


def _emit_minsi(state, op):
    _emit_signed_extremum(state, op, "slt")


def _emit_maxsi(state, op):
    _emit_signed_extremum(state, op, "sgt")


def _emit_assume(state, op):
    _materialize_fact_ids(state, op)


def _materialize_fact_ids(state, op):
    if not op.fact_ids:
        return
    if len(op.fact_target_ids) != len(op.fact_ids):
        fail(
            "TLXW_EMIT_FACT_TARGET_COUNT",
            STAGE,
            "fact materialization requires one target value per fact",
            target_op_id=op.target_op_id,
        )
    if state.fact_program is None:
        fail(
            "TLXW_EMIT_MISSING_FACT_PROGRAM",
            STAGE,
            "fact materialization requires verified fact records",
            target_op_id=op.target_op_id,
        )
    facts = {fact.fact_id: fact for fact in state.fact_program.facts}
    for fact_id, target_value_id in zip(op.fact_ids, op.fact_target_ids):
        fact = facts.get(fact_id)
        if fact is None:
            fail(
                "TLXW_EMIT_UNKNOWN_FACT",
                STAGE,
                f"assume references missing fact {fact_id}",
                target_op_id=op.target_op_id,
                fact_id=fact_id,
            )
        value = _require_value(state, target_value_id, op)
        assumptions = _range_assumptions(state.dsl, fact)
        if assumptions:
            state.values[target_value_id] = state.builder.assume(
                value,
                assumptions,
                name="x",
            )


def _emit_make_range(state, op):
    attrs = target_ir.attrs_dict(op)
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    width = int(target_type.lane_width or 64)
    element_type = _scalar_type(state.dsl, target_type.element_type)
    workitem = state.builder.workitem_id(0, element_type, width)
    start = int(attrs["start"])
    components = []
    if attrs.get("coordinate_mode") == "affine_workitem":
        component_bases = tuple(int(value) for value in attrs["component_bases"])
        stride = int(attrs["workitem_stride"])
        if len(component_bases) != _component_count(state, result_id):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "make_range component bases do not match result component count",
                target_op_id=op.target_op_id,
            )
        for component_base in component_bases:
            value = workitem
            if stride != 1:
                value = _simd_binary_const(
                    state,
                    "muli",
                    value,
                    stride,
                    width,
                    nsw=_LAYOUT_MATH_NSW,
                )
            value = _add_simd_const(
                state,
                value,
                start + int(component_base),
                element_type,
                width,
                nsw=_LAYOUT_MATH_NSW,
            )
            components.append(value)
        state.values[result_id] = _pack_components(tuple(components))
        return
    if attrs.get("coordinate_mode") == "bit_affine_workitem":
        component_bases = tuple(int(value) for value in attrs["component_bases"])
        coefficients = tuple(int(value) for value in attrs["workitem_coefficients"])
        if len(component_bases) != _component_count(state, result_id):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "make_range component bases do not match result component count",
                target_op_id=op.target_op_id,
            )
        dynamic = _bit_affine_thread_offset(
            state,
            workitem,
            0,
            coefficients,
            width,
        )
        for component_base in component_bases:
            components.append(
                _add_simd_const(
                    state,
                    dynamic,
                    start + int(component_base),
                    element_type,
                    width,
                    nsw=_LAYOUT_MATH_NSW,
                ))
        state.values[result_id] = _pack_components(tuple(components))
        return
    if attrs.get("coordinate_mode") == "layout_coordinates":
        shape = tuple(int(value) for value in attrs["coordinate_shape"])
        component_bases = tuple(tuple(int(value) for value in bases) for bases in attrs["component_coordinate_bases"])
        workitem_coefficients = tuple(
            tuple(int(value) for value in coefficients) for coefficients in attrs["workitem_coordinate_coefficients"])
        if len(component_bases) != _component_count(state, result_id):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "make_range coordinate bases do not match result component count",
                target_op_id=op.target_op_id,
            )
        if any(len(bases) != len(shape) for bases in component_bases):
            fail(
                "TLXW_EMIT_BAD_COORDINATES",
                STAGE,
                "make_range component coordinate rank does not match shape",
                target_op_id=op.target_op_id,
            )
        if any(len(coefficients) != len(shape) for coefficients in workitem_coefficients):
            fail(
                "TLXW_EMIT_BAD_COORDINATES",
                STAGE,
                "make_range workitem coordinate rank does not match shape",
                target_op_id=op.target_op_id,
            )
        for component_base in component_bases:
            coords = tuple(
                _bit_linear_thread_coordinate(
                    state,
                    workitem,
                    int(base),
                    tuple(coefficients[dim] for coefficients in workitem_coefficients),
                    width,
                ) for dim, base in enumerate(component_base))
            value = _linearize_coordinates(state, coords, shape, width)
            value = _add_simd_const(
                state,
                value,
                start,
                element_type,
                width,
                nsw=_LAYOUT_MATH_NSW,
            )
            components.append(value)
        state.values[result_id] = _pack_components(tuple(components))
        return
    if attrs.get("coordinate_mode") not in (None, "flat"):
        fail(
            "TLXW_EMIT_UNSUPPORTED_MAKE_RANGE",
            STAGE,
            f"unsupported make_range coordinate mode {attrs['coordinate_mode']}",
            target_op_id=op.target_op_id,
        )
    for component in range(_component_count(state, result_id)):
        component_start = start + component * width
        value = _add_simd_const(
            state,
            workitem,
            component_start,
            element_type,
            width,
            nsw=_LAYOUT_MATH_NSW,
        )
        components.append(value)
    state.values[result_id] = _pack_components(tuple(components))


def _linearize_coordinates(state, coords, shape, lane_width):
    if len(coords) != len(shape):
        fail(
            "TLXW_EMIT_BAD_COORDINATES",
            STAGE,
            "coordinate count does not match shape rank",
        )
    result = state.builder.splat(
        state.builder.constant(state.dsl.i32(), 0),
        state.dsl.i32(),
        int(lane_width),
    )
    for dim, coord in enumerate(coords):
        stride = _product(shape[dim + 1:])
        term = coord
        if int(stride) != 1:
            term = _simd_binary_const(
                state,
                "muli",
                term,
                int(stride),
                lane_width,
                nsw=_LAYOUT_MATH_NSW,
            )
        result = state.builder.binary(
            state.dsl.BinaryKind.AddI,
            result,
            term,
            nsw=_LAYOUT_MATH_NSW,
        )
    return result


def _linearize_coordinates_expr(state, coords, shape):
    result = state.dsl.sym_ctx.int_(0)
    for dim, coord in enumerate(coords):
        stride = _product(shape[dim + 1:])
        term = coord
        if int(stride) != 1:
            term = term * int(stride)
        result = result + term
    return result


def _linearize_coordinates_with_order(state, coords, shape, order, lane_width):
    if len(coords) != len(shape):
        fail(
            "TLXW_EMIT_BAD_COORDINATES",
            STAGE,
            "coordinate count does not match shape rank",
        )
    result = state.builder.splat(
        state.builder.constant(state.dsl.i32(), 0),
        state.dsl.i32(),
        int(lane_width),
    )
    stride = 1
    for dim in order:
        term = coords[int(dim)]
        if int(stride) != 1:
            term = _simd_binary_const(
                state,
                "muli",
                term,
                int(stride),
                lane_width,
                nsw=_LAYOUT_MATH_NSW,
            )
        result = state.builder.binary(
            state.dsl.BinaryKind.AddI,
            result,
            term,
            nsw=_LAYOUT_MATH_NSW,
        )
        stride *= int(shape[int(dim)])
    return result


def _physical_order_from_attrs(attrs, key, shape, op, diagnostic):
    order = tuple(int(dim) for dim in attrs.get(key, ()))
    if not order:
        return _default_physical_order(shape)
    if len(order) > len(shape) or sorted(order) != list(range(len(order))):
        fail(
            diagnostic,
            STAGE,
            f"shared layout order {order} cannot be applied to rank-{len(shape)} shape",
            target_op_id=op.target_op_id,
        )
    prefix_rank = len(shape) - len(order)
    mapped = tuple(prefix_rank + int(dim) for dim in order)
    return mapped + tuple(reversed(range(prefix_rank)))


def _default_physical_order(shape):
    return tuple(reversed(range(len(shape))))


def _bit_linear_thread_coordinate(state, workitem, base, coefficients, lane_width):
    lane_width = int(lane_width)
    result = state.builder.splat(
        state.builder.constant(state.dsl.i32(), int(base)),
        state.dsl.i32(),
        lane_width,
    )
    for bit, coefficient in enumerate(coefficients):
        coefficient = int(coefficient)
        if coefficient == 0:
            continue
        bit_value = _simd_binary_const(state, "divui", workitem, 1 << bit, lane_width)
        bit_value = _simd_binary_const(state, "remui", bit_value, 2, lane_width)
        if coefficient != 1:
            bit_value = _simd_binary_const(
                state,
                "muli",
                bit_value,
                coefficient,
                lane_width,
                nsw=_LAYOUT_MATH_NSW,
            )
        result = state.builder.binary(state.dsl.BinaryKind.XOrI, result, bit_value)
    return result


def _bit_linear_thread_coordinate_expr(state, workitem, base, coefficients):
    result = state.dsl.sym_ctx.int_(int(base))
    for bit, coefficient in enumerate(coefficients):
        coefficient = int(coefficient)
        if coefficient == 0:
            continue
        bit_value = state.dsl.mod(state.dsl.floor(workitem / (1 << bit)), 2)
        if coefficient != 1:
            bit_value *= coefficient
        result = state.dsl.xor(result, bit_value)
    return result


def _add_simd_const(state, value, constant, element_type, width, *, nsw=False):
    if not int(constant):
        return value
    start_value = state.builder.splat(
        state.builder.constant(element_type, int(constant)),
        element_type,
        int(width),
    )
    return state.builder.binary(
        state.dsl.BinaryKind.AddI,
        value,
        start_value,
        nsw=bool(nsw),
    )


def _emit_splat(state, op):
    operand = _operand_values(state, op, 1)[0]
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    if target_type.representation in {"mask", "mask_tuple"}:
        if not _is_scalar_i1_value(state, operand):
            fail(
                "TLXW_EMIT_UNSUPPORTED_MASK_SPLAT",
                STAGE,
                f"mask splat expects scalar i1, got {operand.type}",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        lane_width = int(target_type.lane_width or 64)
        # A tensor splat is a lane mask even when its source is uniform.  Keep
        # that mask in ordinary SIMD data until an operation consumes it.  A
        # scalar i1 is otherwise represented by SCC, which cannot safely stay
        # live across the address arithmetic of one masked memory operation
        # and then be reused by another.  The durable 0/1 payload also lets all
        # memory consumers reconstruct the same explicit Wave execution mask.
        payload = _mask_to_i32_payload(state, operand, lane_width)
        state.values[result_id] = _I32MaskPayload(
            tuple(payload for _ in range(_component_count(state, result_id)))
        )
        return
    splat = state.builder.splat(
        operand,
        _splat_element_type(state.dsl, target_type),
        int(target_type.lane_width or 64),
    )
    component_count = _component_count(state, result_id)
    state.values[result_id] = _pack_components(tuple(splat for _ in range(component_count)))
    if target_type.representation in {"per_lane_pointer", "pointer_tuple"}:
        state.uniform_pointer_bases[result_id] = tuple(operand for _ in range(component_count))


def _emit_broadcast(state, op):
    attrs = target_ir.attrs_dict(op)
    operand = _operand_values(state, op, 1)[0]
    operand_id = op.operands[0]
    result_id = _single_result(op)
    target_count = _component_count(state, result_id)
    if isinstance(operand, _I32MaskPayload):
        source_components = operand.components
        source_predicates = operand.predicates
        component_sources = attrs.get("component_sources")
        if component_sources is not None:
            component_sources = tuple(int(source) for source in component_sources)
            if len(component_sources) != target_count:
                fail(
                    "TLXW_EMIT_UNSUPPORTED_BROADCAST",
                    STAGE,
                    "tt.broadcast mask payload source map does not match the "
                    "result component count",
                    target_op_id=op.target_op_id,
                    target_value_id=result_id,
                )
            if any(source < 0 or source >= len(source_components) for source in component_sources):
                fail(
                    "TLXW_EMIT_UNSUPPORTED_BROADCAST",
                    STAGE,
                    "tt.broadcast mask payload source map references an "
                    "out-of-range source component",
                    target_op_id=op.target_op_id,
                    target_value_id=result_id,
                )
            predicates = (None if source_predicates is None else tuple(
                source_predicates[source] for source in component_sources))
            state.values[result_id] = _I32MaskPayload(
                tuple(source_components[source] for source in component_sources),
                predicates=predicates,
            )
            return
        if target_count == len(source_components):
            state.values[result_id] = operand
            return
        if target_count % len(source_components) != 0:
            fail(
                "TLXW_EMIT_UNSUPPORTED_BROADCAST",
                STAGE,
                "tt.broadcast requires the result component count to be a "
                "multiple of the source component count",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        repeat = target_count // len(source_components)
        predicates = (None if source_predicates is None else tuple(predicate
                                                                   for source_predicate in source_predicates
                                                                   for predicate in (source_predicate, ) * repeat))
        state.values[result_id] = _I32MaskPayload(
            tuple(component for source_component in source_components for component in (source_component, ) * repeat),
            predicates=predicates,
        )
        return
    source_components = _value_components(state, operand, op)
    if "register_payload_source_slots" in attrs:
        state.values[result_id] = _emit_broadcast_register_payload(
            state,
            op,
            source_components,
            attrs,
        )
        return
    component_sources = attrs.get("component_sources")
    if component_sources is not None:
        component_sources = tuple(int(source) for source in component_sources)
        if len(component_sources) != target_count:
            fail(
                "TLXW_EMIT_UNSUPPORTED_BROADCAST",
                STAGE,
                "tt.broadcast component source map does not match the result "
                "component count",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        if any(source < 0 or source >= len(source_components) for source in component_sources):
            fail(
                "TLXW_EMIT_UNSUPPORTED_BROADCAST",
                STAGE,
                "tt.broadcast component source map references an out-of-range "
                "source component",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        state.values[result_id] = _pack_components(tuple(source_components[source] for source in component_sources))
        source_bases = state.uniform_pointer_bases.get(operand_id)
        if source_bases is not None:
            state.uniform_pointer_bases[result_id] = tuple(source_bases[source] for source in component_sources)
        return
    if target_count == len(source_components):
        state.values[result_id] = operand
        _propagate_uniform_pointer_bases(state, operand_id, result_id)
        return
    if target_count % len(source_components) != 0:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BROADCAST",
            STAGE,
            "tt.broadcast requires the result component count to be a "
            "multiple of the source component count",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    repeat = target_count // len(source_components)
    state.values[result_id] = tuple(component for source_component in source_components
                                    for component in (source_component, ) * repeat)
    source_bases = state.uniform_pointer_bases.get(operand_id)
    if source_bases is not None:
        state.uniform_pointer_bases[result_id] = tuple(base for source_base in source_bases
                                                       for base in (source_base, ) * repeat)


def _emit_broadcast_register_payload(state, op, source_components, attrs):
    result_id = _single_result(op)
    result_type = state.target_program.values[result_id].type
    lane_width = int(result_type.lane_width or 64)
    element_type = _scalar_type(state.dsl, result_type.element_type)
    scalar_type = state.dsl.simd_type(element_type, lane_width)
    source_component_count = int(attrs["source_component_count"])
    source_registers = int(attrs["source_registers_per_component"])
    result_registers = int(attrs["result_registers_per_component"])
    if len(source_components) != source_component_count:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BROADCAST",
            STAGE,
            "tt.broadcast source component count does not match register payload attrs",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    source_slots = []
    for component in source_components:
        if state.dsl.FragmentType.isinstance(component.type):
            fail(
                "TLXW_EMIT_FRAGMENT_BOUNDARY",
                STAGE,
                "WaveAMD fragments must not cross a broadcast boundary",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        payload = _simd_1d_vector_payload(state, component)
        if source_registers == 1:
            if payload is None:
                source_slots.append(component)
                continue
            if int(payload[0]) == 1:
                source_slots.append(
                    state.dsl.wave.ExtractOp(scalar_type, component, 0).result
                )
                continue
        elif payload is not None and int(payload[0]) == source_registers:
            source_slots.extend(
                state.dsl.wave.ExtractOp(
                    scalar_type,
                    component,
                    register,
                ).result
                for register in range(source_registers)
            )
            continue
        fail(
            "TLXW_EMIT_UNSUPPORTED_BROADCAST",
            STAGE,
            "tt.broadcast source component does not match its register payload width",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    source_map = tuple(
        int(source) for source in attrs["register_payload_source_slots"]
    )
    if any(source < 0 or source >= len(source_slots) for source in source_map):
        fail(
            "TLXW_EMIT_UNSUPPORTED_BROADCAST",
            STAGE,
            "tt.broadcast register payload map references an invalid source slot",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    result_component_count = int(result_type.component_count)
    if len(source_map) != result_component_count * result_registers:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BROADCAST",
            STAGE,
            "tt.broadcast register payload map does not cover the result",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    components = []
    for component in range(result_component_count):
        first = component * result_registers
        values = tuple(
            source_slots[source]
            for source in source_map[first:first + result_registers]
        )
        if result_registers == 1:
            components.append(values[0])
            continue
        payload_type = state.dsl.simd_type(
            state.dsl.vector_type(result_registers, element_type),
            lane_width,
        )
        components.append(
            state.dsl.wave.PackOp(payload_type, values).result
        )
    return _pack_components(tuple(components))


def _emit_addptr(state, op):
    base, offset = _operand_values(state, op, 2)
    base_id = op.operands[0]
    result_id = _single_result(op)
    count = _component_count(state, result_id)
    base_components, offset_components = _broadcast_components(state, (base, offset), count, op)
    uniform_base_components = state.uniform_pointer_bases.get(base_id)
    if uniform_base_components is not None and len(uniform_base_components) != count:
        uniform_base_components = None
    result_type = _wave_type(state.dsl, state.target_program.values[result_id].type)
    state.values[result_id] = _pack_components(
        tuple(
            state.builder.ptr_add(
                _ptr_add_base_component(
                    state,
                    base_component,
                    offset_component,
                    uniform_base_components[index] if uniform_base_components is not None else None,
                ),
                offset_component,
                result_type=result_type,
            ) for index, (base_component, offset_component) in enumerate(zip(base_components, offset_components))))


def _emit_expand_dims(state, op):
    attrs = target_ir.attrs_dict(op)
    operand = _operand_values(state, op, 1)[0]
    operand_id = op.operands[0]
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    target_count = _component_count(state, result_id)
    if isinstance(operand, _I32MaskPayload):
        if len(operand.components) != target_count:
            fail(
                "TLXW_EMIT_UNSUPPORTED_REMAP",
                STAGE,
                "tt.expand_dims changed mask payload component count; explicit "
                "remap is required",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        state.values[result_id] = operand
        return
    if target_type.representation in _MMA_PACKET_REPRESENTATIONS:
        if attrs.get("result_value_mode") != "mma_packet_remap":
            fail(
                "TLXW_EMIT_UNSUPPORTED_REMAP",
                STAGE,
                "tt.expand_dims to an MMA packet requires packet remap attrs",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        components = _value_components(state, operand, op)
        source_component_count = int(attrs.get("source_component_count", 0))
        if len(components) != source_component_count:
            fail(
                "TLXW_EMIT_UNSUPPORTED_REMAP",
                STAGE,
                "tt.expand_dims MMA packet source component count does not match attrs",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        result_payload_type = _mma_packet_payload_type(
            state,
            attrs,
            target_type.element_type,
            target_type.lane_width,
            op,
        )
        scalar_type = state.dsl.simd_type(
            _scalar_type(state.dsl, target_type.element_type),
            int(target_type.lane_width or attrs.get("lane_width", 64) or 64),
        )
        registers = int(attrs["registers"])
        source_indices = tuple(int(index) for index in attrs.get("packet_source_indices", ()))
        if len(source_indices) != target_count * registers:
            fail(
                "TLXW_EMIT_UNSUPPORTED_REMAP",
                STAGE,
                "tt.expand_dims MMA packet source map does not match result payload",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        packed = []
        if any(str(component.type) != str(scalar_type) for component in components):
            fail(
                "TLXW_EMIT_UNSUPPORTED_REMAP",
                STAGE,
                "tt.expand_dims MMA packet remap requires scalar SIMD components",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        for result_component in range(target_count):
            first = result_component * registers
            packed.append(
                state.dsl.wave.PackOp(
                    result_payload_type,
                    [components[index] for index in source_indices[first:first + registers]],
                ).result)
        state.values[result_id] = _pack_components(tuple(packed))
        return
    result_type = _wave_type(state.dsl, state.target_program.values[result_id].type)
    components = _value_components(state, operand, op)
    if any(str(component.type) != str(result_type) for component in components):
        fail(
            "TLXW_EMIT_UNSUPPORTED_REMAP",
            STAGE,
            "tt.expand_dims changed the emitted Wave type; explicit remap is required",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    if len(components) != target_count:
        fail(
            "TLXW_EMIT_UNSUPPORTED_REMAP",
            STAGE,
            "tt.expand_dims changed component count; explicit remap is required",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    state.values[result_id] = operand
    _propagate_uniform_pointer_bases(state, operand_id, result_id)


def _ptr_add_base_component(state, base_component, offset_component, uniform_base):
    if uniform_base is not None and _is_simd_value(state.dsl, offset_component):
        return uniform_base
    return base_component


def _propagate_uniform_pointer_bases(state, source_id, result_id):
    source_bases = state.uniform_pointer_bases.get(source_id)
    if source_bases is not None:
        state.uniform_pointer_bases[result_id] = source_bases


def _record_local_memory_root(state, result_id):
    state.local_memory_root_sets[int(result_id)] = frozenset({int(result_id)})


def _propagate_local_memory_roots(state, source_id, result_id):
    state.local_memory_root_sets[int(result_id)] = _local_memory_roots(state, source_id)


def _propagate_selected_local_memory_roots(state, source_ids, result_id):
    target_type = state.target_program.values[result_id].type
    if target_type.representation != "memdesc":
        return
    roots = set()
    for source_id in source_ids:
        roots.update(_local_memory_roots(state, source_id))
    state.local_memory_root_sets[int(result_id)] = frozenset(roots)


def _local_memory_roots(state, target_value_id):
    roots = state.local_memory_root_sets.get(int(target_value_id))
    if roots is not None:
        return roots
    return frozenset({int(target_value_id)})


def _local_memory_dependency_token(state, target_value_id, extra_tokens=()):
    roots = _local_memory_roots(state, target_value_id)
    dependency_roots = _local_memory_dependency_roots(state, roots)
    tokens = [state.local_memory_tokens[root] for root in sorted(dependency_roots) if root in state.local_memory_tokens]
    tokens.extend(tuple(extra_tokens))
    return _memory_dependency_token(state, tokens)


def _local_dma_issue_dependency_token(state, issue_dependencies):
    # Async direct-to-LDS issues are ordered by explicit async-token operands.
    # Local readers synchronize through local-memory tokens after async_wait.
    return _memory_dependency_token(state, issue_dependencies)


def _local_dma_component_issue_delay_options(state, op, component):
    """Return explicit commit-group issue metadata for one DMA request."""
    group_id = int(target_ir.attrs_dict(op).get("async_group_id", -1))
    if group_id < 0:
        return {}
    commits = tuple(
        candidate
        for candidate in state.target_program.ops
        if candidate.kind == "async_commit_group"
        and int(target_ir.attrs_dict(candidate).get("group_id", -1)) == group_id
    )
    if not commits:
        return {}
    if len(commits) != 1:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "DMA group has multiple commit operations",
            target_op_id=op.target_op_id,
        )
    schedule = target_ir.attrs_dict(commits[0])
    issue_group_size = int(schedule.get("issue_group_size", 0))
    delay_cycles = int(schedule.get("issue_delay_cycles", 0))
    if issue_group_size <= 0 or delay_cycles <= 0:
        return {}
    request = 1 + int(component)
    for candidate in state.target_program.ops:
        if int(candidate.target_op_id) >= int(op.target_op_id):
            break
        if candidate.kind != "buffer_load_to_local":
            continue
        candidate_attrs = target_ir.attrs_dict(candidate)
        if int(candidate_attrs.get("async_group_id", -1)) != group_id:
            continue
        request += int(candidate_attrs.get("component_count", 0))
    if request % issue_group_size:
        return {}
    options = {"issue_delay_cycles": delay_cycles}
    overlap_cycles = int(schedule.get("issue_delay_overlap_cycles", 0))
    if overlap_cycles:
        options["issue_delay_overlap_cycles"] = overlap_cycles
    skip_threshold = int(schedule.get("issue_delay_skip_thread_threshold", 0))
    if skip_threshold:
        options["issue_delay_skip_thread_threshold"] = skip_threshold
    return options


def _local_memory_access_dependency_token(
    state,
    target_value_id,
    access_kind,
    extra_tokens=(),
    *,
    ignore_async_writes=False,
    ready_async_write_roots=(),
):
    roots = _local_memory_roots(state, target_value_id)
    dependency_roots = _local_memory_dependency_roots(state, roots)
    ready_async_write_roots = frozenset(
        int(root) for root in ready_async_write_roots
    )
    tokens = []
    barrier_roots = []
    token_roots = []
    for root in sorted(dependency_roots):
        token = state.local_memory_tokens.get(root)
        if token is None:
            continue
        pending = state.local_memory_pending_accesses.get(root)
        if (
            _local_memory_access_includes(pending, "async_write")
            and (
                bool(ignore_async_writes)
                or int(root) in ready_async_write_roots
            )
        ):
            continue
        if pending == "read" and access_kind == "read":
            continue
        has_mma_boundary = root in state.pending_mma_read_boundaries
        if ((_local_memory_access_includes(pending, "mma_read") or has_mma_boundary)
                and access_kind in {"write", "async_write"}):
            token_roots.append(root)
            continue
        if pending is not None and _local_memory_access_needs_barrier(pending, access_kind):
            barrier_roots.append(root)
        else:
            tokens.append(token)
    if barrier_roots:
        barrier_token = _sync_pending_local_memory_accesses(state, roots=barrier_roots)
        tokens.append(barrier_token)
    if token_roots:
        tokens.extend(_consume_pending_local_memory_tokens(state, token_roots, "mma_read"))
    tokens.extend(tuple(extra_tokens))
    return _memory_dependency_token(state, tokens)


def _local_load_ready_async_write_roots(state, op, attrs):
    """Return LDS roots made ready by this load's dominating explicit wait.

    Readiness operands are inserted structurally by the token analysis.  They
    identify the completed DMA groups precisely, so a local load need not emit
    a second CTA barrier for those writes.  Other asynchronous or synchronous
    LDS hazards remain in the ordinary access state and are still honored.
    """
    readiness_count = int(attrs.get("readiness_dependency_count", 0))
    dependency_ids = tuple(int(target_id) for target_id in op.operands[1:])
    if readiness_count <= 0:
        return frozenset()
    if readiness_count > len(dependency_ids):
        fail(
            "TLXW_EMIT_LOCAL_READINESS",
            STAGE,
            "local load readiness segment exceeds its dependency operands",
            target_op_id=op.target_op_id,
        )
    readiness_ids = dependency_ids[-readiness_count:]
    # Keep this set exact.  Alias expansion belongs to the access side: a wait
    # for one slice must not make a different overlapping in-flight write look
    # completed merely because both share an aggregate allocation root.
    return _local_memory_roots_for_token_values(state, readiness_ids)


def _local_memory_access_needs_barrier(pending_access, access_kind):
    if _local_memory_access_includes(pending_access, "mma_read"):
        return False
    return (
        _local_memory_access_includes(pending_access, "write")
        or _local_memory_access_includes(pending_access, "async_write")
        or access_kind in {"write", "async_write"}
    )


def _local_memory_access_includes(access_kind, expected_kind):
    if access_kind == expected_kind:
        return True
    if access_kind == "write":
        return expected_kind in {"read", "write"}
    if access_kind == "read_async_write":
        return expected_kind in {"read", "async_write"}
    if access_kind == "mma_read":
        return expected_kind == "mma_read"
    if access_kind == "mma_read_async_write":
        return expected_kind in {"mma_read", "async_write"}
    return False


def _emit_cta_barrier(state, *tokens, boundary_ids=()):
    """Emit a CTA barrier and record the MFMA read boundaries it satisfies.

    An MFMA scheduling boundary is kept virtual until LDS is reused.  Any
    intervening CTA-wide barrier already orders every earlier MFMA payload read,
    though, regardless of why that barrier was emitted.  Remember the first
    such barrier for every boundary that dominates this emission point so a
    later LDS write can depend on it instead of materializing a duplicate.

    Structured control-flow emission snapshots these maps, which naturally
    limits reuse to barriers that dominate the later operation.
    """
    barrier_token = state.builder.barrier(*tokens)
    materialized_boundary_ids = tuple(dict.fromkeys((
        *(int(boundary_id) for boundary_id in boundary_ids),
        *(int(boundary_id) for boundary_id in state.pending_mma_read_boundaries.values()),
    )))
    for boundary_id in materialized_boundary_ids:
        state.materialized_mma_read_boundaries.setdefault(boundary_id, barrier_token)
    return barrier_token


def _consume_pending_local_memory_tokens(state, roots, expected_kind):
    # MMA payload values carry the LDS dependency into MFMA. A pipeline MFMA
    # boundary proves those reads have reached the consuming MFMA, but it is a
    # scheduling marker rather than a memory operation. Materialize the
    # input-free CTA barrier only when a later write actually reuses the slot;
    # this keeps the barrier late without making it consume the read tokens.
    if expected_kind == "mma_read":
        tokens, boundary_ids = _take_local_memory_read_frontier(state, roots)
        _clear_pending_local_memory_accesses(state, roots, expected_kind)
        _clear_pending_mma_read_boundaries(state, roots)
        return _materialize_mma_read_reuse_dependencies(
            state,
            tokens,
            boundary_ids,
        )

    tokens = []
    boundary_ids = []
    for root in roots:
        root = int(root)
        token = state.local_memory_tokens.get(root)
        # A dominating CTA barrier can clear the pending access while leaving
        # its materialized MFMA boundary available for a later LDS reuse.
        if expected_kind == "mma_read" and root in state.pending_mma_read_boundaries:
            boundary_ids.append(state.pending_mma_read_boundaries.pop(root))
        elif token is not None:
            tokens.append(token)
    _clear_pending_local_memory_accesses(state, roots, expected_kind)
    tokens = _unique_tokens(tokens)
    boundary_ids = tuple(dict.fromkeys(boundary_ids))
    if not tokens and not boundary_ids:
        return ()
    if expected_kind == "mma_read" and _kernel_has_multiple_waves(state):
        return _materialize_mma_read_reuse_dependencies(
            state,
            tokens,
            boundary_ids,
        )
    return (_join_memory_tokens(state, tokens), )


def _materialize_mma_read_reuse_dependencies(state, tokens, boundary_ids):
    """Resolve the LDS read frontier before its storage is overwritten.

    A read token only proves completion in the issuing wave.  Reusing LDS in a
    multi-wave workgroup therefore needs CTA convergence before the later write
    is issued.  This barrier orders readers against the write; it does not wait
    for async DMA completion, which remains exclusive to explicit wait_group.
    """
    tokens = _unique_tokens(tokens)
    boundary_ids = tuple(dict.fromkeys(int(boundary_id) for boundary_id in boundary_ids))
    if not tokens and not boundary_ids:
        return ()

    barrier_tokens = [
        state.materialized_mma_read_boundaries[boundary_id]
        for boundary_id in boundary_ids
        if boundary_id in state.materialized_mma_read_boundaries
    ]
    missing_boundary_ids = tuple(
        boundary_id for boundary_id in boundary_ids
        if boundary_id not in state.materialized_mma_read_boundaries
    )
    if missing_boundary_ids or (tokens and _kernel_has_multiple_waves(state)):
        barrier_tokens.append(_emit_cta_barrier(
            state,
            *tokens,
            boundary_ids=missing_boundary_ids,
        ))
        tokens = ()
    dependencies = _unique_tokens((*tokens, *barrier_tokens))
    return ((_join_memory_tokens(state, dependencies), ) if dependencies else ())


def _kernel_has_multiple_waves(state):
    return int(state.target_program.kernel.num_warps or 1) > 1


def _emit_sched_barrier(state, op):
    attrs = target_ir.attrs_dict(op)
    mask = int(attrs.get("mask", 0))
    state.builder.sched_barrier()
    if (
        attrs.get("border") != "mfma"
        or mask != 0
        or not _kernel_has_multiple_waves(state)
    ):
        return
    roots = tuple(
        int(root)
        for root, access_kind in sorted(state.local_memory_pending_accesses.items())
        if _local_memory_access_includes(access_kind, "mma_read") and int(root) in state.local_memory_tokens
    )
    if not roots:
        return
    boundary_id = state.next_mma_read_boundary_id
    state.next_mma_read_boundary_id += 1
    for root in roots:
        state.pending_mma_read_boundaries[int(root)] = boundary_id


def _emit_cond_barrier(state, op):
    (condition, ) = _operand_values(state, op, 1)
    if _is_scalar_i1_value(state, condition):
        with state.builder.if_(condition):
            state.builder.barrier()
        return
    lane_width = int(state.target_program.kernel.threads_per_warp or 64)
    (predicate, ) = _as_mask_predicate_components(
        state,
        condition,
        1,
        lane_width,
        op,
    )
    with state.builder.where(predicate):
        state.builder.barrier()


def _emit_set_priority(state, op):
    attrs = target_ir.attrs_dict(op)
    state.builder.set_priority(int(attrs["priority"]))


def _emit_barrier(state, op):
    attrs = target_ir.attrs_dict(op)
    tokens = tuple(
        _require_value(state, target_value_id, op)
        for target_value_id in op.operands
    )
    barrier_token = _emit_cta_barrier(state, *tokens)
    # A full-memory CTA barrier is also a structural boundary for later memory
    # issue.  Its token orders explicit consumers, while the scheduling cut
    # prevents an otherwise-independent operation from being hoisted to the
    # pre-barrier epoch without turning the barrier into that operation's data
    # dependency. Local-memory publication is represented by explicit tokens
    # and retains legal issue overlap across the physical rendezvous.
    if int(attrs.get("address_space", 0)) == _FULL_MEMORY_BARRIER_ADDRESS_SPACE:
        state.builder.sched_barrier()
    if op.results:
        state.values[_single_result(op)] = barrier_token
        roots = _local_memory_roots_for_token_values(state, op.operands)
        if roots:
            state.token_local_memory_root_sets[int(op.results[0])] = roots
    synchronized_roots = []
    for root, pending in tuple(state.local_memory_pending_accesses.items()):
        if pending == "async_write":
            continue
        synchronized_roots.append(int(root))
        if pending in {"read_async_write", "mma_read_async_write"}:
            state.local_memory_pending_accesses[int(root)] = "async_write"
            continue
        state.local_memory_tokens[int(root)] = barrier_token
        state.local_memory_pending_accesses.pop(int(root), None)
    _clear_local_memory_read_tokens(state, synchronized_roots)


def _clear_pending_local_memory_accesses(state, roots, expected_kind):
    for root in roots:
        root = int(root)
        pending = state.local_memory_pending_accesses.get(root)
        if pending is None or not _local_memory_access_includes(pending, expected_kind):
            continue
        if pending == expected_kind:
            state.local_memory_pending_accesses.pop(root, None)
        elif pending == "mma_read_async_write" and expected_kind == "mma_read":
            state.local_memory_pending_accesses[root] = "async_write"
        elif pending == "read_async_write" and expected_kind == "read":
            state.local_memory_pending_accesses[root] = "async_write"


def _clear_pending_mma_read_boundaries(state, roots):
    for root in _local_memory_dependency_roots(state, roots):
        state.pending_mma_read_boundaries.pop(int(root), None)


def _sync_pending_local_memory_accesses(state, extra_tokens=(), roots=None):
    if roots is None:
        roots_to_sync = tuple(int(root) for root in sorted(state.local_memory_pending_accesses))
    else:
        roots_to_sync = tuple(int(root) for root in sorted({int(root) for root in roots}))
    sync_roots = tuple(
        int(root) for root in roots_to_sync
        if root in state.local_memory_tokens)
    read_roots = tuple(
        int(root) for root in roots_to_sync
        if (
            _local_memory_access_includes(
                state.local_memory_pending_accesses.get(int(root)),
                "read",
            )
            or _local_memory_access_includes(
                state.local_memory_pending_accesses.get(int(root)),
                "mma_read",
            )
        )
    )
    read_tokens, boundary_ids = _take_local_memory_read_frontier(
        state,
        read_roots,
    )
    state_tokens = tuple(
        state.local_memory_tokens[root]
        for root in sync_roots
        if (
            _local_memory_access_includes(
                state.local_memory_pending_accesses.get(int(root)),
                "write",
            )
            or _local_memory_access_includes(
                state.local_memory_pending_accesses.get(int(root)),
                "async_write",
            )
        )
    )
    tokens = _unique_tokens((*state_tokens, *read_tokens, *tuple(extra_tokens)))
    if not tokens and not boundary_ids:
        if roots is None:
            state.local_memory_pending_accesses.clear()
        else:
            for root in roots_to_sync:
                state.local_memory_pending_accesses.pop(int(root), None)
        _clear_local_memory_read_tokens(state, roots_to_sync)
        return state.builder.token()
    if boundary_ids:
        boundary_dependencies = _materialize_mma_read_reuse_dependencies(
            state,
            read_tokens,
            boundary_ids,
        )
        non_read_tokens = _unique_tokens((*state_tokens, *tuple(extra_tokens)))
        if boundary_dependencies and not non_read_tokens:
            barrier_token = _join_memory_tokens(state, boundary_dependencies)
        elif boundary_dependencies or non_read_tokens:
            barrier_token = _emit_cta_barrier(
                state,
                *boundary_dependencies,
                *non_read_tokens,
            )
        else:
            barrier_token = state.builder.token()
    else:
        barrier_token = _emit_cta_barrier(state, *tokens)
    for root in sync_roots:
        state.local_memory_tokens[int(root)] = barrier_token
        state.local_memory_pending_accesses.pop(int(root), None)
    if roots is None:
        state.local_memory_pending_accesses.clear()
    _clear_local_memory_read_tokens(state, roots_to_sync)
    return barrier_token


def _set_local_memory_token(state, target_value_id, token):
    for root in _local_memory_roots(state, target_value_id):
        state.local_memory_tokens[int(root)] = token
        state.local_memory_pending_accesses.pop(int(root), None)


def _set_local_memory_roots_token(state, roots, token):
    for root in roots:
        state.local_memory_tokens[int(root)] = token
        state.local_memory_pending_accesses.pop(int(root), None)


def _set_local_memory_roots_committed_token(state, roots, token):
    # async_commit_group groups in-flight LDS writes; only async_wait/barrier
    # makes the written LDS contents available to later local reads.
    for root in roots:
        root = int(root)
        pending = state.local_memory_pending_accesses.get(root)
        state.local_memory_tokens[root] = token
        if pending is not None:
            state.local_memory_pending_accesses[root] = pending


def _set_local_memory_access_token(
    state,
    target_value_id,
    token,
    access_kind,
):
    roots = _local_memory_roots(state, target_value_id)
    _record_local_memory_access_token(state, roots, token)
    if access_kind in {"read", "mma_read"}:
        _record_local_memory_read_token(state, roots, token)
    else:
        # The write dependency has consumed every earlier read of the aliased
        # storage.  A later overwrite must wait only for reads after this one.
        _clear_local_memory_read_tokens(state, roots)
    if access_kind == "mma_read":
        # A new payload read supersedes any older MFMA-boundary marker for an
        # aliased LDS slot. The older boundary cannot prove this read complete.
        _clear_pending_mma_read_boundaries(state, roots)
    for root in roots:
        root = int(root)
        pending = state.local_memory_pending_accesses.get(root)
        previous_token = state.local_memory_tokens.get(root)
        # Reads do not change the contents or availability of the LDS slot.
        # Keep their completion tokens solely in local_memory_read_tokens so
        # independent reads remain siblings of the same write/wait frontier.
        # A later overwrite consumes that separate read frontier.
        stored_token = (
            previous_token
            if access_kind in {"read", "mma_read"} and previous_token is not None
            else token
        )
        state.local_memory_tokens[root] = stored_token
        state.local_memory_pending_accesses[root] = _merge_local_memory_pending_access(pending, access_kind)
def _record_local_memory_access_token(state, roots, token):
    for dependency_root in _local_memory_dependency_roots(state, roots):
        dependency_root = int(dependency_root)
        if dependency_root not in state.local_memory_allocations:
            continue
        tokens = state.local_memory_access_tokens.get(dependency_root, ())
        if any(existing is token for existing in tokens):
            continue
        state.local_memory_access_tokens[dependency_root] = (*tokens, token)


def _local_memory_allocation_roots(state, roots):
    return tuple(sorted({
        int(dependency_root)
        for root in roots
        for dependency_root in _local_memory_dependency_roots(state, (root, ))
        if int(dependency_root) in state.local_memory_allocations
    }))


def _record_local_memory_read_token(state, roots, token):
    for root in roots:
        root = int(root)
        tokens = state.local_memory_read_tokens.get(root, ())
        if any(existing is token for existing in tokens):
            continue
        state.local_memory_read_tokens[root] = (*tokens, token)


def _clear_local_memory_read_tokens(state, roots):
    for dependency_root in _local_memory_dependency_roots(state, roots):
        state.local_memory_read_tokens.pop(int(dependency_root), None)


def _take_local_memory_read_frontier(state, roots):
    tokens = []
    boundary_ids = []
    for dependency_root in _local_memory_dependency_roots(state, roots):
        dependency_root = int(dependency_root)
        boundary_id = state.pending_mma_read_boundaries.get(dependency_root)
        if boundary_id is not None:
            # The MFMA scheduling boundary supersedes raw LDS read tokens for
            # this alias. Materializing it as an input-free CTA barrier avoids
            # turning a scheduling fact into an LDS wait.
            boundary_ids.append(int(boundary_id))
        else:
            tokens.extend(state.local_memory_read_tokens.get(dependency_root, ()))
        state.local_memory_read_tokens.pop(dependency_root, None)
    return _unique_tokens(tokens), tuple(dict.fromkeys(boundary_ids))


def _local_memory_allocation_read_dependency(state, root):
    return _join_memory_tokens(
        state,
        tuple(
            token
            for dependency_root in sorted(
                _local_memory_dependency_roots(state, (int(root), )))
            for token in state.local_memory_read_tokens.get(int(dependency_root), ())
        ),
    )


def _local_memory_allocation_access_dependency(state, root):
    root = int(root)
    tokens = list(state.local_memory_access_tokens.get(root, ()))
    tokens.extend(
        state.local_memory_tokens[dependency_root]
        for dependency_root in sorted(
            _local_memory_dependency_roots(state, (root, )))
        if dependency_root in state.local_memory_tokens
    )
    return _join_memory_tokens(state, tokens)


def _merge_local_memory_pending_access(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    if lhs == "write" or rhs == "write":
        return "write"
    if {lhs, rhs} == {"read", "async_write"}:
        return "read_async_write"
    if lhs == "read_async_write" or rhs == "read_async_write":
        return "read_async_write"
    if lhs == "read" or rhs == "read":
        return "read"
    if {lhs, rhs} == {"mma_read", "async_write"}:
        return "mma_read_async_write"
    if lhs == "mma_read_async_write" or rhs == "mma_read_async_write":
        return "mma_read_async_write"
    if lhs == "mma_read" or rhs == "mma_read":
        return "mma_read"
    if lhs == "async_write" or rhs == "async_write":
        return "async_write"
    return lhs


def _local_memory_dependency_roots(state, roots):
    dependency_roots = set()
    for root in roots:
        root = int(root)
        dependency_roots.add(root)
        root_interval = state.static_local_memory_root_intervals.get(root)
        circular_region = state.circular_local_memory_root_regions.get(root)
        if root_interval is not None:
            base_root, offset, size = root_interval
            dependency_roots.add(int(base_root))
            for static_root, other_interval in state.static_local_memory_root_intervals.items():
                other_base, other_offset, other_size = other_interval
                if int(other_base) != int(base_root):
                    continue
                if _byte_intervals_overlap(int(offset), int(size), int(other_offset), int(other_size)):
                    dependency_roots.add(int(static_root))
            dependency_roots.update(
                int(dynamic_root)
                for dynamic_root, region in state.circular_local_memory_root_regions.items()
                if int(region.base_root) == int(base_root))
            continue
        if circular_region is not None:
            base_root = int(circular_region.base_root)
            dependency_roots.add(base_root)
            dependency_roots.update(
                int(static_root)
                for static_root, (other_base, _offset, _size) in state.static_local_memory_root_intervals.items()
                if int(other_base) == base_root)
            for dynamic_root, other_region in state.circular_local_memory_root_regions.items():
                if int(other_region.base_root) != base_root:
                    continue
                if not _circular_local_memory_regions_disjoint(circular_region, other_region):
                    dependency_roots.add(int(dynamic_root))
            continue
        dependency_roots.update(
            int(static_root)
            for static_root, (base_root, _offset, _size) in state.static_local_memory_root_intervals.items()
            if int(base_root) == root)
        dependency_roots.update(
            int(dynamic_root)
            for dynamic_root, region in state.circular_local_memory_root_regions.items()
            if int(region.base_root) == root)
    return frozenset(dependency_roots)


def _circular_local_memory_regions_disjoint(lhs, rhs):
    return (int(lhs.base_root) == int(rhs.base_root)
            and int(lhs.index_base_target_id) == int(rhs.index_base_target_id)
            and int(lhs.slot_count) == int(rhs.slot_count)
            and int(lhs.slot_stride_bytes) == int(rhs.slot_stride_bytes)
            and int(lhs.phase) != int(rhs.phase))


def _is_aggregate_local_memory_root(state, root):
    root = int(root)
    if root in state.static_local_memory_root_intervals:
        return False
    return any(
        int(base_root) == root
        for base_root, _offset, _size in state.static_local_memory_root_intervals.values()
    )


def _is_static_descendant_local_memory_root(state, root, candidate):
    interval = state.static_local_memory_root_intervals.get(int(candidate))
    if interval is None:
        return False
    base_root, _offset, _size = interval
    return int(base_root) == int(root)


def _byte_intervals_overlap(lhs_offset, lhs_size, rhs_offset, rhs_size):
    return int(lhs_offset) < int(rhs_offset) + int(rhs_size) and int(rhs_offset) < int(lhs_offset) + int(lhs_size)


def _static_local_memory_roots(state, source_roots, static_byte_offset, byte_size):
    static_byte_offset = int(static_byte_offset)
    byte_size = int(byte_size)
    roots = []
    for source_root in source_roots:
        base_root, base_offset, _base_size = state.static_local_memory_root_intervals.get(
            int(source_root),
            (int(source_root), 0, None),
        )
        key = (int(base_root), int(base_offset) + static_byte_offset, byte_size)
        root = state.static_local_memory_roots.get(key)
        if root is None:
            root = int(state.next_local_memory_root)
            state.next_local_memory_root -= 1
            state.static_local_memory_roots[key] = root
            state.static_local_memory_root_intervals[root] = key
        roots.append(root)
    return frozenset(roots)


def _circular_local_memory_roots(
    state,
    source_roots,
    index_base_target_id,
    phase,
    slot_count,
    slot_stride_bytes,
):
    roots = []
    for source_root in source_roots:
        source_root = int(source_root)
        if (source_root in state.static_local_memory_root_intervals
                or source_root in state.circular_local_memory_root_regions):
            return frozenset(int(root) for root in source_roots)
        key = (
            source_root,
            int(index_base_target_id),
            int(phase),
            int(slot_count),
            int(slot_stride_bytes),
        )
        root = state.circular_local_memory_roots.get(key)
        if root is None:
            root = int(state.next_local_memory_root)
            state.next_local_memory_root -= 1
            state.circular_local_memory_roots[key] = root
            state.circular_local_memory_root_regions[root] = _CircularLocalMemoryRegion(*key)
        roots.append(root)
    return frozenset(roots)


def _target_value_def_op(state, target_value_id):
    if not state.target_value_def_ops:
        for op in state.target_program.ops:
            for result_id in op.results:
                state.target_value_def_ops[int(result_id)] = op
    return state.target_value_def_ops.get(int(target_value_id))


def _target_constant_int(state, target_value_id):
    op = _target_value_def_op(state, target_value_id)
    if op is None or op.kind != "constant":
        return None
    value = target_ir.attrs_dict(op).get("value")
    return int(value) if type(value) is int else None


def _target_additive_base(state, target_value_id, seen=None):
    target_value_id = int(target_value_id)
    seen = set() if seen is None else set(seen)
    if target_value_id in seen:
        return target_value_id, 0
    seen.add(target_value_id)
    op = _target_value_def_op(state, target_value_id)
    if op is None or op.kind != "binary" or len(op.operands) != 2:
        return target_value_id, 0
    attrs = target_ir.attrs_dict(op)
    operation = attrs.get("operation")
    # remui phase equivalence needs unsigned no-wrap. Signed no-wrap alone is
    # insufficient when the circular depth does not divide the integer range.
    if operation not in {"addi", "subi"} or not attrs.get("nuw"):
        return target_value_id, 0
    lhs, rhs = (int(operand) for operand in op.operands)
    lhs_constant = _target_constant_int(state, lhs)
    rhs_constant = _target_constant_int(state, rhs)
    if operation == "addi" and rhs_constant is not None:
        base, offset = _target_additive_base(state, lhs, seen)
        return base, int(offset) + int(rhs_constant)
    if operation == "addi" and lhs_constant is not None:
        base, offset = _target_additive_base(state, rhs, seen)
        return base, int(offset) + int(lhs_constant)
    if operation == "subi" and rhs_constant is not None:
        base, offset = _target_additive_base(state, lhs, seen)
        return base, int(offset) - int(rhs_constant)
    return target_value_id, 0


def _circular_index_phase(state, target_value_id, slot_count, seen=None):
    target_value_id = int(target_value_id)
    slot_count = int(slot_count)
    if slot_count <= 1:
        return None
    seen = set() if seen is None else set(seen)
    if target_value_id in seen:
        return None
    seen.add(target_value_id)
    op = _target_value_def_op(state, target_value_id)
    if op is None or op.kind != "binary" or len(op.operands) != 2:
        return None
    attrs = target_ir.attrs_dict(op)
    operation = attrs.get("operation")
    lhs, rhs = (int(operand) for operand in op.operands)
    # In a two-slot ring, complementing a normalized phase is exactly the
    # other phase: 1 - (x mod 2) == (x + 1) mod 2.  This does not require an
    # arithmetic no-wrap flag because remui proves that the subtrahend is 0 or
    # 1.  Recognizing the identity keeps double-buffer reads and refills in
    # distinct structural alias classes.
    if (
        operation == "subi"
        and slot_count == 2
        and _target_constant_int(state, lhs) == 1
    ):
        nested = _circular_index_phase(state, rhs, slot_count, seen)
        if nested is None:
            return None
        base, nested_phase = nested
        return int(base), (int(nested_phase) + 1) % slot_count
    if operation != "remui":
        return None
    modulus = _target_constant_int(state, rhs)
    if modulus != slot_count:
        return None
    base, offset = _target_additive_base(state, lhs)
    nested = _circular_index_phase(state, base, slot_count, seen)
    if nested is not None:
        base, nested_phase = nested
        offset += int(nested_phase)
    return int(base), int(offset) % slot_count


def _memdesc_index_result_roots(state, source_roots, attrs, index_target_id=None):
    static_byte_offset = attrs.get("static_byte_offset")
    element_byte_width = attrs.get("element_byte_width")
    elements_per_slot = attrs.get("elements_per_slot")
    if static_byte_offset is not None:
        if element_byte_width is None or elements_per_slot is None:
            return frozenset(int(root) for root in source_roots)
        byte_size = int(element_byte_width) * int(elements_per_slot)
        if byte_size <= 0:
            return frozenset(int(root) for root in source_roots)
        return _static_local_memory_roots(
            state,
            source_roots,
            int(static_byte_offset),
            byte_size,
        )
    slot_count = attrs.get("slot_count")
    if (index_target_id is None or element_byte_width is None
            or elements_per_slot is None or slot_count is None):
        return frozenset(int(root) for root in source_roots)
    slot_stride_bytes = int(element_byte_width) * int(elements_per_slot)
    circular_index = _circular_index_phase(state, index_target_id, slot_count)
    if slot_stride_bytes <= 0 or circular_index is None:
        return frozenset(int(root) for root in source_roots)
    index_base_target_id, phase = circular_index
    return _circular_local_memory_roots(
        state,
        source_roots,
        index_base_target_id,
        phase,
        int(slot_count),
        slot_stride_bytes,
    )


def _record_token_local_memory_roots(state, token_target_id, memdesc_target_id):
    state.token_local_memory_root_sets[int(token_target_id)] = _local_memory_roots(state, memdesc_target_id)


def _propagate_token_local_memory_roots(
    state,
    source_token_id,
    result_token_id,
    *,
    preserved_local_memory_roots=(),
):
    roots = state.token_local_memory_root_sets.get(int(source_token_id))
    if not roots:
        return
    state.token_local_memory_root_sets[int(result_token_id)] = roots
    preserved = frozenset(int(root) for root in preserved_local_memory_roots)
    roots_to_set = frozenset(int(root) for root in roots if int(root) not in preserved)
    if roots_to_set:
        _set_local_memory_roots_token(state, roots_to_set, _require_value(state, result_token_id, None))


def _local_memory_roots_for_token_values(state, token_target_ids):
    roots = set()
    for token_target_id in token_target_ids:
        roots.update(state.token_local_memory_root_sets.get(int(token_target_id), ()))
    return frozenset(roots)


def _local_memory_roots_touched_by_region(state, region_id, *, include_root_sets=False):
    root_sets = dict(state.local_memory_root_sets)
    touched_roots = set()
    read_roots = set()
    implicit_root_accesses = {}

    def record_implicit_access(roots, access_kind):
        for root in roots:
            root = int(root)
            implicit_root_accesses[root] = _merge_local_memory_pending_access(
                implicit_root_accesses.get(root),
                access_kind,
            )

    def roots_for(target_value_id):
        target_value_id = int(target_value_id)
        roots = root_sets.get(target_value_id)
        if roots is not None:
            return roots
        target_type = state.target_program.values[target_value_id].type
        if target_type.representation == "memdesc":
            return frozenset({target_value_id})
        return frozenset()

    def visit_region(current_region_id):
        region = state.target_program.regions[int(current_region_id)]
        for op_id in region.op_ids:
            current_op = state.target_program.ops[int(op_id)]
            if current_op.kind == "local_alloc":
                for result_id in current_op.results:
                    root_sets[int(result_id)] = frozenset({int(result_id)})
                continue
            if current_op.kind in {"memdesc_index", "memdesc_view"}:
                if current_op.results:
                    source_roots = roots_for(current_op.operands[0])
                    if current_op.kind == "memdesc_index":
                        root_sets[int(current_op.results[0])] = _memdesc_index_result_roots(
                            state,
                            source_roots,
                            target_ir.attrs_dict(current_op),
                            current_op.operands[1],
                        )
                    else:
                        root_sets[int(current_op.results[0])] = source_roots
                continue
            if current_op.kind == "select" and current_op.results:
                result_id = int(current_op.results[0])
                target_type = state.target_program.values[result_id].type
                if target_type.representation == "memdesc":
                    roots = set()
                    for source_id in current_op.operands[1:]:
                        roots.update(roots_for(source_id))
                    root_sets[result_id] = frozenset(roots)
                continue
            if current_op.kind in {"local_load", "local_load_mma_payload"}:
                current_attrs = target_ir.attrs_dict(current_op)
                if (
                    bool(current_attrs.get("protocol_tracked", False))
                    and bool(current_attrs.get("synced_via_async_wait", False))
                ):
                    # Its completion is an explicit target result carried by
                    # SCF.  Do not synthesize a duplicate emitter-only read
                    # frontier for the same access.
                    continue
                roots = roots_for(current_op.operands[0])
                touched_roots.update(roots)
                read_roots.update(roots)
                access_kind = (
                    "mma_read"
                    if current_op.kind == "local_load_mma_payload"
                    else "read"
                )
                record_implicit_access(roots, access_kind)
            elif current_op.kind == "local_store":
                roots = roots_for(current_op.operands[1])
                touched_roots.update(roots)
                record_implicit_access(roots, "write")
            elif current_op.kind == "buffer_load_to_local":
                roots = roots_for(current_op.operands[0])
                touched_roots.update(roots)
                # All amdg.buffer_load_to_local lowering modes write LDS.  The
                # DMA path returns an async token; scalarized fallback is a
                # normal LDS write and must remain a hard write dependency.
                mode = target_ir.attrs_dict(current_op).get("mode")
                access_kind = "write" if mode == "scalarized_load_store" else "async_write"
                record_implicit_access(roots, access_kind)
            for nested_region_id in current_op.region_ids:
                visit_region(nested_region_id)

    visit_region(region_id)
    result = (frozenset(touched_roots), {
        int(root): access_kind
        for root, access_kind in implicit_root_accesses.items()
    })
    if include_root_sets:
        return (*result, root_sets, frozenset(read_roots))
    return result


def _target_token_local_memory_roots(
    state,
    target_value_id,
    root_sets,
    seen=None,
    *,
    token_root_sets=None,
):
    """Infer the LDS destinations represented by a target async token."""
    target_value_id = int(target_value_id)
    if token_root_sets is not None:
        known = token_root_sets.get(target_value_id)
        if known is not None:
            return known
    known = state.token_local_memory_root_sets.get(target_value_id)
    if known is not None:
        return known
    seen = set() if seen is None else set(seen)
    if target_value_id in seen:
        return None
    seen.add(target_value_id)
    op = _target_value_def_op(state, target_value_id)
    if op is None:
        return None
    if op.kind == "buffer_load_to_local":
        return root_sets.get(int(op.operands[0]), _local_memory_roots(state, op.operands[0]))
    if op.kind in {"async_commit_group", "token_join", "issue_token"}:
        roots = set()
        for operand in op.operands:
            operand_roots = _target_token_local_memory_roots(
                state,
                operand,
                root_sets,
                seen,
                token_root_sets=token_root_sets,
            )
            if operand_roots is None:
                return None
            roots.update(operand_roots)
        return frozenset(roots)
    if op.kind == "async_wait":
        completed_count = int(
            target_ir.attrs_dict(op)["completed_group_dependency_count"]
        )
        roots = set()
        for operand in op.operands[:completed_count]:
            operand_roots = _target_token_local_memory_roots(
                state,
                operand,
                root_sets,
                seen,
                token_root_sets=token_root_sets,
            )
            if operand_roots is None:
                return None
            roots.update(operand_roots)
        return frozenset(roots)
    if op.kind == "token":
        return frozenset()
    return None


def _local_memory_root_interval_at_loop_iteration(
    state,
    root,
    induction_target_id,
    induction_value,
):
    root = int(root)
    static_interval = state.static_local_memory_root_intervals.get(root)
    if static_interval is not None:
        return tuple(int(value) for value in static_interval)
    circular = state.circular_local_memory_root_regions.get(root)
    if circular is None or int(circular.index_base_target_id) != int(induction_target_id):
        return None
    slot = (int(induction_value) + int(circular.phase)) % int(circular.slot_count)
    return (
        int(circular.base_root),
        slot * int(circular.slot_stride_bytes),
        int(circular.slot_stride_bytes),
    )


def _loop_carried_token_roots(
    state,
    init_roots,
    yield_roots,
    induction_target_id,
    lower_target_id,
    step_target_id,
):
    """Express a yielded circular destination in the next iteration's phase."""
    init_roots = frozenset(int(root) for root in init_roots)
    yield_roots = frozenset(int(root) for root in yield_roots)
    # Stable destinations do not need the induction-value interval proof used
    # below for circular aliases.  In particular, a token for an aggregate LDS
    # allocation remains a token for that allocation when carried through a
    # loop.  Dropping this identity made an explicit async group lose its LDS
    # destination at the block boundary and forced later loads onto a redundant
    # bridge-owned state token.
    if (
        init_roots == yield_roots
        and all(
            int(root) not in state.circular_local_memory_root_regions
            for root in init_roots
        )
    ):
        return init_roots
    lower = _target_constant_int(state, lower_target_id)
    step = _target_constant_int(state, step_target_id)
    if lower is None or step is None:
        return None
    carried_roots = []
    for root in yield_roots:
        root = int(root)
        circular = state.circular_local_memory_root_regions.get(root)
        if circular is None:
            carried_roots.append(root)
            continue
        if int(circular.index_base_target_id) != int(induction_target_id):
            return None
        carried_roots.extend(_circular_local_memory_roots(
            state,
            (int(circular.base_root), ),
            induction_target_id,
            (int(circular.phase) - int(step)) % int(circular.slot_count),
            int(circular.slot_count),
            int(circular.slot_stride_bytes),
        ))
    carried_roots = frozenset(carried_roots)
    init_intervals = {
        _local_memory_root_interval_at_loop_iteration(
            state,
            root,
            induction_target_id,
            lower,
        )
        for root in init_roots
    }
    carried_intervals = {
        _local_memory_root_interval_at_loop_iteration(
            state,
            root,
            induction_target_id,
            lower,
        )
        for root in carried_roots
    }
    if None in init_intervals or init_intervals != carried_intervals:
        return None
    return carried_roots


def _loop_exit_token_roots(
    state,
    yield_roots,
    induction_target_id,
    lower_target_id,
    upper_target_id,
    step_target_id,
):
    """Resolve circular destinations to static slots for a constant-trip loop."""
    lower = _target_constant_int(state, lower_target_id)
    upper = _target_constant_int(state, upper_target_id)
    step = _target_constant_int(state, step_target_id)
    if lower is None or upper is None or step is None or step <= 0 or lower >= upper:
        return yield_roots
    last_induction = lower + ((upper - lower - 1) // step) * step
    exit_roots = []
    for root in yield_roots:
        root = int(root)
        circular = state.circular_local_memory_root_regions.get(root)
        if circular is None:
            exit_roots.append(root)
            continue
        if int(circular.index_base_target_id) != int(induction_target_id):
            return yield_roots
        slot = (last_induction + int(circular.phase)) % int(circular.slot_count)
        exit_roots.extend(_static_local_memory_roots(
            state,
            (int(circular.base_root), ),
            slot * int(circular.slot_stride_bytes),
            int(circular.slot_stride_bytes),
        ))
    return frozenset(exit_roots)


def _loop_token_root_sets(state, op, region, region_root_sets):
    block_root_sets = {}
    result_root_sets = {}
    induction_target_id = int(region.block_arg_ids[0])
    for init_id, block_arg_id, yield_id, result_id in zip(
        op.operands[3:],
        region.block_arg_ids[1:],
        region.yield_value_ids,
        op.results,
    ):
        target_type = state.target_program.values[int(init_id)].type
        if target_type.representation != "token":
            continue
        init_roots = state.token_local_memory_root_sets.get(int(init_id))
        yield_roots = _target_token_local_memory_roots(
            state,
            yield_id,
            region_root_sets,
        )
        if init_roots is None or yield_roots is None:
            continue
        carried_roots = _loop_carried_token_roots(
            state,
            init_roots,
            yield_roots,
            induction_target_id,
            op.operands[0],
            op.operands[2],
        )
        if carried_roots is not None:
            block_root_sets[int(block_arg_id)] = carried_roots
        result_root_sets[int(result_id)] = _loop_exit_token_roots(
            state,
            yield_roots,
            induction_target_id,
            op.operands[0],
            op.operands[1],
            op.operands[2],
        )
    return block_root_sets, result_root_sets


def _loop_local_memory_state_carry_required(
    state,
    region,
    candidate_roots,
    implicit_root_accesses,
    outer_pending_accesses,
    region_root_sets,
    loop_token_block_root_sets,
):
    """Return whether the loop body can consume its incoming LDS slot state.

    Async DMA queue state is carried explicitly by the source loop.  LDS read
    completion is carried separately as the overwrite frontier.  The bridge's
    per-slot state is therefore needed only when a synchronous access can see
    the backedge state before an explicit wait/barrier replaces it.  Keeping
    this as a target-IR dataflow fact avoids leaking a recurrent token merely
    because the region happens to mention LDS.
    """
    candidate_roots = frozenset(int(root) for root in candidate_roots)
    if not candidate_roots:
        return False

    synchronized_roots = set()
    directly_synchronized_roots = set()
    token_root_sets = dict(state.token_local_memory_root_sets)
    token_root_sets.update({
        int(target_id): frozenset(int(root) for root in roots)
        for target_id, roots in loop_token_block_root_sets.items()
    })

    def roots_for(target_value_id):
        target_value_id = int(target_value_id)
        roots = region_root_sets.get(target_value_id)
        if roots is not None:
            return frozenset(int(root) for root in roots)
        target_type = state.target_program.values[target_value_id].type
        if target_type.representation == "memdesc":
            return frozenset({target_value_id})
        return frozenset()

    def aliases_candidates(roots):
        for root in roots:
            dependencies = _local_memory_dependency_roots(state, (int(root), ))
            if not candidate_roots.isdisjoint(dependencies):
                return True
            if any(
                int(root) in _local_memory_dependency_roots(state, (candidate, ))
                for candidate in candidate_roots
            ):
                return True
        return False

    def is_synchronized(roots):
        if not roots:
            return False
        for root in roots:
            root = int(root)
            # An exact slot wait can mention its aggregate allocation as an
            # alias dependency, but does not prove that every byte of an
            # aggregate memdesc is ready.
            if _is_aggregate_local_memory_root(state, root):
                if root not in directly_synchronized_roots:
                    return False
            elif root not in synchronized_roots:
                return False
        return True

    def has_backedge_synchronous_write(roots):
        for root in roots:
            for dependency_root in _local_memory_dependency_roots(state, (int(root), )):
                access = _merge_local_memory_pending_access(
                    outer_pending_accesses.get(int(dependency_root)),
                    implicit_root_accesses.get(int(dependency_root)),
                )
                if _local_memory_access_includes(access, "write"):
                    return True
        return False

    for op_id in region.op_ids:
        current_op = state.target_program.ops[int(op_id)]
        if current_op.kind == "async_wait":
            waited_roots = set()
            complete = True
            for operand in current_op.operands:
                operand_roots = _target_token_local_memory_roots(
                    state,
                    operand,
                    region_root_sets,
                    token_root_sets=token_root_sets,
                )
                if operand_roots is None:
                    complete = False
                    break
                waited_roots.update(int(root) for root in operand_roots)
            if complete and waited_roots:
                directly_synchronized_roots.update(waited_roots)
                synchronized_roots.update(
                    _local_memory_dependency_roots(state, waited_roots)
                )
            continue
        if current_op.kind == "barrier":
            directly_synchronized_roots.update(candidate_roots)
            synchronized_roots.update(candidate_roots)
            continue

        if current_op.region_ids:
            nested_roots = set()
            for nested_region_id in current_op.region_ids:
                touched, _accesses = _local_memory_roots_touched_by_region(
                    state,
                    nested_region_id,
                )
                nested_roots.update(int(root) for root in touched)
            if aliases_candidates(nested_roots):
                # Branch/loop path coverage needs a proper structured join.
                # Retain the incoming state until such a proof is available.
                return True

        access_roots = frozenset()
        synchronous_access = False
        if current_op.kind in {"local_load", "local_load_mma_payload"}:
            current_attrs = target_ir.attrs_dict(current_op)
            if (
                bool(current_attrs.get("protocol_tracked", False))
                and bool(current_attrs.get("synced_via_async_wait", False))
            ):
                continue
            access_roots = roots_for(current_op.operands[0])
            synchronous_access = True
        elif current_op.kind == "local_store":
            access_roots = roots_for(current_op.operands[1])
            synchronous_access = True
        elif current_op.kind == "buffer_load_to_local":
            access_roots = roots_for(current_op.operands[0])
            mode = target_ir.attrs_dict(current_op).get("mode")
            synchronous_access = mode == "scalarized_load_store"
            if not synchronous_access and has_backedge_synchronous_write(access_roots):
                # DMA issues never depend on earlier async DMA completion, but
                # they still cannot bypass a synchronous writer to the same LDS.
                synchronous_access = True
        if (
            synchronous_access
            and aliases_candidates(access_roots)
            and not is_synchronized(access_roots)
        ):
            return True
    return False


def _loop_local_memory_history_roots(state, loop_op, allocation_roots):
    """Find loop-touched allocations that a later redistribution can retire."""
    allocation_roots = tuple(sorted(int(root) for root in allocation_roots))
    if not allocation_roots:
        return ()
    loop_source_index = loop_op.source_op_index
    release_roots = set()
    top_level_op_ids = frozenset(
        int(op_id) for op_id in state.target_program.regions[0].op_ids
    )
    for candidate in state.target_program.ops:
        if int(candidate.target_op_id) not in top_level_op_ids:
            continue
        attrs = target_ir.attrs_dict(candidate)
        if (
            candidate.kind != "layout_convert"
            or attrs.get("mode") != "redistribute"
            or not bool(attrs.get("cross_wave", False))
        ):
            continue
        candidate_source_index = candidate.source_op_index
        is_future = (
            loop_source_index is None
            or candidate_source_index is None
            or int(candidate_source_index) > int(loop_source_index)
            or (
                int(candidate_source_index) == int(loop_source_index)
                and int(candidate.target_op_id) > int(loop_op.target_op_id)
            )
        )
        if not is_future:
            continue
        for root in allocation_roots:
            if root in state.local_memory_release_unsafe_roots:
                continue
            if not _local_memory_allocation_has_future_use(state, root, candidate):
                release_roots.add(root)
    return tuple(sorted(release_roots))


def _carried_local_memory_pending_access(state, root, implicit_root_accesses, outer_pending_accesses):
    root = int(root)
    pending = outer_pending_accesses.get(int(root))
    for dep_root in _local_memory_dependency_roots(state, (root, )):
        dep_root = int(dep_root)
        # Aggregate roots represent dynamic LDS indices. Access emission already
        # expands them to the precise static aliases, so carrying descendant
        # pending state on the aggregate as well only adds redundant token deps.
        if (
            dep_root != root
            and _is_aggregate_local_memory_root(state, root)
            and _is_static_descendant_local_memory_root(state, root, dep_root)
        ):
            continue
        pending = _merge_local_memory_pending_access(
            pending,
            outer_pending_accesses.get(dep_root),
        )
        pending = _merge_local_memory_pending_access(
            pending,
            implicit_root_accesses.get(dep_root),
        )
    return pending


_SCRATCH_LAYOUT_CONVERT_MODES = frozenset({
    "cta_exchange_register_remap",
    "dot_operand_vector_payload",
    "mfma_vector_register_remap",
})


def _region_uses_scratch_memory(state, region_id):

    def visit(current_region_id):
        region = state.target_program.regions[int(current_region_id)]
        for op_id in region.op_ids:
            current_op = state.target_program.ops[int(op_id)]
            if current_op.kind == "layout_convert":
                attrs = target_ir.attrs_dict(current_op)
                if (
                    attrs.get("mode") in _SCRATCH_LAYOUT_CONVERT_MODES
                    and int(attrs.get("scratch_allocation_bytes", 0)) > 0
                ):
                    return True
            if any(visit(nested_region_id) for nested_region_id in current_op.region_ids):
                return True
        return False

    return visit(region_id)


def _emit_program_id(state, op):
    attrs = target_ir.attrs_dict(op)
    state.values[_single_result(op)] = state.builder.workgroup_id(int(attrs["axis"]))


def _emit_thread_id(state, op):
    attrs = target_ir.attrs_dict(op)
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    state.values[result_id] = state.builder.workitem_id(
        int(attrs["axis"]),
        # Wave models hardware workitem IDs as i32 SIMD values.  Triton's GPU
        # dialect spells the source result as index and immediately inserts
        # arith.index_cast for the language-level i32 thread_id.
        state.dsl.i32(),
        int(target_type.lane_width or state.target_program.kernel.threads_per_warp or 64),
    )


def _emit_select(state, op):
    result_id = _single_result(op)
    result_type = state.target_program.values[result_id].type
    count = _component_count(state, result_id)
    lane_width = int(result_type.lane_width or 64)
    condition, true_value, false_value = _operand_values(state, op, 3)
    selected_operand_ids = op.operands[1:]
    if result_type.representation in {"mask", "mask_tuple"}:
        cond_components = _as_mask_predicate_components(
            state,
            condition,
            count,
            lane_width,
            op,
        )
        true_components, true_predicates = _as_mask_payload_components_with_predicates(
            state,
            true_value,
            count,
            lane_width,
            op,
        )
        false_components, false_predicates = _as_mask_payload_components_with_predicates(
            state,
            false_value,
            count,
            lane_width,
            op,
        )
        reused = []
        payload_components = tuple(
            _reuse_component_result(
                reused,
                (condition_component, true_component, false_component),
                lambda condition_component=condition_component, true_component=true_component, false_component=
                false_component: state.builder.select(
                    condition_component,
                    true_component,
                    false_component,
                ),
            ) for condition_component, true_component, false_component in zip(
                cond_components,
                true_components,
                false_components,
            ))
        predicates = None
        if true_predicates is not None and false_predicates is not None:
            predicate_reused = []
            predicates = tuple(
                _reuse_component_result(
                    predicate_reused,
                    (condition_component, true_predicate, false_predicate),
                    lambda condition_component=condition_component, true_predicate=true_predicate, false_predicate=
                    false_predicate: state.builder.select(
                        condition_component,
                        true_predicate,
                        false_predicate,
                    ),
                ) for condition_component, true_predicate, false_predicate in zip(
                    cond_components,
                    true_predicates,
                    false_predicates,
                ))
        state.values[result_id] = _I32MaskPayload(payload_components, predicates=predicates)
        _propagate_selected_local_memory_roots(state, selected_operand_ids, result_id)
        return
    true_components, false_components = _broadcast_components(
        state,
        (true_value, false_value),
        count,
        op,
    )
    condition_payloads = (
        tuple(condition.components)
        if isinstance(condition, _I32MaskPayload) and condition.predicates is None
        else None
    )
    raw_cond_components = (
        condition_payloads
        if condition_payloads is not None
        else _mask_predicate_components(state, condition, lane_width)
    )
    expanded = _select_vector_payload_components(
        state,
        raw_cond_components,
        true_components,
        false_components,
        condition_payloads=condition_payloads is not None,
    )
    if expanded is not None:
        state.values[result_id] = _pack_components(expanded)
        _propagate_selected_local_memory_roots(state, selected_operand_ids, result_id)
        return
    if condition_payloads is not None:
        raw_cond_components = tuple(
            _i32_payload_to_mask(state, component, lane_width)
            for component in condition_payloads
        )
    cond_components = _broadcast_component_count(
        raw_cond_components,
        count,
        "mask",
        op,
    )
    reused = []
    state.values[result_id] = _pack_components(
        tuple(
            _reuse_component_result(
                reused,
                (condition_component, true_component, false_component),
                lambda condition_component=condition_component, true_component=true_component, false_component=
                false_component: state.builder.select(
                    condition_component,
                    true_component,
                    false_component,
                ),
            ) for condition_component, true_component, false_component in zip(
                cond_components,
                true_components,
                false_components,
            )))
    _propagate_selected_local_memory_roots(state, selected_operand_ids, result_id)


def _select_vector_payload_components(
    state,
    conditions,
    true_components,
    false_components,
    *,
    condition_payloads=False,
):
    """Select packed SIMD vectors with one predicate per vector element.

    Wave's first-class mask is lane-shaped, so applying it directly to a
    ``simd<vector<NxT>>`` chooses all N registers in a lane together.  Scalar
    tensors in MFMA layouts expose one condition per physical accumulator
    register; scalarize only this selection boundary and repack the result.
    """
    conditions = tuple(conditions)
    true_components = tuple(true_components)
    false_components = tuple(false_components)
    payloads = tuple(
        _simd_1d_vector_payload(state, component)
        for component in true_components
    )
    if any(payload is None for payload in payloads):
        return None
    widths = tuple(int(payload[0]) for payload in payloads)
    if len(conditions) != sum(widths) or len(conditions) == len(true_components):
        return None
    for true_component, false_component, payload in zip(
        true_components,
        false_components,
        payloads,
    ):
        if str(true_component.type) != str(false_component.type):
            return None
        false_payload = _simd_1d_vector_payload(state, false_component)
        if false_payload is None or tuple(map(str, false_payload)) != tuple(map(str, payload)):
            return None

    results = []
    cursor = 0
    for true_component, false_component, payload, width in zip(
        true_components,
        false_components,
        payloads,
        widths,
    ):
        _width, element_type, lane_width = payload
        scalar_type = state.dsl.simd_type(element_type, int(lane_width))
        true_elements = _packed_vector_payload_elements(
            true_component,
            scalar_type,
            width,
        )
        if true_elements is None:
            true_elements = tuple(
                state.dsl.wave.ExtractOp(
                    scalar_type,
                    true_component,
                    element,
                ).result
                for element in range(width)
            )
        false_elements = _packed_vector_payload_elements(
            false_component,
            scalar_type,
            width,
        )
        if false_elements is None:
            false_elements = tuple(
                state.dsl.wave.ExtractOp(
                    scalar_type,
                    false_component,
                    element,
                ).result
                for element in range(width)
            )
        selected = []
        for condition, true_element, false_element in zip(
            conditions[cursor:cursor + width],
            true_elements,
            false_elements,
        ):
            if condition_payloads:
                condition = _i32_payload_to_mask(
                    state,
                    condition,
                    int(lane_width),
                )
            selected.append(
                state.builder.select(condition, true_element, false_element)
            )
        results.append(
            state.dsl.wave.PackOp(true_component.type, selected).result
        )
        cursor += width
    return tuple(results)


def _emit_reduction(state, op):
    attrs = target_ir.attrs_dict(op)
    source = _operand_values(state, op, 1)[0]
    source_components = _as_components(source)
    result_id = _single_result(op)
    component_terms = tuple(tuple(term for term in terms) for terms in attrs["component_terms"])
    if len(component_terms) != _component_count(state, result_id):
        fail(
            "TLXW_EMIT_REDUCTION",
            STAGE,
            "reduction term groups do not match the result component count",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    lane_width = int(attrs["lane_width"])
    registers = int(attrs["source_registers"])
    scalar_type = state.dsl.simd_type(state.dsl.f32(), lane_width)
    lane = state.builder.lane_id(state.dsl.i32(), lane_width)
    identity_lane_map = (0, tuple(1 << bit for bit in range(lane_width.bit_length() - 1)))
    extracted = {}
    lane_maps = {}
    results = []
    for terms in component_terms:
        grouped = {}
        for source_component, source_register, lane_base, lane_coefficients in terms:
            source_component = int(source_component)
            source_register = int(source_register)
            if source_component < 0 or source_component >= len(source_components):
                fail(
                    "TLXW_EMIT_REDUCTION",
                    STAGE,
                    "reduction term references an out-of-range source component",
                    target_op_id=op.target_op_id,
                )
            if source_register < 0 or source_register >= registers:
                fail(
                    "TLXW_EMIT_REDUCTION",
                    STAGE,
                    "reduction term references an out-of-range fragment register",
                    target_op_id=op.target_op_id,
                )
            key = (source_component, source_register)
            value = extracted.get(key)
            if value is None:
                payload = _simd_1d_vector_payload(state, source_components[source_component])
                if payload is None or int(payload[0]) != registers or str(payload[1]) != "f32":
                    fail(
                        "TLXW_EMIT_REDUCTION",
                        STAGE,
                        "reduction input must contain f32 MFMA register vectors",
                        target_op_id=op.target_op_id,
                    )
                value = state.dsl.wave.ExtractOp(
                    scalar_type,
                    source_components[source_component],
                    source_register,
                ).result
                extracted[key] = value
            lane_key = (int(lane_base), tuple(int(coefficient) for coefficient in lane_coefficients))
            grouped.setdefault(lane_key, []).append(value)
        group_results = []
        for lane_key, values in grouped.items():
            value = _emit_reduction_tree(state, attrs, values, op)
            if lane_key != identity_lane_map:
                source_lane = lane_maps.get(lane_key)
                if source_lane is None:
                    source_lane = _bit_linear_thread_offset_index_expr(
                        state,
                        lane,
                        lane_key[0],
                        lane_key[1],
                    )
                    lane_maps[lane_key] = source_lane
                value = state.dsl.wave.ShuffleOp(
                    value.type,
                    value,
                    source_lane,
                ).result
            group_results.append(value)
        results.append(_emit_reduction_tree(state, attrs, group_results, op))
    state.values[result_id] = _pack_components(tuple(results))


def _emit_reduction_tree(state, attrs, values, op):
    values = list(values)
    if not values:
        fail(
            "TLXW_EMIT_REDUCTION",
            STAGE,
            "reduction requires at least one input value",
            target_op_id=op.target_op_id,
        )
    fastmath = _fastmath_attr(state, attrs.get("fastmath"), op)
    operation = attrs["operation"]
    while len(values) > 1:
        next_values = []
        for index in range(0, len(values), 2):
            if index + 1 == len(values):
                next_values.append(values[index])
                continue
            lhs, rhs = values[index:index + 2]
            if operation in {"maximumf", "maxnumf"}:
                next_values.append(state.builder.fmax(lhs, rhs))
            elif operation in {"addf", "mulf"}:
                next_values.append(
                    _emit_wave_float_binary_component(
                        state,
                        operation,
                        lhs,
                        rhs,
                        fastmath,
                        op,
                    ))
            else:
                fail(
                    "TLXW_EMIT_REDUCTION",
                    STAGE,
                    f"unsupported reduction operation {operation}",
                    target_op_id=op.target_op_id,
                )
        values = next_values
    return values[0]


def _emit_if(state, op):
    attrs = target_ir.attrs_dict(op)
    if len(op.operands) != 1:
        fail(
            "TLXW_EMIT_IF_OPERAND_COUNT",
            STAGE,
            "if target op requires one condition operand",
            target_op_id=op.target_op_id,
        )
    if len(op.region_ids) != 2:
        fail(
            "TLXW_EMIT_IF_REGION_COUNT",
            STAGE,
            "if target op requires then and else regions",
            target_op_id=op.target_op_id,
        )
    condition = _require_value(state, op.operands[0], op)
    if not _is_scalar_i1_value(state, condition):
        fail(
            "TLXW_EMIT_IF_CONDITION",
            STAGE,
            "if condition must be a scalar i1 value",
            target_op_id=op.target_op_id,
            target_value_id=op.operands[0],
        )
    result_types, result_shapes = _structured_result_types_and_shapes(
        state,
        op.results,
        op,
    )
    outer_values = dict(state.values)
    outer_uniform_pointer_bases = dict(state.uniform_pointer_bases)
    outer_shared_pointer_dword_bases = dict(state.shared_pointer_dword_bases)
    outer_shared_pointer_offset_cache = dict(state.shared_pointer_offset_cache)
    outer_wave_offset_i32_cache = dict(state.wave_offset_i32_cache)
    outer_scratch_token = state.scratch_token
    outer_scratch_token_needs_write_barrier = state.scratch_token_needs_write_barrier
    outer_local_memory_tokens = dict(state.local_memory_tokens)
    outer_local_memory_pending_accesses = dict(state.local_memory_pending_accesses)
    outer_local_memory_access_tokens = dict(state.local_memory_access_tokens)
    outer_local_memory_read_tokens = dict(state.local_memory_read_tokens)
    outer_pending_mma_read_boundaries = dict(state.pending_mma_read_boundaries)
    outer_materialized_mma_read_boundaries = dict(state.materialized_mma_read_boundaries)
    outer_local_memory_root_sets = dict(state.local_memory_root_sets)
    outer_token_local_memory_root_sets = dict(state.token_local_memory_root_sets)
    outer_local_memory_allocation_roots = frozenset(state.local_memory_allocations)
    conditional_local_memory_roots = set()
    conditional_local_memory_accesses = {}
    for region_id in op.region_ids:
        touched_roots, implicit_accesses = _local_memory_roots_touched_by_region(
            state,
            region_id,
        )
        conditional_local_memory_roots.update(touched_roots)
        for root, access_kind in implicit_accesses.items():
            conditional_local_memory_accesses[int(root)] = _merge_local_memory_pending_access(
                conditional_local_memory_accesses.get(int(root)),
                access_kind,
            )
    hidden_local_memory_roots = tuple(sorted({
        int(dependency_root)
        for root in conditional_local_memory_roots
        for dependency_root in _local_memory_dependency_roots(state, (root, ))
        if int(dependency_root) in outer_local_memory_allocation_roots
    }))
    all_result_types = (
        *result_types,
        *([state.dsl.mem_token_type()] * (2 * len(hidden_local_memory_roots))),
    )
    with state.builder.if_(condition, all_result_types, otherwise=True) as ifop:
        _restore_emission_state(
            state,
            outer_values,
            outer_uniform_pointer_bases,
            outer_shared_pointer_dword_bases,
            outer_shared_pointer_offset_cache,
            outer_wave_offset_i32_cache,
            outer_scratch_token,
            outer_scratch_token_needs_write_barrier,
            outer_local_memory_tokens,
            outer_local_memory_pending_accesses,
            outer_local_memory_access_tokens,
            outer_local_memory_read_tokens,
            outer_pending_mma_read_boundaries,
            outer_materialized_mma_read_boundaries,
            outer_local_memory_root_sets,
            outer_token_local_memory_root_sets,
        )
        then_yields = _emit_structured_branch(
            state,
            op.region_ids[0],
            op.results,
            result_shapes,
            "if then",
            op,
        )
        then_local_memory_yields = tuple(
            _local_memory_allocation_access_dependency(state, root)
            for root in hidden_local_memory_roots
        )
        then_local_memory_read_yields = tuple(
            _local_memory_allocation_read_dependency(state, root)
            for root in hidden_local_memory_roots
        )
        if (
            then_yields
            or then_local_memory_yields
            or then_local_memory_read_yields
        ):
            state.builder.yield_((
                *then_yields,
                *then_local_memory_yields,
                *then_local_memory_read_yields,
            ))
        _restore_emission_state(
            state,
            outer_values,
            outer_uniform_pointer_bases,
            outer_shared_pointer_dword_bases,
            outer_shared_pointer_offset_cache,
            outer_wave_offset_i32_cache,
            outer_scratch_token,
            outer_scratch_token_needs_write_barrier,
            outer_local_memory_tokens,
            outer_local_memory_pending_accesses,
            outer_local_memory_access_tokens,
            outer_local_memory_read_tokens,
            outer_pending_mma_read_boundaries,
            outer_materialized_mma_read_boundaries,
            outer_local_memory_root_sets,
            outer_token_local_memory_root_sets,
        )
        with ifop.otherwise():
            else_yields = _emit_structured_branch(
                state,
                op.region_ids[1],
                op.results,
                result_shapes,
                "if else",
                op,
            )
            else_local_memory_yields = tuple(
                _local_memory_allocation_access_dependency(state, root)
                for root in hidden_local_memory_roots
            )
            else_local_memory_read_yields = tuple(
                _local_memory_allocation_read_dependency(state, root)
                for root in hidden_local_memory_roots
            )
            if (
                else_yields
                or else_local_memory_yields
                or else_local_memory_read_yields
            ):
                state.builder.yield_((
                    *else_yields,
                    *else_local_memory_yields,
                    *else_local_memory_read_yields,
                ))
    _restore_emission_state(
        state,
        outer_values,
        outer_uniform_pointer_bases,
        outer_shared_pointer_dword_bases,
        outer_shared_pointer_offset_cache,
        outer_wave_offset_i32_cache,
        outer_scratch_token,
        outer_scratch_token_needs_write_barrier,
        outer_local_memory_tokens,
        outer_local_memory_pending_accesses,
        outer_local_memory_access_tokens,
        outer_local_memory_read_tokens,
        outer_pending_mma_read_boundaries,
        outer_materialized_mma_read_boundaries,
        outer_local_memory_root_sets,
        outer_token_local_memory_root_sets,
    )
    # An allocation created in only one arm has no dominating handle at the
    # continuation. Existing allocations are handled by the hidden token
    # results above; keep conditionally scoped allocations alive.
    conditionally_scoped_allocations = {
        int(root)
        for root in state.local_memory_allocations
        if int(root) not in outer_local_memory_allocation_roots
    }
    state.local_memory_release_unsafe_roots.update(conditionally_scoped_allocations)

    flat_results = tuple(ifop.results)
    if len(flat_results) != len(all_result_types):
        fail(
            "TLXW_EMIT_IF_RESULT_COMPONENTS",
            STAGE,
            "if result component count must match result and hidden LDS types",
            target_op_id=op.target_op_id,
        )
    source_flat_results = flat_results[:len(result_types)]
    hidden_start = len(result_types)
    hidden_end = hidden_start + len(hidden_local_memory_roots)
    hidden_local_memory_results = flat_results[hidden_start:hidden_end]
    hidden_read_end = hidden_end + len(hidden_local_memory_roots)
    hidden_local_memory_read_results = flat_results[hidden_end:hidden_read_end]
    trailing_results = flat_results[hidden_read_end:]
    if len(hidden_local_memory_results) != len(hidden_local_memory_roots):
        fail(
            "TLXW_EMIT_IF_HIDDEN_LOCAL_TOKEN_COMPONENTS",
            STAGE,
            "if hidden local memory result count must match carried LDS roots",
            target_op_id=op.target_op_id,
        )
    if len(hidden_local_memory_read_results) != len(hidden_local_memory_roots):
        fail(
            "TLXW_EMIT_IF_HIDDEN_LOCAL_READ_TOKEN_COMPONENTS",
            STAGE,
            "if hidden local memory read result count must match carried LDS roots",
            target_op_id=op.target_op_id,
        )
    if trailing_results:
        fail(
            "TLXW_EMIT_IF_TRAILING_COMPONENTS",
            STAGE,
            "if has unexpected trailing hidden token results",
            target_op_id=op.target_op_id,
        )
    for root, token in zip(hidden_local_memory_roots, hidden_local_memory_results):
        root = int(root)
        # Each arm's yielded token includes the full incoming history plus its
        # own accesses, so the merged token replaces the pre-if history.
        state.local_memory_access_tokens[root] = (token, )
        state.local_memory_tokens[root] = token
        pending_access = _carried_local_memory_pending_access(
            state,
            root,
            conditional_local_memory_accesses,
            outer_local_memory_pending_accesses,
        )
        if pending_access is None:
            state.local_memory_pending_accesses.pop(root, None)
        else:
            state.local_memory_pending_accesses[root] = pending_access
    for root, token in zip(hidden_local_memory_roots, hidden_local_memory_read_results):
        state.local_memory_read_tokens[int(root)] = (token, )
    cursor = 0
    for result_id, shape in zip(op.results, result_shapes):
        state.values[result_id] = _pack_structured_value_components(
            state,
            source_flat_results[cursor:cursor + shape.component_count],
            shape,
            "if",
            op,
        )
        cursor += shape.component_count


def _emit_for_loop(state, op):
    attrs = target_ir.attrs_dict(op)
    if len(op.region_ids) != 1:
        fail(
            "TLXW_EMIT_FOR_REGION_COUNT",
            STAGE,
            "for_loop target op requires exactly one region",
            target_op_id=op.target_op_id,
        )
    init_arg_count = int(attrs["init_arg_count"])
    if len(op.operands) != 3 + init_arg_count:
        fail(
            "TLXW_EMIT_FOR_OPERAND_COUNT",
            STAGE,
            "for_loop operand count must be lower, upper, step, and init args",
            target_op_id=op.target_op_id,
        )
    lower, upper, step = tuple(_require_value(state, target_value_id, op) for target_value_id in op.operands[:3])
    init_target_ids = op.operands[3:]
    init_values = tuple(
        _require_value(state, target_value_id, op)
        for target_value_id in init_target_ids
    )
    flat_init_values, init_shapes = _flatten_structured_values(
        state,
        init_values,
        init_target_ids,
        "for_loop",
        op,
        preserve_mask_predicates=False,
        preserve_mma_packet_payloads=True,
    )
    region = state.target_program.regions[op.region_ids[0]]
    if len(region.block_arg_ids) != 1 + init_arg_count:
        fail(
            "TLXW_EMIT_FOR_BLOCK_ARGS",
            STAGE,
            "for_loop region block args must match induction plus init args",
            target_op_id=op.target_op_id,
        )
    if not flat_init_values and op.results:
        fail(
            "TLXW_EMIT_FOR_RESULT_COUNT",
            STAGE,
            "result-bearing for_loop requires init args",
            target_op_id=op.target_op_id,
        )

    outer_values = dict(state.values)
    outer_shared_pointer_dword_bases = dict(state.shared_pointer_dword_bases)
    outer_shared_pointer_offset_cache = dict(state.shared_pointer_offset_cache)
    outer_wave_offset_i32_cache = dict(state.wave_offset_i32_cache)
    outer_lane_mask_loop_phase = state.lane_mask_loop_phase
    outer_scratch_token = state.scratch_token
    outer_scratch_token_needs_write_barrier = state.scratch_token_needs_write_barrier
    outer_local_memory_tokens = dict(state.local_memory_tokens)
    outer_local_memory_pending_accesses = dict(state.local_memory_pending_accesses)
    outer_local_memory_access_tokens = dict(state.local_memory_access_tokens)
    outer_local_memory_read_tokens = dict(state.local_memory_read_tokens)
    outer_pending_mma_read_boundaries = dict(state.pending_mma_read_boundaries)
    outer_materialized_mma_read_boundaries = dict(state.materialized_mma_read_boundaries)
    outer_local_memory_root_sets = dict(state.local_memory_root_sets)
    outer_token_local_memory_root_sets = dict(state.token_local_memory_root_sets)
    (
        touched_local_memory_roots,
        implicit_local_memory_accesses,
        region_local_memory_root_sets,
        region_local_memory_read_roots,
    ) = _local_memory_roots_touched_by_region(
        state,
        region.target_region_id,
        include_root_sets=True,
    )
    loop_token_block_root_sets, loop_token_result_root_sets = _loop_token_root_sets(
        state,
        op,
        region,
        region_local_memory_root_sets,
    )
    implicit_local_memory_roots = frozenset(implicit_local_memory_accesses)
    hidden_local_memory_seed_roots = implicit_local_memory_roots.union(
        root for root in touched_local_memory_roots
        if root in outer_local_memory_tokens
        or any(dep_root in outer_local_memory_tokens
               for dep_root in _local_memory_dependency_roots(state, (root, )))
    )
    hidden_local_memory_candidates = set()
    for root in hidden_local_memory_seed_roots:
        for dep_root in _local_memory_dependency_roots(state, (root, )):
            if root in implicit_local_memory_roots or dep_root in outer_local_memory_tokens:
                hidden_local_memory_candidates.add(dep_root)
    # Async queue token loop carries track logical committed groups, not stable
    # physical LDS slots.  Circular-buffer loops can consume one group's token
    # and yield a different group's token for the same carry position, so the
    # per-root completion state still has to be carried independently.
    if not _loop_local_memory_state_carry_required(
        state,
        region,
        hidden_local_memory_candidates,
        implicit_local_memory_accesses,
        outer_local_memory_pending_accesses,
        region_local_memory_root_sets,
        loop_token_block_root_sets,
    ):
        hidden_local_memory_candidates.clear()
    hidden_local_memory_roots = tuple(sorted(hidden_local_memory_candidates))
    hidden_local_memory_init_tokens = (
        _join_memory_tokens(
            state,
            tuple(
                outer_local_memory_tokens[root]
                for root in hidden_local_memory_roots
                if root in outer_local_memory_tokens
            ),
        ),
    ) if hidden_local_memory_roots else ()
    # Per-slot carries drive ordering within the loop. Keep a separate
    # allocation-wide history carry so every access token also reaches the
    # post-loop lifetime proof, even when an access uses an alias that is later
    # superseded by another physical-slot state token.
    loop_local_memory_allocation_roots = tuple(sorted({
        int(dependency_root)
        for root in touched_local_memory_roots
        for dependency_root in _local_memory_dependency_roots(state, (root, ))
        if int(dependency_root) in state.local_memory_allocations
    }))
    hidden_local_memory_access_roots = _loop_local_memory_history_roots(
        state,
        op,
        loop_local_memory_allocation_roots,
    )
    hidden_local_memory_read_roots = tuple(
        root
        for root in loop_local_memory_allocation_roots
        if any(
            int(dependency_root) in region_local_memory_read_roots
            for dependency_root in _local_memory_dependency_roots(state, (root, ))
        ) or any(
            int(dependency_root) in outer_local_memory_read_tokens
            for dependency_root in _local_memory_dependency_roots(state, (root, ))
        )
    )
    # A loop has one backedge memory frontier.  Carrying a separate cumulative
    # history for every allocation duplicates token phis and can split one
    # commit group's prior-read dependency into incomplete pieces.  Preserve
    # precise per-root slot state above, while carrying allocation lifetime
    # history as one collective token and reattaching it to every touched root
    # inside and after the loop.
    hidden_local_memory_access_init_tokens = (
        _join_memory_tokens(
            state,
            tuple(
                token
                for root in hidden_local_memory_access_roots
                for token in outer_local_memory_access_tokens.get(root, ())
            ),
        ),
    ) if hidden_local_memory_access_roots else ()
    # Overwrite ordering needs a read-only frontier.  Carry it independently
    # from lifetime history so a refill can wait for prior LDS reads without
    # inheriting async DMA completion state from earlier accesses.
    hidden_local_memory_read_init_tokens = (
        _join_memory_tokens(
            state,
            tuple(
                token
                for root in hidden_local_memory_read_roots
                for dependency_root in sorted(
                    _local_memory_dependency_roots(state, (root, )))
                for token in outer_local_memory_read_tokens.get(
                    int(dependency_root), ())
            ),
        ),
    ) if hidden_local_memory_read_roots else ()
    carry_scratch_token = _region_uses_scratch_memory(state, region.target_region_id)
    scratch_init_tokens = (
        outer_scratch_token if outer_scratch_token is not None else state.builder.token(),
    ) if carry_scratch_token else ()
    nonzero_trip = bool(attrs.get("nonzero_trip", False))
    loop_init_values = (
        *flat_init_values,
        *hidden_local_memory_init_tokens,
        *hidden_local_memory_access_init_tokens,
        *hidden_local_memory_read_init_tokens,
        *scratch_init_tokens,
    )
    with state.builder.for_loop(
            lower,
            upper,
            step,
            init_args=loop_init_values,
            nonzero_trip=nonzero_trip,
    ) as loop:
        if loop_init_values:
            induction_value = loop.induction_variable
            loop_iter_values = tuple(loop.inner_iter_args)
            flat_iter_values = loop_iter_values[:len(flat_init_values)]
            hidden_start = len(flat_init_values)
            hidden_end = hidden_start + len(hidden_local_memory_init_tokens)
            hidden_local_memory_iter_tokens = loop_iter_values[hidden_start:hidden_end]
            hidden_access_start = hidden_end
            hidden_access_end = hidden_access_start + len(hidden_local_memory_access_init_tokens)
            hidden_local_memory_access_iter_tokens = loop_iter_values[
                hidden_access_start:hidden_access_end
            ]
            hidden_read_start = hidden_access_end
            hidden_read_end = hidden_read_start + len(hidden_local_memory_read_init_tokens)
            hidden_local_memory_read_iter_tokens = loop_iter_values[
                hidden_read_start:hidden_read_end
            ]
            scratch_start = hidden_read_end
            scratch_end = scratch_start + len(scratch_init_tokens)
            scratch_iter_tokens = loop_iter_values[scratch_start:scratch_end]
        else:
            induction_value = loop
            flat_iter_values = ()
            hidden_local_memory_iter_tokens = ()
            hidden_local_memory_access_iter_tokens = ()
            hidden_local_memory_read_iter_tokens = ()
            scratch_iter_tokens = ()
        lane_mask_loop_phase = induction_value
        if str(lane_mask_loop_phase.type) != str(state.dsl.i32()):
            lane_mask_loop_phase = state.builder.cast(
                lane_mask_loop_phase,
                state.dsl.i32(),
                state.dsl.CastKind.IntConvert,
            )
        state.lane_mask_loop_phase = lane_mask_loop_phase
        if len(hidden_local_memory_iter_tokens) != int(bool(hidden_local_memory_roots)):
            fail(
                "TLXW_EMIT_FOR_HIDDEN_LOCAL_TOKEN_COMPONENTS",
                STAGE,
                "for_loop hidden local memory token component count must match carried LDS roots",
                target_op_id=op.target_op_id,
            )
        if len(hidden_local_memory_access_iter_tokens) != int(bool(hidden_local_memory_access_roots)):
            fail(
                "TLXW_EMIT_FOR_HIDDEN_LOCAL_ACCESS_TOKEN_COMPONENTS",
                STAGE,
                "for_loop must carry one collective local memory access token",
                target_op_id=op.target_op_id,
            )
        if len(hidden_local_memory_read_iter_tokens) != int(bool(hidden_local_memory_read_roots)):
            fail(
                "TLXW_EMIT_FOR_HIDDEN_LOCAL_READ_TOKEN_COMPONENTS",
                STAGE,
                "for_loop must carry one collective local memory read token",
                target_op_id=op.target_op_id,
            )
        if len(scratch_iter_tokens) != len(scratch_init_tokens):
            fail(
                "TLXW_EMIT_FOR_HIDDEN_SCRATCH_TOKEN_COMPONENTS",
                STAGE,
                "for_loop hidden scratch token component count must match carried scratch roots",
                target_op_id=op.target_op_id,
            )
        for root in hidden_local_memory_roots:
            root = int(root)
            state.local_memory_tokens[root] = hidden_local_memory_iter_tokens[0]
            pending_access = _carried_local_memory_pending_access(
                state,
                root,
                implicit_local_memory_accesses,
                outer_local_memory_pending_accesses,
            )
            if pending_access is None:
                state.local_memory_pending_accesses.pop(root, None)
            else:
                state.local_memory_pending_accesses[root] = pending_access
        if hidden_local_memory_access_iter_tokens:
            history_token = hidden_local_memory_access_iter_tokens[0]
            for root in hidden_local_memory_access_roots:
                state.local_memory_access_tokens[int(root)] = (history_token, )
        if hidden_local_memory_read_iter_tokens:
            read_token = hidden_local_memory_read_iter_tokens[0]
            for root in hidden_local_memory_read_roots:
                _clear_local_memory_read_tokens(state, (int(root), ))
                state.local_memory_read_tokens[int(root)] = (read_token, )
        if scratch_iter_tokens:
            state.scratch_token = scratch_iter_tokens[0]
            state.scratch_token_needs_write_barrier = True
        _bind_loop_region_args(
            state,
            region.block_arg_ids,
            induction_value,
            flat_iter_values,
            init_shapes,
            init_target_ids,
            preserved_local_memory_roots=hidden_local_memory_roots,
            op=op,
        )
        state.token_local_memory_root_sets.update(loop_token_block_root_sets)
        yielded_values = _emit_region(state, op.region_ids[0])
        flat_yield_values, yield_shapes = _flatten_structured_values(
            state,
            yielded_values,
            region.yield_value_ids,
            "for_loop",
            op,
            expected_shapes=init_shapes,
            preserve_mask_predicates=False,
            preserve_mma_packet_payloads=True,
        )
        if tuple(yield_shapes) != tuple(init_shapes):
            shape_mismatches = tuple(
                (
                    index,
                    init_target_ids[index],
                    region.yield_value_ids[index],
                    state.target_program.values[init_target_ids[index]].source_value_id,
                    state.target_program.values[
                        region.yield_value_ids[index]
                    ].source_value_id,
                    init_shape,
                    yield_shape,
                )
                for index, (init_shape, yield_shape) in enumerate(
                    zip(init_shapes, yield_shapes)
                )
                if init_shape != yield_shape
            )
            fail(
                "TLXW_EMIT_FOR_YIELD_COMPONENTS",
                STAGE,
                "for_loop yielded component shape must match init args; "
                f"mismatches={shape_mismatches!r}",
                target_op_id=op.target_op_id,
            )
        hidden_local_memory_yield_tokens = (
            _join_memory_tokens(
                state,
                tuple(
                    state.local_memory_tokens.get(
                        int(root), hidden_local_memory_iter_tokens[0])
                    for root in hidden_local_memory_roots
                ),
            ),
        ) if hidden_local_memory_iter_tokens else ()
        hidden_local_memory_access_yield_tokens = (
            _join_memory_tokens(
                state,
                tuple(
                    token
                    for root in hidden_local_memory_access_roots
                    for token in state.local_memory_access_tokens.get(int(root), ())
                ),
            ),
        ) if hidden_local_memory_access_iter_tokens else ()
        hidden_local_memory_read_yield_tokens = (
            _join_memory_tokens(
                state,
                tuple(
                    token
                    for root in hidden_local_memory_read_roots
                    for dependency_root in sorted(
                        _local_memory_dependency_roots(state, (root, )))
                    for token in state.local_memory_read_tokens.get(
                        int(dependency_root), ())
                ),
            ),
        ) if hidden_local_memory_read_iter_tokens else ()
        scratch_yield_tokens = (
            state.scratch_token if state.scratch_token is not None else scratch_iter_tokens[0],
        ) if scratch_iter_tokens else ()
        scratch_yield_needs_write_barrier = state.scratch_token_needs_write_barrier
        if loop_init_values:
            state.builder.yield_((
                *flat_yield_values,
                *hidden_local_memory_yield_tokens,
                *hidden_local_memory_access_yield_tokens,
                *hidden_local_memory_read_yield_tokens,
                *scratch_yield_tokens,
            ))
        elif flat_yield_values:
            fail(
                "TLXW_EMIT_FOR_UNEXPECTED_YIELD",
                STAGE,
                "for_loop without init args must not yield values",
                target_op_id=op.target_op_id,
            )
    inner_token_local_memory_root_sets = dict(state.token_local_memory_root_sets)
    state.values = outer_values
    state.shared_pointer_dword_bases = outer_shared_pointer_dword_bases
    state.shared_pointer_offset_cache = outer_shared_pointer_offset_cache
    state.wave_offset_i32_cache = outer_wave_offset_i32_cache
    state.lane_mask_loop_phase = outer_lane_mask_loop_phase
    state.scratch_token = outer_scratch_token
    state.scratch_token_needs_write_barrier = outer_scratch_token_needs_write_barrier
    state.local_memory_tokens = outer_local_memory_tokens
    state.local_memory_pending_accesses = outer_local_memory_pending_accesses
    state.local_memory_access_tokens = outer_local_memory_access_tokens
    state.local_memory_read_tokens = outer_local_memory_read_tokens
    state.pending_mma_read_boundaries = outer_pending_mma_read_boundaries
    state.materialized_mma_read_boundaries = outer_materialized_mma_read_boundaries
    state.local_memory_root_sets = outer_local_memory_root_sets
    state.token_local_memory_root_sets = outer_token_local_memory_root_sets

    flat_results = tuple(loop.results) if loop_init_values else ()
    if len(flat_results) != len(loop_init_values):
        fail(
            "TLXW_EMIT_FOR_RESULT_COMPONENTS",
            STAGE,
            "for_loop result component count must match init args and hidden local memory tokens",
            target_op_id=op.target_op_id,
        )
    source_flat_results = flat_results[:len(flat_init_values)]
    hidden_start = len(flat_init_values)
    hidden_end = hidden_start + len(hidden_local_memory_init_tokens)
    hidden_local_memory_results = flat_results[hidden_start:hidden_end]
    hidden_access_start = hidden_end
    hidden_access_end = hidden_access_start + len(hidden_local_memory_access_init_tokens)
    hidden_local_memory_access_results = flat_results[
        hidden_access_start:hidden_access_end
    ]
    hidden_read_start = hidden_access_end
    hidden_read_end = hidden_read_start + len(hidden_local_memory_read_init_tokens)
    hidden_local_memory_read_results = flat_results[
        hidden_read_start:hidden_read_end
    ]
    scratch_start = hidden_read_end
    scratch_end = scratch_start + len(scratch_init_tokens)
    scratch_results = flat_results[scratch_start:scratch_end]
    trailing_results = flat_results[scratch_end:]
    if len(hidden_local_memory_results) != int(bool(hidden_local_memory_roots)):
        fail(
            "TLXW_EMIT_FOR_HIDDEN_LOCAL_TOKEN_COMPONENTS",
            STAGE,
            "for_loop hidden local memory result count must match carried LDS roots",
            target_op_id=op.target_op_id,
        )
    # Hidden LDS results are physical slot state, not async queue groups.  A
    # wait inside the loop may have consumed a different logical group, so the
    # post-loop successor must still preserve each carried slot's pending access.
    for root in hidden_local_memory_roots:
        root = int(root)
        token = hidden_local_memory_results[0]
        _record_local_memory_access_token(state, (root, ), token)
        state.local_memory_tokens[root] = token
        pending_access = _carried_local_memory_pending_access(
            state,
            root,
            implicit_local_memory_accesses,
            outer_local_memory_pending_accesses,
        )
        if pending_access is None:
            state.local_memory_pending_accesses.pop(root, None)
        else:
            state.local_memory_pending_accesses[root] = pending_access
    if len(hidden_local_memory_access_results) != int(bool(hidden_local_memory_access_roots)):
        fail(
            "TLXW_EMIT_FOR_HIDDEN_LOCAL_ACCESS_TOKEN_COMPONENTS",
            STAGE,
            "for_loop must return one collective local memory access token",
            target_op_id=op.target_op_id,
        )
    if hidden_local_memory_access_results:
        history_token = hidden_local_memory_access_results[0]
        # The loop result merges the incoming history with the body history and
        # therefore replaces every non-dominating body-local access token.
        for root in hidden_local_memory_access_roots:
            state.local_memory_access_tokens[int(root)] = (history_token, )
    if len(hidden_local_memory_read_results) != int(bool(hidden_local_memory_read_roots)):
        fail(
            "TLXW_EMIT_FOR_HIDDEN_LOCAL_READ_TOKEN_COMPONENTS",
            STAGE,
            "for_loop must return one collective local memory read token",
            target_op_id=op.target_op_id,
        )
    if hidden_local_memory_read_results:
        read_token = hidden_local_memory_read_results[0]
        for root in hidden_local_memory_read_roots:
            _clear_local_memory_read_tokens(state, (int(root), ))
            state.local_memory_read_tokens[int(root)] = (read_token, )
    if len(scratch_results) != len(scratch_init_tokens):
        fail(
            "TLXW_EMIT_FOR_HIDDEN_SCRATCH_TOKEN_COMPONENTS",
            STAGE,
            "for_loop hidden scratch result count must match carried scratch roots",
            target_op_id=op.target_op_id,
        )
    if scratch_results:
        state.scratch_token = scratch_results[0]
        state.scratch_token_needs_write_barrier = scratch_yield_needs_write_barrier
    if trailing_results:
        fail(
            "TLXW_EMIT_FOR_TRAILING_COMPONENTS",
            STAGE,
            "for_loop has unexpected trailing hidden token results",
            target_op_id=op.target_op_id,
        )

    if len(op.results) != init_arg_count:
        fail(
            "TLXW_EMIT_FOR_RESULT_COUNT",
            STAGE,
            "for_loop result count must match init args",
            target_op_id=op.target_op_id,
        )
    if op.results:
        if len(source_flat_results) != len(flat_init_values):
            fail(
                "TLXW_EMIT_FOR_RESULT_COMPONENTS",
                STAGE,
                "for_loop result component count must match init args",
                target_op_id=op.target_op_id,
            )
        cursor = 0
        for yield_index, (result_id, shape) in enumerate(zip(op.results, init_shapes)):
            state.values[result_id] = _pack_loop_value_components(
                state,
                source_flat_results[cursor:cursor + shape.component_count],
                shape,
                op,
            )
            if yield_index < len(region.yield_value_ids):
                roots = loop_token_result_root_sets.get(int(result_id))
                if roots is None:
                    roots = inner_token_local_memory_root_sets.get(int(region.yield_value_ids[yield_index]))
                if roots:
                    state.token_local_memory_root_sets[int(result_id)] = roots
                    _set_local_memory_roots_token(state, roots, state.values[result_id])
            cursor += shape.component_count


def _emit_structured_branch(
    state,
    region_id,
    result_target_ids,
    result_shapes,
    label,
    op,
):
    region = state.target_program.regions[region_id]
    if region.block_arg_ids:
        fail(
            "TLXW_EMIT_IF_BLOCK_ARGS",
            STAGE,
            "if branch regions must not have block arguments",
            target_op_id=op.target_op_id,
        )
    if len(region.yield_value_ids) != len(result_target_ids):
        fail(
            "TLXW_EMIT_IF_YIELD_COUNT",
            STAGE,
            "if branch yield count must match result count",
            target_op_id=op.target_op_id,
        )
    yielded_values = _emit_region(state, region_id)
    flat_yield_values, yield_shapes = _flatten_structured_values(
        state,
        yielded_values,
        region.yield_value_ids,
        label,
        op,
        preserve_mma_packet_payloads=True,
    )
    if tuple(yield_shapes) != tuple(result_shapes):
        fail(
            "TLXW_EMIT_IF_YIELD_COMPONENTS",
            STAGE,
            "if branch yielded component shape must match result types",
            target_op_id=op.target_op_id,
        )
    return flat_yield_values


def _structured_result_types_and_shapes(state, target_value_ids, op):
    result_types = []
    shapes = []
    attrs = target_ir.attrs_dict(op)
    packet_registers = tuple(int(width) for width in attrs.get("result_packet_registers", ()))
    if packet_registers and len(packet_registers) != len(target_value_ids):
        fail(
            "TLXW_EMIT_IF_RESULT_TYPE",
            STAGE,
            "if packet register widths must match the result count",
            target_op_id=op.target_op_id,
        )
    for result_index, target_value_id in enumerate(target_value_ids):
        target_type = state.target_program.values[target_value_id].type
        component_count = int(target_type.component_count)
        if target_type.representation in {"mask", "mask_tuple"}:
            lane_width = int(target_type.lane_width or 64)
            shapes.append(_LoopValueShape(
                component_count,
                is_mask_payload=True,
                mask_payload_component_count=component_count,
            ))
            result_types.extend([state.dsl.simd_type(state.dsl.i32(), lane_width)] * component_count)
            continue
        if target_type.representation == "token":
            shapes.append(_LoopValueShape(component_count))
            result_types.extend([state.dsl.mem_token_type()] * component_count)
            continue
        if target_type.representation in _MMA_PACKET_REPRESENTATIONS:
            registers = packet_registers[result_index] if packet_registers else 0
            if registers <= 0:
                fail(
                    "TLXW_EMIT_IF_RESULT_TYPE",
                    STAGE,
                    "if MMA packet result requires a positive register width",
                    target_op_id=op.target_op_id,
                    target_value_id=target_value_id,
                )
            lane_width = int(target_type.lane_width or 64)
            element_type = _scalar_type(state.dsl, target_type.element_type)
            packet_type = state.dsl.simd_type(
                state.dsl.vector_type(registers, element_type),
                lane_width,
            )
            shapes.append(_LoopValueShape(
                component_count,
                logical_component_count=component_count,
                preserved_vector_payload_key=(registers, str(element_type), lane_width),
                preserved_vector_payload_type=packet_type,
            ))
            result_types.extend([packet_type] * component_count)
            continue
        if target_type.representation in {
                "scalar",
                "uniform_pointer",
                "simd",
                "simd_tuple",
                "per_lane_pointer",
                "pointer_tuple",
        }:
            shapes.append(_LoopValueShape(component_count))
            result_types.extend([_wave_type(state.dsl, target_type)] * component_count)
            continue
        fail(
            "TLXW_EMIT_IF_RESULT_TYPE",
            STAGE,
            f"if result type {target_type.representation} is not supported",
            target_op_id=op.target_op_id,
            target_value_id=target_value_id,
        )
    return tuple(result_types), tuple(shapes)


def _restore_emission_state(
    state,
    values,
    uniform_pointer_bases,
    shared_pointer_dword_bases,
    shared_pointer_offset_cache,
    wave_offset_i32_cache,
    scratch_token,
    scratch_token_needs_write_barrier,
    local_memory_tokens,
    local_memory_pending_accesses,
    local_memory_access_tokens,
    local_memory_read_tokens,
    pending_mma_read_boundaries,
    materialized_mma_read_boundaries,
    local_memory_root_sets,
    token_local_memory_root_sets,
):
    state.values = dict(values)
    state.uniform_pointer_bases = dict(uniform_pointer_bases)
    state.shared_pointer_dword_bases = dict(shared_pointer_dword_bases)
    state.shared_pointer_offset_cache = dict(shared_pointer_offset_cache)
    state.wave_offset_i32_cache = dict(wave_offset_i32_cache)
    state.scratch_token = scratch_token
    state.scratch_token_needs_write_barrier = bool(scratch_token_needs_write_barrier)
    state.local_memory_tokens = dict(local_memory_tokens)
    state.local_memory_pending_accesses = dict(local_memory_pending_accesses)
    state.local_memory_access_tokens = dict(local_memory_access_tokens)
    state.local_memory_read_tokens = dict(local_memory_read_tokens)
    state.pending_mma_read_boundaries = dict(pending_mma_read_boundaries)
    state.materialized_mma_read_boundaries = dict(materialized_mma_read_boundaries)
    state.local_memory_root_sets = dict(local_memory_root_sets)
    state.token_local_memory_root_sets = dict(token_local_memory_root_sets)


def _flatten_structured_values(
    state,
    values,
    target_value_ids,
    context,
    op,
    *,
    expected_shapes=None,
    preserve_mask_predicates=False,
    preserve_mma_packet_payloads=False,
):
    if len(values) != len(target_value_ids):
        fail(
            "TLXW_EMIT_STRUCTURED_COMPONENT_SHAPE",
            STAGE,
            f"{context} value and target id counts do not match",
            target_op_id=op.target_op_id,
        )
    if expected_shapes is not None and len(expected_shapes) != len(values):
        fail(
            "TLXW_EMIT_STRUCTURED_COMPONENT_SHAPE",
            STAGE,
            f"{context} expected shape count does not match values",
            target_op_id=op.target_op_id,
        )
    flat_values = []
    shapes = []
    for index, (value, target_value_id) in enumerate(zip(values, target_value_ids)):
        expected_shape = None if expected_shapes is None else expected_shapes[index]
        target_type = state.target_program.values[target_value_id].type
        component_count = int(target_type.component_count)
        if target_type.representation in {"mask", "mask_tuple"}:
            lane_width = int(target_type.lane_width or 64)
            components, predicates = _as_mask_payload_components_with_predicates(
                state,
                value,
                component_count,
                lane_width,
                op,
            )
            if expected_shape is not None and not expected_shape.is_mask_payload:
                fail(
                    "TLXW_EMIT_STRUCTURED_COMPONENT_SHAPE",
                    STAGE,
                    f"{context} expected mask shape is invalid",
                    target_op_id=op.target_op_id,
                    target_value_id=target_value_id,
                )
            preserve_predicates = bool(preserve_mask_predicates)
            if expected_shape is not None:
                preserve_predicates = int(expected_shape.mask_predicate_component_count) > 0
            predicate_count = 0
            if preserve_predicates:
                if predicates is None:
                    predicates = tuple(_i32_payload_to_mask(state, component, lane_width) for component in components)
                predicate_count = len(predicates)
                components = (*components, *predicates)
            shapes.append(_LoopValueShape(
                len(components),
                is_mask_payload=True,
                mask_payload_component_count=component_count,
                mask_predicate_component_count=predicate_count,
            ))
        elif isinstance(value, _VectorPacketPayload):
            if int(value.logical_component_count) != component_count:
                fail(
                    "TLXW_EMIT_STRUCTURED_COMPONENT_SHAPE",
                    STAGE,
                    "vector packet payload logical component count must match "
                    f"{context} target type",
                    target_op_id=op.target_op_id,
                    target_value_id=target_value_id,
                )
            components = value.packets
            shapes.append(_LoopValueShape(
                len(components),
                packet_width=int(value.packet_width),
                logical_component_count=int(value.logical_component_count),
            ))
        else:
            components = _as_components(value)
            shape = None
            if (preserve_mma_packet_payloads
                    and target_type.representation in _MMA_PACKET_REPRESENTATIONS):
                components, shape = _preserve_loop_vector_payload_components(
                    state,
                    components,
                    target_value_id,
                    context,
                    op,
                )
            if shape is None:
                shape = _LoopValueShape(len(components))
            shapes.append(shape)
        flat_values.extend(components)
    return tuple(flat_values), tuple(shapes)


def _preserve_loop_vector_payload_components(
    state,
    components,
    target_value_id,
    context,
    op,
):
    components = tuple(components)
    if not components:
        return components, _LoopValueShape(0)
    expected_payload = None
    expected_type = None
    for component in components:
        payload = _simd_1d_vector_payload(state, component)
        if payload is None:
            fail(
                "TLXW_EMIT_FOR_MMA_PACKET",
                STAGE,
                f"{context} MMA packet loop carry must be a SIMD vector payload",
                target_op_id=op.target_op_id,
                target_value_id=target_value_id,
            )
        width, element_type, lane_width = payload
        if int(width) <= 0:
            fail(
                "TLXW_EMIT_FOR_MMA_PACKET",
                STAGE,
                f"{context} MMA packet loop carry has an invalid vector width",
                target_op_id=op.target_op_id,
                target_value_id=target_value_id,
            )
        payload_key = (int(width), str(element_type), int(lane_width))
        if expected_payload is None:
            expected_payload = payload_key
            expected_type = component.type
        elif (payload_key != expected_payload
              or str(component.type) != str(expected_type)):
            fail(
                "TLXW_EMIT_FOR_MMA_PACKET",
                STAGE,
                f"{context} MMA packet loop carry components must have matching vector payload types",
                target_op_id=op.target_op_id,
                target_value_id=target_value_id,
            )
    # Keep ordinary typed packets intact through structured control flow.  A
    # fragment is only a zero-cost MMA view of one of these packets, so
    # scalarizing the backedge would just recreate the same tuple at every MMA
    # boundary and enlarge the allocator's simultaneously-live value set.
    shape = _LoopValueShape(
        len(components),
        logical_component_count=len(components),
        preserved_vector_payload_key=expected_payload,
        preserved_vector_payload_type=expected_type,
    )
    return components, shape


def _packed_vector_payload_elements(component, scalar_type, width):
    """Fold scalarization through a matching wave.pack producer."""
    owner = getattr(component, "owner", None)
    operation = getattr(owner, "operation", owner)
    if operation is None or str(getattr(operation, "name", "")) != "wave.pack":
        return None
    operands = tuple(operation.operands)
    if len(operands) != int(width):
        return None
    if any(str(operand.type) != str(scalar_type) for operand in operands):
        return None
    return operands


def _packed_vector_payload_is_splat(elements):
    """Return whether packed payload elements all reference the same value."""
    if not elements:
        return False
    first = elements[0]
    return all(element == first for element in elements[1:])


def _pack_structured_value_components(state, components, shape, context, op):
    components = tuple(components)
    if len(components) != int(shape.component_count):
        fail(
            "TLXW_EMIT_STRUCTURED_COMPONENT_SHAPE",
            STAGE,
            f"{context} component slice does not match recorded value shape",
            target_op_id=None if op is None else op.target_op_id,
        )
    if shape.is_mask_payload:
        payload_count = (int(shape.mask_payload_component_count)
                         if shape.mask_payload_component_count is not None else len(components))
        predicate_count = int(shape.mask_predicate_component_count)
        payload_components = components[:payload_count]
        if predicate_count:
            predicates = components[payload_count:payload_count + predicate_count]
            if len(predicates) != predicate_count or len(payload_components) != predicate_count:
                fail(
                    "TLXW_EMIT_STRUCTURED_COMPONENT_SHAPE",
                    STAGE,
                    f"{context} mask payload predicate shape is invalid",
                    target_op_id=None if op is None else op.target_op_id,
            )
            return _I32MaskPayload(payload_components, predicates=predicates)
        return _I32MaskPayload(payload_components)
    if shape.packet_width is not None:
        return _VectorPacketPayload(
            components,
            int(shape.packet_width),
            int(shape.logical_component_count),
        )
    if shape.preserved_vector_payload_key is not None:
        logical_component_count = int(shape.logical_component_count or 0)
        if logical_component_count <= 0 or len(components) != logical_component_count:
            fail(
                "TLXW_EMIT_STRUCTURED_COMPONENT_SHAPE",
                STAGE,
                f"{context} preserved vector packet shape is invalid",
                target_op_id=None if op is None else op.target_op_id,
            )
        for component in components:
            payload = _simd_1d_vector_payload(state, component)
            if payload is None:
                fail(
                    "TLXW_EMIT_STRUCTURED_COMPONENT_SHAPE",
                    STAGE,
                    f"{context} preserved vector packet component is invalid",
                    target_op_id=None if op is None else op.target_op_id,
                )
            width, element_type, lane_width = payload
            payload_key = (int(width), str(element_type), int(lane_width))
            if (payload_key != shape.preserved_vector_payload_key
                    or str(component.type) != str(shape.preserved_vector_payload_type)):
                fail(
                    "TLXW_EMIT_STRUCTURED_COMPONENT_SHAPE",
                    STAGE,
                    f"{context} preserved vector packet type changed",
                    target_op_id=None if op is None else op.target_op_id,
                )
        return _pack_components(components)
    return _pack_components(components)


def _pack_loop_value_components(state, components, shape, op=None):
    return _pack_structured_value_components(state, components, shape, "for_loop", op)


def _bind_loop_region_args(
    state,
    block_arg_ids,
    induction_value,
    flat_iter_values,
    init_shapes,
    init_target_ids,
    *,
    preserved_local_memory_roots=(),
    op,
):
    state.values[block_arg_ids[0]] = induction_value
    cursor = 0
    for block_arg_id, shape, init_target_id in zip(block_arg_ids[1:], init_shapes, init_target_ids):
        state.values[block_arg_id] = _pack_loop_value_components(
            state,
            flat_iter_values[cursor:cursor + shape.component_count],
            shape,
            op,
        )
        _propagate_token_local_memory_roots(
            state,
            init_target_id,
            block_arg_id,
            preserved_local_memory_roots=preserved_local_memory_roots,
        )
        cursor += shape.component_count
    if cursor != len(flat_iter_values):
        fail(
            "TLXW_EMIT_FOR_BLOCK_COMPONENTS",
            STAGE,
            "for_loop iter block arg component count does not match init args",
            target_op_id=op.target_op_id,
        )


def _emit_local_alloc(state, op):
    attrs = target_ir.attrs_dict(op)
    result_id = _single_result(op)
    value = state.builder.workgroup_alloc(
        int(attrs["allocation_bytes"]),
        int(attrs.get("align", 16)),
        _scalar_type(state.dsl, attrs["element_type"]),
    )
    state.values[result_id] = value
    state.shared_pointer_dword_bases[result_id] = _SharedPointerDwordBase(value)
    _record_local_memory_root(state, result_id)
    state.local_memory_allocations[int(result_id)] = value


def _emit_memdesc_index(state, op):
    attrs = target_ir.attrs_dict(op)
    base, index = _operand_values(state, op, 2)
    result_id = _single_result(op)
    state.local_memory_root_sets[int(result_id)] = _memdesc_index_result_roots(
        state,
        _local_memory_roots(state, op.operands[0]),
        attrs,
        op.operands[1],
    )
    static_byte_offset = attrs.get("static_byte_offset")
    if static_byte_offset is not None:
        target_type = state.target_program.values[result_id].type
        element_byte_width = attrs.get("element_byte_width")
        if element_byte_width is None or int(static_byte_offset) % int(element_byte_width):
            fail(
                "TLXW_EMIT_UNSUPPORTED_MEMDESC_INDEX",
                STAGE,
                "static ttg.memdesc_index offset is not element aligned",
                target_op_id=op.target_op_id,
            )
        pointer_type = state.dsl.ptr_type(
            _scalar_type(state.dsl, target_type.element_type),
            state.dsl.shared_address_space(),
        )
        value = _ptr_cast(state, base, pointer_type)
        element_offset = int(static_byte_offset) // int(element_byte_width)
        if element_offset:
            offset = state.builder.constant(state.dsl.i32(), element_offset)
            value = state.builder.ptr_add(
                value,
                offset,
                result_type=pointer_type,
            )
        state.values[result_id] = value
        _record_static_memdesc_dword_base(
            state,
            result_id,
            op.operands[0],
            int(static_byte_offset),
        )
        return
    if isinstance(index, tuple):
        fail(
            "TLXW_EMIT_UNSUPPORTED_MEMDESC_INDEX",
            STAGE,
            "ttg.memdesc_index requires a scalar slot index",
            target_op_id=op.target_op_id,
        )
    elements_per_slot = int(attrs["elements_per_slot"])
    offset = index
    if elements_per_slot != 1:
        stride = state.builder.constant(index.type, elements_per_slot)
        offset = state.builder.binary(
            state.dsl.BinaryKind.MulI,
            index,
            stride,
            nsw=_LAYOUT_MATH_NSW,
        )
    state.values[result_id] = state.builder.ptr_add(
        base,
        offset,
        result_type=base.type,
    )
    _record_dynamic_memdesc_dword_base(
        state,
        op,
        result_id,
        op.operands[0],
        index,
        elements_per_slot,
        attrs.get("element_byte_width"),
    )


def _record_static_memdesc_dword_base(state, result_id, base_id, byte_offset):
    base_plan = state.shared_pointer_dword_bases.get(base_id)
    if base_plan is None or int(byte_offset) % 4:
        return
    base = _shared_pointer_with_dword_offset(
        state,
        base_plan.base,
        int(byte_offset) // 4,
        cache_key=("memdesc_static", int(byte_offset) // 4),
    )
    state.shared_pointer_dword_bases[result_id] = _SharedPointerDwordBase(
        base,
        base_plan.dword_offset,
    )


def _record_dynamic_memdesc_dword_base(
    state,
    op,
    result_id,
    base_id,
    index,
    elements_per_slot,
    element_byte_width,
):
    base_plan = state.shared_pointer_dword_bases.get(base_id)
    if base_plan is None or element_byte_width is None:
        return
    if str(index.type) != str(state.dsl.i32()):
        return
    slot_bytes = int(elements_per_slot) * int(element_byte_width)
    if slot_bytes % 4:
        return
    slot_dwords = slot_bytes // 4
    static_index = _target_constant_int(state, op.operands[1])
    if static_index is None:
        index_base_target_id, static_slot_offset = _target_additive_base(
            state,
            op.operands[1],
        )
        dynamic_index = state.values.get(int(index_base_target_id), index)
    else:
        static_slot_offset = int(static_index)
        dynamic_index = None
    if dynamic_index is None:
        dword_offset = None
    elif slot_dwords == 1:
        dword_offset = dynamic_index
    else:
        dword_offset = _scalar_binary_const_i32(
            state,
            "muli",
            dynamic_index,
            slot_dwords,
            nsw=_LAYOUT_MATH_NSW,
        )
    base = _shared_pointer_with_dword_offset(
        state,
        base_plan.base,
        int(static_slot_offset) * slot_dwords,
        cache_key=(
            "memdesc_dynamic_static",
            int(static_slot_offset),
            int(slot_dwords),
        ),
    )
    state.shared_pointer_dword_bases[result_id] = _SharedPointerDwordBase(
        base,
        _combine_optional_i32_offsets(
            state,
            base_plan.dword_offset,
            dword_offset,
            nsw=_LAYOUT_MATH_NSW,
        ),
    )


def _emit_memdesc_view(state, op):
    base = _operand_values(state, op, 1)[0]
    result_id = _single_result(op)
    state.values[result_id] = base
    _propagate_local_memory_roots(state, op.operands[0], result_id)
    dword_base = state.shared_pointer_dword_bases.get(op.operands[0])
    if dword_base is not None:
        state.shared_pointer_dword_bases[result_id] = dword_base


def _local_access_result_ids(op):
    attrs = target_ir.attrs_dict(op)
    data_count = int(attrs.get("data_result_count", len(op.results)))
    completion_count = int(attrs.get("completion_result_count", 0))
    if (
        data_count < 0
        or completion_count < 0
        or data_count + completion_count != len(op.results)
    ):
        fail(
            "TLXW_EMIT_LOCAL_ACCESS_RESULTS",
            STAGE,
            "local access result segments do not match target results",
            target_op_id=op.target_op_id,
        )
    return (
        tuple(int(result_id) for result_id in op.results[:data_count]),
        tuple(int(result_id) for result_id in op.results[data_count:]),
    )


def _finish_local_access(state, op, memdesc_target_id, token, access_kind):
    attrs = target_ir.attrs_dict(op)
    _data_result_ids, completion_result_ids = _local_access_result_ids(op)
    for result_id in completion_result_ids:
        state.values[result_id] = token
        state.token_local_memory_root_sets[result_id] = _local_memory_roots(
            state,
            memdesc_target_id,
        )
    if (
        bool(attrs.get("protocol_tracked", False))
        and bool(attrs.get("synced_via_async_wait", False))
        and access_kind in {"read", "mma_read"}
    ):
        # The target SSA completion result is the sole release frontier for a
        # wait-ready DMA consumer.  Retain only allocation-lifetime history;
        # do not rebuild a second pending/read frontier in emitter state.
        _record_local_memory_access_token(
            state,
            _local_memory_roots(state, memdesc_target_id),
            token,
        )
        return
    _set_local_memory_access_token(
        state,
        memdesc_target_id,
        token,
        access_kind,
    )


def _emit_local_store(state, op):
    attrs = target_ir.attrs_dict(op)
    if len(op.operands) < 2:
        fail(
            "TLXW_EMIT_LOCAL_STORE_OPERANDS",
            STAGE,
            "local_store requires value and memdesc operands",
            target_op_id=op.target_op_id,
        )
    values = _require_value(state, op.operands[0], op)
    base = _require_value(state, op.operands[1], op)
    explicit_dependencies = tuple(
        _require_value(state, target_value_id, op)
        for target_value_id in op.operands[2:]
    )
    memdesc_target_id = op.operands[1]
    lane_width = int(attrs["lane_width"])
    component_count = int(attrs["component_count"])
    offsets = _local_access_offsets(state, attrs, component_count, lane_width, op)
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    base = _ptr_cast(
        state,
        base,
        state.dsl.ptr_type(element_type, state.dsl.shared_address_space()),
    )
    dependency = _local_memory_access_dependency_token(
        state,
        memdesc_target_id,
        "write",
        extra_tokens=explicit_dependencies,
    )
    if isinstance(values, _VectorPacketPayload):
        value_components = _value_components(state, values, op)
    else:
        value_components = _broadcast_component(state, values, component_count, op)
    splat_cache = []
    value_components = tuple(
        _memory_simd_component(
            state,
            value_component,
            attrs["element_type"],
            lane_width,
            op,
            splat_cache,
        ) for value_component in value_components)
    token = _emit_symbolic_shared_scatter(
        state,
        value_components,
        offsets,
        base,
        element_type,
        lane_width,
        int(attrs["element_byte_width"]),
        dependency,
        op,
    )
    _finish_local_access(state, op, memdesc_target_id, token, "write")


def _symbolic_shared_packet(state, components, element_type, lane_width):
    components = tuple(components)
    packet_type = state.dsl.simd_type(
        state.dsl.vector_type(len(components), element_type),
        width=int(lane_width),
    )
    return state.dsl.wave.PackOp(packet_type, components).result


def _symbolic_shared_mapping(state, element_byte_width):
    offset_symbol = state.dsl.sym("offset")
    bit_offset = int(element_byte_width) * 8 * offset_symbol
    return offset_symbol, bit_offset


def _symbolic_shared_offset_packet(state, offsets, lane_width):
    offsets = tuple(offsets)
    offset_element_type = state.dsl.SimdType(offsets[0].type).element_type
    return _symbolic_shared_packet(
        state, offsets, offset_element_type, lane_width
    )


def _emit_symbolic_shared_scatter(
    state,
    value_components,
    offsets,
    base,
    element_type,
    lane_width,
    element_byte_width,
    dependency,
    op,
):
    value_components = tuple(value_components)
    offsets = tuple(offsets)
    if not value_components or len(value_components) != len(offsets):
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "symbolic local_store values and offsets must have the same positive component count",
            target_op_id=op.target_op_id,
        )
    value_packet = _symbolic_shared_packet(
        state, value_components, element_type, lane_width
    )
    offset_packet = _symbolic_shared_offset_packet(state, offsets, lane_width)
    offset_symbol, bit_offset = _symbolic_shared_mapping(
        state, element_byte_width
    )
    return state.builder.scatter(
        value_packet,
        [base],
        bit_offset=bit_offset,
        packet_bindings={offset_symbol: offset_packet},
        after=dependency,
    )


def _emit_symbolic_shared_gather(
    state,
    offsets,
    base,
    element_type,
    lane_width,
    element_byte_width,
    dependency,
    op,
):
    offsets = tuple(offsets)
    if not offsets:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "symbolic local_load requires a positive component count",
            target_op_id=op.target_op_id,
        )
    result_type = state.dsl.simd_type(
        state.dsl.vector_type(len(offsets), element_type),
        width=int(lane_width),
    )
    offset_packet = _symbolic_shared_offset_packet(state, offsets, lane_width)
    offset_symbol, bit_offset = _symbolic_shared_mapping(
        state, element_byte_width
    )
    return state.builder.gather(
        [base],
        result_type,
        bit_offset=bit_offset,
        packet_bindings={offset_symbol: offset_packet},
        after=dependency,
    )


def _symbolic_shared_packet_components(
    state, packet, component_count, element_type, lane_width
):
    component_type = state.dsl.simd_type(element_type, int(lane_width))
    return tuple(
        state.dsl.wave.ExtractOp(component_type, packet, index).result
        for index in range(int(component_count))
    )


def _symbolic_contiguous_mapping(
    state,
    offset_element_byte_width,
    packet_element_bit_width,
    static_bit_offset=0,
):
    offset = state.dsl.sym("offset")
    slot = state.dsl.sym("slot")
    bit_offset = (
        int(offset_element_byte_width) * 8 * offset
        + int(packet_element_bit_width) * slot
        + int(static_bit_offset)
    )
    return offset, bit_offset


def _symbolic_packet_value(state, value, element_type, lane_width):
    if _simd_1d_vector_payload(state, value) is not None:
        return value
    packet_type = state.dsl.simd_type(
        state.dsl.vector_type(1, element_type),
        width=int(lane_width),
    )
    return state.dsl.wave.PackOp(packet_type, [value]).result


def _emit_symbolic_contiguous_scatter(
    state,
    value,
    offset,
    base,
    element_type,
    lane_width,
    element_byte_width,
    *,
    dependency=None,
    cache=None,
):
    packet = _symbolic_packet_value(
        state, value, element_type, lane_width
    )
    offset_symbol, bit_offset = _symbolic_contiguous_mapping(
        state,
        element_byte_width,
        int(element_byte_width) * 8,
    )
    return state.builder.scatter(
        packet,
        [base],
        bit_offset=bit_offset,
        bindings={offset_symbol: offset},
        after=dependency,
        cache=cache,
    )


def _emit_contiguous_store(
    state,
    value,
    offset,
    base,
    lane_width,
    *,
    dependency=None,
    cache=None,
):
    ptr = state.builder.ptr_add(
        base,
        _simd_offset_value(state, offset, lane_width),
    )
    return state.builder.store(
        value,
        ptr,
        after=dependency,
        cache=cache,
    )


def _emit_symbolic_contiguous_gather(
    state,
    offset,
    base,
    result_type,
    packet_element_type,
    lane_width,
    offset_element_byte_width,
    packet_element_bit_width,
    *,
    dependency=None,
    cache=None,
    static_bit_offset=0,
):
    try:
        result_simd = state.dsl.SimdType(result_type)
        result_vector = state.dsl.VectorType(result_simd.element_type)
        result_shape = tuple(int(dim) for dim in result_vector.shape)
        packet_payload = result_shape if len(result_shape) == 1 else None
    except Exception:
        packet_payload = None
    if packet_payload is None:
        packet_type = state.dsl.simd_type(
            state.dsl.vector_type(1, packet_element_type),
            width=int(lane_width),
        )
    else:
        packet_type = result_type
    offset_symbol, bit_offset = _symbolic_contiguous_mapping(
        state,
        offset_element_byte_width,
        packet_element_bit_width,
        static_bit_offset,
    )
    packet, token = state.builder.gather(
        [base],
        packet_type,
        bit_offset=bit_offset,
        bindings={offset_symbol: offset},
        after=dependency,
        cache=cache,
    )
    if packet_payload is not None:
        return packet, token
    value = state.dsl.wave.ExtractOp(result_type, packet, 0).result
    return value, token


def _emit_local_load(state, op):
    attrs = target_ir.attrs_dict(op)
    base, *explicit_dependencies = _operand_values(state, op, len(op.operands))
    memdesc_target_id = op.operands[0]
    data_result_ids, _completion_result_ids = _local_access_result_ids(op)
    if len(data_result_ids) != 1:
        fail(
            "TLXW_EMIT_LOCAL_ACCESS_RESULTS",
            STAGE,
            "local_load requires one data result",
            target_op_id=op.target_op_id,
        )
    result_id = data_result_ids[0]
    target_type = state.target_program.values[result_id].type
    lane_width = int(attrs["lane_width"])
    component_count = int(attrs["component_count"])
    raw_packet_indices = _local_load_raw_packet_indices(
        attrs,
        component_count,
        op,
    )
    offsets = _local_access_offsets(
        state,
        attrs,
        component_count,
        lane_width,
        op,
        component_indices=raw_packet_indices,
    )
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    base = _ptr_cast(
        state,
        base,
        state.dsl.ptr_type(element_type, state.dsl.shared_address_space()),
    )
    dependency = _local_memory_access_dependency_token(
        state,
        memdesc_target_id,
        "read",
        extra_tokens=tuple(explicit_dependencies),
        # The explicit source wait is the sole readiness proof for an async
        # DMA write.  It does not waive an unrelated synchronous LDS hazard,
        # such as a preceding local_store to the same allocation.
        ignore_async_writes=bool(attrs.get("synced_via_async_wait", False)),
        ready_async_write_roots=_local_load_ready_async_write_roots(
            state,
            op,
            attrs,
        ),
    )
    if raw_packet_indices is not None:
        payload, token = _emit_local_load_raw_layout_vector_packets(
            state,
            op,
            attrs,
            memdesc_target_id,
            base,
            offsets,
            lane_width,
            dependency,
        )
        state.values[result_id] = payload
        _finish_local_access(state, op, memdesc_target_id, token, "read")
        return
    if attrs.get("result_value_mode") == "mma_packet_payload":
        payload, token = _emit_local_load_mma_packet_payload(
            state,
            op,
            attrs,
            base,
            offsets,
            element_type,
            target_type,
            lane_width,
            dependency,
        )
        state.values[result_id] = payload
        _finish_local_access(state, op, memdesc_target_id, token, "read")
        return
    result_type = _wave_type(state.dsl, target_type)
    vector_packet_indices = _local_load_vector_packet_indices(
        state,
        attrs,
        component_count,
        op,
    )
    if vector_packet_indices is not None:
        result_packet_width = int(attrs["result_packet_width"])
        transpose_packet_width = int(attrs.get("result_transpose_packet_width", result_packet_width))
        loaded_packets = []
        tokens = []
        for index in vector_packet_indices:
            loaded, token = _emit_i8_transpose_gather(
                state,
                attrs,
                index,
                base,
                element_type,
                lane_width,
                dependency,
            )
            loaded_packets.append(loaded)
            tokens.append(token)
        packets = []
        for loaded in loaded_packets:
            packets.extend(
                _split_i8_transpose_packet(
                    state,
                    loaded,
                    element_type,
                    lane_width,
                    transpose_packet_width,
                    result_packet_width,
                ))
        state.values[result_id] = _VectorPacketPayload(
            tuple(packets),
            result_packet_width,
            component_count,
        )
        if tokens:
            _finish_local_access(
                state,
                op,
                memdesc_target_id,
                _join_memory_tokens(state, tokens),
                "read",
            )
        return
    uses_transpose_load = any(
        _local_load_transpose_packet_elements(
            state,
            attrs,
            index,
            component_count,
            op,
        ) > 1
        for index in range(component_count)
    )
    if not uses_transpose_load:
        loaded, token = _emit_symbolic_shared_gather(
            state,
            offsets,
            base,
            element_type,
            lane_width,
            int(attrs["element_byte_width"]),
            dependency,
            op,
        )
        components = _symbolic_shared_packet_components(
            state,
            loaded,
            component_count,
            element_type,
            lane_width,
        )
        state.values[result_id] = _pack_components(components)
        _finish_local_access(state, op, memdesc_target_id, token, "read")
        return
    component_entries = []
    tokens = []
    index = 0
    component_type = state.dsl.simd_type(element_type, lane_width)
    while index < component_count:
        transpose_packet = _local_load_transpose_packet_elements(
            state,
            attrs,
            index,
            component_count,
            op,
        )
        if transpose_packet > 1:
            component_type = state.dsl.simd_type(element_type, lane_width)
            loaded, token = _emit_i8_transpose_gather(
                state,
                attrs,
                index,
                base,
                element_type,
                lane_width,
                dependency,
            )
            component_entries.append(_DeferredPacketLocalLoad(loaded, int(transpose_packet)))
            tokens.append(token)
            index += int(transpose_packet)
            continue
        loaded_packet, token = _emit_symbolic_shared_gather(
            state,
            (offsets[index],),
            base,
            element_type,
            lane_width,
            int(attrs["element_byte_width"]),
            dependency,
            op,
        )
        loaded = state.dsl.wave.ExtractOp(
            component_type,
            loaded_packet,
            0,
        ).result
        component_entries.append(loaded)
        tokens.append(token)
        index += 1
    components = []
    for entry in component_entries:
        if isinstance(entry, _DeferredPacketLocalLoad):
            for element in range(int(entry.element_count)):
                components.append(state.dsl.wave.ExtractOp(
                    component_type,
                    entry.loaded,
                    int(element),
                ).result)
            continue
        components.append(entry)
    state.values[result_id] = _pack_components(tuple(components))
    if tokens:
        _finish_local_access(
            state,
            op,
            memdesc_target_id,
            _join_memory_tokens(state, tokens),
            "read",
        )


def _local_load_raw_packet_indices(attrs, component_count, op):
    if attrs.get("result_value_mode") != "raw_layout_vector_packets":
        return None
    packet_width = int(attrs.get("result_packet_width", 0))
    element_bit_width = int(attrs.get("result_element_bit_width", 0))
    indices = tuple(
        int(index) for index in attrs.get("raw_packet_component_indices", ())
    )
    expected = tuple(range(0, int(component_count), packet_width or 1))
    packet_bits = packet_width * element_bit_width
    if (
        packet_width <= 1
        or element_bit_width <= 0
        or int(component_count) % packet_width
        or indices != expected
        or packet_bits <= 0
        or packet_bits > 128
        or packet_bits % 32
        or 32 % element_bit_width
    ):
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "raw packet local_load attrs do not describe uniform, legal "
            "register packets",
            target_op_id=op.target_op_id,
        )
    return indices


def _emit_local_load_raw_layout_vector_packets(
    state,
    op,
    attrs,
    memdesc_target_id,
    base,
    offsets,
    lane_width,
    dependency,
):
    """Load proven-contiguous LDS elements as untyped register packets."""
    packet_width = int(attrs["result_packet_width"])
    element_bit_width = int(attrs["result_element_bit_width"])
    packet_bits = packet_width * element_bit_width
    load_type = _raw_register_packet_type(
        state,
        packet_bits,
        lane_width,
        op,
    )
    dword_base_type = state.dsl.ptr_type(
        state.dsl.i32(),
        state.dsl.shared_address_space(),
    )
    dword_base_plan = state.shared_pointer_dword_bases.get(
        int(memdesc_target_id)
    )
    if dword_base_plan is None:
        dword_base = _ptr_cast(state, base, dword_base_type)
        dynamic_base_offset = None
    else:
        # Collapse the memdesc ring index and the per-workitem packet address
        # into one dword pointer offset.  Leaving the dynamic ring index in a
        # parent ptr_add gives symbolic pointer lowering two unrelated offset
        # trees to merge.
        dword_base = _ptr_cast(state, dword_base_plan.base, dword_base_type)
        dynamic_base_offset = dword_base_plan.dword_offset
    dword_ptr_type = state.dsl.simd_ptr_type(
        state.dsl.i32(),
        state.dsl.shared_address_space(),
        lane_width,
    )
    elements_per_dword = 32 // element_bit_width
    packets = []
    tokens = []
    for offset in offsets:
        dword_offset = _simd_binary_const(
            state,
            "divui",
            offset,
            elements_per_dword,
            lane_width,
        )
        if dynamic_base_offset is not None:
            dword_offset = state.builder.binary(
                state.dsl.BinaryKind.AddI,
                dword_offset,
                _simd_offset_value(
                    state,
                    dynamic_base_offset,
                    lane_width,
                ),
                nsw=_LAYOUT_MATH_NSW,
            )
        ptr = state.builder.ptr_add(
            dword_base,
            dword_offset,
            result_type=dword_ptr_type,
        )
        packet, token = state.builder.load(
            ptr,
            load_type,
            after=dependency,
        )
        packets.append(packet)
        tokens.append(token)
    return (
        _RawLayoutVectorPacketPayload(
            tuple(packets),
            packet_width,
            int(attrs["component_count"]),
            element_bit_width,
        ),
        _join_memory_tokens(state, tokens),
    )


def _emit_local_load_mma_packet_payload(
    state,
    op,
    attrs,
    base,
    offsets,
    element_type,
    target_type,
    lane_width,
    dependency,
):
    if target_type.representation not in _MMA_PACKET_REPRESENTATIONS:
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "MMA packet local_load requires an MMA packet result type",
            target_op_id=op.target_op_id,
        )
    component_count = int(attrs["component_count"])
    if len(offsets) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "MMA packet local_load offset count does not match attrs",
            target_op_id=op.target_op_id,
        )
    registers = int(attrs.get("registers", 0))
    if registers <= 1:
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "MMA packet local_load requires multiple registers",
            target_op_id=op.target_op_id,
        )
    load_type = state.dsl.simd_type(
        state.dsl.vector_type(registers, element_type),
        width=int(lane_width),
    )
    expanded_offsets = tuple(
        offset
        if element == 0
        else _simd_binary_const(
            state,
            "addi",
            offset,
            element,
            lane_width,
            nsw=_LAYOUT_MATH_NSW,
        )
        for offset in offsets
        for element in range(registers)
    )
    loaded, token = _emit_symbolic_shared_gather(
        state,
        expanded_offsets,
        base,
        element_type,
        lane_width,
        int(attrs["element_byte_width"]),
        dependency,
        op,
    )
    scalar_components = _symbolic_shared_packet_components(
        state,
        loaded,
        component_count * registers,
        element_type,
        lane_width,
    )
    components = tuple(
        state.dsl.wave.PackOp(
            load_type,
            scalar_components[index * registers:(index + 1) * registers],
        ).result
        for index in range(component_count)
    )
    return _pack_components(components), token


def _local_load_vector_packet_indices(state, attrs, component_count, op):
    if attrs.get("result_value_mode") != "transpose_vector_packets":
        return None
    if int(attrs.get("result_packet_width", 0)) <= 1:
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "transpose vector-packet local_load requires result_packet_width",
            target_op_id=op.target_op_id,
        )
    if int(attrs.get("result_transpose_packet_width", 0)) <= 1:
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "transpose vector-packet local_load requires result_transpose_packet_width",
            target_op_id=op.target_op_id,
        )
    if int(attrs["result_transpose_packet_width"]) % int(attrs["result_packet_width"]):
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "transpose vector-packet local_load packet widths are incompatible",
            target_op_id=op.target_op_id,
        )
    if component_count % int(attrs["result_transpose_packet_width"]):
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "transpose vector-packet local_load requires uniform transpose packets",
            target_op_id=op.target_op_id,
        )
    if attrs.get("element_type") != "i8" or int(attrs.get("element_byte_width", 0)) != 1:
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "transpose vector-packet local_load currently requires i8 elements",
            target_op_id=op.target_op_id,
        )
    if int(attrs["result_packet_width"]) != 4 or int(attrs["result_transpose_packet_width"]) != 8:
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "unsupported transpose vector-packet local_load packet shape",
            target_op_id=op.target_op_id,
        )
        return None
    indices = []
    index = 0
    while index < int(component_count):
        transpose_packet = _local_load_transpose_packet_elements(
            state,
            attrs,
            index,
            component_count,
            op,
        )
        if transpose_packet != 8:
            return None
        indices.append(index)
        index += int(transpose_packet)
    return tuple(indices)


def _split_i8_transpose_packet(state, loaded, element_type, lane_width, transpose_packet_width, result_packet_width):
    return tuple(
        _extract_i8_vector_packet(
            state,
            loaded,
            element_type,
            lane_width,
            first_element,
            result_packet_width,
        ) for first_element in range(0, int(transpose_packet_width), int(result_packet_width)))


def _extract_i8_vector_packet(state, loaded, element_type, lane_width, first_element, packet_width):
    component_type = state.dsl.simd_type(element_type, int(lane_width))
    packet_type = state.dsl.simd_type(
        state.dsl.vector_type(int(packet_width), element_type),
        int(lane_width),
    )
    return state.dsl.wave.PackOp(
        packet_type,
        tuple(
            state.dsl.wave.ExtractOp(
                component_type,
                loaded,
                int(first_element) + int(element),
            ).result for element in range(int(packet_width))),
    ).result


def _local_load_transpose_packet_elements(state, attrs, index, component_count, op):
    if int(index) + 8 > int(component_count):
        return 1
    if attrs.get("destination_offset_mode") != "layout_coordinates":
        return 1
    if attrs.get("element_type") != "i8":
        return 1
    if int(attrs.get("element_byte_width", 0)) != 1:
        return 1
    if int(attrs.get("lane_width", 0)) != 64:
        return 1
    if attrs.get("destination_physical_offset_plan") != "swizzled_xor":
        return 1
    if attrs.get("destination_physical_offset_unit") != "element":
        return 1
    if int(attrs.get("destination_physical_element_byte_width", 0)) != 1:
        return 1
    if tuple(int(value) for value in attrs.get("destination_physical_order", ())) != (0, 1):
        return 1
    if int(attrs.get("destination_physical_swizzled_vec", 0)) != 1:
        return 1
    if int(attrs.get("destination_physical_swizzled_per_phase", 0)) != 1:
        return 1
    if int(attrs.get("destination_physical_swizzled_max_phase", 0)) != 1:
        return 1
    shape = tuple(int(value) for value in attrs["destination_coordinate_shape"])
    physical_shape = tuple(int(value) for value in attrs.get("destination_physical_shape", shape))
    logical_origin = tuple(int(value) for value in attrs.get("destination_logical_origin", (0, ) * len(shape)))
    if physical_shape != shape or any(logical_origin):
        return 1
    if len(shape) != 2:
        return 1
    component_bases = tuple(
        tuple(int(value) for value in bases) for bases in attrs["destination_component_coordinate_bases"])
    if int(index) + 8 > len(component_bases):
        return 1
    workitem_coefficients = tuple(
        tuple(int(value)
              for value in coefficients)
        for coefficients in attrs["destination_workitem_coordinate_coefficients"])
    if any(len(coefficients) != len(shape) for coefficients in workitem_coefficients):
        return 1
    # CDNA4 ds_read_b64_tr_b8 loads an 8x16 byte tile from LDS. The Wave op
    # consumes the source-lane pointer for the first logical component and
    # returns the transposed 8-byte payload for the destination lane.
    packet_bases = component_bases[int(index):int(index) + 8]
    first_row, first_col = packet_bases[0]
    if int(first_row) % 32 or int(first_col) % 4:
        return 1
    expected = []
    for row_group in range(4):
        for col_group in range(2):
            expected.append((int(first_row) + row_group * 32, int(first_col) + col_group * 4))
    if tuple(expected) != tuple(packet_bases):
        return 1
    if int(first_row) + 96 >= int(shape[0]) or int(first_col) + 4 >= int(shape[1]):
        return 1
    del state, op
    return 8


def _emit_i8_transpose_gather(
    state,
    attrs,
    index,
    base,
    element_type,
    lane_width,
    dependency,
):
    shape = tuple(int(value) for value in attrs["destination_coordinate_shape"])
    row_extent = int(shape[0])
    component_bases = tuple(
        tuple(int(value) for value in bases)
        for bases in attrs["destination_component_coordinate_bases"]
    )
    first_row, first_col = component_bases[int(index)]
    workitem_coefficients = tuple(
        tuple(int(value) for value in coefficients)
        for coefficients in attrs["destination_workitem_coordinate_coefficients"]
    )
    coefficients = [
        8,
        4 * row_extent,
        32,
        64,
        row_extent,
        2 * row_extent,
    ]
    for bit in range(6, len(workitem_coefficients)):
        row_coeff, col_coeff = workitem_coefficients[bit]
        coefficients.append(int(row_coeff) + int(col_coeff) * row_extent)

    item = state.dsl.sym("item")
    slot = state.dsl.sym("slot")
    lane = state.dsl.mod(item, 64)
    source_item = (
        item
        - lane
        + 16 * state.dsl.floor(lane / 16)
        + state.dsl.floor(state.dsl.mod(lane, 16) / 2)
    )
    source_offset = _bit_affine_thread_offset_expr(
        state,
        source_item,
        int(first_col) * row_extent + int(first_row),
        tuple(coefficients),
    )
    byte_offset = (
        source_offset
        + 4 * state.dsl.mod(lane, 2)
        + state.dsl.floor(slot / 2)
    )
    packet_type = state.dsl.simd_type(
        state.dsl.vector_type(8, element_type),
        width=int(lane_width),
    )
    return state.builder.gather(
        [base],
        packet_type,
        bit_offset=8 * byte_offset,
        after=dependency,
    )


def _b16_transpose_source_item_expr(state, item, slot):
    lane = state.dsl.mod(item, 64)
    return (
        item
        - lane
        + 16 * state.dsl.floor(lane / 16)
        + state.dsl.floor(state.dsl.mod(lane, 16) / 4)
        + 4 * slot
    )


def _emit_b16_transpose_gather(
    state,
    base,
    result_type,
    source_element_offset,
    item,
    dependency,
):
    lane = state.dsl.mod(item, 64)
    element_offset = source_element_offset + state.dsl.mod(lane, 4)
    bit_offset = (16 * element_offset).simplify()
    return state.builder.gather(
        [base],
        result_type,
        bit_offset=bit_offset,
        after=dependency,
    )


def _local_access_offsets(
    state,
    attrs,
    component_count,
    lane_width,
    op,
    *,
    component_indices=None,
):
    if attrs.get("destination_offset_mode") != "layout_coordinates":
        fail(
            "TLXW_EMIT_LOCAL_MEMORY",
            STAGE,
            f"unsupported local-memory offset mode {attrs.get('destination_offset_mode')}",
            target_op_id=op.target_op_id,
        )
    shape = tuple(int(value) for value in attrs["destination_coordinate_shape"])
    component_bases = tuple(
        tuple(int(value) for value in bases) for bases in attrs["destination_component_coordinate_bases"])
    workitem_coefficients = tuple(
        tuple(int(value)
              for value in coefficients)
        for coefficients in attrs["destination_workitem_coordinate_coefficients"])
    if (len(component_bases) != int(component_count) or any(len(bases) != len(shape) for bases in component_bases)
            or any(len(coefficients) != len(shape) for coefficients in workitem_coefficients)):
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "local-memory coordinate offsets do not match component attrs",
            target_op_id=op.target_op_id,
        )
    if component_indices is None:
        component_indices = tuple(range(int(component_count)))
    else:
        component_indices = tuple(int(index) for index in component_indices)
        if (
            len(set(component_indices)) != len(component_indices)
            or any(
                index < 0 or index >= int(component_count)
                for index in component_indices
            )
        ):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "local-memory component selection is invalid",
                target_op_id=op.target_op_id,
            )
    offset_range = _local_access_offset_range(attrs, shape)
    workitem = state.builder.workitem_id(0, state.dsl.i32(), int(lane_width))
    offsets = []
    for component_index in component_indices:
        component_base = component_bases[int(component_index)]
        coords = tuple(
            _bit_linear_thread_coordinate(
                state,
                workitem,
                int(base),
                tuple(coefficients[dim] for coefficients in workitem_coefficients),
                lane_width,
            ) for dim, base in enumerate(component_base))
        offset = _shared_destination_element_offset(
            state,
            attrs,
            coords,
            shape,
            lane_width,
            op,
        )
        offsets.append(_assume_value_range(state, offset, offset_range, op))
    return tuple(offsets)


def _local_access_offset_range(attrs, shape):
    shape = tuple(int(value) for value in attrs.get("destination_physical_shape", shape))
    plan = attrs.get("destination_physical_offset_plan")
    element_count = _product(shape)
    if element_count <= 0:
        return None
    if plan in {"dense_row_major", "linear_shared", "swizzled_xor"}:
        return (0, element_count - 1)
    if plan == "padded_linear":
        upper = element_count - 1
        intervals = tuple(int(value) for value in attrs.get("destination_physical_intervals", ()))
        paddings = tuple(int(value) for value in attrs.get("destination_physical_paddings", ()))
        for interval, padding in zip(intervals, paddings):
            if interval <= 0:
                return None
            upper += (element_count - 1) // int(interval) * int(padding)
        return (0, upper)
    return None


def _emit_buffer_load_to_local(state, op):
    attrs = target_ir.attrs_dict(op)
    issue_dependency_count = int(attrs.get("issue_dependency_count", 0))
    if attrs["mode"] == "dma_packet_lds":
        has_mask = bool(attrs.get("has_mask", False))
        mask_operand_count = int(has_mask)
        operands = _operand_values(
            state,
            op,
            3
            + mask_operand_count
            + issue_dependency_count,
        )
        dest_base, source_base, offsets = operands[:3]
        operand_index = 3
        masks = operands[operand_index] if has_mask else None
        operand_index += mask_operand_count
        issue_dependencies = operands[operand_index:]
        element_type = _scalar_type(state.dsl, attrs["element_type"])
        lane_width = int(attrs["lane_width"])
        component_count = int(attrs["component_count"])
        offset_components = _as_components(offsets)
        if len(offset_components) != component_count:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "packet DMA offset edge does not match component count",
                target_op_id=op.target_op_id,
            )
        mask_components = (
            None
            if masks is None
            else _as_mask_predicate_components(
                state,
                masks,
                component_count,
                lane_width,
                op,
            )
        )
        range_bytes = state.builder.constant(state.dsl.i32(), int(attrs["range_bytes"]))
        buffer_base = state.builder.make_buffer(
            source_base,
            range_bytes,
            result_type=state.dsl.buffer_ptr_type(element_type),
        )
        result_id = _single_result(op)
        token = _emit_buffer_load_to_local_packet_dma(
            state,
            op,
            attrs,
            op.operands[0],
            dest_base,
            buffer_base,
            offset_components,
            issue_dependencies,
            mask_components,
            element_type,
            lane_width,
        )
        state.values[result_id] = token
        _set_local_memory_access_token(state, op.operands[0], token, "async_write")
        _record_token_local_memory_roots(state, result_id, op.operands[0])
        return
    has_mask = bool(attrs.get("has_mask", False))
    mask_operand_count = int(has_mask)
    expected_operand_count = (
        3
        + mask_operand_count
        + issue_dependency_count
    )
    if len(op.operands) != expected_operand_count:
        fail(
            "TLXW_EMIT_OPERAND_COUNT",
            STAGE,
            f"expected {expected_operand_count} operands, got {len(op.operands)}",
            target_op_id=op.target_op_id,
        )
    dest_base = _require_value(state, op.operands[0], op)
    source_base = _require_value(state, op.operands[1], op)
    offsets = _require_value(state, op.operands[2], op)
    operand_index = 3
    masks = (
        _require_value(state, op.operands[operand_index], op)
        if has_mask else None
    )
    operand_index += mask_operand_count
    issue_dependencies = tuple(
        _require_value(state, operand_id, op)
        for operand_id in op.operands[operand_index:]
    )
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    lane_width = int(attrs["lane_width"])
    expected_components = int(attrs["component_count"])
    offset_components = _as_components(offsets)
    if len(offset_components) != expected_components:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "amdg.buffer_load_to_local offset component count does not "
            "match target op attrs",
            target_op_id=op.target_op_id,
        )
    load_offset_components = offset_components
    destination_offset_mode = attrs.get("destination_offset_mode", "affine")
    if destination_offset_mode == "affine":
        destination_offsets = tuple(int(value) for value in attrs["destination_component_offsets"])
        if len(destination_offsets) != expected_components:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "amdg.buffer_load_to_local destination component offsets do "
                "not match target op attrs",
                target_op_id=op.target_op_id,
            )
    elif destination_offset_mode == "layout_coordinates":
        destination_shape = tuple(int(value) for value in attrs["destination_coordinate_shape"])
        destination_component_bases = tuple(
            tuple(int(value) for value in bases) for bases in attrs["destination_component_coordinate_bases"])
        destination_workitem_coefficients = tuple(
            tuple(int(value)
                  for value in coefficients)
            for coefficients in attrs["destination_workitem_coordinate_coefficients"])
        if (len(destination_component_bases) != expected_components
                or any(len(bases) != len(destination_shape) for bases in destination_component_bases) or any(
                    len(coefficients) != len(destination_shape) for coefficients in destination_workitem_coefficients)):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "amdg.buffer_load_to_local coordinate destination offsets "
                "do not match target op attrs",
                target_op_id=op.target_op_id,
            )
    else:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "unsupported amdg.buffer_load_to_local destination offset mode "
            f"{destination_offset_mode}",
            target_op_id=op.target_op_id,
        )
    mask_components = None
    if masks is not None:
        mask_components = _as_mask_predicate_components(
            state,
            masks,
            expected_components,
            lane_width,
            op,
        )
    if attrs["mode"] == "dma_load_lds":
        range_bytes = state.builder.constant(
            state.dsl.i32(),
            int(attrs["range_bytes"]),
        )
        buffer_base = state.builder.make_buffer(
            source_base,
            range_bytes,
            result_type=state.dsl.buffer_ptr_type(element_type),
        )
        result_id = _single_result(op)
        token = _emit_buffer_load_to_local_dma(
            state,
            op,
            attrs,
            op.operands[0],
            dest_base,
            buffer_base,
            offset_components,
            destination_offsets,
            issue_dependencies,
            element_type,
            lane_width,
        )
        state.values[result_id] = token
        _set_local_memory_access_token(state, op.operands[0], token, "async_write")
        _record_token_local_memory_roots(state, result_id, op.operands[0])
        return
    if attrs["mode"] != "scalarized_load_store":
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"unsupported amdg.buffer_load_to_local mode {attrs['mode']}",
            target_op_id=op.target_op_id,
        )
    dependency = _local_memory_access_dependency_token(
        state,
        op.operands[0],
        "write",
        extra_tokens=issue_dependencies,
    )
    component_tokens = []
    workitem = state.builder.workitem_id(0, state.dsl.i32(), lane_width)
    lane_offset = None
    if destination_offset_mode == "affine":
        lane_offset = _local_destination_lane_offset(
            state,
            workitem,
            lane_width,
            int(attrs.get("destination_lane_stride_elements", 1)),
            int(attrs.get("destination_wave_stride_elements", 0)),
        )
    value_type = state.dsl.simd_type(element_type, lane_width)
    mask_mode = attrs.get("mask_mode", "exec_where" if has_mask else "none")
    source_base = _ptr_cast(
        state,
        source_base,
        state.dsl.ptr_type(
            element_type,
            state.dsl.global_address_space(),
        ),
    )
    dest_base = _ptr_cast(
        state,
        dest_base,
        state.dsl.ptr_type(
            element_type,
            state.dsl.shared_address_space(),
        ),
    )

    def component_destination_offset(component_index):
        if destination_offset_mode == "affine":
            dest_offset = lane_offset
            destination_base_offset = destination_offsets[component_index]
            if destination_base_offset:
                base_offset = state.builder.splat(
                    state.builder.constant(
                        state.dsl.i32(),
                        destination_base_offset,
                    ),
                    state.dsl.i32(),
                    lane_width,
                )
                dest_offset = state.builder.binary(
                    state.dsl.BinaryKind.AddI,
                    dest_offset,
                    base_offset,
                    nsw=_LAYOUT_MATH_NSW,
                )
            return dest_offset
        coords = tuple(
            _bit_linear_thread_coordinate(
                state,
                workitem,
                int(base),
                tuple(
                    coefficients[dim]
                    for coefficients in destination_workitem_coefficients
                ),
                lane_width,
            )
            for dim, base in enumerate(
                destination_component_bases[component_index]
            )
        )
        return _shared_destination_element_offset(
            state,
            attrs,
            coords,
            destination_shape,
            lane_width,
            op,
        )

    def emit_component_load_store(component_index, offset_component):
        loaded, _load_token = _emit_symbolic_contiguous_gather(
            state,
            _simd_offset_value(state, offset_component, lane_width),
            source_base,
            value_type,
            element_type,
            lane_width,
            int(attrs["element_byte_width"]),
            int(attrs["element_byte_width"]) * 8,
            dependency=dependency,
        )
        return _emit_symbolic_contiguous_scatter(
            state,
            loaded,
            component_destination_offset(component_index),
            dest_base,
            element_type,
            lane_width,
            int(attrs["element_byte_width"]),
            dependency=dependency,
        )

    if mask_components is None:
        component_tokens.extend(
            emit_component_load_store(index, offset_component)
            for index, offset_component in enumerate(load_offset_components)
        )
    else:
        if mask_mode != "exec_where":
            fail(
                "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC_MASK",
                STAGE,
                f"unsupported buffer_load_to_local mask mode {mask_mode}",
                target_op_id=op.target_op_id,
            )
        for mask_component, component_indices in _group_component_indices_by_identity(
            mask_components
        ):
            def emit_masked_components(component_indices=component_indices):
                return _join_memory_tokens(
                    state,
                    tuple(
                        emit_component_load_store(
                            index,
                            load_offset_components[index],
                        )
                        for index in component_indices
                    ),
                )

            component_tokens.append(
                _emit_masked_token_region(
                    state,
                    mask_component,
                    dependency,
                    emit_masked_components,
                )
            )
    token = _join_memory_tokens(state, component_tokens)
    result_id = _single_result(op)
    state.values[result_id] = token
    _set_local_memory_access_token(state, op.operands[0], token, "write")
    _record_token_local_memory_roots(state, result_id, op.operands[0])


def _local_destination_lane_offset(
    state,
    workitem,
    lane_width,
    lane_stride,
    wave_stride,
):
    lane_width = int(lane_width)
    lane_stride = int(lane_stride)
    wave_stride = int(wave_stride)
    if wave_stride == 0:
        if lane_stride == 1:
            return workitem
        return _simd_binary_const(
            state,
            "muli",
            workitem,
            lane_stride,
            lane_width,
            nsw=_LAYOUT_MATH_NSW,
        )
    lane = _simd_binary_const(state, "remui", workitem, lane_width, lane_width)
    if lane_stride != 1:
        lane = _simd_binary_const(
            state,
            "muli",
            lane,
            lane_stride,
            lane_width,
            nsw=_LAYOUT_MATH_NSW,
        )
    wave_first = state.builder.read_first(workitem)
    wave_id = _scalar_binary_const_i32(state, "divui", wave_first, lane_width)
    wave_offset = _scalar_binary_const_i32(
        state,
        "muli",
        wave_id,
        wave_stride,
        nsw=_LAYOUT_MATH_NSW,
    )
    wave_offset = state.builder.splat(wave_offset, state.dsl.i32(), lane_width)
    return state.builder.binary(
        state.dsl.BinaryKind.AddI,
        lane,
        wave_offset,
        nsw=_LAYOUT_MATH_NSW,
    )


def _shared_destination_element_offset(state, attrs, coords, shape, lane_width, op):
    plan = attrs.get("destination_physical_offset_plan")
    if plan is None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "scalarized shared destination is missing a physical offset plan",
            target_op_id=op.target_op_id,
        )
    unit = attrs.get("destination_physical_offset_unit")
    if unit != "element":
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"unsupported scalarized shared destination offset unit {unit}",
            target_op_id=op.target_op_id,
        )
    if int(attrs.get("destination_physical_element_byte_width", 0)) != int(attrs.get("element_byte_width", 0)):
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "scalarized shared destination offset element width does not match "
            "the op element width",
            target_op_id=op.target_op_id,
        )
    physical_shape = tuple(int(value) for value in attrs.get("destination_physical_shape", shape))
    logical_origin = tuple(
        int(value)
        for value in attrs.get("destination_logical_origin", (0, ) * len(shape))
    )
    if (len(physical_shape) != len(shape) or len(logical_origin) != len(shape)
            or len(coords) != len(shape)):
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "shared destination view ranks do not match its logical coordinates",
            target_op_id=op.target_op_id,
        )
    coords = tuple(
        coord if int(origin) == 0 else _simd_binary_const(
            state,
            "addi",
            coord,
            int(origin),
            lane_width,
            nsw=_LAYOUT_MATH_NSW,
        )
        for coord, origin in zip(coords, logical_origin)
    )
    shape = physical_shape
    if plan == "dense_row_major":
        return _linearize_coordinates(state, coords, shape, lane_width)
    if plan == "linear_shared":
        return _linear_inverse_offset_from_simd_coords(
            state,
            attrs,
            "destination",
            coords,
            lane_width,
            op,
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
        )
    if plan == "padded_linear":
        physical = _linear_component_offset_from_simd_coords(
            state,
            attrs,
            "destination",
            coords,
            lane_width,
            op,
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
        )
        encoded = physical
        intervals = tuple(int(value) for value in attrs.get("destination_physical_intervals", ()))
        paddings = tuple(int(value) for value in attrs.get("destination_physical_paddings", ()))
        for interval, padding in zip(intervals, paddings):
            term = _simd_binary_const(state, "divui", physical, interval, lane_width)
            if padding != 1:
                term = _simd_binary_const(
                    state,
                    "muli",
                    term,
                    padding,
                    lane_width,
                    nsw=_LAYOUT_MATH_NSW,
                )
            encoded = state.builder.binary(
                state.dsl.BinaryKind.AddI,
                encoded,
                term,
                nsw=_LAYOUT_MATH_NSW,
            )
        return encoded
    if plan == "swizzled_xor":
        order = tuple(int(value) for value in attrs["destination_physical_order"])
        minor_dim = int(order[0])
        major_dim = int(order[1])
        minor_extent = int(shape[minor_dim])
        vec = int(attrs["destination_physical_swizzled_vec"])
        per_phase = int(attrs["destination_physical_swizzled_per_phase"])
        max_phase = int(attrs["destination_physical_swizzled_max_phase"])
        major = coords[major_dim]
        minor = coords[minor_dim]
        phase = _simd_binary_const(state, "divui", major, per_phase, lane_width)
        phase = _simd_binary_const(state, "remui", phase, max_phase, lane_width)
        if vec * max_phase <= minor_extent:
            minor_group = _simd_binary_const(
                state,
                "divui",
                minor,
                vec,
                lane_width,
            )
            minor_inner = _simd_binary_const(
                state,
                "remui",
                minor,
                vec,
                lane_width,
            )
            swizzled_minor = state.builder.binary(
                state.dsl.BinaryKind.XOrI,
                minor_group,
                phase,
            )
            if vec != 1:
                swizzled_minor = _simd_binary_const(
                    state,
                    "muli",
                    swizzled_minor,
                    vec,
                    lane_width,
                    nsw=_LAYOUT_MATH_NSW,
                )
            swizzled_minor = state.builder.binary(
                state.dsl.BinaryKind.AddI,
                swizzled_minor,
                minor_inner,
                nsw=_LAYOUT_MATH_NSW,
            )
        else:
            phase_offset = phase
            if vec != 1:
                phase_offset = _simd_binary_const(
                    state,
                    "muli",
                    phase_offset,
                    vec,
                    lane_width,
                    nsw=_LAYOUT_MATH_NSW,
                )
            phase_offset = _simd_binary_const(
                state,
                "remui",
                phase_offset,
                minor_extent,
                lane_width,
            )
            swizzled_minor = state.builder.binary(
                state.dsl.BinaryKind.XOrI,
                minor,
                phase_offset,
            )
        physical_coords = list(coords)
        physical_coords[minor_dim] = swizzled_minor
        return _linearize_coordinates_with_order(
            state,
            physical_coords,
            shape,
            order,
            lane_width,
        )
    fail(
        "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
        STAGE,
        f"unsupported scalarized shared destination physical offset plan {plan}",
        target_op_id=op.target_op_id,
    )


def _emit_buffer_load_to_local_packet_dma(
    state,
    op,
    attrs,
    dest_base_target_id,
    dest_base,
    buffer_base,
    offset_components,
    issue_dependencies,
    mask_components,
    element_type,
    lane_width,
):
    packet_bytes = int(attrs["packet_bytes"])
    element_byte_width = int(attrs["element_byte_width"])
    component_count = int(attrs["component_count"])
    component_thread_count = int(attrs.get("component_thread_count", lane_width))
    destination_offsets = tuple(int(value) for value in attrs["destination_component_offsets"])
    if len(destination_offsets) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "amdg.buffer_load_to_local packet destination offsets do not "
            "match component count",
            target_op_id=op.target_op_id,
        )
    source_ptr_type = state.dsl.simd_ptr_type(
        element_type,
        state.dsl.buffer_address_space(),
        lane_width,
    )
    i32_shared = state.dsl.ptr_type(state.dsl.i32(), state.dsl.shared_address_space())
    dword_base = state.shared_pointer_dword_bases.get(dest_base_target_id)
    dest_base_offset = None
    if dword_base is not None:
        dest_base_i32 = _ptr_cast(state, dword_base.base, i32_shared)
        dest_base_offset = dword_base.dword_offset
    else:
        dest_base_i32 = _ptr_cast(state, dest_base, i32_shared)
    lane = state.builder.workitem_id(0, state.dsl.i32(), lane_width)
    destination_wave_offset_coefficients_dwords = tuple(
        int(value) for value in attrs.get("destination_wave_offset_coefficients_dwords", ()))
    destination_wave_stride_dwords = int(attrs.get("destination_wave_stride_dwords", 0))
    destination_wave_coordinate = _packet_destination_wave_coordinate_value(
        state,
        destination_wave_stride_dwords,
        destination_wave_offset_coefficients_dwords,
        lane,
        lane_width,
        component_thread_count,
        op,
    )
    dependency = _local_dma_issue_dependency_token(state, issue_dependencies)
    component_tokens = []
    if mask_components is not None and len(mask_components) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "amdg.buffer_load_to_local packet mask component count does "
            "not match packet component count",
            target_op_id=op.target_op_id,
        )
    if len(offset_components) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "packet DMA offset edge does not match component count",
            target_op_id=op.target_op_id,
        )
    for component, (source_offset, destination_offset) in enumerate(zip(
        offset_components,
        destination_offsets,
    )):
        issue_delay_options = _local_dma_component_issue_delay_options(
            state, op, component)
        source_offset = _simd_offset_value(state, source_offset, lane_width)
        source_ptr = state.builder.ptr_add(
            buffer_base,
            source_offset,
            result_type=source_ptr_type,
        )
        dest_offset = _packet_destination_offset_value(
            state,
            int(destination_offset),
            element_byte_width,
            destination_wave_coordinate,
            lane_width,
            destination_wave_stride_dwords,
            destination_wave_offset_coefficients_dwords,
            op,
        )
        if dword_base is not None:
            component_base = _shared_pointer_with_dword_offset(
                state,
                dword_base.base,
                dest_offset,
                cache_key=(
                    "packet_dma_component",
                    id(destination_wave_coordinate),
                    int(destination_offset),
                    int(element_byte_width),
                    int(destination_wave_stride_dwords),
                    tuple(destination_wave_offset_coefficients_dwords),
                ),
            )
            if dest_base_offset is None:
                dest_ptr = component_base
            else:
                dest_ptr = state.builder.ptr_add(
                    component_base,
                    dest_base_offset,
                    result_type=i32_shared,
                )
        else:
            dest_offset = _combine_optional_i32_offsets(
                state,
                dest_base_offset,
                dest_offset,
                nsw=_LAYOUT_MATH_NSW,
            )
            if dest_offset is None:
                dest_ptr = dest_base_i32
            else:
                dest_ptr = state.builder.ptr_add(
                    dest_base_i32,
                    dest_offset,
                    result_type=i32_shared,
                )
        mask_component = None if mask_components is None else mask_components[component]

        component_dependency = dependency
        if mask_component is None:
            token = state.builder.dma_load_lds(
                source_ptr,
                dest_ptr,
                after=component_dependency,
                bytes=packet_bytes,
                **issue_delay_options,
            )
            component_tokens.append(token)
            continue
        if _is_scalar_i1_value(state, mask_component):
            fail(
                "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC_MASK",
                STAGE,
                "masked packet DMA requires a SIMD wave mask",
                target_op_id=op.target_op_id,
            )
        if attrs.get("mask_mode") == "exec_where":
            inactive_token = component_dependency or state.builder.token()

            def emit_dma_load(
                source_ptr=source_ptr,
                dest_ptr=dest_ptr,
                component_dependency=component_dependency,
                issue_delay_options=issue_delay_options,
            ):
                return state.builder.dma_load_lds(
                    source_ptr,
                    dest_ptr,
                    after=component_dependency,
                    bytes=packet_bytes,
                    **issue_delay_options,
                )

            token = _emit_masked_token_region(
                state,
                mask_component,
                inactive_token,
                emit_dma_load,
            )
            component_tokens.append(token)
            continue
        inactive_offset = _buffer_inactive_element_offset(state, attrs, lane_width)
        inactive_ptr = state.builder.ptr_add(
            buffer_base,
            inactive_offset,
            result_type=source_ptr_type,
        )
        selected_source = state.builder.select(mask_component, source_ptr, inactive_ptr)
        token = state.builder.dma_load_lds(
            selected_source,
            dest_ptr,
            after=component_dependency,
            bytes=packet_bytes,
            zero_fill_inactive=True,
            **issue_delay_options,
        )
        component_tokens.append(token)
    return _join_memory_tokens(state, component_tokens)


def _emit_buffer_load_to_local_dma(
    state,
    op,
    attrs,
    dest_base_target_id,
    dest_base,
    buffer_base,
    offset_components,
    destination_offsets,
    issue_dependencies,
    element_type,
    lane_width,
):
    packet_bytes = int(attrs["packet_bytes"])
    if packet_bytes <= 0:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local DMA requires a positive packet byte width",
            target_op_id=op.target_op_id,
        )
    dependency = _local_dma_issue_dependency_token(state, issue_dependencies)
    component_tokens = []
    source_ptr_type = state.dsl.simd_ptr_type(
        element_type,
        state.dsl.buffer_address_space(),
        lane_width,
    )
    i32_shared = state.dsl.ptr_type(state.dsl.i32(), state.dsl.shared_address_space())
    dword_base = state.shared_pointer_dword_bases.get(dest_base_target_id)
    dest_base_offset = None
    if dword_base is not None:
        dest_base_i32 = _ptr_cast(state, dword_base.base, i32_shared)
        dest_base_offset = dword_base.dword_offset
    else:
        dest_base_i32 = _ptr_cast(state, dest_base, i32_shared)
    element_byte_width = int(attrs["element_byte_width"])
    for component, (offset_component, destination_base_offset) in enumerate(zip(
            offset_components,
            destination_offsets,
    )):
        offset_component = _simd_offset_value(state, offset_component, lane_width)
        source_ptr = state.builder.ptr_add(
            buffer_base,
            offset_component,
            result_type=source_ptr_type,
        )
        destination_byte_offset = int(destination_base_offset) * element_byte_width
        if destination_byte_offset % 4:
            fail(
                "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "amdg.buffer_load_to_local DMA destination offset must be dword aligned",
                target_op_id=op.target_op_id,
            )
        dest_offset = dest_base_offset
        if destination_byte_offset:
            destination_dword_offset = state.builder.constant(
                state.dsl.i32(),
                destination_byte_offset // 4,
            )
            dest_offset = _combine_optional_i32_offsets(
                state,
                dest_offset,
                destination_dword_offset,
                nsw=_LAYOUT_MATH_NSW,
            )
        if dword_base is not None:
            component_offset = (
                None
                if not destination_byte_offset
                else destination_byte_offset // 4
            )
            component_base = _shared_pointer_with_dword_offset(
                state,
                dword_base.base,
                component_offset,
                cache_key=(
                    "dma_component",
                    int(destination_base_offset),
                    int(element_byte_width),
                ),
            )
            if dest_base_offset is None:
                dest_ptr = component_base
            else:
                dest_ptr = state.builder.ptr_add(
                    component_base,
                    dest_base_offset,
                    result_type=i32_shared,
                )
        elif dest_offset is None:
            dest_ptr = dest_base_i32
        else:
            dest_ptr = state.builder.ptr_add(
                dest_base_i32,
                dest_offset,
                result_type=i32_shared,
            )
        component_dependency = dependency
        token = state.builder.dma_load_lds(
            source_ptr,
            dest_ptr,
            after=component_dependency,
            bytes=packet_bytes,
            **_local_dma_component_issue_delay_options(state, op, component),
        )
        component_tokens.append(token)
    return _join_memory_tokens(state, component_tokens)


def _emit_async_commit_group(state, op):
    tokens = tuple(_require_value(state, target_value_id, op) for target_value_id in op.operands)
    if tokens:
        token = state.builder.join(*tokens)
    else:
        token = state.builder.token()
    if op.results:
        result_id = _single_result(op)
        state.values[result_id] = token
        roots = _local_memory_roots_for_token_values(state, op.operands)
        if roots:
            _set_local_memory_roots_committed_token(state, roots, token)
            state.token_local_memory_root_sets[int(result_id)] = roots


def _emit_token(state, op):
    state.values[_single_result(op)] = state.builder.token()


def _emit_token_join(state, op):
    tokens = tuple(
        _require_value(state, target_value_id, op)
        for target_value_id in op.operands
    )
    state.values[_single_result(op)] = _join_memory_tokens(state, tokens)


def _emit_issue_token(state, op):
    tokens = tuple(
        _require_value(state, target_value_id, op)
        for target_value_id in op.operands
    )
    state.values[_single_result(op)] = state.builder.issue_token(*tokens)


def _join_memory_tokens(state, tokens):
    tokens = _unique_tokens(tokens)
    if not tokens:
        return state.builder.token()
    if len(tokens) == 1:
        return tokens[0]
    return state.builder.join(*tokens)


def _memory_dependency_token(state, tokens):
    tokens = _unique_tokens(tokens)
    if not tokens:
        return state.builder.token()
    return _join_memory_tokens(state, tokens)


def _scratch_write_dependency(state, extra_tokens=()):
    extra_tokens = tuple(extra_tokens)
    dependency = state.scratch_token
    if state.local_memory_pending_accesses:
        dependencies = (dependency, ) if dependency is not None else ()
        dependency = _sync_pending_local_memory_accesses(
            state,
            (*dependencies, *extra_tokens),
        )
        state.scratch_token = dependency
        state.scratch_token_needs_write_barrier = False
    elif ((dependency is not None and state.scratch_token_needs_write_barrier)
          or extra_tokens):
        dependencies = (dependency, ) if dependency is not None else ()
        dependency = _emit_cta_barrier(
            state,
            *_unique_tokens((*dependencies, *extra_tokens)),
        )
        state.scratch_token = dependency
        state.scratch_token_needs_write_barrier = False
    return dependency


def _local_memory_value_descendants(state, root):
    root = int(root)
    cached = state.local_memory_descendants.get(root)
    if cached is not None:
        return cached
    descendants = {root}
    changed = True
    while changed:
        changed = False
        for op in state.target_program.ops:
            sources = set(int(operand) for operand in op.operands)
            for region_id in op.region_ids:
                sources.update(
                    int(value_id)
                    for value_id in state.target_program.regions[int(region_id)].yield_value_ids
                )
            if descendants.isdisjoint(sources):
                continue
            for result_id in op.results:
                result_id = int(result_id)
                if (state.target_program.values[result_id].type.representation == "memdesc"
                        and result_id not in descendants):
                    descendants.add(result_id)
                    changed = True
    result = frozenset(descendants)
    state.local_memory_descendants[root] = result
    return result


def _local_memory_allocation_has_future_use(state, root, current_op):
    current_source_index = current_op.source_op_index
    if current_source_index is None:
        return True
    descendants = _local_memory_value_descendants(state, root)
    for candidate in state.target_program.ops:
        candidate_source_index = candidate.source_op_index
        if candidate_source_index is None:
            continue
        is_future = (
            int(candidate_source_index) > int(current_source_index)
            or (
                int(candidate_source_index) == int(current_source_index)
                and int(candidate.target_op_id) > int(current_op.target_op_id)
            )
        )
        if is_future and not descendants.isdisjoint(int(operand) for operand in candidate.operands):
            return True
    return False


def _release_dead_local_memory_before_redistribute(state, op):
    # Releasing storage from a loop body would be unsound on the next
    # iteration. Restrict bridge-driven lifetime ends to top-level operations;
    # the source/target dataflow check below remains conservative about every
    # later memdesc alias and use.
    if int(op.target_op_id) not in state.target_program.regions[0].op_ids:
        return ()
    released_tokens = []
    for root, allocation in tuple(state.local_memory_allocations.items()):
        root = int(root)
        if root in state.released_local_memory_allocations:
            continue
        if root in state.local_memory_release_unsafe_roots:
            continue
        if _local_memory_allocation_has_future_use(state, root, op):
            continue
        # Completed async groups and earlier disjoint aliases may already have
        # left the pending set. Join every surviving token for this allocation
        # into the lifetime-end proof. The caller puts one collective barrier
        # after all releases, so storage retirement and scratch publication do
        # not require separate hardware barriers.
        release_dependency = _local_memory_allocation_access_dependency(state, root)
        released_tokens.append(
            state.builder.release_alloc(allocation, after=release_dependency)
        )
        state.released_local_memory_allocations.add(root)
    return tuple(released_tokens)


def _record_scratch_read_tokens(state, tokens):
    tokens = _unique_tokens(tokens)
    if not tokens:
        state.scratch_token = state.builder.token()
        state.scratch_token_needs_write_barrier = False
        return
    state.scratch_token = _join_memory_tokens(state, tokens)
    state.scratch_token_needs_write_barrier = True


def _unique_tokens(tokens):
    unique = []
    seen = set()
    for token in tokens:
        key = id(token)
        if key in seen:
            continue
        seen.add(key)
        unique.append(token)
    return tuple(unique)


def _emit_async_wait(state, op):
    attrs = target_ir.attrs_dict(op)
    completed_count = int(attrs["completed_group_dependency_count"])
    retained_issue_count = int(attrs["retained_issue_dependency_count"])
    release_count = int(attrs["lds_release_dependency_count"])
    if (
        completed_count < 0
        or retained_issue_count < 0
        or release_count < 0
        or completed_count + retained_issue_count + release_count
        != len(op.operands)
    ):
        fail(
            "TLXW_EMIT_ASYNC_WAIT_OPERANDS",
            STAGE,
            "async_wait dependency counts do not match its target operands",
            target_op_id=op.target_op_id,
        )
    completed_target_ids = op.operands[:completed_count]
    issue_end = completed_count + retained_issue_count
    retained_issue_target_ids = op.operands[completed_count:issue_end]
    release_target_ids = op.operands[issue_end:]
    completed_tokens = tuple(
        _require_value(state, target_value_id, op)
        for target_value_id in completed_target_ids
    )
    retained_issue_tokens = tuple(
        _require_value(state, target_value_id, op)
        for target_value_id in retained_issue_target_ids
    )
    release_tokens = tuple(
        _require_value(state, target_value_id, op)
        for target_value_id in release_target_ids
    )
    dependencies = (
        *completed_tokens,
        *retained_issue_tokens,
        *release_tokens,
    )
    publication_mode = attrs["publication_mode"]
    if publication_mode == "workgroup":
        token = _emit_cta_barrier(state, *dependencies)
    elif publication_mode == "wave_local":
        token = (
            state.builder.after(*dependencies)
            if dependencies else state.builder.token()
        )
    else:
        fail(
            "TLXW_EMIT_ASYNC_WAIT_MODE",
            STAGE,
            f"unsupported async wait publication mode {publication_mode}",
            target_op_id=op.target_op_id,
        )
    roots = _local_memory_roots_for_token_values(
        state,
        completed_target_ids,
    )
    if op.results:
        result_id = _single_result(op)
        state.values[result_id] = token
        if roots:
            state.token_local_memory_root_sets[int(result_id)] = roots


def _emit_local_load_mma_payload(state, op):
    attrs = target_ir.attrs_dict(op)
    base, *explicit_dependencies = _operand_values(state, op, len(op.operands))
    memdesc_target_id = op.operands[0]
    data_result_ids, _completion_result_ids = _local_access_result_ids(op)
    if len(data_result_ids) != 1:
        fail(
            "TLXW_EMIT_LOCAL_ACCESS_RESULTS",
            STAGE,
            "local_load_mma_payload requires one data result",
            target_op_id=op.target_op_id,
        )
    data_result_id = data_result_ids[0]
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    base = _ptr_cast(
        state,
        base,
        state.dsl.ptr_type(
            element_type,
            state.dsl.shared_address_space(),
        ),
    )
    lane_width = int(attrs["lane_width"])
    registers = int(attrs["registers"])
    expected_components = int(attrs["component_count"])
    warps_per_cta = tuple(int(value) for value in attrs.get("warps_per_cta", (1, 1)))
    wave_tile_axis = attrs.get("wave_tile_axis", "none")
    wave_tile_stride_dwords = int(attrs.get("wave_tile_stride_dwords", 0))
    load_mode = attrs.get("load_mode", "mma_payload_load")
    offset_attr = ("component_dword_offsets" if load_mode == "mma_payload_load" else "component_tile_offsets")
    if len(attrs[offset_attr]) != expected_components:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "local_load_mma_payload component offsets do not match attrs",
            target_op_id=op.target_op_id,
        )
    wi = state.builder.workitem_id(0, state.dsl.i32(), lane_width)
    dependency = _local_memory_access_dependency_token(
        state,
        memdesc_target_id,
        "read",
        extra_tokens=tuple(explicit_dependencies),
        # See _emit_local_load: the wait replaces only async-DMA destination
        # inference.  Synchronous LDS hazards remain structural dependencies.
        ignore_async_writes=bool(attrs.get("synced_via_async_wait", False)),
        ready_async_write_roots=_local_load_ready_async_write_roots(
            state,
            op,
            attrs,
        ),
    )
    # MMA payload values carry the data dependency into MFMA.  The read token is
    # kept only as a lightweight anti-dependency for later writes that reuse the
    # same LDS slot; async_wait does not collect it into an unrelated barrier.
    if load_mode == "transpose_mma_payload_load":
        value, read_token = _emit_transpose_mma_payload_load(
            state,
            op,
            attrs,
            base,
            element_type,
            dependency,
        )
        state.values[data_result_id] = value
        _finish_local_access(
            state, op, memdesc_target_id, read_token, "mma_read"
        )
        return
    if load_mode in {"swizzled_mma_payload_load", "indexed_mma_payload_load"}:
        value, read_token = _emit_swizzled_mma_payload_load(
            state,
            op,
            attrs,
            base,
            element_type,
            wi,
            dependency,
        )
        state.values[data_result_id] = value
        _finish_local_access(
            state, op, memdesc_target_id, read_token, "mma_read"
        )
        return
    if load_mode != "mma_payload_load":
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"unsupported local_load_mma_payload mode {load_mode}",
            target_op_id=op.target_op_id,
        )
    component_offsets = tuple(int(value) for value in attrs["component_dword_offsets"])
    element_byte_width = _element_byte_width(attrs["element_type"], op)
    elements_per_register = 4 // int(element_byte_width)
    load_type = state.dsl.simd_type(
        state.dsl.vector_type(registers * elements_per_register, element_type),
        width=lane_width,
    )
    payloads = []
    load_tokens = []
    for component_offset in component_offsets:
        offset = _linear_local_fragment_index_offset(
            state,
            wi,
            lane_width,
            elements_per_lane=registers,
            wave_tile_axis=wave_tile_axis,
            warps_per_cta=warps_per_cta,
            wave_tile_stride=wave_tile_stride_dwords,
            extra_elements=component_offset,
            op=op,
        )
        if elements_per_register != 1:
            offset = _simd_binary_const(
                state,
                "muli",
                offset,
                elements_per_register,
                lane_width,
                nsw=_LAYOUT_MATH_NSW,
            )
        payload, token = _emit_symbolic_contiguous_gather(
            state,
            offset,
            base,
            load_type,
            element_type,
            lane_width,
            element_byte_width,
            element_byte_width * 8,
            dependency=dependency,
        )
        payloads.append(payload)
        load_tokens.append(token)
    state.values[data_result_id] = _pack_components(tuple(payloads))
    _finish_local_access(
        state,
        op,
        memdesc_target_id,
        _join_memory_tokens(state, load_tokens),
        "mma_read",
    )


def _emit_transpose_mma_payload_load(
    state,
    op,
    attrs,
    base,
    element_type,
    dependency,
):
    lane_width = int(attrs["lane_width"])
    base_type = state.dsl.ptr_type(element_type, state.dsl.shared_address_space())
    base = _ptr_cast(state, base, base_type)
    load_type = state.dsl.simd_type(
        state.dsl.vector_type(int(attrs["chunk_elements"]), element_type),
        width=lane_width,
    )
    component_type = state.dsl.simd_type(element_type, lane_width)
    chunk_element_deltas = attrs.get("chunk_element_deltas")
    item = state.dsl.sym("item")
    slot = state.dsl.sym("slot")
    source_symbol = state.dsl.sym("source")
    source_item = _b16_transpose_source_item_expr(state, item, slot)
    payloads = []
    load_tokens = []
    for component_index, tile_offsets in enumerate(attrs["component_tile_offsets"]):
        components = []
        component_deltas = None
        if chunk_element_deltas is not None:
            component_deltas = tuple(int(value) for value in chunk_element_deltas[component_index])
        for chunk in range(int(attrs["chunks_per_component"])):
            logical_extra_elements = (0 if component_deltas is not None else int(attrs["chunk_elements"]) * chunk)
            physical_extra_elements = (int(component_deltas[chunk]) if component_deltas is not None else 0)
            source_offset = _local_fragment_element_offset_expr(
                state,
                attrs,
                source_symbol,
                tuple(int(value) for value in tile_offsets),
                logical_extra_elements,
                lane_width,
                elements_per_lane=int(attrs["elements_per_lane"]),
                wave_tile_axis=attrs.get("wave_tile_axis", "none"),
                warps_per_cta=tuple(int(value) for value in attrs.get("warps_per_cta", (1, 1))),
                wave_tile_stride=int(attrs.get("wave_tile_stride_elements", 0)),
                op=op,
                physical_extra_elements=physical_extra_elements,
            )
            source_offset = source_offset.simplify().subs(
                source_symbol,
                source_item,
            ).simplify()
            loaded, token = _emit_b16_transpose_gather(
                state,
                base,
                load_type,
                source_offset,
                item,
                dependency,
            )
            load_tokens.append(token)
            for component in range(int(attrs["chunk_elements"])):
                components.append(state.dsl.wave.ExtractOp(
                    component_type,
                    loaded,
                    int(component),
                ).result)
        packed_type = state.dsl.simd_type(
            state.dsl.vector_type(len(components), element_type),
            width=lane_width,
        )
        packed = state.dsl.wave.PackOp(packed_type, components).result
        payloads.append(packed)
    return _pack_components(tuple(payloads)), _join_memory_tokens(state, load_tokens)


def _emit_swizzled_mma_payload_load(
    state,
    op,
    attrs,
    base,
    element_type,
    wi,
    dependency,
):
    lane_width = int(attrs["lane_width"])
    base_type = state.dsl.ptr_type(element_type, state.dsl.shared_address_space())
    base = _ptr_cast(state, base, base_type)
    ptr_type = state.dsl.simd_ptr_type(
        element_type,
        state.dsl.shared_address_space(),
        lane_width,
    )
    load_type = state.dsl.simd_type(
        state.dsl.vector_type(int(attrs["elements_per_lane"]), element_type),
        width=lane_width,
    )
    payloads = []
    load_tokens = []
    component_physical_deltas = attrs.get("component_physical_element_deltas")
    if component_physical_deltas is not None:
        component_physical_deltas = tuple(
            int(value) for value in component_physical_deltas
        )
        if len(component_physical_deltas) != len(attrs["component_tile_offsets"]):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "local_load physical component deltas do not match component tiles",
                target_op_id=op.target_op_id,
            )
        physical_base_attrs = dict(attrs)
        physical_base_attrs["memdesc_logical_origin"] = tuple(
            int(value) for value in attrs["physical_base_logical_origin"]
        )
        physical_base_tile_offsets = tuple(
            int(value) for value in attrs["physical_base_tile_offsets"]
        )
        shared_offset = _local_fragment_element_offset(
            state,
            physical_base_attrs,
            wi,
            physical_base_tile_offsets,
            0,
            lane_width,
            elements_per_lane=int(attrs["elements_per_lane"]),
            wave_tile_axis=attrs.get("wave_tile_axis", "none"),
            warps_per_cta=tuple(int(value) for value in attrs.get("warps_per_cta", (1, 1))),
            wave_tile_stride=int(attrs.get("wave_tile_stride_elements", 0)),
            op=op,
        )
        component_accesses = tuple(
            (shared_offset, int(delta)) for delta in component_physical_deltas
        )
    else:
        component_accesses = tuple(
            (
                _local_fragment_element_offset(
                    state,
                    attrs,
                    wi,
                    tuple(int(value) for value in tile_offsets),
                    0,
                    lane_width,
                    elements_per_lane=int(attrs["elements_per_lane"]),
                    wave_tile_axis=attrs.get("wave_tile_axis", "none"),
                    warps_per_cta=tuple(
                        int(value) for value in attrs.get("warps_per_cta", (1, 1))
                    ),
                    wave_tile_stride=int(attrs.get("wave_tile_stride_elements", 0)),
                    op=op,
                ),
                0,
            )
            for tile_offsets in attrs["component_tile_offsets"]
        )
    for offset, physical_element_delta in component_accesses:
        if physical_element_delta:
            offset = _simd_binary_const(
                state,
                "addi",
                offset,
                int(physical_element_delta),
                lane_width,
                nsw=_LAYOUT_MATH_NSW,
            )
        ptr = state.builder.ptr_add(base, offset, result_type=ptr_type)
        payload, token = state.builder.load(ptr, load_type, after=dependency)
        payloads.append(payload)
        load_tokens.append(token)
    return _pack_components(tuple(payloads)), _join_memory_tokens(state, load_tokens)


def _local_fragment_element_offset(
    state,
    attrs,
    wi,
    tile_offsets,
    extra_elements,
    lane_width,
    *,
    elements_per_lane,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride,
    op,
    elements_per_offset_unit=1,
    physical_extra_elements=0,
):
    wi_sym = state.dsl.sym("wi")
    encoded = _local_fragment_element_offset_expr(
        state,
        attrs,
        wi_sym,
        tile_offsets,
        extra_elements,
        lane_width,
        elements_per_lane=elements_per_lane,
        wave_tile_axis=wave_tile_axis,
        warps_per_cta=warps_per_cta,
        wave_tile_stride=wave_tile_stride,
        op=op,
        elements_per_offset_unit=elements_per_offset_unit,
        physical_extra_elements=physical_extra_elements,
    )
    return state.builder.index_expr(encoded, bindings={wi_sym: wi})


def _local_fragment_element_offset_expr(
    state,
    attrs,
    workitem,
    tile_offsets,
    extra_elements,
    lane_width,
    *,
    elements_per_lane,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride,
    op,
    elements_per_offset_unit=1,
    physical_extra_elements=0,
):
    plan = attrs.get("shared_physical_offset_plan")
    if plan is None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "local_load shared access is missing a physical offset plan",
            target_op_id=op.target_op_id,
        )
    unit = attrs.get("shared_physical_offset_unit")
    if unit != "element":
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"unsupported local_load shared offset unit {unit}",
            target_op_id=op.target_op_id,
        )
    lane_layout = attrs.get("mma_access_lane_layout", "row_major_linear")
    source_shape = tuple(int(dim) for dim in attrs["source_shape"])
    memdesc_shape = tuple(int(dim) for dim in attrs.get("memdesc_shape", attrs["source_shape"]))
    if (lane_layout == "row_major_linear" and plan in {"dense_row_major", "swizzled_xor"}
            and source_shape == memdesc_shape):
        tile_base = _dense_tile_base_elements(
            memdesc_shape,
            tile_offsets,
        )
        logical = _linear_local_fragment_offset_expr(
            state,
            lane_width,
            elements_per_lane=elements_per_lane,
            wave_tile_axis=wave_tile_axis,
            warps_per_cta=warps_per_cta,
            wave_tile_stride=wave_tile_stride,
            extra_elements=int(tile_base) + int(extra_elements),
            op=op,
            workitem=workitem,
        )
    else:
        logical = _local_fragment_logical_offset_expr(
            state,
            attrs,
            workitem,
            tile_offsets,
            int(extra_elements),
            lane_width,
            elements_per_lane=elements_per_lane,
            wave_tile_axis=wave_tile_axis,
            warps_per_cta=warps_per_cta,
            wave_tile_stride=wave_tile_stride,
            op=op,
        )
    physical_shape = tuple(
        int(dim)
        for dim in attrs.get("memdesc_physical_shape", memdesc_shape)
    )
    logical_origin = tuple(
        int(value)
        for value in attrs.get("memdesc_logical_origin", (0, ) * len(memdesc_shape))
    )
    if (len(physical_shape) != len(memdesc_shape)
            or len(logical_origin) != len(memdesc_shape)):
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "local_load memdesc view ranks do not match",
            target_op_id=op.target_op_id,
        )
    if physical_shape != memdesc_shape or any(logical_origin):
        logical_coords = _delinearize_local_fragment_expr(
            state,
            logical,
            memdesc_shape,
        )
        physical_coords = tuple(
            coord + int(origin)
            for coord, origin in zip(logical_coords, logical_origin)
        )
        logical = _linearize_local_fragment_coords(
            physical_shape,
            physical_coords,
        )
    if plan == "dense_row_major":
        encoded = logical
    elif plan == "swizzled_xor":
        encoded = _swizzled_element_offset_expr(state, attrs, logical, op)
    elif plan == "linear_shared":
        encoded = _linear_shared_element_offset_expr(state, attrs, logical, op)
    elif plan == "padded_linear":
        encoded = _padded_element_offset_expr(state, attrs, logical, op)
    else:
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"unsupported local_load shared physical offset plan {plan}",
            target_op_id=op.target_op_id,
        )
    if int(elements_per_offset_unit) != 1:
        encoded = state.dsl.floor(encoded / int(elements_per_offset_unit))
    if int(physical_extra_elements):
        encoded += int(physical_extra_elements)
    return encoded


def _local_fragment_logical_offset_expr(
    state,
    attrs,
    workitem,
    tile_offsets,
    extra_elements,
    lane_width,
    *,
    elements_per_lane,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride,
    op,
):
    lane = state.dsl.mod(workitem, int(lane_width))
    source_shape = tuple(int(dim) for dim in attrs["source_shape"])
    memdesc_shape = tuple(int(dim) for dim in attrs.get("memdesc_shape", attrs["source_shape"]))
    lane_layout = attrs.get("mma_access_lane_layout", "row_major_linear")
    source_coords = _local_fragment_source_coords_expr(
        state,
        lane,
        source_shape,
        int(elements_per_lane),
        int(extra_elements),
        lane_layout,
        op,
    )
    coords = tuple(int(tile_offsets[dim]) + source_coords[dim] for dim in range(len(source_shape)))
    logical = _linearize_local_fragment_coords(memdesc_shape, coords)
    wave_tile = _wave_tile_offset_expr(
        state,
        workitem,
        lane_width,
        wave_tile_axis,
        warps_per_cta,
        wave_tile_stride,
        op,
    )
    if wave_tile is not None:
        logical += wave_tile
    return logical


def _local_fragment_source_coords_expr(
    state,
    lane,
    source_shape,
    elements_per_lane,
    extra_elements,
    lane_layout,
    op,
):
    if lane_layout == "row_major_linear":
        local = lane * int(elements_per_lane)
        if int(extra_elements):
            local += int(extra_elements)
        return _delinearize_local_fragment_expr(state, local, source_shape)
    if lane_layout == "gfx950_mfma_a":
        if len(source_shape) != 2:
            fail(
                "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "gfx950 MFMA A fragment load requires a rank-2 source shape",
                target_op_id=op.target_op_id,
            )
        row = state.dsl.mod(lane, int(source_shape[0]))
        col = state.dsl.floor(lane / int(source_shape[0])) * int(elements_per_lane)
        if int(extra_elements):
            col += int(extra_elements)
        return (row, col)
    if lane_layout == "gfx950_mfma_b":
        if len(source_shape) != 2:
            fail(
                "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "gfx950 MFMA B fragment load requires a rank-2 source shape",
                target_op_id=op.target_op_id,
            )
        non_k_dim = int(source_shape[1])
        col = state.dsl.mod(lane, non_k_dim)
        row = state.dsl.floor(lane / non_k_dim) * int(elements_per_lane)
        if int(extra_elements):
            row += int(extra_elements)
        return (row, col)
    if lane_layout == "gfx950_mfma_b_transpose":
        if len(source_shape) != 2:
            fail(
                "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "gfx950 MFMA B transpose load requires a rank-2 source shape",
                target_op_id=op.target_op_id,
            )
        non_k_dim = int(source_shape[1])
        if non_k_dim % 16:
            fail(
                "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "gfx950 MFMA B transpose load requires the non-K dimension "
                "to be a multiple of the ds_read_tr group width",
                target_op_id=op.target_op_id,
            )
        lane_in_group = state.dsl.mod(lane, 16)
        non_k_group = state.dsl.floor(state.dsl.mod(lane, non_k_dim) / 16)
        k_group = state.dsl.floor(lane / non_k_dim)
        chunk_k = (int(extra_elements) // 4) * 4
        packet_col = int(extra_elements) % 4
        row = (k_group * int(elements_per_lane) + chunk_k + state.dsl.floor(lane_in_group / 4))
        col = non_k_group * 16 + 4 * state.dsl.mod(lane_in_group, 4) + packet_col
        return (row, col)
    fail(
        "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported MMA access lane layout {lane_layout}",
        target_op_id=op.target_op_id,
    )


def _delinearize_local_fragment_expr(state, linear, shape):
    coords = []
    remainder = linear
    for dim, extent in enumerate(shape):
        stride = _product(shape[dim + 1:])
        if stride == 1:
            coord = state.dsl.mod(remainder, int(extent))
        else:
            coord = state.dsl.floor(remainder / int(stride))
            remainder = state.dsl.mod(remainder, int(stride))
        coords.append(coord)
    return tuple(coords)


def _linearize_local_fragment_coords(shape, coords):
    result = 0
    stride = 1
    for dim in reversed(range(len(shape))):
        result += coords[dim] * stride
        stride *= int(shape[dim])
    return result


def _linearize_local_fragment_coords_with_order(shape, coords, order):
    result = 0
    stride = 1
    for dim in order:
        result += coords[int(dim)] * stride
        stride *= int(shape[int(dim)])
    return result


def _linear_local_fragment_index_offset(
    state,
    wi,
    lane_width,
    *,
    elements_per_lane,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride,
    extra_elements,
    op,
):
    expr = _linear_local_fragment_offset_expr(
        state,
        lane_width,
        elements_per_lane=elements_per_lane,
        wave_tile_axis=wave_tile_axis,
        warps_per_cta=warps_per_cta,
        wave_tile_stride=wave_tile_stride,
        extra_elements=extra_elements,
        op=op,
    )
    wi_sym = state.dsl.sym("wi")
    return state.builder.index_expr(expr, bindings={wi_sym: wi})


def _linear_local_fragment_offset_expr(
    state,
    lane_width,
    *,
    elements_per_lane,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride,
    extra_elements,
    op,
    workitem=None,
):
    wi = state.dsl.sym("wi") if workitem is None else workitem
    lane = state.dsl.mod(wi, int(lane_width))
    expr = lane * int(elements_per_lane)
    wave_tile = _wave_tile_offset_expr(
        state,
        wi,
        lane_width,
        wave_tile_axis,
        warps_per_cta,
        wave_tile_stride,
        op,
    )
    if wave_tile is not None:
        expr += wave_tile
    if int(extra_elements):
        expr += int(extra_elements)
    return expr


def _wave_tile_offset_expr(
    state,
    wi,
    lane_width,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride,
    op,
):
    if wave_tile_axis == "none" or not int(wave_tile_stride):
        return None
    if len(warps_per_cta) < 2 or int(warps_per_cta[1]) <= 0:
        fail(
            "TLXW_EMIT_LOCAL_LOAD_TILE_MAP",
            STAGE,
            "local_load_mma_payload requires a valid warps_per_cta mapping",
            target_op_id=op.target_op_id,
        )
    wave_id = state.dsl.floor(wi / int(lane_width))
    if wave_tile_axis == "m":
        wave_coord = state.dsl.floor(wave_id / int(warps_per_cta[1]))
    elif wave_tile_axis == "n":
        wave_coord = state.dsl.mod(wave_id, int(warps_per_cta[1]))
    else:
        fail(
            "TLXW_EMIT_LOCAL_LOAD_TILE_MAP",
            STAGE,
            f"unsupported local_load_mma_payload wave axis {wave_tile_axis}",
            target_op_id=op.target_op_id,
        )
    return wave_coord * int(wave_tile_stride)


def _swizzled_element_offset_expr(state, attrs, logical, op):
    shape = tuple(
        int(dim)
        for dim in attrs.get(
            "memdesc_physical_shape",
            attrs.get("memdesc_shape", attrs["source_shape"]),
        )
    )
    order = _physical_order_from_attrs(
        attrs,
        "shared_physical_order",
        shape,
        op,
        "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
    )
    coords = _delinearize_local_fragment_expr(state, logical, shape)
    minor_dim = int(order[0])
    major_dim = int(order[1])
    minor_extent = int(shape[minor_dim])
    major = coords[major_dim]
    minor = coords[minor_dim]
    vec = int(attrs["shared_physical_swizzled_vec"])
    row_phase = state.dsl.floor(major / int(attrs["shared_physical_swizzled_per_phase"]))
    phase = state.dsl.mod(
        row_phase,
        int(attrs["shared_physical_swizzled_max_phase"]),
    )
    max_phase = int(attrs["shared_physical_swizzled_max_phase"])
    if vec * max_phase <= minor_extent:
        col_group = state.dsl.floor(minor / vec)
        swizzled_group = state.dsl.xor(col_group, phase)
        swizzled_minor = swizzled_group * vec + state.dsl.mod(minor, vec)
    else:
        phase_offset = state.dsl.mod(phase * vec, minor_extent)
        swizzled_minor = state.dsl.xor(minor, phase_offset)
    physical_coords = list(coords)
    physical_coords[minor_dim] = swizzled_minor
    return _linearize_local_fragment_coords_with_order(
        shape,
        physical_coords,
        order,
    )


def _linear_shared_element_offset_expr(state, attrs, logical, op):
    shape = tuple(
        int(dim)
        for dim in attrs.get(
            "memdesc_physical_shape",
            attrs.get("memdesc_shape", attrs["source_shape"]),
        )
    )
    coords = _delinearize_local_fragment_expr(state, logical, shape)
    return _linear_inverse_offset_from_expr_coords(
        state,
        attrs,
        "shared",
        coords,
        op,
        "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
    )


def _padded_element_offset_expr(state, attrs, logical, op):
    shape = tuple(
        int(dim)
        for dim in attrs.get(
            "memdesc_physical_shape",
            attrs.get("memdesc_shape", attrs["source_shape"]),
        )
    )
    coords = _delinearize_local_fragment_expr(state, logical, shape)
    physical = _linear_component_offset_from_expr_coords(
        state,
        attrs,
        "shared",
        coords,
        op,
        "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
    )
    encoded = physical
    for interval, padding in zip(
            attrs.get("shared_physical_intervals", ()),
            attrs.get("shared_physical_paddings", ()),
    ):
        encoded += state.dsl.floor(physical / int(interval)) * int(padding)
    return encoded


def _linear_inverse_offset_from_simd_coords(
    state,
    attrs,
    prefix,
    coords,
    lane_width,
    op,
    diagnostic,
):
    bases = _physical_linear_inverse_offset_bases(
        attrs,
        prefix,
        len(tuple(coords)),
        op,
        diagnostic,
    )
    result = state.builder.splat(
        state.builder.constant(state.dsl.i32(), 0),
        state.dsl.i32(),
        int(lane_width),
    )
    for dim, dim_bases in enumerate(bases):
        for bit, contribution in enumerate(dim_bases):
            if int(contribution) == 0:
                continue
            bit_value = _simd_binary_const(
                state,
                "divui",
                coords[dim],
                1 << int(bit),
                lane_width,
            )
            bit_value = _simd_binary_const(
                state,
                "remui",
                bit_value,
                2,
                lane_width,
            )
            if int(contribution) != 1:
                bit_value = _simd_binary_const(
                    state,
                    "muli",
                    bit_value,
                    int(contribution),
                    lane_width,
                    nsw=_LAYOUT_MATH_NSW,
                )
            result = state.builder.binary(
                state.dsl.BinaryKind.XOrI,
                result,
                bit_value,
            )
    return result


def _linear_inverse_offset_from_expr_coords(
    state,
    attrs,
    prefix,
    coords,
    op,
    diagnostic,
):
    bases = _physical_linear_inverse_offset_bases(
        attrs,
        prefix,
        len(tuple(coords)),
        op,
        diagnostic,
    )
    result = state.dsl.sym_ctx.int_(0)
    for dim, dim_bases in enumerate(bases):
        for bit, contribution in enumerate(dim_bases):
            if int(contribution) == 0:
                continue
            bit_value = state.dsl.mod(
                state.dsl.floor(coords[dim] / (1 << int(bit))),
                2,
            )
            if int(contribution) != 1:
                bit_value *= int(contribution)
            result = state.dsl.xor(result, bit_value)
    return result


def _physical_linear_inverse_offset_bases(
    attrs,
    prefix,
    rank,
    op,
    diagnostic,
):
    key = f"{prefix}_physical_linear_inverse_offset_bases"
    bases = tuple(
        tuple(int(value) for value in dim_bases)
        for dim_bases in attrs.get(key, ()))
    if len(bases) != int(rank):
        fail(
            diagnostic,
            STAGE,
            "shared_linear inverse offset bases do not match coordinate rank; "
            f"{key}={bases}, rank={rank}",
            target_op_id=op.target_op_id,
        )
    return bases


def _linear_component_offset_from_simd_coords(
    state,
    attrs,
    prefix,
    coords,
    lane_width,
    op,
    diagnostic,
):
    bases = _physical_linear_component_bases(attrs, prefix, op, diagnostic)
    result = state.builder.splat(
        state.builder.constant(state.dsl.i32(), 0),
        state.dsl.i32(),
        int(lane_width),
    )
    for bit, dim, value in _iter_linear_component_basis_bits(
            bases,
            len(tuple(coords)),
            op,
            diagnostic,
    ):
        bit_value = _simd_binary_const(
            state,
            "divui",
            coords[int(dim)],
            int(value),
            lane_width,
        )
        bit_value = _simd_binary_const(state, "remui", bit_value, 2, lane_width)
        if int(bit):
            bit_value = _simd_binary_const(
                state,
                "muli",
                bit_value,
                1 << int(bit),
                lane_width,
                nsw=_LAYOUT_MATH_NSW,
            )
        result = state.builder.binary(
            state.dsl.BinaryKind.AddI,
            result,
            bit_value,
            nsw=_LAYOUT_MATH_NSW,
        )
    return result


def _linear_component_offset_from_expr_coords(
    state,
    attrs,
    prefix,
    coords,
    op,
    diagnostic,
):
    bases = _physical_linear_component_bases(attrs, prefix, op, diagnostic)
    result = 0
    for bit, dim, value in _iter_linear_component_basis_bits(
            bases,
            len(tuple(coords)),
            op,
            diagnostic,
    ):
        bit_value = state.dsl.mod(
            state.dsl.floor(coords[int(dim)] / int(value)),
            2,
        )
        if int(bit):
            bit_value *= 1 << int(bit)
        result += bit_value
    return result


def _physical_linear_component_bases(attrs, prefix, op, diagnostic):
    key = f"{prefix}_physical_linear_component_bases"
    bases = tuple(tuple(int(value) for value in basis) for basis in attrs.get(key, ()))
    if not bases:
        fail(
            diagnostic,
            STAGE,
            f"padded shared physical offset is missing {key}",
            target_op_id=op.target_op_id,
        )
    return bases


def _iter_linear_component_basis_bits(bases, rank, op, diagnostic):
    rank = int(rank)
    for bit, basis in enumerate(tuple(bases)):
        basis = tuple(int(value) for value in basis)
        if len(basis) != rank:
            fail(
                diagnostic,
                STAGE,
                "padded shared linearComponent basis rank does not match "
                f"coordinate rank; basis={basis}, rank={rank}",
                target_op_id=op.target_op_id,
            )
        nonzero = [(dim, value) for dim, value in enumerate(basis) if value]
        if len(nonzero) != 1:
            fail(
                diagnostic,
                STAGE,
                "padded shared linearComponent offset basis must move in "
                f"exactly one dimension; basis={basis}",
                target_op_id=op.target_op_id,
            )
        dim, value = nonzero[0]
        if value <= 0 or not _is_power_of_two(value):
            fail(
                diagnostic,
                STAGE,
                "padded shared linearComponent offset basis must be a "
                f"positive power of two; basis={basis}",
                target_op_id=op.target_op_id,
            )
        yield int(bit), int(dim), int(value)


def _simd_binary_const(state, operation, value, constant, lane_width, *, nsw=False):
    constant = int(constant)
    simd = state.dsl.SimdType(value.type)
    element_type = simd.element_type
    lane_width = int(simd.width)
    if operation == "divui" and constant == 1:
        return value
    if operation == "remui" and constant == 1:
        return state.builder.splat(
            state.builder.constant(element_type, 0),
            element_type,
            lane_width,
        )
    operation_kind = _binary_kind(state.dsl, operation)
    rhs = state.builder.splat(
        state.builder.constant(element_type, constant),
        element_type,
        lane_width,
    )
    return state.builder.binary(operation_kind, value, rhs, nsw=bool(nsw))


def _is_power_of_two(value):
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


def _dense_tile_base_elements(shape, tile_offsets):
    element_offset = 0
    stride = 1
    for dim in reversed(range(len(shape))):
        element_offset += int(tile_offsets[dim]) * stride
        stride *= int(shape[dim])
    return int(element_offset)


def _add_optional_offset(state, base, offset):
    if offset is None:
        return base
    return state.builder.binary(state.dsl.BinaryKind.AddI, base, offset)


def _emit_mma_packet_constant(state, op):
    attrs = target_ir.attrs_dict(op)
    lane_width = int(attrs["lane_width"])
    registers = int(attrs["registers"])
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    scalar_payload_type = state.dsl.simd_type(element_type, width=lane_width)
    payload_type = state.dsl.simd_type(
        state.dsl.vector_type(registers, element_type),
        width=lane_width,
    )
    scalar_value = _wave_constant(
        state,
        scalar_payload_type,
        element_type,
        attrs["element_type"],
        attrs["value"],
        op,
    )
    payload = state.dsl.wave.PackOp(
        payload_type,
        tuple(scalar_value for _ in range(registers)),
    ).result
    state.values[_single_result(op)] = _pack_components(tuple(payload for _ in range(int(attrs["component_count"]))))


def _emit_mma(state, op):
    attrs = target_ir.attrs_dict(op)
    lhs, rhs, acc = _operand_values(state, op, 3)
    lhs_components = _as_components(lhs)
    rhs_components = _as_components(rhs)
    acc_components = _as_components(acc)
    lane_width = int(attrs["lane_width"])
    m_tiles = int(attrs["m_tiles"])
    n_tiles = int(attrs["n_tiles"])
    k_tiles = int(attrs.get("k_tiles", 1))
    if len(lhs_components) != m_tiles * k_tiles or len(rhs_components) != n_tiles * k_tiles:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "mma operand component counts do not match tile attrs",
            target_op_id=op.target_op_id,
        )
    if len(acc_components) != m_tiles * n_tiles:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "mma accumulator component count does not match tile attrs",
            target_op_id=op.target_op_id,
        )
    swap_operands = bool(attrs.get("swap_operands_for_transposed_result", False))
    lhs_role = int(attrs["rhs_role"] if swap_operands else attrs["lhs_role"])
    rhs_role = int(attrs["lhs_role"] if swap_operands else attrs["rhs_role"])
    lhs_components = tuple(
        _ensure_mma_fragment(
            state,
            component,
            role=lhs_role,
            element_type=attrs["lhs_element_type"],
            rows=int(attrs["lhs_rows"]),
            columns=int(attrs["lhs_columns"]),
            lane_width=lane_width,
            registers=int(attrs["lhs_registers"]),
        ) for component in lhs_components)
    rhs_components = tuple(
        _ensure_mma_fragment(
            state,
            component,
            role=rhs_role,
            element_type=attrs["rhs_element_type"],
            rows=int(attrs["rhs_rows"]),
            columns=int(attrs["rhs_columns"]),
            lane_width=lane_width,
            registers=int(attrs["rhs_registers"]),
        ) for component in rhs_components)
    acc_components = tuple(
        _ensure_mma_fragment(
            state,
            component,
            role=int(attrs["acc_role"]),
            element_type=attrs["acc_element_type"],
            rows=int(attrs["acc_rows"]),
            columns=int(attrs["acc_columns"]),
            lane_width=lane_width,
            registers=int(attrs["acc_registers"]),
        ) for component in acc_components)
    acc_payload_type = state.dsl.simd_type(
        state.dsl.vector_type(
            int(attrs["acc_registers"]),
            _scalar_type(state.dsl, attrs["acc_element_type"]),
        ),
        width=lane_width,
    )
    results = []
    for m_tile in range(m_tiles):
        for n_tile in range(n_tiles):
            index = m_tile * n_tiles + n_tile
            acc_value = acc_components[index]
            for k_tile in range(k_tiles):
                lhs_value = lhs_components[m_tile * k_tiles + k_tile]
                rhs_value = rhs_components[n_tile * k_tiles + k_tile]
                if swap_operands:
                    lhs_value, rhs_value = rhs_value, lhs_value
                acc_value = state.builder.mma(
                    attrs["kind"],
                    lhs_value,
                    rhs_value,
                    acc_value,
                )
            # Fragments are only MMA-local values. The bridge state carries the
            # accumulator as the SIMD vector payload so control-flow values do
            # not become WaveAMD fragment carriers.
            if not state.dsl.FragmentType.isinstance(acc_value.type):
                fail(
                    "TLXW_EMIT_FRAGMENT_TYPE",
                    STAGE,
                    "mma must produce a WaveAMD accumulator fragment",
                    target_op_id=op.target_op_id,
                )
            acc_value = state.dsl.waveamd.FragmentUnpackOp(
                acc_payload_type,
                acc_value,
            ).result
            results.append(acc_value)
    state.values[_single_result(op)] = _pack_components(tuple(results))


def _emit_mma_scaled(state, op):
    attrs = target_ir.attrs_dict(op)
    has_scales = bool(attrs.get("has_scales", False))
    operand_count = 5 if has_scales else 3
    operands = _operand_values(state, op, operand_count)
    lhs, rhs, acc = operands[:3]
    lhs_scale = operands[3] if has_scales else None
    rhs_scale = operands[4] if has_scales else None
    lhs_components = _as_components(lhs)
    rhs_components = _as_components(rhs)
    acc_components = _as_components(acc)
    lane_width = int(attrs["lane_width"])
    m_tiles = int(attrs["m_tiles"])
    n_tiles = int(attrs["n_tiles"])
    k_tiles = int(attrs["k_tiles"])
    if len(lhs_components) != m_tiles * k_tiles or len(rhs_components) != n_tiles * k_tiles:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "scaled mma operand component counts do not match tile attrs",
            target_op_id=op.target_op_id,
        )
    if len(acc_components) != m_tiles * n_tiles:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "scaled mma accumulator component count does not match tile attrs",
            target_op_id=op.target_op_id,
        )
    swap_operands = bool(attrs.get("swap_operands_for_transposed_result", False))
    lhs_role = int(attrs["rhs_role"] if swap_operands else attrs["lhs_role"])
    rhs_role = int(attrs["lhs_role"] if swap_operands else attrs["rhs_role"])
    lhs_components = tuple(
        _ensure_mma_fragment(
            state,
            component,
            role=lhs_role,
            element_type=attrs["lhs_element_type"],
            rows=int(attrs["lhs_rows"]),
            columns=int(attrs["lhs_columns"]),
            lane_width=lane_width,
            registers=int(attrs["lhs_registers"]),
        ) for component in lhs_components)
    rhs_components = tuple(
        _ensure_mma_fragment(
            state,
            component,
            role=rhs_role,
            element_type=attrs["rhs_element_type"],
            rows=int(attrs["rhs_rows"]),
            columns=int(attrs["rhs_columns"]),
            lane_width=lane_width,
            registers=int(attrs["rhs_registers"]),
        ) for component in rhs_components)
    acc_components = tuple(
        _ensure_mma_fragment(
            state,
            component,
            role=int(attrs["acc_role"]),
            element_type=attrs["acc_element_type"],
            rows=int(attrs["acc_rows"]),
            columns=int(attrs["acc_columns"]),
            lane_width=lane_width,
            registers=int(attrs["acc_registers"]),
        ) for component in acc_components)
    if has_scales:
        lhs_scale_values = _pack_scale_groups(
            state,
            lhs_scale,
            attrs,
            "lhs_scale",
            lane_width,
            op,
        )
        rhs_scale_values = _pack_scale_groups(
            state,
            rhs_scale,
            attrs,
            "rhs_scale",
            lane_width,
            op,
        )
    else:
        zero_scale = _zero_scale_vector(state, lane_width, op)
        lhs_scale_values = (zero_scale, )
        rhs_scale_values = (zero_scale, )
    acc_payload_type = state.dsl.simd_type(
        state.dsl.vector_type(
            int(attrs["acc_registers"]),
            _scalar_type(state.dsl, attrs["acc_element_type"]),
        ),
        width=lane_width,
    )
    results = []
    for m_tile in range(m_tiles):
        for n_tile in range(n_tiles):
            index = m_tile * n_tiles + n_tile
            acc_value = acc_components[index]
            for k_tile in range(k_tiles):
                lhs_value = lhs_components[m_tile * k_tiles + k_tile]
                rhs_value = rhs_components[n_tile * k_tiles + k_tile]
                lhs_scale_value, lhs_scale_idx = _select_packed_scale(
                    lhs_scale_values,
                    attrs,
                    "lhs_scale",
                    m_tile,
                    k_tile,
                    k_tiles,
                    op,
                )
                rhs_scale_value, rhs_scale_idx = _select_packed_scale(
                    rhs_scale_values,
                    attrs,
                    "rhs_scale",
                    n_tile,
                    k_tile,
                    k_tiles,
                    op,
                )
                if swap_operands:
                    lhs_value, rhs_value = rhs_value, lhs_value
                    lhs_scale_value, rhs_scale_value = rhs_scale_value, lhs_scale_value
                    lhs_scale_idx, rhs_scale_idx = rhs_scale_idx, lhs_scale_idx
                acc_value = state.builder.mma_scale(
                    attrs["kind"],
                    lhs_value,
                    lhs_scale_value,
                    rhs_value,
                    rhs_scale_value,
                    acc_value,
                    scale_idx_a=int(lhs_scale_idx),
                    scale_idx_b=int(rhs_scale_idx),
                )
            if not state.dsl.FragmentType.isinstance(acc_value.type):
                fail(
                    "TLXW_EMIT_FRAGMENT_TYPE",
                    STAGE,
                    "scaled mma must produce a WaveAMD accumulator fragment",
                    target_op_id=op.target_op_id,
                )
            acc_value = state.dsl.waveamd.FragmentUnpackOp(
                acc_payload_type,
                acc_value,
            ).result
            results.append(acc_value)
    state.values[_single_result(op)] = _pack_components(tuple(results))


def _pack_scale_groups(state, components, attrs, prefix, lane_width, op):
    pack_width = int(attrs[f"{prefix}_pack_width"])
    group_count = int(attrs[f"{prefix}_group_count"])
    packet_payload = _as_vector_packet_payload(state, components, pack_width)
    if packet_payload is not None:
        if len(packet_payload.packets) < group_count:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "scaled mma scale packet count does not match pack attrs",
                target_op_id=op.target_op_id,
            )
        return tuple(packet_payload.packets[:group_count])
    components = _value_components(state, components, op)
    required = pack_width * group_count
    if len(components) < required:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "scaled mma scale component count does not match pack attrs",
            target_op_id=op.target_op_id,
        )
    zero = _zero_simd_value(
        state,
        state.dsl.simd_type(state.dsl.i8(), int(lane_width)),
        "i8",
        op,
    )
    scale_vector_width = 4
    scale_type = _scale_packet_type(state, lane_width, scale_vector_width)
    packed = []
    for group in range(group_count):
        base = group * pack_width
        values = list(components[base:base + pack_width])
        values.extend([zero] * (scale_vector_width - len(values)))
        packed.append(state.dsl.wave.PackOp(scale_type, values).result)
    return tuple(packed)


def _zero_scale_vector(state, lane_width, op):
    zero = _zero_simd_value(
        state,
        state.dsl.simd_type(state.dsl.i8(), int(lane_width)),
        "i8",
        op,
    )
    scale_type = _scale_packet_type(state, lane_width, 4)
    return state.dsl.wave.PackOp(scale_type, [zero for _ in range(4)]).result


def _as_vector_packet_payload(state, value, pack_width):
    if isinstance(value, _VectorPacketPayload):
        if int(value.packet_width) == 4 and int(pack_width) == int(value.packet_width):
            return value
        return None
    components = _as_components(value)
    if not components:
        return None
    if int(pack_width) != 4:
        return None
    if all(_is_i8_vector_packet(state, component) for component in components):
        return _VectorPacketPayload(
            tuple(components),
            4,
            len(components) * 4,
        )
    return None


def _is_i8_vector_packet(state, value):
    payload = _simd_1d_vector_payload(state, value)
    if payload is None:
        return False
    width, element_type, lane_width = payload
    del lane_width
    return int(width) == 4 and str(element_type) == "i8"


def _scale_packet_type(state, lane_width, packet_width):
    return state.dsl.simd_type(
        state.dsl.vector_type(int(packet_width), state.dsl.i8()),
        int(lane_width),
    )


def _select_packed_scale(scale_values, attrs, prefix, non_k_tile, k_tile, k_tiles, op):
    k_packed_vals = int(attrs.get(f"{prefix}_k_packed_vals", 1))
    non_k_packed_vals = int(attrs.get(f"{prefix}_non_k_packed_vals", 1))
    k_groups = int(attrs.get(f"{prefix}_k_groups", 1))
    k_group = int(k_tile) // k_packed_vals
    non_k_group = int(non_k_tile) // non_k_packed_vals
    group = non_k_group * k_groups + k_group
    if group < 0 or group >= len(scale_values):
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            f"scale group {group} outside packed {prefix} values",
            target_op_id=op.target_op_id,
        )
    scale_idx = (int(non_k_tile) * int(k_tiles) + int(k_tile)) % (non_k_packed_vals * k_packed_vals)
    return scale_values[group], int(scale_idx)


def _ensure_mma_fragment(
    state,
    value,
    *,
    role,
    element_type,
    rows,
    columns,
    lane_width,
    registers,
):
    fragment_type = state.dsl.fragment_type(
        int(role),
        _scalar_type(state.dsl, element_type),
        int(rows),
        int(columns),
        int(lane_width),
        int(registers),
    )
    if state.dsl.FragmentType.isinstance(value.type):
        fail(
            "TLXW_EMIT_FRAGMENT_BOUNDARY",
            STAGE,
            "mma operands must arrive as ordinary typed SIMD packets, not "
            "pre-existing WaveAMD fragments",
        )
    return state.builder.fragment_pack(value, fragment_type)


def _emit_mma_packet_truncf(state, op):
    attrs = target_ir.attrs_dict(op)
    (fragment_value, ) = _operand_values(state, op, 1)
    fragments = _as_components(fragment_value)
    component_count = int(attrs["component_count"])
    if len(fragments) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "MMA packet truncf component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    lane_width = int(attrs["lane_width"])
    registers = int(attrs["registers"])
    result_element_type = attrs.get("result_element_type", "f16")
    result_regs = state.dsl.simd_type(
        state.dsl.vector_type(registers, _scalar_type(state.dsl, result_element_type)),
        width=lane_width,
    )
    packed = []
    for regs in fragments:
        payload = _simd_1d_vector_payload(state, regs)
        if (payload is None or int(payload[0]) != registers
                or str(payload[1]) != "f32" or int(payload[2]) != lane_width):
            fail(
                "TLXW_EMIT_FRAGMENT_TYPE",
                STAGE,
                "MMA packet truncf requires an f32 SIMD packet with the "
                "planned register and lane widths",
                target_op_id=op.target_op_id,
            )
        packed.append(state.builder.fpconvert(regs, result_regs))
    state.values[_single_result(op)] = _pack_components(tuple(packed))


def _emit_redistribute_layout_convert(state, op, value, attrs):
    """Pack bridge components, emit one semantic redistribution, and unpack."""
    result_id = _single_result(op)
    source_type = state.target_program.values[op.operands[0]].type
    result_type = state.target_program.values[result_id].type
    if bool(attrs.get("cross_wave", False)):
        # The bridge owns dependencies from existing LDS accesses. Wave owns
        # only the scratch exchange introduced while lowering redistribution.
        released = _release_dead_local_memory_before_redistribute(state, op)
        _scratch_write_dependency(state, extra_tokens=released)
    lane_width = int(result_type.lane_width or source_type.lane_width or 64)
    mask_payload = result_type.representation in {"mask", "mask_tuple"}
    element_type = (
        state.dsl.i32()
        if mask_payload
        else _scalar_type(state.dsl, attrs["element_type"])
    )
    source_count = int(attrs["source_component_count"])
    source_registers = int(attrs["source_registers_per_component"])
    source_slots = int(attrs["source_slot_count"])
    if isinstance(value, _RawLayoutVectorPacketPayload):
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            "redistribution requires a typed SIMD payload",
            target_op_id=op.target_op_id,
        )
    if source_type.representation in {"mask", "mask_tuple"}:
        components = _as_mask_payload_components(
            state,
            value,
            source_count,
            lane_width,
            op,
        )
    else:
        components = _value_components(state, value, op)
    if len(components) != source_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "redistribution source components do not match packet attrs",
            target_op_id=op.target_op_id,
        )
    source_chunk_type = (
        state.dsl.simd_type(element_type, lane_width)
        if source_registers == 1
        else state.dsl.simd_type(
            state.dsl.vector_type(source_registers, element_type),
            lane_width,
        )
    )
    chunks = tuple(
        _redistribution_component_chunk(
            state,
            component,
            source_chunk_type,
            source_registers,
            op,
        )
        for component in components
    )
    source_packet_type = state.dsl.simd_type(
        state.dsl.vector_type(source_slots, element_type),
        lane_width,
    )
    source_packet = state.dsl.wave.PackOp(source_packet_type, chunks).result
    result_slots = int(attrs["result_slot_count"])
    result_packet_type = state.dsl.simd_type(
        state.dsl.vector_type(result_slots, element_type),
        lane_width,
    )
    item = state.dsl.sym("item")
    slot = state.dsl.sym("slot")
    block = state.dsl.sym("block")
    inputs = {
        "register": slot,
        "lane": state.dsl.mod(item, lane_width),
        "warp": state.dsl.floor(item / lane_width),
        "block": block,
    }
    source_slot = _redistribution_relation_expr(
        state,
        attrs,
        "register",
        inputs,
        op,
    )
    source_lane = _redistribution_relation_expr(
        state,
        attrs,
        "lane",
        inputs,
        op,
    )
    source_warp = _redistribution_relation_expr(
        state,
        attrs,
        "warp",
        inputs,
        op,
    )
    out_names = {str(name) for name, _size in attrs["relation_out_dims"]}
    source_block = (
        _redistribution_relation_expr(
            state,
            attrs,
            "block",
            inputs,
            op,
        )
        if "block" in out_names
        else block
    )
    redistributed = state.builder.redistribute(
        source_packet,
        result_packet_type,
        blocks=int(attrs.get("block_count", 1)),
        items=int(attrs["cta_thread_count"]),
        source_block=source_block,
        source_item=source_lane + lane_width * source_warp,
        source_slot=source_slot,
    )
    result_count = int(attrs["result_component_count"])
    result_registers = int(attrs["result_registers_per_component"])
    result_chunk_type = (
        state.dsl.simd_type(element_type, lane_width)
        if result_registers == 1
        else state.dsl.simd_type(
            state.dsl.vector_type(result_registers, element_type),
            lane_width,
        )
    )
    result_components = tuple(
        state.dsl.wave.ExtractOp(
            result_chunk_type,
            redistributed,
            component * result_registers,
        ).result
        for component in range(result_count)
    )
    state.values[result_id] = (
        _I32MaskPayload(result_components)
        if mask_payload
        else _pack_components(result_components)
    )


def _redistribution_component_chunk(state, component, chunk_type, width, op):
    payload = _simd_1d_vector_payload(state, component)
    if int(width) == 1:
        if payload is None:
            return component
        if int(payload[0]) == 1:
            return state.dsl.wave.ExtractOp(chunk_type, component, 0).result
    elif payload is not None and int(payload[0]) == int(width):
        return component
    fail(
        "TLXW_EMIT_COMPONENT_COUNT",
        STAGE,
        "redistribution component does not match its packet width",
        target_op_id=op.target_op_id,
    )


def _redistribution_relation_expr(state, attrs, output_name, inputs, op):
    out_names = tuple(str(name) for name, _size in attrs["relation_out_dims"])
    if output_name not in out_names:
        return state.dsl.sym_ctx.int_(0)
    output_index = out_names.index(output_name)
    result = state.dsl.sym_ctx.int_(0)
    for input_name, bases in attrs["relation_bases"]:
        if input_name not in inputs:
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                f"redistribution relation has unsupported input {input_name!r}",
                target_op_id=op.target_op_id,
            )
        source = inputs[input_name]
        for bit, basis in enumerate(bases):
            coefficient = int(basis[output_index])
            if coefficient == 0:
                continue
            value = state.dsl.mod(
                state.dsl.floor(source / (1 << bit)),
                2,
            )
            if coefficient != 1:
                value *= coefficient
            result = state.dsl.xor(result, value)
    return result


def _emit_layout_convert(state, op):
    attrs = target_ir.attrs_dict(op)
    (value, ) = _operand_values(state, op, 1)
    mode = attrs["mode"]
    if mode == "alias":
        state.values[_single_result(op)] = _emit_layout_packet_alias(
            state,
            op,
            value,
            attrs,
        )
        return
    if mode == "redistribute":
        _emit_redistribute_layout_convert(state, op, value, attrs)
        return
    fail(
        "TLXW_EMIT_UNSUPPORTED_LAYOUT_CONVERT",
        STAGE,
        f"unsupported layout_convert mode {mode}",
        target_op_id=op.target_op_id,
    )


def _emit_layout_packet_alias(state, op, value, attrs):
    """Regroup an identity layout relation without moving register bits."""
    grouping_keys = {
        "source_component_count",
        "source_packet_width",
        "source_slot_count",
        "result_component_count",
        "result_packet_width",
        "result_slot_count",
    }
    if not grouping_keys.intersection(attrs):
        return value
    if not grouping_keys.issubset(attrs):
        fail(
            "TLXW_EMIT_LAYOUT_ALIAS",
            STAGE,
            "packet alias requires a complete source/result grouping",
            target_op_id=op.target_op_id,
        )
    source_count = int(attrs["source_component_count"])
    source_width = int(attrs["source_packet_width"])
    source_slots = int(attrs["source_slot_count"])
    result_count = int(attrs["result_component_count"])
    result_width = int(attrs["result_packet_width"])
    result_slots = int(attrs["result_slot_count"])
    if (
        source_count <= 0
        or source_width <= 0
        or result_count <= 0
        or result_width <= 0
        or source_count * source_width != source_slots
        or result_count * result_width != result_slots
        or source_slots != result_slots
    ):
        fail(
            "TLXW_EMIT_LAYOUT_ALIAS",
            STAGE,
            "packet alias source/result grouping is inconsistent",
            target_op_id=op.target_op_id,
        )
    if source_count == result_count and source_width == result_width:
        return value

    if isinstance(value, _RawLayoutVectorPacketPayload):
        if (
            int(value.logical_component_count) != source_slots
            or int(value.packet_width) != result_width
            or len(value.packets) != result_count
        ):
            fail(
                "TLXW_EMIT_LAYOUT_ALIAS",
                STAGE,
                "raw register packets do not match the structural alias "
                "result grouping",
                target_op_id=op.target_op_id,
            )
        return _pack_components(value.packets)

    scalar_slots = _layout_alias_scalar_slots(
        state,
        op,
        value,
        source_count,
        source_width,
    )
    if len(scalar_slots) != result_slots:
        fail(
            "TLXW_EMIT_LAYOUT_ALIAS",
            STAGE,
            "packet alias scalar payload does not match its result grouping",
            target_op_id=op.target_op_id,
        )
    if result_width == 1:
        return _pack_components(scalar_slots)

    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    element_type = _scalar_type(state.dsl, target_type.element_type)
    lane_width = int(target_type.lane_width or 64)
    packet_type = state.dsl.simd_type(
        state.dsl.vector_type(result_width, element_type),
        lane_width,
    )
    packets = tuple(
        state.dsl.wave.PackOp(
            packet_type,
            scalar_slots[index:index + result_width],
        ).result
        for index in range(0, result_slots, result_width)
    )
    return _pack_components(packets)


def _layout_alias_scalar_slots(
    state,
    op,
    value,
    source_count,
    source_width,
):
    if isinstance(value, _VectorPacketPayload):
        scalar_slots = _value_components(state, value, op)
        if len(scalar_slots) != source_count * source_width:
            fail(
                "TLXW_EMIT_LAYOUT_ALIAS",
                STAGE,
                "vector packet alias payload has an inconsistent shape",
                target_op_id=op.target_op_id,
            )
        return tuple(scalar_slots)

    components = _as_components(value)
    if len(components) != source_count:
        fail(
            "TLXW_EMIT_LAYOUT_ALIAS",
            STAGE,
            "packet alias source component count does not match its payload",
            target_op_id=op.target_op_id,
        )
    scalar_slots = []
    for component in components:
        payload = _simd_1d_vector_payload(state, component)
        if source_width == 1:
            if payload is None:
                scalar_slots.append(component)
                continue
            width, element_type, lane_width = payload
            if int(width) != 1:
                fail(
                    "TLXW_EMIT_LAYOUT_ALIAS",
                    STAGE,
                    "scalar packet alias component contains multiple elements",
                    target_op_id=op.target_op_id,
                )
            scalar_type = state.dsl.simd_type(element_type, int(lane_width))
            scalar_slots.append(
                state.dsl.wave.ExtractOp(scalar_type, component, 0).result
            )
            continue
        if payload is None or int(payload[0]) != source_width:
            fail(
                "TLXW_EMIT_LAYOUT_ALIAS",
                STAGE,
                "packet alias source component has the wrong vector width",
                target_op_id=op.target_op_id,
            )
        _width, element_type, lane_width = payload
        scalar_type = state.dsl.simd_type(element_type, int(lane_width))
        scalar_slots.extend(
            state.dsl.wave.ExtractOp(
                scalar_type,
                component,
                element,
            ).result
            for element in range(source_width)
        )
    return tuple(scalar_slots)

def _raw_register_packet_type(state, packet_bits, lane_width, op):
    packet_bits = int(packet_bits)
    if packet_bits <= 0 or packet_bits % 32:
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            "raw register packet payload must contain whole dwords",
            target_op_id=op.target_op_id,
        )
    word_count = packet_bits // 32
    if word_count == 1:
        return state.dsl.simd_type(state.dsl.i32(), width=int(lane_width))
    return state.dsl.simd_type(
        state.dsl.vector_type(word_count, state.dsl.i32()),
        width=int(lane_width),
    )


def _bit_linear_thread_offset_index_expr(state, workitem, base, coefficients):
    workitem_symbol = state.dsl.sym("wi")
    expr = state.dsl.sym_ctx.int_(int(base))
    for bit, coefficient in enumerate(coefficients):
        coefficient = int(coefficient)
        if coefficient == 0:
            continue
        bit_value = state.dsl.mod(
            state.dsl.floor(workitem_symbol / (1 << bit)),
            2,
        )
        if coefficient != 1:
            bit_value *= coefficient
        expr = state.dsl.xor(expr, bit_value)
    return state.builder.index_expr(
        expr,
        bindings={workitem_symbol: workitem},
    )


def _bit_affine_thread_offset_expr(state, workitem, base, coefficients):
    result = state.dsl.sym_ctx.int_(int(base))
    for bit, coefficient in enumerate(coefficients):
        coefficient = int(coefficient)
        if coefficient == 0:
            continue
        bit_value = state.dsl.mod(state.dsl.floor(workitem / (1 << bit)), 2)
        if coefficient != 1:
            bit_value *= coefficient
        result += bit_value
    return result


def _bit_affine_thread_offset(state, workitem, base, coefficients, lane_width):
    lane_width = int(lane_width)
    packed = _packed_bit_affine_coefficients(coefficients)
    if packed is not None:
        first_bit, bit_count, stride = packed
        if int(stride) == 0:
            return state.builder.splat(
                state.builder.constant(state.dsl.i32(), int(base)),
                state.dsl.i32(),
                lane_width,
            )
        result = workitem
        if first_bit:
            result = _simd_binary_const(
                state,
                "divui",
                result,
                1 << int(first_bit),
                lane_width,
            )
        result = _simd_binary_const(
            state,
            "remui",
            result,
            1 << int(bit_count),
            lane_width,
        )
        if int(stride) != 1:
            result = _simd_binary_const(
                state,
                "muli",
                result,
                int(stride),
                lane_width,
                nsw=_LAYOUT_MATH_NSW,
            )
        if int(base):
            result = _simd_binary_const(
                state,
                "addi",
                result,
                int(base),
                lane_width,
                nsw=_LAYOUT_MATH_NSW,
            )
        return result
    result = state.builder.splat(
        state.builder.constant(state.dsl.i32(), int(base)),
        state.dsl.i32(),
        lane_width,
    )
    for bit, coefficient in enumerate(coefficients):
        coefficient = int(coefficient)
        if coefficient == 0:
            continue
        bit_value = _simd_binary_const(state, "divui", workitem, 1 << bit, lane_width)
        bit_value = _simd_binary_const(state, "remui", bit_value, 2, lane_width)
        if coefficient != 1:
            bit_value = _simd_binary_const(
                state,
                "muli",
                bit_value,
                coefficient,
                lane_width,
                nsw=_LAYOUT_MATH_NSW,
            )
        result = state.builder.binary(
            state.dsl.BinaryKind.AddI,
            result,
            bit_value,
            nsw=_LAYOUT_MATH_NSW,
        )
    return result


def _packed_bit_affine_coefficients(coefficients):
    nonzero = [(bit, int(coefficient)) for bit, coefficient in enumerate(coefficients) if int(coefficient)]
    if not nonzero:
        return 0, 1, 0
    first_bit, first_coefficient = nonzero[0]
    if first_coefficient <= 0:
        return None
    for expected_index, (bit, coefficient) in enumerate(nonzero):
        if int(bit) != int(first_bit) + int(expected_index):
            return None
        if int(coefficient) != int(first_coefficient) << int(expected_index):
            return None
    return int(first_bit), len(nonzero), int(first_coefficient)


def _direct_buffer_load_cache_attr(state, attrs, op):
    modifier = int(attrs.get("cache_modifier", 1))
    if modifier == 1:
        return None
    kind = {
        2: state.dsl.LoadCacheAttr.CA,
        3: state.dsl.LoadCacheAttr.CG,
        5: state.dsl.LoadCacheAttr.CS,
        7: state.dsl.LoadCacheAttr.CV,
    }.get(modifier)
    if kind is None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_CACHE_MODIFIER",
            STAGE,
            f"unsupported buffer_load cache modifier {modifier}",
            target_op_id=op.target_op_id,
        )
    return state.dsl.load_cache(kind)


def _direct_buffer_store_cache_attr(state, attrs, op):
    modifier = int(attrs.get("cache_modifier", 1))
    if modifier == 1:
        return None
    kind = {
        3: state.dsl.StoreCacheAttr.CG,
        4: state.dsl.StoreCacheAttr.WB,
        5: state.dsl.StoreCacheAttr.CS,
        6: state.dsl.StoreCacheAttr.WT,
    }.get(modifier)
    if kind is None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_CACHE_MODIFIER",
            STAGE,
            f"unsupported buffer_store cache modifier {modifier}",
            target_op_id=op.target_op_id,
        )
    return state.dsl.store_cache(kind)


def _emit_buffer_store(state, op):
    attrs = target_ir.attrs_dict(op)
    has_mask = bool(attrs["has_mask"])
    mask_operand_count = int(has_mask)
    expected_operand_count = 3 + mask_operand_count
    if len(op.operands) != expected_operand_count:
        fail(
            "TLXW_EMIT_OPERAND_COUNT",
            STAGE,
            f"expected {expected_operand_count} operands, got {len(op.operands)}",
            target_op_id=op.target_op_id,
        )
    value = _require_value(state, op.operands[0], op)
    source_base = _require_value(state, op.operands[1], op)
    offsets = _require_value(state, op.operands[2], op)
    masks = (
        _require_value(state, op.operands[3], op)
        if has_mask else None
    )
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    lane_width = int(attrs["lane_width"])
    component_count = int(attrs["component_count"])
    value_payload_width = int(attrs.get("value_payload_width", 1))
    access_component_count = int(
        attrs.get("access_component_count", component_count)
    )
    value_components = _value_components(state, value, op)
    offset_components = _as_components(offsets)
    mask_payload = masks if isinstance(masks, _I32MaskPayload) else None
    mask_components = None
    if mask_payload is not None and mask_payload.predicates is not None:
        mask_components = _broadcast_component_count(
            mask_payload.predicates,
            access_component_count,
            "buffer_store mask predicate",
            op,
        )
        mask_payload = None
    if mask_payload is not None:
        mask_payload = _materialize_mask_payload(
            state,
            mask_payload,
            lane_width,
        )
    if mask_components is None and masks is not None and mask_payload is None:
        mask_components = _as_mask_predicate_components(
            state,
            masks,
            access_component_count,
            int(attrs["lane_width"]),
            op,
        )
    if (
        len(value_components) != component_count
        or len(offset_components) != access_component_count
    ):
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_store value/offset component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    if mask_components is not None and len(mask_components) != access_component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_store mask component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    if mask_payload is not None and len(mask_payload.components) != access_component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_store mask payload component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    mask_mode = attrs.get("mask_mode", "exec_where" if attrs["has_mask"] else "none")
    buffer_base = _ptr_cast(
        state,
        source_base,
        state.dsl.ptr_type(element_type, state.dsl.global_address_space()),
    )
    access_offset_components = offset_components
    store_offset_components = tuple(
        access_offset_components[component * value_payload_width]
        for component in range(component_count)
    )
    zero_mask_payload = None

    def component_mask(index):
        nonlocal zero_mask_payload
        if mask_payload is not None:
            if zero_mask_payload is None:
                zero_mask_payload = _simd_i32_constant(state, lane_width, 0)
            return _cmpi(
                state,
                "ne",
                mask_payload.components[int(index)],
                zero_mask_payload,
            )
        if mask_components is not None:
            return mask_components[int(index)]
        return None

    def packet_mask(index, packet_elements):
        nonlocal zero_mask_payload
        index = int(index)
        packet_elements = int(packet_elements)
        if mask_payload is not None:
            payload = mask_payload.components[index]
            if (not all(mask_payload.components[index + element] is payload for element in range(packet_elements))
                    and not _buffer_store_packet_uses_leading_mask(attrs, index, packet_elements)):
                return None, False
            if zero_mask_payload is None:
                zero_mask_payload = _simd_i32_constant(state, lane_width, 0)
            return _cmpi(state, "ne", payload, zero_mask_payload), True
        if mask_components is not None:
            mask = mask_components[index]
            if (not all(mask_components[index + element] is mask for element in range(packet_elements))
                    and not _buffer_store_packet_uses_leading_mask(attrs, index, packet_elements)):
                return None, False
            return mask, True
        return None, True

    def vector_payload_packet_mask(index, packet_components):
        nonlocal zero_mask_payload
        access_index = int(index) * value_payload_width
        packet_elements = int(packet_components) * value_payload_width
        if mask_payload is not None:
            payload = mask_payload.components[access_index]
            if (not all(mask_payload.components[access_index + element] is payload
                        for element in range(packet_elements))
                    and not _buffer_store_packet_uses_leading_mask(
                        attrs,
                        access_index,
                        packet_elements,
                    )):
                return None, False
            if zero_mask_payload is None:
                zero_mask_payload = _simd_i32_constant(state, lane_width, 0)
            return _cmpi(state, "ne", payload, zero_mask_payload), True
        if mask_components is not None:
            mask = mask_components[access_index]
            if (not all(mask_components[access_index + element] is mask
                        for element in range(packet_elements))
                    and not _buffer_store_packet_uses_leading_mask(
                        attrs,
                        access_index,
                        packet_elements,
                    )):
                return None, False
            return mask, True
        return None, True

    index = 0
    while index < component_count:
        packet_elements = 1
        if value_payload_width == 1:
            packet_elements = _buffer_store_packet_elements(
                state,
                attrs,
                index,
                component_count,
                value_components,
                store_offset_components,
                mask_components,
                mask_payload,
            )
        if packet_elements > 1:
            mask_component, mask_supported = packet_mask(index, packet_elements)
            if mask_supported:
                packet_offset_component = store_offset_components[index]
                _emit_buffer_store_vector_packet(
                    state,
                    op,
                    attrs,
                    index,
                    packet_elements,
                    buffer_base,
                    value_components,
                    packet_offset_component,
                    mask_component,
                    element_type,
                    lane_width,
                    mask_mode,
                )
                index += packet_elements
                continue

        value_component = value_components[index]
        access_index = index * value_payload_width
        mask_payload_component = None
        direct_mask_component = None
        if mask_payload is not None:
            mask_payload_component = mask_payload.components[access_index]
        elif mask_components is not None:
            direct_mask_component = mask_components[access_index]

        value_vector = _simd_1d_vector_payload(state, value_component)
        if value_vector is not None:
            value_length, value_element_type, value_width = value_vector
            if value_width != lane_width:
                fail(
                    "TLXW_EMIT_UNSUPPORTED_BUFFER_STORE",
                    STAGE,
                    "buffer_store vector value payload lane width must match "
                    "the store lane width",
                    target_op_id=op.target_op_id,
                )
            mask_payload_vector = (None if mask_payload_component is None else _simd_1d_vector_payload(
                state, mask_payload_component))
            if mask_payload_vector is not None and (mask_payload_vector[0] != value_length
                                                    or mask_payload_vector[2] != lane_width):
                fail(
                    "TLXW_EMIT_UNSUPPORTED_BUFFER_STORE",
                    STAGE,
                    "buffer_store vector mask payload must match the value "
                    "payload shape",
                    target_op_id=op.target_op_id,
                )
            if direct_mask_component is not None:
                direct_mask_vector = _simd_1d_vector_payload(
                    state,
                    direct_mask_component,
                )
                if direct_mask_vector is not None:
                    fail(
                        "TLXW_EMIT_UNSUPPORTED_BUFFER_STORE",
                        STAGE,
                        "buffer_store does not support vector predicate masks; "
                        "use an i32 mask payload remap",
                        target_op_id=op.target_op_id,
                    )
            vector_packet_components = 0
            if mask_payload_vector is None:
                vector_packet_components = _buffer_store_vector_payload_packet_components(
                    state,
                    attrs,
                    index,
                    component_count,
                    value_components,
                    store_offset_components,
                    value_length,
                    value_element_type,
                    lane_width,
                    None,
                    None,
                )
            if vector_packet_components > 0:
                mask_component, mask_supported = vector_payload_packet_mask(
                    index,
                    vector_packet_components,
                )
                if mask_supported:
                    packet_offset_component = store_offset_components[index]
                    _emit_buffer_store_vector_payload_packet(
                        state,
                        op,
                        attrs,
                        index,
                        vector_packet_components,
                        value_length,
                        buffer_base,
                        value_components,
                        packet_offset_component,
                        mask_component,
                        element_type,
                        lane_width,
                        mask_mode,
                    )
                    index += vector_packet_components
                    continue
            value_scalar_type = state.dsl.simd_type(value_element_type, lane_width)
            for element_index in range(value_length):
                scalar_value = state.dsl.wave.ExtractOp(
                    value_scalar_type,
                    value_component,
                    int(element_index),
                ).result
                scalar_offset = access_offset_components[
                    access_index + element_index
                ]
                scalar_mask = component_mask(access_index + element_index)
                _emit_buffer_store_component(
                    state,
                    op,
                    attrs,
                    buffer_base,
                    lane_width,
                    scalar_value,
                    scalar_offset,
                    scalar_mask,
                    mask_mode,
                )
            index += 1
            continue

        offset_component = store_offset_components[index]
        mask_component = component_mask(access_index)
        _emit_buffer_store_component(
            state,
            op,
            attrs,
            buffer_base,
            lane_width,
            value_component,
            offset_component,
            mask_component,
            mask_mode,
        )
        index += 1


def _buffer_store_packet_elements(
    state,
    attrs,
    index,
    component_count,
    value_components,
    offset_components,
    mask_components,
    mask_payload,
):
    max_packet_elements = _buffer_max_packet_elements(attrs)
    if max_packet_elements <= 1:
        return 1
    max_packet_elements = min(max_packet_elements, int(component_count) - int(index))
    for packet_elements in range(max_packet_elements, 1, -1):
        if not _buffer_packet_payload_is_legal(packet_elements, int(attrs["element_byte_width"])):
            continue
        if not _buffer_store_packet_components_are_scalar(
                state,
                index,
                packet_elements,
                value_components,
                offset_components,
        ):
            continue
        if not _buffer_store_packet_mask_is_supported(attrs, index, packet_elements, mask_components, mask_payload):
            continue
        if _buffer_store_access_group_allows_packet(attrs, index, packet_elements):
            return packet_elements
    return 1


def _buffer_store_vector_payload_packet_components(
    state,
    attrs,
    index,
    component_count,
    value_components,
    offset_components,
    vector_elements,
    vector_element_type,
    lane_width,
    mask_components,
    mask_payload,
):
    index = int(index)
    component_count = int(component_count)
    vector_elements = int(vector_elements)
    if vector_elements <= 1:
        return 0
    if int(attrs.get("access_element_count", 1)) < vector_elements:
        return 0
    max_packet_elements = min(
        _buffer_max_packet_elements(attrs),
        int(attrs.get("access_element_count", 1)),
    )
    max_packet_components = min(
        component_count - index,
        max_packet_elements // vector_elements,
    )
    for packet_components in range(max_packet_components, 0, -1):
        packet_elements = int(packet_components) * vector_elements
        if not _buffer_packet_payload_is_legal(packet_elements, int(attrs["element_byte_width"])):
            continue
        if not _buffer_store_vector_payload_components_match(
                state,
                index,
                packet_components,
                value_components,
                vector_elements,
                vector_element_type,
                lane_width,
        ):
            continue
        if not _buffer_store_vector_payload_offsets_are_scalar(
                state,
            index,
            packet_components,
            offset_components,
        ):
            continue
        if not _buffer_store_vector_payload_packet_mask_is_supported(
                index,
                packet_components,
                mask_components,
                mask_payload,
        ):
            continue
        if packet_components == 1:
            return 1
        if _buffer_store_vector_payload_access_group_allows_packet(
                attrs,
                index,
                packet_components,
                vector_elements,
        ):
            return int(packet_components)
    return 0


def _buffer_store_vector_payload_components_match(
    state,
    index,
    packet_components,
    value_components,
    vector_elements,
    vector_element_type,
    lane_width,
):
    for component in range(int(packet_components)):
        payload = _simd_1d_vector_payload(state, value_components[int(index) + component])
        if payload is None:
            return False
        payload_elements, payload_element_type, payload_lane_width = payload
        if int(payload_elements) != int(vector_elements):
            return False
        if str(payload_element_type) != str(vector_element_type):
            return False
        if int(payload_lane_width) != int(lane_width):
            return False
    return True


def _buffer_store_vector_payload_offsets_are_scalar(
    state,
    index,
    packet_components,
    offset_components,
):
    for component in range(int(packet_components)):
        if _simd_1d_vector_payload(state, offset_components[int(index) + component]) is not None:
            return False
    return True


def _buffer_store_vector_payload_packet_mask_is_supported(index, packet_components, mask_components, mask_payload):
    index = int(index)
    packet_components = int(packet_components)
    if mask_payload is not None:
        payload = mask_payload.components[index]
        return all(mask_payload.components[index + component] is payload for component in range(packet_components))
    if mask_components is not None:
        mask = mask_components[index]
        return all(mask_components[index + component] is mask for component in range(packet_components))
    return True


def _buffer_max_packet_elements(attrs):
    element_byte_width = int(attrs["element_byte_width"])
    if element_byte_width <= 0:
        return 1
    return max(1, 16 // element_byte_width)


def _buffer_packet_payload_is_legal(packet_elements, element_byte_width):
    packet_elements = int(packet_elements)
    element_byte_width = int(element_byte_width)
    if packet_elements <= 1 or element_byte_width <= 0:
        return False
    payload_bits = packet_elements * element_byte_width * 8
    return payload_bits <= 128 and (payload_bits == 16 or payload_bits % 32 == 0)


def _buffer_store_packet_components_are_scalar(
    state,
    index,
    packet_elements,
    value_components,
    offset_components,
):
    for element in range(int(packet_elements)):
        component_index = int(index) + int(element)
        if _simd_1d_vector_payload(state, value_components[component_index]) is not None:
            return False
        if _simd_1d_vector_payload(state, offset_components[component_index]) is not None:
            return False
    return True


def _buffer_store_packet_mask_is_supported(attrs, index, packet_elements, mask_components, mask_payload):
    index = int(index)
    packet_elements = int(packet_elements)
    if mask_payload is not None:
        payload = mask_payload.components[index]
        return (all(mask_payload.components[index + element] is payload for element in range(packet_elements))
                or _buffer_store_packet_uses_leading_mask(attrs, index, packet_elements))
    if mask_components is not None:
        mask = mask_components[index]
        return (all(mask_components[index + element] is mask for element in range(packet_elements))
                or _buffer_store_packet_uses_leading_mask(attrs, index, packet_elements))
    return True


def _buffer_store_packet_uses_leading_mask(attrs, index, packet_elements):
    mask_alignment = int(attrs.get("mask_alignment", 1))
    return mask_alignment >= int(packet_elements) and int(index) % int(packet_elements) == 0


def _buffer_store_access_group_allows_packet(attrs, index, packet_elements):
    access_elements = int(attrs.get("access_element_count", 1))
    if access_elements <= 1:
        return False
    access_packet_elements = _buffer_load_packet_elements(attrs)
    if int(packet_elements) > int(access_packet_elements):
        return False
    if int(index) % int(packet_elements):
        return False
    return int(index) % access_elements + int(packet_elements) <= access_elements


def _buffer_store_vector_payload_access_group_allows_packet(attrs, index, packet_components, vector_elements):
    access_elements = int(attrs.get("access_element_count", 1))
    packet_elements = int(packet_components) * int(vector_elements)
    logical_index = int(index) * int(vector_elements)
    if access_elements <= 1:
        return False
    if packet_elements > access_elements:
        return False
    if logical_index % packet_elements:
        return False
    return logical_index % access_elements + packet_elements <= access_elements


def _ixsimpl_proves(predicate):
    try:
        return int(predicate.simplify()) == 1
    except (TypeError, ValueError):
        return False


def _emit_buffer_store_vector_packet(
    state,
    op,
    attrs,
    index,
    packet_elements,
    buffer_base,
    value_components,
    offset_component,
    mask_component,
    element_type,
    lane_width,
    mask_mode,
    *,
    dependency=None,
):
    cache = _direct_buffer_store_cache_attr(state, attrs, op)
    packet_elements = int(packet_elements)
    packet_type = state.dsl.simd_type(
        state.dsl.vector_type(packet_elements, element_type),
        width=int(lane_width),
    )
    packet_value = state.dsl.wave.PackOp(
        packet_type,
        [value_components[int(index) + element] for element in range(packet_elements)],
    ).result

    def emit_store():
        return _emit_contiguous_store(
            state,
            packet_value,
            offset_component,
            buffer_base,
            lane_width,
            dependency=dependency,
            cache=cache,
        )

    if mask_component is None:
        return emit_store()
    if mask_mode != "exec_where":
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_STORE_MASK",
            STAGE,
            f"unsupported buffer_store mask mode {mask_mode}",
            target_op_id=op.target_op_id,
        )
    return _emit_masked_effect_region(state, mask_component, emit_store)


def _emit_buffer_store_vector_payload_packet(
    state,
    op,
    attrs,
    index,
    packet_components,
    vector_elements,
    buffer_base,
    value_components,
    offset_component,
    mask_component,
    element_type,
    lane_width,
    mask_mode,
    *,
    dependency=None,
):
    cache = _direct_buffer_store_cache_attr(state, attrs, op)
    packet_components = int(packet_components)
    vector_elements = int(vector_elements)
    packet_elements = packet_components * vector_elements
    if packet_components == 1:
        packet_value = value_components[int(index)]
    else:
        packet_type = state.dsl.simd_type(
            state.dsl.vector_type(packet_elements, element_type),
            width=int(lane_width),
        )
        packet_value = state.dsl.wave.PackOp(
            packet_type,
            [value_components[int(index) + component] for component in range(packet_components)],
        ).result

    def emit_store():
        return _emit_contiguous_store(
            state,
            packet_value,
            offset_component,
            buffer_base,
            lane_width,
            dependency=dependency,
            cache=cache,
        )

    if mask_component is None:
        return emit_store()
    if mask_mode != "exec_where":
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_STORE_MASK",
            STAGE,
            f"unsupported buffer_store mask mode {mask_mode}",
            target_op_id=op.target_op_id,
        )
    return _emit_masked_effect_region(state, mask_component, emit_store)


def _emit_buffer_store_component(
    state,
    op,
    attrs,
    buffer_base,
    lane_width,
    value_component,
    offset_component,
    mask_component,
    mask_mode,
    *,
    dependency=None,
):
    cache = _direct_buffer_store_cache_attr(state, attrs, op)

    def emit_store():
        return _emit_contiguous_store(
            state,
            value_component,
            offset_component,
            buffer_base,
            lane_width,
            dependency=dependency,
            cache=cache,
        )

    if mask_component is None:
        return emit_store()
    if mask_mode != "exec_where":
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_STORE_MASK",
            STAGE,
            f"unsupported buffer_store mask mode {mask_mode}",
            target_op_id=op.target_op_id,
        )
    return _emit_masked_effect_region(state, mask_component, emit_store)


def _buffer_inactive_element_offset(state, attrs, lane_width):
    inactive_byte_offset = int(attrs.get("inactive_byte_offset", 1 << 31))
    element_byte_width = int(attrs["element_byte_width"])
    if element_byte_width <= 0 or inactive_byte_offset % element_byte_width:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_STORE_MASK",
            STAGE,
            "buffer_store inactive byte offset must align to element size",
        )
    element_offset = inactive_byte_offset // element_byte_width
    return state.builder.splat(
        state.builder.constant(state.dsl.index_type(), element_offset),
        state.dsl.index_type(),
        lane_width,
    )


def _simd_1d_vector_payload(state, value):
    try:
        simd = state.dsl.SimdType(value.type)
        vector = state.dsl.VectorType(simd.element_type)
    except Exception:
        return None
    shape = tuple(int(dim) for dim in vector.shape)
    if len(shape) != 1:
        return None
    return int(shape[0]), vector.element_type, int(simd.width)


def _mma_packet_payload_type(state, attrs, element_type, lane_width, op):
    registers = int(attrs.get("registers", 0))
    if registers <= 0:
        fail(
            "TLXW_EMIT_FRAGMENT_TYPE",
            STAGE,
            "MMA packet payload requires a positive register count",
            target_op_id=op.target_op_id,
        )
    return state.dsl.simd_type(
        state.dsl.vector_type(registers, _scalar_type(state.dsl, element_type)),
        width=int(lane_width or attrs.get("lane_width", 64) or 64),
    )


def _emit_buffer_load(state, op):
    attrs = target_ir.attrs_dict(op)
    has_mask = bool(attrs["has_mask"])
    mask_operand_count = int(has_mask)
    other_operand_count = int(bool(attrs["has_other"]))
    operand_count = (
        2
        + mask_operand_count
        + other_operand_count
    )
    if len(op.operands) != operand_count:
        fail(
            "TLXW_EMIT_OPERAND_COUNT",
            STAGE,
            f"expected {operand_count} operands, got {len(op.operands)}",
            target_op_id=op.target_op_id,
        )
    source_base = _require_value(state, op.operands[0], op)
    offsets = _require_value(state, op.operands[1], op)
    operand_index = 2
    masks = (
        _require_value(state, op.operands[operand_index], op)
        if has_mask else None
    )
    operand_index += mask_operand_count
    other = (
        _require_value(state, op.operands[operand_index], op)
        if attrs["has_other"] else None
    )
    operand_index += other_operand_count
    component_count = int(attrs["component_count"])
    access_component_count = int(
        attrs.get("access_component_count", component_count)
    )
    offset_components = _as_components(offsets)
    mask_components = (None if masks is None else _as_mask_predicate_components(
        state,
        masks,
        access_component_count,
        int(attrs["lane_width"]),
        op,
    ))
    other_components = None if other is None else _as_components(other)
    if len(offset_components) != access_component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_load offset component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    if mask_components is not None and len(mask_components) != access_component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_load mask component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    if other_components is not None and len(other_components) not in (1, component_count):
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_load other component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    if other_components is not None and mask_components is None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD_OTHER",
            STAGE,
            "buffer_load other requires a mask",
            target_op_id=op.target_op_id,
        )
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    lane_width = int(attrs["lane_width"])
    mask_mode = attrs.get("mask_mode", "exec_where" if attrs["has_mask"] else "none")
    buffer_base = _ptr_cast(
        state,
        source_base,
        state.dsl.ptr_type(element_type, state.dsl.global_address_space()),
    )
    load_offset_components = offset_components
    if attrs.get("result_value_mode") in {"mma_packet_payload", "register_vector_payload"}:
        state.values[result_id] = _emit_buffer_load_mma_packet_payload(
            state,
            op,
            attrs,
            buffer_base,
            load_offset_components,
            mask_components,
            other_components,
            element_type,
            target_type,
            lane_width,
            access_component_count,
            mask_mode,
        )
        return
    result_type = _wave_type(state.dsl, target_type)
    packet_elements = _buffer_load_packet_elements(attrs)
    preserve_raw_layout_packets = (
        attrs.get("result_value_mode") == "raw_layout_vector_packets"
    )
    if preserve_raw_layout_packets:
        _require_buffer_load_raw_layout_packets(
            state,
            op,
            attrs,
            packet_elements,
            component_count,
            mask_components,
            other_components,
        )
    preserve_packets = preserve_raw_layout_packets or _buffer_load_preserves_vector_packets(
        state,
        op,
        attrs,
        packet_elements,
        component_count,
        mask_components,
        other_components,
    )
    loaded_components = []
    loaded_packets = []
    index = 0
    while index < component_count:
        if _can_vectorize_buffer_load_packet(
                attrs,
                index,
                packet_elements,
                component_count,
                mask_components,
                other_components,
        ):
            loaded_packet = _emit_buffer_load_vector_packet(
                state,
                op,
                attrs,
                index,
                packet_elements,
                buffer_base,
                load_offset_components,
                mask_components,
                element_type,
                result_type,
                lane_width,
                mask_mode,
                preserve_packet=preserve_packets,
                preserve_raw_packet=preserve_raw_layout_packets,
            )
            loaded_values, loaded_token = loaded_packet
            if preserve_packets:
                loaded_packets.extend(loaded_values)
            else:
                loaded_components.extend(loaded_values)
            index += packet_elements
            continue
        loaded_components.append(
            _emit_buffer_load_scalar_component(
                state,
                op,
                attrs,
                index,
                buffer_base,
                load_offset_components,
                mask_components,
                other_components,
                result_type,
                mask_mode,
            ))
        index += 1
    if preserve_raw_layout_packets:
        state.values[result_id] = _RawLayoutVectorPacketPayload(
            tuple(loaded_packets),
            int(packet_elements),
            int(component_count),
            int(attrs["result_element_bit_width"]),
        )
    elif preserve_packets:
        state.values[result_id] = _VectorPacketPayload(
            tuple(loaded_packets),
            int(packet_elements),
            int(component_count),
        )
    else:
        state.values[result_id] = _pack_components(tuple(loaded_components))


def _require_buffer_load_raw_layout_packets(
    state,
    op,
    attrs,
    packet_elements,
    component_count,
    mask_components,
    other_components,
):
    if other_components is not None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "raw layout-packet buffer_load does not support an other operand",
            target_op_id=op.target_op_id,
        )
    requested_packet_width = int(attrs.get("result_packet_width", 0))
    element_bit_width = int(attrs.get("result_element_bit_width", 0))
    if (
        requested_packet_width <= 1
        or int(packet_elements) != requested_packet_width
        or int(component_count) % requested_packet_width
        or element_bit_width <= 0
        or 32 % element_bit_width
        or requested_packet_width * element_bit_width % 32
    ):
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "raw layout-packet buffer_load requires a uniform whole-dword packet tiling",
            target_op_id=op.target_op_id,
        )
    if not all(
        _can_vectorize_buffer_load_packet(
            attrs,
            index,
            requested_packet_width,
            component_count,
            mask_components,
            other_components,
        )
        for index in range(0, int(component_count), requested_packet_width)
    ):
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "raw layout-packet buffer_load requested non-contiguous packets",
            target_op_id=op.target_op_id,
        )


def _buffer_load_preserves_vector_packets(
    state,
    op,
    attrs,
    packet_elements,
    component_count,
    mask_components,
    other_components,
):
    if attrs.get("result_value_mode") != "vector_packets":
        return False
    if other_components is not None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "vector-packet buffer_load results do not support an other operand",
            target_op_id=op.target_op_id,
        )
    requested_packet_width = int(attrs.get("result_packet_width", 0))
    if requested_packet_width <= 1:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "vector-packet buffer_load requires result_packet_width",
            target_op_id=op.target_op_id,
        )
    packet_elements = int(packet_elements)
    component_count = int(component_count)
    if requested_packet_width != packet_elements:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "vector-packet buffer_load packet width does not match access width",
            target_op_id=op.target_op_id,
        )
    if packet_elements <= 1 or component_count % packet_elements:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "vector-packet buffer_load requires a uniform packet tiling",
            target_op_id=op.target_op_id,
        )
    if not all(
            _can_vectorize_buffer_load_packet(
                attrs,
                index,
                packet_elements,
                component_count,
                mask_components,
                other_components,
            ) for index in range(0, component_count, packet_elements)):
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "vector-packet buffer_load requested non-contiguous packets",
            target_op_id=op.target_op_id,
        )
    return True


def _emit_buffer_load_mma_packet_payload(
    state,
    op,
    attrs,
    buffer_base,
    offset_components,
    mask_components,
    other_components,
    element_type,
    target_type,
    lane_width,
    access_component_count,
    mask_mode,
):
    result_value_mode = attrs.get("result_value_mode")
    if (result_value_mode == "mma_packet_payload"
            and target_type.representation not in _MMA_PACKET_REPRESENTATIONS):
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "MMA packet payload mode requires an MMA packet result type",
            target_op_id=op.target_op_id,
        )
    if (result_value_mode == "register_vector_payload"
            and target_type.representation not in {"simd", "simd_tuple"}):
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "register vector payload mode requires a SIMD result type",
            target_op_id=op.target_op_id,
        )
    if other_components is not None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "MMA packet buffer_load does not support other operands",
            target_op_id=op.target_op_id,
        )
    supported_mask_mode = "exec_where" if mask_components is not None else "none"
    if mask_mode != supported_mask_mode:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD_MASK",
            STAGE,
            f"unsupported fragment buffer_load mask mode {mask_mode}",
            target_op_id=op.target_op_id,
        )
    component_count = int(attrs["component_count"])
    registers = int(attrs.get("registers", 0))
    if registers <= 1:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "MMA packet buffer_load requires multiple registers",
            target_op_id=op.target_op_id,
        )
    if int(attrs.get("access_element_count", 1)) < registers:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "MMA packet buffer_load requires contiguous access "
            "covering the payload registers",
            target_op_id=op.target_op_id,
        )
    if int(access_component_count) != component_count * registers:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "MMA packet access components must cover every register",
            target_op_id=op.target_op_id,
        )
    scalar_component_type = state.dsl.simd_type(
        _scalar_type(state.dsl, attrs["element_type"]),
        width=int(lane_width),
    )
    payload_type = state.dsl.simd_type(
        state.dsl.vector_type(registers, element_type),
        width=int(lane_width),
    )
    payloads = []
    for component in range(component_count):
        index = component * registers
        if _can_vectorize_buffer_load_packet(
            attrs,
            index,
            registers,
            access_component_count,
            mask_components,
            other_components,
        ):
            loaded_values, _token = _emit_buffer_load_vector_packet(
                state,
                op,
                attrs,
                index,
                registers,
                buffer_base,
                offset_components,
                mask_components,
                element_type,
                scalar_component_type,
                lane_width,
                mask_mode,
                preserve_packet=True,
            )
            payloads.extend(loaded_values)
            continue
        elements = tuple(
            _emit_buffer_load_scalar_component(
                state,
                op,
                attrs,
                index + element,
                buffer_base,
                offset_components,
                mask_components,
                other_components,
                scalar_component_type,
                mask_mode,
            )
            for element in range(registers)
        )
        payloads.append(state.dsl.wave.PackOp(payload_type, elements).result)
    return _pack_components(tuple(payloads))


def _buffer_load_packet_elements(attrs):
    access_elements = int(attrs.get("access_element_count", 1))
    element_byte_width = int(attrs["element_byte_width"])
    if access_elements <= 1 or element_byte_width <= 0:
        return 1
    max_elements = max(1, 16 // element_byte_width)
    packet_elements = min(access_elements, max_elements)
    while packet_elements > 1:
        payload_bits = packet_elements * element_byte_width * 8
        if (access_elements % packet_elements == 0 and payload_bits <= 128
                and (payload_bits == 16 or payload_bits % 32 == 0)):
            return packet_elements
        packet_elements -= 1
    return 1


def _can_vectorize_buffer_load_packet(
    attrs,
    index,
    packet_elements,
    component_count,
    mask_components,
    other_components,
):
    if packet_elements <= 1:
        return False
    if int(index) % int(packet_elements):
        return False
    if int(index) + int(packet_elements) > int(component_count):
        return False
    if other_components is not None:
        return False
    if not _buffer_load_access_group_allows_packet(attrs, index, packet_elements):
        return False
    if mask_components is None:
        return True
    packet_mask = mask_components[int(index)]
    return (all(mask_components[int(index) + element] is packet_mask for element in range(int(packet_elements)))
            or _buffer_load_packet_uses_leading_mask(attrs, index, packet_elements))


def _buffer_load_access_group_allows_packet(attrs, index, packet_elements):
    access_elements = int(attrs.get("access_element_count", 1))
    return _buffer_access_group_allows_packet(access_elements, index, packet_elements)


def _buffer_load_packet_uses_leading_mask(attrs, index, packet_elements):
    mask_alignment = int(attrs.get("mask_alignment", 1))
    return mask_alignment >= int(packet_elements) and int(index) % int(packet_elements) == 0


def _buffer_access_group_allows_packet(access_elements, index, packet_elements):
    access_elements = int(access_elements)
    if access_elements <= 1:
        return False
    if int(packet_elements) > access_elements:
        return False
    if int(index) % int(packet_elements):
        return False
    return int(index) % access_elements + int(packet_elements) <= access_elements


def _emit_buffer_load_scalar_component(
    state,
    op,
    attrs,
    index,
    buffer_base,
    offset_components,
    mask_components,
    other_components,
    result_type,
    mask_mode,
):
    cache = _direct_buffer_load_cache_attr(state, attrs, op)
    offset_component = offset_components[int(index)]
    result_fallback = None
    if other_components is not None:
        result_fallback = (other_components[0] if len(other_components) == 1 else other_components[int(index)])

    def emit_active_load():
        return _emit_symbolic_contiguous_gather(
            state,
            offset_component,
            buffer_base,
            result_type,
            _scalar_type(state.dsl, attrs["element_type"]),
            int(attrs["lane_width"]),
            int(attrs["element_byte_width"]),
            int(attrs["element_byte_width"]) * 8,
            cache=cache,
        )

    if mask_components is None:
        loaded, _token = emit_active_load()
        return loaded
    if mask_mode != "exec_where":
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD_MASK",
            STAGE,
            f"unsupported buffer_load mask mode {mask_mode}",
            target_op_id=op.target_op_id,
        )
    if result_fallback is not None:
        result_fallback = _memory_simd_component(
            state,
            result_fallback,
            attrs["element_type"],
            int(attrs["lane_width"]),
            op,
            [],
        )
    else:
        result_fallback = _zero_simd_value(
            state,
            result_type,
            attrs["element_type"],
            op,
        )
    loaded, _token = _emit_masked_memory_value_region(
        state,
        mask_components[int(index)],
        result_type,
        result_fallback,
        None,
        emit_active_load,
    )
    return loaded


def _emit_buffer_load_vector_packet(
    state,
    op,
    attrs,
    index,
    packet_elements,
    buffer_base,
    offset_components,
    mask_components,
    element_type,
    component_type,
    lane_width,
    mask_mode,
    *,
    preserve_packet=False,
    preserve_raw_packet=False,
):
    cache = _direct_buffer_load_cache_attr(state, attrs, op)
    packet_elements = int(packet_elements)
    if preserve_raw_packet:
        packet_bits = packet_elements * int(attrs["element_byte_width"]) * 8
        load_type = _raw_register_packet_type(state, packet_bits, lane_width, op)
        packet_element_type = state.dsl.i32()
        packet_element_bit_width = 32
        packet_component_type = state.dsl.simd_type(
            packet_element_type,
            width=int(lane_width),
        )
        packet_component_count = packet_bits // 32
    else:
        load_type = state.dsl.simd_type(
            state.dsl.vector_type(packet_elements, element_type),
            width=int(lane_width),
        )
        packet_element_type = element_type
        packet_element_bit_width = int(attrs["element_byte_width"]) * 8
        packet_component_type = component_type
        packet_component_count = packet_elements

    def emit_active_load():
        return _emit_symbolic_contiguous_gather(
            state,
            offset_components[int(index)],
            buffer_base,
            load_type,
            packet_element_type,
            lane_width,
            int(attrs["element_byte_width"]),
            packet_element_bit_width,
            cache=cache,
        )

    if mask_components is None:
        loaded, token = emit_active_load()
    else:
        if mask_mode != "exec_where":
            fail(
                "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD_MASK",
                STAGE,
                f"unsupported buffer_load mask mode {mask_mode}",
                target_op_id=op.target_op_id,
            )
        packet_mask = mask_components[int(index)]
        if packet_component_count == 1:
            inactive_value = _zero_simd_value(
                state,
                load_type,
                "i32" if preserve_raw_packet else attrs["element_type"],
                op,
            )
        else:
            inactive_value = _zero_vector_simd_value(
                state,
                load_type,
                packet_component_type,
                "i32" if preserve_raw_packet else attrs["element_type"],
                packet_component_count,
                op,
            )
        loaded, token = _emit_masked_memory_value_region(
            state,
            packet_mask,
            load_type,
            inactive_value,
            None,
            emit_active_load,
        )
    if preserve_packet:
        return (loaded, ), token
    return tuple(
        state.dsl.wave.ExtractOp(
            component_type,
            loaded,
            element,
        ).result for element in range(packet_elements)), token


def _zero_vector_simd_value(
    state,
    result_type,
    component_type,
    element_type,
    component_count,
    op,
):
    zero = _zero_simd_value(state, component_type, element_type, op)
    return state.dsl.wave.PackOp(
        result_type,
        [zero for _ in range(int(component_count))],
    ).result


def _emit_store(state, op):
    attrs = target_ir.attrs_dict(op)
    operand_count = 3 if attrs["has_mask"] else 2
    operands = _operand_values(state, op, operand_count)
    ptrs, values = operands[:2]
    masks = operands[2] if attrs["has_mask"] else None
    component_count = int(attrs["component_count"])
    ptr_components = _as_components(ptrs)
    value_components = _broadcast_component(state, values, component_count, op)
    splat_cache = []
    value_components = tuple(
        _memory_simd_component(
            state,
            value_component,
            attrs["element_type"],
            int(attrs["lane_width"]),
            op,
            splat_cache,
        ) for value_component in value_components)
    mask_components = None
    if masks is not None:
        mask_components = _as_mask_predicate_components(
            state,
            masks,
            component_count,
            int(attrs["lane_width"]),
            op,
        )
    if len(ptr_components) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "store pointer component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    mask_mode = attrs.get("mask_mode", "exec_where" if attrs["has_mask"] else "none")
    for index, (ptr_component, value_component) in enumerate(zip(ptr_components, value_components)):
        if mask_components is None:
            state.builder.store(value_component, ptr_component)
            continue
        if mask_mode != "exec_where":
            fail(
                "TLXW_EMIT_UNSUPPORTED_STORE_MASK",
                STAGE,
                f"unsupported store mask mode {mask_mode}",
                target_op_id=op.target_op_id,
            )
        _emit_masked_effect_region(
            state,
            mask_components[index],
            lambda value_component=value_component, ptr_component=ptr_component: state.builder.store(
                value_component,
                ptr_component,
            ),
        )


def _emit_load(state, op):
    attrs = target_ir.attrs_dict(op)
    operand_count = 1 + int(bool(attrs["has_mask"])) + int(bool(attrs["has_other"]))
    operands = _operand_values(state, op, operand_count)
    ptrs = operands[0]
    operand_index = 1
    masks = None
    if attrs["has_mask"]:
        masks = operands[operand_index]
        operand_index += 1
    other = operands[operand_index] if attrs["has_other"] else None
    component_count = int(attrs["component_count"])
    ptr_components = _as_components(ptrs)
    mask_components = None
    if masks is not None:
        mask_components = _as_mask_predicate_components(
            state,
            masks,
            component_count,
            int(attrs["lane_width"]),
            op,
        )
    other_components = None
    if other is not None:
        other_components = _broadcast_component(state, other, component_count, op)
        splat_cache = []
        other_components = tuple(
            _memory_simd_component(
                state,
                other_component,
                attrs["element_type"],
                int(attrs["lane_width"]),
                op,
                splat_cache,
            ) for other_component in other_components)
    if len(ptr_components) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "load pointer component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    if other_components is not None and mask_components is None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOAD_OTHER",
            STAGE,
            "load other requires a mask",
            target_op_id=op.target_op_id,
        )
    result_id = _single_result(op)
    result_type = _wave_type(state.dsl, state.target_program.values[result_id].type)
    mask_mode = attrs.get("mask_mode", "exec_where" if attrs["has_mask"] else "none")
    loaded_components = []
    for index, ptr_component in enumerate(ptr_components):
        if mask_components is None:
            loaded, _token = state.builder.load(ptr_component, result_type)
        else:
            if mask_mode != "exec_where":
                fail(
                    "TLXW_EMIT_UNSUPPORTED_LOAD_MASK",
                    STAGE,
                    f"unsupported load mask mode {mask_mode}",
                    target_op_id=op.target_op_id,
                )

            def emit_active_load(ptr_component=ptr_component):
                return state.builder.load(ptr_component, result_type)

            other_component = (None if other_components is None else other_components[index])
            fallback = (
                other_component
                if other_component is not None
                else _zero_simd_value(
                    state,
                    result_type,
                    attrs["element_type"],
                    op,
                )
            )
            loaded, _token = _emit_masked_memory_value_region(
                state,
                mask_components[index],
                result_type,
                fallback,
                None,
                emit_active_load,
            )
        loaded_components.append(loaded)
    state.values[result_id] = _pack_components(tuple(loaded_components))


def _memory_simd_component(state, value, element_type, lane_width, op, splat_cache):
    if _is_simd_value(state.dsl, value):
        return value
    scalar_type = _scalar_type(state.dsl, element_type)
    if str(value.type) != str(scalar_type):
        fail(
            "TLXW_EMIT_UNSUPPORTED_MEMORY_VALUE",
            STAGE,
            f"memory value has type {value.type}, expected {scalar_type}",
            target_op_id=op.target_op_id,
        )
    return _reuse_component_result(
        splat_cache,
        (value, ),
        lambda: state.builder.splat(value, scalar_type, int(lane_width)),
    )


def _emit_masked_effect_region(state, condition, emit_body):
    if _is_scalar_i1_value(state, condition):
        with state.builder.if_(condition):
            emit_body()
        return
    with state.builder.where(condition):
        emit_body()


def _emit_masked_token_region(state, condition, inactive_token, emit_body):
    result_type = state.dsl.mem_token_type()
    if _is_scalar_i1_value(state, condition):
        with state.builder.if_(condition, [result_type], otherwise=True) as ifop:
            state.builder.yield_([emit_body()])
            with ifop.otherwise():
                state.builder.yield_([inactive_token])
        return ifop.results[0]
    with state.builder.where(condition, [result_type]) as where:
        state.builder.yield_([emit_body()])
    with where.otherwise():
        state.builder.yield_([inactive_token])
    return where.results[0]


def _emit_masked_value_region(
    state,
    condition,
    result_type,
    inactive_value,
    emit_body,
):
    if _is_scalar_i1_value(state, condition):
        with state.builder.if_(condition, [result_type], otherwise=True) as ifop:
            state.builder.yield_([emit_body()])
            with ifop.otherwise():
                state.builder.yield_([inactive_value])
        return ifop.results[0]
    with state.builder.where(condition, [result_type]) as where:
        state.builder.yield_([emit_body()])
    with where.otherwise():
        state.builder.yield_([inactive_value])
    return where.results[0]


def _emit_masked_memory_value_region(
    state,
    condition,
    result_type,
    inactive_value,
    dependency,
    emit_body,
):
    token_type = state.dsl.mem_token_type()
    inactive_token = dependency or state.builder.token()
    result_types = [result_type, token_type]
    if _is_scalar_i1_value(state, condition):
        with state.builder.if_(
            condition, result_types, otherwise=True
        ) as ifop:
            value, token = emit_body()
            state.builder.yield_([value, token])
            with ifop.otherwise():
                state.builder.yield_([inactive_value, inactive_token])
        return tuple(ifop.results)
    with state.builder.where(condition, result_types) as where:
        value, token = emit_body()
        state.builder.yield_([value, token])
    with where.otherwise():
        state.builder.yield_([inactive_value, inactive_token])
    return tuple(where.results)


def _zero_simd_value(state, result_type, element_type, op):
    return _wave_constant(
        state,
        result_type,
        _scalar_type(state.dsl, element_type),
        element_type,
        0,
        op,
    )


_TARGET_EMITTERS = {
    "constant": _emit_constant,
    "affine_materialize": _emit_affine_materialize,
    "type_convert": _emit_type_convert,
    "binary": _emit_binary,
    "float_binary": _emit_float_binary,
    "float_unary": _emit_float_unary,
    "float_cast": _emit_float_cast,
    "cmpi": _emit_cmpi,
    "cmpi_select": _emit_cmpi_select,
    "maxsi": _emit_maxsi,
    "minsi": _emit_minsi,
    "assume": _emit_assume,
    "make_range": _emit_make_range,
    "splat": _emit_splat,
    "broadcast": _emit_broadcast,
    "addptr": _emit_addptr,
    "expand_dims": _emit_expand_dims,
    "program_id": _emit_program_id,
    "thread_id": _emit_thread_id,
    "barrier": _emit_barrier,
    "cond_barrier": _emit_cond_barrier,
    "set_priority": _emit_set_priority,
    "sched_barrier": _emit_sched_barrier,
    "for_loop": _emit_for_loop,
    "if": _emit_if,
    "select": _emit_select,
    "reduction": _emit_reduction,
    "local_alloc": _emit_local_alloc,
    "memdesc_index": _emit_memdesc_index,
    "memdesc_view": _emit_memdesc_view,
    "local_store": _emit_local_store,
    "local_load": _emit_local_load,
    "buffer_load_to_local": _emit_buffer_load_to_local,
    "local_load_mma_payload": _emit_local_load_mma_payload,
    "mma_packet_constant": _emit_mma_packet_constant,
    "mma": _emit_mma,
    "mma_scaled": _emit_mma_scaled,
    "mma_packet_truncf": _emit_mma_packet_truncf,
    "layout_convert": _emit_layout_convert,
    "buffer_store": _emit_buffer_store,
    "buffer_load": _emit_buffer_load,
    "store": _emit_store,
    "load": _emit_load,
    "token": _emit_token,
    "token_join": _emit_token_join,
    "issue_token": _emit_issue_token,
    "async_commit_group": _emit_async_commit_group,
    "async_wait": _emit_async_wait,
    "return": _emit_return,
}

_UNOWNED_TARGET_OPS = frozenset(_TARGET_EMITTERS) - domains.all_target_ops()
if _UNOWNED_TARGET_OPS:
    raise RuntimeError(f"unsupported target op domains: {sorted(_UNOWNED_TARGET_OPS)}")


def _scalar_constant(state, scalar_type, element_type, literal, op):
    if element_type == "i1":
        return state.dsl.arith.ConstantOp(
            scalar_type,
            state.ir.IntegerAttr.get(scalar_type, int(_literal_bool(literal, op))),
        ).result
    _require_numeric_literal(literal, op)
    return state.builder.constant(scalar_type, literal)


def _wave_constant(state, result_type, scalar_type, element_type, literal, op):
    if element_type == "i1":
        attr = state.ir.IntegerAttr.get(scalar_type, int(_literal_bool(literal, op)))
    elif _is_float_element(element_type):
        _require_numeric_literal(literal, op)
        attr = state.ir.FloatAttr.get(scalar_type, float(literal))
    else:
        _require_numeric_literal(literal, op)
        attr = state.ir.IntegerAttr.get(scalar_type, int(literal))
    return state.dsl.wave.ConstantOp(result_type, attr).result


def _wave_mask_constant(state, result_type, value):
    return state.dsl.wave.ConstantOp(
        result_type,
        state.ir.Attribute.parse("true" if value else "false"),
    ).result


def _reuse_component_result(reused, operands, create):
    operands = tuple(operands)
    existing = _find_reused_component_result(reused, operands)
    if existing is not None:
        return existing
    value = create()
    reused.append((operands, value))
    return value


def _find_reused_component_result(reused, operands):
    for existing_operands, value in reused:
        if _same_component_operands(existing_operands, operands):
            return value
    return None


def _same_component_operands(lhs, rhs):
    return len(lhs) == len(rhs) and all(
        lhs_operand is rhs_operand
        for lhs_operand, rhs_operand in zip(lhs, rhs)
    )


def _literal_bool(literal, op):
    if isinstance(literal, bool):
        return literal
    if isinstance(literal, int) and literal in (0, 1):
        return bool(literal)
    if isinstance(literal, str) and literal.lower() in {"true", "false"}:
        return literal.lower() == "true"
    fail(
        "TLXW_EMIT_UNSUPPORTED_CONSTANT",
        STAGE,
        f"cannot emit {literal!r} as an i1/mask constant",
        target_op_id=op.target_op_id,
    )


def _require_numeric_literal(literal, op):
    if isinstance(literal, bool) or not isinstance(literal, (int, float)):
        fail(
            "TLXW_EMIT_UNSUPPORTED_CONSTANT",
            STAGE,
            f"cannot emit non-numeric constant literal {literal!r}",
            target_op_id=op.target_op_id,
        )


def _is_float_element(element_type):
    return element_type in {"f16", "bf16", "f32", "f64"}


def _packet_coordinate_values(
    state,
    component,
    lane,
    component_thread_count,
    packet_elements,
    shape,
    packet_order,
):
    linear = _simd_binary_const(
        state,
        "muli",
        lane,
        int(packet_elements),
        int(state.dsl.SimdType(lane.type).width),
        nsw=_LAYOUT_MATH_NSW,
    )
    constant = int(component) * int(component_thread_count) * int(packet_elements)
    if constant:
        linear = _simd_binary_const(
            state,
            "addi",
            linear,
            constant,
            int(state.dsl.SimdType(lane.type).width),
            nsw=_LAYOUT_MATH_NSW,
        )
    lane_width = int(state.dsl.SimdType(lane.type).width)
    coords = [None] * len(shape)
    remainder = linear
    for dim in packet_order:
        extent = int(shape[int(dim)])
        coord = _simd_binary_const(state, "remui", remainder, extent, lane_width)
        coords[int(dim)] = coord
        remainder = _simd_binary_const(state, "divui", remainder, extent, lane_width)
    return tuple(coords)


def _packet_source_offset_index_expr(
    state,
    component,
    component_count,
    wi,
    component_thread_count,
    packet_elements,
    shape,
    packet_order,
    encoded_terms,
    scalar_values,
    op,
    encoded_range,
    scalar_component_sources,
    *,
    coordinate_mode="ordered_linear",
    linear_component_bases=(),
):
    wi_sym = state.dsl.sym("wi")
    scalar_symbols = tuple(state.dsl.sym(f"s{index}") for index, _ in enumerate(scalar_values))
    coords = _packet_coordinate_exprs(
        state,
        int(component),
        wi_sym,
        int(component_thread_count),
        int(packet_elements),
        shape,
        packet_order,
        coordinate_mode=coordinate_mode,
        linear_component_bases=linear_component_bases,
        op=op,
    )
    expr = _affine_offset_expr(state, encoded_terms, coords, scalar_symbols, op)
    bindings = {wi_sym: wi}
    bindings.update({
        symbol: _mapped_affine_component_binding_value(
            state,
            value,
            scalar_component_sources,
            index,
            component_count,
            component,
            op,
        )
        for index, (symbol, value) in enumerate(zip(scalar_symbols, scalar_values))
    })
    assumptions = _index_expr_range_assumptions(expr, encoded_range)
    return state.builder.index_expr(expr, bindings=bindings, assumptions=assumptions)


def _mapped_affine_component_binding_value(
    state,
    value,
    scalar_component_sources,
    scalar_index,
    component_count,
    component,
    op,
):
    if scalar_component_sources:
        if int(scalar_index) >= len(scalar_component_sources):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "affine scalar component source count does not match "
                "affine scalar operands",
                target_op_id=op.target_op_id,
            )
        sources = tuple(int(source) for source in scalar_component_sources[int(scalar_index)])
        if len(sources) != int(component_count):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "affine scalar component source mapping does not match "
                "packet component count",
                target_op_id=op.target_op_id,
            )
        source = int(sources[int(component)])
        components = _value_components(state, value, op)
        if source < 0 or source >= len(components):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                f"affine scalar component source {source} is out of range",
                target_op_id=op.target_op_id,
            )
        return components[source]
    return _affine_component_binding_value(
        state,
        value,
        component_count,
        component,
        op,
    )


def _packet_coordinate_exprs(
    state,
    component,
    wi,
    component_thread_count,
    packet_elements,
    shape,
    packet_order,
    *,
    coordinate_mode="ordered_linear",
    linear_component_bases=(),
    op=None,
):
    linear = wi
    if int(packet_elements) != 1:
        linear *= int(packet_elements)
    constant = int(component) * int(component_thread_count) * int(packet_elements)
    if constant:
        linear += constant
    linear_lower = constant
    linear_upper = constant + (int(component_thread_count) - 1) * int(packet_elements)
    if coordinate_mode == "physical_linear_component":
        return _linear_component_coordinate_exprs(
            state,
            linear,
            tuple(tuple(int(value) for value in basis) for basis in linear_component_bases),
            len(shape),
            op,
        )
    if coordinate_mode != "ordered_linear":
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"unsupported packet DMA source coordinate mode {coordinate_mode}",
            target_op_id=None if op is None else op.target_op_id,
        )
    coords = [None] * len(shape)
    remainder = linear
    for dim in packet_order:
        extent = int(shape[int(dim)])
        if linear_lower >= 0 and linear_upper < extent:
            coords[int(dim)] = remainder
        else:
            coords[int(dim)] = state.dsl.mod(remainder, extent)
        remainder = state.dsl.floor(remainder / extent)
        linear_lower //= extent
        linear_upper //= extent
    return tuple(coords)


def _linear_component_coordinate_exprs(state, linear, bases, rank, op):
    coords = [state.dsl.sym_ctx.int_(0) for _ in range(int(rank))]
    for bit, basis in enumerate(tuple(bases)):
        if len(basis) != int(rank):
            fail(
                "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "packet DMA source linearComponent basis rank does not match "
                f"coordinate rank; basis={basis}, rank={rank}",
                target_op_id=None if op is None else op.target_op_id,
            )
        bit_value = state.dsl.mod(state.dsl.floor(linear / (1 << int(bit))), 2)
        for dim, value in enumerate(tuple(int(value) for value in basis)):
            if not int(value):
                continue
            term = bit_value
            if int(value) != 1:
                term = term * int(value)
            coords[int(dim)] = state.dsl.xor(coords[int(dim)], term)
    return tuple(coords)


def _packet_destination_offset_value(
    state,
    destination_offset,
    element_byte_width,
    destination_wave_coordinate,
    lane_width,
    destination_wave_stride_dwords,
    destination_wave_offset_coefficients_dwords,
    op,
):
    byte_offset = int(destination_offset) * int(element_byte_width)
    if byte_offset % 4:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "packet DMA destination offset must be dword aligned",
            target_op_id=op.target_op_id,
        )
    base_dwords = byte_offset // 4
    if destination_wave_coordinate is None:
        if base_dwords == 0:
            return None
        return state.builder.constant(state.dsl.i32(), int(base_dwords))
    wi_first = state.dsl.sym("wi_first")
    wave_id = state.dsl.floor(wi_first / int(lane_width))
    expr = state.dsl.sym_ctx.int_(int(base_dwords))
    coefficients = tuple(
        int(value) for value in destination_wave_offset_coefficients_dwords)
    if coefficients:
        for bit, coefficient in enumerate(coefficients):
            if not coefficient:
                continue
            bit_value = state.dsl.mod(
                state.dsl.floor(wave_id / (1 << int(bit))),
                2,
            )
            if coefficient != 1:
                bit_value = bit_value * int(coefficient)
            expr = expr + bit_value
    else:
        expr = expr + wave_id * int(destination_wave_stride_dwords)
    return state.builder.index_expr(
        expr,
        bindings={wi_first: destination_wave_coordinate},
    )


def _packet_destination_wave_coordinate_value(
    state,
    destination_wave_stride_dwords,
    destination_wave_offset_coefficients_dwords,
    lane,
    lane_width,
    component_thread_count,
    op,
):
    destination_wave_stride_dwords = int(destination_wave_stride_dwords)
    destination_wave_offset_coefficients_dwords = tuple(
        int(value) for value in destination_wave_offset_coefficients_dwords)
    if not destination_wave_stride_dwords and not destination_wave_offset_coefficients_dwords:
        return None
    cache_key = (
        "packet_destination_wave_offset",
        int(lane_width),
        int(component_thread_count),
        int(destination_wave_stride_dwords),
        destination_wave_offset_coefficients_dwords,
    )
    cached = state.wave_offset_i32_cache.get(cache_key)
    if cached is not None:
        return cached
    wave_first = state.builder.read_first(lane)
    wave_first = _assume_value_range(
        state,
        wave_first,
        (0, max(0,
                int(component_thread_count) - 1)),
        op,
    )
    state.wave_offset_i32_cache[cache_key] = wave_first
    return wave_first


def _scalar_bit_affine_offset(state, value, base, coefficients):
    offset = None
    if int(base):
        offset = state.builder.constant(state.dsl.i32(), int(base))
    for bit, coefficient in enumerate(coefficients):
        coefficient = int(coefficient)
        if coefficient == 0:
            continue
        bit_value = _scalar_binary_const_i32(state, "divui", value, 1 << bit)
        bit_value = _scalar_binary_const_i32(state, "remui", bit_value, 2)
        if coefficient != 1:
            bit_value = _scalar_binary_const_i32(
                state,
                "muli",
                bit_value,
                coefficient,
                nsw=_LAYOUT_MATH_NSW,
            )
        offset = _combine_optional_i32_offsets(
            state,
            offset,
            bit_value,
            nsw=_LAYOUT_MATH_NSW,
        )
    if offset is None:
        return state.builder.constant(state.dsl.i32(), 0)
    return offset


def _affine_offset_value(
    state,
    encoded_terms,
    coords,
    scalar_values,
    op,
    *,
    no_signed_wrap=False,
):
    lane_width = int(state.dsl.SimdType(coords[0].type).width) if coords else 64
    scalar_components = tuple(_splat_i32_scalar(state, value, lane_width, op) for value in scalar_values)
    result = state.builder.splat(
        state.builder.constant(state.dsl.i32(), 0),
        state.dsl.i32(),
        lane_width,
    )
    for encoded in encoded_terms:
        term = _affine_term_i32(
            state,
            encoded,
            coords,
            scalar_components,
            lane_width,
            op,
            no_signed_wrap=bool(no_signed_wrap),
        )
        result = state.builder.binary(
            state.dsl.BinaryKind.AddI,
            result,
            term,
            nsw=bool(no_signed_wrap),
        )
    return result


def _affine_offset_expr(state, encoded_terms, coords, scalar_symbols, op):
    result = 0
    for encoded in encoded_terms:
        result += _affine_term_expr(state, encoded, coords, scalar_symbols, op)
    if isinstance(result, int):
        return state.dsl.sym_ctx.int_(int(result))
    return result


def _affine_term_expr(state, encoded, coords, scalar_symbols, op):
    del state
    kind, coefficient, dim, slots = encoded
    coefficient = int(coefficient)
    dim = int(dim)
    slots = tuple(int(slot) for slot in slots)
    if kind == "const":
        return coefficient
    if kind == "dim":
        return coords[_require_dim_slot(dim, coords, op)] * coefficient
    if kind == "scalar":
        return scalar_symbols[_require_scalar_slot(slots, scalar_symbols, op)] * coefficient
    if kind == "dim_scalar":
        dim_value = coords[_require_dim_slot(dim, coords, op)]
        scalar_value = scalar_symbols[_require_scalar_slot(slots, scalar_symbols, op)]
        return dim_value * scalar_value * coefficient
    if kind == "scalar_product":
        if len(slots) != 2:
            fail(
                "TLXW_EMIT_BAD_AFFINE_TERM",
                STAGE,
                "scalar_product affine term requires two scalar operands",
                target_op_id=op.target_op_id,
            )
        lhs = scalar_symbols[_require_scalar_slot((slots[0], ), scalar_symbols, op)]
        rhs = scalar_symbols[_require_scalar_slot((slots[1], ), scalar_symbols, op)]
        return lhs * rhs * coefficient
    fail(
        "TLXW_EMIT_BAD_AFFINE_TERM",
        STAGE,
        f"unsupported affine term kind {kind}",
        target_op_id=op.target_op_id,
    )


def _index_expr_range_assumptions(expr, encoded_range):
    if encoded_range is None:
        return None
    lower, upper = encoded_range
    assumptions = []
    if lower is not None:
        assumptions.append(expr >= int(lower))
    if upper is not None:
        assumptions.append(expr <= int(upper))
    return assumptions or None


def _affine_term_i32(
    state,
    encoded,
    coords,
    scalar_components,
    lane_width,
    op,
    *,
    no_signed_wrap=False,
):
    kind, coefficient, dim, slots = encoded
    coefficient = int(coefficient)
    dim = int(dim)
    slots = tuple(int(slot) for slot in slots)
    if kind == "const":
        return state.builder.splat(
            state.builder.constant(state.dsl.i32(), coefficient),
            state.dsl.i32(),
            lane_width,
        )
    if kind == "dim":
        return _scale_simd_i32(
            state,
            coords[_require_dim_slot(dim, coords, op)],
            coefficient,
            lane_width,
            no_signed_wrap=bool(no_signed_wrap),
        )
    if kind == "scalar":
        return _scale_simd_i32(
            state,
            scalar_components[_require_scalar_slot(slots, scalar_components, op)],
            coefficient,
            lane_width,
            no_signed_wrap=bool(no_signed_wrap),
        )
    if kind == "dim_scalar":
        dim_value = coords[_require_dim_slot(dim, coords, op)]
        scalar_value = scalar_components[_require_scalar_slot(slots, scalar_components, op)]
        product = state.builder.binary(
            state.dsl.BinaryKind.MulI,
            dim_value,
            scalar_value,
            nsw=bool(no_signed_wrap),
        )
        return _scale_simd_i32(
            state,
            product,
            coefficient,
            lane_width,
            no_signed_wrap=bool(no_signed_wrap),
        )
    if kind == "scalar_product":
        if len(slots) != 2:
            fail(
                "TLXW_EMIT_BAD_AFFINE_TERM",
                STAGE,
                "scalar_product affine term requires two scalar operands",
                target_op_id=op.target_op_id,
            )
        lhs = scalar_components[_require_scalar_slot((slots[0], ), scalar_components, op)]
        rhs = scalar_components[_require_scalar_slot((slots[1], ), scalar_components, op)]
        product = state.builder.binary(
            state.dsl.BinaryKind.MulI,
            lhs,
            rhs,
            nsw=bool(no_signed_wrap),
        )
        return _scale_simd_i32(
            state,
            product,
            coefficient,
            lane_width,
            no_signed_wrap=bool(no_signed_wrap),
        )
    fail(
        "TLXW_EMIT_BAD_AFFINE_TERM",
        STAGE,
        f"unsupported affine term kind {kind}",
        target_op_id=op.target_op_id,
    )


def _scale_simd_i32(state, value, coefficient, lane_width, *, no_signed_wrap=False):
    coefficient = int(coefficient)
    if coefficient == 1:
        return value
    return _simd_binary_const(
        state,
        "muli",
        value,
        coefficient,
        lane_width,
        nsw=bool(no_signed_wrap),
    )


def _splat_i32_scalar(state, value, lane_width, op):
    if str(value.type) != str(state.dsl.i32()):
        fail(
            "TLXW_EMIT_BAD_AFFINE_TERM",
            STAGE,
            f"affine scalar operand must be i32, got {value.type}",
            target_op_id=op.target_op_id,
        )
    return state.builder.splat(value, state.dsl.i32(), lane_width)


def _scalar_binary_const_i32(state, operation, value, constant, *, nsw=False):
    constant = int(constant)
    if operation == "divui" and constant == 1:
        return value
    if operation == "remui" and constant == 1:
        return state.builder.constant(state.dsl.i32(), 0)
    operation_kind = _binary_kind(state.dsl, operation)
    rhs = state.builder.constant(state.dsl.i32(), constant)
    return state.builder.binary(operation_kind, value, rhs, nsw=bool(nsw))


def _combine_optional_i32_offsets(state, lhs, rhs, *, nsw=False):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return state.builder.binary(
        state.dsl.BinaryKind.AddI,
        lhs,
        rhs,
        nsw=bool(nsw),
    )


def _shared_pointer_with_dword_offset(
    state,
    base,
    offset,
    *,
    cache_key,
):
    """Build and reuse an invariant i32 LDS base plus a dword offset.

    Memdesc indexing supplies a dynamic ring offset separately from the
    invariant allocation/view/component address.  Keeping that association in
    the pointer tree lets Wave lower a DMA group to one M0 base plus immediate
    increments instead of carrying one SGPR address per request.
    """
    i32_shared = state.dsl.ptr_type(
        state.dsl.i32(),
        state.dsl.shared_address_space(),
    )
    if offset is None:
        offset = 0
    if type(offset) is int:
        offset_cache_key = ("constant", int(offset))
    else:
        offset_cache_key = tuple(cache_key)
    key = (
        id(base),
        str(i32_shared),
        *offset_cache_key,
    )
    cached = state.shared_pointer_offset_cache.get(key)
    if cached is not None:
        return cached
    base_i32 = _ptr_cast(state, base, i32_shared)
    if type(offset) is int and int(offset) == 0:
        state.shared_pointer_offset_cache[key] = base_i32
        return base_i32
    offset_value = (
        state.builder.constant(state.dsl.i32(), int(offset))
        if type(offset) is int
        else offset
    )
    result = state.builder.ptr_add(
        base_i32,
        offset_value,
        result_type=i32_shared,
    )
    state.shared_pointer_offset_cache[key] = result
    return result


def _require_dim_slot(dim, coords, op):
    if dim < 0 or dim >= len(coords):
        fail(
            "TLXW_EMIT_BAD_AFFINE_TERM",
            STAGE,
            f"affine term references dimension {dim}",
            target_op_id=op.target_op_id,
        )
    return dim


def _require_scalar_slot(slots, scalar_symbols, op):
    if len(slots) != 1 or slots[0] < 0 or slots[0] >= len(scalar_symbols):
        fail(
            "TLXW_EMIT_BAD_AFFINE_TERM",
            STAGE,
            f"affine term references scalar slots {slots}",
            target_op_id=op.target_op_id,
        )
    return slots[0]


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _target_lds_size(target_program):
    del target_program
    return 0


def _align_to(value, alignment):
    value = int(value)
    alignment = int(alignment)
    return ((value + alignment - 1) // alignment) * alignment


def _assume_value_range(state, value, encoded_range, op):
    if encoded_range is None:
        return value
    if len(encoded_range) != 2:
        fail(
            "TLXW_EMIT_BAD_ASSUME_RANGE",
            STAGE,
            "encoded range must contain lower and upper bounds",
            target_op_id=op.target_op_id,
        )
    lower, upper = encoded_range
    x = state.dsl.sym("x")
    assumptions = []
    if lower is not None:
        assumptions.append(x >= int(lower))
    if upper is not None:
        assumptions.append(x <= int(upper))
    if not assumptions:
        return value
    return state.builder.assume(value, tuple(assumptions), name="x")


def _bounded_index_edge(state, value, encoded_range, op):
    """Keep a bounded index edge visible to symbolic pointer lowering.

    ``wave.assume`` carries the whole-value range needed when correlated
    layout terms cannot be bounded independently.  A bare assume, however,
    hides the producing ``wave.index_expr`` from pointer-offset lowering and
    forces the symbolic coordinate expression to be expanded as ordinary
    runtime arithmetic.  The identity index expression makes the structural
    edge explicit while allowing ``wave-combine-pointer-offsets`` to compose
    the nested producer and its range predicates before address selection.
    """
    bounded = _assume_value_range(state, value, encoded_range, op)
    if bounded is value:
        return value
    symbol = state.dsl.sym("x")
    return state.builder.index_expr(
        symbol,
        bindings={symbol: bounded},
        assumptions=_index_expr_range_assumptions(symbol, encoded_range),
    )


def _ptr_cast(state, value, result_type):
    if str(value.type) == str(result_type):
        return value
    return state.dsl.wave.PtrCastOp(result_type, value).result


def _operand_values(state, op, count):
    if len(op.operands) != count:
        fail(
            "TLXW_EMIT_OPERAND_COUNT",
            STAGE,
            f"target op {op.kind} expected {count} operands, got {len(op.operands)}",
            target_op_id=op.target_op_id,
        )
    return tuple(_require_value(state, target_value_id, op) for target_value_id in op.operands)


def _require_value(state, target_value_id, op):
    if target_value_id not in state.values:
        op_kind = "<region-yield>" if op is None else op.kind
        op_id = None if op is None else op.target_op_id
        fail(
            "TLXW_EMIT_UNBOUND_VALUE",
            STAGE,
            f"target value {target_value_id} is not bound before {op_kind}",
            target_op_id=op_id,
            target_value_id=target_value_id,
        )
    return state.values[target_value_id]


def _single_result(op):
    if len(op.results) != 1:
        fail(
            "TLXW_EMIT_RESULT_COUNT",
            STAGE,
            f"target op {op.kind} expected one result, got {len(op.results)}",
            target_op_id=op.target_op_id,
        )
    return op.results[0]


def _component_count(state, target_value_id):
    return int(state.target_program.values[target_value_id].type.component_count)


def _as_components(value):
    return value if isinstance(value, tuple) else (value, )


def _value_components(state, value, op):
    if not isinstance(value, _VectorPacketPayload):
        return _as_components(value)
    packet_width = int(value.packet_width)
    logical_component_count = int(value.logical_component_count)
    if packet_width <= 1 or logical_component_count <= 0:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "vector packet payload has invalid shape",
            target_op_id=op.target_op_id,
        )
    components = []
    for packet in value.packets:
        payload = _simd_1d_vector_payload(state, packet)
        if payload is None:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "vector packet payload contains a non-vector SIMD value",
                target_op_id=op.target_op_id,
            )
        width, element_type, lane_width = payload
        if int(width) != packet_width:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "vector packet payload width does not match its shape",
                target_op_id=op.target_op_id,
            )
        component_type = state.dsl.simd_type(element_type, int(lane_width))
        for element in range(packet_width):
            components.append(state.dsl.wave.ExtractOp(
                component_type,
                packet,
                int(element),
            ).result)
    if len(components) != logical_component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "vector packet payload component count does not match its shape",
            target_op_id=op.target_op_id,
        )
    return tuple(components)


def _pack_components(components):
    return components[0] if len(components) == 1 else tuple(components)


def _as_mask_payload_components(state, value, count, lane_width, op):
    components, _predicates = _as_mask_payload_components_with_predicates(
        state,
        value,
        count,
        lane_width,
        op,
    )
    return components


def _as_mask_payload_components_with_predicates(state, value, count, lane_width, op):
    if isinstance(value, _I32MaskPayload):
        components = tuple(
            _mask_payload_component(state, component, lane_width)
            for component in value.components
        )
        predicates = value.predicates
    else:
        payloads = []
        predicate_components = []
        for component in _as_components(value):
            payload, predicate = _mask_to_i32_payload_with_predicate(
                state,
                component,
                lane_width,
            )
            payloads.append(payload)
            predicate_components.append(predicate)
        components = tuple(payloads)
        predicates = tuple(predicate_components)
    components = _broadcast_component_count(components, count, "mask payload", op)
    if predicates is not None:
        predicates = _broadcast_component_count(predicates, count, "mask predicate", op)
    return components, predicates


def _as_mask_predicate_components(state, value, count, lane_width, op):
    components = _mask_predicate_components(
        state,
        value,
        lane_width,
    )
    return _broadcast_component_count(components, count, "mask", op)


def _mask_predicate_components(state, value, lane_width):
    if isinstance(value, _I32MaskPayload):
        if value.predicates is not None:
            components = value.predicates
        else:
            reused = []
            components = tuple(
                _reuse_component_result(
                    reused,
                    (component, ),
                    lambda component=component: _i32_payload_to_mask(
                        state,
                        component,
                        lane_width,
                    ),
                )
                for component in value.components
            )
    else:
        components = _as_components(value)
    return tuple(components)


def _group_component_indices_by_identity(components):
    groups = []
    for index, component in enumerate(components):
        for existing_component, indices in groups:
            if component is existing_component:
                indices.append(index)
                break
        else:
            groups.append((component, [index]))
    return tuple(
        (component, tuple(indices))
        for component, indices in groups
    )


def _broadcast_component_count(components, count, description, op):
    components = tuple(components)
    if len(components) == count:
        return components
    if len(components) == 1:
        return components * int(count)
    fail(
        "TLXW_EMIT_COMPONENT_COUNT",
        STAGE,
        f"{description} component count does not match attrs",
        target_op_id=op.target_op_id,
    )


def _simd_i32_constant(state, lane_width, value):
    return state.builder.splat(
        state.builder.constant(state.dsl.i32(), int(value)),
        state.dsl.i32(),
        int(lane_width),
    )


def _simd_offset_value(state, value, lane_width):
    if _is_simd_value(state.dsl, value):
        return value
    return state.builder.splat(value, value.type, int(lane_width))


def _simd_i32_from_lane_plan(state, plan, lane_values, lane_width):
    lane_width = int(lane_width)
    plan = tuple(plan)
    lane_values = tuple(int(value) for value in lane_values)
    if not plan:
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            "lane-value SIMD materialization plan must not be empty",
        )
    if (
        len(lane_values) != lane_width
        or any(value < 0 or value >= lane_width for value in lane_values)
    ):
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            "lane-value SIMD materialization plan contains invalid lanes",
        )
    kind = str(plan[0])
    if kind == "bit_affine":
        coefficient_count = max(0, lane_width.bit_length() - 1)
        if (
            lane_width <= 0
            or lane_width & (lane_width - 1)
            or len(plan) != coefficient_count + 2
        ):
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "lane-value bit-affine plan does not match the wave width",
            )
        base = int(plan[1])
        coefficients = tuple(int(value) for value in plan[2:])
        planned_values = tuple(
            base
            + sum(
                coefficient
                for bit, coefficient in enumerate(coefficients)
                if lane & (1 << bit)
            )
            for lane in range(lane_width)
        )
        if planned_values != lane_values:
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "lane-value bit-affine plan does not match its source map",
            )
        lane_id = state.builder.lane_id(state.dsl.i32(), lane_width)
        return _bit_affine_thread_offset(
            state,
            lane_id,
            base,
            coefficients,
            lane_width,
        )
    if kind == "explicit":
        if tuple(int(value) for value in plan[1:]) != lane_values:
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "explicit lane-value plan does not match its source map",
            )
        return _simd_i32_from_lane_values(state, lane_values, lane_width)
    fail(
        "TLXW_EMIT_LAYOUT_REMAP",
        STAGE,
        f"unsupported lane-value SIMD materialization plan {kind!r}",
    )


def _simd_i1_mask_from_lane_plan(state, plan, lanes, lane_width):
    lane_width = int(lane_width)
    plan = tuple(plan)
    lanes = tuple(sorted({int(lane) for lane in lanes}))
    if not plan:
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            "lane-mask materialization plan must not be empty",
        )
    if not lanes or any(lane < 0 or lane >= lane_width for lane in lanes):
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            "lane-mask materialization plan contains invalid lanes",
        )

    kind = str(plan[0])
    if kind == "all":
        if len(plan) != 1 or lanes != tuple(range(lane_width)):
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "all-lanes mask plan does not match its lane set",
            )
        payload = _simd_i32_constant(state, lane_width, 1)
        return _lane_mask_from_i32_payload(state, payload, lane_width)

    lane_id = state.builder.lane_id(state.dsl.i32(), lane_width)
    if kind == "lane_bit":
        if len(plan) != 3:
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "lane-bit mask plan is malformed",
            )
        bit = int(plan[1])
        value = int(plan[2])
        if bit < 0 or bit >= max(0, lane_width.bit_length() - 1) or value not in (0, 1):
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "lane-bit mask plan does not match the wave width",
            )
        planned_lanes = tuple(
            lane
            for lane in range(lane_width)
            if ((lane >> bit) & 1) == value
        )
        if planned_lanes != lanes:
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "lane-bit mask plan does not match its lane set",
            )
        payload = _simd_binary_const(
            state,
            "shrui",
            lane_id,
            bit,
            lane_width,
        )
        payload = _simd_binary_const(
            state,
            "andi",
            payload,
            1,
            lane_width,
        )
        if not value:
            payload = _simd_binary_const(
                state,
                "xori",
                payload,
                1,
                lane_width,
            )
        return _lane_mask_from_i32_payload(state, payload, lane_width)

    if kind == "range":
        if len(plan) != 3:
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "lane-range mask plan is malformed",
            )
        begin = int(plan[1])
        end = int(plan[2])
        if not 0 <= begin < end <= lane_width or lanes != tuple(range(begin, end)):
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "lane-range mask plan does not match its lane set",
            )
        payload = _static_lane_membership_i32_payload(
            state,
            lane_id,
            lanes,
            lane_width,
        )
        return _lane_mask_from_i32_payload(state, payload, lane_width)

    if kind == "explicit":
        if tuple(int(lane) for lane in plan[1:]) != lanes:
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "explicit lane-mask plan does not match its lane set",
            )
        payload = _static_lane_membership_i32_payload(
            state,
            lane_id,
            lanes,
            lane_width,
        )
        return _lane_mask_from_i32_payload(state, payload, lane_width)
    fail(
        "TLXW_EMIT_LAYOUT_REMAP",
        STAGE,
        f"unsupported lane-mask materialization plan {kind!r}",
    )


def _static_lane_membership_i32_payload(state, lane_id, lanes, lane_width):
    """Materialize a static lane set as durable 0/1 VGPR data.

    Wave masks lower to condition-code state.  Keeping a static mask live over
    a structured loop makes that state unnecessarily fragile and also lets CSE
    merge predicates from otherwise independent layout remaps.  A compact
    lane-bit lookup keeps the invariant representation in ordinary SIMD data;
    callers materialize a mask only at the selection point.
    """
    lane_width = int(lane_width)
    if lane_width <= 0 or lane_width > 64:
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            "static lane membership supports wave widths from 1 through 64",
        )
    lane_set = frozenset(int(lane) for lane in lanes)
    words = []
    for word_index in range((lane_width + 31) // 32):
        word = sum(
            1 << (lane - 32 * word_index)
            for lane in lane_set
            if 32 * word_index <= lane < 32 * (word_index + 1)
        )
        if word >= 1 << 31:
            word -= 1 << 32
        words.append(word)

    selected_word = _simd_i32_constant(state, lane_width, words[0])
    if len(words) == 2:
        high_word = _simd_i32_constant(state, lane_width, words[1])
        word_index = _simd_binary_const(
            state,
            "shrui",
            lane_id,
            5,
            lane_width,
        )
        word_mask = state.builder.binary(
            state.dsl.BinaryKind.SubI,
            _simd_i32_constant(state, lane_width, 0),
            word_index,
        )
        word_delta = state.builder.binary(
            state.dsl.BinaryKind.XOrI,
            selected_word,
            high_word,
        )
        selected_word = state.builder.binary(
            state.dsl.BinaryKind.XOrI,
            selected_word,
            state.builder.binary(
                state.dsl.BinaryKind.AndI,
                word_delta,
                word_mask,
            ),
        )

    bit_index = _simd_binary_const(
        state,
        "andi",
        lane_id,
        31,
        lane_width,
    )
    shifted = state.builder.binary(
        state.dsl.BinaryKind.ShRUI,
        selected_word,
        bit_index,
    )
    return _simd_binary_const(
        state,
        "andi",
        shifted,
        1,
        lane_width,
    )


def _lane_mask_from_i32_payload(state, payload, lane_width):
    one = _simd_i32_constant(state, lane_width, 1)
    phase = state.lane_mask_loop_phase
    if phase is None:
        return _cmpi(state, "eq", payload, one)

    # Salt both sides with a lane-varying nonce derived from the current
    # induction value.  The equality remains exactly `payload == 1`, but the
    # physical mask is local to the loop iteration and cannot be hoisted/CSE'd
    # across the loop boundary.  Keeping the nonce lane-varying is essential:
    # a uniform `phase + 1` lowers to an SCC-clobbering SALU op, and Wave's
    # machine loop may keep its continuation condition live in SCC.
    phase = state.builder.splat(phase, state.dsl.i32(), int(lane_width))
    nonce = state.builder.binary(
        state.dsl.BinaryKind.AddI,
        state.builder.lane_id(state.dsl.i32(), int(lane_width)),
        phase,
    )
    payload = state.builder.binary(state.dsl.BinaryKind.AddI, payload, nonce)
    one = state.builder.binary(state.dsl.BinaryKind.AddI, one, nonce)
    return _cmpi(state, "eq", payload, one)


def _simd_i32_from_lane_values(state, lane_values, lane_width):
    lane_width = int(lane_width)
    lane_values = tuple(int(value) for value in lane_values)
    if len(lane_values) != lane_width:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "lane-value SIMD constant map must match lane width",
        )
    result = _simd_i32_constant(state, lane_width, lane_values[0])
    lane_id = state.builder.lane_id(state.dsl.i32(), lane_width)
    for lane_index, value in enumerate(lane_values[1:], start=1):
        if value == lane_values[0]:
            continue
        is_lane = _cmpi(
            state,
            "eq",
            lane_id,
            _simd_i32_constant(state, lane_width, lane_index),
        )
        result = state.builder.select(
            is_lane,
            _simd_i32_constant(state, lane_width, value),
            result,
        )
    return result


def _is_simd_i32_value(state, value):
    try:
        simd = state.dsl.SimdType(value.type)
    except Exception:
        return False
    return str(simd.element_type) == "i32"


def _mask_to_i32_payload_with_predicate(state, component, lane_width):
    if _is_simd_i32_value(state, component):
        return component, None
    return _mask_to_i32_payload(state, component, lane_width), component


def _mask_to_i32_payload(state, component, lane_width):
    if _is_simd_i32_value(state, component):
        return component
    if _is_scalar_i1_value(state, component):
        scalar_payload = state.builder.select(
            component,
            state.builder.constant(state.dsl.i32(), 1),
            state.builder.constant(state.dsl.i32(), 0),
        )
        return state.builder.splat(scalar_payload, state.dsl.i32(), int(lane_width))
    return state.builder.select(
        component,
        _simd_i32_constant(state, lane_width, 1),
        _simd_i32_constant(state, lane_width, 0),
    )


def _mask_and_predicate(state, lhs_component, rhs_component, lane_width, op):
    if _is_scalar_i1_value(state, lhs_component) and _is_scalar_i1_value(state, rhs_component):
        return state.builder.select(
            lhs_component,
            rhs_component,
            _scalar_constant(state, state.dsl.i1(), "i1", False, op),
        )
    if _is_scalar_i1_value(state, rhs_component):
        lhs_component, rhs_component = rhs_component, lhs_component
    return state.builder.select(
        lhs_component,
        rhs_component,
        _wave_mask_constant(state, state.dsl.mask_type(int(lane_width)), False),
    )


def _i32_payload_to_mask(state, component, lane_width):
    # Mask payloads are canonical 0/1 values.  Rebuild predicates as `== 1`
    # rather than `!= 0`: WaveAMDMachine can reuse scalar zero for loop IVs in
    # machine loops, which makes active payload value 1 false on iteration 1.
    if _is_simd_i32_value(state, component):
        return _lane_mask_from_i32_payload(state, component, lane_width)
    one = state.builder.constant(state.dsl.i32(), 1)
    return _cmpi(state, "eq", component, one)


def _mask_payload_component(state, component, lane_width):
    del state, lane_width
    return component


def _materialize_mask_payload(state, payload, lane_width):
    reused = []
    components = tuple(
        _reuse_component_result(
            reused,
            (component, ),
            lambda component=component: _mask_payload_component(
                state,
                component,
                lane_width,
            ),
        )
        for component in payload.components
    )
    return _I32MaskPayload(components)


def _broadcast_components(state, values, count, op):
    return tuple(_broadcast_component(state, value, count, op) for value in values)


def _broadcast_component(state, value, count, op):
    components = _value_components(state, value, op)
    if len(components) == count:
        return components
    if len(components) == 1:
        return components * count
    fail(
        "TLXW_EMIT_COMPONENT_COUNT",
        STAGE,
        f"target op {op.kind} cannot broadcast {len(components)} components "
        f"to {count}",
        target_op_id=op.target_op_id,
    )


def _affine_component_binding_value(state, value, component_count, index, op):
    components = _value_components(state, value, op)
    if len(components) == 1:
        return components[0]
    if len(components) == int(component_count):
        return components[int(index)]
    fail(
        "TLXW_EMIT_COMPONENT_COUNT",
        STAGE,
        f"affine scalar binding cannot map {len(components)} components "
        f"to {int(component_count)} offset components",
        target_op_id=op.target_op_id,
    )


def _range_assumptions(dsl, fact):
    if fact.kind != "range":
        return ()
    x = dsl.sym("x")
    assumptions = []
    if fact.lower is not None:
        assumptions.append(x >= int(fact.lower))
    if fact.upper is not None:
        assumptions.append(x <= int(fact.upper))
    return tuple(assumptions)


def _wave_type(dsl, target_type):
    if target_type.representation == "scalar":
        return _scalar_type(dsl, target_type.element_type)
    if target_type.representation == "uniform_pointer":
        return dsl.ptr_type(_scalar_type(dsl, target_type.element_type))
    if target_type.representation in {"simd", "simd_tuple"}:
        return dsl.simd_type(
            _scalar_type(dsl, target_type.element_type),
            int(target_type.lane_width or 64),
        )
    if target_type.representation in {"mask", "mask_tuple"}:
        return dsl.mask_type(int(target_type.lane_width or 64))
    if target_type.representation in {"per_lane_pointer", "pointer_tuple"}:
        return dsl.simd_ptr_type(
            _scalar_type(dsl, target_type.element_type),
            dsl.global_address_space(),
            int(target_type.lane_width or 64),
        )
    fail(
        "TLXW_EMIT_UNSUPPORTED_TYPE",
        STAGE,
        f"cannot emit target type {target_type}",
    )


def _splat_element_type(dsl, target_type):
    if target_type.representation in {"per_lane_pointer", "pointer_tuple"}:
        return dsl.ptr_type(_scalar_type(dsl, target_type.element_type))
    return _scalar_type(dsl, target_type.element_type)


def _scalar_type(dsl, element_type):
    return {
        "i1": dsl.i1,
        "i8": dsl.i8,
        "i16": lambda: dsl.IntegerType.get_signless(16),
        "i32": dsl.i32,
        "i64": dsl.i64,
        "index": dsl.index_type,
        "f16": dsl.f16,
        "bf16": dsl.bf16,
        "f32": dsl.f32,
    }[element_type]()


def _element_byte_width(element_type, op):
    widths = {
        "i8": 1,
        "i16": 2,
        "i32": 4,
        "i64": 8,
        "f16": 2,
        "bf16": 2,
        "f32": 4,
    }
    width = widths.get(element_type)
    if width is None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_TYPE",
            STAGE,
            f"cannot determine byte width for {element_type}",
            target_op_id=op.target_op_id,
        )
    return int(width)


def _binary_kind(dsl, operation):
    return {
        "addi": dsl.BinaryKind.AddI,
        "subi": dsl.BinaryKind.SubI,
        "muli": dsl.BinaryKind.MulI,
        "shli": dsl.BinaryKind.ShLI,
        "shrui": dsl.BinaryKind.ShRUI,
        "andi": dsl.BinaryKind.AndI,
        "ori": dsl.BinaryKind.OrI,
        "xori": dsl.BinaryKind.XOrI,
        "divui": dsl.BinaryKind.DivUI,
        "divsi": dsl.BinaryKind.DivSI,
        "remui": dsl.BinaryKind.RemUI,
        "remsi": dsl.BinaryKind.RemSI,
    }[operation]


def _is_simd_value(dsl, value):
    try:
        dsl.SimdType(value.type)
    except ValueError:
        return False
    return True


def _is_scalar_i1_value(state, value):
    is_integer = getattr(value.type, "is_integer", None)
    if is_integer is not None and bool(is_integer(1)):
        return True
    return str(value.type) == str(state.dsl.i1())


def _cmpi(state, predicate_name, lhs, rhs):
    predicate = state.dsl.CmpIPredicate[predicate_name]
    lhs_simd = _is_simd_value(state.dsl, lhs)
    rhs_simd = _is_simd_value(state.dsl, rhs)
    if lhs_simd or rhs_simd:
        simd_type = state.dsl.SimdType(lhs.type if lhs_simd else rhs.type)
        if not lhs_simd:
            lhs = state.builder.splat(
                lhs,
                simd_type.element_type,
                int(simd_type.width),
            )
        if not rhs_simd:
            rhs = state.builder.splat(
                rhs,
                simd_type.element_type,
                int(simd_type.width),
            )
        return state.builder.cmpi(predicate, lhs, rhs)
    return state.dsl.arith.CmpIOp(predicate, lhs, rhs).result


def _set_module_attrs(module_builder, dsl, ir, kernel):
    attrs = module_builder.module.operation.attributes
    attrs["tlx_wave.new_converter"] = ir.Attribute.parse("true")
    attrs["tlx_wave.num_ctas"] = ir.IntegerAttr.get(dsl.i32(), int(kernel.num_ctas or 1))
    attrs["tlx_wave.num_warps"] = ir.IntegerAttr.get(dsl.i32(), int(kernel.num_warps or 1))
    attrs["tlx_wave.threads_per_warp"] = ir.IntegerAttr.get(
        dsl.i32(),
        int(kernel.threads_per_warp or 64),
    )
    if kernel.target:
        attrs["tlx_wave.source_target"] = ir.StringAttr.get(kernel.target)
        attrs["waveamdmachine.target"] = ir.StringAttr.get(kernel.target.replace("hip:", "amdgcn-amd-amdhsa--"))


def _kernel_num_warps(kernel):
    return int(kernel.num_warps or 1)


def _kernel_threads_per_warp(kernel):
    return int(kernel.threads_per_warp or 64)


def _kernel_workgroup_size(kernel):
    # Triton supports ND launch grids, but its per-CTA block shape is flat X:
    # AMD/NVIDIA launchers pass (warp_size * num_warps, 1, 1). num_ctas is a
    # cluster/CTA count and is not part of the per-workgroup thread shape.
    return [_kernel_num_warps(kernel) * _kernel_threads_per_warp(kernel), 1, 1]


def _function_attrs(
    dsl,
    ir,
    kernel,
    *,
    enable_split_barriers=False,
    waves_per_eu=0,
):
    num_warps = _kernel_num_warps(kernel)
    workgroup_target_waves = max(1, (num_warps + 3) // 4)
    # Triton's waves_per_eu requests a tighter register budget. It cannot lower
    # the minimum residency needed to place one complete workgroup.
    target_waves = max(workgroup_target_waves, int(waves_per_eu or 0))
    attrs = {
        "tlx_wave.converter.stage": ir.StringAttr.get("structural-emission"),
        "tlx_wave.num_warps": ir.IntegerAttr.get(dsl.i32(), num_warps),
        "tlx_wave.wave_size": ir.IntegerAttr.get(
            dsl.i32(),
            _kernel_threads_per_warp(kernel),
        ),
        "tlx_wave.ttgir.noinline": ir.Attribute.parse("true" if kernel.noinline else "false"),
        "wave.waves_per_workgroup": dsl.i64_attr(num_warps),
        # gfx9/gfx950 exposes four SIMD execution units per CU. Model the
        # requested CTA waves as the resident wave target per SIMD.
        "waveamdmachine.target_waves": dsl.i64_attr(target_waves),
    }
    if enable_split_barriers:
        attrs["waveamdmachine.enable_split_barriers"] = ir.UnitAttr.get()
    return attrs


def _load_wave_dsl():
    third_party = Path(__file__).resolve().parents[3]
    wave_python = (third_party / "wave" / "build" / "wave-build" / "python_packages" / "wave_mlir")
    if not wave_python.exists():
        fail(
            "TLXW_EMIT_BINDINGS_UNAVAILABLE",
            STAGE,
            f"Wave MLIR Python package is missing at {wave_python}",
        )
    path = str(wave_python)
    if path not in sys.path:
        sys.path.insert(0, path)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Attribute builder for .* is already registered",
            category=RuntimeWarning,
        )
        try:
            from mlir import ir
            from mlir.dialects import wave_dsl as dsl
        except Exception as exc:
            fail(
                "TLXW_EMIT_BINDINGS_UNAVAILABLE",
                STAGE,
                f"cannot import Wave MLIR Python bindings: {type(exc).__name__}: {exc}",
            )
    return dsl, ir
