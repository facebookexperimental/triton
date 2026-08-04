"""Structural Wave emission for verified target programs."""

from dataclasses import dataclass, field
from pathlib import Path
import sys
import warnings

from .diagnostics import fail
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


@dataclass(frozen=True)
class EmittedWaveModule:
    text: str
    lds_size: int = 0


@dataclass(frozen=True)
class _SharedPointerDwordBase:
    base: object
    dword_offset: object | None = None
    allocation_base: object | None = None
    allocation_dword_offset: object | None = None
    allocation_bytes: int | None = None
    allocation_dword_range: tuple[int, int] | None = None


@dataclass(frozen=True)
class _VectorPacketPayload:
    packets: tuple[object, ...]
    packet_width: int
    logical_component_count: int


@dataclass(frozen=True)
class _LoopValueShape:
    component_count: int
    packet_width: int | None = None
    logical_component_count: int | None = None
    preserved_vector_payload_key: tuple[int, str, int] | None = None
    preserved_vector_payload_type: object | None = field(default=None, compare=False)


class _PerTargetOpBuilder:
    """Hash-cons pure mechanical expansion within one target operation."""

    def __init__(self, ir, builder):
        self._ir = ir
        self._builder = builder
        self._constants = {}
        self._splats = {}
        self._binaries = {}

    def __getattr__(self, name):
        return getattr(self._builder, name)

    def begin_target_op(self):
        self._constants.clear()
        self._splats.clear()
        self._binaries.clear()

    def _block(self):
        return self._ir.InsertionPoint.current.block

    def constant(self, result_type, value):
        key = (self._block(), str(result_type), value)
        result = self._constants.get(key)
        if result is None:
            result = self._builder.constant(result_type, value)
            self._constants[key] = result
        return result

    def splat(self, value, element_type=None, width=32):
        result_element_type = element_type or value.type
        key = (
            self._block(),
            value,
            str(result_element_type),
            int(width),
        )
        result = self._splats.get(key)
        if result is None:
            result = self._builder.splat(value, element_type, width)
            self._splats[key] = result
        return result

    def binary(self, kind, lhs, rhs, *, nsw=False, nuw=False):
        key = (
            self._block(),
            kind,
            lhs,
            rhs,
            bool(nsw),
            bool(nuw),
        )
        result = self._binaries.get(key)
        if result is None:
            result = self._builder.binary(
                kind,
                lhs,
                rhs,
                nsw=bool(nsw),
                nuw=bool(nuw),
            )
            self._binaries[key] = result
        return result


@dataclass
class _EmissionState:
    dsl: object
    ir: object
    builder: object
    target_program: target_ir.TargetProgram
    values: dict[int, object]
    uniform_pointer_bases: dict[int, tuple[object, ...]] = field(default_factory=dict)
    shared_pointer_dword_bases: dict[int, _SharedPointerDwordBase] = field(default_factory=dict)
    shared_pointer_offset_cache: dict[tuple[object, ...], object] = field(default_factory=dict)
    wave_offset_i32_cache: dict[tuple[object, ...], object] = field(default_factory=dict)


def emit_wave_module(
    target_program,
    *,
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
                        waves_per_eu=waves_per_eu,
                    ),
            ) as builder:
                builder = _PerTargetOpBuilder(ir, builder)
                state = _EmissionState(
                    dsl,
                    ir,
                    builder,
                    target_program,
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
    state.builder.begin_target_op()
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
    if isinstance(value, _VectorPacketPayload):
        return any(_contains_waveamd_fragment(state, packet) for packet in value.packets)
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
            emitted.append(_emit_wave_float_unary_component(
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
            scalars.append(_emit_wave_float_unary_component(
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


def _emit_wave_float_binary_component(state, operation, lhs_component, rhs_component, fastmath, op):
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


def _emit_mma_packet_float_binary_component(state, operation, fastmath, lhs_component, rhs_component, op):
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
    if (operation in {"addf", "subf", "mulf"} and str(lhs_element_type) == "f32" and int(lhs_width) >= 2
            and int(lhs_width) % 2 == 0):
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
        lhs_scalar = state.dsl.wave.ExtractOp(
            scalar_type,
            lhs_component,
            element,
        ).result
        rhs_scalar = state.dsl.wave.ExtractOp(
            scalar_type,
            rhs_component,
            element,
        ).result
        result_scalars.append(
            _emit_wave_float_binary_component(
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
    if result_value_mode == "mma_packet_payload":
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


def _emit_cmpf(state, op):
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
            lambda lhs_component=lhs_component, rhs_component=rhs_component: _cmpf(
                state,
                attrs["predicate"],
                lhs_component,
                rhs_component,
            ),
        ) for lhs_component, rhs_component in zip(lhs_components, rhs_components))
    state.values[result_id] = _pack_components(components)


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
            components.append(component if str(component.type) ==
                              str(cast_type) else state.builder.index_cast(component, cast_type))
        state.values[result_id] = _pack_components(tuple(components))
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
            known_elements = tuple(
                state.dsl.wave.ExtractOp(
                    result_type,
                    packet,
                    element,
                ).result for element in range(packet_width))
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
                components[packet * packet_width:(packet + 1) * packet_width],
            ).result for packet in range(packet_count))
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
    facts = {assumption.assumption_id: assumption for assumption in state.target_program.assumptions}
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
        if fact.kind == "divisible":
            if fact.divisor is None or int(fact.divisor) <= 1:
                fail(
                    "TLXW_EMIT_INVALID_DIVISIBILITY_FACT",
                    STAGE,
                    "divisibility fact requires a divisor greater than one",
                    target_op_id=op.target_op_id,
                    fact_id=fact_id,
                )
            state.values[target_value_id] = state.builder.assume_divisible(
                value,
                int(fact.divisor),
            )
            continue
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
        mask_type = state.dsl.mask_type(lane_width)
        mask = state.builder.select(
            operand,
            _wave_mask_constant(state, mask_type, True),
            _wave_mask_constant(state, mask_type, False),
        )
        state.values[result_id] = _pack_components(tuple(mask for _ in range(_component_count(state, result_id))))
        return
    lane_width = int(target_type.lane_width or 64)
    element_type = _splat_element_type(state.dsl, target_type)
    if _is_simd_value(state.dsl, operand):
        expected_type = state.dsl.simd_type(element_type, lane_width)
        if str(operand.type) != str(expected_type):
            fail(
                "TLXW_EMIT_UNSUPPORTED_SPLAT",
                STAGE,
                f"tensor splat SIMD input has type {operand.type}, expected "
                f"{expected_type}",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        splat = operand
    else:
        splat = state.builder.splat(
            operand,
            element_type,
            lane_width,
        )
    component_count = _component_count(state, result_id)
    state.values[result_id] = _pack_components(tuple(splat for _ in range(component_count)))
    if (target_type.representation in {"per_lane_pointer", "pointer_tuple"} and not _is_simd_value(state.dsl, operand)):
        state.uniform_pointer_bases[result_id] = tuple(operand for _ in range(component_count))


def _emit_broadcast(state, op):
    _emit_packet_layout_transform(
        state,
        op,
        state.dsl.PacketTransform.broadcast(),
    )


def _emit_join(state, op):
    values = _operand_values(state, op, 2)
    source_ids = tuple(int(value_id) for value_id in op.operands)
    result_id = int(_single_result(op))
    source_layouts = tuple(_packet_layout(state, source_id, op) for source_id in source_ids)
    source_packets = tuple(
        _pack_layout_value(state, op, source_id, value, layout)
        for source_id, value, layout in zip(source_ids, values, source_layouts))
    try:
        joined_layout = state.dsl.join_packet_layout(*source_layouts)
    except ValueError as exc:
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            str(exc),
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    joined_type = _layout_packet_type(
        state,
        state.target_program.values[source_ids[0]].type,
        joined_layout,
        op,
    )
    joined_packet = state.dsl.wave.PackOp(
        joined_type,
        source_packets,
    ).result
    result_layout = _packet_layout(state, result_id, op)
    result_type = _layout_packet_type(
        state,
        state.target_program.values[result_id].type,
        result_layout,
        op,
    )
    try:
        result_packet = state.builder.redistribute_layout(
            joined_packet,
            result_type,
            source_layout=joined_layout,
            result_layout=result_layout,
            transform=state.dsl.PacketTransform.identity(),
        )
    except ValueError as exc:
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            str(exc),
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    _unpack_layout_value(state, op, result_id, result_packet, result_layout)
    _propagate_common_uniform_pointer_base(state, source_ids, (result_id, ))


def _emit_split(state, op):
    (value, ) = _operand_values(state, op, 1)
    source_id = int(op.operands[0])
    source_layout = _packet_layout(state, source_id, op)
    source_packet = _pack_layout_value(state, op, source_id, value, source_layout)
    for selector, result_id in enumerate(op.results):
        result_id = int(result_id)
        result_layout = _packet_layout(state, result_id, op)
        result_type = _layout_packet_type(
            state,
            state.target_program.values[result_id].type,
            result_layout,
            op,
        )
        try:
            result_packet = state.builder.redistribute_layout(
                source_packet,
                result_type,
                source_layout=source_layout,
                result_layout=result_layout,
                transform=state.dsl.PacketTransform.split(selector),
            )
        except ValueError as exc:
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                str(exc),
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        _unpack_layout_value(state, op, result_id, result_packet, result_layout)
    _propagate_common_uniform_pointer_base(state, (source_id, ), tuple(int(value_id) for value_id in op.results))


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


def _emit_make_buffer(state, op):
    attrs = target_ir.attrs_dict(op)
    (base, ) = _operand_values(state, op, 1)
    result_id = _single_result(op)
    result_type = _wave_type(
        state.dsl,
        state.target_program.values[result_id].type,
    )
    range_bytes = state.builder.constant(
        state.dsl.i32(),
        int(attrs["range_bytes"]),
    )
    state.values[result_id] = state.builder.make_buffer(
        base,
        range_bytes,
        result_type=result_type,
    )


def _emit_expand_dims(state, op):
    attrs = target_ir.attrs_dict(op)
    _emit_packet_layout_transform(
        state,
        op,
        state.dsl.PacketTransform.expand_dims(int(attrs["axis"])),
    )


def _ptr_add_base_component(state, base_component, offset_component, uniform_base):
    if uniform_base is not None and _is_simd_value(state.dsl, offset_component):
        return uniform_base
    return base_component


def _propagate_uniform_pointer_bases(state, source_id, result_id):
    source_bases = state.uniform_pointer_bases.get(source_id)
    if source_bases is not None:
        state.uniform_pointer_bases[result_id] = source_bases


def _emit_sched_barrier(state, op):
    state.builder.sched_barrier()


def _emit_cond_barrier(state, op):
    (condition, ) = _operand_values(state, op, 1)
    if _is_scalar_i1_value(state, condition):
        with state.builder.if_(condition):
            state.builder.barrier()
        return
    (predicate, ) = _as_mask_components(
        condition,
        1,
        op,
    )
    with state.builder.where(predicate):
        state.builder.barrier()


def _emit_set_priority(state, op):
    attrs = target_ir.attrs_dict(op)
    state.builder.set_priority(int(attrs["priority"]))


def _emit_barrier(state, op):
    attrs = target_ir.attrs_dict(op)
    issue_count = int(attrs.get("barrier_order_dependency_count", 0))
    if issue_count < 0 or issue_count > len(op.operands):
        fail(
            "TLXW_EMIT_BARRIER_OPERANDS",
            STAGE,
            "barrier issue-order segment exceeds its target operands",
            target_op_id=op.target_op_id,
        )
    completion_target_ids = (op.operands[:-issue_count] if issue_count else op.operands)
    issue_target_ids = op.operands[-issue_count:] if issue_count else ()
    completion_tokens = tuple(_require_value(state, target_value_id, op) for target_value_id in completion_target_ids)
    issue_tokens = tuple(_require_value(state, target_value_id, op) for target_value_id in issue_target_ids)
    barrier_token = state.builder.barrier(*completion_tokens)
    if issue_tokens:
        barrier_token = state.builder.after(barrier_token, *issue_tokens)
    if op.results:
        state.values[_single_result(op)] = barrier_token


def _emit_program_id(state, op):
    attrs = target_ir.attrs_dict(op)
    state.values[_single_result(op)] = state.builder.workgroup_id(int(attrs["axis"]))


def _emit_warp_id(state, op):
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    lane_width = int(target_type.lane_width or state.target_program.kernel.threads_per_warp or 64)
    if lane_width <= 0 or lane_width & (lane_width - 1):
        fail(
            "TLXW_EMIT_WARP_ID",
            STAGE,
            f"ttg.warp_id requires a power-of-two wave width, got {lane_width}",
            target_op_id=op.target_op_id,
        )
    workitem = state.builder.workitem_id(0, state.dsl.i32(), lane_width)
    wave_first = state.builder.read_first(workitem)
    shift = state.builder.constant(
        state.dsl.i32(),
        lane_width.bit_length() - 1,
    )
    state.values[result_id] = state.builder.binary(
        state.dsl.BinaryKind.ShRUI,
        wave_first,
        shift,
    )


def _emit_ballot(state, op):
    (predicate, ) = _operand_values(state, op, 1)
    if isinstance(predicate, (tuple, list)):
        fail(
            "TLXW_EMIT_WARP_BALLOT",
            STAGE,
            "warp ballot predicate must contain exactly one mask component",
            target_op_id=op.target_op_id,
        )
    state.values[_single_result(op)] = state.builder.ballot(
        predicate,
        state.dsl.i64(),
    )


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
    count = _component_count(state, result_id)
    condition, true_value, false_value = _operand_values(state, op, 3)
    true_components, false_components = _broadcast_components(
        state,
        (true_value, false_value),
        count,
        op,
    )
    raw_cond_components = _mask_components(condition)
    expanded = _select_vector_payload_components(
        state,
        raw_cond_components,
        true_components,
        false_components,
    )
    if expanded is not None:
        state.values[result_id] = _pack_components(expanded)
        return
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


def _select_vector_payload_components(
    state,
    conditions,
    true_components,
    false_components,
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
    payloads = tuple(_simd_1d_vector_payload(state, component) for component in true_components)
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
        true_elements = tuple(
            state.dsl.wave.ExtractOp(
                scalar_type,
                true_component,
                element,
            ).result for element in range(width))
        false_elements = tuple(
            state.dsl.wave.ExtractOp(
                scalar_type,
                false_component,
                element,
            ).result for element in range(width))
        selected = []
        for condition, true_element, false_element in zip(
                conditions[cursor:cursor + width],
                true_elements,
                false_elements,
        ):
            selected.append(state.builder.select(condition, true_element, false_element))
        results.append(state.dsl.wave.PackOp(true_component.type, selected).result)
        cursor += width
    return tuple(results)


def _emit_reduction(state, op):
    attrs = target_ir.attrs_dict(op)
    if len(op.region_ids) != 1:
        fail(
            "TLXW_EMIT_REDUCTION",
            STAGE,
            "reduction requires one combiner region",
            target_op_id=op.target_op_id,
        )
    region = state.target_program.regions[int(op.region_ids[0])]
    if len(region.block_arg_ids) != 2:
        fail(
            "TLXW_EMIT_REDUCTION",
            STAGE,
            "reduction combiner requires two block arguments",
            target_op_id=op.target_op_id,
        )
    (source, ) = _operand_values(state, op, 1)
    source_id = int(op.operands[0])
    result_id = int(_single_result(op))
    source_layout = _packet_layout(state, source_id, op)
    result_layout = _packet_layout(state, result_id, op)
    source_packet = _pack_layout_value(
        state,
        op,
        source_id,
        source,
        source_layout,
    )
    result_type = _layout_packet_type(
        state,
        state.target_program.values[result_id].type,
        result_layout,
        op,
    )
    outer_values = dict(state.values)
    outer_uniform_pointer_bases = dict(state.uniform_pointer_bases)
    outer_shared_pointer_dword_bases = dict(state.shared_pointer_dword_bases)
    outer_shared_pointer_offset_cache = dict(state.shared_pointer_offset_cache)
    outer_wave_offset_i32_cache = dict(state.wave_offset_i32_cache)
    try:
        reorderable = attrs["reduction_ordering"] == "unordered"
        with state.builder.reduce_layout(
                source_packet,
                result_type,
                source_layout=source_layout,
                result_layout=result_layout,
                axis=int(attrs["axis"]),
                associative=reorderable,
                commutative=reorderable,
        ) as reduction:
            _restore_structural_emission_state(
                state,
                outer_values,
                outer_uniform_pointer_bases,
                outer_shared_pointer_dword_bases,
                outer_shared_pointer_offset_cache,
                outer_wave_offset_i32_cache,
            )
            for target_value_id, argument in zip(
                    region.block_arg_ids,
                    reduction.arguments,
            ):
                state.values[int(target_value_id)] = argument
            yielded = _emit_region(state, op.region_ids[0])
            if len(yielded) != 1:
                fail(
                    "TLXW_EMIT_REDUCTION",
                    STAGE,
                    "reduction combiner must yield one value",
                    target_op_id=op.target_op_id,
                )
            state.builder.yield_(yielded)
        result_packet = reduction.result
    except ValueError as exc:
        fail(
            "TLXW_EMIT_REDUCTION",
            STAGE,
            str(exc),
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    finally:
        _restore_structural_emission_state(
            state,
            outer_values,
            outer_uniform_pointer_bases,
            outer_shared_pointer_dword_bases,
            outer_shared_pointer_offset_cache,
            outer_wave_offset_i32_cache,
        )
    _unpack_layout_value(
        state,
        op,
        result_id,
        result_packet,
        result_layout,
    )


def _emit_if(state, op):
    return _emit_if_structural_only(state, op)


def _emit_if_structural_only(state, op):
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
    with state.builder.if_(condition, result_types, otherwise=True) as ifop:
        _restore_structural_emission_state(
            state,
            outer_values,
            outer_uniform_pointer_bases,
            outer_shared_pointer_dword_bases,
            outer_shared_pointer_offset_cache,
            outer_wave_offset_i32_cache,
        )
        then_yields = _emit_structured_branch(
            state,
            op.region_ids[0],
            op.results,
            result_shapes,
            "if then",
            op,
        )
        if then_yields:
            state.builder.yield_(then_yields)
        _restore_structural_emission_state(
            state,
            outer_values,
            outer_uniform_pointer_bases,
            outer_shared_pointer_dword_bases,
            outer_shared_pointer_offset_cache,
            outer_wave_offset_i32_cache,
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
            if else_yields:
                state.builder.yield_(else_yields)
    _restore_structural_emission_state(
        state,
        outer_values,
        outer_uniform_pointer_bases,
        outer_shared_pointer_dword_bases,
        outer_shared_pointer_offset_cache,
        outer_wave_offset_i32_cache,
    )
    flat_results = tuple(ifop.results)
    if len(flat_results) != len(result_types):
        fail(
            "TLXW_EMIT_IF_RESULT_COMPONENTS",
            STAGE,
            "if result component count must match its explicit result types",
            target_op_id=op.target_op_id,
        )
    cursor = 0
    for result_id, shape in zip(op.results, result_shapes):
        state.values[result_id] = _pack_structured_value_components(
            state,
            flat_results[cursor:cursor + shape.component_count],
            shape,
            "if",
            op,
        )
        cursor += shape.component_count


def _emit_for_loop(state, op):
    return _emit_for_loop_structural_only(state, op)


def _emit_for_loop_structural_only(state, op):
    return _emit_for_loop_literal(state, op)


def _emit_for_loop_literal(state, op):
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
    init_values = tuple(_require_value(state, target_value_id, op) for target_value_id in init_target_ids)
    flat_init_values, init_shapes = _flatten_structured_values(
        state,
        init_values,
        init_target_ids,
        "for_loop",
        op,
        preserve_mma_packet_payloads=True,
        pack_four_f32_simd_tuples=True,
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
    outer_uniform_pointer_bases = dict(state.uniform_pointer_bases)
    outer_shared_pointer_dword_bases = dict(state.shared_pointer_dword_bases)
    outer_shared_pointer_offset_cache = dict(state.shared_pointer_offset_cache)
    outer_wave_offset_i32_cache = dict(state.wave_offset_i32_cache)

    with state.builder.for_loop(
            lower,
            upper,
            step,
            init_args=flat_init_values,
            nonzero_trip=bool(attrs.get("nonzero_trip", False)),
    ) as loop:
        if flat_init_values:
            induction_value = loop.induction_variable
            flat_iter_values = tuple(loop.inner_iter_args)
        else:
            induction_value = loop
            flat_iter_values = ()
        _bind_loop_region_args_structural(
            state,
            region.block_arg_ids,
            induction_value,
            flat_iter_values,
            init_shapes,
            init_target_ids,
            op=op,
        )
        yielded_values = _emit_region(state, op.region_ids[0])
        flat_yield_values, yield_shapes = _flatten_structured_values(
            state,
            yielded_values,
            region.yield_value_ids,
            "for_loop",
            op,
            expected_shapes=init_shapes,
            preserve_mma_packet_payloads=True,
            pack_four_f32_simd_tuples=True,
        )
        if tuple(yield_shapes) != tuple(init_shapes):
            fail(
                "TLXW_EMIT_FOR_YIELD_COMPONENTS",
                STAGE,
                "for_loop yielded component shape must match init args",
                target_op_id=op.target_op_id,
            )
        if flat_init_values:
            state.builder.yield_(flat_yield_values)
        elif flat_yield_values:
            fail(
                "TLXW_EMIT_FOR_UNEXPECTED_YIELD",
                STAGE,
                "for_loop without init args must not yield values",
                target_op_id=op.target_op_id,
            )

    _restore_structural_emission_state(
        state,
        outer_values,
        outer_uniform_pointer_bases,
        outer_shared_pointer_dword_bases,
        outer_shared_pointer_offset_cache,
        outer_wave_offset_i32_cache,
    )

    flat_results = tuple(loop.results) if flat_init_values else ()
    if len(flat_results) != len(flat_init_values):
        fail(
            "TLXW_EMIT_FOR_RESULT_COMPONENTS",
            STAGE,
            "for_loop result component count must match explicit init args",
            target_op_id=op.target_op_id,
        )
    if len(op.results) != init_arg_count:
        fail(
            "TLXW_EMIT_FOR_RESULT_COUNT",
            STAGE,
            "for_loop result count must match init args",
            target_op_id=op.target_op_id,
        )
    cursor = 0
    for result_id, shape, init_target_id in zip(
            op.results,
            init_shapes,
            init_target_ids,
    ):
        value = _pack_loop_value_components(
            state,
            flat_results[cursor:cursor + shape.component_count],
            shape,
            op,
        )
        state.values[result_id] = value
        _propagate_uniform_pointer_bases(
            state,
            init_target_id,
            result_id,
        )
        plan = outer_shared_pointer_dword_bases.get(int(init_target_id))
        if plan is not None:
            state.shared_pointer_dword_bases[int(result_id)] = (_SharedPointerDwordBase(
                value,
                allocation_base=plan.allocation_base,
                allocation_bytes=plan.allocation_bytes,
            ))
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
        expected_shapes=result_shapes,
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
    if packet_registers and len(packet_registers) < len(target_value_ids):
        synthetic_result_ids = target_value_ids[len(packet_registers):]
        if all(state.target_program.values[target_value_id].type.representation == "token"
               for target_value_id in synthetic_result_ids):
            packet_registers = (
                *packet_registers,
                *((0, ) * len(synthetic_result_ids)),
            )
    if packet_registers and len(packet_registers) != len(target_value_ids):
        fail(
            "TLXW_EMIT_IF_RESULT_TYPE",
            STAGE,
            "if packet register widths must match the data result count and "
            "any trailing structural token results",
            target_op_id=op.target_op_id,
        )
    for result_index, target_value_id in enumerate(target_value_ids):
        target_type = state.target_program.values[target_value_id].type
        component_count = int(target_type.component_count)
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
            shapes.append(
                _LoopValueShape(
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
                "uniform_buffer_pointer",
                "simd",
                "simd_tuple",
                "per_lane_pointer",
                "pointer_tuple",
                "buffer_pointer",
                "buffer_pointer_tuple",
                "mask",
                "mask_tuple",
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


def _restore_structural_emission_state(
    state,
    values,
    uniform_pointer_bases,
    shared_pointer_dword_bases,
    shared_pointer_offset_cache,
    wave_offset_i32_cache,
):
    state.values = dict(values)
    state.uniform_pointer_bases = dict(uniform_pointer_bases)
    state.shared_pointer_dword_bases = dict(shared_pointer_dword_bases)
    state.shared_pointer_offset_cache = dict(shared_pointer_offset_cache)
    state.wave_offset_i32_cache = dict(wave_offset_i32_cache)


def _flatten_structured_values(
    state,
    values,
    target_value_ids,
    context,
    op,
    *,
    expected_shapes=None,
    preserve_mma_packet_payloads=False,
    pack_four_f32_simd_tuples=False,
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
    for value, target_value_id in zip(values, target_value_ids):
        target_type = state.target_program.values[target_value_id].type
        component_count = int(target_type.component_count)
        if isinstance(value, _VectorPacketPayload):
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
            shapes.append(
                _LoopValueShape(
                    len(components),
                    packet_width=int(value.packet_width),
                    logical_component_count=int(value.logical_component_count),
                ))
        else:
            components = _value_components(state, value, op)
            shape = None
            if (pack_four_f32_simd_tuples and _is_four_f32_simd_tuple(target_type)):
                components, shape = _pack_four_f32_loop_components(
                    state,
                    components,
                    target_value_id,
                    context,
                    op,
                )
            elif (preserve_mma_packet_payloads and target_type.representation in _MMA_PACKET_REPRESENTATIONS):
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


def _is_four_f32_simd_tuple(target_type):
    return (target_type.representation == "simd_tuple" and target_type.element_type == "f32"
            and int(target_type.component_count) == 4 and int(target_type.lane_width or 0) == 64)


def _pack_four_f32_loop_components(
    state,
    components,
    target_value_id,
    context,
    op,
):
    components = tuple(components)
    target_type = state.target_program.values[target_value_id].type
    if len(components) != 4:
        fail(
            "TLXW_EMIT_FOR_SIMD_PACKET",
            STAGE,
            f"{context} four-f32 loop packet must contain four components",
            target_op_id=op.target_op_id,
            target_value_id=target_value_id,
        )
    scalar_type = _wave_type(state.dsl, target_type)
    if any(str(component.type) != str(scalar_type) for component in components):
        fail(
            "TLXW_EMIT_FOR_SIMD_PACKET",
            STAGE,
            f"{context} four-f32 loop packet components must have matching scalar SIMD types",
            target_op_id=op.target_op_id,
            target_value_id=target_value_id,
        )
    packet_type = state.dsl.simd_type(
        state.dsl.vector_type(4, _scalar_type(state.dsl, "f32")),
        64,
    )
    packet = state.dsl.wave.PackOp(packet_type, components).result
    return (
        (packet, ),
        _LoopValueShape(
            1,
            packet_width=4,
            logical_component_count=4,
        ),
    )


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
        elif (payload_key != expected_payload or str(component.type) != str(expected_type)):
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


def _pack_structured_value_components(state, components, shape, context, op):
    components = tuple(components)
    if len(components) != int(shape.component_count):
        fail(
            "TLXW_EMIT_STRUCTURED_COMPONENT_SHAPE",
            STAGE,
            f"{context} component slice does not match recorded value shape",
            target_op_id=None if op is None else op.target_op_id,
        )
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


def _bind_loop_region_args_structural(
    state,
    block_arg_ids,
    induction_value,
    flat_iter_values,
    init_shapes,
    init_target_ids,
    *,
    op,
):
    state.values[block_arg_ids[0]] = induction_value
    cursor = 0
    for block_arg_id, shape, init_target_id in zip(
            block_arg_ids[1:],
            init_shapes,
            init_target_ids,
    ):
        components = flat_iter_values[cursor:cursor + shape.component_count]
        value = _pack_loop_value_components(
            state,
            components,
            shape,
            op,
        )
        state.values[block_arg_id] = value
        _propagate_uniform_pointer_bases(
            state,
            init_target_id,
            block_arg_id,
        )
        plan = state.shared_pointer_dword_bases.get(int(init_target_id))
        if plan is not None:
            state.shared_pointer_dword_bases[int(block_arg_id)] = (_SharedPointerDwordBase(
                value,
                allocation_base=plan.allocation_base,
                allocation_bytes=plan.allocation_bytes,
            ))
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
    state.shared_pointer_dword_bases[result_id] = _SharedPointerDwordBase(
        value,
        allocation_base=value,
        allocation_bytes=int(attrs["allocation_bytes"]),
        allocation_dword_range=(0, 0),
    )


def _emit_memdesc_index(state, op):
    attrs = target_ir.attrs_dict(op)
    base, index = _operand_values(state, op, 2)
    result_id = _single_result(op)
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
    dword_offset = int(byte_offset) // 4
    allocation_dword_range = None
    if base_plan.allocation_dword_range is not None:
        allocation_dword_range = (
            int(base_plan.allocation_dword_range[0]) + dword_offset,
            int(base_plan.allocation_dword_range[1]) + dword_offset,
        )
    state.shared_pointer_dword_bases[result_id] = _SharedPointerDwordBase(
        base,
        base_plan.dword_offset,
        base_plan.allocation_base,
        _add_constant_to_optional_i32_offset(
            state,
            base_plan.allocation_dword_offset,
            dword_offset,
        ),
        base_plan.allocation_bytes,
        allocation_dword_range,
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
    del op
    if slot_dwords == 1:
        dword_offset = index
    else:
        dword_offset = _scalar_binary_const_i32(
            state,
            "muli",
            index,
            slot_dwords,
            nsw=_LAYOUT_MATH_NSW,
        )
    state.shared_pointer_dword_bases[result_id] = _SharedPointerDwordBase(
        base_plan.base,
        _combine_optional_i32_offsets(
            state,
            base_plan.dword_offset,
            dword_offset,
            nsw=_LAYOUT_MATH_NSW,
        ),
        base_plan.allocation_base,
        _combine_optional_i32_offsets(
            state,
            base_plan.allocation_dword_offset,
            dword_offset,
            nsw=_LAYOUT_MATH_NSW,
        ),
        base_plan.allocation_bytes,
    )


def _emit_memdesc_view(state, op):
    base = _operand_values(state, op, 1)[0]
    result_id = _single_result(op)
    state.values[result_id] = base
    dword_base = state.shared_pointer_dword_bases.get(op.operands[0])
    if dword_base is not None:
        state.shared_pointer_dword_bases[result_id] = dword_base


def _local_access_result_ids(op):
    attrs = target_ir.attrs_dict(op)
    data_count = int(attrs.get("data_result_count", len(op.results)))
    completion_count = int(attrs.get("completion_result_count", 0))
    if (data_count < 0 or completion_count < 0 or data_count + completion_count != len(op.results)):
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
    del memdesc_target_id, access_kind
    _data_result_ids, completion_result_ids = _local_access_result_ids(op)
    for result_id in completion_result_ids:
        state.values[result_id] = token


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
    explicit_dependencies = tuple(_require_value(state, target_value_id, op) for target_value_id in op.operands[2:])
    memdesc_target_id = op.operands[1]
    lane_width = int(attrs["lane_width"])
    component_count = int(attrs["component_count"])
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    base = _ptr_cast(
        state,
        base,
        state.dsl.ptr_type(element_type, state.dsl.shared_address_space()),
    )
    dependency = _memory_dependency_token(state, explicit_dependencies)
    target_type = state.target_program.values[op.operands[0]].type
    packet_count = int(attrs["packet_count"])
    packet_width = int(attrs["packet_width"])
    if target_type.representation in _MMA_PACKET_REPRESENTATIONS:
        payloads = _as_components(values)
        if len(payloads) != packet_count or packet_count != component_count:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "local_store packet payload does not match its relation",
                target_op_id=op.target_op_id,
            )
        for payload in payloads:
            shape = _simd_1d_vector_payload(state, payload)
            if shape is None or int(shape[0]) != packet_width:
                fail(
                    "TLXW_EMIT_COMPONENT_COUNT",
                    STAGE,
                    "local_store packet payload has the wrong vector width",
                    target_op_id=op.target_op_id,
                )
    else:
        if packet_count != 1 or packet_width != component_count:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "local_store scalar payload does not match its packet relation",
                target_op_id=op.target_op_id,
            )
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
        payloads = (_symbolic_shared_packet(
            state,
            value_components,
            element_type,
            lane_width,
        ), )
    packet_bases = tuple(tuple(int(value) for value in packet_base) for packet_base in attrs["packet_coordinate_bases"])
    if len(packet_bases) != packet_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "local_store packet bases do not match its packet count",
            target_op_id=op.target_op_id,
        )
    tokens = []
    for payload, packet_base in zip(payloads, packet_bases):
        tokens.append(
            state.builder.scatter(
                payload,
                [base],
                bit_offset=_symbolic_local_packet_bit_offset(
                    state,
                    attrs,
                    packet_base,
                    op,
                ),
                after=dependency,
            ))
    _finish_local_access(
        state,
        op,
        memdesc_target_id,
        _join_memory_tokens(state, tokens),
        "write",
    )


def _symbolic_shared_packet(state, components, element_type, lane_width):
    components = tuple(components)
    packet_type = state.dsl.simd_type(
        state.dsl.vector_type(len(components), element_type),
        width=int(lane_width),
    )
    return state.dsl.wave.PackOp(packet_type, components).result


def _symbolic_shared_packet_components(state, packet, component_count, element_type, lane_width):
    component_type = state.dsl.simd_type(element_type, int(lane_width))
    return tuple(
        state.dsl.wave.ExtractOp(component_type, packet, index).result for index in range(int(component_count)))


def _symbolic_mask_conditions(state, predicates):
    predicates = tuple(predicates)
    if not predicates:
        raise ValueError("symbolic memory predicates must not be empty")
    if all(_is_scalar_i1_value(state, predicate) for predicate in predicates):
        if not all(predicate is predicates[0] for predicate in predicates):
            fail(
                "TLXW_EMIT_UNSUPPORTED_MEMORY_MASK",
                STAGE,
                "a scalar control predicate must be uniform across a memory packet",
            )
        return predicates[0]
    return predicates


def _symbolic_buffer_mapping(state, element_byte_width):
    """Describe the element offset carried by an ``amdg.buffer_*`` op."""
    offset_symbol = state.dsl.sym("offset")
    return offset_symbol, 8 * int(element_byte_width) * offset_symbol


def _symbolic_buffer_offset_range(attrs):
    element_byte_width = int(attrs["element_byte_width"])
    upper = (int(attrs["range_bytes"]) - element_byte_width + 1) // element_byte_width
    return 0, upper


def _assume_symbolic_element_contiguity(
    state,
    offsets,
    contiguity,
    component_indices=None,
):
    """Express scalar element contiguity with ordinary Wave predicates."""
    offsets = tuple(offsets)
    contiguity = int(contiguity)
    if contiguity <= 1 or len(offsets) <= 1:
        return offsets
    if component_indices is None:
        component_indices = range(len(offsets))
    component_indices = tuple(int(index) for index in component_indices)
    if len(component_indices) != len(offsets):
        raise ValueError("symbolic contiguity indices must match offsets")

    normalized = []
    previous_index = None
    for component_index, offset in zip(component_indices, offsets):
        if (previous_index is None or component_index % contiguity == 0 or component_index != previous_index + 1):
            normalized.append(offset)
            previous_index = component_index
            continue
        previous = normalized[-1]
        delta = state.builder.binary(
            state.dsl.BinaryKind.SubI,
            offset,
            previous,
        )
        unit_delta = state.builder.assume_range(delta, 1, 1)
        normalized.append(state.builder.binary(
            state.dsl.BinaryKind.AddI,
            previous,
            unit_delta,
        ))
        previous_index = component_index
    return tuple(normalized)


def _prepare_symbolic_indexed_gather(
    state,
    offsets,
    base,
    element_type,
    lane_width,
    element_byte_width,
    *,
    offset_range=None,
    op=None,
    dependency=None,
    element_contiguity=1,
    element_component_indices=None,
    cache=None,
):
    """Prepare stable operands and defer access-scoped facts with the gather."""
    offsets = tuple(offsets)
    result_type = state.dsl.simd_type(
        state.dsl.vector_type(len(offsets), element_type),
        width=int(lane_width),
    )
    offset_components = tuple(_simd_offset_value(state, offset, lane_width) for offset in offsets)
    offset_symbol, bit_offset = _symbolic_buffer_mapping(state, element_byte_width)

    def emit(packet_conditions=()):
        access_offsets = tuple(_assume_value_range(state, offset, offset_range, op) for offset in offset_components)
        access_offsets = _assume_symbolic_element_contiguity(
            state,
            access_offsets,
            element_contiguity,
            element_component_indices,
        )
        return state.builder.gather(
            [base],
            result_type,
            bit_offset=bit_offset,
            packet_bindings={offset_symbol: access_offsets},
            packet_conditions=packet_conditions,
            after=dependency,
            cache=cache,
        )

    return emit


def _prepare_symbolic_indexed_scatter(
    state,
    values,
    offsets,
    base,
    element_type,
    lane_width,
    element_byte_width,
    *,
    offset_range=None,
    op=None,
    dependency=None,
    element_contiguity=1,
    element_component_indices=None,
    cache=None,
):
    """Prepare stable operands and defer access-scoped facts with the scatter."""
    values = tuple(values)
    offsets = tuple(offsets)
    if not values or len(values) != len(offsets):
        raise ValueError("symbolic scatter values and offsets must match")
    value_packet = _symbolic_shared_packet(
        state,
        values,
        element_type,
        lane_width,
    )
    offset_components = tuple(_simd_offset_value(state, offset, lane_width) for offset in offsets)
    offset_symbol, bit_offset = _symbolic_buffer_mapping(state, element_byte_width)

    def emit(packet_conditions=()):
        access_offsets = tuple(_assume_value_range(state, offset, offset_range, op) for offset in offset_components)
        access_offsets = _assume_symbolic_element_contiguity(
            state,
            access_offsets,
            element_contiguity,
            element_component_indices,
        )
        return state.builder.scatter(
            value_packet,
            [base],
            bit_offset=bit_offset,
            packet_bindings={offset_symbol: access_offsets},
            packet_conditions=packet_conditions,
            after=dependency,
            cache=cache,
        )

    return emit


def _extract_packet_components(
    state,
    packet,
    element_type,
    lane_width,
    component_count,
):
    component_type = state.dsl.simd_type(element_type, int(lane_width))
    return tuple(
        state.dsl.wave.ExtractOp(component_type, packet, index).result for index in range(int(component_count)))


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
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    base = _ptr_cast(
        state,
        base,
        state.dsl.ptr_type(element_type, state.dsl.shared_address_space()),
    )
    dependency = _memory_dependency_token(
        state,
        tuple(explicit_dependencies),
    )
    payloads, token = _emit_symbolic_local_gather_packets(
        state,
        op,
        attrs,
        base,
        element_type,
        dependency,
    )
    if target_type.representation in _MMA_PACKET_REPRESENTATIONS:
        if len(payloads) != component_count:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "local_load packet relation does not match its result components",
                target_op_id=op.target_op_id,
            )
        state.values[result_id] = _pack_components(payloads)
    else:
        if len(payloads) != 1 or int(attrs["packet_width"]) != component_count:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "local_load scalar packet relation does not match its result",
                target_op_id=op.target_op_id,
            )
        components = _symbolic_shared_packet_components(
            state,
            payloads[0],
            component_count,
            element_type,
            lane_width,
        )
        state.values[result_id] = _pack_components(components)
    _finish_local_access(state, op, memdesc_target_id, token, "read")


def _symbolic_local_destination_bit_offset(
    state,
    attrs,
    component_count,
    element_byte_width,
    op,
):
    """Translate the destination layout into one Wave memory relation."""
    if attrs.get("destination_offset_mode") != "layout_coordinates":
        fail(
            "TLXW_EMIT_LOCAL_MEMORY",
            STAGE,
            f"unsupported local-memory offset mode "
            f"{attrs.get('destination_offset_mode')}",
            target_op_id=op.target_op_id,
        )
    shape = tuple(int(value) for value in attrs["destination_coordinate_shape"])
    component_bases = tuple(
        tuple(int(value) for value in bases) for bases in attrs["destination_component_coordinate_bases"])
    item_coefficients = tuple(
        tuple(int(value)
              for value in coefficients)
        for coefficients in attrs["destination_workitem_coordinate_coefficients"])
    component_count = int(component_count)
    if (component_count <= 0 or component_count & (component_count - 1) or len(component_bases) != component_count
            or any(len(bases) != len(shape) for bases in component_bases)
            or any(len(coefficients) != len(shape) for coefficients in item_coefficients)):
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "local-memory layout relation requires a power-of-two packet "
            "with ranked coordinate bases",
            target_op_id=op.target_op_id,
        )

    slot_bit_count = component_count.bit_length() - 1
    slot_coefficients = tuple(
        tuple(int(component_bases[0][dim]) ^ int(component_bases[1 << bit][dim])
              for dim in range(len(shape)))
        for bit in range(slot_bit_count))
    for slot_index, bases in enumerate(component_bases):
        expected = list(component_bases[0])
        for bit, coefficients in enumerate(slot_coefficients):
            if slot_index & (1 << bit):
                expected = [int(value) ^ int(coefficient) for value, coefficient in zip(expected, coefficients)]
        if tuple(expected) != tuple(bases):
            fail(
                "TLXW_EMIT_BAD_COORDINATES",
                STAGE,
                "local-memory component coordinates are not a bit-linear "
                "slot relation",
                target_op_id=op.target_op_id,
            )

    item = state.dsl.sym("item")
    slot = state.dsl.sym("slot")
    logical_coords = []
    for dim, base in enumerate(component_bases[0]):
        coord = state.dsl.sym_ctx.int_(int(base))
        for bit, coefficients in enumerate(slot_coefficients):
            coefficient = int(coefficients[dim])
            if not coefficient:
                continue
            value = state.dsl.mod(
                state.dsl.floor(slot / (1 << bit)),
                2,
            )
            if coefficient != 1:
                value *= coefficient
            coord = state.dsl.xor(coord, value)
        for bit, coefficients in enumerate(item_coefficients):
            coefficient = int(coefficients[dim])
            if not coefficient:
                continue
            value = state.dsl.mod(
                state.dsl.floor(item / (1 << bit)),
                2,
            )
            if coefficient != 1:
                value *= coefficient
            coord = state.dsl.xor(coord, value)
        logical_coords.append(coord)

    physical_shape = tuple(int(value) for value in attrs["destination_physical_shape"])
    logical_origin = tuple(int(value) for value in attrs["destination_logical_origin"])
    if not (len(logical_coords) == len(shape) == len(physical_shape) == len(logical_origin)):
        fail(
            "TLXW_EMIT_BAD_COORDINATES",
            STAGE,
            "local-memory symbolic destination view ranks do not match",
            target_op_id=op.target_op_id,
        )
    physical_coords = tuple(coord + int(origin) for coord, origin in zip(logical_coords, logical_origin))
    plan = attrs.get("destination_physical_offset_plan")
    if plan == "dense_row_major":
        element_offset = _linearize_local_fragment_coords(
            physical_shape,
            physical_coords,
        )
    elif plan == "linear_shared":
        element_offset = _linear_inverse_offset_from_expr_coords(
            state,
            attrs,
            "destination",
            physical_coords,
            op,
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
        )
    elif plan == "padded_linear":
        physical = _linear_component_offset_from_expr_coords(
            state,
            attrs,
            "destination",
            physical_coords,
            op,
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
        )
        element_offset = physical
        for interval, padding in zip(
                attrs.get("destination_physical_intervals", ()),
                attrs.get("destination_physical_paddings", ()),
        ):
            element_offset += (state.dsl.floor(physical / int(interval)) * int(padding))
    elif plan == "swizzled_xor":
        order = _physical_order_from_attrs(
            attrs,
            "destination_physical_order",
            physical_shape,
            op,
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
        )
        minor_dim = int(order[0])
        major_dim = int(order[1])
        minor_extent = int(physical_shape[minor_dim])
        major = physical_coords[major_dim]
        minor = physical_coords[minor_dim]
        vec = int(attrs["destination_physical_swizzled_vec"])
        phase = state.dsl.mod(
            state.dsl.floor(major / int(attrs["destination_physical_swizzled_per_phase"])),
            int(attrs["destination_physical_swizzled_max_phase"]),
        )
        max_phase = int(attrs["destination_physical_swizzled_max_phase"])
        if vec * max_phase <= minor_extent:
            swizzled_minor = (state.dsl.xor(state.dsl.floor(minor / vec), phase) * vec + state.dsl.mod(minor, vec))
        else:
            phase_offset = state.dsl.mod(phase * vec, minor_extent)
            swizzled_minor = state.dsl.xor(minor, phase_offset)
        encoded_coords = list(physical_coords)
        encoded_coords[minor_dim] = swizzled_minor
        element_offset = _linearize_local_fragment_coords_with_order(
            physical_shape,
            encoded_coords,
            order,
        )
    else:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"unsupported symbolic shared destination offset plan {plan}",
            target_op_id=op.target_op_id,
        )
    return (int(element_byte_width) * 8 * element_offset).simplify()


def _emit_buffer_load_to_local(state, op):
    """Emit one semantic source gather and destination scatter packet."""
    attrs = target_ir.attrs_dict(op)
    if attrs.get("mode") != "symbolic_copy":
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"unsupported amdg.buffer_load_to_local mode {attrs.get('mode')}",
            target_op_id=op.target_op_id,
        )
    source_issue_dependency_count = int(attrs.get("issue_dependency_count", 0))
    barrier_order_dependency_count = int(attrs.get("barrier_order_dependency_count", 0))
    issue_dependency_count = (source_issue_dependency_count + barrier_order_dependency_count)
    has_mask = bool(attrs.get("has_mask", False))
    expected_operand_count = 3 + int(has_mask) + issue_dependency_count
    if len(op.operands) != expected_operand_count:
        fail(
            "TLXW_EMIT_OPERAND_COUNT",
            STAGE,
            f"expected {expected_operand_count} operands, got "
            f"{len(op.operands)}",
            target_op_id=op.target_op_id,
        )

    dest_base = _require_value(state, op.operands[0], op)
    source_base = _require_value(state, op.operands[1], op)
    offsets = _require_value(state, op.operands[2], op)
    operand_index = 3
    masks = (_require_value(state, op.operands[operand_index], op) if has_mask else None)
    operand_index += int(has_mask)
    issue_dependencies = tuple(_require_value(state, operand_id, op) for operand_id in op.operands[operand_index:])
    component_count = int(attrs["component_count"])
    lane_width = int(attrs["lane_width"])
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    offset_components = _as_components(offsets)
    if len(offset_components) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "amdg.buffer_load_to_local offset component count does not "
            "match its symbolic packet",
            target_op_id=op.target_op_id,
        )
    mask_components = (None if masks is None else _as_mask_components(
        masks,
        component_count,
        op,
    ))
    if mask_components is not None and attrs.get("mask_mode") != "exec_where":
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC_MASK",
            STAGE,
            f"unsupported buffer_load_to_local mask mode "
            f"{attrs.get('mask_mode')}",
            target_op_id=op.target_op_id,
        )

    # Direct-to-LDS completion is represented only by the explicit async
    # protocol.  Destination aliasing and prior LDS accesses must not invent
    # an issue dependency here; any required order is carried in the source
    # protocol operands above.
    dependency = _memory_dependency_token(state, issue_dependencies)
    source_base = _symbolic_buffer_base(
        state,
        source_base,
        element_type,
        attrs,
        op,
    )
    dest_base = _ptr_cast(
        state,
        dest_base,
        state.dsl.ptr_type(
            element_type,
            state.dsl.shared_address_space(),
        ),
    )
    element_byte_width = int(attrs["element_byte_width"])
    cache = _direct_buffer_load_cache_attr(state, attrs, op)
    gather = _prepare_symbolic_indexed_gather(
        state,
        offset_components,
        source_base,
        element_type,
        lane_width,
        element_byte_width,
        offset_range=_symbolic_buffer_offset_range(attrs),
        op=op,
        dependency=dependency,
        element_contiguity=int(attrs.get("contiguity", 1)),
        cache=cache,
    )
    if mask_components is None:
        packet, load_token = gather()
    else:
        packet_type = state.dsl.simd_type(
            state.dsl.vector_type(component_count, element_type),
            width=lane_width,
        )
        inactive_component = _zero_simd_value(
            state,
            state.dsl.simd_type(element_type, lane_width),
            attrs["element_type"],
            op,
        )
        inactive_packet = state.dsl.wave.PackOp(
            packet_type,
            [inactive_component] * component_count,
        ).result
        packet, load_token = _emit_masked_memory_value_region(
            state,
            _symbolic_mask_conditions(state, mask_components),
            packet_type,
            inactive_packet,
            dependency,
            gather,
        )
    destination_bit_offset = _symbolic_local_destination_bit_offset(
        state,
        attrs,
        component_count,
        element_byte_width,
        op,
    )
    token = state.builder.scatter(
        packet,
        [dest_base],
        bit_offset=destination_bit_offset,
        after=load_token,
    )

    result_id = _single_result(op)
    state.values[result_id] = token


def _emit_async_commit_group(state, op):
    tokens = tuple(_require_value(state, target_value_id, op) for target_value_id in op.operands)
    if tokens:
        token = state.builder.join(*tokens)
    else:
        token = state.builder.token()
    if op.results:
        result_id = _single_result(op)
        state.values[result_id] = token


def _emit_token(state, op):
    state.values[_single_result(op)] = state.builder.token()


def _emit_token_join(state, op):
    tokens = tuple(_require_value(state, target_value_id, op) for target_value_id in op.operands)
    state.values[_single_result(op)] = _join_memory_tokens(state, tokens)


def _emit_issue_token(state, op):
    tokens = tuple(_require_value(state, target_value_id, op) for target_value_id in op.operands)
    state.values[_single_result(op)] = state.builder.issue_token(*tokens)


def _join_memory_tokens(state, tokens):
    tokens = _unique_tokens(tokens)
    if not tokens:
        return state.builder.token()
    if len(tokens) == 1:
        return tokens[0]
    return state.builder.join(*tokens)


def _barrier_order_dependency(state, op, ordinary_operand_count):
    attrs = target_ir.attrs_dict(op)
    count = int(attrs.get("barrier_order_dependency_count", 0))
    if count not in {0, 1} or len(op.operands) != int(ordinary_operand_count) + count:
        fail(
            "TLXW_EMIT_BARRIER_ORDER_SEGMENT",
            STAGE,
            "memory barrier-order operand segment is malformed",
            target_op_id=op.target_op_id,
        )
    if not count:
        return None
    return _require_value(state, op.operands[-1], op)


def _issue_order_result_ids(op):
    attrs = target_ir.attrs_dict(op)
    count = int(attrs.get("issue_order_result_count", 0))
    if count not in {0, 1} or count > len(op.results):
        fail(
            "TLXW_EMIT_ISSUE_ORDER_RESULT",
            STAGE,
            "memory issue-order result segment is malformed",
            target_op_id=op.target_op_id,
        )
    if not count:
        return tuple(int(result_id) for result_id in op.results), ()
    return (
        tuple(int(result_id) for result_id in op.results[:-1]),
        (int(op.results[-1]), ),
    )


def _finish_issue_order_result(state, op, tokens):
    _data_result_ids, issue_result_ids = _issue_order_result_ids(op)
    if not issue_result_ids:
        return
    state.values[issue_result_ids[0]] = _join_memory_tokens(state, tokens)


def _memory_dependency_token(state, tokens):
    tokens = _unique_tokens(tokens)
    if not tokens:
        return state.builder.token()
    return _join_memory_tokens(state, tokens)


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
    if (completed_count < 0 or retained_issue_count < 0 or completed_count + retained_issue_count != len(op.operands)):
        fail(
            "TLXW_EMIT_ASYNC_WAIT_OPERANDS",
            STAGE,
            "async_wait dependency counts do not match its target operands",
            target_op_id=op.target_op_id,
        )
    completed_target_ids = op.operands[:completed_count]
    retained_issue_target_ids = op.operands[completed_count:]
    completed_tokens = tuple(_require_value(state, target_value_id, op) for target_value_id in completed_target_ids)
    retained_issue_tokens = tuple(
        _require_value(state, target_value_id, op) for target_value_id in retained_issue_target_ids)
    # The issue projection keeps retained groups ordered without contributing
    # their completion: Wave's token analysis stops at wave.issue_token.
    dependencies = (*completed_tokens, *retained_issue_tokens)
    if not dependencies:
        token = state.builder.token()
    else:
        token = state.builder.after(*dependencies)
    if op.results:
        result_id = _single_result(op)
        state.values[result_id] = token


def _emit_symbolic_local_gather_packets(
    state,
    op,
    attrs,
    base,
    element_type,
    dependency,
):
    lane_width = int(attrs["lane_width"])
    packet_width = int(attrs["packet_width"])
    base_type = state.dsl.ptr_type(
        element_type,
        state.dsl.shared_address_space(),
    )
    base = _ptr_cast(state, base, base_type)
    load_type = state.dsl.simd_type(
        state.dsl.vector_type(packet_width, element_type),
        width=lane_width,
    )
    item = state.dsl.sym("item")
    slot = state.dsl.sym("slot")
    payloads = []
    load_tokens = []
    packet_bases = tuple(tuple(int(value) for value in base) for base in attrs["packet_coordinate_bases"])
    if len(packet_bases) != int(attrs["packet_count"]):
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "local_load packet bases do not match its packet count",
            target_op_id=op.target_op_id,
        )
    for packet_base in packet_bases:
        payload, token = state.builder.gather(
            [base],
            load_type,
            bit_offset=_symbolic_local_packet_bit_offset(
                state,
                attrs,
                packet_base,
                op,
                item=item,
                slot=slot,
            ),
            after=dependency,
        )
        payloads.append(payload)
        load_tokens.append(token)
    return tuple(payloads), _join_memory_tokens(state, load_tokens)


def _symbolic_local_packet_bit_offset(
    state,
    attrs,
    packet_base,
    op,
    *,
    item=None,
    slot=None,
):
    item = state.dsl.sym("item") if item is None else item
    slot = state.dsl.sym("slot") if slot is None else slot
    logical_coords = _bit_linear_packet_coordinate_exprs(
        state,
        attrs,
        tuple(int(value) for value in packet_base),
        item,
        slot,
        op,
    )
    element_offset = _local_physical_element_offset_from_coords_expr(
        state,
        attrs,
        logical_coords,
        op,
    )
    return (_element_byte_width(attrs["element_type"], op) * 8 * element_offset).simplify()


def _bit_linear_packet_coordinate_exprs(
    state,
    attrs,
    component_base,
    item,
    slot,
    op,
):
    shape = tuple(int(dim) for dim in attrs["coordinate_shape"])
    slot_coefficients = tuple(tuple(int(value) for value in basis) for basis in attrs["slot_coordinate_coefficients"])
    item_coefficients = tuple(
        tuple(int(value) for value in basis) for basis in attrs["workitem_coordinate_coefficients"])
    if (len(component_base) != len(shape) or any(len(basis) != len(shape) for basis in slot_coefficients)
            or any(len(basis) != len(shape) for basis in item_coefficients)):
        fail(
            "TLXW_EMIT_BAD_COORDINATES",
            STAGE,
            "symbolic local-memory coordinates do not match the tensor rank",
            target_op_id=op.target_op_id,
        )
    coords = []
    for dim, base in enumerate(component_base):
        coord = state.dsl.sym_ctx.int_(int(base))
        for bit, basis in enumerate(slot_coefficients):
            coefficient = int(basis[dim])
            if not coefficient:
                continue
            value = state.dsl.mod(state.dsl.floor(slot / (1 << bit)), 2)
            if coefficient != 1:
                value *= coefficient
            coord = state.dsl.xor(coord, value)
        for bit, basis in enumerate(item_coefficients):
            coefficient = int(basis[dim])
            if not coefficient:
                continue
            value = state.dsl.mod(state.dsl.floor(item / (1 << bit)), 2)
            if coefficient != 1:
                value *= coefficient
            coord = state.dsl.xor(coord, value)
        coords.append(coord)
    return tuple(coords)


def _local_physical_element_offset_from_coords_expr(
    state,
    attrs,
    logical_coords,
    op,
):
    shape = tuple(int(dim) for dim in attrs["coordinate_shape"])
    physical_shape = tuple(int(dim) for dim in attrs["memdesc_physical_shape"])
    logical_origin = tuple(int(value) for value in attrs["memdesc_logical_origin"])
    if not (len(logical_coords) == len(shape) == len(physical_shape) == len(logical_origin)):
        fail(
            "TLXW_EMIT_BAD_COORDINATES",
            STAGE,
            "symbolic local-memory view ranks do not match",
            target_op_id=op.target_op_id,
        )
    physical_coords = tuple(coord + int(origin) for coord, origin in zip(logical_coords, logical_origin))
    logical = _linearize_local_fragment_coords(
        physical_shape,
        physical_coords,
    )
    plan = attrs.get("shared_physical_offset_plan")
    if plan == "dense_row_major":
        return logical
    if plan == "swizzled_xor":
        return _swizzled_element_offset_expr(state, attrs, logical, op)
    if plan == "linear_shared":
        return _linear_shared_element_offset_expr(state, attrs, logical, op)
    if plan == "padded_linear":
        return _padded_element_offset_expr(state, attrs, logical, op)
    fail(
        "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported symbolic local-memory physical offset plan {plan}",
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


def _swizzled_element_offset_expr(state, attrs, logical, op):
    shape = tuple(
        int(dim) for dim in attrs.get(
            "memdesc_physical_shape",
            attrs.get("memdesc_shape", attrs["source_shape"]),
        ))
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
        int(dim) for dim in attrs.get(
            "memdesc_physical_shape",
            attrs.get("memdesc_shape", attrs["source_shape"]),
        ))
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
        int(dim) for dim in attrs.get(
            "memdesc_physical_shape",
            attrs.get("memdesc_shape", attrs["source_shape"]),
        ))
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
    bases = tuple(tuple(int(value) for value in dim_bases) for dim_bases in attrs.get(key, ()))
    if len(bases) != int(rank):
        fail(
            diagnostic,
            STAGE,
            "shared_linear inverse offset bases do not match coordinate rank; "
            f"{key}={bases}, rank={rank}",
            target_op_id=op.target_op_id,
        )
    return bases


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
        if (payload is None or int(payload[0]) != registers or str(payload[1]) != "f32"
                or int(payload[2]) != lane_width):
            fail(
                "TLXW_EMIT_FRAGMENT_TYPE",
                STAGE,
                "MMA packet truncf requires an f32 SIMD packet with the "
                "planned register and lane widths",
                target_op_id=op.target_op_id,
            )
        packed.append(state.builder.fpconvert(regs, result_regs))
    state.values[_single_result(op)] = _pack_components(tuple(packed))


def _packet_layout(state, target_value_id, op):
    target_value = state.target_program.values[int(target_value_id)]
    layout_id = target_value.layout_map_id
    if layout_id is None or not 0 <= int(layout_id) < len(state.target_program.layouts):
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            "layout transform value is missing its symbolic layout",
            target_op_id=op.target_op_id,
            target_value_id=int(target_value_id),
        )
    layout = state.target_program.layouts[int(layout_id)]
    linear = layout.linear_layout
    if linear is None:
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            "layout transform requires a symbolic linear layout",
            target_op_id=op.target_op_id,
            target_value_id=int(target_value_id),
        )
    try:
        return state.dsl.PacketLayout(
            int(layout.lane_width),
            tuple((str(name), int(size)) for name, size in linear.out_dims),
            tuple((
                str(name),
                tuple(tuple(int(component) for component in basis) for basis in bases),
            ) for name, bases in linear.bases),
        )
    except ValueError as exc:
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            str(exc),
            target_op_id=op.target_op_id,
            target_value_id=int(target_value_id),
        )


def _layout_packet_element_type(state, target_type, slot_count, op):
    mask_value = target_type.representation in {"mask", "mask_tuple"}
    pointer_payload = target_type.representation in {
        "buffer_pointer",
        "buffer_pointer_tuple",
        "per_lane_pointer",
        "pointer_tuple",
    }
    if mask_value:
        return state.dsl.i32(), True
    if pointer_payload:
        if int(slot_count) != 1:
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "pointer layout transforms require one packet slot",
                target_op_id=op.target_op_id,
            )
        address_space = (state.dsl.buffer_address_space() if target_type.representation
                         in {"buffer_pointer", "buffer_pointer_tuple"} else state.dsl.global_address_space())
        return (
            state.dsl.ptr_type(
                _scalar_type(state.dsl, target_type.element_type),
                address_space,
            ),
            False,
        )
    return _scalar_type(state.dsl, target_type.element_type), False


def _layout_packet_type(state, target_type, layout, op):
    element_type, _mask_value = _layout_packet_element_type(state, target_type, layout.slot_count, op)
    payload_type = (element_type if int(layout.slot_count) == 1 else state.dsl.vector_type(
        int(layout.slot_count), element_type))
    return state.dsl.simd_type(payload_type, int(layout.lane_width))


def _pack_layout_value(state, op, target_value_id, value, layout):
    target_type = state.target_program.values[int(target_value_id)].type
    component_count = int(target_type.component_count)
    slot_count = int(layout.slot_count)
    if component_count <= 0 or slot_count % component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "layout packet slots do not evenly partition bridge components",
            target_op_id=op.target_op_id,
            target_value_id=int(target_value_id),
        )
    width = slot_count // component_count
    components = _value_components(state, value, op)
    if len(components) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "layout packet source components do not match its target type",
            target_op_id=op.target_op_id,
            target_value_id=int(target_value_id),
        )
    element_type, mask_value = _layout_packet_element_type(state, target_type, slot_count, op)
    if mask_value:
        components = tuple(
            _mask_to_redistribution_value(state, component, int(layout.lane_width)) for component in components)
    chunk_type = state.dsl.simd_type(
        element_type if width == 1 else state.dsl.vector_type(width, element_type),
        int(layout.lane_width),
    )
    chunks = tuple(
        _redistribution_component_chunk(
            state,
            component,
            chunk_type,
            width,
            op,
        ) for component in components)
    if slot_count == 1:
        return chunks[0]
    return state.dsl.wave.PackOp(
        _layout_packet_type(state, target_type, layout, op),
        chunks,
    ).result


def _unpack_layout_value(state, op, target_value_id, packet, layout):
    target_type = state.target_program.values[int(target_value_id)].type
    component_count = int(target_type.component_count)
    slot_count = int(layout.slot_count)
    if component_count <= 0 or slot_count % component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "layout result slots do not evenly partition bridge components",
            target_op_id=op.target_op_id,
            target_value_id=int(target_value_id),
        )
    width = slot_count // component_count
    element_type, mask_value = _layout_packet_element_type(state, target_type, slot_count, op)
    chunk_type = state.dsl.simd_type(
        element_type if width == 1 else state.dsl.vector_type(width, element_type),
        int(layout.lane_width),
    )
    if slot_count == 1:
        components = (packet, )
    else:
        components = tuple(
            state.dsl.wave.ExtractOp(
                chunk_type,
                packet,
                component * width,
            ).result for component in range(component_count))
    if mask_value:
        components = tuple(
            _redistribution_value_to_mask(state, component, int(layout.lane_width)) for component in components)
    state.values[int(target_value_id)] = _pack_components(components)


def _propagate_common_uniform_pointer_base(state, source_ids, result_ids):
    source_bases = tuple(state.uniform_pointer_bases.get(int(source_id)) for source_id in source_ids)
    if not source_bases or any(bases is None or not bases for bases in source_bases):
        return
    first = source_bases[0][0]
    if any(base is not first for bases in source_bases for base in bases):
        return
    for result_id in result_ids:
        count = _component_count(state, int(result_id))
        state.uniform_pointer_bases[int(result_id)] = (first, ) * count


def _emit_packet_layout_transform(state, op, transform):
    (value, ) = _operand_values(state, op, 1)
    source_id = int(op.operands[0])
    result_id = int(_single_result(op))
    source_layout = _packet_layout(state, source_id, op)
    result_layout = _packet_layout(state, result_id, op)
    source_packet = _pack_layout_value(state, op, source_id, value, source_layout)
    result_type = _layout_packet_type(
        state,
        state.target_program.values[result_id].type,
        result_layout,
        op,
    )
    try:
        result_packet = state.builder.redistribute_layout(
            source_packet,
            result_type,
            source_layout=source_layout,
            result_layout=result_layout,
            transform=transform,
        )
    except ValueError as exc:
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            str(exc),
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    _unpack_layout_value(state, op, result_id, result_packet, result_layout)
    _propagate_common_uniform_pointer_base(state, (source_id, ), (result_id, ))


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


def _emit_layout_convert(state, op):
    attrs = target_ir.attrs_dict(op)
    transform = str(attrs.get("transform", "identity"))
    if transform == "identity":
        packet_transform = state.dsl.PacketTransform.identity()
    elif transform == "reshape":
        packet_transform = state.dsl.PacketTransform.reshape()
    elif transform == "trans":
        packet_transform = state.dsl.PacketTransform.transpose(tuple(int(dim) for dim in attrs.get("order", ())))
    else:
        fail(
            "TLXW_EMIT_UNSUPPORTED_LAYOUT_CONVERT",
            STAGE,
            f"unsupported layout transform {transform!r}",
            target_op_id=op.target_op_id,
        )
    _emit_packet_layout_transform(state, op, packet_transform)


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


def _symbolic_buffer_base(state, source_base, element_type, attrs, op):
    """Preserve an amdg.buffer_* resource while Wave plans its accesses."""
    range_bytes = int(attrs["range_bytes"])
    if range_bytes <= 0 or range_bytes > (1 << 31) - 1:
        fail(
            "TLXW_EMIT_INVALID_BUFFER_RANGE",
            STAGE,
            "buffer memory requires a positive signed-i32 byte range",
            target_op_id=op.target_op_id,
        )
    return state.builder.make_buffer(
        source_base,
        state.builder.constant(state.dsl.i32(), range_bytes),
        result_type=state.dsl.buffer_ptr_type(element_type),
    )


def _buffer_memory_masks(
    masks,
    component_count,
    op,
):
    if masks is None:
        return None
    return _as_mask_components(
        masks,
        component_count,
        op,
    )


def _flatten_buffer_store_values(
    state,
    value,
    component_count,
    access_component_count,
    element_type,
    lane_width,
    op,
):
    logical_components = _value_components(state, value, op)
    if len(logical_components) != int(component_count):
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_store value component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    scalar_type = state.dsl.simd_type(element_type, int(lane_width))
    scalar_components = []
    for component in logical_components:
        payload = _simd_1d_vector_payload(state, component)
        if payload is None:
            scalar_components.append(component)
            continue
        width, payload_element_type, payload_lane_width = payload
        if (str(payload_element_type) != str(element_type) or int(payload_lane_width) != int(lane_width)):
            fail(
                "TLXW_EMIT_UNSUPPORTED_BUFFER_STORE",
                STAGE,
                "buffer_store vector payload type does not match its element type",
                target_op_id=op.target_op_id,
            )
        scalar_components.extend(
            state.dsl.wave.ExtractOp(scalar_type, component, element).result for element in range(int(width)))
    if len(scalar_components) != int(access_component_count):
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_store scalar packet does not match its access components",
            target_op_id=op.target_op_id,
        )
    return tuple(scalar_components)


def _emit_buffer_store(state, op):
    attrs = target_ir.attrs_dict(op)
    has_mask = bool(attrs["has_mask"])
    expected_operand_count = 3 + int(has_mask)
    dependency = _barrier_order_dependency(state, op, expected_operand_count)
    _data_result_ids, issue_result_ids = _issue_order_result_ids(op)
    capture_token = bool(issue_result_ids)
    value = _require_value(state, op.operands[0], op)
    source_base = _require_value(state, op.operands[1], op)
    offsets = _require_value(state, op.operands[2], op)
    masks = _require_value(state, op.operands[3], op) if has_mask else None
    lane_width = int(attrs["lane_width"])
    component_count = int(attrs["component_count"])
    access_component_count = int(attrs.get("access_component_count", component_count))
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    offset_components = _as_components(offsets)
    if len(offset_components) != access_component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_store offset component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    value_components = _flatten_buffer_store_values(
        state,
        value,
        component_count,
        access_component_count,
        element_type,
        lane_width,
        op,
    )
    mask_components = _buffer_memory_masks(
        masks,
        access_component_count,
        op,
    )
    redundant_register_mask = int(attrs.get("redundant_register_mask", 0))
    canonical_components = tuple(range(access_component_count))
    if redundant_register_mask:
        canonical_components = tuple(component for component in range(access_component_count)
                                     if (component & redundant_register_mask) == 0)
        value_components = tuple(value_components[component] for component in canonical_components)
        offset_components = tuple(offset_components[component] for component in canonical_components)
        if mask_components is not None:
            mask_components = tuple(mask_components[component] for component in canonical_components)
    if has_mask and attrs.get("mask_mode", "exec_where") != "exec_where":
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_STORE_MASK",
            STAGE,
            f"unsupported buffer_store mask mode {attrs.get('mask_mode')}",
            target_op_id=op.target_op_id,
        )
    buffer_base = _symbolic_buffer_base(state, source_base, element_type, attrs, op)
    cache = _direct_buffer_store_cache_attr(state, attrs, op)
    emit_scatter = _prepare_symbolic_indexed_scatter(
        state,
        value_components,
        offset_components,
        buffer_base,
        element_type,
        lane_width,
        int(attrs["element_byte_width"]),
        offset_range=_symbolic_buffer_offset_range(attrs),
        op=op,
        dependency=dependency,
        element_contiguity=int(attrs.get("contiguity", 1)),
        element_component_indices=canonical_components,
        cache=cache,
    )

    owner_condition = _buffer_store_owner_condition(
        state,
        attrs,
        lane_width,
        op,
    )
    if mask_components is None and owner_condition is None:
        token = emit_scatter()
    else:
        if mask_components is None:
            predicate_conditions = owner_condition
        else:
            if owner_condition is not None:
                mask_components = tuple(
                    _mask_and(
                        state,
                        component,
                        owner_condition,
                        lane_width,
                        op,
                    ) for component in mask_components)
            predicate_conditions = _symbolic_mask_conditions(
                state,
                mask_components,
            )
        if capture_token:
            token = _emit_masked_token_region(
                state,
                predicate_conditions,
                dependency or state.builder.token(),
                emit_scatter,
            )
        else:
            token = _emit_masked_effect_region(
                state,
                predicate_conditions,
                emit_scatter,
            )
    _finish_issue_order_result(
        state,
        op,
        () if token is None else (token, ),
    )


def _buffer_store_owner_condition(state, attrs, lane_width, op):
    lane_mask = int(attrs.get("redundant_lane_mask", 0))
    wave_mask = int(attrs.get("redundant_wave_mask", 0))
    if not lane_mask and not wave_mask:
        return None
    workitem = state.builder.workitem_id(
        0,
        state.dsl.i32(),
        lane_width,
    )
    owner_condition = None
    if lane_mask:
        lane_id = _simd_binary_const(
            state,
            "remui",
            workitem,
            lane_width,
            lane_width,
        )
        masked_lane_id = _simd_binary_const(
            state,
            "andi",
            lane_id,
            lane_mask,
            lane_width,
        )
        owner_condition = _cmpi(
            state,
            "eq",
            masked_lane_id,
            _simd_i32_constant(state, lane_width, 0),
        )
    if wave_mask:
        wave_id = _simd_binary_const(
            state,
            "divui",
            workitem,
            lane_width,
            lane_width,
        )
        masked_wave_id = _simd_binary_const(
            state,
            "andi",
            wave_id,
            wave_mask,
            lane_width,
        )
        wave_owner = _cmpi(
            state,
            "eq",
            masked_wave_id,
            _simd_i32_constant(state, lane_width, 0),
        )
        owner_condition = (wave_owner if owner_condition is None else _mask_and(
            state,
            owner_condition,
            wave_owner,
            lane_width,
            op,
        ))
    return owner_condition


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


def _reconstruct_buffer_load_value(
    state,
    attrs,
    target_type,
    scalar_components,
    component_count,
    access_component_count,
    element_type,
    lane_width,
    op,
):
    scalar_components = tuple(scalar_components)
    if len(scalar_components) != int(access_component_count):
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_load scalar packet does not match its access components",
            target_op_id=op.target_op_id,
        )
    if int(access_component_count) % int(component_count):
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_load access components do not evenly cover result components",
            target_op_id=op.target_op_id,
        )
    payload_width = int(access_component_count) // int(component_count)
    if target_type.representation in _MMA_PACKET_REPRESENTATIONS:
        packet_type = state.dsl.simd_type(
            state.dsl.vector_type(payload_width, element_type),
            width=int(lane_width),
        )
        return _pack_components(
            tuple(
                state.dsl.wave.PackOp(
                    packet_type,
                    scalar_components[index:index + payload_width],
                ).result for index in range(0, int(access_component_count), payload_width)))
    if payload_width != 1:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "ordinary buffer_load result components must have scalar payloads",
            target_op_id=op.target_op_id,
        )
    return _pack_components(scalar_components)


def _emit_buffer_load(state, op):
    attrs = target_ir.attrs_dict(op)
    has_mask = bool(attrs["has_mask"])
    operand_count = 2 + int(has_mask) + int(bool(attrs["has_other"]))
    dependency = _barrier_order_dependency(state, op, operand_count)
    source_base = _require_value(state, op.operands[0], op)
    offsets = _require_value(state, op.operands[1], op)
    operand_index = 2
    masks = _require_value(state, op.operands[operand_index], op) if has_mask else None
    operand_index += int(has_mask)
    other = (_require_value(state, op.operands[operand_index], op) if attrs["has_other"] else None)
    component_count = int(attrs["component_count"])
    access_component_count = int(attrs.get("access_component_count", component_count))
    lane_width = int(attrs["lane_width"])
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    offset_components = _as_components(offsets)
    if len(offset_components) != access_component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_load offset component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    mask_components = _buffer_memory_masks(
        masks,
        access_component_count,
        op,
    )
    if has_mask and attrs.get("mask_mode", "exec_where") != "exec_where":
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD_MASK",
            STAGE,
            f"unsupported buffer_load mask mode {attrs.get('mask_mode')}",
            target_op_id=op.target_op_id,
        )
    fallback_components = None
    scalar_result_type = state.dsl.simd_type(element_type, lane_width)
    if other is not None:
        fallback_components = _broadcast_component(
            state,
            other,
            access_component_count,
            op,
        )
        splat_cache = []
        fallback_components = tuple(
            _memory_simd_component(
                state,
                component,
                attrs["element_type"],
                lane_width,
                op,
                splat_cache,
            ) for component in fallback_components)
    elif mask_components is not None:
        zero = _zero_simd_value(
            state,
            scalar_result_type,
            attrs["element_type"],
            op,
        )
        fallback_components = tuple(zero for _ in range(access_component_count))
    buffer_base = _symbolic_buffer_base(state, source_base, element_type, attrs, op)
    cache = _direct_buffer_load_cache_attr(state, attrs, op)
    packet_type = state.dsl.simd_type(
        state.dsl.vector_type(access_component_count, element_type),
        width=lane_width,
    )

    emit_gather = _prepare_symbolic_indexed_gather(
        state,
        offset_components,
        buffer_base,
        element_type,
        lane_width,
        int(attrs["element_byte_width"]),
        offset_range=_symbolic_buffer_offset_range(attrs),
        op=op,
        dependency=dependency,
        element_contiguity=int(attrs.get("contiguity", 1)),
        cache=cache,
    )

    if mask_components is None:
        packet, token = emit_gather()
    else:
        fallback_packet = state.dsl.wave.PackOp(
            packet_type,
            fallback_components,
        ).result
        predicate_conditions = _symbolic_mask_conditions(
            state,
            mask_components,
        )
        packet, token = _emit_masked_memory_value_region(
            state,
            predicate_conditions,
            packet_type,
            fallback_packet,
            dependency,
            emit_gather,
        )
    loaded_components = _extract_packet_components(
        state,
        packet,
        element_type,
        lane_width,
        access_component_count,
    )
    data_result_ids, _issue_result_ids = _issue_order_result_ids(op)
    if len(data_result_ids) != 1:
        fail(
            "TLXW_EMIT_ISSUE_ORDER_RESULT",
            STAGE,
            "buffer_load requires one data result",
            target_op_id=op.target_op_id,
        )
    result_id = data_result_ids[0]
    target_type = state.target_program.values[result_id].type
    state.values[result_id] = _reconstruct_buffer_load_value(
        state,
        attrs,
        target_type,
        loaded_components,
        component_count,
        access_component_count,
        element_type,
        lane_width,
        op,
    )
    _finish_issue_order_result(state, op, (token, ))


def _emit_store(state, op):
    attrs = target_ir.attrs_dict(op)
    operand_count = 3 if attrs["has_mask"] else 2
    dependency = _barrier_order_dependency(state, op, operand_count)
    operands = _operand_values(state, op, len(op.operands))[:operand_count]
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
        mask_components = _as_mask_components(
            masks,
            component_count,
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
    if mask_components is not None and mask_mode != "exec_where":
        fail(
            "TLXW_EMIT_UNSUPPORTED_STORE_MASK",
            STAGE,
            f"unsupported store mask mode {mask_mode}",
            target_op_id=op.target_op_id,
        )
    _data_result_ids, issue_result_ids = _issue_order_result_ids(op)
    capture_token = bool(issue_result_ids)
    lane_width = int(attrs["lane_width"])
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    packet_type = state.dsl.simd_type(
        state.dsl.vector_type(component_count, element_type),
        width=lane_width,
    )
    packet = state.dsl.wave.PackOp(packet_type, value_components).result
    slot = state.dsl.sym("slot")
    zero = state.dsl.sym_ctx.int_(0)

    def emit_store(packet_conditions=()):
        return state.builder.scatter(
            packet,
            ptr_components,
            base=slot,
            bit_offset=zero,
            packet_conditions=packet_conditions,
            after=dependency,
        )

    if mask_components is None:
        token = emit_store()
    else:
        condition = _symbolic_mask_conditions(state, mask_components)
        if capture_token:
            token = _emit_masked_token_region(
                state,
                condition,
                dependency or state.builder.token(),
                emit_store,
            )
        else:
            _emit_masked_effect_region(state, condition, emit_store)
            token = None
    _finish_issue_order_result(state, op, () if token is None else (token, ))


def _emit_load(state, op):
    attrs = target_ir.attrs_dict(op)
    operand_count = 1 + int(bool(attrs["has_mask"])) + int(bool(attrs["has_other"]))
    dependency = _barrier_order_dependency(state, op, operand_count)
    operands = _operand_values(state, op, len(op.operands))[:operand_count]
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
        mask_components = _as_mask_components(
            masks,
            component_count,
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
    data_result_ids, _issue_result_ids = _issue_order_result_ids(op)
    if len(data_result_ids) != 1:
        fail(
            "TLXW_EMIT_ISSUE_ORDER_RESULT",
            STAGE,
            "load requires one data result",
            target_op_id=op.target_op_id,
        )
    result_id = data_result_ids[0]
    mask_mode = attrs.get("mask_mode", "exec_where" if attrs["has_mask"] else "none")
    if mask_components is not None and mask_mode != "exec_where":
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOAD_MASK",
            STAGE,
            f"unsupported load mask mode {mask_mode}",
            target_op_id=op.target_op_id,
        )
    lane_width = int(attrs["lane_width"])
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    component_type = state.dsl.simd_type(element_type, lane_width)
    packet_type = state.dsl.simd_type(
        state.dsl.vector_type(component_count, element_type),
        width=lane_width,
    )
    slot = state.dsl.sym("slot")
    zero = state.dsl.sym_ctx.int_(0)

    def emit_active_load(packet_conditions=()):
        return state.builder.gather(
            ptr_components,
            packet_type,
            base=slot,
            bit_offset=zero,
            packet_conditions=packet_conditions,
            after=dependency,
        )

    if mask_components is None:
        packet, token = emit_active_load()
    else:
        if other_components is None:
            fallback = _zero_simd_value(
                state,
                component_type,
                attrs["element_type"],
                op,
            )
            fallback_components = (fallback, ) * component_count
        else:
            fallback_components = other_components
        fallback_packet = state.dsl.wave.PackOp(
            packet_type,
            fallback_components,
        ).result
        packet, token = _emit_masked_memory_value_region(
            state,
            _symbolic_mask_conditions(state, mask_components),
            packet_type,
            fallback_packet,
            dependency,
            emit_active_load,
        )
    loaded_components = _symbolic_shared_packet_components(
        state,
        packet,
        component_count,
        element_type,
        lane_width,
    )
    state.values[result_id] = _pack_components(tuple(loaded_components))
    _finish_issue_order_result(state, op, (token, ))


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


def _explicit_packet_conditions(condition):
    if isinstance(condition, (tuple, list)) and len(condition) > 1:
        return tuple(condition)
    return ()


def _emit_masked_effect_region(state, condition, emit_body):
    if not isinstance(condition, (tuple, list)) and _is_scalar_i1_value(state, condition):
        with state.builder.if_(condition):
            emit_body()
        return
    packet_conditions = _explicit_packet_conditions(condition)
    with state.builder.where(condition):
        emit_body(packet_conditions)


def _emit_masked_token_region(state, condition, inactive_token, emit_body):
    result_type = state.dsl.mem_token_type()
    if not isinstance(condition, (tuple, list)) and _is_scalar_i1_value(state, condition):
        with state.builder.if_(condition, [result_type], otherwise=True) as ifop:
            state.builder.yield_([emit_body()])
            with ifop.otherwise():
                state.builder.yield_([inactive_token])
        return ifop.results[0]
    packet_conditions = _explicit_packet_conditions(condition)
    with state.builder.where(condition, [result_type]) as where:
        state.builder.yield_([emit_body(packet_conditions)])
    with where.otherwise():
        state.builder.yield_([inactive_token])
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
    if not isinstance(condition, (tuple, list)) and _is_scalar_i1_value(state, condition):
        with state.builder.if_(condition, result_types, otherwise=True) as ifop:
            value, token = emit_body()
            state.builder.yield_([value, token])
            with ifop.otherwise():
                state.builder.yield_([inactive_value, inactive_token])
        return tuple(ifop.results)
    packet_conditions = _explicit_packet_conditions(condition)
    with state.builder.where(condition, result_types) as where:
        value, token = emit_body(packet_conditions)
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
    "type_convert": _emit_type_convert,
    "binary": _emit_binary,
    "float_binary": _emit_float_binary,
    "float_unary": _emit_float_unary,
    "float_cast": _emit_float_cast,
    "cmpf": _emit_cmpf,
    "cmpi": _emit_cmpi,
    "maxsi": _emit_maxsi,
    "minsi": _emit_minsi,
    "assume": _emit_assume,
    "make_range": _emit_make_range,
    "splat": _emit_splat,
    "broadcast": _emit_broadcast,
    "join": _emit_join,
    "split": _emit_split,
    "addptr": _emit_addptr,
    "make_buffer": _emit_make_buffer,
    "expand_dims": _emit_expand_dims,
    "program_id": _emit_program_id,
    "warp_id": _emit_warp_id,
    "ballot": _emit_ballot,
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
    return len(lhs) == len(rhs) and all(lhs_operand is rhs_operand for lhs_operand, rhs_operand in zip(lhs, rhs))


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


def _add_constant_to_optional_i32_offset(state, value, constant):
    constant = int(constant)
    if constant == 0:
        return value
    constant_value = state.builder.constant(state.dsl.i32(), constant)
    return _combine_optional_i32_offsets(
        state,
        value,
        constant_value,
        nsw=_LAYOUT_MATH_NSW,
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
    offset_value = (state.builder.constant(state.dsl.i32(), int(offset)) if type(offset) is int else offset)
    result = state.builder.ptr_add(
        base_i32,
        offset_value,
        result_type=i32_shared,
    )
    state.shared_pointer_offset_cache[key] = result
    return result


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _target_lds_size(target_program):
    del target_program
    return 0


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
    packet_components = []
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
            packet_components.append(state.dsl.wave.ExtractOp(
                component_type,
                packet,
                int(element),
            ).result)
    if len(packet_components) != logical_component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "vector packet payload component count does not match its shape",
            target_op_id=op.target_op_id,
        )
    return tuple(packet_components)


def _pack_components(components):
    return components[0] if len(components) == 1 else tuple(components)


def _as_mask_components(value, count, op):
    components = _mask_components(value)
    return _broadcast_component_count(components, count, "mask", op)


def _mask_components(value):
    return tuple(_as_components(value))


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


def _mask_to_redistribution_value(state, component, lane_width):
    if _is_scalar_i1_value(state, component):
        scalar = state.builder.select(
            component,
            state.builder.constant(state.dsl.i32(), 1),
            state.builder.constant(state.dsl.i32(), 0),
        )
        return state.builder.splat(scalar, state.dsl.i32(), int(lane_width))
    return state.builder.select(
        component,
        _simd_i32_constant(state, lane_width, 1),
        _simd_i32_constant(state, lane_width, 0),
    )


def _redistribution_value_to_mask(state, component, lane_width):
    return _cmpi(
        state,
        "ne",
        component,
        _simd_i32_constant(state, lane_width, 0),
    )


def _mask_and(state, lhs_component, rhs_component, lane_width, op):
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
    if target_type.representation == "uniform_buffer_pointer":
        return dsl.buffer_ptr_type(_scalar_type(dsl, target_type.element_type))
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
    if target_type.representation in {
            "buffer_pointer",
            "buffer_pointer_tuple",
    }:
        return dsl.simd_ptr_type(
            _scalar_type(dsl, target_type.element_type),
            dsl.buffer_address_space(),
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
    except Exception:
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


def _cmpf(state, predicate_name, lhs, rhs):
    predicate = state.dsl.CmpFPredicate[predicate_name.upper()]
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
        return state.builder.cmpf(predicate, lhs, rhs)
    return state.dsl.arith.CmpFOp(predicate, lhs, rhs).result


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
        "wave.address_arithmetic_no_overflow": ir.UnitAttr.get(),
        # gfx9/gfx950 exposes four SIMD execution units per CU. Model the
        # requested CTA waves as the resident wave target per SIMD.
        "waveamdmachine.target_waves": dsl.i64_attr(target_waves),
    }
    if kernel.enable_split_barriers:
        attrs["waveamdmachine.enable_split_barriers"] = ir.UnitAttr.get()
    if kernel.enable_multi_wave_specialization:
        attrs["waveamdmachine.enable_multi_wave_specialization"] = ir.UnitAttr.get()
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
