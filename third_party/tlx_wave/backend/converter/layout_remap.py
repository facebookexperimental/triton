"""Structural layout-remap helpers for TLX Wave conversion."""

from .diagnostics import fail
from . import layouts

STAGE = "op_conversion"

_DISTRIBUTED_REMAP_KINDS = frozenset({"blocked", "linear", "generic_linear"})
_DISTRIBUTED_REMAP_REPRESENTATIONS = frozenset({
    "mask",
    "mask_tuple",
    "per_lane_pointer",
    "pointer_tuple",
    "simd",
    "simd_tuple",
})
_MMA_PACKET_REPRESENTATIONS = frozenset({
    "simd_packet",
    "simd_packet_tuple",
})
_STRUCTURAL_TENSOR_REPRESENTATIONS = frozenset({
    "simd",
    "simd_tuple",
    *_MMA_PACKET_REPRESENTATIONS,
})


def redistribution_plan(operand, result, operand_layout, result_layout, op):
    """Build the complete destination-to-source packet relation.

    Layout coordinates are the semantic witness for a layout conversion.  The
    bridge carries a checked bit-linear gather relation to emission; movement
    classification and lowering belong to Wave's redistribute pass.
    """
    if operand_layout is None or result_layout is None:
        return None
    if operand.type.element_type != result.type.element_type:
        return None
    if tuple(operand_layout.shape) != tuple(result_layout.shape):
        return None

    source_layout = _redistribution_linear_layout(operand_layout, operand, op)
    destination_layout = _redistribution_linear_layout(result_layout, result, op)
    source_slots = layouts.linear_layout_in_dim_size(source_layout, "register")
    destination_slots = layouts.linear_layout_in_dim_size(
        destination_layout,
        "register",
    )
    source_components = int(operand.type.component_count)
    destination_components = int(result.type.component_count)
    if source_slots % source_components or destination_slots % destination_components:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "redistribution packet slots must evenly partition bridge components",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    source_warps = _redistribution_warp_count(operand_layout)
    destination_warps = _redistribution_warp_count(result_layout)
    if source_warps != destination_warps:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "redistribution source and destination workgroup sizes must match",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    source_blocks = layouts.linear_layout_in_dim_size(source_layout, "block")
    destination_blocks = layouts.linear_layout_in_dim_size(
        destination_layout,
        "block",
    )
    if source_blocks != destination_blocks:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "redistribution source and destination cluster sizes must match",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    relation_bases, relation_out_dims, is_identity, cross_wave = (
        _redistribution_relation_plan(
            source_layout,
            destination_layout,
            source_slots,
            destination_slots,
            lane_width,
            source_warps,
            source_blocks,
            len(result_layout.shape),
            op,
            operand.value_id,
            result.value_id,
        )
    )
    if is_identity and source_slots == destination_slots:
        packet_grouping = _identity_packet_grouping_plan(
            operand,
            result,
            source_slots,
            destination_slots,
            op,
        )
        if packet_grouping is not None:
            return {
                "group_size": 1,
                "mode": "alias",
                **packet_grouping,
            }
    if (
        is_identity
        and operand.type.representation == result.type.representation
        and source_components == destination_components
        and source_slots == destination_slots
    ):
        return {
            "group_size": 1,
            "mode": "alias",
            "result_component_count": destination_components,
        }
    return {
        "mode": "redistribute",
        "block_count": source_blocks,
        "cta_thread_count": lane_width * source_warps,
        "cross_wave": cross_wave,
        "element_type": result.type.element_type,
        "relation_bases": relation_bases,
        "relation_out_dims": relation_out_dims,
        "source_component_count": source_components,
        "source_registers_per_component": source_slots // source_components,
        "source_slot_count": source_slots,
        "result_component_count": destination_components,
        "result_registers_per_component": destination_slots // destination_components,
        "result_slot_count": destination_slots,
    }


def structural_view_alias_plan(
    operand,
    result,
    operand_layout,
    result_layout,
    op,
):
    """Prove and describe a zero-cost reshape/transpose packet regrouping.

    TritonGPU tensor views preserve the physical ``register/lane/warp/block``
    inputs while changing only their logical output coordinates.  Prove that
    relation explicitly, then describe any ordinary-SIMD packet grouping the
    Wave emitter must reconstruct at the view boundary.  WaveAMD fragment
    types are deliberately not part of this contract.
    """
    if operand_layout is None or result_layout is None:
        _structural_view_fail(
            op,
            result.value_id,
            f"{op.name} requires source and result distributed layouts",
        )
    if operand.type.kind != "tensor" or result.type.kind != "tensor":
        _structural_view_fail(
            op,
            result.value_id,
            f"{op.name} requires tensor operands",
        )
    if operand.type.element_type != result.type.element_type:
        _structural_view_fail(
            op,
            result.value_id,
            f"{op.name} changed its element type",
        )
    if int(operand.type.lane_width or 64) != int(result.type.lane_width or 64):
        _structural_view_fail(
            op,
            result.value_id,
            f"{op.name} changed its wave width",
        )
    source_shape = tuple(int(dim) for dim in operand_layout.shape)
    result_shape = tuple(int(dim) for dim in result_layout.shape)
    if _product(source_shape) != _product(result_shape):
        _structural_view_fail(
            op,
            result.value_id,
            f"{op.name} source and result element counts must match",
        )

    source_linear = _redistribution_linear_layout(operand_layout, operand, op)
    result_linear = _redistribution_linear_layout(result_layout, result, op)
    source_slots = layouts.linear_layout_in_dim_size(source_linear, "register")
    result_slots = layouts.linear_layout_in_dim_size(result_linear, "register")
    source_warps = _redistribution_warp_count(operand_layout)
    result_warps = _redistribution_warp_count(result_layout)
    source_blocks = layouts.linear_layout_in_dim_size(source_linear, "block")
    result_blocks = layouts.linear_layout_in_dim_size(result_linear, "block")
    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    if (
        source_slots != result_slots
        or source_warps != result_warps
        or source_blocks != result_blocks
    ):
        _structural_view_fail(
            op,
            result.value_id,
            f"{op.name} changed its physical register packet dimensions",
        )

    order = _structural_view_order(op, source_shape, result_shape)
    description = f"{op.name} structural packet alias"
    for block in range(int(source_blocks)):
        for warp in range(int(source_warps)):
            for lane in range(lane_width):
                for slot in range(int(source_slots)):
                    source_coords = _redistribution_coords(
                        source_linear,
                        slot,
                        lane,
                        warp,
                        block,
                        len(source_shape),
                        op,
                        operand.value_id,
                        description,
                    )
                    result_coords = _redistribution_coords(
                        result_linear,
                        slot,
                        lane,
                        warp,
                        block,
                        len(result_shape),
                        op,
                        result.value_id,
                        description,
                    )
                    expected = _structural_view_result_coords(
                        op.name,
                        source_coords,
                        source_shape,
                        result_shape,
                        order,
                    )
                    if result_coords != expected:
                        _structural_view_fail(
                            op,
                            result.value_id,
                            f"{op.name} changed physical register ownership",
                        )

    packet_grouping = _identity_packet_grouping_plan(
        operand,
        result,
        source_slots,
        result_slots,
        op,
    )
    if packet_grouping is None:
        _structural_view_fail(
            op,
            result.value_id,
            f"{op.name} has unsupported tensor packet representations "
            f"{operand.type.representation} -> {result.type.representation}",
        )
    return {
        "group_size": 1,
        "mode": "alias",
        **packet_grouping,
    }


def _identity_packet_grouping_plan(
    operand,
    result,
    source_slots,
    result_slots,
    op,
):
    source_components = int(operand.type.component_count)
    result_components = int(result.type.component_count)
    if source_components <= 0 or result_components <= 0:
        return None
    if source_slots % source_components or result_slots % result_components:
        return None
    source_width = int(source_slots) // source_components
    result_width = int(result_slots) // result_components
    source_representation = operand.type.representation
    result_representation = result.type.representation
    if (
        source_representation not in _STRUCTURAL_TENSOR_REPRESENTATIONS
        or result_representation not in _STRUCTURAL_TENSOR_REPRESENTATIONS
    ):
        return None
    if (
        source_representation not in _MMA_PACKET_REPRESENTATIONS
        and source_width != 1
    ):
        return None
    if (
        result_representation not in _MMA_PACKET_REPRESENTATIONS
        and result_width != 1
    ):
        return None
    return {
        "source_component_count": source_components,
        "source_packet_width": source_width,
        "source_slot_count": int(source_slots),
        "result_component_count": result_components,
        "result_packet_width": result_width,
        "result_slot_count": int(result_slots),
    }


def _structural_view_order(op, source_shape, result_shape):
    if op.name == "tt.reshape":
        return ()
    if op.name != "tt.trans":
        _structural_view_fail(
            op,
            None,
            f"unsupported structural tensor view {op.name}",
        )
    order = tuple(int(dim) for dim in op.attrs.get("order", ()))
    if sorted(order) != list(range(len(source_shape))):
        _structural_view_fail(
            op,
            None,
            "tt.trans requires a complete permutation",
        )
    if tuple(source_shape[dim] for dim in order) != tuple(result_shape):
        _structural_view_fail(
            op,
            None,
            "tt.trans permutation does not match the result shape",
        )
    return order


def _structural_view_result_coords(
    op_name,
    source_coords,
    source_shape,
    result_shape,
    order,
):
    if op_name == "tt.trans":
        return tuple(int(source_coords[dim]) for dim in order)
    linear = 0
    for coord, extent in zip(source_coords, source_shape):
        linear = linear * int(extent) + int(coord)
    result = [0] * len(result_shape)
    for dim in reversed(range(len(result_shape))):
        result[dim] = linear % int(result_shape[dim])
        linear //= int(result_shape[dim])
    return tuple(result)


def _structural_view_fail(op, value_id, message):
    fail(
        "TLXW_OP_STRUCTURAL_VIEW",
        STAGE,
        message,
        source_op_index=op.index,
        source_value_id=value_id,
    )


def _redistribution_relation_plan(
    source_layout,
    destination_layout,
    source_slots,
    destination_slots,
    lane_width,
    warp_count,
    block_count,
    rank,
    op,
    source_value_id,
    result_value_id,
):
    description = "wave.redistribute layout conversion"
    dimensions = {
        "register": int(destination_slots),
        "lane": int(lane_width),
        "warp": int(warp_count),
        "block": int(block_count),
    }
    if any(not _is_positive_power_of_two(size) for size in dimensions.values()):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} requires power-of-two destination dimensions",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )

    source_by_coord = {}
    for block in range(int(block_count)):
        for warp in range(int(warp_count)):
            for lane in range(int(lane_width)):
                for slot in range(int(source_slots)):
                    coords = _redistribution_coords(
                        source_layout,
                        slot,
                        lane,
                        warp,
                        block,
                        rank,
                        op,
                        source_value_id,
                        description,
                    )
                    source_by_coord.setdefault(coords, []).append(
                        (int(slot), int(lane), int(warp), int(block))
                    )

    relation = {}
    is_identity = int(source_slots) == int(destination_slots)
    cross_wave = False
    for block in range(int(block_count)):
        for warp in range(int(warp_count)):
            for lane in range(int(lane_width)):
                for slot in range(int(destination_slots)):
                    coords = _redistribution_coords(
                        destination_layout,
                        slot,
                        lane,
                        warp,
                        block,
                        rank,
                        op,
                        result_value_id,
                        description,
                    )
                    candidates = source_by_coord.get(coords)
                    if not candidates:
                        fail(
                            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                            STAGE,
                            f"{description} result coordinate is not covered "
                            "by the source distributed layout",
                            source_op_index=op.index,
                            source_value_id=result_value_id,
                        )
                    destination = (int(slot), int(lane), int(warp), int(block))
                    source = min(
                        candidates,
                        key=lambda candidate: _redistribution_replica_score(
                            candidate,
                            destination,
                        ),
                    )
                    relation[destination] = source
                    is_identity = is_identity and source == destination
                    cross_wave = cross_wave or source[2:] != destination[2:]

    output_dims = (
        ("register", int(source_slots)),
        ("lane", int(lane_width)),
        ("warp", int(warp_count)),
        ("block", int(block_count)),
    )
    bases = []
    zero = (0, 0, 0, 0)
    if relation[zero] != zero:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} produced a non-zero relation origin",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    for input_index, (name, size) in enumerate(dimensions.items()):
        input_bases = []
        for bit in range(int(size).bit_length() - 1):
            point = [0, 0, 0, 0]
            point[input_index] = 1 << bit
            input_bases.append(tuple(int(value) for value in relation[tuple(point)]))
        bases.append((name, tuple(input_bases)))

    encoded_bases = tuple(bases)
    for destination, source in relation.items():
        if _apply_redistribution_bases(encoded_bases, destination) != source:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} requires a non-bit-linear gather relation",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
    return encoded_bases, output_dims, bool(is_identity), bool(cross_wave)


def _redistribution_coords(
    linear,
    slot,
    lane,
    warp,
    block,
    rank,
    op,
    value_id,
    description,
):
    available = {
        "block": int(block),
        "lane": int(lane),
        "register": int(slot),
        "warp": int(warp),
    }
    input_names = tuple(str(name) for name in linear.get_in_dim_names())
    if any(name not in available for name in input_names):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} has unsupported layout input dimensions",
            source_op_index=op.index,
            source_value_id=value_id,
        )
    outputs = linear.apply({name: available[name] for name in input_names})
    try:
        return tuple(int(outputs[f"dim{dim}"]) for dim in range(int(rank)))
    except KeyError:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} has malformed layout output dimensions",
            source_op_index=op.index,
            source_value_id=value_id,
        )


def _redistribution_replica_score(source, destination):
    return (
        source[3] != destination[3],
        source[2] != destination[2],
        source[1] != destination[1],
        source[0] != destination[0],
        source,
    )


def _apply_redistribution_bases(bases, inputs):
    result = [0, 0, 0, 0]
    for input_value, (_name, input_bases) in zip(inputs, bases):
        for bit, basis in enumerate(input_bases):
            if int(input_value) & (1 << bit):
                result = [
                    int(current) ^ int(coefficient)
                    for current, coefficient in zip(result, basis)
                ]
    return tuple(result)


def _is_positive_power_of_two(value):
    value = int(value)
    return value > 0 and not value & (value - 1)


def _redistribution_linear_layout(layout, value, op):
    if layout.kind != "dot_operand":
        return _distributed_linear_layout(layout, op)
    parent = layout.properties.get("parent_properties", {})
    instr_shape = tuple(int(item) for item in parent.get("instr_shape", ()))
    warps_per_cta = tuple(
        int(item) for item in parent.get("warps_per_cta", ())
    )
    registers = _dot_operand_fragment_registers(
        value.type.element_type,
        instr_shape,
        op,
    )
    elements_per_lane = _fragment_elements_per_lane(
        value.type.element_type,
        registers,
        op,
        value.value_id,
    )
    return _dot_operand_payload_linear_layout(
        layout,
        elements_per_lane,
        instr_shape,
        warps_per_cta,
        int(value.type.lane_width or 64),
        _dot_operand_parent_warp_count(layout),
        op,
    )


def _redistribution_warp_count(layout):
    if layout.kind == "dot_operand":
        return _dot_operand_parent_warp_count(layout)
    return _layout_warp_count(layout)


def register_remap(
    operand,
    result,
    operand_layout,
    result_layout,
    op,
    *,
    physical_element_bit_width=None,
    target=None,
):
    if operand_layout is None or result_layout is None:
        return None
    if not (operand_layout.kind == "amd_mfma" and result_layout.kind in {"blocked", "linear", "generic_linear"}):
        return None
    if operand.type.element_type != result.type.element_type:
        return None
    if result.type.representation not in {"simd", "simd_tuple"}:
        return None

    description = f"MFMA to {result_layout.kind} convert_layout"
    if result_layout.kind == "generic_linear":
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} involves #ttg.generic_linear; representative "
            "semantics for non-alias convert_layout are not declared",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    source_layout = _distributed_linear_layout(operand_layout, op)
    result_layout_ll = _distributed_linear_layout(result_layout, op)
    source_register_count = layouts.linear_layout_in_dim_size(
        source_layout,
        "register",
    )
    result_register_count = layouts.linear_layout_in_dim_size(
        result_layout_ll,
        "register",
    )
    source_registers_per_component = layouts.mfma_registers_per_component(
        operand_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    source_scalar_count = (int(operand.type.component_count) * source_registers_per_component)
    if source_scalar_count != source_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} source component model does not match "
            "the source register layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    if int(result.type.component_count) != result_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} requires a warp-aware result "
            f"component model: result has {int(result.type.component_count)} "
            f"components but the per-wave register layout has "
            f"{result_register_count}",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    cta_warp_count = max(
        _layout_warp_count(operand_layout),
        _layout_warp_count(result_layout),
    )
    source_by_coord = _source_slots_by_coord(
        source_layout,
        source_register_count,
        lane_width,
        cta_warp_count,
        op,
        operand.value_id,
        description=description,
    )

    result_sources = tuple(
        _sources_for_result_slot(
            result_layout_ll,
            result_register,
            source_by_coord,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            description=description,
        ) for result_register in range(result_register_count))

    simple_remap = _simple_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        source_registers_per_component,
        op,
        result.value_id,
        description=description,
    )
    if simple_remap is not None:
        return {
            "source_component_count": int(operand.type.component_count),
            "source_registers_per_component": int(source_registers_per_component),
            **simple_remap,
        }

    movement = _distributed_movement_class(
        result_sources,
        lane_width,
        cta_warp_count,
    )
    if movement == "lane_mux":
        lane_mux_remap = _lane_mux_register_remap(
            result_sources,
            lane_width,
            cta_warp_count,
            source_registers_per_component,
            op,
            result.value_id,
            description=description,
        )
        if lane_mux_remap is not None:
            return {
                "source_component_count": int(operand.type.component_count),
                "source_registers_per_component": int(source_registers_per_component),
                **lane_mux_remap,
            }

    exchange_remap = _cta_exchange_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        op,
        result.value_id,
        source_layout=source_layout,
        result_layout=result_layout_ll,
        physical_element_bit_width=physical_element_bit_width,
        target=target,
        description=description,
    )
    return {
        "mode": "cta_exchange_register_remap",
        "source_component_count": int(operand.type.component_count),
        "source_registers_per_component": int(source_registers_per_component),
        **exchange_remap,
    }


def distributed_remap(
    operand,
    result,
    operand_layout,
    result_layout,
    op,
    *,
    physical_element_bit_width=None,
    target=None,
):
    if operand_layout is None or result_layout is None:
        return None
    if operand_layout.kind not in _DISTRIBUTED_REMAP_KINDS:
        return None
    if result_layout.kind not in _DISTRIBUTED_REMAP_KINDS:
        return None
    if operand.type.element_type != result.type.element_type:
        return None
    if operand.type.kind != result.type.kind:
        return None
    if operand.type.representation not in _DISTRIBUTED_REMAP_REPRESENTATIONS:
        return None
    if result.type.representation not in _DISTRIBUTED_REMAP_REPRESENTATIONS:
        return None

    source_layout = _distributed_linear_layout(operand_layout, op)
    result_layout_ll = _distributed_linear_layout(result_layout, op)
    description = (f"{operand_layout.kind} to {result_layout.kind} convert_layout")
    if "generic_linear" in {operand_layout.kind, result_layout.kind}:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} involves #ttg.generic_linear; representative "
            "semantics for non-alias convert_layout are not declared",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    _require_injective_layout(
        source_layout,
        op,
        operand.value_id,
        f"{description} source layout",
    )
    _require_injective_layout(
        result_layout_ll,
        op,
        result.value_id,
        f"{description} result layout",
    )
    source_register_count = layouts.linear_layout_in_dim_size(
        source_layout,
        "register",
    )
    result_register_count = layouts.linear_layout_in_dim_size(
        result_layout_ll,
        "register",
    )
    if int(operand.type.component_count) != source_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} source component model does not match the "
            "distributed register layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    if int(result.type.component_count) != result_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} result component model does not match the "
            "distributed register layout",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    cta_warp_count = max(
        _layout_warp_count(operand_layout),
        _layout_warp_count(result_layout),
    )
    source_by_coord = _source_slots_by_coord(
        source_layout,
        source_register_count,
        lane_width,
        cta_warp_count,
        op,
        operand.value_id,
        description=description,
    )
    result_sources = tuple(
        _sources_for_result_slot(
            result_layout_ll,
            result_register,
            source_by_coord,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            description=description,
        ) for result_register in range(result_register_count))
    remap = _simple_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        1,
        op,
        result.value_id,
        description=description,
        allow_fallback=True,
    )
    if remap is None:
        movement = _distributed_movement_class(
            result_sources,
            lane_width,
            cta_warp_count,
        )
        if movement == "lane_mux":
            remap = _lane_mux_register_remap(
                result_sources,
                lane_width,
                cta_warp_count,
                1,
                op,
                result.value_id,
                description=description,
            )
            if remap is not None:
                return {
                    "source_component_count": int(operand.type.component_count),
                    "source_registers_per_component": 1,
                    **remap,
                }
        if movement != "cross_warp":
            remap = _simple_register_remap(
                result_sources,
                lane_width,
                cta_warp_count,
                1,
                op,
                result.value_id,
                description=description,
            )
            if remap is not None:
                return remap
            _reject_distributed_movement(
                result_sources,
                lane_width,
                cta_warp_count,
                op,
                result.value_id,
                description,
            )
        if result.type.representation not in {
                "mask",
                "mask_tuple",
                "simd",
                "simd_tuple",
        }:
            _reject_distributed_movement(
                result_sources,
                lane_width,
                cta_warp_count,
                op,
                result.value_id,
                description,
            )
        return {
            "mode":
            "cta_exchange_register_remap",
            "source_component_count":
            int(operand.type.component_count),
            "source_registers_per_component":
            1,
            **_cta_exchange_register_remap(
                result_sources,
                lane_width,
                cta_warp_count,
                op,
                result.value_id,
                source_layout=source_layout,
                result_layout=result_layout_ll,
                physical_element_bit_width=physical_element_bit_width,
                target=target,
                description=description,
            ),
        }
    if remap["mode"] == "cross_lane_register_remap" and result.type.representation not in {
            "mask",
            "mask_tuple",
            "simd",
            "simd_tuple",
    }:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} requires cross-lane movement for "
            f"{result.type.representation}; only Wave SIMD payloads can be "
            "shuffled",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    return {
        "source_component_count": int(operand.type.component_count),
        "source_registers_per_component": 1,
        **remap,
    }


def _lane_mux_register_remap(
    result_sources,
    lane_width,
    cta_warp_count,
    source_registers_per_component,
    op,
    result_value_id,
    *,
    description="distributed convert_layout",
):
    lane_width = int(lane_width)
    cta_warp_count = int(cta_warp_count)
    source_indices_by_result = []
    source_element_indices_by_result = []
    source_lanes_by_result = []
    source_lane_plans = []
    candidate_mask_plans = []
    for sources in result_sources:
        if len(sources) != cta_warp_count * lane_width:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} produced a malformed lane-mux source map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        reference = None
        for result_warp in range(cta_warp_count):
            source_indices = []
            source_element_indices = []
            source_lanes = []
            for lane in range(lane_width):
                source_warp, source_lane, source_register = sources[result_warp * lane_width + lane]
                if int(source_warp) != int(result_warp):
                    return None
                source_indices.append(int(source_register) // int(source_registers_per_component))
                source_element_indices.append(int(source_register) % int(source_registers_per_component))
                source_lanes.append(int(source_lane))
            current = (
                tuple(source_indices),
                tuple(source_element_indices),
                tuple(source_lanes),
            )
            if reference is None:
                reference = current
            elif current != reference:
                return None
        source_indices, source_element_indices, source_lanes = reference
        source_indices_by_result.append(source_indices)
        source_element_indices_by_result.append(source_element_indices)
        source_lanes_by_result.append(source_lanes)
        source_lane_plans.append(
            _lane_value_materialization_plan(source_lanes, lane_width)
        )
        candidate_mask_plans.append(
            tuple(
                _lane_selection_mask_plan(
                    tuple(
                        lane
                        for lane, source in enumerate(
                            zip(source_indices, source_element_indices)
                        )
                        if source == candidate
                    ),
                    lane_width,
                )
                for candidate in sorted(
                    set(zip(source_indices, source_element_indices))
                )
            )
        )
    return {
        "mode": "lane_mux_register_remap",
        "lane_mux_candidate_mask_plans": tuple(candidate_mask_plans),
        "lane_mux_source_element_indices": tuple(source_element_indices_by_result),
        "lane_mux_source_indices": tuple(source_indices_by_result),
        "lane_mux_source_lane_plans": tuple(source_lane_plans),
        "lane_mux_source_lanes": tuple(source_lanes_by_result),
    }


def _lane_value_materialization_plan(values, lane_width):
    values = tuple(int(value) for value in values)
    lane_width = int(lane_width)
    if (
        len(values) == lane_width
        and lane_width > 0
        and not lane_width & (lane_width - 1)
    ):
        base = values[0]
        coefficients = tuple(
            values[1 << bit] - base
            for bit in range(lane_width.bit_length() - 1)
        )
        if all(
            values[lane]
            == base
            + sum(
                int(coefficient)
                for bit, coefficient in enumerate(coefficients)
                if lane & (1 << bit)
            )
            for lane in range(lane_width)
        ):
            return ("bit_affine", int(base), *coefficients)
    return ("explicit", *values)


def _lane_selection_mask_plan(lanes, lane_width):
    lanes = tuple(sorted({int(lane) for lane in lanes}))
    lane_width = int(lane_width)
    if lanes == tuple(range(lane_width)):
        return ("all",)
    for bit in range(max(0, lane_width.bit_length() - 1)):
        for value in (0, 1):
            selected = tuple(
                lane
                for lane in range(lane_width)
                if ((lane >> bit) & 1) == value
            )
            if lanes == selected:
                return ("lane_bit", int(bit), int(value))
    if lanes:
        begin = lanes[0]
        end = lanes[-1] + 1
        if lanes == tuple(range(begin, end)):
            return ("range", int(begin), int(end))
    return ("explicit", *lanes)


def _lane_mux_vector_pack_plan(remap, vector_length):
    """Group compatible scalar lane muxes into vector payload chunks.

    The lane bijection is still described per scalar, but shuffling a packed
    vector lets the Wave machine lowering issue one cross-lane operation per
    physical register instead of one per sub-dword element.  A chunk contains
    only consecutive result scalars with identical lane and candidate-mask
    behavior; source register identities may differ and become the elements of
    each packed candidate.
    """
    vector_length = int(vector_length)
    source_indices_by_result = tuple(remap["lane_mux_source_indices"])
    source_element_indices_by_result = tuple(
        remap["lane_mux_source_element_indices"]
    )
    source_lanes_by_result = tuple(remap["lane_mux_source_lanes"])
    source_lane_plans = tuple(remap["lane_mux_source_lane_plans"])
    candidate_mask_plans = tuple(remap["lane_mux_candidate_mask_plans"])
    scalar_count = len(source_indices_by_result)
    if vector_length <= 0 or scalar_count % vector_length:
        return None
    if not (
        len(source_element_indices_by_result)
        == len(source_lanes_by_result)
        == len(source_lane_plans)
        == len(candidate_mask_plans)
        == scalar_count
    ):
        return None

    scalar_descriptors = []
    for source_indices, source_element_indices, source_lanes, source_lane_plan, mask_plans in zip(
        source_indices_by_result,
        source_element_indices_by_result,
        source_lanes_by_result,
        source_lane_plans,
        candidate_mask_plans,
    ):
        candidate_keys = sorted(
            set(zip(source_indices, source_element_indices))
        )
        if len(candidate_keys) != len(mask_plans):
            return None
        candidate_lanes = tuple(
            tuple(
                lane
                for lane, source in enumerate(
                    zip(source_indices, source_element_indices)
                )
                if source == candidate
            )
            for candidate in candidate_keys
        )
        candidate_shuffles = tuple(
            not all(int(source_lanes[lane]) == lane for lane in lanes)
            for lanes in candidate_lanes
        )
        scalar_descriptors.append((
            tuple(source_lane_plan),
            tuple(int(lane) for lane in source_lanes),
            tuple(tuple(plan) for plan in mask_plans),
            candidate_lanes,
            candidate_shuffles,
            tuple(
                (int(component_index), int(element_index))
                for component_index, element_index in candidate_keys
            ),
        ))

    vector_plan = []
    chunk_sizes = tuple(
        size
        for size in range(vector_length, 0, -1)
        if vector_length % size == 0
    )
    for vector_start in range(0, scalar_count, vector_length):
        for chunk_size in chunk_sizes:
            compatible = True
            for chunk_start in range(
                vector_start,
                vector_start + vector_length,
                chunk_size,
            ):
                signature = scalar_descriptors[chunk_start][:-1]
                if any(
                    scalar_descriptors[result_index][:-1] != signature
                    for result_index in range(
                        chunk_start + 1,
                        chunk_start + chunk_size,
                    )
                ):
                    compatible = False
                    break
            if compatible:
                break
        if not compatible:
            return None

        chunks = []
        for chunk_start in range(
            vector_start,
            vector_start + vector_length,
            chunk_size,
        ):
            (
                source_lane_plan,
                source_lanes,
                mask_plans,
                candidate_lanes,
                candidate_shuffles,
                _,
            ) = scalar_descriptors[chunk_start]
            candidate_sources = tuple(
                tuple(
                    scalar_descriptors[result_index][-1][candidate_index]
                    for result_index in range(
                        chunk_start,
                        chunk_start + chunk_size,
                    )
                )
                for candidate_index in range(len(mask_plans))
            )
            chunks.append((
                source_lane_plan,
                source_lanes,
                mask_plans,
                candidate_lanes,
                candidate_sources,
                candidate_shuffles,
            ))
        vector_plan.append((int(chunk_size), tuple(chunks)))
    return tuple(vector_plan)


def mfma_component_metadata_remap(
    operand,
    result,
    operand_layout,
    result_layout,
    op,
    *,
    physical_element_bit_width=None,
    target=None,
):
    if operand_layout is None or result_layout is None:
        return None
    if operand_layout.kind not in _DISTRIBUTED_REMAP_KINDS:
        return None
    if result_layout.kind != "amd_mfma":
        return None
    if operand.type.element_type != result.type.element_type:
        return None
    value_metadata = operand.type.representation in {
        "simd",
        "simd_tuple",
    } and result.type.representation in {"simd", "simd_tuple"}
    mask_metadata = operand.type.representation in {
        "mask",
        "mask_tuple",
    } and result.type.representation in {"mask", "mask_tuple"}
    if not value_metadata and not mask_metadata:
        return None
    if operand_layout.kind == "generic_linear":
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "generic_linear to MFMA metadata convert_layout requires declared "
            "representative semantics",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    if tuple(operand_layout.shape) != tuple(result_layout.shape):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to MFMA metadata convert_layout requires matching "
            "source and result shapes",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    packed_registers_per_component = layouts.mfma_registers_per_component(
        result_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    result_count = int(result.type.component_count)
    source_count = int(operand.type.component_count)
    source_layout = _distributed_linear_layout(operand_layout, op)
    result_layout_ll = _distributed_linear_layout(result_layout, op)
    source_register_count = layouts.linear_layout_in_dim_size(
        source_layout,
        "register",
    )
    if source_count != int(source_register_count):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to MFMA metadata source component model does not "
            "match the source register layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    result_register_count = layouts.linear_layout_in_dim_size(
        result_layout_ll,
        "register",
    )
    if result_count == int(result_register_count):
        registers_per_component = 1
    elif result_count * int(packed_registers_per_component) == int(result_register_count):
        registers_per_component = int(packed_registers_per_component)
    else:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to MFMA metadata result component model does not "
            "match the MFMA register layout",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    cta_warp_count = max(
        _layout_warp_count(operand_layout),
        _layout_warp_count(result_layout),
    )
    source_by_coord = _source_slots_by_coord(
        source_layout,
        source_register_count,
        lane_width,
        cta_warp_count,
        op,
        operand.value_id,
        description="distributed to MFMA metadata convert_layout",
    )
    result_sources = []
    component_vectors_are_contiguous = True
    vector_axis = (1 if bool(result_layout.properties.get("is_transposed", False)) else 0)
    for component in range(result_count):
        base_register = int(component) * int(registers_per_component)
        if not _mfma_component_vector_is_contiguous(
                result_layout_ll,
                base_register,
                int(registers_per_component),
                vector_axis,
                lane_width,
                cta_warp_count,
        ):
            component_vectors_are_contiguous = False
        result_sources.append(
            _sources_for_result_slot(
                result_layout_ll,
                base_register,
                source_by_coord,
                lane_width,
                cta_warp_count,
                op,
                result.value_id,
                description="distributed to MFMA metadata convert_layout",
            ))
    result_sources = tuple(result_sources)

    if not component_vectors_are_contiguous:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to MFMA metadata convert_layout requires each "
            "MFMA component vector to cover its logical vector dimension",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    base_attrs = {
        "source_component_count": int(source_count),
        "source_registers_per_component": 1,
    }
    simple_remap = _simple_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        1,
        op,
        result.value_id,
        description="distributed to MFMA metadata convert_layout",
        allow_fallback=True,
    )
    if simple_remap is not None:
        return {**base_attrs, **simple_remap}

    movement = _distributed_movement_class(
        result_sources,
        lane_width,
        cta_warp_count,
    )
    if movement == "unknown":
        strict_remap = _simple_register_remap(
            result_sources,
            lane_width,
            cta_warp_count,
            1,
            op,
            result.value_id,
            description="distributed to MFMA metadata convert_layout",
        )
        if strict_remap is not None:
            return {**base_attrs, **strict_remap}
        _reject_distributed_movement(
            result_sources,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            "distributed to MFMA metadata convert_layout",
        )
    if movement not in {"cross_warp", "lane_mux"}:
        _reject_distributed_movement(
            result_sources,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            "distributed to MFMA metadata convert_layout",
        )
    return {
        "mode":
        "cta_exchange_register_remap",
        **base_attrs,
        **_cta_exchange_register_remap(
            result_sources,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            source_layout=source_layout,
            result_layout=result_layout_ll,
            physical_element_bit_width=physical_element_bit_width,
            target=target,
            description="distributed to MFMA metadata convert_layout",
        ),
    }


def dot_operand_vector_payload(
    operand,
    result,
    operand_layout,
    result_layout,
    op,
    *,
    physical_element_bit_width=None,
    target=None,
):
    if operand_layout is None or result_layout is None:
        return None
    source_is_distributed = operand_layout.kind in _DISTRIBUTED_REMAP_KINDS
    source_is_mfma = operand_layout.kind == "amd_mfma"
    if not source_is_distributed and not source_is_mfma:
        return None
    if result_layout.kind != "dot_operand":
        return None
    if operand_layout.kind == "generic_linear":
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "generic_linear to dot_operand convert_layout requires declared "
            "representative semantics before MMA payload packing",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    if operand.type.element_type != result.type.element_type:
        return None
    if (source_is_distributed
            and operand.type.representation not in {"simd", "simd_tuple"}):
        return None
    if (source_is_mfma
            and operand.type.representation not in _MMA_PACKET_REPRESENTATIONS):
        return None
    if result.type.representation not in _MMA_PACKET_REPRESENTATIONS:
        return None
    if tuple(operand_layout.shape) != tuple(result_layout.shape):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand convert_layout requires matching "
            "source and result shapes",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(int(value) for value in parent.get("instr_shape", ()))
    warps_per_cta = tuple(int(value) for value in parent.get("warps_per_cta", ()))
    if instr_shape not in {(16, 16, 32), (32, 32, 16)} or len(warps_per_cta) < 2:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand convert_layout requires a supported "
            "MFMA parent layout",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    registers = _dot_operand_fragment_registers(
        result.type.element_type,
        instr_shape,
        op,
    )
    elements_per_lane = _fragment_elements_per_lane(
        result.type.element_type,
        registers,
        op,
        result.value_id,
    )
    k_width = int(result_layout.properties.get("k_width", 0))
    if (k_width <= 0 or int(elements_per_lane) % k_width):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand convert_layout requires kWidth to be "
            "a positive divisor of the MMA packet width: "
            f"kWidth={k_width}, "
            f"payload_width={int(elements_per_lane)}",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    source_layout = _distributed_linear_layout(operand_layout, op)
    _require_injective_layout(
        source_layout,
        op,
        operand.value_id,
        "distributed to dot_operand source layout",
    )
    source_component_count = layouts.linear_layout_in_dim_size(
        source_layout,
        "register",
    )
    source_registers_per_component = (
        layouts.mfma_registers_per_component(
            operand_layout,
            stage=STAGE,
            source_op_index=op.index,
        ) if source_is_mfma else 1
    )
    source_scalar_count = (
        int(operand.type.component_count) * int(source_registers_per_component)
    )
    if source_scalar_count != source_component_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand source payload model does not match "
            "the source register layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    source_warp_count = _layout_warp_count(operand_layout)
    result_warp_count = _dot_operand_parent_warp_count(result_layout)
    if int(source_warp_count) != int(result_warp_count):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand convert_layout requires matching "
            "source and dot-parent wave counts",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    cta_warp_count = int(source_warp_count)
    source_component_count = int(source_component_count)
    cta_thread_count = int(lane_width) * int(cta_warp_count)
    if physical_element_bit_width is None or target is None:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand convert_layout requires a physical "
            "element width and target for conflict-aware LDS planning",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    result_layout_ll = _dot_operand_payload_linear_layout(
        result_layout,
        int(elements_per_lane),
        instr_shape,
        warps_per_cta,
        int(lane_width),
        int(cta_warp_count),
        op,
    )
    result_scalar_count = layouts.linear_layout_in_dim_size(
        result_layout_ll,
        "register",
    )
    expected_result_scalar_count = (
        int(result.type.component_count) * int(elements_per_lane)
    )
    if int(result_scalar_count) != int(expected_result_scalar_count):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand scalar payload model does not match "
            "the destination LinearLayout",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    source_by_coord = _source_slots_by_coord(
        source_layout,
        source_component_count,
        lane_width,
        cta_warp_count,
        op,
        operand.value_id,
        description="distributed to dot_operand convert_layout",
    )
    result_sources = tuple(
        _sources_for_result_slot(
            result_layout_ll,
            result_register,
            source_by_coord,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            description="distributed to dot_operand convert_layout",
        )
        for result_register in range(int(result_scalar_count))
    )
    register_remap = _dot_operand_vector_register_remap_attrs(
        result_sources,
        lane_width,
        cta_warp_count,
        int(source_registers_per_component),
        int(operand.type.component_count),
        int(elements_per_lane),
        int(registers),
        instr_shape,
        int(result_layout.properties["op_idx"]),
        op,
        result.value_id,
    )
    if register_remap is not None:
        return {
            "element_type": result.type.element_type,
            "elements_per_lane": int(elements_per_lane),
            "result_scalar_count": int(result_scalar_count),
            **register_remap,
        }
    exchange_remap = _cta_exchange_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        op,
        result.value_id,
        source_layout=source_layout,
        result_layout=result_layout_ll,
        physical_element_bit_width=physical_element_bit_width,
        target=target,
        description="distributed to dot_operand convert_layout",
    )
    if exchange_remap.get("scratch_physical_plan") != "optimal_swizzling_ldst":
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand convert_layout did not produce a "
            "conflict-aware LDS plan",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    return {
        "cta_thread_count": int(cta_thread_count),
        "element_type": result.type.element_type,
        "elements_per_lane": int(elements_per_lane),
        "mode": "dot_operand_vector_payload",
        "registers": int(registers),
        "result_scalar_count": int(result_scalar_count),
        "role": int(result_layout.properties["op_idx"]),
        "rows": int(instr_shape[0]),
        "columns": int(instr_shape[1]),
        "source_component_count": int(operand.type.component_count),
        "source_registers_per_component": int(source_registers_per_component),
        **exchange_remap,
    }


def _dot_operand_vector_register_remap_attrs(
    result_sources,
    lane_width,
    cta_warp_count,
    source_registers_per_component,
    source_component_count,
    vector_length,
    registers,
    instr_shape,
    role,
    op,
    result_value_id,
):
    """Describe an in-wave scalar remap packed as dot-operand vectors.

    Dot operands are represented as a tuple of SIMD vectors, while the layout
    bijection is scalar.  Keep the scalar mapping explicit in target IR and let
    emission pack consecutive destination scalars into the declared vector
    payload.  A CTA exchange remains the fallback whenever the composed map
    genuinely crosses waves or cannot be represented by the supported lane
    remaps.
    """
    description = "distributed to dot_operand convert_layout"
    base_attrs = {
        "mode": "mfma_vector_register_remap",
        "columns": int(instr_shape[1]),
        "registers": int(registers),
        "role": int(role),
        "rows": int(instr_shape[0]),
        "scalar_result_component_count": len(result_sources),
        "source_component_count": int(source_component_count),
        "source_registers_per_component": int(source_registers_per_component),
        "vector_length": int(vector_length),
    }
    simple_remap = _simple_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        source_registers_per_component,
        op,
        result_value_id,
        description=description,
        allow_fallback=True,
    )
    if simple_remap is not None:
        return {
            **base_attrs,
            "scalar_mode": simple_remap["mode"],
            "scalar_source_element_indices": tuple(
                simple_remap["source_element_indices"]
            ),
            "scalar_source_indices": tuple(simple_remap["source_indices"]),
            **{
                key: value
                for key, value in simple_remap.items()
                if key.startswith("source_lane_")
            },
        }

    movement = _distributed_movement_class(
        result_sources,
        lane_width,
        cta_warp_count,
    )
    if movement == "cross_warp":
        return None
    lane_mux_remap = _lane_mux_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        source_registers_per_component,
        op,
        result_value_id,
        description=description,
    )
    if lane_mux_remap is None:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} has a wave-varying same-wave remap that cannot "
            "be represented by one SIMD lane-mux plan",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    vector_pack_plan = _lane_mux_vector_pack_plan(
        lane_mux_remap,
        vector_length,
    )
    if vector_pack_plan is None:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} has an invalid vector lane-mux packing plan",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    return {
        **base_attrs,
        "scalar_mode": lane_mux_remap["mode"],
        "lane_mux_vector_pack_plan": vector_pack_plan,
        **{
            key: value
            for key, value in lane_mux_remap.items()
            if key != "mode"
        },
    }


def distributed_to_mfma_base_remap(
    operand,
    result,
    operand_layout,
    result_layout,
    op,
    *,
    physical_element_bit_width=None,
    target=None,
):
    if operand_layout is None or result_layout is None:
        return None
    if operand_layout.kind not in _DISTRIBUTED_REMAP_KINDS:
        return None
    if result_layout.kind != "amd_mfma":
        return None
    if operand.type.element_type != result.type.element_type:
        return None
    value_remap = (operand.type.representation in {"simd", "simd_tuple"}
                   and result.type.representation in _MMA_PACKET_REPRESENTATIONS
                   and operand.type.element_type != "i1")
    mask_remap = (operand.type.representation in {"mask", "mask_tuple"}
                  and result.type.representation in {"mask", "mask_tuple"} and operand.type.element_type == "i1")
    if not value_remap and not mask_remap:
        return None
    if tuple(operand_layout.shape) != tuple(result_layout.shape):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to MFMA base convert_layout requires matching "
            "source and result shapes",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    source_layout = _distributed_linear_layout(operand_layout, op)
    result_layout_ll = _distributed_linear_layout(result_layout, op)
    description = "distributed to MFMA base convert_layout"
    _require_injective_layout(
        source_layout,
        op,
        operand.value_id,
        f"{description} source layout",
    )
    source_register_count = layouts.linear_layout_in_dim_size(
        source_layout,
        "register",
    )
    if int(operand.type.component_count) != int(source_register_count):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to MFMA base source component model does not match "
            "the source register layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    registers_per_component = layouts.mfma_registers_per_component(
        result_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    result_register_count = layouts.linear_layout_in_dim_size(
        result_layout_ll,
        "register",
    )
    if int(result.type.component_count) * int(registers_per_component) != int(result_register_count):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to MFMA base result component model does not match "
            "the MFMA register layout",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    cta_warp_count = max(
        _layout_warp_count(operand_layout),
        _layout_warp_count(result_layout),
    )
    source_by_coord = _source_slots_by_coord(
        source_layout,
        source_register_count,
        lane_width,
        cta_warp_count,
        op,
        operand.value_id,
        description=description,
    )
    result_sources = []
    component_vectors_are_contiguous = True
    vector_axis = (1 if bool(result_layout.properties.get("is_transposed", False)) else 0)
    for component in range(int(result.type.component_count)):
        base_register = int(component) * int(registers_per_component)
        if not _mfma_component_vector_is_contiguous(
                result_layout_ll,
                base_register,
                int(registers_per_component),
                vector_axis,
                lane_width,
                cta_warp_count,
        ):
            component_vectors_are_contiguous = False
        result_sources.append(
            _sources_for_result_slot(
                result_layout_ll,
                base_register,
                source_by_coord,
                lane_width,
                cta_warp_count,
                op,
                result.value_id,
                description=description,
            ))
    result_sources = tuple(result_sources)

    if value_remap and result.type.element_type == "f32":
        scalar_result_sources = _mfma_scalar_result_sources(
            result_layout_ll,
            source_by_coord,
            int(result.type.component_count),
            int(registers_per_component),
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            description,
        )
        return _mfma_vector_register_remap_attrs(
            result,
            result_layout,
            source_layout,
            result_layout_ll,
            scalar_result_sources,
            lane_width,
            cta_warp_count,
            int(operand.type.component_count),
            source_registers_per_component=1,
            registers_per_component=int(registers_per_component),
            op=op,
            result_value_id=result.value_id,
            description=description,
            physical_element_bit_width=physical_element_bit_width,
            target=target,
        )

    if not component_vectors_are_contiguous:
        _reject_distributed_movement(
            result_sources,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            description,
        )

    remap = _simple_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        1,
        op,
        result.value_id,
        description=description,
    )
    if remap is not None and (remap["mode"] == "same_lane_register_remap" or value_remap):
        return {
            "source_component_count": int(operand.type.component_count),
            "source_registers_per_component": 1,
            **remap,
        }

    return {
        "mode":
        "cta_exchange_register_remap",
        "source_component_count":
        int(operand.type.component_count),
        "source_registers_per_component":
        1,
        **_cta_exchange_register_remap(
            result_sources,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            source_layout=source_layout,
            result_layout=result_layout_ll,
            physical_element_bit_width=physical_element_bit_width,
            target=target,
            description=description,
        ),
    }


def _mfma_scalar_result_sources(
    result_layout,
    source_by_coord,
    result_component_count,
    registers_per_component,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
    description,
):
    scalar_result_sources = []
    for component in range(int(result_component_count)):
        base_register = int(component) * int(registers_per_component)
        for element in range(int(registers_per_component)):
            scalar_result_sources.append(
                _sources_for_result_slot(
                    result_layout,
                    base_register + int(element),
                    source_by_coord,
                    lane_width,
                    cta_warp_count,
                    op,
                    result_value_id,
                    description=description,
                ))
    return tuple(scalar_result_sources)


def _mfma_vector_register_remap_attrs(
    result,
    result_layout,
    source_layout,
    result_layout_ll,
    scalar_result_sources,
    lane_width,
    cta_warp_count,
    source_component_count,
    *,
    source_registers_per_component,
    registers_per_component,
    op,
    result_value_id,
    description,
    physical_element_bit_width=None,
    target=None,
):
    instr_shape = tuple(int(entry) for entry in result_layout.properties.get("instr_shape", ()))
    scalar_remap = _simple_register_remap(
        scalar_result_sources,
        lane_width,
        cta_warp_count,
        source_registers_per_component,
        op,
        result_value_id,
        description=description,
        allow_fallback=True,
    )
    attrs = {
        "mode": "mfma_vector_register_remap",
        "columns": int(instr_shape[1]),
        "registers": int(registers_per_component),
        "role": 2,
        "rows": int(instr_shape[0]),
        "scalar_result_component_count": len(scalar_result_sources),
        "source_component_count": int(source_component_count),
        "source_registers_per_component": int(source_registers_per_component),
        "vector_length": int(registers_per_component),
    }
    if scalar_remap is not None:
        return {
            **attrs,
            "scalar_mode": scalar_remap["mode"],
            "scalar_source_element_indices": tuple(scalar_remap["source_element_indices"]),
            "scalar_source_indices": tuple(scalar_remap["source_indices"]),
            **{key: value
               for key, value in scalar_remap.items() if key.startswith("source_lane_")},
        }
    return {
        **attrs,
        "scratch_reuse_lds":
        True,
        **_cta_exchange_register_remap(
            scalar_result_sources,
            lane_width,
            cta_warp_count,
            op,
            result_value_id,
            source_layout=source_layout,
            result_layout=result_layout_ll,
            physical_element_bit_width=physical_element_bit_width,
            target=target,
            description=description,
        ),
    }


def reject_unsupported_pair(operand_layout, result_layout, op):
    operand_kind = "none" if operand_layout is None else operand_layout.kind
    result_kind = "none" if result_layout is None else result_layout.kind
    if operand_kind in {"slice", "dot_operand"} or result_kind in {"slice", "dot_operand"}:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{operand_kind} to {result_kind} convert_layout requires parent "
            "layout movement support",
            source_op_index=op.index,
        )
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"{operand_kind} to {result_kind} convert_layout has unknown "
        "movement class",
        source_op_index=op.index,
    )


def _source_slots_by_coord(
    source_layout,
    source_register_count,
    lane_width,
    cta_warp_count,
    op,
    source_value_id,
    *,
    description="MFMA convert_layout",
    allow_replicated_warps=False,
):
    source_by_coord = {}
    for source_warp in range(int(cta_warp_count)):
        for source_register in range(int(source_register_count)):
            for lane in range(int(lane_width)):
                coords = layouts.linear_layout_coords(
                    source_layout,
                    source_register,
                    lane,
                    warp=source_warp,
                )
                source = (source_warp, lane, source_register)
                if coords in source_by_coord:
                    if allow_replicated_warps:
                        existing = source_by_coord[coords]
                        if existing and isinstance(existing[0], tuple):
                            source_by_coord[coords] = (
                                *existing,
                                tuple(int(value) for value in source),
                            )
                        else:
                            source_by_coord[coords] = (
                                tuple(int(value) for value in existing),
                                tuple(int(value) for value in source),
                            )
                        continue
                    fail(
                        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                        STAGE,
                        f"{description} source layout is not injective "
                        "within the CTA distributed map",
                        source_op_index=op.index,
                        source_value_id=source_value_id,
                    )
                source_by_coord[coords] = tuple(int(value) for value in source)
    return source_by_coord


def _sources_for_result_slot(
    result_layout,
    result_register,
    source_by_coord,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
    *,
    description="MFMA to blocked convert_layout",
):
    sources = []
    for result_warp in range(int(cta_warp_count)):
        for lane in range(int(lane_width)):
            coords = layouts.linear_layout_coords(
                result_layout,
                result_register,
                lane,
                warp=result_warp,
            )
            source = source_by_coord.get(coords)
            if source is None:
                fail(
                    "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                    STAGE,
                    f"{description} result coordinate is not "
                    "covered by the source distributed layout",
                    source_op_index=op.index,
                    source_value_id=result_value_id,
                )
            if source and isinstance(source[0], tuple):
                same_warp_sources = [candidate for candidate in source if int(candidate[0]) == int(result_warp)]
                if not same_warp_sources:
                    fail(
                        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                        STAGE,
                        f"{description} replicated source coordinate is not "
                        "available in the result wave",
                        source_op_index=op.index,
                        source_value_id=result_value_id,
                    )
                source = same_warp_sources[0]
            sources.append(tuple(int(value) for value in source))
    return tuple(sources)


def _mfma_component_vector_is_contiguous(
    result_layout,
    base_register,
    registers_per_component,
    vector_axis,
    lane_width,
    cta_warp_count,
):
    if int(registers_per_component) <= 1:
        return True
    for result_warp in range(int(cta_warp_count)):
        for lane in range(int(lane_width)):
            base_coords = layouts.linear_layout_coords(
                result_layout,
                int(base_register),
                lane,
                warp=result_warp,
            )
            for element in range(1, int(registers_per_component)):
                coords = layouts.linear_layout_coords(
                    result_layout,
                    int(base_register) + int(element),
                    lane,
                    warp=result_warp,
                )
                expected = list(base_coords)
                expected[int(vector_axis)] += int(element)
                if tuple(coords) != tuple(expected):
                    return False
    return True


def _simple_register_remap(
    result_sources,
    lane_width,
    cta_warp_count,
    source_registers_per_component,
    op,
    result_value_id,
    *,
    description="MFMA to blocked convert_layout",
    allow_fallback=False,
):
    source_indices = []
    source_element_indices = []
    source_lane_maps = []
    for sources in result_sources:
        if len(sources) != int(cta_warp_count) * int(lane_width):
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} produced a malformed "
                "source map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        lane_maps = []
        registers = []
        for result_warp in range(int(cta_warp_count)):
            wave_lane_map = []
            for lane in range(int(lane_width)):
                source_warp, source_lane, source_register = sources[result_warp * int(lane_width) + lane]
                if source_warp != result_warp:
                    return None
                registers.append(source_register)
                wave_lane_map.append(source_lane)
            lane_maps.append(tuple(int(lane) for lane in wave_lane_map))

        first_source = registers[0]
        if not all(source == first_source for source in registers):
            return None
        first_lane_map = lane_maps[0]
        if not all(lane_map == first_lane_map for lane_map in lane_maps):
            if allow_fallback:
                return None
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} requires a wave-varying source "
                "lane map; explicit CTA-wave remap support is required",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        if all(source_lane == lane for lane, source_lane in enumerate(first_lane_map)):
            source_lane_map = None
        else:
            source_lane_map = first_lane_map
        source_indices.append(first_source // int(source_registers_per_component))
        source_element_indices.append(first_source % int(source_registers_per_component))
        source_lane_maps.append(source_lane_map)

    if allow_fallback:
        lane_map_attrs = _classify_source_lane_maps_or_none(
            source_lane_maps,
            lane_width,
        )
        if lane_map_attrs is None and any(lane_map is not None for lane_map in source_lane_maps):
            return None
    else:
        lane_map_attrs = _classify_source_lane_maps(
            source_lane_maps,
            lane_width,
            op,
            result_value_id,
            description=description,
        )
    return {
        "mode": "cross_lane_register_remap" if lane_map_attrs is not None else "same_lane_register_remap",
        "source_element_indices": tuple(source_element_indices),
        "source_indices": tuple(source_indices),
        **(lane_map_attrs or {}),
    }


def _classify_source_lane_maps_or_none(source_lane_maps, lane_width):
    concrete_maps = [lane_map for lane_map in source_lane_maps if lane_map is not None]
    if not concrete_maps:
        return None
    first = concrete_maps[0]
    if not all(lane_map == first for lane_map in concrete_maps):
        return None
    if len(first) != int(lane_width):
        return None
    if any(int(lane) < 0 or int(lane) >= int(lane_width) for lane in first):
        return None
    kind, attrs = _classify_lane_map(first, lane_width)
    if kind is None:
        return None
    return {
        "source_lane_map": tuple(int(lane) for lane in first),
        "source_lane_map_kind": kind,
        **attrs,
    }


def _cta_exchange_register_remap(
    result_sources,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
    *,
    source_layout=None,
    result_layout=None,
    physical_element_bit_width=None,
    target=None,
    description="MFMA to blocked convert_layout",
):
    cta_thread_count = int(lane_width) * int(cta_warp_count)
    if (
        source_layout is not None
        and result_layout is not None
        and physical_element_bit_width is not None
        and target is not None
    ):
        source_register_count = layouts.linear_layout_in_dim_size(
            source_layout,
            "register",
        )
        result_register_count = layouts.linear_layout_in_dim_size(
            result_layout,
            "register",
        )
        referenced_source_registers = {
            int(source_register)
            for sources in result_sources
            for _, _, source_register in sources
        }
        if (
            len(result_sources) == int(result_register_count)
            and referenced_source_registers == set(range(int(source_register_count)))
        ):
            return _optimal_swizzled_cta_exchange_register_remap(
                result_sources,
                source_layout,
                result_layout,
                int(physical_element_bit_width),
                str(target),
                int(lane_width),
                int(cta_warp_count),
                op,
                result_value_id,
                description=description,
            )

    groups = {}
    max_group_slots = 0
    for result_index, sources in enumerate(result_sources):
        if len(sources) != cta_thread_count:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} produced a malformed CTA source map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        source_slots = tuple(sorted({int(source[2]) for source in sources}))
        if not source_slots:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} produced an empty CTA source map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        max_group_slots = max(max_group_slots, len(source_slots))
        source_slot_indices = {source_slot: index for index, source_slot in enumerate(source_slots)}
        load_offsets = []
        for source_warp, source_lane, source_register in sources:
            load_offsets.append(source_slot_indices[int(source_register)] * cta_thread_count +
                                int(source_warp) * int(lane_width) + int(source_lane))
        base, coefficients = _fit_bit_affine_offsets(
            load_offsets,
            cta_thread_count,
            op,
            result_value_id,
        )
        groups.setdefault(source_slots, []).append(
            (int(result_index), int(base), tuple(int(value) for value in coefficients)))

    exchange_groups = []
    for source_slots, result_entries in groups.items():
        exchange_groups.append((
            tuple(int(slot) for slot in source_slots),
            tuple(int(entry[0]) for entry in result_entries),
            tuple(int(entry[1]) for entry in result_entries),
            tuple(tuple(int(value) for value in entry[2]) for entry in result_entries),
        ))
    return {
        "barrier_scope": "cta",
        "cta_thread_count": int(cta_thread_count),
        "exchange_groups": tuple(exchange_groups),
        "scratch_element_count": int(max_group_slots) * int(cta_thread_count),
    }


def _optimal_swizzled_cta_exchange_register_remap(
    result_sources,
    source_layout,
    result_layout,
    physical_element_bit_width,
    target,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
    *,
    description,
):
    cta_thread_count = int(lane_width) * int(cta_warp_count)
    try:
        plan = layouts.optimal_swizzled_ldst_plan(
            source_layout,
            result_layout,
            int(physical_element_bit_width),
            target,
        )
    except (RuntimeError, ValueError) as exc:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} could not build an optimal LDS swizzle: {exc}",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )

    repetitions = int(plan["repetitions"])
    store_tile_size = int(plan["store_tile_size"])
    load_tile_size = int(plan["load_tile_size"])
    scratch_elements = int(plan["scratch_elements"])
    store_vector_elements = int(plan["store_vector_elements"])
    load_vector_elements = int(plan["load_vector_elements"])
    store_registers = tuple(int(value) for value in plan["store_registers"])
    load_registers = tuple(int(value) for value in plan["load_registers"])
    source_register_count = layouts.linear_layout_in_dim_size(source_layout, "register")
    result_register_count = layouts.linear_layout_in_dim_size(result_layout, "register")
    if (
        repetitions <= 0
        or store_tile_size <= 0
        or load_tile_size <= 0
        or scratch_elements <= 0
        or store_vector_elements <= 0
        or load_vector_elements <= 0
        or store_tile_size % store_vector_elements
        or load_tile_size % load_vector_elements
        or len(store_registers) != repetitions * store_tile_size
        or len(load_registers) != repetitions * load_tile_size
        or len(store_registers) != int(source_register_count)
        or len(load_registers) != int(result_register_count)
        or sorted(store_registers) != list(range(int(source_register_count)))
        or sorted(load_registers) != list(range(int(result_register_count)))
    ):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} produced a malformed optimal LDS swizzle plan",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )

    store_layout = plan["store_layout"]
    load_layout = plan["load_layout"]
    exchange_repetitions = []
    covered_results = set()
    for repetition in range(repetitions):
        store_register_base = repetition * store_tile_size
        load_register_base = repetition * load_tile_size
        store_packets = _swizzled_cta_exchange_packets(
            store_layout,
            store_registers[
                store_register_base:store_register_base + store_tile_size
            ],
            store_vector_elements,
            cta_thread_count,
            lane_width,
            scratch_elements,
            op,
            result_value_id,
            description=description,
        )
        load_packets = _swizzled_cta_exchange_packets(
            load_layout,
            load_registers[
                load_register_base:load_register_base + load_tile_size
            ],
            load_vector_elements,
            cta_thread_count,
            lane_width,
            scratch_elements,
            op,
            result_value_id,
            description=description,
        )

        stored_sources = {}
        for source_indices, base, coefficients in store_packets:
            for thread in range(cta_thread_count):
                packet_offset = _apply_bit_linear_offset(base, coefficients, thread)
                for element, source_register in enumerate(source_indices):
                    offset = int(packet_offset) + int(element)
                    source = (
                        int(thread) // int(lane_width),
                        int(thread) % int(lane_width),
                        int(source_register),
                    )
                    previous = stored_sources.setdefault(offset, source)
                    if previous != source:
                        fail(
                            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                            STAGE,
                            f"{description} optimal LDS swizzle aliases two stores",
                            source_op_index=op.index,
                            source_value_id=result_value_id,
                        )
        if len(stored_sources) != scratch_elements:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} optimal LDS swizzle does not cover its scratch tile",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )

        for result_indices, base, coefficients in load_packets:
            covered_results.update(int(index) for index in result_indices)
            for thread in range(cta_thread_count):
                packet_offset = _apply_bit_linear_offset(base, coefficients, thread)
                for element, result_index in enumerate(result_indices):
                    offset = int(packet_offset) + int(element)
                    actual_source = stored_sources.get(offset)
                    expected_source = tuple(
                        int(value) for value in result_sources[int(result_index)][int(thread)])
                    if actual_source != expected_source:
                        fail(
                            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                            STAGE,
                            f"{description} optimal LDS swizzle does not preserve the semantic remap",
                            source_op_index=op.index,
                            source_value_id=result_value_id,
                        )
        exchange_repetitions.append((store_packets, load_packets))

    if covered_results != set(range(len(result_sources))):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} optimal LDS swizzle does not produce every result",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    return {
        "barrier_scope": "cta",
        "cta_thread_count": int(cta_thread_count),
        "scratch_element_count": int(scratch_elements),
        "scratch_exchange_repetitions": tuple(exchange_repetitions),
        "scratch_load_vector_elements": int(load_vector_elements),
        "scratch_physical_plan": "optimal_swizzling_ldst",
        "scratch_read_bank_conflicts": int(plan["read_bank_conflicts"]),
        "scratch_store_vector_elements": int(store_vector_elements),
        "scratch_write_bank_conflicts": int(plan["write_bank_conflicts"]),
    }


def _swizzled_cta_exchange_packets(
    linear,
    register_order,
    vector_elements,
    cta_thread_count,
    lane_width,
    scratch_elements,
    op,
    result_value_id,
    *,
    description,
):
    register_order = tuple(int(value) for value in register_order)
    vector_elements = int(vector_elements)
    packets = []
    for register in range(0, len(register_order), vector_elements):
        packet_registers = register_order[register:register + vector_elements]
        offsets = []
        for thread in range(int(cta_thread_count)):
            packet_offset = _swizzled_cta_exchange_offset(
                linear,
                register,
                thread,
                lane_width,
                op,
                result_value_id,
                description=description,
            )
            if packet_offset < 0 or packet_offset + vector_elements > int(scratch_elements):
                fail(
                    "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                    STAGE,
                    f"{description} optimal LDS packet exceeds scratch bounds",
                    source_op_index=op.index,
                    source_value_id=result_value_id,
                )
            for element in range(vector_elements):
                element_offset = _swizzled_cta_exchange_offset(
                    linear,
                    register + element,
                    thread,
                    lane_width,
                    op,
                    result_value_id,
                    description=description,
                )
                if element_offset != packet_offset + element:
                    fail(
                        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                        STAGE,
                        f"{description} optimal LDS packet is not contiguous",
                        source_op_index=op.index,
                        source_value_id=result_value_id,
                    )
            offsets.append(int(packet_offset))
        base, coefficients = _fit_bit_linear_offsets(
            offsets,
            cta_thread_count,
            op,
            result_value_id,
            description=description,
        )
        packets.append((
            tuple(packet_registers),
            int(base),
            tuple(int(value) for value in coefficients),
        ))
    return tuple(packets)


def _swizzled_cta_exchange_offset(
    linear,
    register,
    thread,
    lane_width,
    op,
    result_value_id,
    *,
    description,
):
    available = {
        "block": 0,
        "lane": int(thread) % int(lane_width),
        "register": int(register),
        "warp": int(thread) // int(lane_width),
    }
    input_names = tuple(str(name) for name in linear.get_in_dim_names())
    if any(name not in available for name in input_names):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} optimal LDS plan has unknown input dimensions",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    outputs = linear.apply({name: available[name] for name in input_names})
    if int(outputs.get("block", 0)) != 0 or "offset" not in outputs:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} optimal LDS plan is not block-local",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    return int(outputs["offset"])


def _fit_bit_linear_offsets(
    offsets,
    cta_thread_count,
    op,
    result_value_id,
    *,
    description,
):
    offsets = tuple(int(offset) for offset in offsets)
    if len(offsets) != int(cta_thread_count):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} produced a malformed optimal LDS address map",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    base = offsets[0]
    coefficients = tuple(
        int(offsets[1 << bit]) ^ int(base)
        for bit in range(int(cta_thread_count).bit_length() - 1)
    )
    for thread, offset in enumerate(offsets):
        if int(offset) != _apply_bit_linear_offset(base, coefficients, thread):
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} produced a non-linear optimal LDS address map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
    return int(base), coefficients


def _apply_bit_linear_offset(base, coefficients, thread):
    result = int(base)
    for bit, coefficient in enumerate(coefficients):
        if int(thread) & (1 << bit):
            result ^= int(coefficient)
    return int(result)


def cta_exchange_has_packet_group(exchange_groups, packet_elements):
    packet_elements = int(packet_elements)
    limit = len(exchange_groups) - packet_elements + 1
    for index in range(max(0, limit)):
        if cta_exchange_groups_form_packet(exchange_groups[index:index + packet_elements]):
            return True
    return False


def cta_exchange_groups_form_packet(groups):
    groups = tuple(groups)
    if len(groups) < 2:
        return False
    first_sources, first_results, first_bases, first_coefficients = groups[0]
    source_count = len(first_sources)
    result_count = len(first_results)
    if result_count == 0 or len(first_bases) != result_count or len(first_coefficients) != result_count:
        return False
    reference_bases = tuple(int(value) for value in first_bases)
    reference_coefficients = tuple(tuple(int(value) for value in coefficients) for coefficients in first_coefficients)
    result_indices = set()
    for source_slots, results, bases, coefficients in groups:
        if len(source_slots) != source_count:
            return False
        if len(results) != result_count or len(bases) != result_count or len(coefficients) != result_count:
            return False
        if tuple(int(value) for value in bases) != reference_bases:
            return False
        if tuple(tuple(int(value) for value in item) for item in coefficients) != reference_coefficients:
            return False
        for result_index in results:
            result_index = int(result_index)
            if result_index in result_indices:
                return False
            result_indices.add(result_index)
    return True


def _reject_distributed_movement(
    result_sources,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
    description,
):
    for sources in result_sources:
        for result_warp in range(int(cta_warp_count)):
            for lane in range(int(lane_width)):
                source_warp, _source_lane, _source_register = sources[result_warp * int(lane_width) + lane]
                if int(source_warp) != int(result_warp):
                    fail(
                        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                        STAGE,
                        f"{description} requires cross-warp movement",
                        source_op_index=op.index,
                        source_value_id=result_value_id,
                    )
        source_registers = {int(source[2]) for source in sources}
        if len(source_registers) > 1:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} requires per-lane source component selection",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"{description} has unknown movement class",
        source_op_index=op.index,
        source_value_id=result_value_id,
    )


def _distributed_movement_class(
    result_sources,
    lane_width,
    cta_warp_count,
):
    has_cross_warp = False
    has_lane_mux = False
    for sources in result_sources:
        for result_warp in range(int(cta_warp_count)):
            for lane in range(int(lane_width)):
                source_warp, _source_lane, _source_register = sources[result_warp * int(lane_width) + lane]
                if int(source_warp) != int(result_warp):
                    has_cross_warp = True
        source_registers = {int(source[2]) for source in sources}
        if len(source_registers) > 1:
            has_lane_mux = True
    if has_cross_warp:
        return "cross_warp"
    if has_lane_mux:
        return "lane_mux"
    return "unknown"


def _require_injective_layout(linear, op, source_value_id, description):
    if linear.is_injective():
        return
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"{description} is non-injective",
        source_op_index=op.index,
        source_value_id=source_value_id,
    )


def _fit_bit_affine_offsets(
    load_offsets,
    cta_thread_count,
    op,
    result_value_id,
    *,
    description="MFMA to blocked convert_layout",
):
    load_offsets = tuple(int(offset) for offset in load_offsets)
    if len(load_offsets) != int(cta_thread_count):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} produced a malformed CTA exchange load map",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if int(cta_thread_count) <= 0 or int(cta_thread_count) & (int(cta_thread_count) - 1):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} CTA exchange requires a power-of-two CTA "
            "thread count",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    base = load_offsets[0]
    coefficients = []
    for bit in range(int(cta_thread_count).bit_length() - 1):
        coefficients.append(load_offsets[1 << bit] - base)
    for thread in range(int(cta_thread_count)):
        expected = base
        for bit, coefficient in enumerate(coefficients):
            if thread & (1 << bit):
                expected += int(coefficient)
        if load_offsets[thread] != expected:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} requires a non-bit-affine CTA exchange "
                "load map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
    return int(base), tuple(int(value) for value in coefficients)


def _dot_operand_fragment_registers(element_type, instr_shape, op):
    if element_type in {"f16", "bf16"} and instr_shape in {
        (16, 16, 32),
        (32, 32, 16),
    }:
        return 4
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        "distributed to dot_operand fragment registers are not known for "
        f"element_type={element_type}, instr_shape={instr_shape}",
        source_op_index=op.index,
    )


def _fragment_elements_per_lane(element_type, registers, op, source_value_id):
    element_bits = {"f16": 16, "bf16": 16}.get(element_type)
    if element_bits is None:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"distributed to dot_operand does not support {element_type} fragments",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    return int(registers) * 32 // int(element_bits)


def _dot_operand_parent_warp_count(layout):
    parent = layout.properties.get("parent_properties", {})
    warps_per_cta = tuple(int(value) for value in parent.get("warps_per_cta", ()))
    result = 1
    for value in warps_per_cta:
        result *= max(1, int(value))
    return result


def _dot_operand_payload_linear_layout(
    layout,
    elements_per_lane,
    instr_shape,
    warps_per_cta,
    lane_width,
    cta_warp_count,
    op,
):
    register_count = int(layout.component_count) * int(elements_per_lane)
    input_sizes = (
        ("register", int(register_count)),
        ("lane", int(lane_width)),
        ("warp", int(cta_warp_count)),
    )
    for input_name, input_size in input_sizes:
        if input_size <= 0 or input_size & (input_size - 1):
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                "distributed to dot_operand LinearLayout requires a "
                f"power-of-two {input_name} extent, got {input_size}",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )

    def payload_coords(register, lane, warp):
        component, element = divmod(int(register), int(elements_per_lane))
        return _dot_operand_payload_coords(
            layout,
            component,
            element,
            int(lane),
            int(warp),
            int(elements_per_lane),
            instr_shape,
            warps_per_cta,
            op,
        )

    if any(payload_coords(0, 0, 0)):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand payload is not a zero-based "
            "LinearLayout",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )

    bases = []
    for input_name, input_size in input_sizes:
        input_bases = []
        for bit in range(input_size.bit_length() - 1):
            register = (1 << bit) if input_name == "register" else 0
            lane = (1 << bit) if input_name == "lane" else 0
            warp = (1 << bit) if input_name == "warp" else 0
            input_bases.append(list(payload_coords(register, lane, warp)))
        bases.append((input_name, input_bases))
    bases.append(("block", []))
    result = layouts.LinearLayout.from_bases(
        bases,
        [f"dim{dim}" for dim in range(len(layout.shape))],
        [int(extent) for extent in layout.shape],
        False,
    )

    for register in range(int(register_count)):
        for warp in range(int(cta_warp_count)):
            for lane in range(int(lane_width)):
                actual = layouts.linear_layout_coords(
                    result,
                    register,
                    lane,
                    warp=warp,
                )
                expected = payload_coords(register, lane, warp)
                if actual != expected:
                    fail(
                        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                        STAGE,
                        "distributed to dot_operand payload is not representable "
                        "as a bit-linear layout",
                        source_op_index=op.index,
                        source_value_id=layout.value_id,
                    )
    return result


def _dot_operand_payload_coords(
    layout,
    component,
    element,
    lane,
    result_warp,
    elements_per_lane,
    instr_shape,
    warps_per_cta,
    op,
):
    shape = tuple(int(dim) for dim in layout.shape)
    if len(shape) < 2:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand requires rank-2 tensors",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    op_idx = int(layout.properties.get("op_idx", -1))
    component = int(component)
    parent = layout.properties.get("parent_properties", {})
    parent_transposed = bool(parent.get("is_transposed", False))
    tiles_per_warp = tuple(int(value) for value in parent.get("tiles_per_warp", (1, 1)))
    if len(tiles_per_warp) < 2:
        tiles_per_warp = (1, 1)
    physical_role = 1 - op_idx if parent_transposed and op_idx in {0, 1} else op_idx
    payload_row, payload_col = _dot_operand_payload_local_coords(
        physical_role,
        lane,
        element,
        elements_per_lane,
        instr_shape,
        op,
        layout.value_id,
    )
    if op_idx == 0:
        k_tiles = _ceil_div(shape[1], instr_shape[2])
        if int(layout.component_count) % k_tiles:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                "A dot_operand component count is not divisible by K tiles",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        m_tile = component // k_tiles
        k_tile = component % k_tiles
        row_base = layouts.mfma_cta_tile_coordinate(
            m_tile,
            _dot_operand_wave_tile_coord(result_warp, warps_per_cta, "m"),
            warps_per_cta[0],
            tiles_per_warp[0],
        ) * int(instr_shape[0])
        col_base = k_tile * int(instr_shape[2])
        if parent_transposed:
            row = row_base + payload_col
            col = col_base + payload_row
        else:
            row = row_base + payload_row
            col = col_base + payload_col
    elif op_idx == 1:
        k_tiles = _ceil_div(shape[0], instr_shape[2])
        if int(layout.component_count) % k_tiles:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                "B dot_operand component count is not divisible by K tiles",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        n_tile = component // k_tiles
        k_tile = component % k_tiles
        row_base = k_tile * int(instr_shape[2])
        col_base = layouts.mfma_cta_tile_coordinate(
            n_tile,
            _dot_operand_wave_tile_coord(result_warp, warps_per_cta, "n"),
            warps_per_cta[1],
            tiles_per_warp[1],
        ) * int(instr_shape[1])
        if parent_transposed:
            row = row_base + payload_col
            col = col_base + payload_row
        else:
            row = row_base + payload_row
            col = col_base + payload_col
    else:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"unsupported dot operand index {op_idx}",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    coords = (int(row), int(col))
    if any(coord < 0 or coord >= extent for coord, extent in zip(coords, shape)):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand payload coordinate exceeds tensor shape",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    return coords


def _dot_operand_payload_local_coords(
    physical_role,
    lane,
    element,
    elements_per_lane,
    instr_shape,
    op,
    source_value_id,
):
    if len(instr_shape) < 3:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand payload requires rank-3 MFMA instrShape",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    if int(physical_role) == 0:
        rows = int(instr_shape[0])
        row = int(lane) % rows
        col = (int(lane) // rows) * int(elements_per_lane) + int(element)
        return int(row), int(col)
    if int(physical_role) == 1:
        columns = int(instr_shape[1])
        row = (int(lane) // columns) * int(elements_per_lane) + int(element)
        col = int(lane) % columns
        return int(row), int(col)
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"unsupported dot operand physical role {physical_role}",
        source_op_index=op.index,
        source_value_id=source_value_id,
    )


def _dot_operand_wave_tile_coord(result_warp, warps_per_cta, axis):
    warps_n = max(1, int(warps_per_cta[1]))
    if axis == "m":
        wave_coord = int(result_warp) // warps_n
        return wave_coord
    if axis == "n":
        return int(result_warp) % warps_n
    return 0


def _ceil_div(lhs, rhs):
    return (int(lhs) + int(rhs) - 1) // int(rhs)


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def _classify_source_lane_maps(
    source_lane_maps,
    lane_width,
    op,
    result_value_id,
    *,
    description="MFMA to blocked convert_layout",
):
    concrete_maps = [lane_map for lane_map in source_lane_maps if lane_map is not None]
    if not concrete_maps:
        return None
    first = concrete_maps[0]
    if not all(lane_map == first for lane_map in concrete_maps):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} requires per-component source "
            "lane maps; explicit lane-map remap support is required",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if len(first) != int(lane_width):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} produced a malformed source "
            "lane map",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if any(int(lane) < 0 or int(lane) >= int(lane_width) for lane in first):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} produced an out-of-range source "
            "lane map",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    kind, attrs = _classify_lane_map(first, lane_width)
    if kind is None:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} requires a non-affine source "
            "lane map; Wave shuffle emission only supports affine and "
            "2-D transpose lane maps",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    return {
        "source_lane_map": tuple(int(lane) for lane in first),
        "source_lane_map_kind": kind,
        **attrs,
    }


def _classify_lane_map(lane_map, lane_width):
    lane_width = int(lane_width)
    identity = tuple(range(lane_width))
    if tuple(lane_map) == identity:
        return None, {}
    if lane_width > 1:
        stride = int(lane_map[1]) - int(lane_map[0])
        base = int(lane_map[0])
        if all(int(value) == base + lane * stride for lane, value in enumerate(lane_map)):
            return "affine", {
                "source_lane_affine_base": int(base),
                "source_lane_affine_stride": int(stride),
            }
    for inner in _lane_width_factors(lane_width):
        outer = lane_width // inner
        transposed = tuple((lane % inner) * outer + lane // inner for lane in range(lane_width))
        if tuple(lane_map) == transposed:
            return "transpose", {
                "source_lane_transpose_inner": int(inner),
                "source_lane_transpose_outer": int(outer),
            }
    return None, {}


def _lane_width_factors(lane_width):
    lane_width = int(lane_width)
    for factor in range(2, lane_width):
        if lane_width % factor == 0:
            yield factor


def _distributed_linear_layout(layout, op):
    if layout.kind not in {"blocked", "linear", "generic_linear", "amd_mfma"}:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"layout {layout.kind} is not converted through linear-layout remap",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    return layouts.distributed_linear_layout(
        layout,
        stage=STAGE,
        source_op_index=op.index,
    )


def _layout_warp_count(layout):
    return layouts.layout_warp_count(layout)
