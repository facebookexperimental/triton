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

    source_layout = _redistribution_linear_layout(operand_layout, op)
    destination_layout = _redistribution_linear_layout(result_layout, op)
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

    source_linear = _redistribution_linear_layout(operand_layout, op)
    result_linear = _redistribution_linear_layout(result_layout, op)
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


def _redistribution_linear_layout(layout, op):
    return _distributed_linear_layout(layout, op)


def _redistribution_warp_count(layout):
    return _layout_warp_count(layout)


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


def _apply_bit_linear_offset(base, coefficients, thread):
    result = int(base)
    for bit, coefficient in enumerate(coefficients):
        if int(thread) & (1 << bit):
            result ^= int(coefficient)
    return int(result)


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


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def _distributed_linear_layout(layout, op):
    if layout.kind not in {
        "blocked",
        "linear",
        "generic_linear",
        "amd_mfma",
        "dot_operand",
    }:
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
