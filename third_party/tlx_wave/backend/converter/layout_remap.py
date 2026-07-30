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
def redistribution_plan(
    operand,
    result,
    operand_layout,
    result_layout,
    op,
    *,
    source_coordinate_transform=None,
):
    """Build the complete destination-to-source packet relation.

    Layout coordinates are the semantic witness for a layout conversion.  The
    bridge carries a checked bit-linear gather relation to emission; movement
    classification and lowering belong to Wave's redistribute pass.
    """
    if operand_layout is None or result_layout is None:
        return None
    if operand.type.element_type != result.type.element_type:
        return None
    if (
        source_coordinate_transform is None
        and tuple(operand_layout.shape) != tuple(result_layout.shape)
    ):
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
    relation_bases, relation_out_dims = _redistribution_relation_plan(
        source_layout,
        destination_layout,
        source_slots,
        destination_slots,
        lane_width,
        source_warps,
        source_blocks,
        len(operand_layout.shape),
        len(result_layout.shape),
        op,
        operand.value_id,
        result.value_id,
        source_coordinate_transform=source_coordinate_transform,
    )
    return {
        "mode": "redistribute",
        "block_count": source_blocks,
        "cta_thread_count": lane_width * source_warps,
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


def structural_view_redistribution_plan(
    operand,
    result,
    operand_layout,
    result_layout,
    op,
):
    """Translate a structural view to the generic symbolic layout relation."""
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

    order = _structural_view_order(op, source_shape, result_shape)
    plan = redistribution_plan(
        operand,
        result,
        operand_layout,
        result_layout,
        op,
        source_coordinate_transform=lambda coords: _structural_view_result_coords(
            op.name,
            coords,
            source_shape,
            result_shape,
            order,
        ),
    )
    if plan is None:
        _structural_view_fail(
            op,
            result.value_id,
            f"{op.name} cannot be represented as a symbolic layout relation",
        )
    return plan


def structural_join_plan(
    operands,
    result,
    operand_layouts,
    result_layout,
    op,
):
    """Describe a same-workitem ``tt.join`` as scalar register selection.

    Triton's join appends a two-element minor dimension.  It is structural
    when every result register slot is supplied by a register slot owned by
    the same block, warp, and lane in one input.  The plan is expressed over
    scalar register slots and separately records bridge packet grouping, so
    vector-valued MFMA payloads can be unpacked and regrouped without turning
    an in-thread representation change into a semantic redistribution.
    """
    if len(operands) != 2 or len(operand_layouts) != 2:
        _structural_join_fail(op, result.value_id, "tt.join requires two operands")
    if result_layout is None or any(layout is None for layout in operand_layouts):
        _structural_join_fail(
            op,
            result.value_id,
            "tt.join requires distributed operand and result layouts",
        )
    first, second = operands
    source_shape = tuple(int(dim) for dim in operand_layouts[0].shape)
    if tuple(int(dim) for dim in operand_layouts[1].shape) != source_shape:
        _structural_join_fail(op, result.value_id, "tt.join operand shapes must match")
    result_shape = tuple(int(dim) for dim in result_layout.shape)
    if result_shape != source_shape + (2,):
        _structural_join_fail(
            op,
            result.value_id,
            "tt.join result shape must append a two-element minor dimension",
        )
    if (
        first.type.kind != "tensor"
        or second.type.kind != "tensor"
        or result.type.kind != "tensor"
        or first.type.element_type != second.type.element_type
        or first.type.element_type != result.type.element_type
    ):
        _structural_join_fail(
            op,
            result.value_id,
            "tt.join requires matching tensor element types",
        )

    lane_width = int(result.type.lane_width or 64)
    if any(int(value.type.lane_width or 64) != lane_width for value in operands):
        _structural_join_fail(op, result.value_id, "tt.join changed its wave width")
    source_linears = tuple(
        _redistribution_linear_layout(layout, op) for layout in operand_layouts
    )
    result_linear = _redistribution_linear_layout(result_layout, op)
    source_slots = tuple(
        layouts.linear_layout_in_dim_size(linear, "register")
        for linear in source_linears
    )
    result_slots = layouts.linear_layout_in_dim_size(result_linear, "register")
    source_components = tuple(int(value.type.component_count) for value in operands)
    result_components = int(result.type.component_count)
    if any(slots % components for slots, components in zip(source_slots, source_components)):
        _structural_join_fail(
            op,
            result.value_id,
            "tt.join source packets do not evenly cover register slots",
        )
    if result_slots % result_components:
        _structural_join_fail(
            op,
            result.value_id,
            "tt.join result packets do not evenly cover register slots",
        )

    source_warps = tuple(_redistribution_warp_count(layout) for layout in operand_layouts)
    result_warps = _redistribution_warp_count(result_layout)
    source_blocks = tuple(
        layouts.linear_layout_in_dim_size(linear, "block")
        for linear in source_linears
    )
    result_blocks = layouts.linear_layout_in_dim_size(result_linear, "block")
    if any(count != result_warps for count in source_warps):
        _structural_join_fail(op, result.value_id, "tt.join changed its workgroup size")
    if any(count != result_blocks for count in source_blocks):
        _structural_join_fail(op, result.value_id, "tt.join changed its cluster size")

    source_by_item_and_coordinate = []
    for linear, slot_count, operand in zip(source_linears, source_slots, operands):
        source_map = {}
        for block in range(int(result_blocks)):
            for warp in range(int(result_warps)):
                for lane in range(lane_width):
                    for slot in range(int(slot_count)):
                        coordinate = _redistribution_coords(
                            linear,
                            slot,
                            lane,
                            warp,
                            block,
                            len(source_shape),
                            op,
                            operand.value_id,
                            "tt.join structural component mapping",
                        )
                        source_map.setdefault(
                            (block, warp, lane, coordinate),
                            [],
                        ).append(slot)
        source_by_item_and_coordinate.append(source_map)

    scalar_sources = []
    for result_slot in range(int(result_slots)):
        sources = set()
        for block in range(int(result_blocks)):
            for warp in range(int(result_warps)):
                for lane in range(lane_width):
                    coordinate = _redistribution_coords(
                        result_linear,
                        result_slot,
                        lane,
                        warp,
                        block,
                        len(result_shape),
                        op,
                        result.value_id,
                        "tt.join structural component mapping",
                    )
                    operand_index = int(coordinate[-1])
                    if operand_index not in (0, 1):
                        _structural_join_fail(
                            op,
                            result.value_id,
                            "tt.join result selector is outside the appended dimension",
                        )
                    candidates = source_by_item_and_coordinate[operand_index].get(
                        (block, warp, lane, tuple(coordinate[:-1])),
                        (),
                    )
                    if not candidates:
                        _structural_join_fail(
                            op,
                            result.value_id,
                            "tt.join result coordinate is not owned by the selected operand workitem",
                        )
                    sources.add((operand_index, min(int(candidate) for candidate in candidates)))
        if len(sources) != 1:
            _structural_join_fail(
                op,
                result.value_id,
                "tt.join scalar source varies across workitems",
            )
        scalar_sources.append(sources.pop())

    return {
        "scalar_sources": tuple(scalar_sources),
        "source_component_counts": source_components,
        "source_packet_widths": tuple(
            slots // components
            for slots, components in zip(source_slots, source_components)
        ),
        "source_slot_counts": source_slots,
        "result_component_count": result_components,
        "result_packet_width": result_slots // result_components,
        "result_slot_count": result_slots,
    }


def _structural_join_fail(op, value_id, message):
    fail(
        "TLXW_OP_STRUCTURAL_JOIN",
        STAGE,
        message,
        source_op_index=op.index,
        source_value_id=value_id,
    )


def structural_split_plan(
    operand,
    results,
    operand_layout,
    result_layouts,
    op,
):
    """Describe a same-workitem ``tt.split`` as scalar register selection.

    Triton's split removes a trailing two-element dimension and returns one
    tensor for each selector value.  The lowering is structural exactly when
    every output register slot is already owned by the same block, warp, and
    lane in the input.  Packet grouping is recorded separately so this also
    handles vector-valued register payloads without exposing their physical
    representation through source operation boundaries.
    """
    if len(results) != 2 or len(result_layouts) != 2:
        _structural_split_fail(
            op,
            operand.value_id,
            "tt.split requires two results",
        )
    if operand_layout is None or any(layout is None for layout in result_layouts):
        _structural_split_fail(
            op,
            operand.value_id,
            "tt.split requires distributed operand and result layouts",
        )
    source_shape = tuple(int(dim) for dim in operand_layout.shape)
    if not source_shape or source_shape[-1] != 2:
        _structural_split_fail(
            op,
            operand.value_id,
            "tt.split operand shape must end in a two-element dimension",
        )
    result_shape = source_shape[:-1]
    if any(
        tuple(int(dim) for dim in layout.shape) != result_shape
        for layout in result_layouts
    ):
        _structural_split_fail(
            op,
            operand.value_id,
            "tt.split result shapes must remove the trailing selector dimension",
        )
    if (
        operand.type.kind != "tensor"
        or any(result.type.kind != "tensor" for result in results)
        or any(
            result.type.element_type != operand.type.element_type
            for result in results
        )
    ):
        _structural_split_fail(
            op,
            operand.value_id,
            "tt.split requires matching tensor element types",
        )

    lane_width = int(operand.type.lane_width or 64)
    if any(int(result.type.lane_width or 64) != lane_width for result in results):
        _structural_split_fail(op, operand.value_id, "tt.split changed its wave width")
    source_linear = _redistribution_linear_layout(operand_layout, op)
    result_linears = tuple(
        _redistribution_linear_layout(layout, op) for layout in result_layouts
    )
    source_slots = layouts.linear_layout_in_dim_size(source_linear, "register")
    result_slots = tuple(
        layouts.linear_layout_in_dim_size(linear, "register")
        for linear in result_linears
    )
    source_components = int(operand.type.component_count)
    result_components = tuple(int(result.type.component_count) for result in results)
    if source_slots % source_components:
        _structural_split_fail(
            op,
            operand.value_id,
            "tt.split source packets do not evenly cover register slots",
        )
    if any(slots % components for slots, components in zip(result_slots, result_components)):
        _structural_split_fail(
            op,
            operand.value_id,
            "tt.split result packets do not evenly cover register slots",
        )

    source_warps = _redistribution_warp_count(operand_layout)
    result_warps = tuple(
        _redistribution_warp_count(layout) for layout in result_layouts
    )
    source_blocks = layouts.linear_layout_in_dim_size(source_linear, "block")
    result_blocks = tuple(
        layouts.linear_layout_in_dim_size(linear, "block")
        for linear in result_linears
    )
    if any(count != source_warps for count in result_warps):
        _structural_split_fail(
            op,
            operand.value_id,
            "tt.split changed its workgroup size",
        )
    if any(count != source_blocks for count in result_blocks):
        _structural_split_fail(
            op,
            operand.value_id,
            "tt.split changed its cluster size",
        )

    source_by_item_and_coordinate = {}
    for block in range(int(source_blocks)):
        for warp in range(int(source_warps)):
            for lane in range(lane_width):
                for slot in range(int(source_slots)):
                    coordinate = _redistribution_coords(
                        source_linear,
                        slot,
                        lane,
                        warp,
                        block,
                        len(source_shape),
                        op,
                        operand.value_id,
                        "tt.split structural component mapping",
                    )
                    source_by_item_and_coordinate.setdefault(
                        (block, warp, lane, coordinate),
                        [],
                    ).append(slot)

    scalar_source_slots = []
    for selector, (result, result_linear, slot_count) in enumerate(
        zip(results, result_linears, result_slots)
    ):
        result_sources = []
        for result_slot in range(int(slot_count)):
            sources = set()
            for block in range(int(source_blocks)):
                for warp in range(int(source_warps)):
                    for lane in range(lane_width):
                        coordinate = _redistribution_coords(
                            result_linear,
                            result_slot,
                            lane,
                            warp,
                            block,
                            len(result_shape),
                            op,
                            result.value_id,
                            "tt.split structural component mapping",
                        )
                        candidates = source_by_item_and_coordinate.get(
                            (block, warp, lane, tuple(coordinate) + (selector,)),
                            (),
                        )
                        if not candidates:
                            _structural_split_fail(
                                op,
                                result.value_id,
                                "tt.split result coordinate is not owned by the same input workitem",
                            )
                        sources.add(min(int(candidate) for candidate in candidates))
            if len(sources) != 1:
                _structural_split_fail(
                    op,
                    result.value_id,
                    "tt.split scalar source varies across workitems",
                )
            result_sources.append(sources.pop())
        scalar_source_slots.append(tuple(result_sources))

    return {
        "source_component_count": source_components,
        "source_packet_width": source_slots // source_components,
        "source_slot_count": source_slots,
        "scalar_source_slots": tuple(scalar_source_slots),
        "result_component_counts": result_components,
        "result_packet_widths": tuple(
            slots // components
            for slots, components in zip(result_slots, result_components)
        ),
        "result_slot_counts": result_slots,
    }


def _structural_split_fail(op, value_id, message):
    fail(
        "TLXW_OP_STRUCTURAL_SPLIT",
        STAGE,
        message,
        source_op_index=op.index,
        source_value_id=value_id,
    )


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
    source_rank,
    destination_rank,
    op,
    source_value_id,
    result_value_id,
    *,
    source_coordinate_transform=None,
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
                    source_coords = _redistribution_coords(
                        source_layout,
                        slot,
                        lane,
                        warp,
                        block,
                        source_rank,
                        op,
                        source_value_id,
                        description,
                    )
                    coords = (
                        source_coords
                        if source_coordinate_transform is None
                        else tuple(source_coordinate_transform(source_coords))
                    )
                    source_by_coord.setdefault(coords, []).append(
                        (int(slot), int(lane), int(warp), int(block))
                    )

    relation = {}
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
                        destination_rank,
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
    return encoded_bases, output_dims


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
        f"{operand_kind} to {result_kind} convert_layout cannot be represented "
        "as a symbolic layout relation",
        source_op_index=op.index,
    )


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
        "slice",
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
