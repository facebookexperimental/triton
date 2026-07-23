"""Structural coordinate materialization plans for TLX Wave conversion."""

from dataclasses import dataclass

from .diagnostics import fail
from . import layouts

STAGE = "op_conversion"


@dataclass(frozen=True)
class CoordinatePlan:
    shape: tuple[int, ...]
    component_bases: tuple[tuple[int, ...], ...]
    workitem_coefficients: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PacketCoordinatePlan:
    shape: tuple[int, ...]
    component_bases: tuple[tuple[int, ...], ...]
    slot_coefficients: tuple[tuple[int, ...], ...]
    workitem_coefficients: tuple[tuple[int, ...], ...]


def layout_coordinate_plan(
    layout,
    component_count,
    lane_width,
    warp_count,
    op,
    source_value_id,
):
    if layout.kind not in {
            "blocked",
            "linear",
            "generic_linear",
            "slice",
            "amd_mfma",
    }:
        return None
    shape = tuple(int(dim) for dim in layout.shape)
    if not shape:
        _fail(
            "layout coordinate materialization requires ranked tensor shape",
            layout,
            op,
            source_value_id,
        )
    linear = layouts.distributed_linear_layout(
        layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    component_registers = layouts.linear_layout_component_registers(
        linear,
        layout,
        component_count,
        stage=STAGE,
        source_op_index=op.index,
        source_value_id=source_value_id,
    )
    lane_width = int(lane_width)
    warp_count = max(1, int(warp_count))
    if not _is_power_of_two(lane_width):
        _fail(
            f"lane width {lane_width} is not a power of two",
            layout,
            op,
            source_value_id,
        )
    if not _is_power_of_two(warp_count):
        _fail(
            f"warp count {warp_count} is not a power of two",
            layout,
            op,
            source_value_id,
        )
    lane_bits = lane_width.bit_length() - 1
    warp_bits = warp_count.bit_length() - 1
    workitem_coefficients = _workitem_coefficients(linear, lane_bits, warp_bits)
    component_bases = tuple(
        layouts.linear_layout_coords(linear, register, 0, warp=0) for register in component_registers)
    _validate_physical_domain(
        linear,
        shape,
        component_bases,
        workitem_coefficients,
        lane_width,
        warp_count,
        layout,
        op,
        source_value_id,
        component_registers,
    )
    return CoordinatePlan(
        shape=shape,
        component_bases=tuple(tuple(int(value) for value in bases) for bases in component_bases),
        workitem_coefficients=tuple(
            tuple(int(value) for value in coefficients) for coefficients in workitem_coefficients),
    )


def packet_layout_coordinate_plan(
    layout,
    component_count,
    packet_width,
    lane_width,
    warp_count,
    op,
    source_value_id,
):
    """Split a distributed register basis into component, slot, and item."""
    if layout.kind not in {
        "blocked",
        "linear",
        "generic_linear",
        "slice",
        "amd_mfma",
        "dot_operand",
    }:
        return None
    shape = tuple(int(dim) for dim in layout.shape)
    component_count = int(component_count)
    packet_width = int(packet_width)
    lane_width = int(lane_width)
    warp_count = int(warp_count)
    if (
        component_count <= 0
        or not _is_power_of_two(packet_width)
        or not _is_power_of_two(lane_width)
        or not _is_power_of_two(warp_count)
    ):
        _packet_fail(
            "packet coordinate materialization requires positive components "
            "and power-of-two packet, lane, and warp extents",
            layout,
            op,
            source_value_id,
        )
    linear = layouts.distributed_linear_layout(
        layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    register_count = layouts.linear_layout_in_dim_size(linear, "register")
    if register_count != component_count * packet_width:
        _packet_fail(
            "packet component grouping does not cover the layout register dimension",
            layout,
            op,
            source_value_id,
        )
    block_bases = layouts.linear_layout_bases(linear, "block")
    if any(any(int(value) for value in basis) for basis in block_bases):
        _packet_fail(
            "packet coordinate materialization does not support nontrivial block bases",
            layout,
            op,
            source_value_id,
        )

    register_bases = _logical_dim_bases(linear, "register")
    slot_bits = packet_width.bit_length() - 1
    if len(register_bases) < slot_bits:
        _packet_fail(
            "packet width exceeds the layout register basis",
            layout,
            op,
            source_value_id,
        )
    slot_coefficients = tuple(register_bases[:slot_bits])
    workitem_coefficients = _workitem_coefficients(
        linear,
        lane_width.bit_length() - 1,
        warp_count.bit_length() - 1,
    )
    component_bases = tuple(
        layouts.linear_layout_coords(
            linear,
            component * packet_width,
            0,
            warp=0,
        )
        for component in range(component_count)
    )

    for component, component_base in enumerate(component_bases):
        for slot in range(packet_width):
            register = component * packet_width + slot
            for warp in range(warp_count):
                for lane in range(lane_width):
                    workitem = warp * lane_width + lane
                    planned = _coords_from_packet_plan(
                        component_base,
                        slot_coefficients,
                        workitem_coefficients,
                        slot,
                        workitem,
                    )
                    actual = layouts.linear_layout_coords(
                        linear,
                        register,
                        lane,
                        warp=warp,
                    )
                    if planned != actual:
                        _packet_fail(
                            "packet coordinate plan does not match linear layout bases",
                            layout,
                            op,
                            source_value_id,
                        )
    return PacketCoordinatePlan(
        shape=shape,
        component_bases=tuple(
            tuple(int(value) for value in base) for base in component_bases
        ),
        slot_coefficients=tuple(
            tuple(int(value) for value in basis)
            for basis in slot_coefficients
        ),
        workitem_coefficients=tuple(
            tuple(int(value) for value in basis)
            for basis in workitem_coefficients
        ),
    )


def packet_slots_are_contiguous_along_dimension(plan, dimension):
    dimension = int(dimension)
    if dimension < 0 or dimension >= len(plan.shape):
        return False
    expected = []
    for bit in range(len(plan.slot_coefficients)):
        basis = [0] * len(plan.shape)
        basis[dimension] = 1 << bit
        expected.append(tuple(basis))
    return tuple(plan.slot_coefficients) == tuple(expected)


def is_default_flat_make_range(plan, lane_width):
    if len(plan.shape) != 1:
        return False
    if plan.workitem_coefficients != tuple((1 << bit, ) for bit in range(int(lane_width).bit_length() - 1)):
        return False
    return plan.component_bases == tuple(
        (component * int(lane_width), ) for component in range(len(plan.component_bases)))


def is_flat_affine_make_range(plan, lane_width, warp_count):
    if len(plan.shape) != 1:
        return None
    lane_width = int(lane_width)
    warp_count = int(warp_count)
    total_bits = (lane_width * warp_count).bit_length() - 1
    if total_bits == 0:
        stride = 0
    elif not plan.workitem_coefficients:
        return None
    else:
        stride = int(plan.workitem_coefficients[0][0])
    if plan.workitem_coefficients != tuple(((stride * (1 << bit)), ) for bit in range(total_bits)):
        return None
    bases = tuple(int(base[0]) for base in plan.component_bases)
    coefficients = tuple(int(coefficient[0]) for coefficient in plan.workitem_coefficients)
    if not _xor_masks_are_additive(bases, coefficients):
        return None
    if not bases:
        return (), stride
    return bases, stride


def is_flat_bit_affine_make_range(plan):
    if len(plan.shape) != 1:
        return None
    bases = tuple(int(base[0]) for base in plan.component_bases)
    coefficients = tuple(int(coefficient[0]) for coefficient in plan.workitem_coefficients)
    if not _xor_masks_are_additive(bases, coefficients):
        return None
    return bases, coefficients


def _workitem_coefficients(linear, lane_bits, warp_bits):
    lane_bases = _logical_dim_bases(linear, "lane")
    warp_bases = _logical_dim_bases(linear, "warp")
    rank = _basis_rank(linear)
    zero = tuple(0 for _ in range(rank))
    coefficients = []
    for bit in range(int(lane_bits)):
        coefficients.append(lane_bases[bit] if bit < len(lane_bases) else zero)
    for bit in range(int(warp_bits)):
        coefficients.append(warp_bases[bit] if bit < len(warp_bases) else zero)
    return tuple(tuple(int(value) for value in basis) for basis in coefficients)


def _logical_dim_bases(linear, in_dim):
    return tuple(_basis_in_logical_dim_order(linear, basis) for basis in layouts.linear_layout_bases(linear, in_dim))


def _basis_in_logical_dim_order(linear, basis):
    basis = tuple(int(value) for value in basis)
    out_dims = tuple(linear.out_dims)
    if len(out_dims) != len(basis):
        return basis
    dims = []
    for out_dim in out_dims:
        name = str(out_dim[0])
        if not name.startswith("dim"):
            return basis
        try:
            dims.append(int(name[3:]))
        except ValueError:
            return basis
    if sorted(dims) != list(range(len(dims))):
        return basis
    result = [0] * len(dims)
    for index, dim in enumerate(dims):
        result[dim] = basis[index]
    return tuple(result)


def _basis_rank(linear):
    for _name, bases in linear.bases:
        if bases:
            return len(bases[0])
    return len(linear.out_dims)


def _validate_physical_domain(
    linear,
    shape,
    component_bases,
    workitem_coefficients,
    lane_width,
    warp_count,
    layout,
    op,
    source_value_id,
    component_registers,
):
    seen = set()
    duplicate_seen = False
    physical_slots = 0
    for component, component_base in enumerate(component_bases):
        component_register = int(component_registers[component])
        for warp in range(int(warp_count)):
            for lane in range(int(lane_width)):
                physical_slots += 1
                workitem = warp * int(lane_width) + lane
                planned = _coords_from_plan(
                    component_base,
                    workitem_coefficients,
                    workitem,
                )
                actual = layouts.linear_layout_coords(
                    linear,
                    component_register,
                    lane,
                    warp=warp,
                )
                if tuple(planned) != tuple(actual):
                    _fail(
                        "layout coordinate plan does not match linear layout bases",
                        layout,
                        op,
                        source_value_id,
                    )
                if len(actual) != len(shape):
                    _fail(
                        "layout coordinate rank does not match tensor rank",
                        layout,
                        op,
                        source_value_id,
                    )
                for coord, extent in zip(actual, shape):
                    if int(coord) < 0 or int(coord) >= int(extent):
                        _fail(
                            "layout coordinate map produces out-of-bounds coordinates",
                            layout,
                            op,
                            source_value_id,
                        )
                duplicate_seen = duplicate_seen or tuple(actual) in seen
                seen.add(tuple(actual))
    block_count = layouts.linear_layout_in_dim_size(linear, "block")
    total_logical_slots = _product(shape)
    if total_logical_slots % int(block_count):
        _fail(
            "layout coordinate block domain does not divide the tensor shape",
            layout,
            op,
            source_value_id,
        )
    local_logical_slots = total_logical_slots // int(block_count)
    if _is_mfma_component_layout(layout):
        return
    if len(seen) != local_logical_slots:
        if duplicate_seen and int(physical_slots) <= int(local_logical_slots):
            _fail(
                "layout coordinate map is non-injective over physical lanes",
                layout,
                op,
                source_value_id,
            )
        _fail(
            "layout coordinate map does not cover the tensor shape exactly",
            layout,
            op,
            source_value_id,
        )


def _coords_from_plan(component_base, workitem_coefficients, workitem):
    coords = [int(value) for value in component_base]
    for bit, coefficients in enumerate(workitem_coefficients):
        if not (int(workitem) & (1 << bit)):
            continue
        for dim, coefficient in enumerate(coefficients):
            coords[dim] ^= int(coefficient)
    return tuple(coords)


def _coords_from_packet_plan(
    component_base,
    slot_coefficients,
    workitem_coefficients,
    slot,
    workitem,
):
    coords = list(
        _coords_from_plan(component_base, workitem_coefficients, workitem)
    )
    for bit, coefficients in enumerate(slot_coefficients):
        if not (int(slot) & (1 << bit)):
            continue
        for dim, coefficient in enumerate(coefficients):
            coords[dim] ^= int(coefficient)
    return tuple(coords)


def _xor_masks_are_additive(bases, coefficients):
    occupied = 0
    for coefficient in coefficients:
        coefficient = int(coefficient)
        if coefficient < 0:
            return False
        if occupied & coefficient:
            return False
        occupied |= coefficient
    for base in bases:
        base = int(base)
        if base < 0 or (base & occupied):
            return False
    return True


def _basis_pattern(layout):
    if layout.kind == "linear":
        return {
            "register": tuple(layout.properties.get("register_bases", ())),
            "lane": tuple(layout.properties.get("lane_bases", ())),
            "warp": tuple(layout.properties.get("warp_bases", ())),
            "block": tuple(layout.properties.get("block_bases", ())),
        }
    if layout.kind == "blocked":
        return {
            "size_per_thread": tuple(layout.properties.get("size_per_thread", ())),
            "threads_per_warp": tuple(layout.properties.get("threads_per_warp", ())),
            "warps_per_cta": tuple(layout.properties.get("warps_per_cta", ())),
            "order": tuple(layout.properties.get("order", ())),
        }
    return dict(layout.properties)


def _is_mfma_component_layout(layout):
    if layout.kind == "amd_mfma":
        return True
    if layout.kind == "slice":
        return layout.properties.get("parent_kind") == "amd_mfma"
    return False


def _fail(message, layout, op, source_value_id):
    fail(
        "TLXW_OP_MAKE_RANGE_LAYOUT",
        STAGE,
        f"tt.make_range {message}; layout={layout.kind} "
        f"shape={tuple(layout.shape)} bases={_basis_pattern(layout)}",
        source_op_index=op.index,
        source_value_id=source_value_id,
    )


def _packet_fail(message, layout, op, source_value_id):
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"local_load {message}; layout={layout.kind} "
        f"shape={tuple(layout.shape)} bases={_basis_pattern(layout)}",
        source_op_index=op.index,
        source_value_id=source_value_id,
    )


def _is_power_of_two(value):
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result
