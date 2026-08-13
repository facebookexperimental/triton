"""Structural layout maps for the TLX Wave converter."""

from dataclasses import dataclass

from triton._C.libtriton import linear_layout as _linear_layout

from ..wave_bridge_tools import load_wave_dsl
from .diagnostics import fail
from .layout_domains import _is_power_of_two, classify_coordinate_domain

LinearLayout = _linear_layout.LinearLayout

STAGE = "type_layout"


@dataclass(frozen=True)
class LayoutMap:
    layout_map_id: int
    value_id: int
    kind: str
    shape: tuple[int, ...]
    element_type: str | None
    component_count: int
    lane_width: int
    properties: dict
    linear_layout: object | None = None
    component_grid: tuple[int, int] | None = None
    component_relation: tuple[int, ...] | None = None
    active_relation: tuple[int, ...] | None = None


@dataclass(frozen=True)
class MfmaInstructionSpec:
    rows: int
    columns: int
    operand_registers: int
    accumulator_registers: int
    operand_element_types: tuple[str, ...]


_MFMA_INSTRUCTION_SPECS = {
    (16, 16, 32): MfmaInstructionSpec(16, 16, 4, 4, ("f16", "bf16")),
    (16, 16, 128): MfmaInstructionSpec(16, 16, 4, 4, ("i8", )),
    (32, 32, 16): MfmaInstructionSpec(32, 32, 4, 16, ("f16", "bf16")),
    (32, 32, 64): MfmaInstructionSpec(32, 32, 4, 16, ("i8", )),
}


def mfma_instruction_spec(instr_shape):
    return _MFMA_INSTRUCTION_SPECS.get(tuple(int(value) for value in instr_shape))


@dataclass(frozen=True)
class _SharedAddressMap:
    linear_layout: object
    prefix_shape: tuple[int, ...]
    inner_shape: tuple[int, ...]


@dataclass(frozen=True)
class _MmaComponentModel:
    component_count: int
    grid: tuple[int, int]
    relation: tuple[int, ...]


def build_layout_map(layout_map_id, value_id, source_type, lane_width, warp_count=1, block_count=1):
    if source_type.kind not in {"tensor", "memdesc"}:
        return None
    attr = source_type.encoding_attr
    kind, properties = _layout_kind_and_properties(attr, value_id, encoding=str(source_type.encoding or ""))
    distributed_kinds = {
        "amd_mfma",
        "blocked",
        "dot_operand",
        "generic_linear",
        "linear",
        "slice",
    }
    canonical_shape = tuple(source_type.shape)
    if source_type.kind == "memdesc" and source_type.alloc_shape:
        canonical_shape = tuple(source_type.alloc_shape[-len(source_type.shape):])
    if kind in distributed_kinds:
        linear_layout = _linear_layout.to_linear_layout(canonical_shape, attr)
    elif kind in {"shared_linear", "swizzled_shared"}:
        component_rank = len(properties["order"])
        if component_rank <= 0 or component_rank > len(canonical_shape):
            fail(
                "TLXW_TYPE_MALFORMED_LAYOUT",
                STAGE,
                "shared layout rank does not fit the allocation shape",
                source_value_id=value_id,
            )
        linear_layout = _linear_layout.to_linear_layout(canonical_shape[-component_rank:], attr)
    elif kind == "padded_shared":
        # Padding is composed after the canonical linear component.
        linear_layout = properties["linear_component"]
    else:
        linear_layout = None
    mma_model = None
    if kind in {"amd_mfma", "dot_operand"}:
        mma_model = _mma_component_model(kind, source_type.element_type, properties, linear_layout,
                                         source_value_id=value_id)
        component_count = int(mma_model.component_count)
    elif kind in {"blocked", "linear", "generic_linear"}:
        coordinate_domain = classify_coordinate_domain(source_type.shape, lane_width, linear=linear_layout)
        _require_supported_coordinate_domain(source_type.shape, coordinate_domain, value_id)
        properties = {**properties, "coordinate_domain": coordinate_domain}
        component_count = int(coordinate_domain["component_count"])
    else:
        component_count = _layout_component_count(source_type, lane_width, linear_layout)
    active_relation = (_distributed_active_relation(
        linear_layout,
        lane_width=int(lane_width),
        warp_count=int(warp_count),
        block_count=int(block_count),
        source_value_id=value_id,
    ) if kind in distributed_kinds else None)
    return LayoutMap(
        layout_map_id,
        value_id,
        kind,
        tuple(source_type.shape),
        source_type.element_type,
        int(component_count),
        int(lane_width),
        properties,
        linear_layout,
        None if mma_model is None else mma_model.grid,
        None if mma_model is None else mma_model.relation,
        active_relation,
    )


def _layout_kind_and_properties(attr, value_id, *, encoding=None):
    if attr is None:
        return "none", {}
    if _attr_bool(attr, "is_tlx_no_verify_layout"):
        inner = _attr_value(attr, "get_tlx_no_verify_layout")
        return _layout_kind_and_properties(inner, value_id, encoding=str(inner))
    if _attr_bool(attr, "is_tlx_user_layout"):
        inner = _attr_value(attr, "get_tlx_user_layout")
        return _layout_kind_and_properties(inner, value_id, encoding=str(inner))
    if _attr_bool(attr, "is_blocked_encoding"):
        return "blocked", {
            "size_per_thread": _int_tuple(_attr_value(attr, "get_blocked_size_per_thread")),
            "threads_per_warp": _int_tuple(_attr_value(attr, "get_blocked_threads_per_warp")),
            "warps_per_cta": _int_tuple(_attr_value(attr, "get_blocked_warps_per_cta")),
            "order": _int_tuple(_attr_value(attr, "get_blocked_order")),
        }
    if _attr_bool(attr, "is_linear_encoding"):
        kind = ("generic_linear" if str(encoding or "").startswith("#ttg.generic_linear") else "linear")
        return kind, {
            "register_bases": _basis_tuple(_attr_value(attr, "get_linear_register_bases")),
            "lane_bases": _basis_tuple(_attr_value(attr, "get_linear_lane_bases")),
            "warp_bases": _basis_tuple(_attr_value(attr, "get_linear_warp_bases")),
            "block_bases": _basis_tuple(_attr_value(attr, "get_linear_block_bases")),
            "linear_encoding_kind": kind,
        }
    if _attr_bool(attr, "is_slice_encoding"):
        parent = _attr_value(attr, "get_slice_parent")
        parent_kind, parent_properties = _layout_kind_and_properties(parent, value_id, encoding=str(parent or ""))
        return "slice", {
            "dim": int(_attr_value(attr, "get_slice_dim")),
            "parent_kind": parent_kind,
            "parent_properties": parent_properties,
        }
    if _attr_bool(attr, "is_dot_operand_encoding"):
        parent = _attr_value(attr, "get_dot_operand_parent")
        parent_kind, parent_properties = _layout_kind_and_properties(parent, value_id, encoding=str(parent or ""))
        return "dot_operand", {
            "op_idx": int(_attr_value(attr, "get_dot_operand_op_idx")),
            "k_width": int(_attr_value(attr, "get_dot_operand_k_width")),
            "parent_kind": parent_kind,
            "parent_properties": parent_properties,
        }
    if _attr_bool(attr, "is_amd_mfma_encoding"):
        return "amd_mfma", {
            "version": int(_attr_value(attr, "get_amd_mfma_version")),
            "warps_per_cta": _int_tuple(_attr_value(attr, "get_amd_mfma_warps_per_cta")),
            "instr_shape": _int_tuple(_attr_value(attr, "get_amd_mfma_instr_shape")),
            "is_transposed": bool(_attr_value(attr, "get_amd_mfma_is_transposed")),
            "tiles_per_warp": _int_tuple(_attr_value(attr, "get_amd_mfma_tiles_per_warp")),
            "element_bit_width": int(_attr_value(attr, "get_amd_mfma_element_bit_width")),
        }
    if _attr_bool(attr, "is_swizzled_shared_encoding"):
        return "swizzled_shared", {
            "vec": int(_attr_value(attr, "get_swizzled_shared_vec")),
            "per_phase": int(_attr_value(attr, "get_swizzled_shared_per_phase")),
            "max_phase": int(_attr_value(attr, "get_swizzled_shared_max_phase")),
            "order": _int_tuple(_attr_value(attr, "get_swizzled_shared_order")),
        }
    if _attr_bool(attr, "is_shared_linear_encoding"):
        return "shared_linear", {
            "alignment": int(_attr_value(attr, "get_shared_linear_alignment")),
            "linear_component": _attr_value(attr, "get_shared_linear_layout"),
            "order": _int_tuple(_attr_value(attr, "get_shared_linear_order")),
        }
    if _attr_bool(attr, "is_padded_shared_encoding"):
        return "padded_shared", {
            "intervals": _int_tuple(_attr_value(attr, "get_padded_shared_intervals")),
            "paddings": _int_tuple(_attr_value(attr, "get_padded_shared_paddings")),
            "order": _int_tuple(_attr_value(attr, "get_padded_shared_order")),
            "linear_component": _attr_value(attr, "get_padded_shared_linear_component"),
        }
    fail("TLXW_TYPE_UNSUPPORTED_LAYOUT", STAGE, f"unsupported layout encoding {attr}", source_value_id=value_id)


def _layout_component_count(source_type, lane_width, linear):
    if linear is not None:
        return linear_layout_in_dim_size(linear, "register")
    element_count = _product(source_type.shape)
    return max(1, _ceil_div(element_count, int(lane_width)))


def _mma_fragment_abi(
    kind,
    element_type,
    properties,
    lane_count,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_MALFORMED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):

    def reject(message):
        _layout_fail(diagnostic, stage, message, source_op_index=source_op_index, source_value_id=source_value_id)

    if kind == "amd_mfma":
        instr_shape = tuple(int(value) for value in properties.get("instr_shape", ()))
        op_idx = None
    elif kind == "dot_operand":
        parent = properties.get("parent_properties", {})
        instr_shape = tuple(int(value) for value in parent.get("instr_shape", ()))
        op_idx = int(properties.get("op_idx", -1))
    else:
        reject("MFMA component model requires an accumulator or operand layout")
    spec = mfma_instruction_spec(instr_shape)
    if spec is None or len(instr_shape) != 3 or (op_idx is not None and op_idx not in (0, 1)):
        reject("MFMA fragment quotient requires a supported instruction ABI")

    lane_count = int(lane_count)
    if lane_count <= 0:
        reject("MFMA instruction ABI requires a nonempty lane domain")

    if kind == "amd_mfma":
        fragment = (int(spec.rows), int(spec.columns))
        if _product(fragment) % lane_count:
            reject("MFMA accumulator fragment does not divide the lane domain")
        payload_elements = _product(fragment) // lane_count
        payload_element_bits = 32
        payload_registers = int(spec.accumulator_registers)
    else:
        element_bits = {
            "bf16": 16,
            "f16": 16,
            "i8": 8,
        }.get(str(element_type))
        if element_bits is None or element_type not in spec.operand_element_types:
            reject("MFMA operand fragment payload type is not supported by the instruction ABI")
        payload_element_bits = int(element_bits)
        payload_registers = int(spec.operand_registers)
        payload_bits = payload_registers * 32
        if payload_bits % payload_element_bits:
            reject("MFMA operand register payload does not contain whole elements")
        payload_elements = payload_bits // payload_element_bits
        fragment_elements = payload_elements * lane_count
        fragment = [0, 0]
        fragment[op_idx] = int(instr_shape[op_idx])
        if fragment_elements % fragment[op_idx]:
            reject("MFMA operand payload does not form an integral instruction fragment")
        fragment[1 - op_idx] = fragment_elements // fragment[op_idx]
        fragment = tuple(fragment)

    if (payload_elements <= 0 or not _is_power_of_two(payload_elements)
            or payload_elements * payload_element_bits != payload_registers * 32
            or any(extent <= 0 or not _is_power_of_two(extent) for extent in fragment)):
        reject("MFMA instruction ABI does not define a binary physical fragment")
    return payload_elements, payload_element_bits, payload_registers, fragment


def _mma_component_model(
    kind,
    element_type,
    properties,
    linear,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_MALFORMED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    """Prove the component/tile decomposition of one canonical MMA layout."""
    if kind not in {"amd_mfma", "dot_operand"}:
        return None

    def reject(message):
        _layout_fail(diagnostic, stage, message, source_op_index=source_op_index, source_value_id=source_value_id)

    out_dims = tuple((str(name), int(size)) for name, size in linear.out_dims)
    if tuple(name for name, _size in out_dims) != ("dim0", "dim1") or any(not _is_power_of_two(size)
                                                                          for _name, size in out_dims):
        reject("MFMA component model requires a binary rank-2 canonical LinearLayout")

    lane_count = linear_layout_in_dim_size(linear, "lane")
    payload_elements, _element_bits, _registers, fragment = _mma_fragment_abi(
        kind,
        element_type,
        properties,
        lane_count,
        stage=stage,
        diagnostic=diagnostic,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )
    shape = tuple(size for _name, size in out_dims)
    visible_fragment = tuple(min(logical, physical) for logical, physical in zip(shape, fragment))
    if payload_elements * lane_count % _product(visible_fragment):
        reject("MFMA payload and lane domains do not cover the visible fragment")

    register_count = linear_layout_in_dim_size(linear, "register")
    if register_count % payload_elements:
        reject("MFMA payload does not divide the canonical register domain")
    component_count = register_count // payload_elements
    tile_shape = tuple(_ceil_div(logical, physical) for logical, physical in zip(shape, fragment))
    if any(not _is_power_of_two(extent) for extent in tile_shape):
        reject("MFMA canonical layout has a non-binary fragment quotient")

    input_bases = {
        str(name): tuple(tuple(int(value)
                               for value in basis)
                         for basis in bases)
        for name, bases in linear.bases
    }
    input_names = tuple(str(name) for name in linear.get_in_dim_names())
    if set(input_names) - {"register", "lane", "warp", "block"}:
        reject("MFMA canonical layout has an unsupported physical input dimension")
    register_bases = input_bases.get("register", ())
    payload_bits = payload_elements.bit_length() - 1
    if len(register_bases) < payload_bits:
        reject("MFMA canonical register domain is smaller than one payload")

    def quotient_basis(basis, source):
        quotient = []
        for value, fragment_extent, tile_extent in zip(basis, fragment, tile_shape):
            if int(value) % int(fragment_extent):
                reject(f"MFMA {source} basis crosses fragment and tile coordinates")
            value = int(value) // int(fragment_extent)
            if value < 0 or value >= int(tile_extent):
                reject(f"MFMA {source} tile basis exceeds the fragment quotient")
            quotient.append(value)
        return tuple(quotient)

    component_bases = tuple(quotient_basis(basis, "component") for basis in register_bases[payload_bits:])
    partition_bases = {
        name: tuple(quotient_basis(basis, name)
                    for basis in input_bases.get(name, ()))
        for name in ("warp", "block")
    }

    def basis_masks(bases):
        masks = [0, 0]
        for basis in bases:
            for dim, value in enumerate(basis):
                masks[dim] |= value
        return tuple(masks)

    component_masks = basis_masks(component_bases)
    partition_masks = basis_masks(tuple(basis for bases in partition_bases.values() for basis in bases))

    for dim, extent in enumerate(tile_shape):
        if (component_masks[dim] & partition_masks[dim]
                or (component_masks[dim] | partition_masks[dim]) != int(extent) - 1):
            reject("MFMA tile layout is not a direct component and workgroup quotient")

    grid = tuple(1 << int(mask).bit_count() for mask in component_masks)
    if _product(grid) != component_count:
        reject("MFMA canonical component bits do not form a dense tile grid")

    # Invert the exact GF(2) component map in physical register order.
    positions = tuple((dim, bit)
                      for dim, mask in enumerate(component_masks)
                      for bit in range(int(mask).bit_length())
                      if int(mask) & (1 << bit))
    columns = tuple(
        sum(int(bool(basis[dim] & (1 << bit))) << row
            for row, (dim, bit) in enumerate(positions))
        for basis in component_bases)
    width = len(columns)
    if len(positions) != width:
        reject("MFMA component quotient is not square")
    rows = [
        sum(((column >> row) & 1) << bit
            for bit, column in enumerate(columns)) | (1 << (width + row))
        for row in range(width)
    ]
    for pivot in range(width):
        source = next((row for row in range(pivot, width) if rows[row] & (1 << pivot)), None)
        if source is None:
            reject("MFMA component quotient is not invertible")
        rows[pivot], rows[source] = rows[source], rows[pivot]
        for row in range(width):
            if row != pivot and rows[row] & (1 << pivot):
                rows[row] ^= rows[pivot]
    inverse_rows = tuple(row >> width for row in rows)

    dsl = load_wave_dsl()
    axes = tuple(dsl.sym(f"mma_axis{dim}") for dim in range(2))
    symbols = {name: dsl.sym(f"mma_{name}") for name in ("intra", "lane", "warp", "block")}

    def extract_bit(expression, bit):
        return dsl.mod(dsl.floor(expression / (1 << bit)), 2)

    axis_bits = tuple(
        extract_bit(axes[dim], compact_bit)
        for dim, mask in enumerate(component_masks)
        for compact_bit, _bit in enumerate(bit for bit in range(int(mask).bit_length()) if int(mask) & (1 << bit)))
    component = dsl.ixs_int(0)
    for physical_bit, inverse_mask in enumerate(inverse_rows):
        bit_value = dsl.ixs_int(0)
        for axis_bit, value in enumerate(axis_bits):
            if inverse_mask & (1 << axis_bit):
                bit_value = dsl.xor(bit_value, value)
        component += bit_value * (1 << physical_bit)
    register = component * payload_elements + symbols["intra"]
    extents = {
        "intra": payload_elements,
        "lane": lane_count,
        "warp": linear_layout_in_dim_size(linear, "warp"),
        "block": linear_layout_in_dim_size(linear, "block"),
    }
    facts = tuple(predicate for name, extent in extents.items() for predicate in _symbolic_range_predicates(
        dsl, symbols[name], 0,
        int(extent) - 1)) + tuple(predicate for dim, extent in enumerate(grid)
                                  for predicate in _symbolic_range_predicates(dsl, axes[dim], 0,
                                                                              int(extent) - 1))
    physical = {"register": register, **{name: symbols[name] for name in ("lane", "warp", "block")}}
    logical = _symbolic_layout_formula(dsl, linear, {name: physical[name] for name in input_names})

    def remap_bits(expression, source_bits, destination_bits):
        return sum(
            (extract_bit(expression, source_bit) * (1 << destination_bit)
             for source_bit, destination_bit in zip(source_bits, destination_bits)),
            dsl.ixs_int(0),
        )

    def xor_basis(coordinates, source, bases):
        for bit, basis in enumerate(bases):
            bit_value = extract_bit(source, bit)
            for dim, coefficient in enumerate(basis):
                if coefficient:
                    coordinates[dim] = dsl.xor(coordinates[dim], int(coefficient) * bit_value)

    fragment_coords = [dsl.ixs_int(0), dsl.ixs_int(0)]
    xor_basis(fragment_coords, symbols["intra"], register_bases[:payload_bits])
    xor_basis(fragment_coords, symbols["lane"], input_bases.get("lane", ()))
    partition_coords = [dsl.ixs_int(0), dsl.ixs_int(0)]
    for name in ("warp", "block"):
        xor_basis(partition_coords, symbols[name], partition_bases[name])

    goals = [
        ("component_lower", component >= 0),
        ("component_upper", component < component_count),
        ("register_lower", register >= 0),
        ("register_upper", register < register_count),
        ("component_roundtrip", dsl.ixs_eq(dsl.floor(register / payload_elements), component)),
        ("intra_roundtrip", dsl.ixs_eq(dsl.mod(register, payload_elements), symbols["intra"])),
    ]
    for dim, ((name, logical_extent), fragment_extent, tile_extent) in enumerate(zip(out_dims, fragment, tile_shape)):
        tile = dsl.floor(logical[name] / fragment_extent)
        fragment_coord = dsl.mod(logical[name], fragment_extent)
        component_bits = tuple(bit for bit in range(int(component_masks[dim]).bit_length())
                               if int(component_masks[dim]) & (1 << bit))
        deposited_component = remap_bits(axes[dim], range(len(component_bits)), component_bits)
        compact_component = remap_bits(tile, component_bits, range(len(component_bits)))
        merged_tile = deposited_component + partition_coords[dim]
        goals.extend((
            (f"{name}_component_order", dsl.ixs_eq(compact_component, axes[dim])),
            (f"{name}_fragment_model", dsl.ixs_eq(fragment_coord, fragment_coords[dim])),
            (f"{name}_tile_composition", dsl.ixs_eq(tile, merged_tile)),
            (f"{name}_fragment_lower", fragment_coord >= 0),
            (f"{name}_fragment_upper", fragment_coord < fragment_extent),
            (f"{name}_tile_lower", tile >= 0),
            (f"{name}_tile_upper", tile < tile_extent),
            (f"{name}_partition_lower", partition_coords[dim] >= 0),
            (f"{name}_partition_upper", partition_coords[dim] < tile_extent),
            (f"{name}_logical_lower", logical[name] >= 0),
            (f"{name}_logical_upper", logical[name] < logical_extent),
        ))
    proofs, normalized = dsl.ixs_check((*tuple(goal for _name, goal in goals), component), facts)
    goal_proofs = proofs[:len(goals)]
    if any(proof is not True for proof in goal_proofs):
        status = "False" if False in goal_proofs else "Unknown"
        failed = tuple(name for (name, _goal), proof in zip(goals, goal_proofs) if proof is not True)
        reject("MFMA canonical component proof returned "
               f"{status} for goals {failed}")
    return _MmaComponentModel(component_count, grid, dsl.ixs_serialize(normalized[len(goals)]))


def mma_tile_grid(
    result_layout,
    lhs_layout,
    rhs_layout,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_MALFORMED_LAYOUT",
    source_op_index=None,
):
    """Return the MFMA component tile grid proved by the three layouts."""

    def component_grid(layout):
        if layout.linear_layout is None:
            _layout_fail(
                diagnostic,
                stage,
                "MFMA lowering requires the canonical LinearLayout model",
                source_op_index=source_op_index,
                source_value_id=layout.value_id,
            )
        grid = layout.component_grid
        relation = layout.component_relation
        if (grid is None or len(grid) != 2 or any(type(extent) is not int or extent <= 0 for extent in grid)
                or not isinstance(relation, tuple) or not relation
                or any(type(byte) is not int or not 0 <= byte <= 255 for byte in relation)):
            _layout_fail(
                diagnostic,
                stage,
                "MFMA layout does not have a proved instruction-component relation",
                source_op_index=source_op_index,
                source_value_id=layout.value_id,
            )
        return tuple(int(extent) for extent in grid)

    result = component_grid(result_layout)
    lhs = component_grid(lhs_layout)
    rhs = component_grid(rhs_layout)
    dsl = load_wave_dsl()
    compatibility = (
        dsl.ixs_eq(dsl.ixs_int(lhs[0]), dsl.ixs_int(result[0])),
        dsl.ixs_eq(dsl.ixs_int(rhs[1]), dsl.ixs_int(result[1])),
        dsl.ixs_eq(dsl.ixs_int(rhs[0]), dsl.ixs_int(lhs[1])),
    )
    proofs, _normalized = dsl.ixs_check(compatibility, ())
    if any(proof is not True for proof in proofs):
        _layout_fail(
            diagnostic,
            stage,
            "MFMA operand and result LinearLayout component grids have incompatible "
            "tile grids",
            source_op_index=source_op_index,
        )
    return (
        result[0],
        result[1],
        lhs[1],
    )


def distributed_linear_layout(
    layout,
    *,
    stage=STAGE,
    source_op_index=None,
):
    if layout.linear_layout is not None:
        return layout.linear_layout
    _layout_fail(
        "TLXW_TYPE_UNSUPPORTED_LAYOUT",
        stage,
        f"layout {layout.kind} does not have a distributed register map",
        source_op_index=source_op_index,
        source_value_id=layout.value_id,
    )


def linear_layout_in_dim_size(linear, dim):
    for in_dim, bases in linear.bases:
        if in_dim == dim:
            return 1 << len(bases)
    return 1


def linear_layout_bases(linear, in_dim):
    for name, bases in linear.bases:
        if name == in_dim:
            return tuple(tuple(int(value) for value in basis) for basis in bases)
    return ()


def _distributed_active_relation(
    linear,
    *,
    lane_width,
    warp_count,
    block_count,
    source_value_id,
):
    """Return the zero-key predicate for one linear section of the layout."""
    if linear is None:
        return None

    def reject(code, message):
        _layout_fail(code, STAGE, message, source_value_id=source_value_id)

    lane_width = int(lane_width)
    warp_count = int(warp_count)
    block_count = int(block_count)
    if min(lane_width, warp_count, block_count) <= 0:
        reject("TLXW_TYPE_MALFORMED_LAYOUT", "distributed ownership requires positive kernel dimensions")

    out_dims = tuple((str(name), int(extent)) for name, extent in linear.out_dims)
    if any(not _is_power_of_two(extent) for _name, extent in out_dims):
        reject("TLXW_TYPE_UNSUPPORTED_LAYOUT", "replicated ownership requires a binary LinearLayout output domain")
    out_rank = len(out_dims)
    provided = {
        str(name): tuple(tuple(int(value)
                               for value in basis)
                         for basis in bases)
        for name, bases in linear.bases
    }
    unknown = tuple(sorted(set(provided) - {"register", "lane", "warp", "block"}))
    if unknown:
        reject("TLXW_TYPE_UNSUPPORTED_LAYOUT", f"distributed ownership has unsupported physical dimensions {unknown}")
    register_extent = 1 << len(provided.get("register", ()))
    physical_extents = {
        "register": register_extent,
        "lane": lane_width,
        "warp": warp_count,
        "block": block_count,
    }
    columns = []
    for name in ("register", "lane", "warp", "block"):
        extent = int(physical_extents[name])
        if not _is_power_of_two(extent):
            reject("TLXW_TYPE_UNSUPPORTED_LAYOUT", f"distributed ownership requires a binary {name} domain")
        width = extent.bit_length() - 1
        bases = provided.get(name, ())
        if len(bases) > width or any(len(basis) != out_rank for basis in bases):
            reject("TLXW_TYPE_MALFORMED_LAYOUT", f"LinearLayout {name} bases do not fit the kernel domain")
        for bit in range(width):
            basis = bases[bit] if bit < len(bases) else (0, ) * out_rank
            if any(value < 0 or value >= extent for value, (_out_name, extent) in zip(basis, out_dims)):
                reject("TLXW_TYPE_MALFORMED_LAYOUT", f"LinearLayout {name} basis exceeds its logical domain")
            shift = 0
            column = 0
            for value, (_out_name, extent) in zip(basis, out_dims):
                column |= int(value) << shift
                shift += int(extent).bit_length() - 1
            columns.append((name, bit, column))

    physical_size = _product(physical_extents.values())
    logical_size = _product(extent for _name, extent in out_dims)
    if physical_size < logical_size or physical_size % logical_size:
        reject("TLXW_TYPE_MALFORMED_LAYOUT", "LinearLayout physical domain does not cover its logical domain")
    replication = physical_size // logical_size
    if not _is_power_of_two(replication):
        reject("TLXW_TYPE_UNSUPPORTED_LAYOUT", "distributed ownership requires a binary replication domain")

    # Preserve the execution hierarchy in the canonical linear section.
    logical_width = sum(extent.bit_length() - 1 for _name, extent in out_dims)
    pivots = {}

    def add_independent(column):
        reduced = int(column)
        while reduced:
            pivot = reduced.bit_length() - 1
            known = pivots.get(pivot)
            if known is None:
                pivots[pivot] = reduced
                return True
            reduced ^= known
        return False

    block_columns = tuple(column for column in columns if column[0] == "block")
    if any(not add_independent(column) for _name, _bit, column in block_columns):
        reject("TLXW_TYPE_UNSUPPORTED_LAYOUT", "replicated block ownership cannot be suppressed inside one workgroup")
    nonpivots = []
    for physical_dim in ("lane", "warp", "register"):
        for name, bit, column in columns:
            if name == physical_dim and not add_independent(column):
                nonpivots.append((name, bit))
    if len(pivots) != logical_width or not linear.is_surjective():
        reject("TLXW_TYPE_UNSUPPORTED_LAYOUT", "distributed ownership requires a surjective binary LinearLayout")
    kernel_width = replication.bit_length() - 1
    if len(nonpivots) != kernel_width:
        reject("TLXW_TYPE_MALFORMED_LAYOUT", "LinearLayout section does not span the physical replication domain")
    if not nonpivots:
        return None

    dsl = load_wave_dsl()
    item = dsl.sym("item")
    slot = dsl.sym("slot")
    inputs = {
        "register": slot,
        "lane": dsl.mod(item, lane_width),
        "warp": dsl.floor(item / lane_width),
    }

    def extract_bit(expression, bit):
        return dsl.mod(dsl.floor(expression / (1 << bit)), 2)

    # Pack the non-pivot coordinates of the replication kernel directly.
    digits = tuple(extract_bit(inputs[name], bit) for name, bit in nonpivots)
    active_key = sum(
        (digit * (1 << bit) for bit, digit in enumerate(digits)),
        dsl.ixs_int(0),
    )
    facts = (
        *_symbolic_range_predicates(
            dsl,
            item,
            0,
            lane_width * warp_count - 1,
        ),
        *_symbolic_range_predicates(
            dsl,
            slot,
            0,
            register_extent - 1,
        ),
    )
    goals = (
        dsl.ixs_eq(active_key, dsl.floor(active_key)),
        active_key >= 0,
        active_key <= replication - 1,
    )
    proofs, normalized = dsl.ixs_check((*goals, active_key), facts)
    if any(proof is not True for proof in proofs[:len(goals)]):
        status = "False" if False in proofs[:len(goals)] else "Unknown"
        reject("TLXW_TYPE_UNSUPPORTED_LAYOUT", f"distributed ownership proof returned {status}")
    owner_proofs, _ = dsl.ixs_check(
        tuple(dsl.ixs_eq(digit, 0) for digit in digits),
        (*facts, dsl.ixs_eq(active_key, 0)),
    )
    if any(proof is not True for proof in owner_proofs):
        reject("TLXW_TYPE_UNSUPPORTED_LAYOUT", "ownership key is not injective")
    return dsl.ixs_serialize(normalized[len(goals)])


def packet_layout_relations(
        sources,
        results,
        *,
        transform="identity",
        axis=None,
        order=(),
):
    sources = tuple(sources)
    results = tuple(results)
    expected = ((2, 1) if transform == "join" else ((1, 2) if transform == "split" else (1, 1)))
    if (len(sources), len(results)) != expected:
        raise ValueError(f"{transform} packet relation requires {expected[0]} source layout(s) "
                         f"and {expected[1]} result layout(s)")
    if any(layout.linear_layout is None for layout in (*sources, *results)):
        raise ValueError("packet redistribution requires distributed linear layouts")
    lane_widths = {int(layout.lane_width) for layout in (*sources, *results)}
    if len(lane_widths) != 1:
        raise ValueError("packet redistribution requires equal source/result lane widths")

    if transform == "join":
        first, second = sources
        result = results[0]
        if first.linear_layout != second.linear_layout:
            raise ValueError("packet join requires identical distributed operand layouts")
        relation = _packet_relation_blob(
            _joined_packet_layout(first),
            result.linear_layout,
            tuple(int(dim) for dim in result.shape),
            tuple(int(dim) for dim in result.shape),
            lane_width=int(result.lane_width),
            source_components=int(first.component_count) + int(second.component_count),
            destination_components=int(result.component_count),
        )
        return (relation, )

    source = sources[0]
    if transform == "split" and axis is None:
        axis = len(source.shape) - 1
    return tuple(
        _packet_relation_blob(
            source.linear_layout,
            result.linear_layout,
            tuple(int(dim) for dim in source.shape),
            tuple(int(dim) for dim in result.shape),
            lane_width=int(result.lane_width),
            source_components=int(source.component_count),
            destination_components=int(result.component_count),
            transform=transform,
            axis=axis,
            order=order,
            selector=selector if transform == "split" else None,
        ) for selector, result in enumerate(results))


def _symbolic_range_predicates(dsl, expression, lower, upper):
    """Describe an integer-valued symbolic coordinate over a closed range."""
    return (
        dsl.ixs_eq(expression, dsl.floor(expression)),
        expression >= int(lower),
        expression <= int(upper),
    )


def make_range_relation(
    layout,
    component_count,
    lane_width,
    warp_count,
    start,
    end,
    *,
    source_op_index=None,
    source_value_id=None,
):
    """Map the full physical packet domain to a distributed range value."""
    linear = _complete_packet_physical_dims(
        distributed_linear_layout(
            layout,
            stage=STAGE,
            source_op_index=source_op_index,
        ))
    shape = tuple(int(extent) for extent in layout.shape)
    if not shape or _product(shape) != int(end) - int(start):
        _layout_fail(
            "TLXW_TYPE_MALFORMED_LAYOUT",
            STAGE,
            "make_range layout shape does not match its value interval",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    register_count = linear_layout_in_dim_size(linear, "register")
    if int(component_count) != register_count:
        _layout_fail(
            "TLXW_TYPE_UNSUPPORTED_LAYOUT",
            STAGE,
            "scalar make_range components must match its distributed register domain",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    dsl = load_wave_dsl()
    item = dsl.sym("item")
    block = dsl.sym("block")
    slot = dsl.sym("slot")
    inputs = {
        "block": block,
        "item": item,
        "lane": dsl.mod(item, int(lane_width)),
        "warp": dsl.floor(item / int(lane_width)),
        "register": slot,
    }
    goals = []
    coords = _symbolic_layout_formula(dsl, linear, inputs)
    flat = dsl.ixs_int(0)
    for dim, extent in enumerate(shape):
        coord = coords[f"dim{dim}"]
        goals.extend((coord >= 0, coord < extent))
        flat = flat * extent + coord
    value = int(start) + flat
    goals.extend((value >= int(start), value < int(end)))
    facts = (
        *_symbolic_range_predicates(
            dsl,
            item,
            0,
            int(lane_width) * int(warp_count) - 1,
        ),
        *_symbolic_range_predicates(
            dsl,
            block,
            0,
            linear_layout_in_dim_size(linear, "block") - 1,
        ),
        *_symbolic_range_predicates(
            dsl,
            slot,
            0,
            register_count - 1,
        ),
    )
    checked, normalized = dsl.ixs_check((*goals, value), facts)
    goal_proofs = checked[:len(goals)]
    if any(proof is not True for proof in goal_proofs):
        status = "False" if False in goal_proofs else "Unknown"
        raise ValueError(f"symbolic make_range layout proof returned {status}")
    return dsl.ixs_serialize(normalized[len(goals)])


def global_memory_bit_offset_relation(
    layout,
    index_expr,
    *,
    element_byte_width,
    element_contiguity=1,
    wrap_i32=True,
):
    """Compose one logical index expression with its final packet layout."""
    if layout is None or layout.linear_layout is None:
        raise ValueError("global memory relation requires a distributed linear layout")
    dsl = load_wave_dsl()
    linear = _complete_packet_physical_dims(layout.linear_layout)
    linear = _packet_item_linear_layout(
        linear,
        int(layout.lane_width),
        layout_warp_count(layout),
        preserve_block=True,
    )
    physical_inputs = {
        "block": dsl.sym("block"),
        "slot": dsl.sym("slot"),
        "item": dsl.sym("item"),
    }
    logical = _symbolic_layout_field_formula(
        dsl,
        linear,
        physical_inputs,
        combine_with_xor=False,
    )
    if logical is None:
        logical = _symbolic_layout_formula(dsl, linear, physical_inputs)
    element_offset = index_expr.subs({dsl.sym(name): expression for name, expression in logical.items()})
    element_contiguity = int(element_contiguity)
    if element_contiguity <= 0:
        raise ValueError("global memory relation requires positive element contiguity")
    if element_contiguity > 1:
        slot = dsl.sym("slot")
        group_slot = element_contiguity * dsl.floor(slot / element_contiguity)
        group_origin = element_offset.subs({slot: group_slot})
        element_offset = group_origin + dsl.mod(slot, element_contiguity)
    byte_offset = int(element_byte_width) * element_offset
    if wrap_i32:
        byte_offset = dsl.mod(byte_offset, 1 << 32)
    bit_offset = 8 * byte_offset
    return tuple(int(byte) for byte in dsl.ixs_serialize(bit_offset))


def local_memory_bit_offset_relation(
    distributed_layout,
    shared_layout,
    logical_shape,
    physical_shape,
    logical_origin,
    *,
    lane_width,
    warp_count,
    element_byte_width,
    allocation_bytes,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    """Compose a distributed placement and shared layout into bit offsets."""
    if distributed_layout is None or distributed_layout.linear_layout is None:
        raise ValueError("local memory relation requires a distributed linear layout")
    logical_shape = tuple(int(value) for value in logical_shape)
    physical_shape = tuple(int(value) for value in physical_shape)
    logical_origin = tuple(int(value) for value in logical_origin)
    lane_width = int(lane_width)
    warp_count = int(warp_count)
    element_byte_width = int(element_byte_width)
    allocation_bytes = int(allocation_bytes)
    if (not logical_shape or len(logical_shape) != len(physical_shape) or len(logical_shape) != len(logical_origin)):
        raise ValueError("local memory relation requires matching ranked shapes and origin")
    if any(value <= 0 for value in (*logical_shape, *physical_shape)) or any(value < 0 for value in logical_origin):
        raise ValueError("local memory relation requires positive shapes and nonnegative origin")
    if (lane_width <= 0 or warp_count <= 0 or element_byte_width <= 0 or allocation_bytes <= 0):
        raise ValueError("local memory relation requires positive physical extents")

    linear = _complete_packet_physical_dims(distributed_layout.linear_layout)
    register_count = linear_layout_in_dim_size(linear, "register")
    if linear_layout_in_dim_size(linear, "lane") > lane_width:
        raise ValueError("local memory relation lane domain exceeds the hardware wave")
    if linear_layout_in_dim_size(linear, "warp") > warp_count:
        raise ValueError("local memory relation warp domain exceeds the workgroup")
    block_bases = linear_layout_bases(linear, "block")
    if any(any(int(value) for value in basis) for basis in block_bases):
        raise ValueError("local memory relation does not support nontrivial block bases")
    expected_outputs = {f"dim{dim}": int(extent) for dim, extent in enumerate(logical_shape)}
    if {str(name): int(extent) for name, extent in linear.out_dims} != expected_outputs:
        raise ValueError("local memory relation logical dimensions do not match its shape")

    address_layout, order, intervals, paddings = _shared_address_layout(
        shared_layout,
        physical_shape,
        stage=stage,
        diagnostic=diagnostic,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )
    dsl = load_wave_dsl()
    item = dsl.sym("item")
    slot = dsl.sym("slot")
    goals = []
    element_bits = element_byte_width * 8
    allocation_bits = allocation_bytes * 8
    item_linear = _packet_item_bit_linear_layout(
        linear,
        lane_width,
        warp_count,
    )
    lane_bits = lane_width.bit_length() - 1

    def item_bit(bit):
        if bit < lane_bits:
            return dsl.floor(dsl.mod(item, 1 << (bit + 1)) / (1 << bit))
        warp_bit = bit - lane_bits
        warp = dsl.floor(item / lane_width)
        return dsl.floor(dsl.mod(warp, 1 << (warp_bit + 1)) / (1 << warp_bit))

    physical_inputs = {
        "slot": slot,
        **{f"item{bit}": item_bit(bit)
           for bit in range((lane_width * warp_count).bit_length() - 1)},
    }
    logical_by_name = _symbolic_layout_formula(
        dsl,
        item_linear,
        physical_inputs,
    )
    logical_coords = tuple(logical_by_name[f"dim{dim}"] for dim in range(len(logical_shape)))
    for coord, extent in zip(logical_coords, logical_shape):
        goals.extend((coord >= 0, coord < extent))
    physical_coords = tuple(coord + origin for coord, origin in zip(logical_coords, logical_origin))
    for coord, extent in zip(physical_coords, physical_shape):
        goals.extend((coord >= 0, coord < extent))
    if (address_layout is not None and not address_layout.prefix_shape and logical_shape == physical_shape
            and not any(logical_origin) and all(_is_power_of_two(interval) for interval in intervals)):
        physical_layout = _compose_linear_layouts(
            linear,
            address_layout.linear_layout,
        )
        layout_inputs = {
            "register": slot,
            "lane": dsl.mod(item, lane_width),
            "warp": dsl.floor(item / lane_width),
            "block": dsl.ixs_int(0),
        }
        mapped = None
        if not any(paddings):
            mapped = _symbolic_layout_field_formula(
                dsl,
                physical_layout,
                layout_inputs,
                combine_with_xor=False,
            )
        physical_bits = (None if mapped is not None else _symbolic_layout_bits(dsl, physical_layout, layout_inputs))
        goals.append(
            dsl.ixs_eq(
                mapped["block"] if mapped is not None else _symbolic_bits_to_int(dsl, physical_bits["block"]),
                dsl.ixs_int(0),
            ))
        if mapped is not None:
            element_offset = mapped["offset"]
        else:
            offset_bits = physical_bits["offset"]
            element_offset = _symbolic_bits_to_int(dsl, offset_bits)
            for interval, padding in zip(intervals, paddings):
                first_padding_bit = int(interval).bit_length() - 1
                element_offset += int(padding) * _symbolic_bits_to_int(
                    dsl,
                    offset_bits[first_padding_bit:],
                )
    else:
        element_offset = _symbolic_shared_element_offset(
            dsl,
            address_layout,
            physical_shape,
            physical_coords,
            goals,
            order=order,
            intervals=intervals,
            paddings=paddings,
        )
    bit_offset = element_bits * element_offset
    goals.extend((
        bit_offset >= 0,
        bit_offset + element_bits <= allocation_bits,
    ))
    facts = (
        *_symbolic_range_predicates(
            dsl,
            item,
            0,
            lane_width * warp_count - 1,
        ),
        *_symbolic_range_predicates(
            dsl,
            slot,
            0,
            register_count - 1,
        ),
    )
    checked, _normalized = dsl.ixs_check(tuple(goals), facts)
    goal_proofs = checked
    if any(proof is not True for proof in goal_proofs):
        status = "False" if False in goal_proofs else "Unknown"
        raise ValueError(f"symbolic local memory address proof returned {status}")
    return dsl.ixs_serialize(bit_offset)


def memdesc_index_element_offset_relation(
    parent_layout,
    child_layout,
    parent_shape,
    child_shape,
    *,
    element_byte_width,
    allocation_bytes,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    """Return the exact physical element offset of a memdesc slot.

    A ``ttg.memdesc_index`` drops the leading logical slot dimensions.  Its
    pointer is therefore the shared-layout image of the row-major logical
    element ``slot * product(child_shape)``.  The same address maps are used
    for the parent base and for accesses through the resulting child view;
    proving their translation identity here makes that pointer view exact.
    """
    parent_shape = tuple(int(extent) for extent in parent_shape)
    child_shape = tuple(int(extent) for extent in child_shape)
    element_byte_width = int(element_byte_width)
    allocation_bytes = int(allocation_bytes)
    if (not parent_shape or not child_shape or any(extent <= 0 for extent in (*parent_shape, *child_shape))):
        raise ValueError("memdesc index relation requires positive ranked shapes")
    if element_byte_width <= 0 or allocation_bytes <= 0:
        raise ValueError("memdesc index relation requires positive physical extents")
    parent_elements = _product(parent_shape)
    child_elements = _product(child_shape)
    if parent_elements % child_elements:
        raise ValueError("memdesc index child shape does not evenly tile its parent shape")
    slot_count = parent_elements // child_elements
    if slot_count <= 0:
        raise ValueError("memdesc index relation requires a nonempty slot domain")

    parent_address, parent_order, parent_intervals, parent_paddings = (_shared_address_layout(
        parent_layout,
        parent_shape,
        stage=stage,
        diagnostic=diagnostic,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    ))
    child_address, child_order, child_intervals, child_paddings = (_shared_address_layout(
        child_layout,
        child_shape,
        stage=stage,
        diagnostic=diagnostic,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    ))
    dsl = load_wave_dsl()
    slot = dsl.sym("slot")
    child_linear = dsl.sym("child")

    def row_major_coords(linear, shape, goals):
        coords = []
        for dim, extent in enumerate(shape):
            stride = _product(shape[dim + 1:])
            coord = dsl.mod(dsl.floor(linear / stride), int(extent))
            goals.extend((coord >= 0, coord < int(extent)))
            coords.append(coord)
        return tuple(coords)

    base_goals = []
    base_linear = slot * child_elements
    base_coords = row_major_coords(base_linear, parent_shape, base_goals)
    base_offset = _symbolic_shared_element_offset(
        dsl,
        parent_address,
        parent_shape,
        base_coords,
        base_goals,
        order=parent_order,
        intervals=parent_intervals,
        paddings=parent_paddings,
    )
    base_byte_offset = base_offset * element_byte_width
    base_goals.extend((
        dsl.ixs_eq(base_offset, dsl.floor(base_offset)),
        dsl.ixs_eq(
            dsl.mod(base_byte_offset, element_byte_width),
            dsl.ixs_int(0),
        ),
        base_byte_offset >= 0,
        base_byte_offset + element_byte_width <= allocation_bytes,
    ))
    slot_facts = _symbolic_range_predicates(
        dsl,
        slot,
        0,
        slot_count - 1,
    )
    checked, normalized = dsl.ixs_check(
        (*base_goals, base_offset),
        slot_facts,
    )
    if any(proof is not True for proof in checked[:len(base_goals)]):
        proofs = checked[:len(base_goals)]
        status = "False" if False in proofs else "Unknown"
        raise ValueError(f"symbolic memdesc index base proof returned {status}")

    # A child memdesc remains a normal shared-layout view only when the parent
    # map restricted to each logical slot is a translated copy of the child's
    # map.  Establish that identity over the complete child domain instead of
    # inferring it from sampled slot offsets.
    translation_goals = []
    parent_linear = base_linear + child_linear
    parent_coords = row_major_coords(
        parent_linear,
        parent_shape,
        translation_goals,
    )
    parent_offset = _symbolic_shared_element_offset(
        dsl,
        parent_address,
        parent_shape,
        parent_coords,
        translation_goals,
        order=parent_order,
        intervals=parent_intervals,
        paddings=parent_paddings,
    )
    child_coords = row_major_coords(
        child_linear,
        child_shape,
        translation_goals,
    )
    child_offset = _symbolic_shared_element_offset(
        dsl,
        child_address,
        child_shape,
        child_coords,
        translation_goals,
        order=child_order,
        intervals=child_intervals,
        paddings=child_paddings,
    )
    translation_goals.extend((
        dsl.ixs_eq(parent_offset, base_offset + child_offset),
        parent_offset * element_byte_width + element_byte_width <= allocation_bytes,
    ))
    translation_facts = (
        *slot_facts,
        *_symbolic_range_predicates(
            dsl,
            child_linear,
            0,
            child_elements - 1,
        ),
    )
    proofs, _ = dsl.ixs_check(tuple(translation_goals), translation_facts)
    if any(proof is not True for proof in proofs):
        status = "False" if False in proofs else "Unknown"
        raise ValueError("symbolic memdesc index child-translation proof returned "
                         f"{status}")
    return (
        dsl.ixs_serialize(normalized[len(base_goals)]),
        int(slot_count),
    )


def shared_allocation_size_bytes(
    layout,
    shape,
    element_byte_width,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    """Return the allocation size required by one shared address layout."""
    shape = tuple(int(extent) for extent in shape)
    element_byte_width = int(element_byte_width)
    if not shape or any(extent <= 0 for extent in shape) or element_byte_width <= 0:
        raise ValueError("shared allocation requires positive shape and element size")
    _address, _order, intervals, paddings = _shared_address_layout(
        layout,
        shape,
        stage=stage,
        diagnostic=diagnostic,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )
    logical_extent = _product(shape)
    padding_extent = sum((logical_extent // interval - int(logical_extent % interval == 0)) * padding
                         for interval, padding in zip(intervals, paddings))
    return (logical_extent + padding_extent) * element_byte_width


def _packet_relation_blob(
        source,
        result,
        source_shape,
        result_shape,
        *,
        lane_width,
        source_components,
        destination_components,
        transform="identity",
        axis=None,
        order=(),
        selector=None,
):
    if source is None or result is None:
        raise ValueError("packet redistribution requires distributed linear layouts")
    source = _complete_packet_physical_dims(source)
    result = _complete_packet_physical_dims(result)
    if not source.is_surjective():
        raise ValueError("packet redistribution requires a surjective source layout")
    source_extents = {name: linear_layout_in_dim_size(source, name) for name in ("register", "lane", "warp", "block")}
    result_extents = {name: linear_layout_in_dim_size(result, name) for name in ("register", "lane", "warp", "block")}
    if source_extents["lane"] != result_extents["lane"]:
        raise ValueError("packet redistribution requires equal source/result lane domains")
    if source_extents["block"] != result_extents["block"]:
        raise ValueError("packet redistribution requires equal source/result block domains")
    mapped_dims, structural = _packet_transform_descriptor(
        source_shape,
        result_shape,
        transform=transform,
        axis=axis,
        order=order,
    )
    dsl = load_wave_dsl()
    item = dsl.sym("item")
    packet = {
        "block": dsl.sym("block"),
        "item": item,
        "lane": dsl.mod(item, int(lane_width)),
        "warp": dsl.floor(item / int(lane_width)),
        "register": dsl.sym("slot"),
    }
    if structural:
        name, _extent, _source_dim = structural
        packet[name] = dsl.sym(name) if selector is None else dsl.ixs_int(int(selector))
    source_dims = tuple((str(name), int(size)) for name, size in source.out_dims)
    result_dims = tuple((str(name), int(size)) for name, size in result.out_dims)
    if (int(destination_components) <= 0 or result_extents["register"] % int(destination_components)):
        raise ValueError("packet component count does not split the register domain")
    source_slots = linear_layout_in_dim_size(source, "register")
    source_warps = linear_layout_in_dim_size(source, "warp")
    source_items = int(lane_width) * source_warps
    result_physical = {
        "register": dsl.mod(packet["register"], result_extents["register"]),
        "lane": dsl.mod(packet["lane"], result_extents["lane"]),
        "warp": dsl.mod(packet["warp"], result_extents["warp"]),
        "block": dsl.mod(packet["block"], result_extents["block"]),
    }
    result_logical = _symbolic_layout_formula(dsl, result, result_physical)
    expected_logical = _transform_logical_coordinates(
        dsl,
        result_logical,
        source_dims,
        result_dims,
        mapped_dims,
        structural,
        packet,
        reshape=transform == "reshape",
    )
    if transform == "reshape":
        # A reshape through non-power-of-two dimensions is not a GF(2) linear
        # map.  Keep its exact integer coordinate composition.
        inverse = source.pseudoinvert()
        mapped = _symbolic_layout_formula(dsl, inverse, expected_logical)
    else:
        destination = _packet_destination_layout(
            result,
            source_dims,
            result_dims,
            mapped_dims,
            structural,
            reshape=False,
        )
        same_layout = (transform == "identity" and not structural and tuple(source.bases) == tuple(result.bases)
                       and tuple(source.out_dims) == tuple(result.out_dims))
        if same_layout:
            mapped = dict(result_physical)
        elif all(_is_power_of_two(extent) for _name, extent in source_dims):
            relation = _preferred_packet_relation(source, destination)
            mapped = _symbolic_layout_formula(dsl, relation, packet)
        elif transform == "identity" and not structural:
            raise ValueError("packet redistribution between distinct non-power-of-two layouts "
                             "is not representable by the GF(2) layout relation")
        else:
            inverse = source.pseudoinvert()
            mapped = _symbolic_layout_formula(dsl, inverse, expected_logical)
    mapped_item = mapped["warp"] * int(lane_width) + mapped["lane"]
    packed = ((mapped["block"] * source_items + mapped_item) * source_slots + mapped["register"])
    return dsl.ixs_serialize(packed)


def _symbolic_layout_formula(
    dsl,
    linear,
    inputs,
    component_count=None,
):
    out_dims = tuple((str(name), int(size)) for name, size in linear.out_dims)
    if component_count is not None:
        register_count = linear_layout_in_dim_size(linear, "register")
        if component_count <= 0 or register_count % component_count:
            raise ValueError("packet component count does not split the register domain")
    coordinates = [dsl.ixs_int(0) for _name, _size in out_dims]
    for name, values in linear.bases:
        source = inputs[str(name)]
        for bit, basis in enumerate(values):
            bit_value = dsl.mod(dsl.floor(source / (1 << bit)), 2)
            for dim, coefficient in enumerate(basis):
                if not coefficient:
                    continue
                if _is_power_of_two(out_dims[dim][1]):
                    coordinates[dim] = dsl.xor(coordinates[dim], int(coefficient) * bit_value)
                else:
                    coordinates[dim] += int(coefficient) * bit_value
    return {
        name: coordinate if _is_power_of_two(size) else dsl.mod(coordinate, size)
        for (name, size), coordinate in zip(out_dims, coordinates)
    }


def _symbolic_layout_field_formula(
    dsl,
    linear,
    inputs,
    *,
    output_weights=None,
    combine_with_xor=True,
):
    """Compose a bit-permutation layout as maximal integer input fields."""
    out_dims = tuple((str(name), int(size)) for name, size in linear.out_dims)
    if any(not _is_power_of_two(size) for _name, size in out_dims):
        return None
    output_bits = [size.bit_length() - 1 for _name, size in out_dims]
    if output_weights is None:
        output_weights = {
            name: tuple(1 << bit
                        for bit in range(bit_count))
            for (name, _size), bit_count in zip(out_dims, output_bits)
        }
    weights = [tuple(map(int, output_weights[name])) for name, _size in out_dims]
    if any(len(values) != bit_count for values, bit_count in zip(weights, output_bits)):
        return None
    occupied = [set() for _ in out_dims]
    fields_by_output = [[] for _ in out_dims]
    for name, bases in linear.bases:
        source = inputs.get(str(name))
        if source is None:
            return None
        fields = []
        current = None
        for input_bit, basis in enumerate(bases):
            nonzero = [(output, int(coefficient)) for output, coefficient in enumerate(basis) if int(coefficient)]
            if not nonzero:
                current = None
                continue
            if len(nonzero) != 1:
                return None
            output, coefficient = nonzero[0]
            if coefficient <= 0 or coefficient & (coefficient - 1):
                return None
            output_bit = coefficient.bit_length() - 1
            if (output_bit >= output_bits[output] or output_bit in occupied[output]):
                return None
            occupied[output].add(output_bit)
            if (current is not None and current[0] == output and current[1] + current[3] == input_bit
                    and current[2] + current[3] == output_bit
                    and weights[output][output_bit] == 2 * weights[output][output_bit - 1]):
                current[3] += 1
            else:
                current = [output, input_bit, output_bit, 1]
                fields.append(current)
        for output, input_bit, output_bit, width in fields:
            value = source if input_bit == 0 and width == len(bases) else dsl.mod(
                dsl.floor(source / (1 << input_bit)),
                1 << width,
            )
            fields_by_output[output].append(weights[output][output_bit] * value)
    coordinates = []
    for fields in fields_by_output:
        coordinate = dsl.ixs_int(0)
        for field in fields:
            coordinate = (dsl.xor(coordinate, field) if combine_with_xor else coordinate + field)
        coordinates.append(coordinate)
    return {name: coordinate for (name, _size), coordinate in zip(out_dims, coordinates)}


def _symbolic_layout_bits(dsl, linear, inputs):
    """Compose each output bit of a GF(2) linear layout."""
    out_dims = tuple((str(name), int(size)) for name, size in linear.out_dims)
    if any(not _is_power_of_two(size) for _name, size in out_dims):
        raise ValueError("bitwise layout formula requires power-of-two output domains")
    bits = [[dsl.ixs_int(0) for _ in range(size.bit_length() - 1)] for _name, size in out_dims]
    for name, bases in linear.bases:
        source = inputs[str(name)]
        for input_bit, basis in enumerate(bases):
            value = dsl.mod(dsl.floor(source / (1 << input_bit)), 2)
            for output, coefficient in enumerate(basis):
                for output_bit in range(len(bits[output])):
                    if int(coefficient) & (1 << output_bit):
                        bits[output][output_bit] = dsl.xor(
                            bits[output][output_bit],
                            value,
                        )
    return {name: tuple(values) for (name, _size), values in zip(out_dims, bits)}


def _symbolic_bits_to_int(dsl, bits):
    result = dsl.ixs_int(0)
    for bit, value in enumerate(bits):
        result = dsl.xor(result, (1 << bit) * value)
    return result


def _symbolic_layout_bit_formula(dsl, linear, inputs):
    """Serialize each GF(2) output bit before forming its integer value."""
    return {name: _symbolic_bits_to_int(dsl, bits) for name, bits in _symbolic_layout_bits(dsl, linear, inputs).items()}


def _symbolic_shared_element_offset(
        dsl,
        address_layout,
        shape,
        coords,
        goals,
        *,
        order=(),
        intervals=(),
        paddings=(),
):
    if address_layout is None:
        offset = dsl.ixs_int(0)
        stride = 1
        for dim in order:
            offset += coords[int(dim)] * stride
            stride *= int(shape[int(dim)])
    else:
        inner_rank = len(address_layout.inner_shape)
        prefix_coords = tuple(coords[:-inner_rank]) if inner_rank else tuple(coords)
        inner_coords = tuple(coords[-inner_rank:]) if inner_rank else ()
        mapped = _symbolic_layout_formula(
            dsl,
            address_layout.linear_layout,
            {f"dim{dim}": coord
             for dim, coord in enumerate(inner_coords)},
        )
        outer = dsl.ixs_int(0)
        for coord, extent in zip(prefix_coords, address_layout.prefix_shape):
            outer = outer * int(extent) + coord
        offset = outer * _product(address_layout.inner_shape) + mapped["offset"]
        goals.append(dsl.ixs_eq(mapped["block"], dsl.ixs_int(0)))
    goals.extend((offset >= 0, offset < _product(shape)))
    encoded = offset
    for interval, padding in zip(intervals, paddings):
        encoded += dsl.floor(offset / int(interval)) * int(padding)
    return encoded


def _shared_address_layout(
    layout,
    shape,
    *,
    stage,
    diagnostic,
    source_op_index,
    source_value_id,
):
    """Return the logical-coordinate to unpadded physical-offset map."""
    shape = tuple(int(extent) for extent in shape)

    def reject(message):
        _shared_layout_fail(
            diagnostic,
            stage,
            message,
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )

    kind = "dense" if layout is None or layout.kind == "none" else str(layout.kind)
    order = tuple(reversed(range(len(shape))))
    if layout is not None and kind in {
            "padded_shared",
            "shared_linear",
            "swizzled_shared",
    }:
        inner_order = tuple(int(dim) for dim in layout.properties.get("order", ()))
        if inner_order:
            if len(inner_order) > len(shape) or sorted(inner_order) != list(range(len(inner_order))):
                reject(f"shared layout order {inner_order} cannot be applied to "
                       f"rank-{len(shape)} shape")
            prefix_rank = len(shape) - len(inner_order)
            order = tuple(prefix_rank + dim for dim in inner_order) + tuple(reversed(range(prefix_rank)))
    if kind == "dense":
        return None, tuple(order), (), ()
    if kind not in {"padded_shared", "shared_linear", "swizzled_shared"}:
        reject(f"shared address layout does not support {kind}")

    forward = layout.linear_layout
    if forward is None:
        reject(f"{kind} is missing its canonical LinearLayout")
    in_dims = tuple(str(name) for name in forward.get_in_dim_names())
    if in_dims != ("offset", "block"):
        reject(f"shared address map must use [offset, block] input dims; got {in_dims}")
    out_dims = tuple((str(name), int(extent)) for name, extent in forward.out_dims)
    component_rank = len(out_dims)
    if component_rank == 0 or component_rank > len(shape):
        reject("shared address map rank does not fit the memdesc shape; "
               f"map={out_dims}, shape={shape}")
    inner_shape = shape[-component_rank:]
    prefix_shape = shape[:-component_rank]
    expected_out_dims = tuple((f"dim{dim}", int(extent)) for dim, extent in enumerate(inner_shape))
    if out_dims != expected_out_dims:
        reject("shared address map output dims do not match the trailing "
               f"memdesc shape; got {out_dims}, expected {expected_out_dims}")
    if not forward.is_invertible():
        reject("shared address layout must be bijective")
    inverse = forward.invert()
    outputs = {str(name): int(extent) for name, extent in inverse.out_dims}
    if outputs != {"offset": _product(inner_shape), "block": 1}:
        reject(f"shared address layout has invalid physical domain {outputs}")
    if kind == "padded_shared":
        intervals = tuple(int(value) for value in layout.properties.get("intervals", ()))
        paddings = tuple(int(value) for value in layout.properties.get("paddings", ()))
        description = f"order={inner_order}, intervals={intervals}, paddings={paddings}"
        if len(intervals) != len(paddings):
            reject(f"padded shared interval/padding counts differ; {description}")
        if any(interval <= 0 for interval in intervals):
            reject(f"padded shared intervals must be positive; {description}")
        if any(padding < 0 for padding in paddings):
            reject(f"padded shared paddings must be nonnegative; {description}")
    else:
        intervals = paddings = ()
    return (
        _SharedAddressMap(
            inverse,
            tuple(prefix_shape),
            tuple(inner_shape),
        ),
        tuple(order),
        tuple(intervals),
        tuple(paddings),
    )


def _complete_packet_physical_dims(linear):
    bases = {str(name): values for name, values in linear.bases}
    unknown = tuple(sorted(set(bases) - {"register", "lane", "warp", "block"}))
    if unknown:
        raise ValueError("packet layout has unbound physical input dimensions "
                         f"{unknown}; compose helper dimensions into the packet domain first")
    if all(name in bases for name in ("register", "lane", "warp", "block")):
        return linear
    names = tuple(bases) + tuple(name for name in ("register", "lane", "warp", "block") if name not in bases)
    return LinearLayout.from_bases(
        tuple((name, bases.get(name, ())) for name in names),
        [str(name) for name, _size in linear.out_dims],
        [int(size) for _name, size in linear.out_dims],
        False,
    )


def _packet_item_linear_layout(
    linear,
    lane_width,
    warp_count,
    *,
    preserve_block=False,
):
    """Rebase lane and warp packet inputs onto the physical item index."""
    lane_width = int(lane_width)
    warp_count = int(warp_count)
    physical_extents = {name: linear_layout_in_dim_size(linear, name) for name in _PACKET_PHYSICAL_DIMS}
    lane_bits = lane_width.bit_length() - 1
    warp_bits = warp_count.bit_length() - 1
    if 1 << lane_bits != lane_width or 1 << warp_bits != warp_count:
        raise ValueError("packet item layout requires power-of-two hardware extents")
    register_bits = physical_extents["register"].bit_length() - 1
    physical_names = list(_PACKET_PHYSICAL_DIMS)

    def basis(**values):
        return [int(values.get(name, 0)) for name in physical_names]

    slot_bases = [basis(register=1 << bit) for bit in range(register_bits)]
    item_bases = [basis(lane=(1 << bit) if (1 << bit) < physical_extents["lane"] else 0) for bit in range(lane_bits)]
    item_bases.extend(
        basis(warp=(1 << bit) if (1 << bit) < physical_extents["warp"] else 0) for bit in range(warp_bits))
    adapter_bases = [
        ("slot", slot_bases),
        ("item", item_bases),
    ]
    if preserve_block:
        adapter_bases.append((
            "block",
            [basis(block=1 << bit) for bit in range(physical_extents["block"].bit_length() - 1)],
        ))
    adapter = LinearLayout.from_bases(
        adapter_bases,
        physical_names,
        [physical_extents[name] for name in physical_names],
        False,
    )
    return _compose_linear_layouts(adapter, linear)


def _packet_item_bit_linear_layout(linear, lane_width, warp_count):
    """Rebase packet ownership onto individual bits of the physical item."""
    lane_width = int(lane_width)
    warp_count = int(warp_count)
    physical_extents = {name: linear_layout_in_dim_size(linear, name) for name in _PACKET_PHYSICAL_DIMS}
    lane_bits = lane_width.bit_length() - 1
    warp_bits = warp_count.bit_length() - 1
    if 1 << lane_bits != lane_width or 1 << warp_bits != warp_count:
        raise ValueError("packet item layout requires power-of-two hardware extents")
    physical_names = list(_PACKET_PHYSICAL_DIMS)

    def basis(**values):
        return [int(values.get(name, 0)) for name in physical_names]

    register_bits = physical_extents["register"].bit_length() - 1
    adapter_bases = [
        ("slot", [basis(register=1 << bit) for bit in range(register_bits)]),
    ]
    adapter_bases.extend((
        f"item{bit}",
        [
            basis(lane=(1 << bit) if (1 << bit) < physical_extents["lane"] else 0, ) if bit < lane_bits else basis(
                warp=(1 << (bit - lane_bits)) if (1 << (bit - lane_bits)) < physical_extents["warp"] else 0, )
        ],
    ) for bit in range(lane_bits + warp_bits))
    adapter = LinearLayout.from_bases(
        adapter_bases,
        physical_names,
        [physical_extents[name] for name in physical_names],
        False,
    )
    return _compose_linear_layouts(adapter, linear)


def _compose_linear_layouts(inner, outer):
    """Compose equivalent LinearLayout objects from either Python binding."""
    inner_outputs = tuple(str(name) for name, _extent in inner.out_dims)
    outer_outputs = tuple(str(name) for name, _extent in outer.out_dims)
    outer_bases = {str(name): tuple(tuple(map(int, basis)) for basis in values) for name, values in outer.bases}

    def apply_outer(value):
        result = [0] * len(outer_outputs)
        for input_name, input_value in zip(inner_outputs, map(int, value)):
            for bit, basis in enumerate(outer_bases[input_name]):
                if input_value & (1 << bit):
                    result = [lhs ^ rhs for lhs, rhs in zip(result, basis)]
        return result

    return LinearLayout.from_bases(
        [(str(name), [apply_outer(value) for value in values]) for name, values in inner.bases],
        outer_outputs,
        [int(extent) for _name, extent in outer.out_dims],
        False,
    )


def _joined_packet_layout(layout):
    linear = layout.linear_layout
    if linear is None:
        _layout_fail(
            "TLXW_TYPE_UNSUPPORTED_LAYOUT",
            STAGE,
            "packet join requires a distributed linear layout",
            source_value_id=layout.value_id,
        )
    out_dims = tuple((str(name), int(size)) for name, size in linear.out_dims)
    selector = f"dim{len(out_dims)}"
    bases = []
    saw_register = False
    for name, values in linear.bases:
        values = [[*map(int, basis), 0] for basis in values]
        if str(name) == "register":
            saw_register = True
            values.append([0] * len(out_dims) + [1])
        bases.append((str(name), values))
    if not saw_register:
        bases.append(("register", [[0] * len(out_dims) + [1]]))
    return LinearLayout.from_bases(
        bases,
        [name for name, _size in out_dims] + [selector],
        [size for _name, size in out_dims] + [2],
        False,
    )


def _packet_transform_descriptor(source_shape, result_shape, *, transform, axis, order):
    source_shape = tuple(int(dim) for dim in source_shape)
    result_shape = tuple(int(dim) for dim in result_shape)
    structural = None
    if transform == "identity":
        if source_shape != result_shape:
            raise ValueError("identity layout conversion requires equal shapes")
        mapped_dims = tuple(range(len(result_shape)))
    elif transform == "broadcast":
        if len(source_shape) != len(result_shape) or any(src not in (1, dst)
                                                         for src, dst in zip(source_shape, result_shape)):
            raise ValueError("broadcast layout conversion has incompatible shapes")
        mapped_dims = tuple(None if size == 1 else dim for dim, size in enumerate(source_shape))
    elif transform == "expand_dims":
        if (axis is None or not 0 <= axis < len(result_shape) or result_shape[axis] != 1
                or result_shape[:axis] + result_shape[axis + 1:] != source_shape):
            raise ValueError("expand_dims layout conversion has incompatible shapes")
        mapped_dims = tuple(None if dim == axis else dim - (dim > axis) for dim in range(len(result_shape)))
    elif transform == "trans":
        if (len(order) != len(source_shape) or sorted(order) != list(range(len(source_shape)))
                or tuple(source_shape[dim] for dim in order) != result_shape):
            raise ValueError("transpose layout conversion has an invalid order")
        mapped_dims = tuple(int(dim) for dim in order)
    elif transform in {"split", "reduction"}:
        if (axis is None or not 0 <= axis < len(source_shape)
                or source_shape[:axis] + source_shape[axis + 1:] != result_shape
                or (transform == "split" and source_shape[axis] != 2)):
            raise ValueError(f"{transform} layout conversion has incompatible shapes")
        mapped_dims = tuple(dim + (dim >= axis) for dim in range(len(result_shape)))
        name = "selector" if transform == "split" else "reduction"
        structural = (name, int(source_shape[axis]), int(axis))
    elif transform == "reshape":
        if _product(source_shape) != _product(result_shape):
            raise ValueError("reshape layout conversion changes the element count")
        mapped_dims = None
    else:
        raise ValueError(f"unsupported packet layout transform {transform!r}")
    return mapped_dims, structural


def _packet_destination_layout(result, source_dims, result_dims, mapped_dims, structural, *, reshape):
    """Express destination physical coordinates in the source logical domain."""
    source_shape = tuple(size for _name, size in source_dims)
    result_shape = tuple(size for _name, size in result_dims)

    def transform_coordinate(coordinate):
        if reshape:
            linear = 0
            for component, extent in zip(coordinate, result_shape, strict=True):
                linear = linear * extent + component
            values = [0] * len(source_shape)
            for dim in reversed(range(len(source_shape))):
                values[dim] = linear % source_shape[dim]
                linear //= source_shape[dim]
            return values
        values = [0] * len(source_shape)
        for result_dim, source_dim in enumerate(mapped_dims):
            if source_dim is not None:
                values[source_dim] = coordinate[result_dim]
        return values

    bases = []
    for name, values in result.bases:
        bases.append((str(name), [transform_coordinate(tuple(map(int, basis))) for basis in values]))
    if structural:
        name, extent, source_dim = structural
        selector_bases = []
        bit = 1
        while bit < extent:
            basis = [0] * len(source_shape)
            basis[source_dim] = bit
            selector_bases.append(basis)
            bit <<= 1
        bases.append((name, selector_bases))
    return LinearLayout.from_bases(
        bases,
        [name for name, _size in source_dims],
        source_shape,
        False,
    )


_PACKET_PHYSICAL_DIMS = ("register", "lane", "warp", "block")


def _packet_layout_columns(linear):
    return tuple(((str(name), bit), tuple(int(component)
                                          for component in basis))
                 for name, bases in linear.bases
                 for bit, basis in enumerate(bases))


def _flatten_packet_coordinate(linear, coordinate):
    value = 0
    offset = 0
    for component, (_name, extent) in zip(coordinate, linear.out_dims, strict=True):
        value |= int(component) << offset
        offset += int(extent).bit_length() - 1
    return value


def _packet_source_equations(linear, columns):
    logical_bits = sum(int(extent).bit_length() - 1 for _name, extent in linear.out_dims)
    equations = [0] * logical_bits
    offsets = []
    offset = 0
    for _name, extent in linear.out_dims:
        offsets.append(offset)
        offset += int(extent).bit_length() - 1
    for column, (_physical, basis) in enumerate(columns):
        for dim, component in enumerate(basis):
            for bit in range(int(linear.out_dims[dim][1]).bit_length() - 1):
                if component & (1 << bit):
                    equations[offsets[dim] + bit] |= 1 << column
    return tuple(equations)


def _solve_packet_gf2(equations, rhs, variable_count, constraints=()):
    rows = [[mask, (rhs >> bit) & 1] for bit, mask in enumerate(equations)]
    rows.extend([[1 << variable, value] for variable, value in constraints])
    pivots = []
    pivot_row = 0
    for column in range(variable_count):
        pivot = next((row for row in range(pivot_row, len(rows)) if rows[row][0] & (1 << column)), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        pivot_mask, pivot_rhs = rows[pivot_row]
        for row, values in enumerate(rows):
            if row != pivot_row and values[0] & (1 << column):
                values[0] ^= pivot_mask
                values[1] ^= pivot_rhs
        pivots.append((column, pivot_row))
        pivot_row += 1
    if any(mask == 0 and value for mask, value in rows):
        return None
    return sum(1 << column for column, row in pivots if rows[row][1])


def _preferred_packet_solution(equations, rhs, source_columns, destination_column):
    groups = {
        name: tuple(index
                    for index, ((column_name, _bit), _basis) in enumerate(source_columns)
                    if column_name == name)
        for name in _PACKET_PHYSICAL_DIMS
    }
    preferred = {index: int(destination_column == column) for index, (column, _basis) in enumerate(source_columns)}
    constraints = []
    variable_count = len(source_columns)
    for name in reversed(_PACKET_PHYSICAL_DIMS):
        proposed = constraints + [(variable, preferred[variable]) for variable in groups[name]]
        if _solve_packet_gf2(equations, rhs, variable_count, proposed) is not None:
            constraints = proposed
    constrained = {variable for variable, _value in constraints}
    for name in _PACKET_PHYSICAL_DIMS:
        for variable in sorted(groups[name], key=lambda index: source_columns[index][0][1], reverse=True):
            if variable in constrained:
                continue
            proposed = [*constraints, (variable, 0)]
            value = int(_solve_packet_gf2(equations, rhs, variable_count, proposed) is None)
            constraints.append((variable, value))
            constrained.add(variable)
    solution = _solve_packet_gf2(equations, rhs, variable_count, constraints)
    if solution is None:
        raise ValueError("packet layouts do not define a total redistribution")
    return solution


def _preferred_packet_relation(source, destination):
    source_columns = _packet_layout_columns(source)
    equations = _packet_source_equations(source, source_columns)
    destination_columns = _packet_layout_columns(destination)
    solutions = tuple(
        _preferred_packet_solution(
            equations,
            _flatten_packet_coordinate(source, basis),
            source_columns,
            column,
        ) for column, basis in destination_columns)
    physical_extents = {name: linear_layout_in_dim_size(source, name) for name in _PACKET_PHYSICAL_DIMS}

    def physical_coordinate(solution):
        return tuple(
            sum(1 << bit
                for index, ((column_name, bit), _basis) in enumerate(source_columns)
                if column_name == name and solution & (1 << index))
            for name in _PACKET_PHYSICAL_DIMS)

    relation_bases = []
    solution = iter(solutions)
    for name, bases in destination.bases:
        relation_bases.append((str(name), [physical_coordinate(next(solution)) for _basis in bases]))
    return LinearLayout.from_bases(
        relation_bases,
        list(_PACKET_PHYSICAL_DIMS),
        [physical_extents[name] for name in _PACKET_PHYSICAL_DIMS],
        False,
    )


def _transform_logical_coordinates(
    dsl,
    result_coordinates,
    source_dims,
    result_dims,
    mapped_dims,
    structural,
    packet,
    *,
    reshape,
):
    """Apply an ordinary tensor coordinate transform exactly."""
    source_shape = tuple(size for _name, size in source_dims)
    result_shape = tuple(size for _name, size in result_dims)
    result_values = tuple(result_coordinates[str(name)] for name, _size in result_dims)
    if reshape:
        linear = dsl.ixs_int(0)
        for coordinate, extent in zip(result_values, result_shape):
            linear = linear * int(extent) + coordinate
        values = [dsl.ixs_int(0)] * len(source_shape)
        stride = 1
        for dim in reversed(range(len(source_shape))):
            values[dim] = dsl.mod(dsl.floor(linear / stride), source_shape[dim])
            stride *= int(source_shape[dim])
    else:
        values = [dsl.ixs_int(0)] * len(source_shape)
        for result_dim, source_dim in enumerate(mapped_dims):
            if source_dim is not None:
                values[source_dim] = result_values[result_dim]
        if structural:
            name, _extent, source_dim = structural
            values[source_dim] = packet[name]
    return {str(name): value for (name, _size), value in zip(source_dims, values)}


def layout_warp_count(layout):
    if layout.linear_layout is None:
        raise ValueError("distributed layout is missing its LinearLayout model")
    return linear_layout_in_dim_size(layout.linear_layout, "warp")


def _shared_layout_fail(
    code,
    stage,
    message,
    *,
    layout=None,
    source_op_index=None,
    source_value_id=None,
):
    if source_value_id is None and layout is not None:
        source_value_id = layout.value_id
    fail(
        code,
        stage,
        message,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def _require_supported_coordinate_domain(
    shape,
    coordinate_domain,
    source_value_id,
):
    if coordinate_domain["coverage"] in {"exact", "replicated"}:
        return
    _layout_fail(
        "TLXW_TYPE_UNSUPPORTED_LAYOUT",
        STAGE,
        "unsupported distributed layout coordinate domain "
        f"{coordinate_domain['coverage']}; shape={tuple(shape)} "
        f"domain={coordinate_domain}",
        source_value_id=source_value_id,
    )


def _layout_fail(
    code,
    stage,
    message,
    *,
    source_op_index=None,
    source_value_id=None,
):
    fail(
        code,
        stage,
        message,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def _attr_bool(attr, method):
    fn = getattr(attr, method, None)
    return bool(fn()) if fn is not None else False


def _attr_value(attr, method):
    fn = getattr(attr, method, None)
    if fn is None:
        fail(
            "TLXW_TYPE_MALFORMED_LAYOUT",
            STAGE,
            f"layout encoding is missing {method}",
        )
    return fn()


def _int_tuple(values):
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _basis_tuple(values):
    if values is None:
        return ()
    return tuple(tuple(int(dim) for dim in basis) for basis in values)


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _ceil_div(lhs, rhs):
    return (int(lhs) + int(rhs) - 1) // int(rhs)
