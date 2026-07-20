"""Structural layout maps for the TLX Wave converter."""

from dataclasses import dataclass

from triton._C.libtriton import linear_layout as _linear_layout

LinearLayout = _linear_layout.LinearLayout

from .diagnostics import fail

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


@dataclass(frozen=True)
class PhysicalOffsetRecord:
    element_offset: int
    byte_offset: int
    dword_offset: int | None
    element_byte_width: int
    layout_kind: str
    order: tuple[int, ...]
    logical_coords: tuple[int, ...]
    logical_linear_offset: int
    bindings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    proof_status: str = "static"
    provenance: str = "shared_physical_offset"


@dataclass(frozen=True)
class PhysicalOffsetExpressionPlan:
    expression_kind: str
    offset_unit: str
    element_byte_width: int
    layout_kind: str
    order: tuple[int, ...]
    bindings: tuple[str, ...] = ("logical_coords", )
    assumptions: tuple[str, ...] = ()
    proof_status: str = "symbolic_verified"
    provenance: str = "shared_physical_offset"
    intervals: tuple[int, ...] = ()
    paddings: tuple[int, ...] = ()
    swizzled_vec: int | None = None
    swizzled_per_phase: int | None = None
    swizzled_max_phase: int | None = None
    linear_component_bases: tuple[tuple[int, ...], ...] = ()
    linear_inverse_offset_bases: tuple[tuple[int, ...], ...] = ()


def build_layout_map(layout_map_id, value_id, source_type, lane_width):
    if source_type.kind not in {"tensor", "memdesc"}:
        return None
    attr = source_type.encoding_attr
    kind, properties = _layout_kind_and_properties(
        attr,
        value_id,
        encoding=str(source_type.encoding or ""),
    )
    if kind in {"blocked", "linear", "generic_linear"}:
        coordinate_domain = _layout_coordinate_domain(
            kind,
            source_type.shape,
            properties,
            lane_width,
            value_id,
        )
        _require_supported_coordinate_domain(
            kind,
            source_type.shape,
            properties,
            coordinate_domain,
            value_id,
        )
        properties = {**properties, "coordinate_domain": coordinate_domain}
        component_count = int(coordinate_domain["component_count"])
    else:
        component_count = _layout_component_count(
            source_type,
            kind,
            properties,
            lane_width,
            value_id,
        )
    return LayoutMap(
        layout_map_id,
        value_id,
        kind,
        tuple(source_type.shape),
        source_type.element_type,
        int(component_count),
        int(lane_width),
        properties,
    )


def _layout_kind_and_properties(attr, value_id, *, encoding=None):
    if attr is None:
        return "none", {}
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
        parent_kind, parent_properties = _layout_kind_and_properties(
            parent,
            value_id,
            encoding=str(parent or ""),
        )
        return "slice", {
            "dim": int(_attr_value(attr, "get_slice_dim")),
            "parent_kind": parent_kind,
            "parent_properties": parent_properties,
        }
    if _attr_bool(attr, "is_dot_operand_encoding"):
        parent = _attr_value(attr, "get_dot_operand_parent")
        parent_kind, parent_properties = _layout_kind_and_properties(
            parent,
            value_id,
            encoding=str(parent or ""),
        )
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
            "linear_component": _attr_value(
                attr,
                "get_padded_shared_linear_component",
            ),
        }
    fail(
        "TLXW_TYPE_UNSUPPORTED_LAYOUT",
        STAGE,
        f"unsupported layout encoding {attr}",
        source_value_id=value_id,
    )


def _layout_component_count(source_type, kind, properties, lane_width, value_id):
    if kind in {"blocked", "linear", "generic_linear"}:
        coordinate_domain = _layout_coordinate_domain(
            kind,
            source_type.shape,
            properties,
            lane_width,
            source_value_id=value_id,
        )
        _require_supported_coordinate_domain(
            kind,
            source_type.shape,
            properties,
            coordinate_domain,
            value_id,
        )
        return int(coordinate_domain["component_count"])
    if kind == "dot_operand":
        parent_properties = properties.get("parent_properties", {})
        instr_shape = parent_properties.get("instr_shape", ())
        warps_per_cta = parent_properties.get("warps_per_cta", ())
        if len(instr_shape) >= 3 and len(warps_per_cta) >= 2 and len(source_type.shape) >= 2:
            k_tile_extent = dot_operand_storage_k_tile_extent(
                source_type.element_type,
                properties,
            )
            op_idx = int(properties.get("op_idx", -1))
            if op_idx == 0:
                m_tiles = _per_wave_tile_count(
                    int(source_type.shape[0]),
                    int(instr_shape[0]),
                    int(warps_per_cta[0]),
                )
                k_tiles = _ceil_div(int(source_type.shape[1]), k_tile_extent)
                return m_tiles * k_tiles
            if op_idx == 1:
                n_tiles = _per_wave_tile_count(
                    int(source_type.shape[1]),
                    int(instr_shape[1]),
                    int(warps_per_cta[1]),
                )
                k_tiles = _ceil_div(int(source_type.shape[0]), k_tile_extent)
                return n_tiles * k_tiles
        return 1
    if kind == "slice":
        parent_kind = properties.get("parent_kind")
        parent_properties = properties.get("parent_properties", {})
        if parent_kind in {"blocked", "linear", "generic_linear", "slice", "amd_mfma"}:
            dim = int(properties.get("dim", 0))
            shape = tuple(int(value) for value in source_type.shape)
            if dim < 0 or dim > len(shape):
                _layout_fail(
                    "TLXW_TYPE_MALFORMED_LAYOUT",
                    STAGE,
                    "slice layout dimension is outside the parent rank",
                    source_value_id=value_id,
                )
            parent_shape = list(shape)
            parent_shape.insert(dim, 1)
            if parent_kind == "amd_mfma":
                return _mfma_component_count(
                    tuple(parent_shape),
                    parent_properties,
                    lane_width,
                    source_value_id=value_id,
                )
            linear = distributed_linear_layout_from_parts(
                kind,
                shape,
                properties,
                lane_width,
                source_value_id=value_id,
            )
            return linear_layout_in_dim_size(linear, "register")
    if kind == "amd_mfma":
        return _mfma_component_count(
            source_type.shape,
            properties,
            lane_width,
            source_value_id=value_id,
        )
    element_count = _product(source_type.shape)
    return max(1, _ceil_div(element_count, int(lane_width)))


def distributed_register_count(
    kind,
    shape,
    properties,
    lane_width,
    *,
    stage=STAGE,
    source_op_index=None,
    source_value_id=None,
):
    linear = distributed_linear_layout_from_parts(
        kind,
        shape,
        properties,
        lane_width,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )
    return linear_layout_in_dim_size(linear, "register")


def distributed_linear_layout(
    layout,
    *,
    stage=STAGE,
    source_op_index=None,
):
    return distributed_linear_layout_from_parts(
        layout.kind,
        layout.shape,
        layout.properties,
        layout.lane_width,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=layout.value_id,
    )


def distributed_linear_layout_from_parts(
    kind,
    shape,
    properties,
    lane_width,
    *,
    stage=STAGE,
    source_op_index=None,
    source_value_id=None,
):
    shape = tuple(int(dim) for dim in shape)
    if kind == "blocked":
        return _blocked_linear_layout(
            shape,
            properties,
            stage=stage,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if kind in {"linear", "generic_linear"}:
        return _linear_encoding_layout(
            shape,
            properties,
            stage=stage,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if kind == "slice":
        return _slice_linear_layout(
            shape,
            properties,
            lane_width,
            stage=stage,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if kind == "amd_mfma":
        return _mfma_linear_layout(
            shape,
            properties,
            lane_width,
            stage=stage,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    _layout_fail(
        "TLXW_TYPE_UNSUPPORTED_LAYOUT",
        stage,
        f"layout {kind} does not have a distributed register map",
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def linear_layout_in_dim_size(linear, dim):
    for in_dim, bases in linear.bases:
        if in_dim == dim:
            return 1 << len(bases)
    return 1


def optimal_swizzled_ldst_plan(source, result, bitwidth, target):
    """Build the physical LDS conversion plan used by Triton's LLVM lowering."""
    arch = str(target or "").rsplit(":", 1)[-1]
    if arch == "gfx950":
        num_banks = 64
        load_lane_addr_128 = (0, 1, 3, 4)
    elif arch == "gfx942":
        num_banks = 32
        load_lane_addr_128 = (0, 1, 4)
    else:
        raise ValueError(f"unsupported AMD LDS swizzle target {target!r}")

    vector_bitwidth = int(
        _linear_layout.get_vec_bitwidth_ld_st(
            source,
            result,
            int(bitwidth),
        ))
    # Match AMD TargetInfo::getSharedLdStTiles. Stores use the regular lane
    # tile; 128-bit loads use the architecture-specific ds_read_b128 tile.
    load_lane_addr = load_lane_addr_128 if vector_bitwidth == 128 else ()
    return _linear_layout.optimal_swizzled_ldst_plan(
        source,
        result,
        int(bitwidth),
        num_banks=num_banks,
        dst_lane_addr=load_lane_addr,
    )


def linear_layout_out_dim_size(linear, dim, *, stage=STAGE):
    for out_dim, size in linear.out_dims:
        if out_dim == dim:
            return int(size)
    _layout_fail(
        "TLXW_TYPE_MALFORMED_LAYOUT",
        stage,
        f"linear layout is missing output dimension {dim}",
    )


def linear_layout_coords(linear, register, lane, *, warp):
    available = {
        "block": 0,
        "register": int(register),
        "lane": int(lane),
        "warp": int(warp),
    }
    coords = linear.apply({name: available[name] for name in linear.get_in_dim_names()})
    return tuple(int(coords[f"dim{dim}"]) for dim in range(len(coords)))


def linear_layout_bases(linear, in_dim):
    for name, bases in linear.bases:
        if name == in_dim:
            return tuple(tuple(int(value) for value in basis) for basis in bases)
    return ()


def linear_layout_component_registers(
    linear,
    layout,
    component_count,
    *,
    stage=STAGE,
    source_op_index=None,
    source_value_id=None,
):
    component_count = int(component_count)
    if component_count <= 0:
        _layout_fail(
            "TLXW_TYPE_MALFORMED_LAYOUT",
            stage,
            "layout component register mapping requires a positive component count",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    register_count = linear_layout_in_dim_size(linear, "register")
    if int(register_count) == component_count:
        stride = 1
    elif _is_mfma_component_layout(layout) and int(register_count) % component_count == 0:
        stride = int(register_count) // component_count
    else:
        _layout_fail(
            "TLXW_TYPE_UNSUPPORTED_LAYOUT",
            stage,
            "layout component model does not match distributed register layout",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    return tuple(int(component) * int(stride) for component in range(component_count))


def dot_operand_storage_k_tile_extent(element_type, properties):
    parent_properties = properties.get("parent_properties", {})
    instr_shape = parent_properties.get("instr_shape", ())
    if len(instr_shape) < 3:
        return 1
    instr_k = int(instr_shape[2])
    k_width = int(properties.get("k_width", 0) or 0)
    if element_type == "i8" and k_width > 0 and 32 % k_width == 0:
        storage_pack = max(1, 32 // int(k_width))
        if storage_pack > 1 and instr_k % storage_pack == 0:
            return instr_k // storage_pack
    return instr_k


def _is_mfma_component_layout(layout):
    return _layout_parts_are_mfma(layout.kind, layout.properties)


def _layout_parts_are_mfma(kind, properties):
    if kind == "amd_mfma":
        return True
    if kind == "slice":
        return _layout_parts_are_mfma(
            properties.get("parent_kind"),
            properties.get("parent_properties", {}),
        )
    return False


def layout_warp_count(layout):
    if layout.kind in {"linear", "generic_linear"}:
        return 1 << len(tuple(layout.properties.get("warp_bases", ())))
    if layout.kind == "slice":
        return _layout_warp_count_from_parts(
            layout.properties.get("parent_kind"),
            layout.properties.get("parent_properties", {}),
        )
    warps_per_cta = tuple(int(value) for value in layout.properties.get("warps_per_cta", ()))
    result = 1
    for value in warps_per_cta:
        result *= max(1, int(value))
    return result


def shared_physical_offset(
    layout,
    shape,
    coords,
    element_byte_width,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    shape = tuple(int(dim) for dim in shape)
    coords = tuple(int(coord) for coord in coords)
    element_byte_width = int(element_byte_width)
    if element_byte_width <= 0:
        _shared_layout_fail(
            diagnostic,
            stage,
            "shared physical offset requires a positive element byte width",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if len(coords) != len(shape):
        _shared_layout_fail(
            diagnostic,
            stage,
            "shared physical offset coordinate rank does not match shape rank",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    for coord, extent in zip(coords, shape):
        if int(coord) < 0 or int(coord) >= int(extent):
            _shared_layout_fail(
                diagnostic,
                stage,
                f"shared physical offset coordinate {coords} exceeds shape {shape}",
                layout=layout,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )

    kind = shared_layout_kind(layout)
    assumptions = ()
    if kind == "dense":
        order = default_physical_order(shape)
        element_offset = static_linear_offset(shape, coords)
        logical_linear_offset = element_offset
        provenance = "dense_row_major"
    elif kind == "swizzled_shared":
        if is_identity_swizzled_shared(layout, shape):
            order = default_physical_order(shape)
            element_offset = static_linear_offset(shape, coords)
            logical_linear_offset = element_offset
            provenance = "identity_swizzled_row_major"
        else:
            order, vec, per_phase, max_phase = swizzled_shared_parameters(
                layout,
                shape,
                stage=stage,
                diagnostic=diagnostic,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )
            minor_dim = int(order[0])
            major_dim = int(order[1])
            minor_extent = int(shape[minor_dim])
            major = int(coords[major_dim])
            minor = int(coords[minor_dim])
            phase = (major // int(per_phase)) % int(max_phase)
            if int(vec) * int(max_phase) <= minor_extent:
                swizzled_minor = (
                    ((minor // int(vec)) ^ phase) * int(vec)
                    + (minor % int(vec))
                )
            else:
                phase_offset = (int(vec) * int(phase)) % minor_extent
                swizzled_minor = int(minor) ^ int(phase_offset)
            if swizzled_minor >= minor_extent:
                _shared_layout_fail(
                    diagnostic,
                    stage,
                    "swizzled shared physical offset produces an "
                    f"out-of-bounds minor coordinate {swizzled_minor}; "
                    f"{swizzled_shared_description(layout)}",
                    layout=layout,
                    source_op_index=source_op_index,
                    source_value_id=source_value_id,
                )
            physical_coords = list(coords)
            physical_coords[minor_dim] = int(swizzled_minor)
            element_offset = ordered_linear_offset(shape, physical_coords, order)
            logical_linear_offset = ordered_linear_offset(shape, coords, order)
            provenance = "swizzled_shared"
            assumptions = ("minor_extent_divisible_by_vec", )
    elif kind == "shared_linear":
        order = shared_layout_physical_order(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        inverse_offset_bases = shared_linear_inverse_offset_bases(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        element_offset = offset_from_inverse_offset_bases(
            inverse_offset_bases,
            coords,
        )
        logical_linear_offset = static_linear_offset(shape, coords)
        provenance = "shared_linear_inverse"
        assumptions = ("single_block_bijective_shared_linear", )
    elif kind == "padded_shared":
        intervals, paddings = padded_shared_parameters(
            layout,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        order = shared_layout_physical_order(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        offset_bases = padded_shared_linear_component_bases(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        logical_linear_offset = offset_from_linear_component_bases(
            offset_bases,
            coords,
            layout,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        element_offset = int(logical_linear_offset)
        for interval, padding in zip(intervals, paddings):
            element_offset += (logical_linear_offset // int(interval)) * int(padding)
        provenance = "padded_shared"
        assumptions = ("valid_padded_intervals", )
    else:
        _shared_layout_fail(
            diagnostic,
            stage,
            f"shared physical offset does not support layout {kind}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )

    byte_offset = int(element_offset) * int(element_byte_width)
    dword_offset = byte_offset // 4 if byte_offset % 4 == 0 else None
    return PhysicalOffsetRecord(
        element_offset=int(element_offset),
        byte_offset=int(byte_offset),
        dword_offset=dword_offset,
        element_byte_width=int(element_byte_width),
        layout_kind=kind,
        order=tuple(int(dim) for dim in order),
        logical_coords=coords,
        logical_linear_offset=int(logical_linear_offset),
        assumptions=tuple(assumptions),
        provenance=provenance,
    )


def shared_physical_offset_from_linear(
    layout,
    shape,
    linear,
    element_byte_width,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    shape = tuple(int(dim) for dim in shape)
    linear = int(linear)
    if linear < 0 or linear >= _product(shape):
        return None
    coords = static_delinearize_row_major(
        linear,
        shape,
        stage=stage,
        diagnostic=diagnostic,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
        layout=layout,
    )
    return shared_physical_offset(
        layout,
        shape,
        coords,
        int(element_byte_width),
        stage=stage,
        diagnostic=diagnostic,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def shared_physical_offset_expression_plan(
    layout,
    shape,
    element_byte_width,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    shape = tuple(int(dim) for dim in shape)
    element_byte_width = int(element_byte_width)
    if element_byte_width <= 0:
        _shared_layout_fail(
            diagnostic,
            stage,
            "shared physical offset expression requires a positive element byte width",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    kind = shared_layout_kind(layout)
    if kind == "dense":
        return PhysicalOffsetExpressionPlan(
            expression_kind="dense_row_major",
            offset_unit="element",
            element_byte_width=element_byte_width,
            layout_kind=kind,
            order=default_physical_order(shape),
            provenance="dense_row_major",
        )
    if kind == "swizzled_shared":
        if is_identity_swizzled_shared(layout, shape):
            return PhysicalOffsetExpressionPlan(
                expression_kind="dense_row_major",
                offset_unit="element",
                element_byte_width=element_byte_width,
                layout_kind=kind,
                order=default_physical_order(shape),
                assumptions=("identity_swizzled_shared", ),
                provenance="identity_swizzled_row_major",
            )
        order, vec, per_phase, max_phase = swizzled_shared_parameters(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        return PhysicalOffsetExpressionPlan(
            expression_kind="swizzled_xor",
            offset_unit="element",
            element_byte_width=element_byte_width,
            layout_kind=kind,
            order=tuple(int(dim) for dim in order),
            assumptions=("minor_extent_divisible_by_vec", ),
            provenance="swizzled_shared",
            swizzled_vec=int(vec),
            swizzled_per_phase=int(per_phase),
            swizzled_max_phase=int(max_phase),
        )
    if kind == "shared_linear":
        order = shared_layout_physical_order(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        inverse_offset_bases = shared_linear_inverse_offset_bases(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        return PhysicalOffsetExpressionPlan(
            expression_kind="linear_shared",
            offset_unit="element",
            element_byte_width=element_byte_width,
            layout_kind=kind,
            order=tuple(int(dim) for dim in order),
            assumptions=("single_block_bijective_shared_linear", ),
            provenance="shared_linear_inverse",
            linear_inverse_offset_bases=inverse_offset_bases,
        )
    if kind == "padded_shared":
        intervals, paddings = padded_shared_parameters(
            layout,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        order = shared_layout_physical_order(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        offset_bases = padded_shared_linear_component_bases(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        return PhysicalOffsetExpressionPlan(
            expression_kind="padded_linear",
            offset_unit="element",
            element_byte_width=element_byte_width,
            layout_kind=kind,
            order=tuple(int(dim) for dim in order),
            assumptions=("valid_padded_intervals", ),
            provenance="padded_shared",
            intervals=tuple(int(value) for value in intervals),
            paddings=tuple(int(value) for value in paddings),
            linear_component_bases=tuple(tuple(int(value) for value in basis) for basis in offset_bases),
        )
    _shared_layout_fail(
        diagnostic,
        stage,
        f"shared physical offset expression does not support layout {kind}",
        layout=layout,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def physical_offset_expression_plan_attrs(plan, prefix):
    prefix = str(prefix)
    attrs = {
        f"{prefix}_physical_offset_plan": plan.expression_kind,
        f"{prefix}_physical_offset_unit": plan.offset_unit,
        f"{prefix}_physical_element_byte_width": int(plan.element_byte_width),
        f"{prefix}_physical_layout_kind": plan.layout_kind,
        f"{prefix}_physical_order": tuple(int(dim) for dim in plan.order),
        f"{prefix}_physical_bindings": tuple(str(name) for name in plan.bindings),
        f"{prefix}_physical_assumptions": tuple(str(assumption) for assumption in plan.assumptions),
        f"{prefix}_physical_proof_status": plan.proof_status,
        f"{prefix}_physical_provenance": plan.provenance,
    }
    if plan.expression_kind == "padded_linear" or plan.intervals:
        attrs[f"{prefix}_physical_intervals"] = tuple(int(value) for value in plan.intervals)
    if plan.expression_kind == "padded_linear" or plan.paddings:
        attrs[f"{prefix}_physical_paddings"] = tuple(int(value) for value in plan.paddings)
    if plan.swizzled_vec is not None:
        attrs[f"{prefix}_physical_swizzled_vec"] = int(plan.swizzled_vec)
    if plan.swizzled_per_phase is not None:
        attrs[f"{prefix}_physical_swizzled_per_phase"] = int(plan.swizzled_per_phase)
    if plan.swizzled_max_phase is not None:
        attrs[f"{prefix}_physical_swizzled_max_phase"] = int(plan.swizzled_max_phase)
    if plan.linear_component_bases:
        attrs[f"{prefix}_physical_linear_component_bases"] = tuple(
            tuple(int(value) for value in basis) for basis in plan.linear_component_bases)
    if plan.linear_inverse_offset_bases:
        attrs[f"{prefix}_physical_linear_inverse_offset_bases"] = tuple(
            tuple(int(value) for value in bases)
            for bases in plan.linear_inverse_offset_bases)
    return attrs


def shared_layout_kind(layout):
    if layout is None or layout.kind == "none":
        return "dense"
    if layout.kind in {"linear", "generic_linear"}:
        return "linear_shared"
    return str(layout.kind)


def default_physical_order(shape):
    return tuple(reversed(range(len(tuple(shape)))))


def expand_physical_order(
    order,
    rank,
    *,
    layout=None,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    order = tuple(int(dim) for dim in order)
    rank = int(rank)
    if len(order) > rank or sorted(order) != list(range(len(order))):
        _shared_layout_fail(
            diagnostic,
            stage,
            f"shared layout order {order} cannot be applied to rank-{rank} shape",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    prefix_rank = rank - len(order)
    mapped = tuple(prefix_rank + int(dim) for dim in order)
    return mapped + tuple(reversed(range(prefix_rank)))


def shared_layout_physical_order(
    layout,
    shape,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    shape = tuple(int(dim) for dim in shape)
    if not shape:
        return ()
    if layout is not None and layout.kind in {
            "padded_shared",
            "shared_linear",
            "swizzled_shared",
    }:
        order = tuple(int(dim) for dim in layout.properties.get("order", ()))
        if order:
            return expand_physical_order(
                order,
                len(shape),
                layout=layout,
                stage=stage,
                diagnostic=diagnostic,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )
    return default_physical_order(shape)


def shared_linear_inverse_offset_bases(
    layout,
    shape,
    *,
    stage,
    diagnostic,
    source_op_index,
    source_value_id,
):
    """Import a shared-linear logical-coordinate to LDS-offset map.

    ``SharedLinearEncodingAttr`` stores the forward GF(2) map from physical
    ``(offset, block)`` bits to logical coordinates.  Address generation needs
    the inverse map.  Preserve that map as one offset-bit mask per logical
    coordinate bit; masks may contain multiple bits and are combined with XOR.
    """
    shape = tuple(int(dim) for dim in shape)
    linear = layout.properties.get("linear_component")
    if linear is None:
        _shared_layout_fail(
            diagnostic,
            stage,
            "shared_linear layout is missing its linear map",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    in_dims = tuple(str(dim) for dim in linear.get_in_dim_names())
    if in_dims != ("offset", "block"):
        _shared_layout_fail(
            diagnostic,
            stage,
            "shared_linear map must use [offset, block] input dims; "
            f"got {in_dims}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    out_dims = tuple((str(name), int(size)) for name, size in linear.out_dims)
    expected_out_dims = tuple(
        (f"dim{dim}", int(extent)) for dim, extent in enumerate(shape))
    if out_dims != expected_out_dims:
        _shared_layout_fail(
            diagnostic,
            stage,
            "shared_linear output dims do not match memdesc shape; "
            f"got {out_dims}, expected {expected_out_dims}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if not linear.is_invertible():
        _shared_layout_fail(
            diagnostic,
            stage,
            "shared_linear map must be bijective for Wave LDS addressing",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    inverse = linear.invert()
    inverse_in_dims = tuple(str(dim) for dim in inverse.get_in_dim_names())
    expected_in_dims = tuple(name for name, _size in expected_out_dims)
    inverse_out_dims = tuple((str(name), int(size)) for name, size in inverse.out_dims)
    if inverse_in_dims != expected_in_dims or tuple(name for name, _size in inverse_out_dims) != (
            "offset",
            "block",
    ):
        _shared_layout_fail(
            diagnostic,
            stage,
            "shared_linear inverse map has unexpected dimensions; "
            f"inputs={inverse_in_dims}, outputs={inverse_out_dims}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    offset_size = dict(inverse_out_dims)["offset"]
    block_size = dict(inverse_out_dims)["block"]
    if int(block_size) != 1 or int(offset_size) != _product(shape):
        _shared_layout_fail(
            diagnostic,
            stage,
            "Wave shared_linear addressing requires one block and one unique "
            f"offset per element; inverse outputs={inverse_out_dims}, shape={shape}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    result = []
    for dim, extent in enumerate(shape):
        name = f"dim{dim}"
        bases = linear_layout_bases(inverse, name)
        expected_bits = _power_of_two_log2(
            extent,
            layout=layout,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        if len(bases) != expected_bits or any(len(basis) != 2 for basis in bases):
            _shared_layout_fail(
                diagnostic,
                stage,
                "shared_linear inverse bases do not match the logical shape; "
                f"dim={name}, bases={bases}, extent={extent}",
                layout=layout,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )
        if any(int(basis[1]) != 0 for basis in bases):
            _shared_layout_fail(
                diagnostic,
                stage,
                "shared_linear logical coordinates must not select another "
                f"CTA block; dim={name}, bases={bases}",
                layout=layout,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )
        offset_bases = tuple(int(basis[0]) for basis in bases)
        if any(value < 0 or value >= int(offset_size) for value in offset_bases):
            _shared_layout_fail(
                diagnostic,
                stage,
                "shared_linear inverse offset basis exceeds the allocation; "
                f"dim={name}, bases={offset_bases}, offset_size={offset_size}",
                layout=layout,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )
        result.append(offset_bases)
    return tuple(result)


def offset_from_inverse_offset_bases(inverse_offset_bases, coords):
    offset = 0
    for dim, bases in enumerate(tuple(inverse_offset_bases)):
        coord = int(coords[dim])
        for bit, contribution in enumerate(tuple(bases)):
            if coord & (1 << int(bit)):
                offset ^= int(contribution)
    return int(offset)


def padded_shared_linear_component_bases(
    layout,
    shape,
    *,
    stage,
    diagnostic,
    source_op_index,
    source_value_id,
):
    shape = tuple(int(dim) for dim in shape)
    linear_component = layout.properties.get("linear_component")
    if linear_component is not None:
        bases = _padded_linear_component_offset_bases(
            layout,
            shape,
            linear_component,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    else:
        order = shared_layout_physical_order(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        bases = identity_offset_bases(shape, order)
    _validate_padded_offset_bases(
        layout,
        shape,
        bases,
        stage=stage,
        diagnostic=diagnostic,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )
    return tuple(tuple(int(value) for value in basis) for basis in bases)


def _padded_linear_component_offset_bases(
    layout,
    shape,
    linear_component,
    *,
    stage,
    diagnostic,
    source_op_index,
    source_value_id,
):
    shape = tuple(int(dim) for dim in shape)
    in_dims = tuple(str(dim) for dim in linear_component.get_in_dim_names())
    if in_dims != ("offset", "block"):
        _shared_layout_fail(
            diagnostic,
            stage,
            "padded shared linearComponent must use [offset, block] input dims; "
            f"got {in_dims}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    out_dims = tuple((str(name), int(size)) for name, size in linear_component.out_dims)
    component_shape = tuple(int(size) for _name, size in out_dims)
    component_rank = len(component_shape)
    if component_rank > len(shape):
        _shared_layout_fail(
            diagnostic,
            stage,
            "padded shared linearComponent rank exceeds memdesc shape rank; "
            f"linearComponent dims={out_dims}, shape={shape}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    expected_names = tuple((f"dim{dim}", int(extent)) for dim, extent in enumerate(component_shape))
    if out_dims != expected_names or component_shape != shape[-component_rank:]:
        _shared_layout_fail(
            diagnostic,
            stage,
            "padded shared linearComponent output dims do not match trailing "
            f"memdesc shape; got {out_dims}, shape={shape}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    prefix_rank = len(shape) - component_rank
    bases = [
        tuple((0, ) * prefix_rank + tuple(int(value)
                                          for value in basis))
        for basis in linear_layout_bases(linear_component, "offset")
    ]
    for dim in reversed(range(prefix_rank)):
        bits = _power_of_two_log2(
            int(shape[dim]),
            layout=layout,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        for bit in range(bits):
            basis = [0] * len(shape)
            basis[dim] = 1 << bit
            bases.append(tuple(basis))
    return tuple(bases)


def identity_offset_bases(shape, order):
    shape = tuple(int(dim) for dim in shape)
    bases = []
    for dim in tuple(int(value) for value in order):
        extent = int(shape[dim])
        bits = _power_of_two_log2(extent)
        for bit in range(bits):
            basis = [0] * len(shape)
            basis[dim] = 1 << bit
            bases.append(tuple(basis))
    return tuple(bases)


def offset_from_linear_component_bases(
    bases,
    coords,
    layout,
    *,
    stage,
    diagnostic,
    source_op_index,
    source_value_id,
):
    offset = 0
    for bit, dim, value in _iter_padded_offset_basis_bits(
            layout,
            bases,
            len(tuple(coords)),
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
    ):
        if int(coords[dim]) & int(value):
            offset += 1 << int(bit)
    return int(offset)


def _validate_padded_offset_bases(
    layout,
    shape,
    bases,
    *,
    stage,
    diagnostic,
    source_op_index,
    source_value_id,
):
    seen = set()
    for _bit, dim, value in _iter_padded_offset_basis_bits(
            layout,
            bases,
            len(tuple(shape)),
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
    ):
        key = (int(dim), int(value))
        if key in seen:
            _shared_layout_fail(
                diagnostic,
                stage,
                "padded shared linearComponent repeats an offset basis bit; "
                f"{padded_shared_description(layout)}",
                layout=layout,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )
        seen.add(key)


def _iter_padded_offset_basis_bits(
    layout,
    bases,
    rank,
    *,
    stage,
    diagnostic,
    source_op_index,
    source_value_id,
):
    rank = int(rank)
    for bit, basis in enumerate(tuple(bases)):
        basis = tuple(int(value) for value in basis)
        if len(basis) != rank:
            _shared_layout_fail(
                diagnostic,
                stage,
                "padded shared linearComponent basis rank does not match shape; "
                f"basis={basis}, rank={rank}",
                layout=layout,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )
        nonzero = [(dim, value) for dim, value in enumerate(basis) if value]
        if len(nonzero) != 1:
            _shared_layout_fail(
                diagnostic,
                stage,
                "padded shared linearComponent offset basis must move in "
                f"exactly one dimension; basis={basis}",
                layout=layout,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )
        dim, value = nonzero[0]
        if value <= 0 or not _is_power_of_two(value):
            _shared_layout_fail(
                diagnostic,
                stage,
                "padded shared linearComponent offset basis must be a "
                f"positive power of two; basis={basis}",
                layout=layout,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )
        yield int(bit), int(dim), int(value)


def _power_of_two_log2(
    value,
    *,
    layout=None,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    value = int(value)
    if value <= 0 or not _is_power_of_two(value):
        _shared_layout_fail(
            diagnostic,
            stage,
            f"padded shared physical offset requires power-of-two shape dims; got {value}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    return value.bit_length() - 1


def _is_power_of_two(value):
    value = int(value)
    return value > 0 and value & (value - 1) == 0


def static_linear_offset(shape, coords):
    offset = 0
    shape = tuple(int(dim) for dim in shape)
    for dim, coord in enumerate(coords):
        stride = _product(shape[dim + 1:])
        offset += int(coord) * stride
    return int(offset)


def ordered_linear_offset(shape, coords, order):
    offset = 0
    stride = 1
    shape = tuple(int(dim) for dim in shape)
    for dim in order:
        offset += int(coords[int(dim)]) * stride
        stride *= int(shape[int(dim)])
    return int(offset)


def ordered_coords_from_linear(linear, shape, order):
    coords = [0] * len(shape)
    remainder = int(linear)
    for dim in order:
        extent = int(shape[int(dim)])
        coords[int(dim)] = remainder % extent
        remainder //= extent
    return tuple(coords)


def static_delinearize_row_major(
    linear,
    shape,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
    layout=None,
):
    shape = tuple(int(dim) for dim in shape)
    coords = [0] * len(shape)
    remainder = int(linear)
    for dim in reversed(range(len(shape))):
        extent = int(shape[dim])
        coords[dim] = remainder % extent
        remainder //= extent
    if remainder:
        _shared_layout_fail(
            diagnostic,
            stage,
            f"linear index {linear} exceeds shape {shape}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    return tuple(coords)


def swizzled_shared_parameters(
    layout,
    shape,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    order = tuple(int(dim) for dim in layout.properties.get("order", ()))
    shape = tuple(int(dim) for dim in shape)
    if len(shape) < 2 or len(order) != len(shape) or sorted(order) != list(range(len(shape))):
        _shared_layout_fail(
            diagnostic,
            stage,
            "swizzled shared physical offsets require a rank-matching "
            f"permutation; shape={shape}, {swizzled_shared_description(layout)}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    vec = int(layout.properties["vec"])
    per_phase = int(layout.properties["per_phase"])
    max_phase = int(layout.properties["max_phase"])
    if vec <= 0 or per_phase <= 0 or max_phase <= 0:
        _shared_layout_fail(
            diagnostic,
            stage,
            "swizzled shared layout requires positive vec/perPhase/maxPhase; "
            f"got {swizzled_shared_description(layout)}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    minor_extent = int(shape[int(order[0])])
    if minor_extent % vec:
        _shared_layout_fail(
            diagnostic,
            stage,
            f"swizzled shared minor extent {minor_extent} is not divisible "
            f"by vec={vec}; {swizzled_shared_description(layout)}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    return order, vec, per_phase, max_phase


def swizzled_shared_linear_component_bases(
    layout,
    shape,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    """Return Triton's physical-offset to logical-coordinate GF(2) map."""
    shape = tuple(int(dim) for dim in shape)
    order, vec, per_phase, max_phase = swizzled_shared_parameters(
        layout,
        shape,
        stage=stage,
        diagnostic=diagnostic,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )
    minor_dim = int(order[0])
    major_dim = int(order[1])
    minor_extent = int(shape[minor_dim])
    bases = []
    for dim in order:
        dim = int(dim)
        bits = _power_of_two_log2(
            shape[dim],
            layout=layout,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        for bit in range(bits):
            value = 1 << int(bit)
            basis = [0] * len(shape)
            basis[dim] = value
            if dim == major_dim:
                phase = (value // int(per_phase)) % int(max_phase)
                basis[minor_dim] = (int(vec) * int(phase)) % minor_extent
            bases.append(tuple(int(component) for component in basis))
    return tuple(bases)


def padded_shared_parameters(
    layout,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    if tuple(layout.properties.get("order", ())) not in {(0, 1), (1, 0), (0, ), ()}:
        _shared_layout_fail(
            diagnostic,
            stage,
            f"unsupported padded shared order; {padded_shared_description(layout)}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    intervals = tuple(int(value) for value in layout.properties.get("intervals", ()))
    paddings = tuple(int(value) for value in layout.properties.get("paddings", ()))
    if len(intervals) != len(paddings):
        _shared_layout_fail(
            diagnostic,
            stage,
            "padded shared layout requires matching interval/padding counts; "
            f"{padded_shared_description(layout)}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if any(interval <= 0 for interval in intervals):
        _shared_layout_fail(
            diagnostic,
            stage,
            "padded shared intervals must be positive; "
            f"{padded_shared_description(layout)}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    return intervals, paddings


def is_identity_swizzled_shared(layout, shape):
    props = layout.properties
    order = tuple(props.get("order", ()))
    return (int(props.get("vec", 0)) == 1 and int(props.get("per_phase", 0)) == 1
            and int(props.get("max_phase", 0)) == 1
            and order == default_physical_order(shape))


def require_identity_swizzled_shared(
    layout,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    props = layout.properties
    if (int(props.get("vec", 0)) == 1 and int(props.get("per_phase", 0)) == 1 and int(props.get("max_phase", 0)) == 1):
        return
    _shared_layout_fail(
        diagnostic,
        stage,
        "swizzled shared layout requires an explicit remap target op",
        layout=layout,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def swizzled_shared_description(layout):
    props = layout.properties
    return (f"order={tuple(props.get('order', ()))}, "
            f"vec={int(props.get('vec', 0))}, "
            f"per_phase={int(props.get('per_phase', 0))}, "
            f"max_phase={int(props.get('max_phase', 0))}")


def padded_shared_description(layout):
    props = layout.properties
    return (f"order={tuple(props.get('order', ()))}, "
            f"intervals={tuple(int(value) for value in props.get('intervals', ()))}, "
            f"paddings={tuple(int(value) for value in props.get('paddings', ()))}")


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


def _layout_coordinate_domain(kind, shape, properties, lane_width, source_value_id):
    linear = distributed_linear_layout_from_parts(
        kind,
        shape,
        properties,
        lane_width,
        source_value_id=source_value_id,
    )
    component_count = linear_layout_in_dim_size(linear, "register")
    warp_count = _layout_warp_count_from_parts(kind, properties)
    block_count = linear_layout_in_dim_size(linear, "block")
    shape = tuple(int(dim) for dim in shape)
    total_elements = _product(shape)
    seen = set()
    duplicate_slots = 0
    out_of_bounds_slots = 0
    for component in range(int(component_count)):
        for warp in range(int(warp_count)):
            for lane in range(int(lane_width)):
                coords = linear_layout_coords(linear, component, lane, warp=warp)
                if len(coords) != len(shape) or any(
                        int(coord) < 0 or int(coord) >= int(extent) for coord, extent in zip(coords, shape)):
                    out_of_bounds_slots += 1
                    continue
                if coords in seen:
                    duplicate_slots += 1
                seen.add(coords)
    physical_slots = int(component_count) * int(lane_width) * int(warp_count)
    if int(block_count) <= 0 or total_elements % int(block_count):
        coverage = "block_mismatch"
        local_elements = total_elements
    else:
        local_elements = total_elements // int(block_count)
        if out_of_bounds_slots:
            coverage = "out_of_bounds"
        elif len(seen) == local_elements and duplicate_slots == 0:
            coverage = "exact"
        elif len(seen) == local_elements:
            coverage = "replicated"
        elif duplicate_slots:
            coverage = "duplicate_partial"
        else:
            coverage = "partial"
    return {
        "coverage": coverage,
        "component_count": int(component_count),
        "covered_elements": int(len(seen)),
        "duplicate_slots": int(duplicate_slots),
        "local_elements": int(local_elements),
        "physical_slots": int(physical_slots),
        "out_of_bounds_slots": int(out_of_bounds_slots),
        "block_count": int(block_count),
    }


def _require_supported_coordinate_domain(
    kind,
    shape,
    properties,
    coordinate_domain,
    source_value_id,
):
    if coordinate_domain["coverage"] in {"exact", "replicated"}:
        return
    _layout_fail(
        "TLXW_TYPE_UNSUPPORTED_LAYOUT",
        STAGE,
        "unsupported distributed layout coordinate domain "
        f"{coordinate_domain['coverage']}; kind={kind} shape={tuple(shape)} "
        f"domain={coordinate_domain} bases={_basis_pattern(kind, properties)}",
        source_value_id=source_value_id,
    )


def _layout_warp_count_from_parts(kind, properties):
    if kind in {"linear", "generic_linear"}:
        return 1 << len(tuple(properties.get("warp_bases", ())))
    if kind == "slice":
        return _layout_warp_count_from_parts(
            properties.get("parent_kind"),
            properties.get("parent_properties", {}),
        )
    warps_per_cta = tuple(int(value) for value in properties.get("warps_per_cta", ()))
    result = 1
    for value in warps_per_cta:
        result *= max(1, int(value))
    return result


def _basis_pattern(kind, properties):
    if kind in {"linear", "generic_linear"}:
        return {
            "register": tuple(properties.get("register_bases", ())),
            "lane": tuple(properties.get("lane_bases", ())),
            "warp": tuple(properties.get("warp_bases", ())),
            "block": tuple(properties.get("block_bases", ())),
        }
    if kind == "blocked":
        return {
            "size_per_thread": tuple(properties.get("size_per_thread", ())),
            "threads_per_warp": tuple(properties.get("threads_per_warp", ())),
            "warps_per_cta": tuple(properties.get("warps_per_cta", ())),
            "order": tuple(properties.get("order", ())),
        }
    return dict(properties)


def mfma_registers_per_component(
    layout,
    *,
    stage=STAGE,
    source_op_index=None,
):
    instr_shape = tuple(int(value) for value in layout.properties.get("instr_shape", ()))
    if instr_shape == (16, 16, 32):
        return 4
    if instr_shape == (16, 16, 128):
        return 4
    if instr_shape == (32, 32, 16):
        return 16
    _layout_fail(
        "TLXW_TYPE_UNSUPPORTED_LAYOUT",
        stage,
        f"unsupported MFMA register count for instrShape={instr_shape}",
        source_op_index=source_op_index,
        source_value_id=layout.value_id,
    )


def _blocked_linear_layout(
    shape,
    properties,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    rank = len(shape)
    size_per_thread = tuple(int(value) for value in properties["size_per_thread"])
    threads_per_warp = tuple(int(value) for value in properties["threads_per_warp"])
    warps_per_cta = tuple(int(value) for value in properties["warps_per_cta"])
    order = tuple(int(value) for value in properties["order"])
    if not (len(size_per_thread) == len(threads_per_warp) == len(warps_per_cta) == len(order) == rank):
        _layout_fail(
            "TLXW_TYPE_MALFORMED_LAYOUT",
            stage,
            "blocked layout requires rank-matched metadata",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    linear = (_identity_standard_nd("register", size_per_thread, order) *
              _identity_standard_nd("lane", threads_per_warp, order) *
              _identity_standard_nd("warp", warps_per_cta, order))
    return _ensure_layout_matches_shape(
        linear,
        shape,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def _linear_encoding_layout(
    shape,
    properties,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    rank = len(shape)
    out_dims = [f"dim{dim}" for dim in range(rank)]
    bases = [
        ("register", [list(basis) for basis in properties.get("register_bases", ())]),
        ("lane", [list(basis) for basis in properties.get("lane_bases", ())]),
        ("warp", [list(basis) for basis in properties.get("warp_bases", ())]),
        ("block", [list(basis) for basis in properties.get("block_bases", ())]),
    ]
    for in_dim, in_bases in bases:
        for basis in in_bases:
            if len(basis) != rank:
                _layout_fail(
                    "TLXW_TYPE_MALFORMED_LAYOUT",
                    stage,
                    f"linear layout {in_dim} basis rank does not match tensor rank",
                    source_op_index=source_op_index,
                    source_value_id=source_value_id,
                )
    construction_shape = tuple(
        max(int(shape[dim]), _minimum_linear_basis_extent(bases, dim)) for dim in range(rank))
    linear = LinearLayout.from_bases(bases, out_dims, list(construction_shape), False)
    return _ensure_layout_matches_shape(
        linear,
        shape,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def _minimum_linear_basis_extent(bases, dim):
    extent = 1
    for _in_dim, in_bases in bases:
        for basis in in_bases:
            value = int(basis[int(dim)])
            if value:
                extent = max(extent, 1 << int(value).bit_length())
    return extent


def _slice_linear_layout(
    shape,
    properties,
    lane_width,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    parent_kind = properties.get("parent_kind")
    parent_properties = properties.get("parent_properties", {})
    if parent_kind not in {"blocked", "linear", "generic_linear", "slice", "amd_mfma"}:
        _layout_fail(
            "TLXW_TYPE_UNSUPPORTED_LAYOUT",
            stage,
            f"slice parent layout {parent_kind} does not have a distributed register map",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    dim = int(properties.get("dim", 0))
    shape = tuple(int(value) for value in shape)
    parent_shape = list(shape)
    if dim < 0 or dim > len(parent_shape):
        _layout_fail(
            "TLXW_TYPE_MALFORMED_LAYOUT",
            stage,
            "slice layout dimension is outside the parent rank",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if parent_kind in {"linear", "generic_linear"}:
        return _project_explicit_linear_slice_layout(
            shape,
            parent_properties,
            dim,
            stage=stage,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    parent_shape.insert(dim, 1)
    parent = distributed_linear_layout_from_parts(
        parent_kind,
        tuple(parent_shape),
        parent_properties,
        lane_width,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )
    parent_dim_to_basis_index = {str(name): index for index, (name, _size) in enumerate(parent.out_dims)}
    kept_parent_dims = [f"dim{index}" for index in range(len(parent_shape)) if index != dim]
    out_dims = [f"dim{index}" for index in range(len(shape))]
    bases = []
    for in_dim, in_bases in parent.bases:
        projected = []
        for basis in in_bases:
            basis = tuple(int(value) for value in basis)
            projected.append([basis[parent_dim_to_basis_index[parent_dim]] for parent_dim in kept_parent_dims])
        bases.append((in_dim, projected))
    return LinearLayout.from_bases(bases, out_dims, list(shape), False)


def _project_explicit_linear_slice_layout(
    shape,
    parent_properties,
    dim,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    shape = tuple(int(value) for value in shape)
    parent_rank = len(shape) + 1
    dim = int(dim)
    out_dims = [f"dim{index}" for index in range(len(shape))]
    bases = []
    for in_dim in ("register", "lane", "warp", "block"):
        projected = []
        for basis in parent_properties.get(f"{in_dim}_bases", ()):
            basis = tuple(int(value) for value in basis)
            if len(basis) != parent_rank:
                _layout_fail(
                    "TLXW_TYPE_MALFORMED_LAYOUT",
                    stage,
                    f"slice parent linear layout {in_dim} basis rank does not match parent rank",
                    source_op_index=source_op_index,
                    source_value_id=source_value_id,
                )
            projected_basis = [basis[index] for index in range(parent_rank) if index != dim]
            if in_dim == "register" and not any(projected_basis) and any(basis):
                continue
            projected.append(projected_basis)
        bases.append((in_dim, projected))
    return LinearLayout.from_bases(bases, out_dims, list(shape), False)


def _mfma_linear_layout(
    shape,
    properties,
    lane_width,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    if len(shape) != 2:
        _layout_fail(
            "TLXW_TYPE_UNSUPPORTED_LAYOUT",
            stage,
            "MFMA distributed layout currently requires rank-2 tensors",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    instr_shape = tuple(int(value) for value in properties.get("instr_shape", ()))
    if instr_shape not in {(16, 16, 32), (16, 16, 128), (32, 32, 16)}:
        _layout_fail(
            "TLXW_TYPE_UNSUPPORTED_LAYOUT",
            stage,
            f"unsupported MFMA instruction shape {instr_shape}",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    element_bit_width = int(properties.get("element_bit_width", 32))
    height = 1 if element_bit_width == 64 else 4
    m_dim, n_dim = int(instr_shape[0]), int(instr_shape[1])
    warp_size = int(lane_width)
    tiles = (m_dim * n_dim) // (warp_size * height)
    if tiles <= 0:
        _layout_fail(
            "TLXW_TYPE_UNSUPPORTED_LAYOUT",
            stage,
            "MFMA distributed layout requires at least one register tile",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    dim_m = "dim0"
    dim_n = "dim1"
    if bool(properties.get("is_transposed", False)):
        linear = LinearLayout.identity_1d(height, "register", dim_n)
        linear *= (LinearLayout.identity_1d(m_dim, "lane", dim_m) *
                   LinearLayout.identity_1d(warp_size // m_dim, "lane", dim_n))
        linear *= LinearLayout.identity_1d(tiles, "register", dim_n)
    else:
        linear = LinearLayout.identity_1d(height, "register", dim_m)
        linear *= (LinearLayout.identity_1d(n_dim, "lane", dim_n) *
                   LinearLayout.identity_1d(warp_size // n_dim, "lane", dim_m))
        linear *= LinearLayout.identity_1d(tiles, "register", dim_m)
    linear = _linear_layout_transpose_outs(linear, (dim_n, dim_m))
    tiles_per_warp = tuple(int(value) for value in properties.get("tiles_per_warp", ()))
    if len(tiles_per_warp) < 2:
        tiles_per_warp = (1, 1)
    warps_per_cta = tuple(int(value) for value in properties.get("warps_per_cta", ()))
    if len(warps_per_cta) != 2:
        _layout_fail(
            "TLXW_TYPE_MALFORMED_LAYOUT",
            stage,
            "MFMA distributed layout requires rank-2 warpsPerCTA metadata",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    tiles_per_warp_m = max(1, int(tiles_per_warp[0]))
    tiles_per_warp_n = max(1, int(tiles_per_warp[1]))
    warps_per_cta_m = max(1, int(warps_per_cta[0]))
    warps_per_cta_n = max(1, int(warps_per_cta[1]))
    linear *= LinearLayout.identity_1d(tiles_per_warp_n, "register", dim_n)
    linear *= LinearLayout.identity_1d(warps_per_cta_n, "warp", dim_n)
    n_remainder = shape[1] // (n_dim * warps_per_cta_n * tiles_per_warp_n)
    linear *= LinearLayout.identity_1d(max(1, n_remainder), "register", dim_n)
    linear *= LinearLayout.identity_1d(tiles_per_warp_m, "register", dim_m)
    linear *= LinearLayout.identity_1d(warps_per_cta_m, "warp", dim_m)
    return _ensure_layout_matches_shape(
        linear,
        shape,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def _linear_layout_transpose_outs(linear, out_dim_names):
    out_dim_names = tuple(str(name) for name in out_dim_names)
    old_out_dims = tuple((str(name), int(size)) for name, size in linear.out_dims)
    old_index = {name: index for index, (name, _size) in enumerate(old_out_dims)}
    old_size = {name: size for name, size in old_out_dims}
    bases = []
    for in_dim, in_bases in linear.bases:
        bases.append((
            str(in_dim),
            [[int(basis[old_index[name]]) for name in out_dim_names] for basis in in_bases],
        ))
    return LinearLayout.from_bases(
        bases,
        list(out_dim_names),
        [old_size[name] for name in out_dim_names],
        False,
    )


def _identity_standard_nd(in_dim, shape, order):
    linear = LinearLayout()
    for dim in order:
        linear *= LinearLayout.identity_1d(int(shape[dim]), in_dim, f"dim{dim}")
    return linear


def _ensure_layout_matches_shape(
    linear,
    shape,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    shape_by_dim = {f"dim{dim}": int(size) for dim, size in enumerate(shape)}
    linear = _ensure_layout_not_smaller_than(
        linear,
        shape_by_dim,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )
    linear = _ensure_layout_not_larger_than(linear, shape_by_dim)
    # Triton's combineCtaCgaWithShape canonicalizes distributed layouts to
    # standard tensor-dimension order after applying the shape.  Keep the
    # bridge model identical: physical swizzle planning flattens output dims,
    # so retaining an encoding's construction order changes its bit bases.
    standard_out_dims = tuple(f"dim{dim}" for dim in range(len(shape)))
    if tuple(str(name) for name, _size in linear.out_dims) != standard_out_dims:
        linear = _linear_layout_transpose_outs(linear, standard_out_dims)
    return linear


def _ensure_layout_not_smaller_than(
    linear,
    shape_by_dim,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    for dim, desired_size in shape_by_dim.items():
        actual_size = linear_layout_out_dim_size(linear, dim, stage=stage)
        if desired_size > actual_size:
            if desired_size % actual_size:
                _layout_fail(
                    "TLXW_TYPE_UNSUPPORTED_LAYOUT",
                    stage,
                    "layout remap requires non-integral shape extension",
                    source_op_index=source_op_index,
                    source_value_id=source_value_id,
                )
            linear *= LinearLayout.identity_1d(
                desired_size // actual_size,
                "register",
                dim,
            )
    return linear


def _ensure_layout_not_larger_than(linear, shape_by_dim):
    out_dims = []
    out_sizes = []
    for dim, size in linear.out_dims:
        resized = min(int(size), int(shape_by_dim[dim]))
        out_dims.append(dim)
        out_sizes.append(resized)
    bases = []
    for in_dim, in_bases in linear.bases:
        rewritten = []
        for basis in in_bases:
            basis = [int(value) for value in basis]
            was_zero = all(value == 0 for value in basis)
            for index, value in enumerate(tuple(basis)):
                if value >= out_sizes[index]:
                    basis[index] = 0
            is_zero = all(value == 0 for value in basis)
            if in_dim == "register":
                if was_zero or not is_zero:
                    rewritten.append(basis)
            else:
                rewritten.append(basis)
        bases.append((in_dim, rewritten))
    return LinearLayout.from_bases(
        bases,
        out_dims,
        out_sizes,
        False,
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


def _per_wave_mfma_tiles(shape, instr_shape, warps_per_cta):
    total_m_tiles = _ceil_div(int(shape[0]), int(instr_shape[0]))
    total_n_tiles = _ceil_div(int(shape[1]), int(instr_shape[1]))
    warps_m = max(1, int(warps_per_cta[0]))
    warps_n = max(1, int(warps_per_cta[1]))
    return _ceil_div(total_m_tiles, warps_m), _ceil_div(total_n_tiles, warps_n)


def mfma_cta_tile_coordinate(per_wave_tile, wave_coordinate, wave_count, tiles_per_warp):
    """Map a per-wave MFMA tile index to its CTA tile coordinate.

    Triton assigns ``tiles_per_warp`` adjacent instruction tiles to a wave,
    then advances the wave coordinate, and only then repeats the pattern for
    larger tensors.  With the default value of one this reduces to the usual
    strided ``tile * wave_count + wave`` mapping.
    """
    per_wave_tile = int(per_wave_tile)
    wave_coordinate = int(wave_coordinate)
    wave_count = max(1, int(wave_count))
    tiles_per_warp = max(1, int(tiles_per_warp))
    local_tile = per_wave_tile % tiles_per_warp
    repeat = per_wave_tile // tiles_per_warp
    return local_tile + tiles_per_warp * (wave_coordinate + wave_count * repeat)


def _mfma_component_count(shape, properties, lane_width, *, source_value_id=None):
    del lane_width, source_value_id
    shape = tuple(int(value) for value in shape)
    instr_shape = properties.get("instr_shape", ())
    warps_per_cta = properties.get("warps_per_cta", ())
    if len(instr_shape) >= 2 and len(warps_per_cta) >= 2 and len(shape) >= 2:
        m_tiles, n_tiles = _per_wave_mfma_tiles(
            shape,
            instr_shape,
            warps_per_cta,
        )
        return m_tiles * n_tiles
    return 1


def _per_wave_tile_count(extent, instr_extent, warp_extent):
    return _ceil_div(_ceil_div(int(extent), int(instr_extent)), max(1, int(warp_extent)))


def _ceil_div(lhs, rhs):
    return (int(lhs) + int(rhs) - 1) // int(rhs)
