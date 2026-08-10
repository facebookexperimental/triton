"""Coordinate-domain classification for distributed LinearLayouts."""


def _in_dim_size(linear, dim):
    for name, bases in linear.bases:
        if name == dim:
            return 1 << len(bases)
    return 1


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _is_power_of_two(value):
    value = int(value)
    return value > 0 and value & (value - 1) == 0


def classify_coordinate_domain(shape, lane_width, linear):
    """Classify the exact ownership coverage of a distributed layout."""
    component_count = _in_dim_size(linear, "register")
    warp_count = _in_dim_size(linear, "warp")
    block_count = _in_dim_size(linear, "block")
    shape = tuple(int(dim) for dim in shape)
    total_elements = _product(shape)
    physical_slots = int(component_count) * int(lane_width) * int(warp_count)
    if int(block_count) <= 0 or total_elements % int(block_count):
        coverage = "block_mismatch"
        local_elements = total_elements
        covered_elements = 0
        duplicate_slots = 0
    else:
        local_elements = total_elements // int(block_count)
        modular = any(not _is_power_of_two(extent) for _name, extent in linear.out_dims)
        if modular:
            # Triton can represent modular LinearLayouts, but its injectivity
            # query is defined only for binary-coordinate domains.
            coverage = "modular"
            covered_elements = 0
            duplicate_slots = 0
        else:
            surjective = bool(linear.is_surjective())
            injective = bool(linear.is_injective())
            if surjective and injective and physical_slots == local_elements:
                coverage = "exact"
            elif surjective:
                coverage = "replicated"
            elif injective:
                coverage = "partial"
            else:
                coverage = "duplicate_partial"
            covered_elements = local_elements if surjective else 0
            duplicate_slots = (max(0, physical_slots - local_elements) if surjective else 0)
    return {
        "coverage": coverage,
        "component_count": int(component_count),
        "covered_elements": int(covered_elements),
        "duplicate_slots": int(duplicate_slots),
        "local_elements": int(local_elements),
        "physical_slots": int(physical_slots),
        # LinearLayout construction constrains each output basis to its
        # declared dimension, so there is no separate out-of-bounds image.
        "out_of_bounds_slots": 0,
        "block_count": int(block_count),
    }
