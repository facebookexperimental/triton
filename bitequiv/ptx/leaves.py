"""Leaf labeling — the PTX layout analog.

For each global load that feeds the reduction tree, recover the tensor element it reads
as a canonical coordinate: the load address evaluated to an :class:`~bitequiv.ptx.affine.
Affine` over ``tid``/``ctaid``/params (the PTX echo of the LinearLayout bases), plus the
vector-element slot and whether the load is masked. Two configs that map the same
``(lane, warp, block, slot)`` to the same element produce identical coordinates; a
different ``sizePerThread`` / ``num_warps`` / transpose produces a different one. This is
the PTX counterpart of TTGIR's ``sublayout({register,lane,warp,block},{dim<axis>})``.
"""

from bitequiv.ptx.affine import Affine, canon

# Byte widths for the load-result types we may see on a reduction's leaf loads.
_WIDTH = {
    ".b8": 1,
    ".b16": 2,
    ".b32": 4,
    ".b64": 8,
    ".b128": 16,
    ".f16": 2,
    ".f32": 4,
    ".f64": 8,
    ".u8": 1,
    ".u16": 2,
    ".u32": 4,
    ".u64": 8,
    ".s8": 1,
    ".s16": 2,
    ".s32": 4,
    ".s64": 8,
}


def _load_width(modifiers):
    for m in reversed(modifiers):
        if m in _WIDTH:
            return _WIDTH[m]
    return None


def leaf_coord(ev, du, load_inst, slot):
    """Canonical element-coordinate string for one (vector-)load destination.

    ``ev`` is an :class:`AffineEval`, ``du`` a :class:`DefUse`, ``load_inst`` the
    ``ld.global`` instruction, ``slot`` the vector-element index (0 for a scalar load)."""
    at = du.index_of(load_inst)
    addr = ev.of_address(load_inst.operands[1], at)
    width = _load_width(load_inst.modifiers)
    # Element byte-address = base address + slot * element width (v4 loads are contiguous).
    if isinstance(addr, Affine) and width is not None and slot:
        addr = Affine(addr.const + slot * width, addr.terms)
    coord = canon(addr) if isinstance(addr, Affine) else canon(addr)
    if not isinstance(addr, Affine) and slot:
        coord = f"{coord}+slot{slot}"
    masked = "m" if load_inst.predicate is not None else "u"
    w = "" if width is None else f"/{width}"
    return f"{coord}{w}|{masked}"
