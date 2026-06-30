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


def leaf_columns(ev, du, load_inst, slot):
    """Layout-invariant element-index image of one (vector-)load across the WHOLE thread grid.

    The load address is an :class:`Affine` ``const + sum(coeff*sym)``. The columns this load
    reads, as the thread index ranges over the launch grid, are ``{(numeric_part)/width}``
    expanded over every ``%tid`` symbol's range. Symbolic terms that are CONSTANT across a
    single reduction (the base pointer param, the row ``%ctaid`` offset) are dropped — they are
    identical for every config compared and for every leaf of one row, so dropping them yields
    a layout-invariant per-row column set. Returns ``frozenset[int]`` or ``None`` when the
    image cannot be proven (non-affine address, unknown ``%tid`` range, or a non-tid varying
    symbol), in which case the collapse pass must NOT fire (stays conservative)."""
    at = du.index_of(load_inst)
    addr = ev.of_address(load_inst.operands[1], at)
    width = _load_width(load_inst.modifiers)
    if not isinstance(addr, Affine) or width is None or width == 0:
        return None
    base = addr.const + (slot * width if slot else 0)
    grid = []  # (coeff, n) for each thread-index symbol that selects a column
    for sym, coeff in addr.terms:
        if sym.startswith("%tid"):
            n = ev.reqntid.get(sym.split(".")[-1])
            if n is None:
                return None
            grid.append((coeff, n))
        elif sym.startswith("%laneid"):
            grid.append((coeff, 32))
        elif (sym.startswith("%ctaid") or sym.startswith("%nctaid") or sym.startswith("%ntid")
              or sym.startswith("param:") or sym.startswith("sym:") or sym.startswith("reg:")
              or sym.startswith("opq:")):
            # constant across this reduction (row index / base ptr / unmodellable row math)
            # -> drop; it is identical for every config and every leaf of one row.
            continue
        else:
            return None  # an unknown varying symbol -> cannot prove the image; be conservative
    cols = {base}
    for coeff, n in grid:
        cols = {c + coeff * t for c in cols for t in range(n)}
        if len(cols) > 1 << 20:  # safety cap; refuse to materialize an absurd image
            return None
    if any(c % width for c in cols):
        return None
    return frozenset(c // width for c in cols)
