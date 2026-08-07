"""Layout-canonical leaf coordinates for AMDGCN loads — the AMD peer of ``bitequiv.ptx.leaves``.

A reduction-tree leaf is a loaded tensor element. Its ``coord`` is a layout-invariant label (the
element each thread reads), and its ``cols`` is the element-index IMAGE of the load across the whole
thread grid (used by the balanced-collapse pass to prove two configs reduce the SAME element set).

AMD addresses live in registers (``buffer_load`` voffset / ``global_load`` vaddr), not in a
``[base+off]`` operand, so we evaluate the offset REGISTER through the affine engine. The coord keeps
only the ``%tid``/``%laneid``-varying part + the numeric per-load constant (+ ``slot*width`` for a
sub-register of a vector load) and DROPS block-uniform row bases (``param:``/``vec:``/``opq:``…),
which differ only by register allocation and would otherwise split every config into a singleton.
"""
from __future__ import annotations

from bitequiv.amdgcn.affine import canon
from bitequiv.amdgcn.core.affine_algebra import Affine

# (element byte width, n sub-register slots) per load mnemonic suffix.
_LOAD_SHAPE = {
    "dwordx4": (4, 4), "dwordx3": (4, 3), "dwordx2": (4, 2), "dword": (4, 1),
    "ushort": (2, 1), "short_d16": (2, 1), "short_d16_hi": (2, 1), "ubyte": (1, 1), "sbyte": (1, 1),
    "b128": (4, 4), "b96": (4, 3), "b64": (4, 2), "b32": (4, 1), "b16": (2, 1), "b8": (1, 1),
}

_LANEID_RANGE = 64  # wave64


def load_shape(opcode):
    """(element byte width, n slots) of a load opcode, or ``(None, 1)`` if unrecognized."""
    for suf, shape in _LOAD_SHAPE.items():
        if opcode.endswith(suf) or opcode.endswith(suf + "_e32") or opcode.endswith(suf + "_e64"):
            return shape
    return (None, 1)


def _offset_operand(load_inst):
    """The address/offset operand of a load (``buffer_load`` voffset / ``global_load`` vaddr =
    operand[1]); ``None`` if absent."""
    return load_inst.operands[1] if len(load_inst.operands) > 1 else None


def _tid_range(sym, reqntid):
    """The value range of a thread-index symbol: a ``.bit<i>`` symbol is [0,2); a plain
    ``%tid.<dim>`` uses the workgroup size; ``%laneid`` is [0,64). ``None`` for a non-tid / unknown
    varying symbol (caller stays conservative)."""
    parts = sym.split(".")
    if parts[-1].startswith("bit") and parts[0] in ("%tid", "%laneid"):
        return 2
    if parts[0] == "%tid" and len(parts) == 2:
        return reqntid.get(parts[1])
    if parts[0] == "%laneid":
        return _LANEID_RANGE
    return None


def _is_row_constant(sym):
    return sym.startswith(("opq:", "param:", "reg:", "vec:", "sym:", "%ctaid", "%nctaid", "%ntid"))


def leaf_coord(ev, du, load_inst, slot):
    """Layout-invariant element label for one loaded element (sub-register ``slot`` of a vector
    load). Keeps only the ``%tid``/``%laneid`` coefficients + numeric const (+ ``slot*width``);
    drops row-constant bases. Non-affine address -> a conservative opaque coord string."""
    at = du.index_of(load_inst)
    off = _offset_operand(load_inst)
    addr = ev.of_operand(off, at) if off is not None else None
    width, _ = load_shape(load_inst.opcode)
    wtag = "" if width is None else f"/{width}"
    if isinstance(addr, Affine):
        const = addr.const + (slot * width if (width is not None and slot) else 0)
        tid_terms = sorted((s, c) for s, c in addr.terms if s.startswith(("%tid", "%laneid")))
        body = "+".join(f"{c}*{s}" for s, c in tid_terms)
        coord = f"{const}+{body}" if body else str(const)
        return f"{coord}{wtag}"
    coord = canon(addr) if addr is not None else "?"
    return f"{coord}+slot{slot}{wtag}" if slot else f"{coord}{wtag}"


def leaf_columns(ev, du, load_inst, slot):
    """Layout-invariant element-index IMAGE of the load across the whole grid, or ``None`` if not
    recoverable. Expands each ``%tid``/``%laneid`` symbol over its range (bit->2, plain->workgroup,
    laneid->64), drops row-constants; bails on any unknown varying symbol or a mis-aligned address.
    Must be evaluated with the absorb-opaque evaluator so an opaque row base folds into a droppable
    symbol rather than poisoning the whole address."""
    at = du.index_of(load_inst)
    off = _offset_operand(load_inst)
    addr = ev.of_operand(off, at) if off is not None else None
    width, _ = load_shape(load_inst.opcode)
    if not isinstance(addr, Affine) or width is None or width == 0:
        return None
    base = addr.const + (slot * width if slot else 0)
    grid = []
    for sym, coeff in addr.terms:
        if _is_row_constant(sym):
            continue
        n = _tid_range(sym, ev.reqntid)
        if n is None:
            return None
        grid.append((coeff, n))
    cols = {base}
    for coeff, n in grid:
        cols = {c + coeff * t for c in cols for t in range(n)}
        if len(cols) > (1 << 20):
            return None
    if any(c % width for c in cols):
        return None
    return frozenset(c // width for c in cols)
