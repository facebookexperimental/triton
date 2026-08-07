"""Conservative symbolic evaluator for AMDGCN integer address registers — the AMD peer of
``bitequiv.ptx.affine``'s ``AffineEval``.

Given the def-use index, evaluate an address-computing register to an :class:`Affine` over a
fixed symbol basis (``%tid.x`` from ``v0``, kernel ``param:`` bases, base-pointer pairs) or to
:class:`Opaque` when it cannot be *proven* affine. The symbolic ALGEBRA (``Affine`` / ``Opaque``
/ ``canon`` / range math) is the ISA-neutral core (:mod:`bitequiv.amdgcn.core.affine_algebra`);
only the opcode dispatch, the ``v0``->``%tid.x`` basis, and the wave-size are AMD-specific.

Two AMD facts vs PTX:
  * there is no ``%tid.x`` register READ — the flat workitem id sits in ``v0`` at kernel entry,
    so an unwritten ``v0`` IS ``%tid.x`` (its low 6 bits are the wavefront lane, the high bits
    the wave index within the block — wave64);
  * addresses are computed in registers (``buffer_load`` voffset / ``global_load`` vaddr pair),
    there is no ``[base+offset]`` operand — so the leaf/exchange code evaluates the offset
    register directly through ``of_operand``.

Conservative floor: only a fixed opcode set is modeled; anything else (and every fp / DPP /
shuffle op, which is a value not an address) becomes ``Opaque``, so two addresses compare equal
only when provably the same computation — it over-splits, never over-merges.
"""
from __future__ import annotations

from bitequiv.amdgcn.core.affine_algebra import (
    Affine,
    Opaque,
    _add_terms,
    _const,
    _is_pow2,
    _parse_int,
    _rng_add,
    _rng_scale,
    _set_bits_mask,
    _symbol,
    canon,
)
from bitequiv.amdgcn.parser import ImmediateOperand, RegisterOperand, VectorOperand

# AMD integer-arithmetic opcode groups (base mnemonic, size suffix already stripped). From the
# validated backward AMD checker's affine dispatch. Reversed-operand shifts are separated out
# (``v_lshlrev_b32 dst, shift, val`` == ``val << shift``).
_CO_ADD = frozenset({"v_add_co_u32", "v_add_co_ci_u32", "v_addc_co_u32", "v_add_co_u32_dpp"})
_CO_SUB = frozenset({"v_sub_co_u32", "v_subb_co_u32", "v_subrev_co_u32"})
_ADD = frozenset({"v_add_u32", "v_add_nc_u32", "v_add_i32", "s_add_u32", "s_add_i32", "s_addk_i32"})
_SUB = frozenset({"v_sub_u32", "v_sub_nc_u32", "v_sub_i32", "s_sub_u32", "s_sub_i32"})
_MUL = frozenset({"v_mul_lo_u32", "v_mul_lo_i32", "v_mul_u32_u24", "v_mul_i32_i24", "s_mul_i32"})
_SHL = frozenset({"v_lshl_b32", "s_lshl_b32", "s_lshl_b64", "v_lshl_b64"})
_SHR = frozenset({"v_lshr_b32", "s_lshr_b32", "s_lshr_b64", "v_lshr_b64"})
_SHLREV = frozenset({"v_lshlrev_b32", "v_lshlrev_b64"})
_SHRREV = frozenset({"v_lshrrev_b32", "v_lshrrev_b64"})
_AND = frozenset({"v_and_b32", "s_and_b32", "s_and_b64"})
_ORXOR = frozenset({"v_or_b32", "s_or_b32", "v_xor_b32", "s_xor_b32", "v_or3_b32"})
_MOV = frozenset({"v_mov_b32", "s_mov_b32", "s_movk_i32"})  # s_movk_i32 dst, simm16 = a 16-bit
#   constant materialize; its source operand IS the immediate, so the mov path returns it. Leaving it
#   opaque made every `and(addr, MASK)` where MASK came from s_movk (the LDS swizzle masks) fall to
#   opaque -> the cross-warp ds_read address never resolved at low num_warps.
_MAD = frozenset({"v_mad_u32_u24", "v_mad_i32_i24", "v_mad_u64_u32"})

_MAX_IMAGE = 1 << 16  # cap on grid addresses thread_image materializes
_WARP_BITS = 6  # wave64: %tid.x bits [0,6) are the lane, [6,..) the wave index within the block


def _lane_bit(sym):
    """The bit index if ``sym`` is a ``%tid.<dim>.bit<i>`` symbol, else ``None``."""
    parts = sym.split(".")
    if len(parts) == 3 and parts[0] == "%tid" and parts[2].startswith("bit"):
        try:
            return int(parts[2][3:])
        except ValueError:
            return None
    return None


def _strip_size(op):
    """Drop the encoding-size / DPP suffix so the dispatch keys on the base mnemonic."""
    for suf in ("_e64_dpp", "_e32_dpp", "_dpp", "_e32", "_e64"):
        if op.endswith(suf):
            return op[:-len(suf)]
    return op


def reqntid_of(func):
    """The launch geometry the parser recovered from ``.amdgpu_metadata`` — ``{'x': flat_wg}``.
    num_warps = flat_wg // 64 (wave64); the affine bit-basis expands ``%tid.x`` over it."""
    rq = getattr(func, "reqntid", None)
    return dict(rq) if rq else {}


class AffineEval:
    """Evaluate integer registers to :class:`Affine` / :class:`Opaque`, memoized per def."""

    def __init__(self, defuse, reqntid=None, absorb_opaque=False):
        self.du = defuse
        self.reqntid = reqntid or {}
        self._memo = {}
        self._stack = set()
        # When True an unmodellable value becomes a fresh DROPPABLE symbol ("opq:N") instead of a
        # poisoning Opaque — used only by the column-image extractor (leaf_columns), which keeps
        # the %tid-linear part and drops opq:/param:/vec: symbols.
        self.absorb_opaque = absorb_opaque
        self._opq = 0

    # -- operand entry points -------------------------------------------------

    def of_operand(self, operand, before_index):
        if isinstance(operand, ImmediateOperand):
            v = _parse_int(operand.text)
            return _const(v) if v is not None else Opaque(f"imm({operand.text})")
        if isinstance(operand, RegisterOperand):
            return self.of_reg(operand.name, before_index)
        if isinstance(operand, VectorOperand):
            # A register range is a 64-bit base-pointer pair (or a value vector); treat it as a
            # stable row-constant symbol — it never carries the tid-varying column offset.
            return _symbol("vec:" + (operand.text or ",".join(e.name for e in operand.elements)))
        return Opaque(f"operand({type(operand).__name__})")

    def of_reg(self, name, before_index):
        d = self.du.last_writer(name, before_index)
        if d is None:
            # Hardware-initialized / kernel input. v0 = flat workitem id (%tid.x); its low 6 bits
            # are the wave64 lane, the rest the wave index. Any other unwritten reg = row constant.
            if name == "v0":
                n = self.reqntid.get("x")
                return _symbol("%tid.x", rng=(0, n) if n else None)
            return _symbol("reg:" + name)
        if id(d) in self._memo:
            return self._memo[id(d)]
        if id(d) in self._stack:
            return Opaque("cycle:" + name)
        self._stack.add(id(d))
        val = self._eval_def(d)
        self._stack.discard(id(d))
        self._memo[id(d)] = val
        return val

    # -- core evaluation ------------------------------------------------------

    def _eval_def(self, d):
        inst = d.inst
        raw = inst.opcode
        at = d.index
        ops = inst.operands
        srcs = ops[1:]

        def ev(o):
            return self.of_operand(o, at)

        # A floating-point / packed / DPP / shuffle op is a VALUE, not an address -> opaque.
        if raw.endswith("_dpp") or any(w in raw for w in ("_f32", "_f16", "_f64", "_bf16")):
            return self._opaque_def(d)
        op = _strip_size(raw)

        if op in _MOV and srcs:
            return ev(srcs[0])
        if op.startswith("s_load") or op.startswith("s_buffer_load"):  # kernarg -> param symbol
            base = getattr(srcs[0], "name", None) or getattr(srcs[0], "text", "?") if srcs else "?"
            return _symbol("param:" + str(base))
        if op in _CO_ADD:  # carry-out dst is operand[0]; the addends are operands[1:] after it
            vs = ops[2:]
            return self._binop_add(ev(vs[0]), ev(vs[1]), 1) if len(vs) >= 2 else self._opaque_def(d)
        if op in _CO_SUB:
            vs = ops[2:]
            return self._binop_add(ev(vs[0]), ev(vs[1]), -1) if len(vs) >= 2 else self._opaque_def(d)
        if op in _ADD and len(srcs) >= 2:
            return self._binop_add(ev(srcs[0]), ev(srcs[1]), 1)
        if op in _SUB and len(srcs) >= 2:
            return self._binop_add(ev(srcs[0]), ev(srcs[1]), -1)
        if op in _MUL and len(srcs) >= 2:
            return self._mul(ev(srcs[0]), ev(srcs[1]))
        if op in _MAD and len(srcs) >= 3:  # a*b + c
            return self._binop_add(self._mul(ev(srcs[0]), ev(srcs[1])), ev(srcs[2]), 1)
        if op == "v_add_lshl_u32" and len(srcs) >= 3:  # (s0 + s1) << s2
            return self._shl(self._binop_add(ev(srcs[0]), ev(srcs[1]), 1), ev(srcs[2]))
        if op == "v_lshl_add_u32" and len(srcs) >= 3:  # (s0 << s1) + s2
            return self._binop_add(self._shl(ev(srcs[0]), ev(srcs[1])), ev(srcs[2]), 1)
        if op.startswith("s_lshl") and op.endswith("_add_u32") and len(op) > 7 and op[6].isdigit() and len(srcs) >= 2:
            n = int(op[6])  # s_lshlN_add_u32 dst, a, b = (a << N) + b
            return self._binop_add(self._shl(ev(srcs[0]), _const(n)), ev(srcs[1]), 1)
        if op == "v_lshl_or_b32" and len(srcs) >= 3:  # (s0 << s1) | s2
            return self._or_xor(self._shl(ev(srcs[0]), ev(srcs[1])), ev(srcs[2]))
        if op in ("v_add3_u32", "v_add3_nc_u32") and len(srcs) >= 3:  # s0 + s1 + s2
            return self._binop_add(self._binop_add(ev(srcs[0]), ev(srcs[1]), 1), ev(srcs[2]), 1)
        if op == "v_and_or_b32" and len(srcs) >= 3:  # (s0 & s1) | s2
            return self._or_xor(self._and(ev(srcs[0]), ev(srcs[1])), ev(srcs[2]))
        if op in _SHLREV and len(srcs) >= 2:  # dst = src1 << src0
            return self._shl(ev(srcs[1]), ev(srcs[0]))
        if op in _SHRREV and len(srcs) >= 2:  # dst = src1 >> src0
            return self._shr(ev(srcs[1]), ev(srcs[0]))
        if op in _SHL and len(srcs) >= 2:
            return self._shl(ev(srcs[0]), ev(srcs[1]))
        if op in _SHR and len(srcs) >= 2:
            return self._shr(ev(srcs[0]), ev(srcs[1]))
        if op in _AND and len(srcs) >= 2:
            return self._and(ev(srcs[0]), ev(srcs[1]))
        if op in _ORXOR and len(srcs) >= 2:
            return self._or_xor(ev(srcs[0]), ev(srcs[1]))
        if op == "v_bfe_u32" and len(srcs) >= 3:  # unsigned bit-field extract (signed sign-extends -> opaque)
            return self._bfe(ev(srcs[0]), ev(srcs[1]), ev(srcs[2]))
        if op == "v_cndmask_b32" and len(srcs) >= 2:
            # masked-address select dst = cond ? src1 : src0: one side is the real affine address,
            # the other a poison constant. Take the affine side.
            a, b = ev(srcs[0]), ev(srcs[1])
            if isinstance(a, Affine) and a.terms and not (isinstance(b, Affine) and b.terms):
                return a
            if isinstance(b, Affine) and b.terms:
                return b
            return a if isinstance(a, Affine) else b
        if op == "v_readfirstlane_b32" and srcs:  # value at lane 0 = wave-uniform (lane bits cleared)
            return self._wave_uniform(ev(srcs[0]))
        if op == "v_readlane_b32" and len(srcs) >= 2 and isinstance(srcs[1], ImmediateOperand):
            lane = _parse_int(srcs[1].text)  # value at lane N = wave base + N (in lane units)
            base = self._wave_uniform(ev(srcs[0]))
            if lane is not None and isinstance(base, Affine):
                return self._binop_add(base, _const(lane), 1)
        return self._opaque_def(d)

    # -- modeled operators (ISA-neutral algebra; identical to the PTX evaluator) ---------------

    def _binop_add(self, a, b, sign):
        if isinstance(a, Opaque) or isinstance(b, Opaque):
            return self._opaque_pair(a, "add" if sign > 0 else "sub", b)
        terms = _add_terms(a, b, sign)
        return Affine(a.const + sign * b.const, terms, rng=_rng_add(a.rng, b.rng, sign))

    def _mul(self, a, b):
        ca = a if isinstance(a, Affine) and not a.terms else None
        cb = b if isinstance(b, Affine) and not b.terms else None
        if isinstance(a, Affine) and cb is not None:
            return self._scale(a, cb.const)
        if isinstance(b, Affine) and ca is not None:
            return self._scale(b, ca.const)
        return self._opaque_pair(a, "mul", b)

    def _scale(self, a, k):
        terms = frozenset((s, c * k) for s, c in a.terms) if k != 0 else frozenset()
        return Affine(a.const * k, terms, rng=_rng_scale(a.rng, k))

    def _shl(self, a, b):
        if isinstance(a, Affine) and isinstance(b, Affine) and not b.terms:
            return self._scale(a, 1 << b.const)
        return self._opaque_pair(a, "shl", b)

    def _shr(self, a, b):
        if isinstance(a, Affine) and isinstance(b, Affine) and not b.terms:
            s = b.const
            if a.tz >= s:
                terms = frozenset((sym, c >> s) for sym, c in a.terms)
                rng = None if a.rng is None else (a.rng[0] >> s, ((a.rng[1] - 1) >> s) + 1)
                return Affine(a.const >> s, terms, rng=rng)
            exp = self._tid_bit_expand(a)
            pos = self._bit_positions(exp) if exp is not None else None
            if pos is not None and s >= 0:
                kept = frozenset((sym, 1 << (p - s)) for p, sym in pos.items() if p >= s)
                hi = sum(1 << (p - s) for p in pos if p >= s) + 1
                return Affine(0, kept, rng=(0, hi))
        return self._opaque_pair(a, "shr", b)

    def _tid_bit_expand(self, aff):
        """Rewrite an integer affine into the ``%tid`` BIT basis (each ``%tid.<dim>`` term ->
        ``sum_i (coeff*2**i) %tid.<dim>.bit<i>`` over ``i in [0, log2 N)``, ``N = reqntid[dim]``).
        Exact because ``N`` is a power of two. ``None`` if a term is not so decomposable."""
        if not isinstance(aff, Affine) or aff.const != 0:
            return None
        terms = {}
        for sym, coeff in aff.terms:
            parts = sym.split(".")
            is_bit = len(parts) == 3 and parts[0] == "%tid" and parts[2].startswith("bit")
            is_plain = len(parts) == 2 and parts[0] == "%tid"
            if is_bit:
                terms[sym] = terms.get(sym, 0) + coeff
                continue
            if not is_plain:
                return None
            n = self.reqntid.get(parts[1])
            if not _is_pow2(n):
                return None
            for i in range(n.bit_length() - 1):
                bit = f"{sym}.bit{i}"
                terms[bit] = terms.get(bit, 0) + coeff * (1 << i)
        return Affine(0, frozenset((s, c) for s, c in terms.items() if c != 0), rng=aff.rng)

    def _wave_uniform(self, aff):
        """The value seen after a wave broadcast (``v_readfirstlane`` = lane 0): lane bits of the
        thread index are cleared, so only the wave-index (warp) component of ``%tid.x`` survives.
        A pure constant / already block-uniform value passes through; a value not decomposable into
        the ``%tid`` bit basis becomes opaque (its per-wave dependence is unknown -> fail closed)."""
        if not isinstance(aff, Affine) or not aff.terms:
            return aff
        exp = self._tid_bit_expand(Affine(0, aff.terms, rng=aff.rng))
        if exp is None:
            return Opaque(f"readfirstlane({canon(aff)})")
        kept = frozenset((s, c) for s, c in exp.terms
                         if not ((bi := _lane_bit(s)) is not None and bi < _WARP_BITS))
        return Affine(aff.const, kept, rng=None)

    @staticmethod
    def _bit_positions(exp):
        pos = {}
        for sym, coeff in exp.terms:
            if coeff <= 0 or (coeff & (coeff - 1)) != 0:
                return None
            p = coeff.bit_length() - 1
            if p in pos:
                return None
            pos[p] = sym
        return pos

    def _and(self, a, b):
        # AND is commutative; the mask is whichever operand is a bare constant (an AMD `v_and_b32`
        # freely emits `and const, val` or `and val, const`).
        if isinstance(b, Affine) and not b.terms:
            val, mask = a, b.const
        elif isinstance(a, Affine) and not a.terms:
            val, mask = b, a.const
        else:
            return self._opaque_pair(a, "and", b)
        if isinstance(val, Affine):
            bits = _set_bits_mask(val)
            if bits is not None and (mask & bits) == bits:
                return val  # no-op mask (covers every possibly-set bit)
            exp = self._tid_bit_expand(val)
            pos = self._bit_positions(exp) if exp is not None else None
            if pos is not None and mask >= 0:
                kept = frozenset((sym, 1 << p) for p, sym in pos.items() if (mask >> p) & 1)
                hi = sum(1 << p for p in pos if (mask >> p) & 1) + 1
                return Affine(0, kept, rng=(0, hi))
        return self._opaque_pair(a, "and", b)

    def _or_xor(self, a, b):
        if isinstance(a, Affine) and isinstance(b, Affine):
            ba, bb = _set_bits_mask(a), _set_bits_mask(b)
            if ba is not None and bb is not None and (ba & bb) == 0 and (a.const & bb) == 0 and (b.const & ba) == 0:
                return self._binop_add(a, b, 1)
        return self._opaque_pair(a, "orxor", b)

    def _bfe(self, a, pos, length):
        if not (isinstance(pos, Affine) and not pos.terms and isinstance(length, Affine) and not length.terms):
            return self._opaque_pair(a, "bfe", pos)
        p, ln = pos.const, length.const
        exp = self._tid_bit_expand(a)
        bpos = self._bit_positions(exp) if exp is not None else None
        if bpos is not None and p >= 0 and ln > 0:
            kept = frozenset((sym, 1 << (bp - p)) for bp, sym in bpos.items() if p <= bp < p + ln)
            hi = sum(1 << (bp - p) for bp in bpos if p <= bp < p + ln) + 1
            return Affine(0, kept, rng=(0, hi))
        return self._opaque_pair(a, "bfe", pos)

    # -- opaque construction --------------------------------------------------

    def _fresh_opq(self):
        self._opq += 1
        return _symbol(f"opq:{self._opq}")

    def _opaque_pair(self, a, name, b):
        if self.absorb_opaque:
            return self._fresh_opq()
        return Opaque(f"{name}({canon(a)},{canon(b)})")

    def _opaque_def(self, d):
        if self.absorb_opaque:
            return self._fresh_opq()
        inst = d.inst
        parts = [canon(self.of_operand(o, d.index)) for o in inst.operands[1:]]
        slot = "" if d.slot is None else f"#{d.slot}"
        return Opaque(f"{inst.opcode}{''.join(inst.modifiers)}{slot}({','.join(parts)})")


def _bit_basis(ev, aff):
    """Rewrite ``aff``'s thread-index terms into the ``%tid.<dim>.bit<i>`` basis — the (warp, lane)
    decomposition. For wave64, ``%tid.x`` bits [0, 6) are the LANE, bits [6, log2 ntid.x) are the
    WARP. Returns ``(uniform_terms, bit_terms, const)`` or ``None`` (fail closed) when a term is not
    decomposable (non-power-of-two / unknown ntid, or an unknown varying symbol)."""
    if not isinstance(aff, Affine):
        return None
    uniform, bits = [], {}
    for sym, coeff in aff.terms:
        parts = sym.split(".")
        if parts[-1].startswith("bit") and parts[0] in ("%tid", "%laneid"):
            bits[sym] = bits.get(sym, 0) + coeff
            continue
        if parts[0] == "%laneid":
            stem, n = "%laneid", 64  # wave64 lane id range (PTX was 32)
        elif parts[0] == "%tid" and len(parts) == 2:
            stem, n = sym, ev.reqntid.get(parts[1])
        elif sym.startswith(("sym:", "reg:", "param:", "vec:", "opq:", "%ctaid", "%nctaid", "%ntid")):
            uniform.append((sym, coeff))
            continue
        else:
            return None
        if not _is_pow2(n):
            return None
        for i in range(n.bit_length() - 1):
            b = f"{stem}.bit{i}"
            bits[b] = bits.get(b, 0) + coeff * (1 << i)
    return frozenset(uniform), {s: c for s, c in bits.items() if c != 0}, aff.const


def thread_image(ev, aff):
    """``(uniform_key, offsets)`` of an address over the whole thread grid, or ``None``.

    ``uniform_key`` identifies the memory object (block-uniform base + row offset); ``offsets`` is
    the frozenset of byte addresses the instruction touches as the thread index ranges over the
    block. Two accesses alias only when their ``uniform_key`` matches, on the intersection of their
    ``offsets`` — this is how a cross-warp LDS store slot is matched to a read."""
    got = _bit_basis(ev, aff)
    if got is None:
        return None
    uniform, bits, const = got
    offs = {const}
    for coeff in bits.values():
        offs = {o for base in offs for o in (base, base + coeff)}
        if len(offs) > _MAX_IMAGE:
            return None
    return uniform, frozenset(offs)
