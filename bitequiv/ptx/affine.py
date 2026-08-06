"""Conservative symbolic evaluator for **integer address registers**.

Given the def-use index (:mod:`bitequiv.ptx.linker`), evaluate an address-computing
register to an :class:`Affine` form over a fixed symbol basis (``%tid.*``, ``%ctaid.*``,
``%ntid.*``, kernel ``.param`` names, base pointers) — or to an :class:`Opaque` token
when it cannot be *proven* affine. This is the PTX echo of the LinearLayout bases: the
coefficients of ``tid``/``ctaid``/per-load-offset over the reduce axis are the recovered
thread<->data map used to label tree leaves.

**Conservative floor.** Only a fixed opcode set is modeled, and bit-slicing (``and``/``or``/
``xor``/``shr``) is resolved only when *provably* exact (trailing-zero + range analysis,
using ``.reqntid`` for the ``tid`` range). Anything else becomes an ``Opaque`` whose token
is the *structural* expression (register names inlined to their defs), so two address values
compare equal only when provably the same computation (it over-splits rather than
over-merging). Caveat: a token that bottoms out at an unresolved register name can collide
across two separately-compiled modules — see :mod:`bitequiv.ptx_reduction` for the
module-level guarantee and its residuals.
"""

from dataclasses import dataclass, field

from pyptx.ir.nodes import (
    AddressOperand,
    ImmediateOperand,
    RegisterOperand,
)

_INF_TZ = 64  # "infinitely many" trailing zeros (a value known to be 0)

# Floating-point type modifiers; an op carrying one is a value combine, not address math.
_FP_WIDTHS = frozenset({".f16", ".f16x2", ".f32", ".f64", ".bf16", ".bf16x2"})

_SPECIAL_PREFIXES = (
    "%tid",
    "%ntid",
    "%ctaid",
    "%nctaid",
    "%laneid",
    "%warpid",
    "%smid",
    "%nsmid",
    "%gridid",
    "%lanemask",
    "%clock",
    "%pm",
    "%envreg",
)


def _parse_int(text):
    """Parse a PTX integer immediate (decimal/hex/signed); ``None`` if not an integer
    (e.g. a float literal ``0d...`` / ``0f...``)."""
    try:
        return int(text, 0)
    except (ValueError, TypeError):
        return None


def _ctz(n):
    """Count trailing zero bits; a 0 value has "infinitely" many."""
    n = abs(n)
    if n == 0:
        return _INF_TZ
    z = 0
    while (n & 1) == 0:
        z += 1
        n >>= 1
    return z


def _is_pow2(n):
    return n is not None and n > 0 and (n & (n - 1)) == 0


def _is_special(name):
    return any(name.startswith(p) for p in _SPECIAL_PREFIXES)


@dataclass(frozen=True)
class Affine:
    """``const + sum(coeff * symbol)`` over integer symbols.

    Value identity (``==``/hash) is the (const, terms) pair only; ``rng`` is a derived
    range hint excluded from equality so two equal affines stay equal regardless of how
    their ranges were inferred."""

    const: int
    terms: frozenset  # frozenset[(symbol: str, coeff: int)], coeff != 0
    rng: tuple | None = field(default=None, compare=False)  # (lo, hi) exclusive-hi, or None

    @property
    def tz(self):
        vals = [_ctz(self.const)]
        vals += [_ctz(c) for _, c in self.terms]
        return min(vals) if vals else _INF_TZ

    def to_str(self):
        if not self.terms:
            return str(self.const)
        body = "+".join(f"{c}*{s}" for s, c in sorted(self.terms))
        return body if self.const == 0 else f"{self.const}+{body}"


@dataclass(frozen=True)
class Opaque:
    """Top of the lattice: a value not provably affine. ``token`` is the structural
    expression text (register names inlined to their defs) so two opaques are equal iff
    they are the same computation."""

    token: str


def canon(value):
    """Canonical comparison string for an :class:`Affine` or :class:`Opaque`."""
    return value.to_str() if isinstance(value, Affine) else f"opq({value.token})"


def _const(v, lo=None):
    return Affine(v, frozenset(), rng=(v, v + 1))


def _symbol(name, rng=None):
    return Affine(0, frozenset({(name, 1)}), rng=rng)


def _add_terms(a, b, sign=1):
    """Merge term maps of two affines (b scaled by sign), dropping zero coeffs."""
    m = {}
    for s, c in a.terms:
        m[s] = m.get(s, 0) + c
    for s, c in b.terms:
        m[s] = m.get(s, 0) + sign * c
    return frozenset((s, c) for s, c in m.items() if c != 0)


def _rng_add(ra, rb, sign=1):
    if ra is None or rb is None:
        return None
    lo = ra[0] + (rb[0] if sign > 0 else -(rb[1] - 1))
    hi = (ra[1] - 1) + (rb[1] - 1 if sign > 0 else -rb[0]) + 1
    return (lo, hi)


def _rng_scale(r, k):
    if r is None:
        return None
    if k >= 0:
        return (r[0] * k, (r[1] - 1) * k + 1)
    return ((r[1] - 1) * k, r[0] * k + 1)


def _set_bits_mask(aff):
    """Mask of bit positions ``aff`` could possibly set, or ``None`` if unknown.
    Uses trailing zeros (low bound) and the range (high bound)."""
    if aff.rng is None or aff.rng[1] is None:
        return None
    hi = aff.rng[1] - 1  # max value
    if hi < 0:
        return None
    msb = hi.bit_length()  # bits [0, msb)
    low = aff.tz
    full = (1 << msb) - 1
    return full & ~((1 << low) - 1) if low < _INF_TZ else 0


class AffineEval:
    """Evaluate integer registers to :class:`Affine` / :class:`Opaque`, memoized per def."""

    def __init__(self, defuse, reqntid=None, absorb_opaque=False):
        self.du = defuse
        self.reqntid = reqntid or {}  # {'x': N, 'y': .., 'z': ..}
        self._memo = {}  # id(Def) -> value
        self._stack = set()
        # When True, an unmodellable value becomes a fresh DROPPABLE symbol ("opq:N") instead
        # of an Opaque that poisons the whole expression. Used only by the column-image
        # extractor (leaf_columns), which keeps the %tid-linear part and drops opq:/param:/
        # ctaid symbols — so an opaque ROW base (e.g. mul(stride, ctaid)) no longer hides the
        # tid-dependent COLUMN offset. Never used for the identity-bearing leaf_coord.
        self.absorb_opaque = absorb_opaque
        self._opq = 0

    # -- operand entry points -------------------------------------------------

    def of_operand(self, operand, before_index):
        if isinstance(operand, ImmediateOperand):
            v = _parse_int(operand.text)
            return _const(v) if v is not None else Opaque(f"imm({operand.text})")
        if isinstance(operand, RegisterOperand):
            return self.of_reg(operand.name, before_index)
        if isinstance(operand, AddressOperand):
            return self.of_address(operand, before_index)
        return Opaque(f"operand({type(operand).__name__})")

    def of_address(self, addr, before_index):
        """Affine of a ``[base(+offset)]`` memory address."""
        base = (self.of_reg(addr.base, before_index) if addr.base.startswith("%") else _symbol("sym:" + addr.base))
        if addr.offset is None:
            return base
        off_int = _parse_int(addr.offset)
        if off_int is not None:
            off = _const(off_int)
        elif addr.offset.startswith("%"):
            off = self.of_reg(addr.offset, before_index)
        else:
            off = _symbol("sym:" + addr.offset)
        return self._binop_add(base, off, 1)

    def of_reg(self, name, before_index):
        if _is_special(name):
            return self._special(name)
        d = self.du.last_writer(name, before_index)
        if d is None:
            # An input register not produced in-body (e.g. a kernel pointer used
            # before its ld.param is visible, or an undefined source): a stable symbol.
            return _symbol("reg:" + name)
        if id(d) in self._memo:
            return self._memo[id(d)]
        if id(d) in self._stack:  # cycle (not expected in single-reduction PTX)
            return Opaque("cycle:" + name)
        self._stack.add(id(d))
        val = self._eval_def(d)
        self._stack.discard(id(d))
        self._memo[id(d)] = val
        return val

    # -- core evaluation ------------------------------------------------------

    def _special(self, name):
        if name.startswith("%tid.") and (dim := name.split(".")[-1]) in self.reqntid:
            return _symbol(name, rng=(0, self.reqntid[dim]))
        if name.startswith("%laneid"):
            return _symbol(name, rng=(0, 32))
        return _symbol(name)  # range unknown

    def _eval_def(self, d):
        inst = d.inst
        op, mods = inst.opcode, inst.modifiers
        at = d.index
        ops = inst.operands
        srcs = ops[1:]

        def ev(o):
            return self.of_operand(o, at)

        # copies / casts
        if op == "mov" and len(srcs) == 1:
            return ev(srcs[0])
        if op == "cvta":  # address-space cast: identity on the address value
            return ev(srcs[-1])
        if op == "ld" and ".param" in mods and len(srcs) == 1:
            a = srcs[0]
            base = a.base if isinstance(a, AddressOperand) else getattr(a, "name", str(a))
            return _symbol("param:" + base)

        # Address arithmetic is integer-only. A floating-point combine (add.f32, mul.f64,
        # ...) is a value, not an address — never model it as affine (its operand chain can
        # be arbitrarily deep). This keeps the evaluator confined to shallow address math.
        if any(m in _FP_WIDTHS for m in mods):
            return self._opaque_def(d)

        # additive
        if op == "add" and len(srcs) == 2:
            return self._binop_add(ev(srcs[0]), ev(srcs[1]), 1)
        if op == "sub" and len(srcs) == 2:
            return self._binop_add(ev(srcs[0]), ev(srcs[1]), -1)

        # multiply / fused-multiply-add by a constant
        if op == "mul" and len(srcs) == 2:  # mul.lo / mul.wide
            return self._mul(ev(srcs[0]), ev(srcs[1]), inst)
        if op == "mad" and len(srcs) == 3:  # mad.lo / mad.wide : a*b + c
            return self._binop_add(self._mul(ev(srcs[0]), ev(srcs[1]), inst), ev(srcs[2]), 1)

        # shifts
        if op == "shl" and len(srcs) == 2:
            return self._shl(ev(srcs[0]), ev(srcs[1]))
        if op == "shr" and len(srcs) == 2:
            return self._shr(ev(srcs[0]), ev(srcs[1]))

        # bit-slicing
        if op == "and" and len(srcs) == 2:
            return self._and(ev(srcs[0]), ev(srcs[1]))
        if op in ("or", "xor") and len(srcs) == 2:
            return self._or_xor(ev(srcs[0]), ev(srcs[1]))
        if op == "bfe" and len(srcs) == 3:  # bit-field extract: bits [pos, pos+len) of a
            return self._bfe(ev(srcs[0]), ev(srcs[1]), ev(srcs[2]))

        return self._opaque_def(d)

    # -- modeled operators ----------------------------------------------------

    def _binop_add(self, a, b, sign):
        if isinstance(a, Opaque) or isinstance(b, Opaque):
            return self._opaque_pair(a, "add" if sign > 0 else "sub", b)
        terms = _add_terms(a, b, sign)
        const = a.const + sign * b.const
        return Affine(const, terms, rng=_rng_add(a.rng, b.rng, sign))

    def _mul(self, a, b, inst):
        # affine * const only (const * affine handled symmetrically)
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
        # exact only when the value is provably a multiple of 2**s (tz >= s)
        if isinstance(a, Affine) and isinstance(b, Affine) and not b.terms:
            s = b.const
            if a.tz >= s:
                terms = frozenset((sym, c >> s) for sym, c in a.terms)
                return Affine(a.const >> s, terms, rng=_rng_scale(a.rng, 1) if a.rng is None else
                              (a.rng[0] >> s, ((a.rng[1] - 1) >> s) + 1))
            # tid bit-slice: decompose to the bit basis, drop bits < s, shift the rest down.
            exp = self._tid_bit_expand(a)
            pos = self._bit_positions(exp) if exp is not None else None
            if pos is not None and s >= 0:
                kept = frozenset((sym, 1 << (p - s)) for p, sym in pos.items() if p >= s)
                hi = sum(1 << (p - s) for p in pos if p >= s) + 1
                return Affine(0, kept, rng=(0, hi))
        return self._opaque_pair(a, "shr", b)

    def _tid_bit_expand(self, aff):
        """Rewrite an integer affine into the ``%tid`` BIT basis: each ``%tid.<dim>`` term is
        replaced by ``sum_i (coeff * 2**i) * %tid.<dim>.bit<i>`` over ``i`` in ``[0, log2(N))``
        where ``N = reqntid[dim]`` (a bit symbol has range ``[0, 2)`` — see :func:`leaf_columns`).

        This is the tid->(warp, lane, tile) basis change: because ``N`` is a power of two, bits
        ``[0, log2 N)`` cover ``[0, N)`` EXACTLY, so the rewrite is bit-for-bit the same integer —
        the sound precondition for turning ``and`` / ``shr`` / ``bfe`` on ``tid`` (opaque today)
        into an exact affine. Returns ``None`` (caller stays opaque) when a term is not so
        decomposable: a non-power-of-two / unknown ``ntid`` (bits would over-cover ``[0, N)``), or a
        NON-tid varying symbol (``%ctaid`` / ``param:`` / ``%laneid`` / opaque) whose bits are
        unknown, or a non-zero constant (its bits could carry into the slice — kept out of scope)."""
        if not isinstance(aff, Affine) or aff.const != 0:
            return None
        terms = {}
        for sym, coeff in aff.terms:
            parts = sym.split(".")
            is_bit = len(parts) == 3 and parts[0] == "%tid" and parts[2].startswith("bit")
            is_plain = len(parts) == 2 and parts[0] == "%tid"
            if is_bit:  # already a bit symbol — carry through
                terms[sym] = terms.get(sym, 0) + coeff
                continue
            if not is_plain:
                return None  # a non-tid varying symbol -> not bit-decomposable
            n = self.reqntid.get(parts[1])
            if not _is_pow2(n):
                return None
            for i in range(n.bit_length() - 1):  # log2(N) bits
                bit = f"{sym}.bit{i}"
                terms[bit] = terms.get(bit, 0) + coeff * (1 << i)
        frozen = frozenset((s, c) for s, c in terms.items() if c != 0)
        return Affine(0, frozen, rng=aff.rng)

    @staticmethod
    def _bit_positions(exp):
        """The bit position of every term of a pure bit-basis affine (const 0, each coeff a single
        power of two), or ``None`` if any coeff is not a single set bit or two terms collide on one
        position (so a bitwise mask/shift would not distribute over the sum)."""
        pos = {}
        for sym, coeff in exp.terms:
            if coeff <= 0 or (coeff & (coeff - 1)) != 0:
                return None  # not a single power of two
            p = coeff.bit_length() - 1
            if p in pos:
                return None  # two bits collide on one position
            pos[p] = sym
        return pos

    def _and(self, a, b):
        # x & mask == x  when mask covers every bit x can possibly set (no-op mask).
        if isinstance(a, Affine) and isinstance(b, Affine) and not b.terms:
            mask = b.const
            bits = _set_bits_mask(a)
            if bits is not None and (mask & bits) == bits:
                return a
            # tid bit-slice: decompose to the bit basis and keep only bits fully inside the mask.
            exp = self._tid_bit_expand(a)
            pos = self._bit_positions(exp) if exp is not None else None
            if pos is not None and mask >= 0:
                kept = frozenset((sym, 1 << p) for p, sym in pos.items() if (mask >> p) & 1)
                hi = sum(1 << p for p in pos if (mask >> p) & 1) + 1
                return Affine(0, kept, rng=(0, hi))
        return self._opaque_pair(a, "and", b)

    def _or_xor(self, a, b):
        # x | imm == x ^ imm == x + imm  when imm is disjoint from x's possible set bits (no carry),
        # or (a | b) == (a ^ b) == (a + b) when a, b set DISJOINT bits — both add with no carry.
        if isinstance(a, Affine) and isinstance(b, Affine):
            ba, bb = _set_bits_mask(a), _set_bits_mask(b)
            if ba is not None and bb is not None and (ba & bb) == 0 and (a.const & bb) == 0 and (b.const & ba) == 0:
                return self._binop_add(a, b, 1)
        return self._opaque_pair(a, "orxor", b)

    def _bfe(self, a, pos, length):
        """``bfe`` (bit-field extract): bits ``[p, p+len)`` of ``a`` shifted down to bit 0. Exact on
        the tid bit basis (``= shr(and(a, ((1<<len)-1)<<p), p)``); opaque otherwise."""
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

    # -- opaque construction (structural, register-name independent) -----------

    def _fresh_opq(self):
        self._opq += 1
        return _symbol(f"opq:{self._opq}")  # droppable symbol (range unknown)

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


def reqntid_of(func):
    """Extract ``.reqntid`` (or ``.maxntid``) x/y/z bounds from a function's directives."""
    out = {}
    for fd in getattr(func, "directives", ()) or ():
        nm = getattr(fd, "name", "").lstrip(".")
        if nm in ("reqntid", "maxntid"):
            vals = list(getattr(fd, "values", ()))
            for dim, v in zip(("x", "y", "z"), vals):
                iv = v if isinstance(v, int) else _parse_int(str(v))
                if iv is not None:
                    out.setdefault(dim, iv)
    return out
