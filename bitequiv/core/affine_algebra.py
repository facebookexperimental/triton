"""ISA-neutral symbolic integer algebra: the ``Affine`` / ``Opaque`` lattice, its canonical
serialization, and the range / trailing-zero / bit-mask helpers an address evaluator builds on.

Moved VERBATIM out of :mod:`bitequiv.ptx.affine`, which keeps the PTX-specific parts (the
``%tid``-family special registers, :class:`~bitequiv.ptx.affine.AffineEval`, ``thread_image``,
``reqntid_of``) and imports this algebra."""

from dataclasses import dataclass, field

_INF_TZ = 64  # "infinitely many" trailing zeros (a value known to be 0)


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
