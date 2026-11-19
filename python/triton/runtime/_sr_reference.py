"""Bit-level host reference for stochastic store RNG ABI v1."""

_MASK32 = (1 << 32) - 1
_MASK64 = (1 << 64) - 1


def _mix64(value):
    value &= _MASK64
    for multiplier in (0xff51afd7ed558ccd, 0xc4ceb9fe1a85ec53):
        value = ((value ^ (value >> 33)) * multiplier) & _MASK64
    return value ^ (value >> 33)


def _domain_hash(a, b, domain):
    return _mix64(_mix64(a ^ domain) ^ b)


def random_bits(logical_index, base_seed, dynamic_seed, invocation_offset=0, rounds=10, site_id=0, entropy="element"):
    """Return Philox r0; signed indices are interpreted modulo 2**64."""
    if rounds not in (7, 10):
        raise ValueError("ABI v1 supports seven or ten Philox rounds")
    if entropy not in ("element", "packed8"):
        raise ValueError("SR entropy must be element or packed8")
    seed = _domain_hash(base_seed, dynamic_seed, 0x747269746f6e5301)
    invocation = _domain_hash(invocation_offset, site_id, 0x747269746f6e5302)
    index = logical_index & _MASK64
    counter_index = index >> 3 if entropy == "packed8" else index
    counter = (counter_index & _MASK32, counter_index >> 32, invocation & _MASK32, invocation >> 32)
    key0, key1 = seed & _MASK32, seed >> 32
    for _ in range(rounds):
        p0, p1 = counter[0] * 0xd2511f53, counter[2] * 0xcd9e8d57
        counter = ((p1 >> 32) ^ counter[1] ^ key0, p1 & _MASK32, (p0 >> 32) ^ counter[3] ^ key1, p0 & _MASK32)
        key0 = (key0 + 0x9e3779b9) & _MASK32
        key1 = (key1 + 0xbb67ae85) & _MASK32
    if entropy == "packed8":
        return (counter[(index & 7) >> 1] >> ((index & 1) * 16)) & 0xffff
    return counter[0]


def bf16_bits(fp32_bits, entropy, overflow="ieee"):
    """IEEE BF16: preserve infinities/zeros/subnormals, canonicalize NaNs."""
    bits = fp32_bits & _MASK32
    if bits & 0x7fffffff > 0x7f800000:
        return 0x7fff
    if bits & 0x7fffffff == 0x7f800000:
        return bits >> 16
    rounded = ((bits + (entropy & 0xffff)) >> 16) & 0xffff
    if overflow == "satfinite" and rounded & 0x7fff >= 0x7f80:
        rounded = (rounded & 0x8000) | 0x7f7f
    elif overflow not in ("ieee", "satfinite"):
        raise ValueError("SR overflow must be ieee or satfinite")
    return rounded
