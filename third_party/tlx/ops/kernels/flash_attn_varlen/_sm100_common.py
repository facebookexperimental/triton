"""JIT helpers shared by the sm100 varlen forward and backward (ported from ads_mkl TLX JFA)."""

import triton
import triton.language as tl


@triton.jit
def _reduce_or(x, y):
    return x | y


@triton.jit
def _split_n(x, SPLIT_FACTOR: tl.constexpr):
    if SPLIT_FACTOR == 1:
        return (x, )
    else:
        x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        return _split_n(x0, SPLIT_FACTOR // 2) + _split_n(x1, SPLIT_FACTOR // 2)


@triton.jit
def _join_n(xs):
    if len(xs) == 1:
        return xs[0]
    else:
        x0 = _join_n(xs[:len(xs) // 2])
        x1 = _join_n(xs[len(xs) // 2:])
        x = tl.join(x0, x1).permute(0, 2, 1).reshape([x0.shape[0], x0.shape[1] * 2])
        return x


@triton.jit
def _keep_range_scalar(qk, row, s, i, lo0, hi0, cap, other):
    # Bitmask over a block of 16 columns: column s + i of row `row` is kept iff
    # row + lo0 <= s + i < min(row + hi0, cap).
    lo_c = min(max(row + lo0 - s, 0), 16)
    hi_c = min(max(min(row + hi0, cap) - s, 0), 16)
    bits = (-1 << lo_c) & ~(-1 << hi_c)
    return tl.where((bits & (1 << i)) != 0, qk, other)


@triton.jit
def _apply_column_range(qk, lo0, hi0, cap, other: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Keeps columns [row + lo0, min(row + hi0, cap)) of each tile row. A per-16-column bitmask lets ptxas use R2P
    # (register to predicate) instead of per-element compares, as the dense sm100 flash attention does for its
    # causal diagonal. Only iotas and scalars go in, so no operand needs a layout conversion to qk's TMEM layout.
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    lo0 = tl.full([1, 1], lo0, tl.int32)
    hi0 = tl.full([1, 1], hi0, tl.int32)
    cap = tl.full([1, 1], cap, tl.int32)
    other = tl.full([1, 1], other, qk.dtype)
    return tl.map_elementwise(_keep_range_scalar, qk, rows, cols & ~0xF, cols & 0xF, lo0, hi0, cap, other)


@triton.jit
def _row_has_key(offs_m, diag, klen, WINDOW_LEFT: tl.constexpr, WINDOW_RIGHT: tl.constexpr):
    has_key = (offs_m >= 0) & (klen > 0)
    if WINDOW_RIGHT is not None:
        has_key &= offs_m + diag + WINDOW_RIGHT >= 0
    if WINDOW_LEFT is not None:
        has_key &= offs_m + diag - WINDOW_LEFT < klen
    return has_key


@triton.jit
def _kv_range(start_m, qlen, klen, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, WINDOW_LEFT: tl.constexpr,
              WINDOW_RIGHT: tl.constexpr):
    # Keys [lo, hi) of M tile start_m. Empty for an empty key sequence, or for a
    # band that covers no key; every warp group must then skip the tile.
    lo, hi = 0, klen
    diag = klen - qlen
    if WINDOW_LEFT is not None:
        lo = max(lo, ((start_m * BLOCK_M + diag - WINDOW_LEFT) // BLOCK_N) * BLOCK_N)
    if WINDOW_RIGHT is not None:
        hi = min(hi, (start_m + 1) * BLOCK_M + diag + WINDOW_RIGHT)
    return lo, hi


@triton.jit
def _unmasked_kv_range(m0, lo, hi, qlen, klen, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, WINDOW_LEFT: tl.constexpr,
                       WINDOW_RIGHT: tl.constexpr):
    # Key tiles [mid_lo, mid_hi) of [lo, hi) are below klen and inside the band for every row of [m0, m0 + BLOCK_M),
    # so they need no mask. lo is a multiple of BLOCK_N, and so are both bounds.
    diag = klen - qlen
    mid_lo = lo
    end = klen
    if WINDOW_LEFT is not None:
        first = max(m0 + BLOCK_M - 1 + diag - WINDOW_LEFT, 0)
        mid_lo = max(mid_lo, tl.cdiv(first, BLOCK_N) * BLOCK_N)
    if WINDOW_RIGHT is not None:
        end = min(end, max(m0 + diag + WINDOW_RIGHT + 1, 0))
    mid_lo = min(mid_lo, hi)
    mid_hi = max(min((end // BLOCK_N) * BLOCK_N, hi), mid_lo)
    return mid_lo, mid_hi


@triton.jit
def _tile_in_band(m0, n0, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, diag, WINDOW_LEFT: tl.constexpr,
                  WINDOW_RIGHT: tl.constexpr):
    # Whether every (query, key) of rows [m0, m0 + BLOCK_M) and keys [n0, n0 + BLOCK_N) lies inside the band.
    inside = n0 >= 0
    if WINDOW_RIGHT is not None:
        inside &= n0 + BLOCK_N - 1 <= m0 + diag + WINDOW_RIGHT
    if WINDOW_LEFT is not None:
        inside &= n0 >= m0 + BLOCK_M - 1 + diag - WINDOW_LEFT
    return inside


@triton.jit
def bwd_lookup_offsets(
    tile_to_batch,
    tile_to_head,
    tile_to_block,
    G,
    tile_idx,
    BROADCAST_Q: tl.constexpr,
):
    off_z = tl.load(tile_to_batch + tile_idx)
    off_h = tl.load(tile_to_head + tile_idx)
    pid = tl.load(tile_to_block + tile_idx)
    off_h_kv = off_h // G
    if BROADCAST_Q:
        off_q_z = 0
    else:
        off_q_z = off_z
    return off_z, off_h, off_h_kv, off_q_z, pid


@triton.jit
def bwd_calculate_num_steps(
    qlen,
    klen,
    start_n,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
):
    # Query rows [start_m, end_m) that see some key of the K tile at start_n, in BLOCK_M1 steps.
    diag = klen - qlen
    start_m = 0
    end_m = qlen
    if WINDOW_RIGHT is not None:
        start_m = max(start_m, start_n - diag - WINDOW_RIGHT)
    if WINDOW_LEFT is not None:
        end_m = min(end_m, start_n + BLOCK_N1 - diag + WINDOW_LEFT)
    start_m = (start_m // BLOCK_M1) * BLOCK_M1
    num_steps = (tl.cdiv(end_m, BLOCK_M1) * BLOCK_M1 - start_m) // BLOCK_M1
    return num_steps.to(tl.int32), start_m
