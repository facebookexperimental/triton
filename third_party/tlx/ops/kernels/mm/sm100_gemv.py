"""Blackwell (sm100) GEMV for `tlx.ops.mm` at M == 1 with a K-major B (the `x @ w.T` linear layer at decode).

One row of A only streams B, so each program loads BN rows of W = B.T straight into registers, multiplies them by
the matching slice of x with FP32 FMAs, and reduces once at the end. Programs are small (one or two rows, one to
eight warps) so that the grid covers B in one wave.

When a program's rows fit in registers (up to 128 elements of W per thread, K <= 16384 for two rows and eight
warps), K is fully unrolled into NC chunks of BK columns, 16 bytes per thread per load, and every chunk is requested
before any is used: the most W in flight per program. x is read once per program. Longer rows loop over K in
UNROLL loads of BK columns, which keeps more of W in flight than one wide load.

The kernel is launched with programmatic dependent launch and waits on earlier kernels before it reads x and
writes C. By default it also waits before reading W. With `b_static=True` the caller promises that no kernel that
may still be running writes B (a model weight, for instance); the first chunks of W are then requested before the
wait, so they stream while the previous kernel finishes, as decode GEMMs that read static weights do. The unrolled
form bounds that prefetch to NPRE chunks: requesting every chunk before the wait competes for memory bandwidth with
a previous kernel that is still running.
"""

import triton
import triton.language as tl

# Measured at M = 1 on GB300 against the recipe's decode GEMMs, caches flushed. Unrolled form: (N, K) -> (BN, BK,
# num_warps). Short rows stream better through the looped form: (N, K) -> (BN, BK, UNROLL, num_warps, trigger
# dependents before the first load).
_TUNED = {
    (4608, 8192): (2, 4096, 8),
    (8192, 2048): (2, 1024, 2),
    (2048, 4096): (2, 2048, 4),
    (1024, 8192): (2, 4096, 8),
    (1024, 6144): (2, 1024, 4),
}
_TUNED_LOOPED = {
    (8192, 1024): (4, 128, 4, 4, True),
    (4096, 1024): (2, 256, 4, 4, True),
}
_MIN_BLOCK_K = 128
_MAX_W_PER_THREAD = 128
# Chunks of W the unrolled form requests before the wait when B is static.
_NPRE = 2


@triton.jit
def _load_w(w_rows, k, kk):
    return tl.load(w_rows + (k + kk), eviction_policy="evict_first")


@triton.jit
def _times_x(w, x_rows, k, kk):
    return w.to(tl.float32) * tl.load(x_rows + (k + kk)).to(tl.float32)


@triton.jit
def gemv_unrolled_kernel(x_ptr, w_ptr, c_ptr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                         B_STATIC: tl.constexpr, NPRE: tl.constexpr):
    NC: tl.constexpr = K // BK
    offs_n = tl.program_id(0) * BN + tl.arange(0, BN)
    w_rows = w_ptr + offs_n.to(tl.int64)[:, None] * K  # a row can start past 2^31 elements
    x_rows = x_ptr + offs_n[:, None] * 0
    kk = tl.arange(0, BK)[None, :]
    if not B_STATIC:
        tl.extra.cuda.gdc_wait()  # W may be written by an earlier kernel
    ws = ()
    for c in tl.static_range(NC):
        if B_STATIC and c == NPRE:
            tl.extra.cuda.gdc_wait()  # x and C may be in use by earlier kernels
        ws = ws + (_load_w(w_rows, c * BK, kk), )
    if B_STATIC and NPRE >= NC:
        tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    acc = tl.zeros([BN, BK], dtype=tl.float32)
    for c in tl.static_range(NC):
        acc += _times_x(ws[c], x_rows, c * BK, kk)
    tl.store(c_ptr + offs_n, tl.sum(acc, axis=1).to(c_ptr.dtype.element_ty))


@triton.jit
def gemv_kernel(x_ptr, w_ptr, c_ptr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, UNROLL: tl.constexpr,
                B_STATIC: tl.constexpr, TRIGGER_FIRST: tl.constexpr):
    offs_n = tl.program_id(0) * BN + tl.arange(0, BN)
    w_rows = w_ptr + offs_n.to(tl.int64)[:, None] * K  # a row can start past 2^31 elements
    x_rows = x_ptr + offs_n[:, None] * 0
    kk = tl.arange(0, BK)[None, :]
    STEP: tl.constexpr = UNROLL * BK
    K_MAIN: tl.constexpr = K // STEP * STEP
    if TRIGGER_FIRST:
        tl.extra.cuda.gdc_launch_dependents()  # dependents still wait for this kernel before reading C
    if not B_STATIC:
        tl.extra.cuda.gdc_wait()  # W may be written by an earlier kernel
    w0 = _load_w(w_rows, 0, kk)
    if UNROLL >= 2:
        w1 = _load_w(w_rows, BK, kk)
    if UNROLL >= 4:
        w2 = _load_w(w_rows, 2 * BK, kk)
        w3 = _load_w(w_rows, 3 * BK, kk)
    if UNROLL >= 8:
        w4 = _load_w(w_rows, 4 * BK, kk)
        w5 = _load_w(w_rows, 5 * BK, kk)
        w6 = _load_w(w_rows, 6 * BK, kk)
        w7 = _load_w(w_rows, 7 * BK, kk)
    if B_STATIC:
        tl.extra.cuda.gdc_wait()  # x and C may be in use by earlier kernels
    if not TRIGGER_FIRST:
        tl.extra.cuda.gdc_launch_dependents()
    acc = tl.zeros([BN, BK], dtype=tl.float32)
    for k0 in range(0, K_MAIN, STEP):
        if k0 > 0:
            w0 = _load_w(w_rows, k0, kk)
            if UNROLL >= 2:
                w1 = _load_w(w_rows, k0 + BK, kk)
            if UNROLL >= 4:
                w2 = _load_w(w_rows, k0 + 2 * BK, kk)
                w3 = _load_w(w_rows, k0 + 3 * BK, kk)
            if UNROLL >= 8:
                w4 = _load_w(w_rows, k0 + 4 * BK, kk)
                w5 = _load_w(w_rows, k0 + 5 * BK, kk)
                w6 = _load_w(w_rows, k0 + 6 * BK, kk)
                w7 = _load_w(w_rows, k0 + 7 * BK, kk)
        acc += _times_x(w0, x_rows, k0, kk)
        if UNROLL >= 2:
            acc += _times_x(w1, x_rows, k0 + BK, kk)
        if UNROLL >= 4:
            acc += _times_x(w2, x_rows, k0 + 2 * BK, kk)
            acc += _times_x(w3, x_rows, k0 + 3 * BK, kk)
        if UNROLL >= 8:
            acc += _times_x(w4, x_rows, k0 + 4 * BK, kk)
            acc += _times_x(w5, x_rows, k0 + 5 * BK, kk)
            acc += _times_x(w6, x_rows, k0 + 6 * BK, kk)
            acc += _times_x(w7, x_rows, k0 + 7 * BK, kk)
    for k0 in range(K_MAIN, K, BK):
        acc += _times_x(_load_w(w_rows, k0, kk), x_rows, k0, kk)
    tl.store(c_ptr + offs_n, tl.sum(acc, axis=1).to(c_ptr.dtype.element_ty))


def unrolled_config(N, K):
    """(BN, BK, num_warps) of the unrolled form, or None for the looped form (rows that do not fit in registers, and
    shapes measured faster looped)."""
    if (N, K) in _TUNED_LOOPED:
        return None
    if (N, K) in _TUNED:
        return _TUNED[(N, K)]
    bn = 2 if N % 2 == 0 else 1
    bk = next(b for b in (4096, 2048, 1024, 512, 256, _MIN_BLOCK_K) if K % b == 0)
    num_warps = max(1, min(8, bk * bn // 256))  # 16 bytes per thread per load
    return (bn, bk, num_warps) if bn * K <= _MAX_W_PER_THREAD * num_warps * 32 else None


def config(N, K):
    """(BN, BK, UNROLL, num_warps, trigger first) of the looped form."""
    if (N, K) in _TUNED_LOOPED:
        return _TUNED_LOOPED[(N, K)]
    bn = next(b for b in (4, 2, 1) if N % b == 0 and (b == 1 or N >= 1024 * b))
    bk = next(b for b in (512, 256, _MIN_BLOCK_K) if K % b == 0)
    unroll = next(u for u in (8, 4, 2, 1) if u * bk <= K)
    return bn, bk, unroll, 1, True


def eligible(a, b, out=None):
    """Whether `a @ b` takes the GEMV: one row, A contiguous along K, and B a column-major view of a contiguous W."""
    M, K = a.shape
    return (M == 1 and a.stride(1) == 1 and b.T.is_contiguous() and K > 0 and K % _MIN_BLOCK_K == 0
            and (out is None or out.is_contiguous()))


def mm(a, b, c, *, b_static=False):
    K = a.shape[1]
    N = b.shape[1]
    unrolled = unrolled_config(N, K)
    if unrolled is not None:
        bn, bk, num_warps = unrolled
        gemv_unrolled_kernel[(N // bn, )](a, b.T, c, K=K, BN=bn, BK=bk, B_STATIC=b_static, NPRE=_NPRE,
                                          num_warps=num_warps, launch_pdl=True)
        return c
    bn, bk, unroll, num_warps, trigger_first = config(N, K)
    gemv_kernel[(N // bn, )](a, b.T, c, K=K, BN=bn, BK=bk, UNROLL=unroll, B_STATIC=b_static,
                             TRIGGER_FIRST=trigger_first, num_warps=num_warps, launch_pdl=True)
    return c
