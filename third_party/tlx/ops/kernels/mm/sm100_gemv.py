"""Blackwell (sm100) GEMV for `tlx.ops.mm` at M == 1 with a K-major B (the `x @ w.T` linear layer at decode).

One row of A only streams B, so each program loads BN rows of W = B.T straight into registers, multiplies them by
the matching slice of x with FP32 FMAs, and reduces once at the end. Programs are small (one to four rows, two to
eight warps) so that the grid covers B in one wave. Each loop iteration issues UNROLL separate loads of BK columns
before using any, which keeps more of W in flight than one wide load; chunks past the last full iteration are read
one at a time. x is loaded in the shape of the W tile so that both share a register layout; a [BK] load of x would
be moved into that layout through shared memory at every step.

The kernel is launched with programmatic dependent launch and waits on earlier kernels before it reads x and
writes C. By default it also waits before reading W. With `b_static=True` the caller promises that no kernel that
may still be running writes B (a model weight, for instance); the first iteration's chunks of W are then loaded
before the wait, so they stream while the previous kernel finishes, as decode GEMMs that read static weights do.
"""

import triton
import triton.language as tl

# Measured per kernel at M = 1 on GB300 with caches flushed: (N, K) -> (BN, BK, UNROLL, num_warps, trigger
# dependents before the first load).
_TUNED = {
    (4608, 8192): (1, 1024, 4, 8, False),
    (8192, 2048): (2, 1024, 2, 2, False),
    (2048, 4096): (2, 2048, 2, 4, False),
    (1024, 8192): (1, 1024, 4, 8, False),
    (8192, 1024): (4, 256, 2, 2, False),
    (1024, 6144): (1, 1024, 4, 8, False),
    (4096, 1024): (4, 256, 2, 4, True),
}
_MIN_BLOCK_K = 128


@triton.jit
def _load_w(w_rows, k, kk):
    return tl.load(w_rows + (k + kk)[None, :], eviction_policy="evict_first")


@triton.jit
def _times_x(w, x_rows, k, kk):
    return w.to(tl.float32) * tl.load(x_rows + (k + kk)[None, :]).to(tl.float32)


@triton.jit
def gemv_kernel(x_ptr, w_ptr, c_ptr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, UNROLL: tl.constexpr,
                B_STATIC: tl.constexpr, TRIGGER_FIRST: tl.constexpr):
    offs_n = tl.program_id(0) * BN + tl.arange(0, BN)
    w_rows = w_ptr + offs_n.to(tl.int64)[:, None] * K  # a row can start past 2^31 elements
    x_rows = x_ptr + offs_n[:, None] * 0
    kk = tl.arange(0, BK)
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


def config(N, K):
    """(BN, BK, UNROLL, num_warps, trigger first): measured for the tuned shapes, else derived from N and K."""
    if (N, K) in _TUNED:
        return _TUNED[(N, K)]
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
    bn, bk, unroll, num_warps, trigger_first = config(N, K)
    gemv_kernel[(N // bn, )](a, b.T, c, K=K, BN=bn, BK=bk, UNROLL=unroll, B_STATIC=b_static,
                             TRIGGER_FIRST=trigger_first, num_warps=num_warps, launch_pdl=True)
    return c
