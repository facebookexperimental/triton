"""The Triton kernels that reproduce cuBLAS's arithmetic, and their launchers.

Nothing here decides anything.  A `CublasGemmPlan` names one of these and its parameters; this
module only implements the orders.  There is no architecture in this file: the one hardware
constant it used to read from an `ArchProfile` -- the BM floor below which Triton stops using
the native fp8 tensor-core path -- is now an argument (`fp8_min_bm`) the caller passes in.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .ltapi import DEVICE
import contextlib
from .arch import platform
import math


# --------------------------------------------------------------------------- #
# Triton kernels (scale-capable, single FP32 accumulator)
# --------------------------------------------------------------------------- #
@triton.jit
def _plain_gemm(A, B, C, M, N, K, am, ak, bk, bn, cm, cn, s, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    # int64 indices: an operand with more than 2^31 elements (e.g. M=8192, K=300000) overflows
    # 32-bit address arithmetic and faults. Every kernel here does the same cast.
    om = ((pm * BM + tl.arange(0, BM)) % M).to(tl.int64)
    on = ((pn * BN + tl.arange(0, BN)) % N).to(tl.int64)
    ok = tl.arange(0, BK).to(tl.int64)
    ap = A + om[:, None] * am + ok[None, :] * ak
    bp = B + ok[:, None] * bk + on[None, :] * bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(ap, mask=ok[None, :] < K - k * BK, other=0.0),
                     tl.load(bp, mask=ok[:, None] < K - k * BK, other=0.0), acc)
        ap += BK * ak
        bp += BK * bk
    acc = acc * s
    ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], acc.to(C.dtype.element_ty),
             mask=(ocm[:, None] < M) & (ocn[None, :] < N))


@triton.jit
def _plain_gemm_k_per_dot(A, B, C, M, N, K, am, ak, bk, bn, cm, cn, s, KPD, RES, BM: tl.constexpr, BN: tl.constexpr,
                          BK: tl.constexpr):
    """Plain GEMM that rounds its accumulator exactly where cuBLAS's CUTLASS kernel does.

    A non-aligned shape sends cuBLAS to a CUTLASS kernel whose MMA takes 8 real k-elements
    (`s1688`) or 16 (`s16816`, `s161616`), and which updates the fp32 accumulator once per
    MMA. Two things are needed to match it. (a) A k16 MMA whose upper k-lanes are exactly zero
    is bit-identical to a k8 MMA, so Triton never has to emit `m16n8k8` -- which it cannot do
    anyway; we zero-pad each dot's k group to BK with a load mask. (b) The accumulator has to round at
    the same k boundaries, so we consume exactly KPD real k-elements per `tl.dot`.

    Where the partial group sits is the subtle part. CUTLASS handles a K that is not a whole
    number of mainloop block_k steps in its FIRST iteration, but the MMAs inside that iteration
    still march on the tile's own grid from 0, so the partial MMA lands at index
    `floor(RES/KPD)`, not at position 0. RES is `K % block_k` for the kernel's block_k (32, 64 or
    128). The accumulator therefore rounds at

        0, KPD, 2*KPD, ..., floor(RES/KPD)*KPD, RES, RES+KPD, ..., K

    Putting the whole leftover at position 0 instead is only equivalent when `RES < KPD`,
    which is why that earlier version matched just a third of these shapes. KPD and RES are
    runtime args so one compiled kernel serves every combination."""
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    om = ((pm * BM + tl.arange(0, BM)) % M).to(tl.int64)
    on = ((pn * BN + tl.arange(0, BN)) % N).to(tl.int64)
    ok = tl.arange(0, BK).to(tl.int64)
    pre = RES // KPD  # whole groups that precede the partial one
    part = RES - pre * KPD  # size of the partial group, 0 when RES is a multiple of KPD
    has_part = tl.where(part > 0, 1, 0)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(0, pre + has_part + (K - RES) // KPD):
        is_pre = g < pre
        is_part = (part > 0) & (g == pre)
        k0 = tl.where(is_pre, g * KPD, tl.where(is_part, pre * KPD, RES + (g - pre - has_part) * KPD))
        klen = tl.where(is_part, part, KPD)
        real = ok < klen
        a = tl.load(A + om[:, None] * am + (k0 + ok)[None, :] * ak, mask=real[None, :], other=0.0)
        b = tl.load(B + (k0 + ok)[:, None] * bk + on[None, :] * bn, mask=real[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
    acc = acc * s
    ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], acc.to(C.dtype.element_ty),
             mask=(ocm[:, None] < M) & (ocn[None, :] < N))


@triton.jit
def _splitk_partial(A, B, W, M, N, K, am, ak, bk, bn, ws, wm, wn, CHUNK, BM: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr):
    """Pass 1: slice `sk` covers [sk*CHUNK, min((sk+1)*CHUNK, K)) -> one FP32 partial.
    CHUNK is a runtime arg, so one compiled kernel serves any num_split / granularity."""
    pid = tl.program_id(0)
    sk = tl.program_id(1)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    om = ((pm * BM + tl.arange(0, BM)) % M).to(tl.int64)
    on = ((pn * BN + tl.arange(0, BN)) % N).to(tl.int64)
    k0 = sk * CHUNK
    klen = tl.minimum(CHUNK, K - k0)
    ok = tl.arange(0, BK).to(tl.int64)
    ap = A + om[:, None] * am + (k0 + ok)[None, :] * ak
    bp = B + (k0 + ok)[:, None] * bk + on[None, :] * bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(0, tl.cdiv(CHUNK, BK)):
        km = (ki * BK + ok) < klen
        acc = tl.dot(tl.load(ap, mask=km[None, :], other=0.0), tl.load(bp, mask=km[:, None], other=0.0), acc)
        ap += BK * ak
        bp += BK * bk
    ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    tl.store(W + sk.to(tl.int64) * ws + ocm[:, None] * wm + ocn[None, :] * wn, acc,
             mask=(ocm[:, None] < M) & (ocn[None, :] < N))


@triton.jit
def _splitk_partial_blocks(A, B, W, M, N, K, am, ak, bk, bn, ws, wm, wn, CHUNK, BLOCK_K, BM: tl.constexpr,
                           BN: tl.constexpr, BK: tl.constexpr):
    """Pass 1 with a SECOND accumulation level at the kernel's own threadblock k step.

    Slice `sk` covers [sk*CHUNK, min((sk+1)*CHUNK, K)) as in `_splitk_partial`, but inside the
    slice each BLOCK_K-long block is summed into its OWN fp32 accumulator and the block totals
    are then added forward, instead of one flat accumulator running the whole slice. Two
    forward chains, nested.

    That is not a refinement of the flat form, it is a different sum. The two agree exactly
    while a slice fits one block -- which is why the flat form matched cuBLAS on H100 fp8 up to
    K = 128 and missed every K above it, even multiples of 128.

    CHUNK and BLOCK_K are runtime args so one compiled kernel serves every combination."""
    pid = tl.program_id(0)
    sk = tl.program_id(1)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    om = ((pm * BM + tl.arange(0, BM)) % M).to(tl.int64)
    on = ((pn * BN + tl.arange(0, BN)) % N).to(tl.int64)
    k0 = sk * CHUNK
    kend = tl.minimum(k0 + CHUNK, K)
    ok = tl.arange(0, BK).to(tl.int64)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for blk in range(0, tl.cdiv(CHUNK, BLOCK_K)):
        b0 = k0 + blk * BLOCK_K
        btop = tl.minimum(b0 + BLOCK_K, kend)
        sub = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(0, tl.cdiv(BLOCK_K, BK)):
            kk = b0 + ki * BK + ok
            real = kk < btop
            a = tl.load(A + om[:, None] * am + kk[None, :] * ak, mask=real[None, :], other=0.0)
            b = tl.load(B + kk[:, None] * bk + on[None, :] * bn, mask=real[:, None], other=0.0)
            sub = tl.dot(a, b, sub)
        acc = tl.where(blk == 0, sub, acc + sub)
    ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    tl.store(W + sk.to(tl.int64) * ws + ocm[:, None] * wm + ocn[None, :] * wn, acc,
             mask=(ocm[:, None] < M) & (ocn[None, :] < N))


@triton.jit
def _splitk_combine(W, C, M, N, ws, wm, wn, cm, cn, S, s, BM: tl.constexpr, BN: tl.constexpr):
    """Pass 2: sum the S FP32 partials in FORWARD order (slice 0..S-1), scale, cast. The sum
    starts at partial 0 rather than a zero accumulator, which matters only for signed zero:
    `(+0.0) + (-0.0) = +0.0` would drop the sign a cuBLAS epilogue with beta=0 keeps."""
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    om = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    on = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    m2 = (om[:, None] < M) & (on[None, :] < N)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for i in range(0, S):
        p = tl.load(W + i.to(tl.int64) * ws + om[:, None] * wm + on[None, :] * wn, mask=m2, other=0.0)
        acc = tl.where(i == 0, p, acc + p)
    acc = acc * s
    tl.store(C + cm * om[:, None] + cn * on[None, :], acc.to(C.dtype.element_ty), mask=m2)


@triton.jit
def _splitk_partial_k_per_dot(A, B, W, M, N, K, am, ak, bk, bn, ws, wm, wn, CHUNK, KPD, BLOCK_K, BM: tl.constexpr,
                              BN: tl.constexpr, BK: tl.constexpr):
    """Pass 1 for the CUTLASS split-K path: slice `sk` covers [sk*CHUNK, min((sk+1)*CHUNK, K))
    and accumulates it in groups of KPD real k-elements, with the slice's own residue tile.

    Same idea as `_plain_gemm_k_per_dot` but applied PER SLICE: a CUTLASS slice whose length is
    not a whole number of threadblock block_k steps runs a partial first tile of `klen % BLOCK_K`
    elements, and the KPD-sized MMAs inside that tile start at its beginning, so the short
    group lands at the END of the residue tile -- in the middle of the slice, not at either
    end. Putting it first is only equivalent when the residue is under one group."""
    pid = tl.program_id(0)
    sk = tl.program_id(1)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    om = ((pm * BM + tl.arange(0, BM)) % M).to(tl.int64)
    on = ((pn * BN + tl.arange(0, BN)) % N).to(tl.int64)
    k0 = sk * CHUNK
    klen = tl.minimum(CHUNK, K - k0)
    rbk = klen % BLOCK_K
    rbk = tl.minimum(tl.where(rbk == 0, BLOCK_K, rbk), klen)  # length of the leading residue tile
    nfirst = tl.cdiv(rbk, KPD)
    ok = tl.arange(0, BK).to(tl.int64)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(0, nfirst + (klen - rbk) // KPD):
        off = tl.where(g < nfirst, g * KPD, rbk + (g - nfirst) * KPD)
        glen = tl.where(g < nfirst, tl.minimum(KPD, rbk - g * KPD), KPD)
        real = ok < glen
        kk = k0 + off + ok
        a = tl.load(A + om[:, None] * am + kk[None, :] * ak, mask=real[None, :], other=0.0)
        b = tl.load(B + kk[:, None] * bk + on[None, :] * bn, mask=real[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
    ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    tl.store(W + sk.to(tl.int64) * ws + ocm[:, None] * wm + ocn[None, :] * wn, acc,
             mask=(ocm[:, None] < M) & (ocn[None, :] < N))


@triton.jit
def _splitk_combine_modes(W, C, M, N, ws, wm, wn, cm, cn, S, s, CMODE: tl.constexpr, BM: tl.constexpr,
                          BN: tl.constexpr):
    """Pass 2 with the three merge schemes cuBLAS actually uses, which `SPLITK_NUM` does not
    distinguish -- only the launched kernel names do:

      CMODE 0  fp32 workspace, forward fp32 sum                     (`splitKreduce<..., float>`)
      CMODE 2  partials rounded to the output dtype, then fp32 sum  (`splitKreduce<..., __half>`)
      CMODE 3  serial chain kept in the output dtype between slices (`GemmSplitKSerial`)

    The rounding is to the OUTPUT dtype, not to fp16. Hard-coding fp16 here was invisible for a
    long time because only fp16 was ever fuzzed; it made every bf16 shape on this path wrong.

    Measured shares of the non-aligned split-K family: CMODE 3 is 70%, CMODE 2 17%, CMODE 0 12%
    -- so the fp32 assumption the aligned path uses is the minority case here.

    The running sum starts AT partial 0 rather than at a zero accumulator. That matters only
    for signed zero -- `(+0.0) + (-0.0) = +0.0` would destroy a `-0.0` partial, while a CUTLASS
    epilogue with beta=0 writes the value out and keeps it -- and it never loses (169 wins, 0
    losses over 1600 recipe pairs)."""
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    om = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    on = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    m2 = (om[:, None] < M) & (on[None, :] < N)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for i in range(0, S):
        p = tl.load(W + i.to(tl.int64) * ws + om[:, None] * wm + on[None, :] * wn, mask=m2, other=0.0)
        if CMODE == 2:  # the GEMM wrote an output-dtype workspace, so each partial is rounded once
            p = p.to(C.dtype.element_ty).to(tl.float32)
        acc = tl.where(i == 0, p, acc + p)
        if CMODE == 3:  # the running total lives in the output dtype between slices
            acc = acc.to(C.dtype.element_ty).to(tl.float32)
    acc = acc * s
    tl.store(C + cm * om[:, None] + cn * on[None, :], acc.to(C.dtype.element_ty), mask=m2)


# --------------------------------------------------------------------------- #
# Triton kernels for the CUDA-core (SIMT) families
#
# None of these may use `tl.dot`. cuBLAS runs them on the CUDA cores, one fp32 accumulator per
# output element; a tensor-core dot is off by 1 ulp against that even at K == 2, so no
# regrouping of a `tl.dot` can ever match and the products have to be written out by hand.
# fp16 x fp16 and bf16 x bf16 are exact in fp32, so `fma` and (multiply, then add) give the
# same bits and only the association order matters.
# --------------------------------------------------------------------------- #
@triton.jit
def _simt_chain_gemm(A, B, C, M, N, K, CH, SUB, am, ak, bk, bn, cm, cn, BM: tl.constexpr, BN: tl.constexpr,
                     S: tl.constexpr):
    """`gemmSN_NN_kernel` (ALGO_ID 11) and `magma_sgemmEx_kernel` (16): three accumulation
    levels, all of them left to right.

        level 1  an ascending chain inside a sub-block of SUB k
        level 2  the sub-block totals, added into the thread partial
        level 3  the S thread partials

    S threads share one output column, so thread t owns the contiguous k chunk
    [t*CH, min((t+1)*CH, K)) with CH = ceil(K/S), and splits it into SUB-long sub-blocks.
    Level 2 only bites once a chunk is longer than SUB, i.e. K > S*SUB = 512 for ALGO 11;
    below that there is one sub-block per thread and this collapses to a plain S-way split.
    ALGO 16 is S = 1 with SUB = K: a single ascending chain.

    The tile shape is a pure performance knob -- every output element is its own independent
    reduction, so BM, BN and num_warps cannot change a single bit."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    on = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    mm = om < M
    mn = on < N
    total = tl.zeros((BM, BN), dtype=tl.float32)
    for t in tl.static_range(S):
        lo = t * CH
        hi = tl.minimum(lo + CH, K)
        part = tl.zeros((BM, BN), dtype=tl.float32)
        for base in range(lo, hi, SUB):
            top = tl.minimum(base + SUB, hi)
            sub = tl.zeros((BM, BN), dtype=tl.float32)
            for k in range(base, top):
                kk = k.to(tl.int64)
                av = tl.load(A + om * am + kk * ak, mask=mm, other=0.0).to(tl.float32)
                bv = tl.load(B + kk * bk + on * bn, mask=mn, other=0.0).to(tl.float32)
                sub += av[:, None] * bv[None, :]
            part += sub
        total += part
    tl.store(C + om[:, None] * cm + on[None, :] * cn, total.to(C.dtype.element_ty), mask=mm[:, None] & mn[None, :])


@triton.jit
def _bitrev(idx, W: tl.constexpr):
    """Reverse the log2(W) low bits of `idx`. Reversing them turns a count-down butterfly
    (offset W/2 first, the classic `__shfl_down` reduction) into a neighbour-pair tree, which
    is the shape Triton can express with reshape + split."""
    r = idx * 0
    if W >= 2:
        r += (idx & 1) * (W // 2)
    if W >= 4:
        r += ((idx >> 1) & 1) * (W // 4)
    if W >= 8:
        r += ((idx >> 2) & 1) * (W // 8)
    if W >= 16:
        r += ((idx >> 3) & 1) * (W // 16)
    if W >= 32:
        r += ((idx >> 4) & 1) * (W // 32)
    if W >= 64:
        r += ((idx >> 5) & 1) * (W // 64)
    if W >= 128:
        r += ((idx >> 6) & 1) * (W // 128)
    return r


@triton.jit
def _pair_fold(x, BE: tl.constexpr, n: tl.constexpr):
    """Add neighbouring columns: (BE, n) -> (BE, n // 2)."""
    lo, hi = tl.split(tl.reshape(x, (BE, n // 2, 2)))
    return lo + hi


@triton.jit
def _butterfly_down(x, BE: tl.constexpr, W: tl.constexpr):
    """The count-down butterfly over W lane values, as a chain of neighbour-pair folds. The
    columns of `x` must already be in bit-reversed lane order (see `_bitrev`)."""
    if W >= 128:
        x = _pair_fold(x, BE, 128)
    if W >= 64:
        x = _pair_fold(x, BE, 64)
    if W >= 32:
        x = _pair_fold(x, BE, 32)
    if W >= 16:
        x = _pair_fold(x, BE, 16)
    if W >= 8:
        x = _pair_fold(x, BE, 8)
    if W >= 4:
        x = _pair_fold(x, BE, 4)
    if W >= 2:
        x = _pair_fold(x, BE, 2)
    return tl.reshape(x, (BE, ))


@triton.jit
def _gemv_lane(A, B, Cout, NEL, K0, KE, sa_e, sa_k, sb_k, sb_e, sc, NC, CC, V: tl.constexpr, W: tl.constexpr,
               DOWN: tl.constexpr, BE: tl.constexpr):
    """`gemv2T_kernel` / `gemv2N_kernel` (ALGO_ID 13) over the k range [K0, KE).

    W lanes cooperate on one output element and a lane loads V consecutive k at a time, so
    lane l owns, inside tile t, the V k at t*V*W + l*V. CC tiles form a chunk: a chunk is
    summed on its own in (tile, sub) order and the chunk totals are then added left to right.
    The W lane totals combine either as a count-down butterfly (DOWN) or left to right.

    One program handles BE output elements; that is a performance knob only, because the
    elements are independent reductions."""
    pe = tl.program_id(0)
    oe = (pe * BE + tl.arange(0, BE)).to(tl.int64)
    em = oe < NEL
    lid = tl.arange(0, W)
    # Column c of the accumulator holds lane `slot[c]`: the lanes in order for the sequential
    # combine, bit-reversed for the butterfly.
    if DOWN:
        slot = _bitrev(lid, W)
    else:
        slot = lid
    lane = slot.to(tl.int64) * V
    acc = tl.zeros((BE, W), dtype=tl.float32)
    for c in range(0, NC):
        cacc = tl.zeros((BE, W), dtype=tl.float32)
        base = K0 + c * CC * V * W
        for t in range(0, CC):
            tb = base + t * V * W
            for s in tl.static_range(V):
                k = tb + lane + s
                km = k < KE
                a = tl.load(A + oe[:, None] * sa_e + k[None, :] * sa_k, mask=em[:, None] & km[None, :], other=0.0)
                bb = tl.load(B + k[None, :] * sb_k + oe[:, None] * sb_e, mask=em[:, None] & km[None, :], other=0.0)
                cacc += a.to(tl.float32) * bb.to(tl.float32)
        acc = tl.where(c == 0, cacc, acc + cacc)
    if DOWN:
        tot = _butterfly_down(acc, BE, W)
    else:
        tot = tl.zeros((BE, ), dtype=tl.float32)
        for j in tl.static_range(W):
            v = tl.sum(tl.where(lid[None, :] == j, acc, 0.0), axis=1)
            tot = v if j == 0 else tot + v
    tl.store(Cout + oe * sc, tot.to(Cout.dtype.element_ty), mask=em)


@triton.jit
def _gemv_cslice(A, B, Cout, NEL, K, sa_e, sa_k, sb_k, sb_e, sc, S, NCH, C: tl.constexpr, W: tl.constexpr,
                 BE: tl.constexpr):
    """`gemv2N_kernel<..., 128, 32, 4, 4, 1, false>` (ALGO_ID 13, `CUSTOM_OPTION` 5).

    Same algo as `_gemv_lane` but a different lane layout, which is why it needs its own
    kernel. 128 threads and 4 output elements per CTA, so W = 32 lanes per element, and lane l
    takes the CONTIGUOUS k slice [l*S, l*S + S) with S = ceil(K/W) -- not the strided tile the
    other `CUSTOM_OPTION` values use. Inside its slice the lane sums chunks of C = 16 k, each
    an ascending chain, and adds the chunk totals left to right; the W lane totals then combine
    left to right too.

    Worked example at K = 581, so S = 19: lane 0 is (x0 + ... + x15) + (x16 + x17 + x18) and
    lane 1 starts at k = 19. The chunk boundary sits at 15, 31, 47 for every S measured."""
    pe = tl.program_id(0)
    oe = (pe * BE + tl.arange(0, BE)).to(tl.int64)
    em = oe < NEL
    base = tl.arange(0, W).to(tl.int64) * S
    acc = tl.zeros((BE, W), dtype=tl.float32)
    for ch in range(0, NCH):
        cacc = tl.zeros((BE, W), dtype=tl.float32)
        for s in tl.static_range(C):
            off = ch * C + s
            k = base + off
            km = (k < K) & (off < S)
            a = tl.load(A + oe[:, None] * sa_e + k[None, :] * sa_k, mask=em[:, None] & km[None, :], other=0.0)
            bb = tl.load(B + k[None, :] * sb_k + oe[:, None] * sb_e, mask=em[:, None] & km[None, :], other=0.0)
            cacc += a.to(tl.float32) * bb.to(tl.float32)
        acc = tl.where(ch == 0, cacc, acc + cacc)
    lid = tl.arange(0, W)
    tot = tl.zeros((BE, ), dtype=tl.float32)
    for j in tl.static_range(W):
        v = tl.sum(tl.where(lid[None, :] == j, acc, 0.0), axis=1)
        tot = v if j == 0 else tot + v
    tl.store(Cout + oe * sc, tot.to(Cout.dtype.element_ty), mask=em)


@triton.jit
def _gemv_slice_combine(Wsp, Cout, NEL, ws, sc, S, BE: tl.constexpr):
    """ALGO_ID 13 split-K: add the S fp32 slice partials left to right, then cast."""
    pe = tl.program_id(0)
    oe = (pe * BE + tl.arange(0, BE)).to(tl.int64)
    em = oe < NEL
    tot = tl.zeros((BE, ), dtype=tl.float32)
    for i in range(0, S):
        v = tl.load(Wsp + i.to(tl.int64) * ws + oe, mask=em, other=0.0)
        tot = tl.where(i == 0, v, tot + v)
    tl.store(Cout + oe * sc, tot.to(Cout.dtype.element_ty), mask=em)


@triton.jit
def _gemv_block_dot(A, B, Wsp, NEL, K, sa_e, sa_k, sb_k, sb_e, ws, NB, PER, V: tl.constexpr, BE: tl.constexpr):
    """`dot_kernel` (ALGO_ID 14), pass 1: one fp32 partial per block into the workspace.

    Three things differ from ALGO 13 and all three change the bits. Tiles of V*128 k are owned
    by blocks STRIDED by NB = SPLITK_NUM, not cut into contiguous slices. A thread keeps V
    separate accumulator chains -- accumulator s sums x[i*TILE + t*V + s] over the block's
    tiles i ascending -- and they are only joined at the end. The 128 thread values then
    combine as a count-down butterfly."""
    pe = tl.program_id(0)
    blk = tl.program_id(1)
    oe = (pe * BE + tl.arange(0, BE)).to(tl.int64)
    em = oe < NEL
    thr = _bitrev(tl.arange(0, 128), 128).to(tl.int64)  # column s holds thread bitrev(s)
    accs = tl.zeros((BE, 128), dtype=tl.float32)
    for s in tl.static_range(V):
        acc = tl.zeros((BE, 128), dtype=tl.float32)
        for g in range(0, PER):
            k = (g * NB + blk) * (V * 128) + thr * V + s
            km = k < K
            a = tl.load(A + oe[:, None] * sa_e + k[None, :] * sa_k, mask=em[:, None] & km[None, :], other=0.0)
            bb = tl.load(B + k[None, :] * sb_k + oe[:, None] * sb_e, mask=em[:, None] & km[None, :], other=0.0)
            acc += a.to(tl.float32) * bb.to(tl.float32)
        accs += acc
    tl.store(Wsp + blk.to(tl.int64) * ws + oe, _butterfly_down(accs, BE, 128), mask=em)


@triton.jit
def _gemv_block_reduce(Wsp, Cout, NEL, ws, sc, NB, MM, BE: tl.constexpr):
    """`reduce_1Block_kernel<float, 128, 7>` (ALGO_ID 14), pass 2: 128 threads, i.e. 4 warps.

        q[t]  = p[t] + p[t+128] + p[t+256] + ...    t = 0..127, ascending, missing entries 0
        q[t] += q[t+64]                             butterfly across two warps
        q[t] += q[t+32]                             butterfly across one warp
        u[j]  = q[j] + q[4+j] + ... + q[28+j]       j = 0..3, ascending
        total = ((u0 + u1) + u2) + u3

    Both butterfly folds add exact zeros while NB <= 32, so reading the partials as a flat
    [NB/4][4] sequence -- what this did before -- is right only there. At NB = 64 it is not:
    the flat read gives u0 = p0 + p4 + ... + p60, the kernel gives
    u0 = (p0 + p32) + (p4 + p36) + ... + (p28 + p60).

    The two folds are the top two steps of a count-down butterfly, so they are two neighbour-
    pair folds once the columns are loaded in the right order: column c holds thread
    c // 4 + {0, 64, 32, 96}[c % 4], which puts each fold's two operands side by side.
    MM = ceil(NB / 128) is how many stride-128 passes the load loop makes."""
    pe = tl.program_id(0)
    oe = (pe * BE + tl.arange(0, BE)).to(tl.int64)
    em = oe < NEL
    c = tl.arange(0, 128)
    thr = c // 4 + (c & 1) * 64 + ((c >> 1) & 1) * 32
    q = tl.zeros((BE, 128), dtype=tl.float32)
    for m in range(0, MM):
        i = m * 128 + thr
        v = tl.load(Wsp + i.to(tl.int64)[None, :] * ws + oe[:, None], mask=em[:, None] & (i < NB)[None, :], other=0.0)
        q = tl.where(m == 0, v, q + v)
    # Column l now holds (q[l] + q[l+64]) + (q[l+32] + q[l+96]). Split it so that axis 1 is the
    # group g and axis 2 the component j of `u[j] = sum over g of lane[4g + j]`.
    lanes = tl.reshape(_pair_fold(_pair_fold(q, BE, 128), BE, 64), (BE, 8, 4))
    gid = tl.arange(0, 8)[None, :, None]
    u = tl.sum(tl.where(gid == 0, lanes, 0.0), axis=1)
    for g in tl.static_range(1, 8):
        u += tl.sum(tl.where(gid == g, lanes, 0.0), axis=1)
    tot = tl.zeros((BE, ), dtype=tl.float32)
    for j in tl.static_range(4):
        v = tl.sum(tl.where(tl.arange(0, 4)[None, :] == j, u, 0.0), axis=1)
        tot = v if j == 0 else tot + v
    tl.store(Cout + oe * sc, tot.to(Cout.dtype.element_ty), mask=em)


def _tile(M: int, N: int, dtype=None, fp8_min_bm: int = 64) -> tuple[int, int]:
    """Partial tile: small tiles for small M,N, capped at 128.

    Bit-neutral EXCEPT for one hard constraint: fp8 needs BM >= 64. Below that Triton
    cannot use the native fp8 tensor-core path -- it upcasts fp8 to f16 and emits
    `mma.sync.m16n8k16` instead of `tcgen05.mma.kind::f8f6f4`, which rounds differently
    from cuBLAS. (fp16/bf16 agree on both paths, so they are free to use a small tile.)"""
    BM = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
    BN = 16 if N <= 16 else 32 if N <= 32 else 64 if N <= 64 else 128
    if dtype == torch.float8_e4m3fn:
        BM = max(BM, fp8_min_bm)
    return BM, BN


def _kcontig(b):
    return b if b.stride(1) == 1 else b.contiguous()


def _triton_plain(a, b, out_dtype, scale=1.0, BM=128, BN=128, BK=64):
    if _is_sm103_plain():
        M, N = a.shape[0], b.shape[1]
        BM, BN, nw, ns = _sm103_config(M, N, a.dtype == torch.float8_e4m3fn)
        return _triton_plain_tma_sm103(a, b, out_dtype, scale=scale, BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    b = _kcontig(b)
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), )
    _plain_gemm[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                      scale, BM=BM, BN=BN, BK=BK, num_warps=8)
    return c


def _triton_plain_k_per_dot(a, b, out_dtype, k_per_dot, res, scale=1.0, BM=128, BN=128, BK=16):
    """Plain GEMM accumulating `k_per_dot` real k-elements per dot, with the partial group
    ending at `res`, so the accumulator rounds where cuBLAS's CUTLASS kernel rounds. The tile
    and the MMA Triton picks are bit-neutral here (measured 2000/2000 both ways), so BM, BN,
    BK and num_warps are free."""
    if _is_sm103_k_per_dot():
        c = _triton_plain_k_per_dot_sm103(a, b, out_dtype, k_per_dot, res, scale)
        if c is not None:
            return c
    b = _kcontig(b)
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), )
    _plain_gemm_k_per_dot[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                                c.stride(1), scale, k_per_dot, res, BM=BM, BN=BN, BK=BK, num_warps=8)
    return c


def _triton_splitk_groups(a, b, chunk, k_per_dot, block_k, cmode, out_dtype, scale=1.0, BK=16, fp8_min_bm=64):
    """Two-pass split-K for the CUTLASS path: `chunk`-element slices, each accumulated in
    groups of `k_per_dot` real k with a per-slice residue tile of `block_k`, merged by `cmode`."""
    if _is_sm103_splitk_groups():
        c = _triton_splitk_groups_sm103(a, b, chunk, k_per_dot, block_k, cmode, out_dtype, scale, fp8_min_bm)
        if c is not None:
            return c
    b = _kcontig(b)
    M, K = a.shape
    N = b.shape[1]
    nsplit = (K + chunk - 1) // chunk
    if nsplit < 1 or nsplit > 8192:
        return None
    BM, BN = _tile(M, N, a.dtype, fp8_min_bm)
    ntile = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    w = torch.empty(nsplit, M, N, device=DEVICE, dtype=torch.float32)
    _splitk_partial_k_per_dot[(ntile, nsplit)](a, b, w, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                               w.stride(0), w.stride(1), w.stride(2), chunk, k_per_dot, block_k, BM=BM,
                                               BN=BN, BK=BK, num_warps=4)
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    _splitk_combine_modes[(ntile, )](w, c, M, N, w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1),
                                     nsplit, scale, CMODE=cmode, BM=BM, BN=BN, num_warps=4)
    return c


def _triton_splitk(a, b, chunk, out_dtype, scale=1.0, BK=64, fp8_min_bm=64):
    """Deterministic two-pass split-K at `chunk`-element early slices (uneven tail),
    forward FP32 combine. Returns None if the chunk does not yield a real split (nsplit<2)."""
    if _split_is_sm103():
        return _triton_splitk_fast(a, b, chunk, out_dtype, scale, BK, fp8_min_bm)
    b = _kcontig(b)
    M, K = a.shape
    N = b.shape[1]
    nsplit = (K + chunk - 1) // chunk
    if nsplit < 2 or nsplit > 8192:
        return None
    BM, BN = _tile(M, N, a.dtype, fp8_min_bm)
    ntile = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    w = torch.empty(nsplit, M, N, device=DEVICE, dtype=torch.float32)
    _splitk_partial[(ntile, nsplit)](a, b, w, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), w.stride(0),
                                     w.stride(1), w.stride(2), chunk, BM=BM, BN=BN, BK=BK, num_warps=4)
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    _splitk_combine[(ntile, )](w, c, M, N, w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1), nsplit,
                               scale, BM=BM, BN=BN, num_warps=4)
    return c


def _triton_splitk_blocks(a, b, chunk, block_k, out_dtype, scale=1.0, BK=32, fp8_min_bm=64):
    """Two nested forward chains: `chunk`-long slices, each the forward sum of its `block_k`-long
    block totals, and the slice partials added forward in fp32. `chunk == K` is the
    SPLITK_NUM 1 case -- one slice, so only the block level is left."""
    b = _kcontig(b)
    M, K = a.shape
    N = b.shape[1]
    nsplit = (K + chunk - 1) // chunk
    if nsplit < 1 or nsplit > 8192:
        return None
    BM, BN = _tile(M, N, a.dtype, fp8_min_bm)
    ntile = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    w = torch.empty(nsplit, M, N, device=DEVICE, dtype=torch.float32)
    _splitk_partial_blocks[(ntile, nsplit)](a, b, w, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                            w.stride(0), w.stride(1), w.stride(2), chunk, block_k, BM=BM, BN=BN, BK=BK,
                                            num_warps=4)
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    _splitk_combine[(ntile, )](w, c, M, N, w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1), nsplit,
                               scale, BM=BM, BN=BN, num_warps=4)
    return c


def _triton_gemmsn(a, b, out_dtype, split, sub):
    """ALGO_ID 11 and 16: `split` contiguous k chunks, each accumulated in `sub`-long
    sub-blocks (`sub` 0 = one sub-block for the whole chunk, which is ALGO 16)."""
    M, K = a.shape
    N = b.shape[1]
    ch = -(-K // split)
    # `_simt_chain_gemm_wide` carries one accumulator plane per chunk, so it needs `split` to be
    # a power of two (`tl.arange`) and every chunk to be non-empty. The second only fails for K
    # below about S*(S-1), where there is nothing to win anyway.
    if _sm103_gemmsn() and split & (split - 1) == 0 and (split - 1) * ch < K:
        BM, BN, warps, unroll = _GEMMSN_WIDE_TILE
        sb = sub or K
        c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        _simt_chain_gemm_wide[grid](a, b, c, M, N, K, ch, K - (split - 1) * ch, -(-ch // sb), sb, a.stride(0),
                                    a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BM=BM, BN=BN,
                                    S=split, U=unroll, num_warps=warps, num_stages=1)
        return c
    BM = min(128, max(16, triton.next_power_of_2(M)))
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    _simt_chain_gemm[(triton.cdiv(M, BM), triton.cdiv(N, 64))](a, b, c, M, N, K, -(-K // split), sub or K, a.stride(0),
                                                               a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                                                               c.stride(1), BM=BM, BN=64, S=split, num_warps=4)
    return c


def _gemv_axis(a, b, c):
    """A gemv has one output per row (N == 1) or per column (M == 1). Returns the number of
    output elements and the strides that walk the two operands and the output along it."""
    if b.shape[1] == 1:
        return a.shape[0], a.stride(0), a.stride(1), b.stride(0), 0, c.stride(0)
    return b.shape[1], 0, a.stride(1), b.stride(0), b.stride(1), c.stride(1)


def _gemv_vlen(v, chunk, W):
    """The k a lane loads at once. A positive `v` is a fixed vector width; `v <= 0` is the "one
    contiguous slice per lane" family, capped at -v (0 = uncapped).

    `chunk` is the k per split-K slice, NOT the length of the slice being launched. The two
    differ in the last slice, and sizing the lane vector from the short tail instead is wrong:
    it costs 40/248 (shape, seed) byte-compares across the capped family, and every one of them
    comes back when the chunk is used. That is also what the hardware must be doing -- cuBLAS
    launches one kernel for the whole split, so one lane layout serves every slice and a short
    tail just runs out of k early."""
    per_lane = -(-chunk // W)
    return v if v > 0 else (per_lane if v == 0 else min(-v, per_lane))


def _launch_gemv_lane(a, b, out, nel, k0, ke, strides, sc, recipe, vlen, BE):
    _, W, CC, down = recipe
    ntile = -(-(ke - k0) // (vlen * W))
    per_chunk = CC or ntile
    _gemv_lane[(triton.cdiv(nel, BE), )](a, b, out, nel, k0, ke, *strides, sc, -(-ntile // per_chunk), per_chunk,
                                         V=vlen, W=W, DOWN=down, BE=BE, num_warps=4)


def _triton_gemv13(a, b, out_dtype, recipe, nsplit, BE=16):
    """ALGO_ID 13. With SPLITK_NUM > 1 the k range is first cut into ceil(K/SPLITK_NUM)-long
    contiguous slices, each reduced with the base order, and the slice partials added left to
    right in fp32."""
    K = a.shape[1]
    c = torch.empty(a.shape[0], b.shape[1], device=DEVICE, dtype=out_dtype)
    nel, *strides, sc = _gemv_axis(a, b, c)
    if _sm103_gemv() and _gemv_vlen(recipe[0], -(-K // nsplit) if nsplit > 1 else K, recipe[1]) == 1:
        # One k per lane per tile only. Above that the merged kernel is not safe to substitute:
        # the frozen kernel is launched once per slice with `NC` and `CC` both equal to 1, which
        # Triton specializes into the code it generates, and on a short partly masked tile that
        # specialized form computes a different fp32 partial from any form that carries the two
        # as runtime values. Measured over 4320 (nel, K, orientation, V, W, CC, DOWN,
        # SPLITK_NUM) cases: 4 differ, all of them `V = -64, W = 4, CC = 0, DOWN` at K = 110,
        # where `ceil(chunk / W)` lands on 2. Rather than fit a guard to those four, anything
        # that does not load exactly one k per lane keeps the shipped per-slice launch.
        v, W = recipe[0], recipe[1]
        chunk = -(-K // nsplit) if nsplit > 1 else K
        vlen = _gemv_vlen(v, chunk, W)
        per_est = recipe[2] or max(1, -(-chunk // (vlen * W)))
        be, warps, uf = _gemv13_shape(nel, nsplit, W, vlen, strides[0] == 0, per_est)
        return _gemv13_launch(a, b, c, nel, strides, sc, recipe, nsplit, be, warps, uf=uf)
    chunk = -(-K // nsplit) if nsplit > 1 else K
    vlen = _gemv_vlen(recipe[0], chunk, recipe[1])
    if nsplit <= 1:
        _launch_gemv_lane(a, b, c, nel, 0, K, strides, sc, recipe, vlen, BE)
        return c
    w = torch.zeros(nsplit, nel, device=DEVICE, dtype=torch.float32)
    for sl in range(nsplit):
        k0 = sl * chunk
        if k0 >= K:
            continue  # the workspace is zeroed, so a slice past K contributes an exact 0
        _launch_gemv_lane(a, b, w[sl], nel, k0, min(k0 + chunk, K), strides, 1, recipe, vlen, BE)
    _gemv_slice_combine[(triton.cdiv(nel, BE), )](w, c, nel, w.stride(0), sc, nsplit, BE=BE, num_warps=4)
    return c


def _triton_gemv_cslice(a, b, out_dtype, w_lanes, chunk, BE=16):
    """ALGO_ID 13 `CUSTOM_OPTION` 5: one contiguous k slice per lane, chunked inside the lane."""
    K = a.shape[1]
    c = torch.empty(a.shape[0], b.shape[1], device=DEVICE, dtype=out_dtype)
    nel, *strides, sc = _gemv_axis(a, b, c)
    slice_k = -(-K // w_lanes)
    if _sm103_gemv():
        be, warps = _CSLICE_LAUNCH
        return _cslice_launch(a, b, c, nel, strides, sc, K, w_lanes, chunk, be, warps)
    _gemv_cslice[(triton.cdiv(nel, BE), )](a, b, c, nel, K, *strides, sc, slice_k, -(-slice_k // chunk), C=chunk,
                                           W=w_lanes, BE=BE, num_warps=4)
    return c


def _triton_gemv14(a, b, out_dtype, nblock, v, BE=16):
    """ALGO_ID 14: `nblock` = SPLITK_NUM strided block partials in an fp32 workspace, merged by
    the 128-thread 4-warp reduce."""
    K = a.shape[1]
    c = torch.empty(a.shape[0], b.shape[1], device=DEVICE, dtype=out_dtype)
    nel, *strides, sc = _gemv_axis(a, b, c)
    ntile = -(-K // (v * 128))
    if _sm103_gemv():
        be, warps, rbe, rwarps, uf = _GEMV14_LAUNCH
        return _gemv14_launch(a, b, c, nel, strides, sc, K, nblock, v, be, warps, rbe, rwarps, uf=uf)
    w = torch.zeros(nblock, nel, device=DEVICE, dtype=torch.float32)
    _gemv_block_dot[(triton.cdiv(nel, BE), nblock)](a, b, w, nel, K, *strides, w.stride(0), nblock, -(-ntile // nblock),
                                                    V=v, BE=BE, num_warps=4)
    _gemv_block_reduce[(triton.cdiv(nel, BE), )](w, c, nel, w.stride(0), sc, nblock, -(-nblock // 128), BE=BE,
                                                 num_warps=4)
    return c


# ---------------------------------------------------------------------------
# sm_103 fast paths, merged from the per-mode tuning runs.
# ---------------------------------------------------------------------------

@triton.jit
def _plain_gemm_sm103(A, B, C, M, N, K, am, ak, bk, bn, cm, cn, s, GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
                      I64: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """`_plain_gemm` for sm_103. Same arithmetic, different launch shape.

    Every difference is addressing or scheduling:
      GROUP_M  visit GROUP_M rows of tiles before moving on, so a B tile is read by GROUP_M
               programs that are resident at the same time instead of by one. Row-major order
               re-streams the whole of B once per tile row.
      EVEN_K   drop the k mask when K is a whole number of BK. The mask only ever contributed
               exact zeros, so the sum is the same one.
      I64      32-bit offsets when neither operand nor the output can reach 2^31 elements.

    The k loop still consumes BK real k-elements per `tl.dot`, in increasing k, into one fp32
    accumulator, and still scales and rounds once at the end. That is the whole bit contract."""
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    if GROUP_M == 1:
        pm = pid // npn
        pn = pid % npn
    else:
        per_group = GROUP_M * npn
        gid = pid // per_group
        first = gid * GROUP_M
        rows = tl.minimum(npm - first, GROUP_M)
        pm = first + (pid % per_group) % rows
        pn = (pid % per_group) // rows
    if I64:
        om = ((pm * BM + tl.arange(0, BM)) % M).to(tl.int64)
        on = ((pn * BN + tl.arange(0, BN)) % N).to(tl.int64)
        ok = tl.arange(0, BK).to(tl.int64)
    else:
        om = (pm * BM + tl.arange(0, BM)) % M
        on = (pn * BN + tl.arange(0, BN)) % N
        ok = tl.arange(0, BK)
    ap = A + om[:, None] * am + ok[None, :] * ak
    bp = B + ok[:, None] * bk + on[None, :] * bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        if EVEN_K:
            acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        else:
            acc = tl.dot(tl.load(ap, mask=ok[None, :] < K - k * BK, other=0.0),
                         tl.load(bp, mask=ok[:, None] < K - k * BK, other=0.0), acc)
        ap += BK * ak
        bp += BK * bk
    acc = acc * s
    if I64:
        ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
        ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    else:
        ocm = pm * BM + tl.arange(0, BM)
        ocn = pn * BN + tl.arange(0, BN)
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], acc.to(C.dtype.element_ty),
             mask=(ocm[:, None] < M) & (ocn[None, :] < N))


@triton.jit
def _plain_gemm_tma_sm103(a_desc, b_desc, c_desc, M, N, K, s, B_NK: tl.constexpr, GROUP_M: tl.constexpr,
                          NUM_SMS: tl.constexpr, SUBTILE: tl.constexpr, WS: tl.constexpr, FLATTEN: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """The same GEMM again, fed by TMA and run as a persistent grid.

    Nothing here touches the arithmetic. The k loop still takes BK real k-elements per `tl.dot`,
    in increasing k, into one fp32 accumulator, and still scales once and rounds once. TMA only
    changes how the operand tiles reach shared memory, warp specialization only changes which
    warp issues the copy, and the persistent loop only changes which program owns a tile.

    `B_NK` says B's descriptor is over the [N,K] weight (the fp8 case, where the caller is handed
    `w.t()`); the load is transposed back before the dot, which reads the same values in the same
    k order. `SUBTILE` splits the epilogue in two along N -- two disjoint sets of output elements,
    so no sum is reassociated; it is worth 9% because the half-width store buffer leaves room for
    a deeper operand pipeline. `FLATTEN` must stay off: flattening the persistent loop drops the
    warp specialization, and the shared memory the pipeline then needs does not fit."""
    start_pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    num_tiles = npm * npn
    per_group = GROUP_M * npn
    tile_id_c = start_pid - NUM_SMS

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WS):
        gid = tile_id // per_group
        first = gid * GROUP_M
        rows = tl.minimum(npm - first, GROUP_M)
        om = (first + (tile_id % per_group) % rows) * BM
        on = ((tile_id % per_group) // rows) * BN
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(k_tiles):
            ok = ki * BK
            a = a_desc.load([om, ok])
            if B_NK:
                acc = tl.dot(a, b_desc.load([on, ok]).T, acc)
            else:
                acc = tl.dot(a, b_desc.load([ok, on]), acc)
        acc = acc * s

        tile_id_c += NUM_SMS
        gidc = tile_id_c // per_group
        firstc = gidc * GROUP_M
        rowsc = tl.minimum(npm - firstc, GROUP_M)
        omc = (firstc + (tile_id_c % per_group) % rowsc) * BM
        onc = ((tile_id_c % per_group) // rowsc) * BN
        if SUBTILE:
            half = tl.permute(tl.reshape(acc, (BM, 2, BN // 2)), (0, 2, 1))
            acc0, acc1 = tl.split(half)
            c_desc.store([omc, onc], acc0.to(c_desc.dtype))
            c_desc.store([omc, onc + BN // 2], acc1.to(c_desc.dtype))
        else:
            c_desc.store([omc, onc], acc.to(c_desc.dtype))


_SM103_plain = None


_SM103_splitk_groups = None


_SM103_k_per_dot = None


_SM103_gemmsn: bool | None = None


def _is_sm103_plain() -> bool:
    """True on GB300. Cached, because it is read once per GEMM call."""
    global _SM103_plain
    if _SM103_plain is None:
        from .arch import platform
        _SM103_plain = platform().name.startswith("sm_103")
    return _SM103_plain


def _is_sm103_splitk_groups():
    """Cached: only sm_103 takes the tuned path, every other GPU keeps the shipped one."""
    global _SM103_splitk_groups
    if _SM103_splitk_groups is None:
        from .arch import platform
        _SM103_splitk_groups = platform().name.startswith("sm_103")
    return _SM103_splitk_groups


def _is_sm103_k_per_dot():
    """Cached: only sm_103 takes the tuned path, every other GPU keeps the shipped one."""
    global _SM103_k_per_dot
    if _SM103_k_per_dot is None:
        from .arch import platform
        _SM103_k_per_dot = platform().name.startswith("sm_103")
    return _SM103_k_per_dot


_I32_MAX = 2**31 - 1


def _fits_i32(a, b, c) -> bool:
    """Can every byte one program addresses be reached with a 32-bit offset? The largest offset
    a kernel forms is (dim - 1) * stride summed over both dims, per tensor."""
    for t in (a, b, c):
        if sum((d - 1) * s for d, s in zip(t.shape, t.stride())) > _I32_MAX:
            return False
    return True


def _triton_plain_sm103(a, b, out_dtype, scale=1.0, BM=128, BN=128, BK=64, GROUP_M=8, num_warps=8, num_stages=3):
    """sm_103 plain GEMM. Same kernel arithmetic, a different launch.

    `_plain_gemm_sm103` reads B through the runtime strides `bk`/`bn`, so a B that is already
    contiguous along k needs no copy -- and an fp8 B arrives that way (`w.t()` of an [N,K]
    weight). The copy the general launcher makes is pure overhead here."""
    if b.stride(0) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), )
    _plain_gemm_sm103[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                            c.stride(1), scale, GROUP_M=GROUP_M, EVEN_K=(K % BK == 0), I64=not _fits_i32(a, b, c),
                            BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages)
    return c


_NUM_SMS = None


def _num_sms() -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    return _NUM_SMS


def _tma_ok(t, block_inner_bytes: int) -> bool:
    """TMA needs the innermost dimension packed and every other stride a whole number of 16-byte
    lines; the block's innermost extent has to be a multiple of 16 bytes too."""
    if t.stride(-1) != 1 or block_inner_bytes % 16:
        return False
    e = t.element_size()
    return all(s * e % 16 == 0 for s in t.stride()[:-1])


@contextlib.contextmanager
def _meta_ws():
    """Compile the next kernel through Meta's warp-specialization passes.

    `warp_specialize=True` on a `tl.range` is refused by the upstream pipeline on sm_103 (the
    partition verifier rejects the predicate the TMA load lowering makes). Meta's passes accept
    the same kernel. The knob is only read while a kernel is being compiled, and it is put back
    on the way out, so nothing outside this launch sees it.

    The one knob is saved and restored by hand rather than through `knobs.nvidia.scope()`:
    `scope()` snapshots the whole nvidia knob group and costs 20 us, which is most of a small
    GEMM's launch, while this is a plain attribute read and write."""
    import triton.knobs
    prev = triton.knobs.nvidia.use_meta_ws
    triton.knobs.nvidia.use_meta_ws = True
    try:
        yield
    finally:
        triton.knobs.nvidia.use_meta_ws = prev


def _triton_plain_tma_sm103(a, b, out_dtype, scale=1.0, BM=128, BN=256, BK=64, GROUP_M=8, num_warps=8, num_stages=3,
                            SUBTILE=True, WS=True, FLATTEN=False, META_WS=True):
    """TMA + persistent + warp-specialized launch of the plain GEMM. Falls back to the plain
    launcher whenever an operand cannot carry a tensor descriptor."""
    from triton.tools.tensor_descriptor import TensorDescriptor
    M, K = a.shape
    N = b.shape[1]
    b_nk = b.stride(0) == 1 and b.stride(1) != 1  # fp8: the caller was handed `w.t()`
    bt = b.t() if b_nk else b
    bb = [BN, BK] if b_nk else [BK, BN]
    cbn = BN // 2 if SUBTILE else BN
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    if not (_tma_ok(a, BK * a.element_size()) and _tma_ok(bt, bb[1] * bt.element_size())
            and _tma_ok(c, cbn * c.element_size())):
        return _triton_plain_sm103(a, b, out_dtype, scale=scale, BM=BM, BN=BN, BK=BK, GROUP_M=GROUP_M)
    a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(bt, bb)
    c_desc = TensorDescriptor.from_tensor(c, [BM, cbn])
    # The persistent loop strides by the program count, so the grid and NUM_SMS must agree.
    nprog = min(_num_sms(), triton.cdiv(M, BM) * triton.cdiv(N, BN))
    grid = (nprog, )
    with (_meta_ws() if META_WS else contextlib.nullcontext()):
        _plain_gemm_tma_sm103[grid](a_desc, b_desc, c_desc, M, N, K, scale, B_NK=b_nk, GROUP_M=GROUP_M,
                                    NUM_SMS=nprog, SUBTILE=SUBTILE, WS=WS, FLATTEN=FLATTEN, BM=BM, BN=BN,
                                    BK=BK, num_warps=num_warps, num_stages=num_stages)
    return c


_SM103_LADDER_FP8 = ((256, 256, 8, 8), (128, 256, 4, 8), (128, 128, 4, 8), (128, 64, 4, 8), (64, 128, 4, 8),
                     (64, 64, 4, 8))


_SM103_LADDER_F16 = ((128, 256, 4, 8), (128, 128, 4, 8), (128, 64, 4, 8), (64, 128, 4, 8), (64, 64, 4, 8))


_SM103_MIN_TILES = 80


_SM103_MIN_WAVE_FILL = 0.6


_SM103_MAX_STEP_DOWN = 2


def _sm103_bm_cap(M: int) -> int:
    """The tallest BM worth using at this M: the next power of two at or above M, never under 64.

    BM is allowed to exceed M. The rows past M are masked, load zeros and contribute nothing --
    the shipped launcher already runs a 128-row tile at every M below 128. Capping BM at M
    instead was a mistake: for 64 < M < 128 it forced two tile rows where one would do, so B was
    read twice over, and on wide N that costs more than TMA and warp specialization win back.

    The floor of 64 is the fp8 constraint from `_tile`: below BM 64 Triton drops
    `tcgen05.mma.kind::f8f6f4` for `mma.sync.m16n8k16`, which rounds differently. Every ladder
    rung is already at BM >= 64, so no shape can be given a smaller one."""
    return max(64, 1 << (M - 1).bit_length())


def _sm103_wave_fill(M: int, N: int, BM: int, BN: int) -> float:
    """Share of the launched programs that still have a tile in the final wave."""
    tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    return tiles / (triton.cdiv(tiles, _num_sms()) * _num_sms())


_SM103_CONFIG_CACHE: dict[tuple, tuple] = {}


def _sm103_config(M: int, N: int, fp8: bool):
    """The largest tile that fills the machine, then one step down if the last wave is half empty.

    A tile choice cannot move a bit: it only says which output elements one program owns, and the
    k loop inside is the same either way. Memoised because it is on the launch path of every
    call and a small GEMM only takes tens of microseconds."""
    hit = _SM103_CONFIG_CACHE.get((M, N, fp8))
    if hit is not None:
        return hit
    cap = _sm103_bm_cap(M)
    ladder = [r for r in (_SM103_LADDER_FP8 if fp8 else _SM103_LADDER_F16) if r[0] <= cap]
    idx = len(ladder) - 1
    for i, (BM, BN, _, _) in enumerate(ladder):
        if triton.cdiv(M, BM) * triton.cdiv(N, BN) >= _SM103_MIN_TILES:
            idx = i
            break
    for _ in range(_SM103_MAX_STEP_DOWN):
        if idx + 1 >= len(ladder):
            break
        here = _sm103_wave_fill(M, N, ladder[idx][0], ladder[idx][1])
        if here >= _SM103_MIN_WAVE_FILL:
            break
        if _sm103_wave_fill(M, N, ladder[idx + 1][0], ladder[idx + 1][1]) <= here:
            break
        idx += 1
    _SM103_CONFIG_CACHE[(M, N, fp8)] = ladder[idx]
    return ladder[idx]


@triton.jit
def _splitk_partial_fast(A, B, W, M, N, K, am, ak, bk, bn, ws, wm, wn, CHUNK, BM: tl.constexpr, BN: tl.constexpr,
                         BK: tl.constexpr):
    """`_splitk_partial` with int32 index arithmetic.

    Statement for statement the same kernel: slice `sk` covers
    [sk*CHUNK, min((sk+1)*CHUNK, K)), the loop runs exactly `cdiv(CHUNK, BK)` times, each pass
    loads BK k-elements with the same out-of-slice mask and zero fill, and one fp32 accumulator
    is chained through the dots in ascending k. Only the width of the integers that compute the
    addresses changes, and an address is not a value that gets added.

    The shipped kernel casts every index to int64 because an operand can exceed 2^31 elements.
    The launcher checks that here instead and only calls this kernel when it cannot happen, which
    halves the registers the address arithmetic needs.
"""
    pid = tl.program_id(0)
    sk = tl.program_id(1)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    om = (pm * BM + tl.arange(0, BM)) % M
    on = (pn * BN + tl.arange(0, BN)) % N
    k0 = sk * CHUNK
    klen = tl.minimum(CHUNK, K - k0)
    ok = tl.arange(0, BK)
    ap = A + om[:, None] * am + (k0 + ok)[None, :] * ak
    bp = B + (k0 + ok)[:, None] * bk + on[None, :] * bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(0, tl.cdiv(CHUNK, BK)):
        km = (ki * BK + ok) < klen
        acc = tl.dot(tl.load(ap, mask=km[None, :], other=0.0), tl.load(bp, mask=km[:, None], other=0.0), acc)
        ap += BK * ak
        bp += BK * bk
    ocm = pm * BM + tl.arange(0, BM)
    ocn = pn * BN + tl.arange(0, BN)
    tl.store(W + sk * ws + ocm[:, None] * wm + ocn[None, :] * wn, acc,
             mask=(ocm[:, None] < M) & (ocn[None, :] < N))


@triton.jit
def _splitk_combine_flat(W, C, NEL, ws, S, s, BLOCK: tl.constexpr):
    """Pass 2 for a contiguous workspace and a contiguous output.

    Same arithmetic as `_splitk_combine`, element for element: partial 0 is the starting value
    (so a lone `-0.0` survives), partials 1..S-1 are added onto it in FORWARD order, the total
    is scaled once and cast once. The only difference is the shape of the work: `_splitk_combine`
    walks a BM x BN tile grid sized for the GEMM, which on these shapes is a few dozen CTAs of
    four warps for what is a pure copy-and-add of S*M*N floats. This walks the output as one
    flat vector instead, so the launch is sized by how much memory has to move.

    Valid only when W is [S, M, N] contiguous and C is [M, N] contiguous; the launcher owns both.
    """
    pid = tl.program_id(0)
    off = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = off < NEL
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    for i in range(0, S):
        p = tl.load(W + i.to(tl.int64) * ws + off, mask=m, other=0.0)
        acc = tl.where(i == 0, p, acc + p)
    acc = acc * s
    tl.store(C + off, acc.to(C.dtype.element_ty), mask=m)


_COMBINE_BLOCK = 512  # output elements per combine program; see the pass-2 sweep in REPORT.md


_IS_SM103: dict[int, bool] = {}


_P1_WAVE_SPLIT = 1.5  # programs per SM above which the grid counts as "several waves"


_P1_LAUNCH = {
    (16, 32, 2, 0): (4, 16), (16, 128, 2, 1): (2, 3), (32, 16, 2, 0): (2, 8), (32, 32, 2, 1): (2, 4),
    (32, 64, 2, 0): (8, 12), (32, 64, 2, 1): (2, 6), (32, 128, 2, 1): (2, 3), (32, 256, 2, 0): (4, 4),
    (32, 256, 2, 1): (4, 4), (64, 16, 1, 0): (4, 8), (64, 16, 1, 1): (2, 4), (64, 16, 2, 0): (4, 12),
    (64, 16, 2, 1): (4, 6), (64, 32, 1, 0): (4, 12), (64, 32, 1, 1): (4, 6), (64, 32, 2, 0): (4, 16),
    (64, 32, 2, 1): (4, 6), (64, 64, 1, 0): (4, 8), (64, 64, 1, 1): (4, 4), (64, 64, 2, 0): (4, 8),
    (64, 64, 2, 1): (4, 6), (64, 128, 1, 0): (4, 6), (64, 128, 1, 1): (4, 4), (64, 128, 2, 0): (4, 6),
    (64, 128, 2, 1): (4, 4), (64, 256, 1, 0): (4, 8), (64, 256, 2, 0): (4, 4), (64, 256, 2, 1): (4, 4),
    (128, 16, 1, 0): (4, 8), (128, 16, 2, 0): (4, 6), (128, 32, 1, 0): (4, 8), (128, 32, 1, 1): (4, 6),
    (128, 32, 2, 0): (4, 8), (128, 32, 2, 1): (4, 4), (128, 64, 1, 0): (4, 6), (128, 64, 1, 1): (4, 4),
    (128, 64, 2, 0): (4, 8), (128, 64, 2, 1): (4, 4), (128, 128, 1, 0): (4, 8), (128, 128, 1, 1): (4, 8),
    (128, 128, 2, 0): (4, 6), (128, 128, 2, 1): (4, 3), (128, 256, 1, 0): (8, 6), (128, 256, 2, 0): (8, 4),
    (128, 256, 2, 1): (8, 4), (256, 16, 1, 0): (4, 6), (256, 32, 1, 0): (4, 8), (256, 64, 2, 0): (4, 4),
    (256, 128, 1, 0): (8, 6), (256, 128, 2, 0): (8, 4), (256, 128, 2, 1): (8, 4), (256, 256, 2, 0): (8, 3)
}


_P1_TILES = tuple(sorted({(k[0], k[1]) for k in _P1_LAUNCH}))


_P1_TIE = 0.02  # two tiles within 2% of the cheapest count as a tie; the wider one wins


_P1_DEEP_LIMIT = 2.0  # programs per SM above which the pipeline is capped


_P1_DEEP_STAGES = 3  # the depth the shipped launcher uses


_INT32_MAX = 2**31


_SMEM_LIMIT = 228 * 1024  # sm_103 shared memory a CTA may ask for


_P1_CACHE: dict[tuple, tuple] = {}


def _split_tile(M, N, K, nsplit, elsize, min_bm, sm_count, BK, vectorises=True):
    """Tile, warps and stages for pass 1 on sm_103.

    `_tile` sizes the tile to the output and stops at 128, which on a skinny deep-K split-K
    leaves far too few CTAs: 92x672x12432 gets 6 tiles x 2 slices = 12 programs for 152 SMs and
    runs at a twentieth of the bandwidth it could. Sizing it the other way round -- smallest tile,
    most CTAs -- is just as wrong, because the operand traffic is
    `ceil(N/BN)*M*K + ceil(M/BM)*N*K` elements and a narrow tile reads the other operand many
    times over. So score every tile by the bytes it moves, divided by how much of the machine it
    can keep busy, and take the cheapest.

    None of this touches the arithmetic: the tile decides which program owns which output
    element and how many k-elements are loaded at once for it, never the order the products are
    added in. Each output element is still one accumulator fed by dots of BK real k in ascending
    order."""
    key = (M, N, K, nsplit, elsize, min_bm, sm_count, BK, vectorises)
    hit = _P1_CACHE.get(key)
    if hit is not None:
        return hit
    if not vectorises:
        # Without `cp.async` every tile load is a scalar 2-byte load, and the table below was read
        # off shapes where that never happens. Its rungs are the worst possible place to be here:
        # a 32x256 tile on two warps with a 4-deep pipeline spills to 255 registers and runs 24x
        # slower than the shipped launch on 17x11816x17408. Nothing in the table was measured in
        # this regime, and a sweep of it found no rung that beats the shipped launch reliably --
        # 30% better on one shape, 17% worse on another. So hand the shape back: same tile, same
        # warps, same depth, same kernel as ships today.
        BM, BN = _tile(M, N, torch.float8_e4m3fn if elsize == 1 else torch.float16, min_bm)
        hit = _P1_CACHE[key] = (BM, BN, 4, 3)
        return hit
    cands = []
    for BM, BN in _P1_TILES:
        if BM < min_bm or BM > max(64, 2 * triton.next_power_of_2(M)):
            continue
        if BN > max(16, 2 * triton.next_power_of_2(N)):
            continue
        tm, tn = triton.cdiv(M, BM), triton.cdiv(N, BN)
        ctas = tm * tn * nsplit
        wave = 0 if ctas <= _P1_WAVE_SPLIT * sm_count else 1
        if (BM, BN, elsize, wave) not in _P1_LAUNCH:
            continue
        # elements actually loaded, tile padding included: a tile wider than N still reads BN
        # columns, which is what makes an oversized tile expensive on a narrow shape.
        byts = tm * tn * (BM + BN) * K * elsize + nsplit * M * N * 4
        # A grid of 189 programs on 152 SMs costs two waves and is only 62% busy, so what divides
        # the bytes is the wave-quantised occupancy, not the raw program count.
        cands.append((byts * triton.cdiv(ctas, sm_count) * sm_count / ctas, BM, BN, wave))
    lo = min(c[0] for c in cands)
    # Two tiles of the same area move nearly the same bytes and the model cannot separate them;
    # the one with the wider m side wins, because it reads A once for more output and gives the
    # MMA its larger m shape.
    _, BM, BN, wave = max((c for c in cands if c[0] <= lo * (1 + _P1_TIE)), key=lambda c: (c[1], c[2]))
    num_warps, num_stages = _P1_LAUNCH[(BM, BN, elsize, wave)]
    # Past two waves the pipeline depth stops being free: it is shared memory, and shared memory
    # is how many programs an SM can hold. The table's several-waves rows were fitted mostly from
    # shapes that were only just over one wave, so they are too deep out here -- 64x15360x17408
    # runs 360 programs and is 11% slower at 4 stages than at the 3 the shipped launch uses.
    if triton.cdiv(M, BM) * triton.cdiv(N, BN) * nsplit > _P1_DEEP_LIMIT * sm_count:
        num_stages = min(num_stages, _P1_DEEP_STAGES)
    while num_stages > 2 and (BM + BN) * BK * elsize * num_stages > _SMEM_LIMIT:
        num_stages -= 1
    hit = _P1_CACHE[key] = (BM, BN, num_warps, num_stages)
    return hit


def _split_vectorises(a, b, N) -> bool:
    """Can pass 1 issue wide async copies for this shape?

    Triton decides the width of a tile load from one fact per kernel argument: whether its value
    divides by 16. It reads that as a count of ELEMENTS, so an fp16 B whose rows are 23632 bytes
    apart -- which is 16-byte aligned -- still fails it, because 11816 % 16 == 8. When it fails,
    `cp.async` disappears from the loop entirely and every load becomes a scalar `ld.global.b16`.

    Measured on 32x11816x9752 by slicing the operands so that each fact could be varied on its
    own: it is N and the row strides that decide it, and both have to divide by 16. N alone with
    an aligned stride is as bad as neither (8 async copies instead of 40), and an aligned N over
    an unaligned stride is equally bad -- so there is no split of the N range that recovers it.
    Restating the divisibility inside the kernel with `tl.multiple_of`, or passing the stride
    pre-divided and rebuilding it from a literal, changes nothing: the fact is read off the
    argument, not the body. M does not enter into it, which is what makes the frozen shapes
    (M = 19, 31, 39, ...) fast in the first place.

    fp8 never reaches here -- cuBLAS refuses fp8 unless every dimension is a multiple of 16."""
    if N % 16:
        return False
    for stride in (a.stride(0), a.stride(1), b.stride(0), b.stride(1)):
        if stride > 1 and stride % 16:
            return False
    return True


def _split_is_sm103() -> bool:
    """True on a GB300. Cached per device index, because the launcher is on the hot path for
    shapes whose GEMM is only a few microseconds long."""
    dev = torch.cuda.current_device()
    hit = _IS_SM103.get(dev)
    if hit is None:
        hit = _IS_SM103[dev] = platform().name.startswith("sm_103")
    return hit


def _triton_splitk_fast(a, b, chunk, out_dtype, scale, BK, fp8_min_bm):
    """`_triton_splitk` for sm_103.

    `_splitk_partial` reads B through the (bk, bn) stride pair, so a B that is already
    K-contiguous -- what an fp8 `w.t()` weight is -- can be handed to it as it stands. Making it
    N-contiguous first copies the whole weight on every call and is by far the largest cost on
    the fp8 shapes here (0.57 ms of a 0.62 ms call at N=10352, K=14848).

    The load reads the same K*N values into the same tile positions either way, so every
    `tl.dot` sees exactly the operands it saw before: the copy moves bytes, not arithmetic."""
    M, K = a.shape
    N = b.shape[1]
    nsplit = (K + chunk - 1) // chunk
    if nsplit < 2 or nsplit > 8192:
        return None
    elsize = a.element_size()
    min_bm = fp8_min_bm if a.dtype == torch.float8_e4m3fn else 16
    vectorises = _split_vectorises(a, b, N)
    BM, BN, num_warps, num_stages = _split_tile(M, N, K, nsplit, elsize, min_bm, platform().sm_count, BK, vectorises)
    ntile = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    w = torch.empty(nsplit, M, N, device=DEVICE, dtype=torch.float32)
    # The int32 kernel is a small win when the loads vectorise and a loss when they do not (its
    # narrower address arithmetic frees registers the scalar-load schedule then spends elsewhere),
    # so the non-vectorising path takes the shipped kernel too and matches pristine exactly.
    fits32 = max(M * K, K * N, nsplit * M * N) < _INT32_MAX
    pass1 = _splitk_partial_fast if (fits32 and vectorises) else _splitk_partial
    pass1[(ntile, nsplit)](a, b, w, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), w.stride(0),
                           w.stride(1), w.stride(2), chunk, BM=BM, BN=BN, BK=BK, num_warps=num_warps,
                           num_stages=num_stages)
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    _splitk_combine_flat[(triton.cdiv(M * N, _COMBINE_BLOCK), )](w, c, M * N, w.stride(0), nsplit, scale,
                                                                 BLOCK=_COMBINE_BLOCK, num_warps=4)
    return c


@triton.jit
def _splitk_partial_kpd_wide(A, B, W, M, N, K, am, ak, bk, bn, ws, wm, wn, CHUNK, KPD: tl.constexpr,
                             BLOCK_K: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                             GROUP_M: tl.constexpr, NSTAGE: tl.constexpr, IDX: tl.constexpr, NPAD: tl.constexpr,
                             SK0, KALIGN: tl.constexpr):
    """`_splitk_partial_k_per_dot` with the same accumulator boundaries, but G groups per dot.

    The order this kernel adds in is the order `_splitk_partial_k_per_dot` adds in, group for
    group. A `tl.dot` whose k extent is 16*G lowers to G chained k16 MMAs on ONE fp32
    accumulator, in increasing k, and each MMA rounds the accumulator exactly once. So lanes
    [16g, 16g+16) of one wide dot are the same accumulator update as the g-th narrow dot was.

    Two group widths are in play and both fit that grid:
      KPD 16  the group IS the MMA, lanes [16g, 16g+16) carry 16 real k
      KPD 8   the group is half an MMA, so lanes [16g, 16g+8) carry the 8 real k and
              [16g+8, 16g+16) are masked to zero -- which is the same trick the narrow kernel
              already uses to stand in for `s1688`, just repeated G times in one tile

    Only the residue tile needs group-at-a-time treatment (its last group can be short), and it
    is at most BLOCK_K/KPD == 4 groups. After it the k grid is `rbk + KPD*n` with every group
    full, so the wide loop starts there.

    NPAD says B has whole BN-wide columns, so the n index needs no wrap. `x % N` has contiguity
    `gcd(BN, BN, divisibility(N))`, and `divisibility(N)` is 1 for an N that is not a multiple
    of 16 -- which reads every B element with its own 2-byte `ld.global`. The plain
    `pn*BN + arange(BN)` keeps contiguity BN and loads 16 bytes at a time. Measured 1.33x-1.57x
    on this pass. The columns past N still get computed; the store drops them.

    KALIGN is what the wide loop's k start is really a multiple of. A's fast axis is k, and the
    k start `sk*CHUNK + rbk` is opaque to the compiler, so without this the whole A tile is read
    2 bytes at a time. The caller works the value out for the slices in THIS launch and splits
    the launch when they disagree -- a value that is not true faults with a misaligned address,
    so it is computed the same way the kernel computes the k start, not guessed."""
    IT: tl.constexpr = tl.int64 if IDX == 64 else tl.int32
    pid = tl.program_id(0)
    sk = tl.program_id(1) + SK0
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    ingrp = GROUP_M * npn
    m0 = (pid // ingrp) * GROUP_M
    gsz = tl.minimum(npm - m0, GROUP_M)
    pm = m0 + (pid % ingrp) % gsz
    pn = (pid % ingrp) // gsz
    rn = pn * BN + tl.arange(0, BN)
    om = ((pm * BM + tl.arange(0, BM)) % M).to(IT)
    on = (rn if NPAD else rn % N).to(IT)
    k0 = sk * CHUNK
    klen = tl.minimum(CHUNK, K - k0)
    rbk = klen % BLOCK_K
    rbk = tl.minimum(tl.where(rbk == 0, BLOCK_K, rbk), klen)
    nfirst = tl.cdiv(rbk, KPD)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    ok = tl.arange(0, 16).to(IT)
    for g in tl.range(0, nfirst, num_stages=1):
        off = g * KPD
        glen = tl.minimum(KPD, rbk - off)
        real = ok < glen
        kk = (k0 + off).to(IT) + ok
        a = tl.load(A + om[:, None] * am + kk[None, :] * ak, mask=real[None, :], other=0.0)
        b = tl.load(B + kk[:, None] * bk + on[None, :] * bn, mask=real[:, None], other=0.0)
        acc = tl.dot(a, b, acc)

    BKW: tl.constexpr = 16 * G
    STEP: tl.constexpr = G * KPD
    ix = tl.arange(0, BKW)
    koff = ((ix // 16) * KPD + ix % 16).to(IT)
    wr = ix % 16 < KPD
    base = k0 + rbk
    if KALIGN > 1:
        base = tl.multiple_of(base, KALIGN)
    nwide = (klen - rbk) // STEP
    ap = A + om[:, None] * am + (base.to(IT) + koff)[None, :] * ak
    bp = B + (base.to(IT) + koff)[:, None] * bk + on[None, :] * bn
    for _ in tl.range(0, nwide, num_stages=NSTAGE):
        if KPD == 16:  # every lane is real, so no mask and no out-of-range address
            a = tl.load(ap)
            b = tl.load(bp)
        else:
            a = tl.load(ap, mask=wr[None, :], other=0.0)
            b = tl.load(bp, mask=wr[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        ap += STEP * ak
        bp += STEP * bk

    done = rbk + nwide * STEP  # 0 unless G*KPD does not divide the post-residue length
    for g in tl.range(0, (klen - done) // KPD, num_stages=1):
        kk = (k0 + done + g * KPD).to(IT) + ok
        real = ok < KPD
        a = tl.load(A + om[:, None] * am + kk[None, :] * ak, mask=real[None, :], other=0.0)
        b = tl.load(B + kk[:, None] * bk + on[None, :] * bn, mask=real[:, None], other=0.0)
        acc = tl.dot(a, b, acc)

    ocm = (pm * BM + tl.arange(0, BM)).to(IT)
    tl.store(W + sk.to(IT) * ws + ocm[:, None] * wm + rn.to(IT)[None, :] * wn, acc,
             mask=(ocm[:, None] < M) & (rn[None, :] < N))


_SM_COUNT = None


def _sm_count():
    global _SM_COUNT
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    return _SM_COUNT


def _splitk_use_align(BM, ntile):
    """Whether pass 1 pays for the aligned-A path: a copy of A, the k-alignment claim, and the
    last slice in its own launch. One decision, not three -- splitting it into "copy when A is
    re-read enough times" plus "split when each launch still fills the machine" was measured on
    103 shapes and was 1.007x-1.023x SLOWER overall, with 1.9x-2.1x tails on narrow-N shapes."""
    return BM >= 128 and ntile >= _sm_count()


def _splitk_slice_kstart(K, chunk, block_k, sk):
    """The k index slice `sk`'s wide loop starts at -- `_splitk_partial_kpd_wide`'s `k0 + rbk`,
    recomputed here so the alignment claim it is turned into cannot drift from the kernel."""
    k0 = sk * chunk
    klen = min(chunk, K - k0)
    r = klen % block_k
    return k0 + min(block_k if r == 0 else r, klen)


def _pow2_gcd(vals, cap=8):
    g = 0
    for v in vals:
        g = math.gcd(g, v)
    a = 1
    while a < cap and g and g % (a * 2) == 0:
        a *= 2
    return a


def _splitk_groups_cfg(tBM, M, N, nsplit):
    """(groups per wide dot, BM, BN, GROUP_M, num_warps, num_stages) for pass 1 on sm_103.

    The only case split of my own is an occupancy one, and it is a statement about the machine
    rather than a fitted constant: when the whole grid -- every tile of every slice -- is
    smaller than the GPU has SMs, half the SMs sit idle, so a 64-row tile that makes twice as
    many of them wins. Verified on 79 shapes the rule was not fitted on: 0.978x the time of not
    doing it, never more than 1.06x slower on any single shape, and up to 1.42x faster.

    An earlier version instead sent N <= 2048 to the 64-row tile, because that won on four of
    the 24 tuning shapes. On the unseen shapes that branch cost up to 1.71x, worst of all right
    at its own boundary (3000x2047x5408). A threshold on N is not a fact about anything; the
    grid being smaller than the machine is.

    The 128-row tile takes a 4th pipeline stage and the two 64-row ones do not: 4 stages is
    1.02x-1.11x on all 19 shapes that use the wide tile and 1.4x SLOWER on both shapes that use
    a 64-row one. That split is the same one the tile itself makes, not a new threshold."""
    BN = min(256, max(16, triton.next_power_of_2(N)))
    if tBM < 128:
        return 4, tBM, min(128, BN), 8, 4, 3
    if triton.cdiv(M, 128) * triton.cdiv(N, BN) * nsplit < _sm_count():
        return 4, 64, BN, 8, 8, 3
    return 2, 128, BN, 8, 4, 4


def _triton_splitk_groups_sm103(a, b, chunk, k_per_dot, block_k, cmode, out_dtype, scale, fp8_min_bm):
    """sm_103 only. Same two passes and the same accumulator boundaries as
    `_triton_splitk_groups`, with the pass-1 k loop widened to G groups per `tl.dot`."""
    if block_k % k_per_dot or k_per_dot not in (8, 16):
        return None
    b = _kcontig(b)
    M, K = a.shape
    N = b.shape[1]
    nsplit = (K + chunk - 1) // chunk
    if nsplit < 1 or nsplit > 8192:
        return None
    tBM, tBN = _tile(M, N, a.dtype, fp8_min_bm)
    G, BM, BN, GROUP_M, nw, ns = _splitk_groups_cfg(tBM, M, N, nsplit)
    # Copying B into whole BN-wide columns lets pass 1 read it 16 bytes at a time instead of 2
    # (see `_splitk_partial_kpd_wide`). Measured 1.04x-2.10x on pass 1 for every shape whose B
    # row stride is not a multiple of 16, net of the copy. Skipped below one full row tile,
    # where the copy is a whole pass over B against a GEMM that reads it once.
    npad = 0
    if (N % 16 or b.stride(0) % 16) and M >= BM:
        bp = torch.empty(K, triton.cdiv(N, BN) * BN, device=DEVICE, dtype=b.dtype)
        bp[:, :N] = b  # the pad columns stay uninitialised: they only feed outputs the store drops
        b, npad = bp, 1
    # And the same for A's rows, for the k axis rather than the n axis: unless the row stride is
    # a multiple of 16 elements, no k start can be proved aligned and the A tile is read 2 bytes
    # at a time. Worth up to 1.5x on pass 1. Measured on all 24 shapes, it only pays on the
    # 128-row tile, and only once there is at least a CTA per SM -- below that the copy and the
    # second launch cost more than the wider loads save.
    ntile = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    use_align = _splitk_use_align(BM, ntile)
    if use_align and a.stride(0) % 16:
        ap = torch.empty(M, triton.cdiv(K, 16) * 16, device=DEVICE, dtype=a.dtype)
        ap[:, :K] = a
        a = ap[:, :K]
    aligned = use_align and a.stride(0) % 16 == 0
    # int32 addressing is only safe while no operand's linear index can reach 2^31
    lim = 2**31 - 1
    idx = 32 if max((M + BM) * abs(a.stride(0)) + K, (K + 1) * abs(b.stride(0)) + N + BN,
                    nsplit * (M + BM) * (N + BN)) < lim else 64
    # The workspace rows are deliberately NOT padded to 16 fp32. It does buy the combine 1.27x,
    # but an aligned row stride makes the pass-1 store vectorise, and on the 64-row tile that
    # costs pass 1 up to 1.8x -- far more than the combine gains.
    w = torch.empty(nsplit, M, N, device=DEVICE, dtype=torch.float32)
    # Every slice but the last starts its wide loop at `sk*chunk + chunk%block_k`, a multiple of
    # 8; the last one starts at an offset congruent to K, which is usually not. Running the last
    # slice as its own launch keeps the weaker claim off the other slices, for one extra ~2 us
    # kernel launch.
    starts = [_splitk_slice_kstart(K, chunk, block_k, sk) for sk in range(nsplit)]
    groups = [(0, nsplit)]
    if aligned and nsplit > 1 and _pow2_gcd(starts[:-1]) != _pow2_gcd(starts[-1:]):
        groups = [(0, nsplit - 1), (nsplit - 1, nsplit)]
    for s0, s1 in groups:
        kalign = _pow2_gcd(starts[s0:s1]) if aligned else 1
        _splitk_partial_kpd_wide[(ntile, s1 - s0)](a, b, w, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                                   w.stride(0), w.stride(1), w.stride(2), chunk, KPD=k_per_dot,
                                                   BLOCK_K=block_k, G=G, BM=BM, BN=BN, GROUP_M=GROUP_M, NSTAGE=ns,
                                                   IDX=idx, NPAD=npad, SK0=s0, KALIGN=kalign, num_warps=nw)
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    # An 8x64 tile with 2 warps, from 1440 timings over 60 tiles: within 2% of the best tile on
    # every shape and 1.33x ahead of 32x64. The combine is pure streaming, so it wants many
    # small CTAs, not the GEMM's 128x128.
    cBM, cBN = 8, 64
    _splitk_combine_modes[(triton.cdiv(M, cBM) * triton.cdiv(N, cBN), )](w, c, M, N, w.stride(0), w.stride(1),
                                                                        w.stride(2), c.stride(0), c.stride(1), nsplit,
                                                                        scale, CMODE=cmode, BM=cBM, BN=cBN,
                                                                        num_warps=2)
    return c


@triton.jit
def _pad_rows(SRC, DST, K, sm, dm, DOFF, BS: tl.constexpr):
    """Copy a [rows, K] operand into a buffer whose row stride is a multiple of 16 elements,
    starting at column DOFF. One program per (row, column block).

    torch's own strided `copy_` runs this at about 1.4 TB/s; this reaches 2.5-3.2 TB/s, which is
    worth 5-9% of the whole call on the shapes that repack both operands. Columns outside
    [DOFF, DOFF+K) are left as `torch.empty` gave them: the GEMM either masks them or feeds them
    to output columns its store drops."""
    r = tl.program_id(0).to(tl.int64)
    o = tl.program_id(1) * BS + tl.arange(0, BS)
    v = tl.load(SRC + r * sm + o, mask=o < K, other=0)
    tl.store(DST + r * dm + DOFF + o, v, mask=o < K)


@triton.jit
def _pad_rows2(SA, DA, KA, sam, dam, DOFFA, NA, JA, SB, DB, KB, sbm, dbm, DOFFB, JB, BS: tl.constexpr):
    """Both repacks in one launch. A kernel launch is 5.3 us of device time on this GPU even when
    the kernel does nothing, and on a 15 us GEMM two of them are most of the call. The grid is
    flattened over (row, column block) for each operand in turn so neither pays for the other's
    row length."""
    pid = tl.program_id(0)
    if pid < NA * JA:
        r = (pid // JA).to(tl.int64)
        o = (pid % JA) * BS + tl.arange(0, BS)
        tl.store(DA + r * dam + DOFFA + o, tl.load(SA + r * sam + o, mask=o < KA, other=0), mask=o < KA)
    else:
        q = pid - NA * JA
        r = (q // JB).to(tl.int64)
        o = (q % JB) * BS + tl.arange(0, BS)
        tl.store(DB + r * dbm + DOFFB + o, tl.load(SB + r * sbm + o, mask=o < KB, other=0), mask=o < KB)


@triton.jit
def _kpd_wide_loop(ap, bp, acc, nok, nwide, ak, bk, KPD: tl.constexpr, BKW: tl.constexpr, BN: tl.constexpr,
                   STEP: tl.constexpr, NSTAGE: tl.constexpr, MASKN: tl.constexpr):
    """The main k loop. A mask on the n axis is not free -- carrying one where it is not needed
    measured 2.5x slower on the KPD 8 shapes -- so MASKN is only set when the last column of
    tiles really does run past N and the caller has not repacked B into whole BN-wide columns.

    Compiling the masked and unmasked forms into one kernel and picking between them per program
    was tried and does not fit: Triton gives each `scf.if` branch its own multi-buffered shared
    memory, so the 512-wide tile needs 327736 bytes against a 232448 limit."""
    ix = tl.arange(0, BKW)
    wr = ix % 16 < KPD
    for _ in tl.range(0, nwide, num_stages=NSTAGE):
        if KPD == 16:  # every lane is real, so no k mask and no out-of-range k
            a = tl.load(ap)
            b = tl.load(bp, mask=nok[None, :], other=0.0) if MASKN else tl.load(bp)
        else:
            a = tl.load(ap, mask=wr[None, :], other=0.0)
            bm = wr[:, None] & nok[None, :] if MASKN else tl.broadcast_to(wr[:, None], (BKW, BN))
            b = tl.load(bp, mask=bm, other=0.0)
        acc = tl.dot(a, b, acc)
        ap += STEP * ak
        bp += STEP * bk
    return acc


@triton.jit
def _plain_gemm_kpd_wide(A, B, C, M, N, K, am, ak, bk, bn, cm, cn, s, RES, AK0, KPD: tl.constexpr, G: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr, NSTAGE: tl.constexpr,
                         RSTAGE: tl.constexpr, IDX: tl.constexpr, NGUARD: tl.constexpr, KALIGN: tl.constexpr,
                         NR: tl.constexpr):
    """`_plain_gemm_k_per_dot` with the same accumulator boundaries, but G groups per `tl.dot`.

    The order this kernel adds in is the order `_plain_gemm_k_per_dot` adds in, group for group.
    A `tl.dot` whose k extent is 16*G lowers to G chained k16 MMAs on ONE fp32 accumulator, in
    increasing k, and each MMA rounds the accumulator exactly once. So lanes [16g, 16g+16) of one
    wide dot are the same accumulator update as the g-th narrow dot was.

    Two group widths are in play and both fit that grid:
      KPD 16  the group IS the MMA, lanes [16g, 16g+16) carry 16 real k
      KPD 8   the group is half an MMA, so lanes [16g, 16g+8) carry the 8 real k and
              [16g+8, 16g+16) are masked to zero -- the same stand-in for `s1688` the narrow
              kernel already uses, repeated G times in one tile

    Only the leading residue needs group-at-a-time treatment, because its last group can be
    short. It is `ceil(RES/KPD)` groups, at most 16. After it the k grid is `RES + KPD*n` with
    every group full, so the wide loop starts there and a short tail finishes whatever G*KPD
    does not divide.

    The n index is never wrapped. `x % N` has contiguity `gcd(BN, BN, divisibility(N))`, and
    `divisibility(N)` is 1 for an N that is not a multiple of 16 -- which reads every B element
    with its own 2-byte `ld.global`. The plain `pn*BN + arange(BN)` keeps contiguity BN. NGUARD
    then says the last n tile runs past N and the B load needs a column mask; the caller clears
    it either by copying B into whole BN-wide columns or by observing that BN divides N. Columns
    past N still get computed; the store drops them, so which of a wrap, a mask or an
    uninitialised pad column feeds them cannot change a stored value.

    AK0 shifts every A k-index. A's fast axis is k, so the wide loop's row start has to be a
    multiple of 16 elements for the tile to be read 16 bytes at a time, and the natural start
    `RES` is whatever `K % block_k` happens to be. The caller copies A into a buffer whose row
    stride is a multiple of 16 and whose column 0 sits at AK0 = -RES mod 16, which makes
    `AK0 + RES` a multiple of 16; KALIGN is that claim. A claim that is not true faults with a
    misaligned address, so it is computed the same way the kernel computes the start."""
    IT: tl.constexpr = tl.int64 if IDX == 64 else tl.int32
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    ingrp = GROUP_M * npn
    m0 = (pid // ingrp) * GROUP_M
    gsz = tl.minimum(npm - m0, GROUP_M)
    pm = m0 + (pid % ingrp) % gsz
    pn = (pid % ingrp) // gsz
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    om = (rm % M).to(IT)
    on = rn.to(IT)
    nok = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    ok = tl.arange(0, 16).to(IT)
    wg = ok < KPD  # constant-true at KPD 16, so the whole-group loads carry no mask there
    if NR > 0:
        # The whole leading residue as ONE dot of NR groups, with the groups it does not need
        # padded on at the FRONT and masked to zero on both operands.
        #
        # A leading all-zero group is exact here and only here. Its products are all +0.0, so the
        # MMA adds +0.0 to the accumulator, and the accumulator at that point is still the +0.0
        # that `tl.zeros` put there -- and +0.0 + (+0.0) is +0.0. Putting the pad groups at the
        # BACK instead would add +0.0 to a real partial sum, which is a no-op for every value
        # except -0.0, where it would flip the sign. Front is provable, back is not.
        jx = tl.arange(0, 16 * NR)
        gpad = NR - tl.cdiv(RES, KPD)  # groups of head padding, 0 when NR is exactly what is needed
        kj = (jx // 16 - gpad) * KPD + jx % 16
        rj = (jx % 16 < KPD) & (kj >= 0) & (kj < RES)
        kjc = tl.maximum(kj, 0).to(IT)
        aj = tl.load(A + om[:, None] * am + (AK0 + kjc)[None, :] * ak, mask=rj[None, :], other=0.0)
        bmj = rj[:, None] & nok[None, :] if NGUARD else tl.broadcast_to(rj[:, None], (16 * NR, BN))
        bj = tl.load(B + kjc[:, None] * bk + on[None, :] * bn, mask=bmj, other=0.0)
        acc = tl.dot(aj, bj, acc)
    npre = RES // KPD if NR == 0 else 0  # whole groups before the short one
    for g in tl.range(0, npre, num_stages=RSTAGE):
        kk = ok + g * KPD
        bq = B + kk[:, None] * bk + on[None, :] * bn
        if KPD == 16:
            a = tl.load(A + om[:, None] * am + (AK0 + kk)[None, :] * ak)
            b = tl.load(bq, mask=nok[None, :], other=0.0) if NGUARD else tl.load(bq)
        else:
            a = tl.load(A + om[:, None] * am + (AK0 + kk)[None, :] * ak, mask=wg[None, :], other=0.0)
            bm = wg[:, None] & nok[None, :] if NGUARD else tl.broadcast_to(wg[:, None], (16, BN))
            b = tl.load(bq, mask=bm, other=0.0)
        acc = tl.dot(a, b, acc)
    if NR == 0 and RES - npre * KPD > 0:  # the short group, only when KPD does not divide RES
        kk = ok + npre * KPD
        real = ok < RES - npre * KPD
        a = tl.load(A + om[:, None] * am + (AK0 + kk)[None, :] * ak, mask=real[None, :], other=0.0)
        bm = real[:, None] & nok[None, :] if NGUARD else tl.broadcast_to(real[:, None], (16, BN))
        b = tl.load(B + kk[:, None] * bk + on[None, :] * bn, mask=bm, other=0.0)
        acc = tl.dot(a, b, acc)

    BKW: tl.constexpr = 16 * G
    STEP: tl.constexpr = G * KPD
    ix = tl.arange(0, BKW)
    # At KPD 16 the group index and the lane index coincide. Spelling that out matters: the
    # compiler reads contiguity off the expression, and `(ix // 16) * 16 + ix % 16` is a div and
    # a mod it cannot see through, so the tile would come out contiguity 16 instead of 16*G.
    koff = ix.to(IT) if KPD == 16 else ((ix // 16) * KPD + ix % 16).to(IT)
    wr = ix % 16 < KPD
    abase = AK0 + RES
    if KALIGN > 1:
        abase = tl.multiple_of(abase, KALIGN)
    nwide = (K - RES) // STEP
    ap = A + om[:, None] * am + (koff + abase)[None, :] * ak
    bp = B + (koff + RES)[:, None] * bk + on[None, :] * bn
    acc = _kpd_wide_loop(ap, bp, acc, nok, nwide, ak, bk, KPD, BKW, BN, STEP, NSTAGE, MASKN=NGUARD)

    done = RES + nwide * STEP  # < K only when G*KPD does not divide the post-residue length
    for g in tl.range(0, (K - done) // KPD, num_stages=RSTAGE):
        kk = ok + (done + g * KPD)
        bq = B + kk[:, None] * bk + on[None, :] * bn
        if KPD == 16:
            a = tl.load(A + om[:, None] * am + (AK0 + kk)[None, :] * ak)
            b = tl.load(bq, mask=nok[None, :], other=0.0) if NGUARD else tl.load(bq)
        else:
            a = tl.load(A + om[:, None] * am + (AK0 + kk)[None, :] * ak, mask=wg[None, :], other=0.0)
            bm = wg[:, None] & nok[None, :] if NGUARD else tl.broadcast_to(wg[:, None], (16, BN))
            b = tl.load(bq, mask=bm, other=0.0)
        acc = tl.dot(a, b, acc)

    acc = acc * s
    ocm = rm.to(IT)
    tl.store(C + cm * ocm[:, None] + cn * on[None, :], acc.to(C.dtype.element_ty),
             mask=(ocm[:, None] < M) & nok[None, :])


_KPD_CFG: dict[tuple, tuple] = {}


def _kpd_cfg(M, N, K, kpd, res):
    """(G, BM, BN, GROUP_M, num_warps, num_stages, residue stages) for `_plain_gemm_kpd_wide`.

    Start from the widest tile the machine likes -- 128 rows, and BN 256 or 512 -- then, only for
    a GEMM small enough that filling the machine matters more than re-reading less, step the tile
    down until the grid covers the SMs.

    BN 256 vs 512 is a data-reuse choice made through `wavework(BN) = ceil(programs / SM) * BN`,
    the tile work a full wave of the machine costs. The wider tile halves the number of programs,
    which is a win while the grid still fills the machine and a loss the moment the last wave
    empties out: at 4455x4262 it drops the grid from 3.9 waves to 2.07, so a third of the machine
    idles through the last wave. A tie goes to the wider tile by up to 25%, which is the gap
    between the closest pair measured either way (1.167 for 512, 1.333 for 256).

    The step-down only runs below a million MACs squared -- `M*N*K < 1e9`, about 4 us of work at
    the rate these kernels reach, i.e. the same order as one kernel launch (5.3 us here, measured
    with an empty kernel). Above that the big tile always won: on 501x5714x1942 and three
    neighbours the grid is under one wave and stepping down still costs 6%, because each program
    has enough k to hide everything. Below it the reverse: 224x3308x86 and 40x11778x126 ran 0.86x
    and 0.67x of the shipped kernel on the wide tile and 1.4-1.6x on a stepped one. Measured over
    58 shapes in that band, the step-down is 1.40x the shipped kernel against 1.14x without it,
    and it changes nothing on the 18 large frozen shapes.

    Below 64 columns the step needs one more condition, and only when the tile is already at the
    16-row floor: with 16 rows there is nothing left to trade, so a halving that pushes the grid
    into a second wave loses (16x7290x87 measured 1.221x at BN 64 and 1.107x at BN 32).

    G is the groups per dot. On the 128-row tile it is 2, or 4 when the grid is under one wave and
    each program needs more work in flight. On the stepped tiles the k loop is a handful of
    dependent steps, so G takes the whole post-residue k in one dot, held to G * BN <= 256 --
    past that the tile stops fitting the pipeline (G=8 with BN=64 measured 0.81x against 0.99x at
    BN=32 on the same shape)."""
    hit = _KPD_CFG.get((M, N, K, kpd, res))  # host cost only; the smallest shapes here are 9 us
    if hit is not None:
        return hit
    sm = _sm_count()
    BM = 16 if M <= 16 else max(32, min(128, triton.next_power_of_2(M)))
    wide = [b for b in (256, 512) if b <= max(256, triton.next_power_of_2(N))]
    BN = wide[0]
    for bn in wide[1:]:  # ties, and near-ties up to 25%, go to the wider tile
        if _wavework(M, N, BM, bn, sm) <= 1.25 * _wavework(M, N, BM, BN, sm):
            BN = bn
    nt = lambda: triton.cdiv(M, BM) * triton.cdiv(N, BN)  # noqa: E731
    if M * N * K < 10**9:
        while nt() < sm and BN > 64:
            BN //= 2
        while nt() < sm and BM > 32:
            BM //= 2
        while nt() < sm and BN > 32 and (BM > 16 or _wavework(M, N, BM, BN // 2, sm) < _wavework(M, N, BM, BN, sm)):
            BN //= 2
    if BM >= 128 and BN >= 512:
        cfg = (2, 128, BN, 8, 8, 4, 1)
    elif BM >= 128:
        cfg = (4, 128, BN, 8, 8, 4, 1) if nt() < sm else (2, 128, BN, 8, 4, 4, 1)
    else:
        G = max(1, min(8, 256 // BN, triton.next_power_of_2(max(1, (K - res) // kpd))))
        cfg = (G, BM, BN, 1, 8, 4, 1)
    _KPD_CFG[(M, N, K, kpd, res)] = cfg
    return cfg


def _wavework(M, N, BM, BN, sm):
    """The tile work one full wave of the machine costs: `ceil(programs / SM) * BM * BN`."""
    return -(-triton.cdiv(M, BM) * triton.cdiv(N, BN) // sm) * BM * BN


def _copy_padded(src, dst, doff):
    """`dst[:, doff:doff+src.shape[1]] = src`, but 1.8x faster than torch's strided copy."""
    rows, cols = src.shape
    BS = 1024
    _pad_rows[(rows, triton.cdiv(cols, BS))](src, dst, cols, src.stride(0), dst.stride(0), doff, BS=BS,
                                             num_warps=8 if cols >= 4096 else 4)


def _triton_plain_k_per_dot_sm103(a, b, out_dtype, k_per_dot, res, scale, cfg=None, copy_a=None, copy_b=None):
    """sm_103 only. Same accumulator boundaries as `_triton_plain_k_per_dot`, with the k loop
    widened to G groups per `tl.dot` and the two operands repacked so the tile can be read
    16 bytes at a time. Returns None for anything the wide form does not cover, and the caller
    then runs the shipped launch."""
    if k_per_dot not in (8, 16):
        return None
    M, K = a.shape
    N = b.shape[1]
    if (K - res) % k_per_dot:
        return None  # the post-residue grid is not whole groups; the shipped kernel is the safe form
    G, BM, BN, GROUP_M, nw, ns, rs = cfg or _kpd_cfg(M, N, K, k_per_dot, res)
    # A repack is one whole pass over the operand plus a kernel launch, so it pays only when the
    # GEMM reads that operand more than once and is long enough for the launch to disappear in
    # it. Both are counted in tiles, not in bytes: `nm`/`nn` is how many times the GEMM re-reads
    # B/A, and BM < 64 is the rung whose whole runtime is a couple of launches.
    nm, nn = triton.cdiv(M, BM), triton.cdiv(N, BN)
    big = BM >= 64 and nm * nn >= _sm_count() // 2
    if copy_b is None:
        copy_b = big and nm >= 2 and b.stride(0) % 16
    nguard = int(bool(N % BN))  # cleared below when B is repacked into whole BN-wide columns
    if copy_a is None:
        copy_a = big and nn >= 2 and (a.stride(1) != 1 or a.stride(0) % 16 or res % 16)
    copy_a = bool(copy_a) and a.stride(1) == 1
    ak0, kalign, ap, bp = 0, 1, None, None
    if copy_a:
        ak0 = -res % 16
        ap = torch.empty(M, triton.cdiv(K + ak0, 16) * 16, device=DEVICE, dtype=a.dtype)
    if copy_b:
        bp = torch.empty(K, nn * BN, device=DEVICE, dtype=b.dtype)
    if copy_a and copy_b:  # one launch, not two: an empty kernel already costs 5.3 us here
        BS = 1024
        ja, jb = triton.cdiv(K, BS), triton.cdiv(N, BS)
        _pad_rows2[(M * ja + K * jb, )](a, ap, K, a.stride(0), ap.stride(0), ak0, M, ja, b, bp, N, b.stride(0),
                                        bp.stride(0), 0, jb, BS=BS, num_warps=8)
    elif copy_a:
        _copy_padded(a, ap, ak0)
    elif copy_b:
        _copy_padded(b, bp, 0)
    if copy_a:  # the pad columns stay uninitialised; the store drops the outputs they reach
        a, kalign = ap, 16
    if copy_b:
        b, nguard = bp, 0
    # The leading residue as one dot, when its tile fits shared memory next to the main one.
    nres = -(-res // k_per_dot)
    nr = triton.next_power_of_2(nres) if nres else 0
    if nr and 16 * nr * (BM + BN) * 2 > 100000:
        nr = 0
    lim = 2**31 - 1
    idx = 32 if max((M + BM) * abs(a.stride(0)) + K + ak0, (K + 16) * abs(b.stride(0)) + N + BN,
                    (M + BM) * (N + BN)) < lim else 64
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    _plain_gemm_kpd_wide[(nm * nn, )](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                                      c.stride(1), scale, res, ak0, KPD=k_per_dot, G=G, BM=BM, BN=BN, GROUP_M=GROUP_M,
                                      NSTAGE=ns, RSTAGE=rs, IDX=idx, NGUARD=nguard, KALIGN=kalign, NR=nr, num_warps=nw)
    return c


def _sm103_gemmsn() -> bool:
    global _SM103_gemmsn
    if _SM103_gemmsn is None:
        from .arch import platform
        try:
            _SM103_gemmsn = platform().name.startswith("sm_103")
        except Exception:
            _SM103_gemmsn = False
    return _SM103_gemmsn


def _sm103_gemv() -> bool:
    """Only sm_103 gets the tuned launches; every other architecture keeps the shipped path."""
    return platform().name.startswith("sm_103")


@triton.jit
def _simt_chain_gemm_wide(A, B, C, M, N, K, CH, TAIL, NSUB, SUB, am, ak, bk, bn, cm, cn, BM: tl.constexpr,
                          BN: tl.constexpr, S: tl.constexpr, U: tl.constexpr):
    """`_simt_chain_gemm`, sm_103 only, with the S chunks walked side by side.

    Same three accumulation levels, same operands in the same places, only the loop nest is
    turned inside out. The shipped kernel finishes chunk 0 before it starts chunk 1, so the k
    loop is `S` runs of one k-element with two global loads each and nothing else in flight.
    Here the S chunks advance together: step `j` handles k = 0*CH+j, 1*CH+j, ... (S-1)*CH+j at
    once, as one (BM, S) load of A and one (BN, S) load of B, and the accumulator carries an
    (BM, BN, S) tile with one plane per chunk.

    WHY THE ORDER CANNOT MOVE. Plane t of `sub` only ever receives a[m, t*CH+j] * b[t*CH+j, n],
    with j running upward, so each plane is the same ascending chain the shipped kernel builds
    for chunk t -- the planes are independent accumulators that happen to be updated in the same
    instruction, not a regrouping. The three levels are still nested the same way: `sub` per
    sub-block, `accs` gathering the sub-block totals, and the S planes added into `acc` left to
    right at the end.

    The two tails are what keeps that true when the chunks are not all the same length. Chunk
    S-1 is the short one (TAIL = K - (S-1)*CH elements, the rest have CH). The first loop runs
    while every chunk still has a k to contribute; the second finishes the S-1 longer ones under
    a `tl.where` that leaves the finished plane's bits untouched -- not a masked load adding an
    exact zero, which would turn a -0.0 partial into +0.0. `accs` is likewise only advanced for
    the planes whose chunk actually reaches this sub-block.

    The final level-3 sum extracts plane t as `sum(where(t, accs, 0))`, i.e. it adds three exact
    zeros to it. That is safe here and only here: `acc` starts at +0.0, `+0.0 + x` is x for every
    x except -0.0, and it maps -0.0 to +0.0 -- so `acc` is never -0.0 and adding +0.0 to it in
    place of a -0.0 plane cannot change it. Overflow to inf/NaN is impossible: |sum| <= K *
    65504^2 fits fp32 for any K these kernels see.

    THE BACKEND WILL REASSOCIATE THIS CHAIN IF LET. LLVM's SLP pass can turn the accumulate
    chain into a horizontal reduction, which moves the result by about one fp32 ulp -- exactly
    what this kernel exists to prevent. It shows up as `foldExtExtBinop` in the LLVM IR and as
    independent `mul.f32` in the PTX. It is not fp contraction: `enable_fp_fusion=False` does not
    stop it.

    Writing the unroll out in source (`tl.static_range(U)` inside a runtime block loop, as the
    shipped kernel does) is deliberate but is NOT the protection, and must not be read as one.
    Measured: at BM=4, BN=16, num_warps=1, U=8 the source form and `tl.range(loop_unroll_factor
    =8)` compile to the same LLVM IR and give the same wrong answer -- 26/26 shape checks differ,
    the same 98 bytes. What decides it is the width of the per-thread accumulator, not who
    unrolled. At the tuned tile that width is one fp32 and nothing reassociates: over 348
    compiled specializations (split 1, 2, 4, 8, 16 x six K x two N x two M x three sub)
    `foldExtExtBinop` is 0 and `mul.f32` is 0 in every one. The evidence that counts is
    behavioural -- the recipe-space sweep in REPORT.md. Re-run it if this tile is ever retuned.

    BM, BN, U and num_warps are pure performance knobs, as in the shipped kernel: every output
    element is its own independent reduction."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    on = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    mm = om < M
    mn = on < N
    tv = tl.arange(0, S)
    kb = (tv * CH).to(tl.int64)  # where each chunk starts
    lent = tl.where(tv < S - 1, CH, TAIL)  # how long each chunk is
    accs = tl.zeros((BM, BN, S), dtype=tl.float32)
    for i in range(0, NSUB):
        jlo = i * SUB
        jhi = tl.minimum(jlo + SUB, CH)
        jfull = tl.minimum(jlo + SUB, TAIL)  # past this the short chunk is done
        sub = tl.zeros((BM, BN, S), dtype=tl.float32)
        # `nfull` can land below `jlo` when this sub-block is past the short chunk's end; both
        # loops below are then empty, whichever way the division rounds.
        nfull = jlo + ((jfull - jlo) // U) * U
        for j0 in range(jlo, nfull, U):
            for u in tl.static_range(U):
                kk = kb + (j0 + u)
                av = tl.load(A + om[:, None] * am + kk[None, :] * ak, mask=mm[:, None], other=0.0).to(tl.float32)
                bv = tl.load(B + kk[None, :] * bk + on[:, None] * bn, mask=mn[:, None], other=0.0).to(tl.float32)
                sub += av[:, None, :] * bv[None, :, :]
        for j in range(tl.maximum(nfull, jlo), jfull):
            kk = kb + j
            av = tl.load(A + om[:, None] * am + kk[None, :] * ak, mask=mm[:, None], other=0.0).to(tl.float32)
            bv = tl.load(B + kk[None, :] * bk + on[:, None] * bn, mask=mn[:, None], other=0.0).to(tl.float32)
            sub += av[:, None, :] * bv[None, :, :]
        for j in range(tl.maximum(jfull, jlo), jhi):
            kk = kb + j
            act = j < lent
            av = tl.load(A + om[:, None] * am + kk[None, :] * ak, mask=act[None, :] & mm[:, None], other=0.0)
            bv = tl.load(B + kk[None, :] * bk + on[:, None] * bn, mask=act[None, :] & mn[:, None], other=0.0)
            sub = tl.where(act[None, None, :], sub + av.to(tl.float32)[:, None, :] * bv.to(tl.float32)[None, :, :], sub)
        accs = tl.where((jlo < lent)[None, None, :], accs + sub, accs)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for t in tl.static_range(S):
        acc += tl.sum(tl.where(tv[None, None, :] == t, accs, 0.0), axis=2)
    tl.store(C + om[:, None] * cm + on[None, :] * cn, acc.to(C.dtype.element_ty), mask=mm[:, None] & mn[None, :])


_GEMMSN_WIDE_TILE = (4, 8, 4, 8)


_GEMV13_TILE = 256


_GEMV13_EPT_M1 = 8  # matrix contiguous along the output axis


_GEMV13_EPT_N1 = 4  # matrix contiguous along k


_GEMV13_CTA_FLOOR = 304  # 2 x the 152 SMs


_GEMV13_MAX_UNROLL = 8


_GEMV13_WARPS_PER_SM = 16  # below this the launch cannot fill the GPU, so spend warps instead


_GEMV13_COMBINE = (32, 4)  # (BE, num_warps) for the fp32 slice combine


_CSLICE_LAUNCH = (8, 4)  # (BE, num_warps): a 256-element (BE, W=32) tile on 128 threads


_GEMV14_LAUNCH = (8, 4, 16, 4, 4)  # (dot BE, dot warps, reduce BE, reduce warps, dot unroll)


@triton.jit
def _lane_lr(x, BE: tl.constexpr, W: tl.constexpr):
    """Add the W lane totals left to right: ((l0 + l1) + l2) + ...

    This is the frozen kernels' extraction, character for character, and it stays that way. A
    `tl.split` column peel replaced it for a while -- W-1 splits instead of W full reductions,
    worth 2% on ALGO 13 and about 2x on the cslice path -- and it was wrong twice over. First,
    the masked sum adds W-1 exact `+0.0`s to the column and `-0.0 + 0.0` is `+0.0`, so it
    quietly normalises a negative-zero column while the split keeps the sign; on a tile with
    planted negative zeros all 100 (W, BE, num_warps) combinations differed. Second, even after
    that was put back, 3 of those 100 still differed by an fp32 ulp on ordinary values, with no
    mechanism found. A 2% knob is not worth a difference nobody can explain."""
    lid = tl.arange(0, W)
    tot = tl.zeros((BE, ), dtype=tl.float32)
    for j in tl.static_range(W):
        v = tl.sum(tl.where(lid[None, :] == j, x, 0.0), axis=1)
        tot = v if j == 0 else tot + v
    return tot


@triton.jit
def _gemv_lane_x(A, B, Cout, NEL, K, CHUNK, sa_e, sa_k, sb_k, sb_e, sc, ws, CC, V: tl.constexpr, W: tl.constexpr,
                 DOWN: tl.constexpr, BE: tl.constexpr, BC: tl.constexpr, NS: tl.constexpr,
                 UF: tl.constexpr = 1):
    """`_gemv_lane` with the SPLITK_NUM slices folded into grid axis 1.

    Nothing about the arithmetic moves.

      * slice `sl` still owns exactly [sl*CHUNK, min((sl+1)*CHUNK, K)) and still walks it in
        the same (chunk, tile, sub) order, so every lane's chain is the one `_gemv_lane` runs.
        `_triton_gemv13` used to launch one kernel per slice from a Python loop; at
        SPLITK_NUM 32 that is 32 launches of a few CTAs each, nearly all of it latency.
      * a slice that starts past K reduces nothing and stores an exact +0.0, which is the value
        the zero-filled workspace used to carry.
      * `BC` says which operand is the vector: a gemv has one output per row or per column, so
        exactly one operand is read once per output element and the other is read once, full
        stop.  Loading the shared one as a 1-D vector and broadcasting it reads the same numbers
        as the (BE, W) tile of identical rows it replaces.
    """
    pe = tl.program_id(0)
    sl = tl.program_id(1)
    oe = (pe * BE + tl.arange(0, BE)).to(tl.int64)
    em = oe < NEL
    k0 = sl * CHUNK
    ke = tl.minimum(k0 + CHUNK, K)
    ntile = tl.cdiv(tl.maximum(ke - k0, 0), V * W)
    per = tl.maximum(tl.where(CC > 0, CC, ntile), 1)
    nc = tl.cdiv(ntile, per)
    lid = tl.arange(0, W)
    if DOWN:
        slot = _bitrev(lid, W)
    else:
        slot = lid
    lane = slot.to(tl.int64) * V
    ap = A + oe[:, None] * sa_e
    bp = B + oe[:, None] * sb_e
    acc = tl.zeros((BE, W), dtype=tl.float32)
    for c in range(0, nc):
        cacc = tl.zeros((BE, W), dtype=tl.float32)
        base = k0 + c * per * V * W
        # The unroll is written out here rather than asked for with `loop_unroll_factor`. That
        # attribute reassociates the accumulator: at factor 4 and 8 the PTX grows separate
        # `mul.f32`s and a shallower add tree, and 89 of 3581 fuzzed gemv shapes came back one
        # fp16 ulp off on two to four output elements. `tl.static_range` stamps the copies out
        # in source order, so the chain stays `((p0 + p1) + p2) + ...` exactly as the frozen
        # kernel runs it. The rounded-up tiles are fully masked, and a fully masked tile adds
        # `+0.0`: `cacc` starts at `+0.0` and can never become `-0.0`, so that is a no-op.
        # Masking against the CHUNK end, not the slice end, is what lets the trip count be
        # rounded up to a whole number of unrolled groups: for the last chunk the two are the
        # same value, and for a middle chunk it stops the rounded-up tiles from reading the next
        # chunk's real data. One `tl.minimum` per chunk, and the load mask stays a single
        # compare.
        cend = tl.minimum(base + per * V * W, ke)
        for tg in range(0, tl.cdiv(per, UF)):
            for u in tl.static_range(UF):
                t = tg * UF + u
                tb = base + t * V * W
                for s in tl.static_range(V):
                    k = tb + lane + s
                    km = k < cend
                    if BC == 1:  # A is the shared vector, B carries one column per element
                        a = tl.load(A + k * sa_k, mask=km, other=0.0)[None, :]
                    else:
                        a = tl.load(ap + k[None, :] * sa_k, mask=em[:, None] & km[None, :], other=0.0)
                    if BC == 2:  # B is the shared vector, A carries one row per element
                        bb = tl.load(B + k * sb_k, mask=km, other=0.0)[None, :]
                    else:
                        bb = tl.load(bp + k[None, :] * sb_k, mask=em[:, None] & km[None, :], other=0.0)
                    cacc += a.to(tl.float32) * bb.to(tl.float32)
        acc = tl.where(c == 0, cacc, acc + cacc)
    if DOWN:
        tot = _butterfly_down(acc, BE, W)
    else:
        tot = _lane_lr(acc, BE, W)
    if NS:
        tl.store(Cout + sl.to(tl.int64) * ws + oe * sc, tot.to(Cout.dtype.element_ty), mask=em)
    else:
        tl.store(Cout + oe * sc, tot.to(Cout.dtype.element_ty), mask=em)


@triton.jit
def _gemv_cslice_x(A, B, Cout, NEL, K, sa_e, sa_k, sb_k, sb_e, sc, S, NCH, C: tl.constexpr, W: tl.constexpr,
                   BE: tl.constexpr, BC: tl.constexpr):
    """`_gemv_cslice` with the shared vector read once and the lane combine done by extraction.

    The lane layout, the C-long chunks inside a lane, and the left-to-right combine of the W
    lane totals are all the ones the frozen kernel runs; `_lane_lr` builds the same left-nested
    chain out of the same column values."""
    pe = tl.program_id(0)
    oe = (pe * BE + tl.arange(0, BE)).to(tl.int64)
    em = oe < NEL
    base = tl.arange(0, W).to(tl.int64) * S
    ap = A + oe[:, None] * sa_e
    bp = B + oe[:, None] * sb_e
    acc = tl.zeros((BE, W), dtype=tl.float32)
    for ch in range(0, NCH):
        cacc = tl.zeros((BE, W), dtype=tl.float32)
        for s in tl.static_range(C):
            off = ch * C + s
            k = base + off
            km = (k < K) & (off < S)
            if BC == 1:
                a = tl.load(A + k * sa_k, mask=km, other=0.0)[None, :]
            else:
                a = tl.load(ap + k[None, :] * sa_k, mask=em[:, None] & km[None, :], other=0.0)
            if BC == 2:
                bb = tl.load(B + k * sb_k, mask=km, other=0.0)[None, :]
            else:
                bb = tl.load(bp + k[None, :] * sb_k, mask=em[:, None] & km[None, :], other=0.0)
            cacc += a.to(tl.float32) * bb.to(tl.float32)
        acc = tl.where(ch == 0, cacc, acc + cacc)
    tl.store(Cout + oe * sc, _lane_lr(acc, BE, W).to(Cout.dtype.element_ty), mask=em)


@triton.jit
def _gemv_block_dot_x(A, B, Wsp, NEL, K, sa_e, sa_k, sb_k, sb_e, ws, NB, PER, V: tl.constexpr, BE: tl.constexpr,
                      BC: tl.constexpr, UF: tl.constexpr):
    """`_gemv_block_dot` with the shared vector read once. The V separate accumulator chains,
    the strided block tiling and the 128-lane count-down butterfly are untouched."""
    pe = tl.program_id(0)
    blk = tl.program_id(1)
    oe = (pe * BE + tl.arange(0, BE)).to(tl.int64)
    em = oe < NEL
    thr = _bitrev(tl.arange(0, 128), 128).to(tl.int64)
    ap = A + oe[:, None] * sa_e
    bp = B + oe[:, None] * sb_e
    accs = tl.zeros((BE, 128), dtype=tl.float32)
    for s in tl.static_range(V):
        acc = tl.zeros((BE, 128), dtype=tl.float32)
        # Written-out unroll, not `loop_unroll_factor` -- see the note in `_gemv_lane_x`; that
        # attribute reassociates the accumulator chain and so moves bits.
        for gg in range(0, tl.cdiv(PER, UF)):
            for u in tl.static_range(UF):
                g = gg * UF + u
                # Tile g*NB + blk is past the last tile once g reaches PER, so `k < K` already
                # masks every rounded-up group; no extra guard is needed here.
                k = (g * NB + blk) * (V * 128) + thr * V + s
                km = k < K
                if BC == 1:
                    a = tl.load(A + k * sa_k, mask=km, other=0.0)[None, :]
                else:
                    a = tl.load(ap + k[None, :] * sa_k, mask=em[:, None] & km[None, :], other=0.0)
                if BC == 2:
                    bb = tl.load(B + k * sb_k, mask=km, other=0.0)[None, :]
                else:
                    bb = tl.load(bp + k[None, :] * sb_k, mask=em[:, None] & km[None, :], other=0.0)
                acc += a.to(tl.float32) * bb.to(tl.float32)
        accs += acc
    tl.store(Wsp + blk.to(tl.int64) * ws + oe, _butterfly_down(accs, BE, 128), mask=em)


@triton.jit
def _gemv_block_reduce_x(Wsp, Cout, NEL, ws, sc, NB, MM, BE: tl.constexpr):
    """`_gemv_block_reduce` with the shared vector read once; both ascending sums are the frozen
    masked extraction, unchanged."""
    pe = tl.program_id(0)
    oe = (pe * BE + tl.arange(0, BE)).to(tl.int64)
    em = oe < NEL
    c = tl.arange(0, 128)
    thr = c // 4 + (c & 1) * 64 + ((c >> 1) & 1) * 32
    q = tl.zeros((BE, 128), dtype=tl.float32)
    for m in range(0, MM):
        i = m * 128 + thr
        v = tl.load(Wsp + i.to(tl.int64)[None, :] * ws + oe[:, None], mask=em[:, None] & (i < NB)[None, :], other=0.0)
        q = tl.where(m == 0, v, q + v)
    lanes = tl.reshape(_pair_fold(_pair_fold(q, BE, 128), BE, 64), (BE, 8, 4))
    gid = tl.arange(0, 8)[None, :, None]
    u = tl.sum(tl.where(gid == 0, lanes, 0.0), axis=1)
    for g in tl.static_range(1, 8):
        u += tl.sum(tl.where(gid == g, lanes, 0.0), axis=1)
    tot = _lane_lr(u, BE, 4)
    tl.store(Cout + oe * sc, tot.to(Cout.dtype.element_ty), mask=em)


def _gemv13_shape(nel, nsplit, W, vlen, m1, per_est):
    """BE, num_warps and the loop unroll for one ALGO 13 launch.

    All three are free. Every output element of a gemv is its own independent reduction, so how
    many of them one program owns (BE), how many warps run that program, and how many copies of
    the loop body the compiler stamps out cannot move a single addition. What they do change is
    how wide a load each thread issues and how many are in flight: for the M == 1 orientation
    the matrix is contiguous along the output axis, so a thread wants several output elements,
    and for N == 1 it is contiguous along k, which is the W axis and only W long. `m1` says
    which of the two this launch is."""
    be = min(256, max(16, _GEMV13_TILE // W))
    while be > 16 and -(-nel // be) * max(nsplit, 1) < _GEMV13_CTA_FLOOR:
        be //= 2
    # `tl.static_range(V)` already stamps out V copies of the load pair, so a thread's live
    # values are BE * W * V / threads, not BE * W / threads. Counting V here is what keeps the
    # V = 64 recipe off 1 warp, where it spills and runs 9x slower. `vlen` is not always a power
    # of two -- the "one contiguous slice per lane" recipes take it from ceil(chunk / W) -- so
    # the share is rounded down to one below, or Triton asserts on `num_warps`.
    ept = _GEMV13_EPT_M1 if m1 else _GEMV13_EPT_N1
    share = min(8, max(1, be * W * vlen // (32 * ept)))
    # A warp with no accumulator element of its own adds nothing but a duplicate of the tile, so
    # one element per thread is the widest the CTA may ever get.
    cap = max(1, min(8, be * W // 32))
    warps = min(cap, 1 << (share.bit_length() - 1))
    # 8 elements per thread otherwise leaves a small launch running on 32-thread CTAs, which on
    # the three smallest shapes measured leaves most of the GPU idle. Once the whole launch is
    # under a few warps per SM there is nothing to save registers for, so widen the CTA instead.
    ncta = -(-nel // be) * max(nsplit, 1)
    floor = platform().sm_count * _GEMV13_WARPS_PER_SM
    while warps < cap and ncta * warps < floor:
        warps *= 2
    # The written-out unroll rounds the tile count up to a whole group, and every rounded-up
    # tile is real masked work. On a short chunk that is most of the loop -- 11 tiles unrolled
    # by 8 runs 16 -- so take the widest group that wastes no more than a quarter of the trip
    # count. Correctness does not depend on this; only how much masked work is done.
    widest = min(8, max(1, _GEMV13_MAX_UNROLL // vlen))
    uf = 1
    for cand in (8, 4, 2):
        if cand <= widest and -(-per_est // cand) * cand <= per_est + per_est // 4:
            uf = cand
            break
    return be, warps, uf


def _gemv13_launch(a, b, c, nel, strides, sc, recipe, nsplit, BE, warps, uf=1, bcast=True, combine=None):
    """The sm_103 launch: one `_gemv_lane_x` for the whole split, then the fp32 combine."""
    K = a.shape[1]
    v, W, CC, down = recipe
    chunk = -(-K // nsplit) if nsplit > 1 else K
    vlen = _gemv_vlen(v, chunk, W)
    bc = 0 if not bcast else (1 if strides[0] == 0 else 2)
    grid = (triton.cdiv(nel, BE), max(nsplit, 1))
    if nsplit <= 1:
        _gemv_lane_x[grid](a, b, c, nel, K, K, *strides, sc, 0, CC, V=vlen, W=W, DOWN=down, BE=BE, BC=bc, NS=False,
                           num_warps=warps, UF=uf)
        return c
    w = torch.empty(nsplit, nel, device=DEVICE, dtype=torch.float32)
    _gemv_lane_x[grid](a, b, w, nel, K, chunk, *strides, 1, w.stride(0), CC, V=vlen, W=W, DOWN=down, BE=BE, BC=bc,
                       NS=True, num_warps=warps, UF=uf)
    cbe, cwarps = combine or _GEMV13_COMBINE
    _gemv_slice_combine[(triton.cdiv(nel, cbe), )](w, c, nel, w.stride(0), sc, nsplit, BE=cbe, num_warps=cwarps)
    return c


def _cslice_launch(a, b, c, nel, strides, sc, K, w_lanes, chunk, BE, warps, bcast=True):
    """The sm_103 launch for `CUSTOM_OPTION` 5."""
    slice_k = -(-K // w_lanes)
    bc = 0 if not bcast else (1 if strides[0] == 0 else 2)
    _gemv_cslice_x[(triton.cdiv(nel, BE), )](a, b, c, nel, K, *strides, sc, slice_k, -(-slice_k // chunk), C=chunk,
                                             W=w_lanes, BE=BE, BC=bc, num_warps=warps)
    return c


def _gemv14_launch(a, b, c, nel, strides, sc, K, nblock, v, BE, warps, rbe, rwarps, bcast=True, uf=1):
    """The sm_103 launch for ALGO_ID 14.

    The workspace is `torch.empty` rather than `torch.zeros`: pass 1 writes every (block,
    element) pair the reduce reads -- the grid covers block 0..NB-1 and both kernels mask on the
    same `oe < NEL` -- so the zero fill only ever wrote over values that were about to be
    overwritten, at the cost of a whole extra kernel in a two-kernel sequence."""
    ntile = -(-K // (v * 128))
    bc = 0 if not bcast else (1 if strides[0] == 0 else 2)
    per = max(1, -(-ntile // nblock))
    # Same round-up cost as ALGO 13: never unroll a group wider than the trip count can fill.
    uf = max([1] + [c for c in (2, 4, 8) if c <= uf and -(-per // c) * c <= per + per // 4])
    w = torch.empty(nblock, nel, device=DEVICE, dtype=torch.float32)
    _gemv_block_dot_x[(triton.cdiv(nel, BE), nblock)](a, b, w, nel, K, *strides, w.stride(0), nblock, per, V=v,
                                                      BE=BE, BC=bc, UF=uf, num_warps=warps)
    _gemv_block_reduce_x[(triton.cdiv(nel, rbe), )](w, c, nel, w.stride(0), sc, nblock, -(-nblock // 128), BE=rbe,
                                                    num_warps=rwarps)
    return c
