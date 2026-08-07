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
    w = torch.zeros(nblock, nel, device=DEVICE, dtype=torch.float32)
    _gemv_block_dot[(triton.cdiv(nel, BE), nblock)](a, b, w, nel, K, *strides, w.stride(0), nblock, -(-ntile // nblock),
                                                    V=v, BE=BE, num_warps=4)
    _gemv_block_reduce[(triton.cdiv(nel, BE), )](w, c, nel, w.stride(0), sc, nblock, -(-nblock // 128), BE=BE,
                                                 num_warps=4)
    return c
