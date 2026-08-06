"""Bit-for-bit cuBLAS-equivalent Triton GEMM, driven by the cuBLASLt heuristic.

Top-level API (a cuBLAS-like matmul):

    from bitequiv.cublas_equiv_gemm import cublas_equivalent_gemm, cublas_equivalent_scaled_mm
    c = cublas_equivalent_gemm(a_fp16, b_fp16)               # == cuBLAS fp16/bf16 GEMM, bit-for-bit
    c = cublas_equivalent_scaled_mm(a_fp8, b_fp8_col, sa, sb) # == cuBLAS fp8 GEMM, bit-for-bit

How it works
------------
For a shape (M, N, K, dtype) we ask cuBLAS's OWN heuristic what it would do, then
rebuild that exact computation in Triton:

  1. Query `cublasLtMatmulAlgoGetHeuristic` (ctypes) -> the top algo's split count
     `SPLITK_NUM` (config attr 2).  This is faithful to what cuBLAS runs.
  2. `SPLITK_NUM == 1`  -> plain GEMM (one FP32 accumulator over K).
     `SPLITK_NUM  > 1`  -> split-K: cut K into k-slices, one FP32 partial each,
     combine the partials in FORWARD order in FP32, cast to the output dtype.
  3. The partition is fully determined by `SPLITK_NUM`: at grain `G` (64 for
     fp16/bf16, 128 for fp8), `chunk = ceil(ceil(K/G)/SPLITK_NUM)*G`, slices uneven
     with the remainder in the LAST one.  Verified against cuBLAS-direct over
     131,533 fp8 split-K shapes with 0 counterexamples; a wide sweep of thousands of
     alternative partitions per shape never found a different chunk that works.
  4. HOW THE RECIPE IS DECIDED -- three tiers, tried in order (see `_resolve`):

       static         the heuristic alone settles it and nothing is executed.  Only the
                      PLAIN branch qualifies, for both dtypes.
       pseudo-static  one profiled cuBLAS launch; the remaining knobs are READ off the
                      launched kernel name (threadblock k-tile, MMA shape) and off
                      `REDUCTION_SCHEME` (which of the three split-K merge schemes).
                      Costs ~2 ms for the profile against ~0.1 ms for the GEMM, but it
                      observes what cuBLAS did instead of inferring it, so it cannot pick
                      a recipe that merely happens to agree on the seeds it was tested
                      against.  It also proves nothing: no bytes are compared here.
       runtime        search the candidate recipes and keep the one that byte-matches
                      cuBLAS on 32 selection seeds plus 8 hold-out seeds.  Proves the
                      match on those inputs; but picking the best of ~36 candidates is a
                      selection, so a recipe that fits by luck can still slip through.

     `enable_pseudo_static` (default on) and `enable_runtime_match` gate the last two.
     A shape no tier can serve raises `CublasUnsupportedShape` (or, in static-only mode,
     `CublasNeedRuntimeMatch`) -- the search path never returns an unverified guess.

Known unsupported (raise `CublasUnsupportedShape`): cuBLAS's SIMT `gemv` fallbacks for
very thin M or N, which are not CUTLASS at all; plus a thin split-K residual that
appears when K is not a whole number of k-tiles, so the last slice is a partial one and
cuBLAS handles that leftover in a way the two-pass split-K cannot match at any
partition.  fp8 needs `BM >= 64` (see `_tile`) or it silently uses a different MMA.

Depends on `torch`, `triton`, `ctypes` (stdlib), and the CUDA `libcublasLt` shared
library (always present with a CUDA runtime).  Measured on GB200 / sm_100.
"""
from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import glob
import os
import re
import threading
import warnings

import torch
import triton
import triton.language as tl

DEVICE = "cuda"
_WS_BYTES = 33554432  # 32 MiB scratch for the cuBLAS-direct reference execute
_SEEDS = tuple(range(32))  # seeds used to SELECT a reconstruction among the candidates
_CONFIRM_SEEDS = tuple(range(1000, 1008))  # held out: only the winner is checked here, so a candidate that
# fits the selection seeds by luck is caught rather than shipped. See `_calibrate`.


class CublasUnsupportedShape(Exception):
    """No Triton reconstruction bit-matches cuBLAS for this shape, even with a runtime
    byte-compare (e.g. fp8 vertical/cluster split-K, fp16 non-aligned s1688 K=8 / odd-K)."""


class CublasUnsupportedPlatform(CublasUnsupportedShape):
    """No measured strategy for this GPU architecture.

    Every reconstruction rule in this file is architecture-specific and was measured, not
    derived, so it must not be extrapolated: the sm_103 rules did not carry over to sm_100 when
    that move was tried. Subclasses `CublasUnsupportedShape` on purpose, so a caller that
    already writes `except CublasUnsupportedShape: <fall back to cuBLAS>` keeps working
    unchanged on a machine we have not measured."""


class CublasNeedRuntimeMatch(Exception):
    """The shape cannot be reconstructed from the heuristic STATICALLY with confidence; a
    runtime byte-compare against cuBLAS is needed to confirm (or reject) it. Raised only in
    static mode (`enable_runtime_match=False`). Re-call with `enable_runtime_match=True`."""


# --------------------------------------------------------------------------- #
# Per-platform strategy
#
# PORTING GUIDE. Everything cuBLAS-specific in this file lives in an ArchProfile below, so
# adding a GPU is filling in one dataclass -- no dispatch code changes. Nothing here may be
# guessed from another architecture: when the sm_103 rules were carried over to sm_100 they
# were wrong, and the same is expected in reverse. Each field says how to measure it.
#
# The gate is (compute capability, cuBLASLt version). The cuBLASLt library we call through
# ctypes is what decides the kernels -- its name literally carries the arch, e.g.
# `nvjet_sm100_...` -- so it, not the CUDA toolkit torch was built against, is what we pin.
# --------------------------------------------------------------------------- #
# What a profile that does not name its own versions -- every placeholder, until someone
# measures it -- is assumed to hold for. The newest version the rules have been measured on.
_DEFAULT_CUBLASLT = (13, 1)

@dataclasses.dataclass(frozen=True)
class ArchProfile:
    """The measured cuBLAS behaviour of one GPU architecture.

    `measured=False` means the entry is a placeholder: the dispatch knows the architecture
    exists but has no rules for it, and the API raises `CublasUnsupportedPlatform`."""

    name: str                        # human-readable, e.g. "sm_100 (NVIDIA GB200 / B200)"
    measured: bool = False
    # (major, minor) cuBLASLt versions these rules were measured against. A patch bump inside a
    # listed (major, minor) is silent; any other version warns and runs anyway, because a version
    # change *can* move which kernel a shape lands on but usually does not. A placeholder profile
    # inherits `_DEFAULT_CUBLASLT` until whoever fills it in measures which versions it holds for.
    cublaslt_versions: tuple = (_DEFAULT_CUBLASLT, )
    measured_cublaslt: str = ""      # exact version, for the record

    # -- alignment gate -------------------------------------------------------------------
    # Which contiguous dims must be aligned for cuBLAS to stay on its own (nvjet) kernels.
    # Measure: sweep shapes, profile the launched kernel name, find where it flips to CUTLASS.
    align_elems: tuple = ()          # sm_100: (("fp16", 8), ("fp8", 16)) -- see `_aligned`

    # -- nvjet split-K --------------------------------------------------------------------
    # k-slice grain of cuBLAS's own split-K, per dtype. Measure: byte-compare
    # `chunk = ceil(ceil(K/G)/SPLITK_NUM)*G` for candidate G against cuBLAS-direct.
    nvjet_grain: tuple = ()          # sm_100: (("fp16", 64), ("fp8", 128))

    # -- Triton-side constraint -----------------------------------------------------------
    # Minimum BM below which Triton stops using the native fp8 tensor-core path. Measure: dump
    # PTX over a (BM, BN, BK, num_warps) grid and look for the MMA instruction changing.
    fp8_min_bm: int = 64

    # -- CUTLASS fallback (only reached when the shape is not aligned) ---------------------
    # Real k-elements per `tl.dot`, i.e. the MMA's K. Measure: read the `s<MMA>gemm` token.
    k_per_dot_choices: tuple = ()    # sm_100: (16, 8)
    # Threadblock k-tiles CUTLASS uses here; decides where the partial group sits. Measure:
    # read the `<tbK>x<stages>` token from the launched name over many shapes.
    ktile_choices: tuple = ()        # sm_100: (32, 64, 128)
    # Default k-tile when the name omits that field (two-stage sm_75 instances do).
    ktile_default: int = 32
    # Partition grains to try for CUTLASS split-K. Measure: byte-compare, and cross-check
    # against `kAlignK` in `cutlass/.../params_universal_base.h`.
    splitk_grains: tuple = ()        # sm_100: (8, 64)
    # CUBLASLT_REDUCTION_SCHEME -> our merge scheme. Measure: cross-tabulate attr 3 against the
    # merge scheme that byte-matches. sm_100: NONE -> serial fp16, COMPUTE_TYPE -> fp32 forward
    # sum, OUTPUT_TYPE -> fp16 partials.
    reduction_to_cmode: tuple = ()   # sm_100: ((0, 3), (2, 0), (4, 2))


_SM100 = ArchProfile(
    name="sm_100 (NVIDIA GB200 / B200)",
    measured=True,
    cublaslt_versions=((13, 1), (12, 8)),
    measured_cublaslt="13.1.1 and 12.8.5",
    align_elems=(("fp16", 8), ("bf16", 8), ("fp8", 16)),
    nvjet_grain=(("fp16", 64), ("bf16", 64), ("fp8", 128)),
    fp8_min_bm=64,
    k_per_dot_choices=(16, 8),
    ktile_choices=(32, 64, 128),
    ktile_default=32,
    splitk_grains=(8, 64),
    reduction_to_cmode=((0, 3), (2, 0), (4, 2)),
)

# Placeholders. Fill in the fields above on the machine in question and flip `measured=True`;
# the dispatch, the kernels and the eval need no changes. Re-measure every field -- the values
# in `_SM100` are not a starting point, they are a different architecture's answer.
_SM103 = ArchProfile(name="sm_103 (NVIDIA GB300)")
_SM90 = ArchProfile(name="sm_90 (NVIDIA H100)")

_PROFILES = {(10, 0): _SM100, (10, 3): _SM103, (9, 0): _SM90}


def _cublaslt_version():
    """(major, minor, patch) of the cuBLASLt we actually call."""
    return _lt_version_of(_load_lt())


def _platform():
    """The ArchProfile for the current device, or raise CublasUnsupportedPlatform."""
    cap = torch.cuda.get_device_capability()
    major, minor, patch = _cublaslt_version()
    if (cap, major, minor) in _PLATFORM_CACHE:
        return _PLATFORM_CACHE[(cap, major, minor)]
    prof = _PROFILES.get(cap)
    dev = torch.cuda.get_device_name()
    if prof is None or not prof.measured:
        known = ", ".join(p.name for p in _PROFILES.values() if p.measured)
        raise CublasUnsupportedPlatform(
            f"unsupported on sm_{cap[0]}{cap[1]} ({dev}): no cuBLAS-equivalence strategy has been "
            f"measured for this GPU. Supported: {known}. The reconstruction rules are "
            f"architecture-specific and must be re-measured, not extrapolated -- see ArchProfile.")
    if (major, minor) not in prof.cublaslt_versions:
        want = ", ".join(f"{a}.{b}.x" for a, b in prof.cublaslt_versions)
        exact = f" (exactly {prof.measured_cublaslt})" if prof.measured_cublaslt else ""
        # Warn, do not refuse. A cuBLASLt update can move which kernel a shape lands on, but in
        # practice most shapes are unaffected, and every reconstruction is still either checked
        # against cuBLAS (runtime tier) or read off the kernel cuBLAS actually launched
        # (pseudo-static tier). Only the static tier trusts the rules blind. Warn once per
        # architecture: the cache below is filled exactly once, so a fuzzer loop stays quiet.
        warnings.warn(
            f"cuBLASLt {major}.{minor}.{patch} on {prof.name} was not measured: the rules come from "
            f"{want}{exact}. Results should still be bit-identical to cuBLAS, but a version bump can "
            f"change which kernel a shape lands on. Re-measure and add {major}.{minor} to "
            f"ArchProfile.cublaslt_versions to silence this.", RuntimeWarning, stacklevel=2)
    _PLATFORM_CACHE[(cap, major, minor)] = prof
    return prof


_PLATFORM_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# Triton kernels (scale-capable, single FP32 accumulator)
# --------------------------------------------------------------------------- #
@triton.jit
def _plain_gemm(A, B, C, M, N, K, am, ak, bk, bn, cm, cn, sa, sb, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr):
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
    acc = acc * sa * sb
    ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], acc.to(C.dtype.element_ty),
             mask=(ocm[:, None] < M) & (ocn[None, :] < N))


@triton.jit
def _plain_gemm_k_per_dot(A, B, C, M, N, K, am, ak, bk, bn, cm, cn, sa, sb, KPD, RES, BM: tl.constexpr,
                          BN: tl.constexpr, BK: tl.constexpr):
    """Plain GEMM that rounds its accumulator exactly where cuBLAS's CUTLASS kernel does.

    A non-aligned shape sends cuBLAS to a CUTLASS kernel whose MMA takes 8 real k-elements
    (`s1688`) or 16 (`s16816`, `s161616`), and which updates the fp32 accumulator once per
    MMA. Two things are needed to match it. (a) A k16 MMA whose upper k-lanes are exactly zero
    is bit-identical to a k8 MMA, so Triton never has to emit `m16n8k8` -- which it cannot do
    anyway; we zero-pad the k-tile to BK with a load mask. (b) The accumulator has to round at
    the same k boundaries, so we consume exactly KPD real k-elements per `tl.dot`.

    Where the partial group sits is the subtle part. CUTLASS handles a K that is not a whole
    number of mainloop k-tiles in its FIRST iteration, but the MMAs inside that iteration
    still march on the tile's own grid from 0, so the partial MMA lands at index
    `floor(RES/KPD)`, not at position 0. RES is `K % ktile` for the kernel's k-tile (32, 64 or
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
    pre = RES // KPD             # whole groups that precede the partial one
    part = RES - pre * KPD       # size of the partial group, 0 when RES is a multiple of KPD
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
    acc = acc * sa * sb
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
def _splitk_combine(W, C, M, N, ws, wm, wn, cm, cn, S, sa, sb, BM: tl.constexpr, BN: tl.constexpr):
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
    acc = acc * sa * sb
    tl.store(C + cm * om[:, None] + cn * on[None, :], acc.to(C.dtype.element_ty), mask=m2)


@triton.jit
def _splitk_partial_k_per_dot(A, B, W, M, N, K, am, ak, bk, bn, ws, wm, wn, CHUNK, KPD, KTILE, BM: tl.constexpr,
                              BN: tl.constexpr, BK: tl.constexpr):
    """Pass 1 for the CUTLASS split-K path: slice `sk` covers [sk*CHUNK, min((sk+1)*CHUNK, K))
    and accumulates it in groups of KPD real k-elements, with the slice's own residue tile.

    Same idea as `_plain_gemm_k_per_dot` but applied PER SLICE: a CUTLASS slice whose length is
    not a whole number of threadblock k-tiles runs a partial first tile of `klen % KTILE`
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
    rbk = klen % KTILE
    rbk = tl.minimum(tl.where(rbk == 0, KTILE, rbk), klen)  # length of the leading residue tile
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
def _splitk_combine_modes(W, C, M, N, ws, wm, wn, cm, cn, S, sa, sb, CMODE: tl.constexpr, BM: tl.constexpr,
                          BN: tl.constexpr):
    """Pass 2 with the three merge schemes cuBLAS actually uses, which `SPLITK_NUM` does not
    distinguish -- only the launched kernel names do:

      CMODE 0  fp32 workspace, forward fp32 sum                     (`splitKreduce<..., float>`)
      CMODE 2  partials rounded to fp16, then forward fp32 sum      (`splitKreduce<..., __half>`)
      CMODE 3  serial chain kept in fp16 between slices             (`GemmSplitKSerial`)

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
        if CMODE == 2:  # the GEMM wrote an fp16 workspace, so each partial is rounded once
            p = p.to(tl.float16).to(tl.float32)
        acc = tl.where(i == 0, p, acc + p)
        if CMODE == 3:  # the running total lives in fp16 between slices
            acc = acc.to(tl.float16).to(tl.float32)
    acc = acc * sa * sb
    tl.store(C + cm * om[:, None] + cn * on[None, :], acc.to(C.dtype.element_ty), mask=m2)


def _tile(M: int, N: int, dtype=None) -> tuple[int, int]:
    """Partial tile: small tiles for small M,N, capped at 128.

    Bit-neutral EXCEPT for one hard constraint: fp8 needs BM >= 64. Below that Triton
    cannot use the native fp8 tensor-core path -- it upcasts fp8 to f16 and emits
    `mma.sync.m16n8k16` instead of `tcgen05.mma.kind::f8f6f4`, which rounds differently
    from cuBLAS. (fp16/bf16 agree on both paths, so they are free to use a small tile.)"""
    BM = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
    BN = 16 if N <= 16 else 32 if N <= 32 else 64 if N <= 64 else 128
    if dtype == torch.float8_e4m3fn:
        BM = max(BM, _platform().fp8_min_bm)
    return BM, BN


def _kcontig(b):
    return b if b.stride(1) == 1 else b.contiguous()


def _triton_plain(a, b, out_dtype, sa=1.0, sb=1.0, BM=128, BN=128, BK=64):
    b = _kcontig(b)
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), )
    _plain_gemm[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                      sa, sb, BM=BM, BN=BN, BK=BK, num_warps=8)
    return c


def _triton_plain_k_per_dot(a, b, out_dtype, k_per_dot, res, sa=1.0, sb=1.0, BM=128, BN=128, BK=16):
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
                                c.stride(1), sa, sb, k_per_dot, res, BM=BM, BN=BN, BK=BK, num_warps=8)
    return c


def _k_per_dot_candidates(K):
    """`(k_per_dot, res)` pairs to try for a non-aligned shape, deduped.

    `k_per_dot` is 8 for a `s1688` kernel and 16 for `s16816` / `s161616`; `res = K % ktile`
    for the kernel's mainloop k-tile, which cuBLAS picks for performance and is 32, 64 or 128.
    Neither is derivable from the shape -- both are visible only in the launched kernel name,
    and parity gets `k_per_dot` wrong for the WMMA kernels -- so try all six. Measured over
    1,806 non-aligned single-pass fp16 shapes at 32 seeds: exactly one of the six byte-matches
    every shape, with no exceptions, taking that family from 33.4% to 100%."""
    prof = _platform()
    out, seen = [], set()
    for k_per_dot in prof.k_per_dot_choices:
        for ktile in prof.ktile_choices:
            cand = (k_per_dot, K % ktile)
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def _triton_splitk_groups(a, b, chunk, k_per_dot, ktile, cmode, out_dtype, sa=1.0, sb=1.0, BK=16):
    """Two-pass split-K for the CUTLASS path: `chunk`-element slices, each accumulated in
    groups of `k_per_dot` real k with a per-slice residue tile of `ktile`, merged by `cmode`."""
    b = _kcontig(b)
    M, K = a.shape
    N = b.shape[1]
    nsplit = (K + chunk - 1) // chunk
    if nsplit < 1 or nsplit > 8192:
        return None
    BM, BN = _tile(M, N, a.dtype)
    ntile = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    w = torch.empty(nsplit, M, N, device=DEVICE, dtype=torch.float32)
    _splitk_partial_k_per_dot[(ntile, nsplit)](a, b, w, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                               w.stride(0), w.stride(1), w.stride(2), chunk, k_per_dot, ktile, BM=BM,
                                               BN=BN, BK=BK, num_warps=4)
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    _splitk_combine_modes[(ntile, )](w, c, M, N, w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1),
                                     nsplit, sa, sb, CMODE=cmode, BM=BM, BN=BN, num_warps=4)
    return c


def _splitk_group_candidates(K, nsplit):
    """`(chunk, k_per_dot, ktile, cmode)` recipes to try for a non-aligned split-K shape.

    The partition grain is CUTLASS's `kAlignK` (8 = 128 bits / 16, not the 64 the nvjet path
    uses, though a small tail needs 64), the k per dot follows the MMA (`s1688` -> 8,
    `s16816`/`s161616` -> 16), the merge scheme is one of three, and the threadblock k-tile is a
    cuBLAS performance choice. Ordered by measured frequency so the seed-0 prefilter usually
    stops on the first few. Measured over 520 shapes: 100% reconstructed, median calibration
    0.1 s per shape.

    Two of these are in fact readable from the heuristic and this search does not yet use them:
    `REDUCTION_SCHEME` (attr 3) gives the merge scheme exactly (`COMPUTE_TYPE` -> fp32 forward,
    `OUTPUT_TYPE` -> fp16 partials, `NONE` -> serial fp16; 119 shapes, no mixing), and
    `(ALGO_ID, CUSTOM_OPTION, STAGES_ID)` gives the CUTLASS subtype and hence the k per dot.
    Using them would cut the list from 36 candidates to 3. The k-tile is the one knob no
    heuristic field pins -- config groups identical in every readable field still differ in it --
    so a byte-compare is still needed, which is why these shapes are `runtime`, not `static`."""
    prof = _platform()
    chunks = []
    for g in prof.splitk_grains:
        chunk = _chunk_from_ns(K, nsplit, g)
        if g <= chunk <= K and chunk not in chunks:
            chunks.append(chunk)
    out = []
    for cmode in (3, 2, 0):        # serial-fp16 70%, fp16-partials 17%, fp32-forward 12%
        for k_per_dot in reversed(prof.k_per_dot_choices):  # 8 is 80%
            for ktile in prof.ktile_choices:  # 32 is 86%
                for chunk in chunks:
                    out.append((chunk, k_per_dot, ktile, cmode))
    return out


def _triton_splitk(a, b, chunk, out_dtype, sa=1.0, sb=1.0, BK=64):
    """Deterministic two-pass split-K at `chunk`-element early slices (uneven tail),
    forward FP32 combine. Returns None if the chunk does not yield a real split (nsplit<2)."""
    b = _kcontig(b)
    M, K = a.shape
    N = b.shape[1]
    nsplit = (K + chunk - 1) // chunk
    if nsplit < 2 or nsplit > 8192:
        return None
    BM, BN = _tile(M, N, a.dtype)
    ntile = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    w = torch.empty(nsplit, M, N, device=DEVICE, dtype=torch.float32)
    _splitk_partial[(ntile, nsplit)](a, b, w, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), w.stride(0),
                                     w.stride(1), w.stride(2), chunk, BM=BM, BN=BN, BK=BK, num_warps=4)
    c = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
    _splitk_combine[(ntile, )](w, c, M, N, w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1), nsplit, sa,
                               sb, BM=BM, BN=BN, num_warps=4)
    return c


# --------------------------------------------------------------------------- #
# cuBLASLt via ctypes: heuristic query + run the chosen algo directly (reference)
# --------------------------------------------------------------------------- #
_CUDA_R_16F, _CUDA_R_32F, _CUDA_R_16BF, _CUDA_R_8F_E4M3 = 2, 0, 14, 28
_CUBLAS_COMPUTE_32F = 68
_OP_N, _OP_T = 0, 1
_DESC_TRANSA, _DESC_TRANSB = 3, 4
_DESC_A_SCALE_PTR, _DESC_B_SCALE_PTR = 17, 18
_PREF_MAX_WS = 1
_LAYOUT_ORDER, _ORDER_ROW = 1, 1
_CFG_ID, _CFG_SPLITK, _CFG_REDUCTION = 0, 2, 3
_LT_SPEC = None   # process-wide choice from set_cublaslt(); None = newest installed
_TLS = threading.local()   # per-CALL overrides live here, never in a global -- see _using_cublaslt
_UNSET = object()
_LT_LIBS: dict = {}   # resolved path -> loaded library, so switching back and forth is free
_LT_PATHS: dict = {}  # spec -> resolved path
_ONES = None  # device fp32 [1.0] for fp8 scale pointers
_WS = None    # device scratch for the reference execute


class _HeurResult(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_char * 64), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


def set_cublaslt(spec=None):
    """Choose which cuBLAS to be bit-identical to, for the rest of the process.

    This is process-wide, as the name says. A single call overrides it with `cublaslt=`, and
    that override is thread-local, so one thread cannot retarget a call in flight on another.

    `spec` is a version prefix ("12.8", "13.1.1") or a full path to a libcublasLt; `None`
    restores auto-detection, which takes the newest installed. Which cuBLAS is the target is
    part of the claim and not an implementation detail -- a box can carry several, and torch's
    bundled one is usually not the newest. A single call can override this with `cublaslt=`."""
    global _LT_SPEC
    _LT_SPEC = spec


def _lt_spec():
    """Which cuBLAS the call on THIS thread is targeting: its own override if it set one,
    otherwise the process-wide choice."""
    v = getattr(_TLS, "lt_spec", _UNSET)
    return _LT_SPEC if v is _UNSET else v


@contextlib.contextmanager
def _using_cublaslt(spec):
    """Select `spec` for one call. Held in thread-local state rather than a global: two threads
    can be matching two different cuBLAS at the same time, and a global would let one of them
    retarget the other mid-flight. Libraries are cached per path, so switching is cheap, and the
    plan cache keys on the version, so a plan made against one is never reused for another."""
    if spec is None:
        yield
        return
    prev = getattr(_TLS, "lt_spec", _UNSET)
    _TLS.lt_spec = spec
    try:
        yield
    finally:
        if prev is _UNSET:
            del _TLS.lt_spec
        else:
            _TLS.lt_spec = prev


def _lt_candidates():
    """Installed libcublasLt paths, newest first. The version comes from the file name, so
    picking one does not mean loading all of them."""
    seen, out = set(), []
    for p in (glob.glob("/usr/local/cuda*/lib64/libcublasLt.so*") +
              glob.glob("/usr/local/cuda*/targets/*/lib/libcublasLt.so*")):
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        seen.add(rp)
        m = re.search(r"\.so\.([\d.]+)$", rp)
        out.append((tuple(int(x) for x in m.group(1).split(".")) if m else (), rp))
    out.sort(reverse=True)
    return [p for _, p in out] + ["libcublasLt.so"]


def _lt_version_of(lib):
    lib.cublasLtGetVersion.restype = ctypes.c_size_t
    v = int(lib.cublasLtGetVersion())
    return v // 10000, (v // 100) % 100, v % 100


def _resolve_lt(spec):
    """`spec` (path, version prefix, or None) -> the path to load."""
    if spec in _LT_PATHS:
        return _LT_PATHS[spec]
    if spec and ("/" in spec or spec.endswith(".so")):
        cands, want = [spec], None
    else:
        cands = _lt_candidates()
        want = tuple(int(x) for x in spec.split(".")) if spec else None
    for c in cands:
        try:
            lib = ctypes.CDLL(c)
        except OSError:
            continue
        if want is not None and _lt_version_of(lib)[:len(want)] != want:
            continue
        _LT_PATHS[spec] = os.path.realpath(c)
        return _LT_PATHS[spec]
    raise OSError(f"no libcublasLt matching {spec!r}; tried {cands[:4]}")


def _load_lt():
    global _ONES, _WS
    path = _resolve_lt(_lt_spec())
    lib = _LT_LIBS.get(path)
    if lib is not None:
        return lib
    lib = ctypes.CDLL(path)
    P, PP = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    ci, cu64, ci64, csz = ctypes.c_int, ctypes.c_uint64, ctypes.c_int64, ctypes.c_size_t
    lib.cublasLtCreate.argtypes = [PP]
    lib.cublasLtDestroy.argtypes = [P]
    lib.cublasLtMatmulDescCreate.argtypes = [PP, ci, ci]
    lib.cublasLtMatmulDescDestroy.argtypes = [P]
    lib.cublasLtMatmulDescSetAttribute.argtypes = [P, ci, P, csz]
    lib.cublasLtMatrixLayoutCreate.argtypes = [PP, ci, cu64, cu64, ci64]
    lib.cublasLtMatrixLayoutDestroy.argtypes = [P]
    lib.cublasLtMatrixLayoutSetAttribute.argtypes = [P, ci, P, csz]
    lib.cublasLtMatmulPreferenceCreate.argtypes = [PP]
    lib.cublasLtMatmulPreferenceDestroy.argtypes = [P]
    lib.cublasLtMatmulPreferenceSetAttribute.argtypes = [P, ci, P, csz]
    lib.cublasLtMatmulAlgoGetHeuristic.argtypes = [P, P, P, P, P, P, P, ci, P, ctypes.POINTER(ci)]
    lib.cublasLtMatmulAlgoConfigGetAttribute.argtypes = [P, ci, P, csz, ctypes.POINTER(csz)]
    lib.cublasLtMatmul.restype = ci
    lib.cublasLtMatmul.argtypes = [P] * 14 + [csz, P]  # 14 ptrs, workspaceSize, stream
    if _ONES is None:
        _ONES = torch.ones(1, dtype=torch.float32, device=DEVICE)
        _WS = torch.empty(_WS_BYTES, dtype=torch.uint8, device=DEVICE)
    _LT_LIBS[path] = lib
    return lib


def _ck(nm, rc):
    if rc != 0:
        raise RuntimeError(f"cublasLt {nm} failed (status {rc})")


def _cfg(algo_ptr, attr):
    lib = _load_lt()
    o = ctypes.c_int(-1)
    w = ctypes.c_size_t(0)
    r = lib.cublasLtMatmulAlgoConfigGetAttribute(algo_ptr, attr, ctypes.byref(o), 4, ctypes.byref(w))
    return o.value if r == 0 else None


def _lt_dtypes(kind, out_dtype):
    """(ab_dt, cd_dt, transa, transb) for the cuBLASLt layouts, per input dtype."""
    cd = _CUDA_R_16BF if out_dtype == torch.bfloat16 else _CUDA_R_16F
    if kind == "fp8":
        return _CUDA_R_8F_E4M3, cd, _OP_T, _OP_N  # e4m3 TN
    ab = _CUDA_R_16BF if kind == "bf16" else _CUDA_R_16F
    return ab, cd, _OP_N, _OP_N


def _set_row(lib, layout):
    v = ctypes.c_int(_ORDER_ROW)
    _ck("layout-order", lib.cublasLtMatrixLayoutSetAttribute(layout, _LAYOUT_ORDER, ctypes.byref(v), 4))


def _make_layouts(lib, desc, M, N, K, kind, out_dtype):
    """Row-major layouts matching a standard torch GEMM dispatch; set transa/transb
    (+ fp8 scale pointers). Returns (A, B, C, D) layout handles."""
    ab, cd, transa, transb = _lt_dtypes(kind, out_dtype)
    ta, tb = ctypes.c_int(transa), ctypes.c_int(transb)
    _ck("transa", lib.cublasLtMatmulDescSetAttribute(desc, _DESC_TRANSA, ctypes.byref(ta), 4))
    _ck("transb", lib.cublasLtMatmulDescSetAttribute(desc, _DESC_TRANSB, ctypes.byref(tb), 4))
    A, B, C, D = (ctypes.c_void_p() for _ in range(4))
    if kind != "fp8":
        _ck("LA", lib.cublasLtMatrixLayoutCreate(ctypes.byref(A), ab, M, K, K))
        _set_row(lib, A)
        _ck("LB", lib.cublasLtMatrixLayoutCreate(ctypes.byref(B), ab, K, N, N))
        _set_row(lib, B)
    else:  # fp8 e4m3 TN: operands K-contiguous (col-major), output row-major
        _ck("LA", lib.cublasLtMatrixLayoutCreate(ctypes.byref(A), ab, K, M, K))
        _ck("LB", lib.cublasLtMatrixLayoutCreate(ctypes.byref(B), ab, K, N, K))
        sp = ctypes.c_void_p(_ONES.data_ptr())
        _ck("Ascale", lib.cublasLtMatmulDescSetAttribute(desc, _DESC_A_SCALE_PTR, ctypes.byref(sp), 8))
        _ck("Bscale", lib.cublasLtMatmulDescSetAttribute(desc, _DESC_B_SCALE_PTR, ctypes.byref(sp), 8))
    _ck("LC", lib.cublasLtMatrixLayoutCreate(ctypes.byref(C), cd, M, N, N))
    _set_row(lib, C)
    _ck("LD", lib.cublasLtMatrixLayoutCreate(ctypes.byref(D), cd, M, N, N))
    _set_row(lib, D)
    return A, B, C, D


def _cublas_direct(a, b, kind, out_dtype, execute=True):
    """Query cuBLAS's top-heuristic algo and (if execute) run it DIRECTLY via ctypes
    cublasLtMatmul. Returns (D, nsplit): with execute, D is cuBLAS's exact output; without,
    D is None and only the heuristic's SPLITK_NUM is read (pure-static, no GEMM run). No torch."""
    lib = _load_lt()
    M, K = a.shape
    N = b.shape[1]
    handle, desc, pref = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    A = B = C = D = None
    try:
        _ck("create", lib.cublasLtCreate(ctypes.byref(handle)))
        _ck("desc", lib.cublasLtMatmulDescCreate(ctypes.byref(desc), _CUBLAS_COMPUTE_32F, _CUDA_R_32F))
        A, B, C, D = _make_layouts(lib, desc, M, N, K, kind, out_dtype)
        _ck("pref", lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)))
        ws = ctypes.c_size_t(_WS_BYTES)
        _ck("pref-set", lib.cublasLtMatmulPreferenceSetAttribute(pref, _PREF_MAX_WS, ctypes.byref(ws), 8))
        results = (_HeurResult * 16)()
        ret = ctypes.c_int(0)
        rc = lib.cublasLtMatmulAlgoGetHeuristic(handle, desc, A, B, C, D, pref, 16,
                                                ctypes.cast(results, ctypes.c_void_p), ctypes.byref(ret))
        if rc != 0 or ret.value == 0:
            raise CublasUnsupportedShape(f"no cuBLAS algo for {M}x{N}x{K} {kind} (rc={rc}, ret={ret.value})")
        algo_ptr = ctypes.cast(ctypes.byref(results[0]), ctypes.c_void_p)
        nsplit = _cfg(algo_ptr, _CFG_SPLITK)
        nsplit = nsplit if isinstance(nsplit, int) and nsplit >= 1 else 1
        reduction = _cfg(algo_ptr, _CFG_REDUCTION)
        if not execute:
            return None, nsplit, reduction
        out = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = lib.cublasLtMatmul(handle, desc, ctypes.cast(ctypes.pointer(alpha), ctypes.c_void_p),
                                ctypes.c_void_p(a.data_ptr()), A, ctypes.c_void_p(b.data_ptr()), B,
                                ctypes.cast(ctypes.pointer(beta), ctypes.c_void_p), ctypes.c_void_p(out.data_ptr()), C,
                                ctypes.c_void_p(out.data_ptr()), D, algo_ptr,
                                ctypes.c_void_p(_WS.data_ptr()), ctypes.c_size_t(_WS_BYTES), stream)
        _ck("matmul", rc)
        torch.cuda.synchronize()
        return out, nsplit, reduction
    finally:
        for h, d in [(pref, lib.cublasLtMatmulPreferenceDestroy), (D, lib.cublasLtMatrixLayoutDestroy),
                     (C, lib.cublasLtMatrixLayoutDestroy), (B, lib.cublasLtMatrixLayoutDestroy),
                     (A, lib.cublasLtMatrixLayoutDestroy), (desc, lib.cublasLtMatmulDescDestroy),
                     (handle, lib.cublasLtDestroy)]:
            try:
                if h is not None and h.value:
                    d(h)
            except Exception:
                pass


def cublas_matmul(a, b, out_dtype=None, cublaslt=None):
    """cuBLAS's OWN output (run directly via cublasLtMatmul), the reference our
    reconstruction must match. fp16/bf16: `b` is [K,N]. fp8: `b` is [K,N] column-major.
    `cublaslt` picks which cuBLAS, as in `cublas_equivalent_gemm`."""
    kind = _kind_of(a)
    out_dtype = out_dtype or (torch.float16 if kind == "fp8" else a.dtype)
    with _using_cublaslt(cublaslt):
        return _cublas_direct(a, b, kind, out_dtype)[0]


# --------------------------------------------------------------------------- #
# Reconstruction planning (verify-then-use) + per-shape cache
# --------------------------------------------------------------------------- #
# (capability,cublaslt,M,N,K,kind,out_dtype) -> (origin, plan). The capability and the cuBLASLt
# version keep a process that switched either one from reusing the old recipe. origin is
# "static" | "pseudo-static" | "runtime" | "unsupported"; a "runtime" plan is withheld in
# static mode.
_PLAN: dict[tuple, tuple] = {}


def _kind_of(a) -> str:
    if a.dtype == torch.float8_e4m3fn:
        return "fp8"
    if a.dtype == torch.bfloat16:
        return "bf16"
    return "fp16"


def _chunk_from_ns(K, ns, G):
    total = (K + G - 1) // G
    return ((total + ns - 1) // ns) * G


def _grain(kind):
    """k-slice granularity of cuBLAS's own (nvjet) split-K, from the platform profile."""
    return dict(_platform().nvjet_grain)[kind]


def _static_chunk(K, nsplit, kind):
    """The k-slice partition the cuBLAS heuristic implies: G-grain, uneven, remainder last."""
    return _chunk_from_ns(K, nsplit, _grain(kind))


def _splitk_candidate_chunks(K, nsplit, kind):
    """The single partition cuBLAS uses -- there is nothing to search.

    `chunk = chunk_from_ns(K, SPLITK_NUM, G)`, uneven, remainder in the LAST slice. Verified
    over 131,533 fp8 split-K shapes with 0 counterexamples, and for fp16 a wide re-sweep of
    7k-21k alternative partitions per failing shape recovered the rule chunk on controls and
    found an alternative for only 4 of 195. So if this chunk does not byte-verify, the shape
    is a genuine residual, not a missed split count."""
    G = _grain(kind)
    chunk = _chunk_from_ns(K, nsplit, G)
    return [chunk] if G <= chunk < K else []


def _make_inputs(M, N, K, kind, seed):
    """Calibration inputs, alternating TWO distributions across seeds. A reconstruction has to
    pass both, because neither alone discriminates.

      even seed -- FLAT: same-scale gaussians, so every k term contributes comparably and a
        wrong grouping of the bulk of K changes the sum.
      odd seed  -- SPREAD: a logspace magnitude ramp with alternating sign along K, which
        stresses cancellation and the placement of the large terms.

    SPREAD alone is a trap: with a 1e-3..1e3 ramp the largest-k products dominate and the rest
    are absorbed in fp32, so the result barely depends on how the bulk of K is grouped and a
    wrong reconstruction passes. Measured: with SPREAD-only calibration, 180 of 281 non-aligned
    shapes that "passed" 32 seeds were then wrong on ordinary flat inputs.

    Magnitudes stay in range so the output is finite (fp8 values must fit e4m3, max 448)."""
    torch.manual_seed(seed)
    flat = seed % 2 == 0
    sign = torch.where(torch.arange(K, device=DEVICE) % 2 == 0, 1.0, -1.0).unsqueeze(0)
    if kind == "fp8":
        af = torch.randn(M, K, device=DEVICE)
        if not flat:
            af = af * torch.logspace(-1, 1, K, device=DEVICE, dtype=torch.float32).unsqueeze(0) * sign
        a = (af * (0.5 if flat else 1.0)).to(torch.float8_e4m3fn)
        b = (torch.randn(N, K, device=DEVICE) * 0.25).to(torch.float8_e4m3fn).t()  # [K,N] col-major
        return a, b
    dt = torch.bfloat16 if kind == "bf16" else torch.float16
    af = torch.randn(M, K, device=DEVICE)
    if not flat:
        af = af * torch.logspace(-3, 3, K, device=DEVICE, dtype=torch.float32).unsqueeze(0) * sign
    a = (af * (0.3 if flat else 1.0)).to(dt)
    b = (torch.randn(K, N, device=DEVICE) * (0.3 if flat else 0.05)).to(dt)
    return a, b


def _bits_eq(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


def _aligned(M, N, K, kind):
    """Contiguous-dim alignment cuBLAS needs to stay on its reconstructable nvjet path.
    fp16/bf16: K%8==0 and N%8==0 (M free). fp8: all dims %16."""
    n = dict(_platform().align_elems)[kind]
    return (M % n == 0 and N % n == 0 and K % n == 0) if kind == "fp8" else (K % n == 0 and N % n == 0)


def _is_static_exact(kind, nsplit, aligned):
    """Can we reconstruct from the heuristic ALONE (no runtime byte-compare) and be sure it is
    bit-exact? Only the PLAIN branch, for both dtypes.

    Measured against cuBLAS-direct over 251,704 shapes / 1.07M byte-compares (GB300/sm_100):
      - aligned + SPLITK_NUM == 1: fp8 0 violations in 48,635, fp16 3 in 16,902 (0.018%);
      - aligned SPLIT-K: NOT safe. 450 of 40,097 aligned fp16 split-K shapes (1.12%) are not
        bit-exact, and no partition fixes them (K is not a whole number of k-tiles there, and
        cuBLAS treats the partial last slice in a way the two-pass split-K cannot express);
        fp8 split-K has a 0.41% residual. Both go through the runtime match;
      - non-aligned: NO (CUTLASS s1688 K=8 / odd-K tails).
    An earlier fp8-only `ceil(M/64)*ceil(N/64) >= 64` gate was dropped: 0 violations in 33,013
    small-output plain fp8 shapes, so `SPLITK_NUM == 1` alone is the right condition."""
    return aligned and nsplit <= 1


def _reconstruct(a, b, out_dtype, plan, sa=1.0, sb=1.0):
    mode, arg = plan
    if mode == "plain":
        return _triton_plain(a, b, out_dtype, sa=sa, sb=sb)
    if mode == "k_per_dot":
        return _triton_plain_k_per_dot(a, b, out_dtype, arg[0], arg[1], sa=sa, sb=sb)
    if mode == "splitk_groups":
        return _triton_splitk_groups(a, b, arg[0], arg[1], arg[2], arg[3], out_dtype, sa=sa, sb=sb)
    return _triton_splitk(a, b, arg, out_dtype, sa=sa, sb=sb)


def _launched_kernel_names(a, b, out_dtype):
    """Run cuBLAS once under the profiler and return the CUDA kernels it launched.

    The launched kernel name is the only place cuBLAS states its threadblock k-tile, and that
    is the one knob no heuristic field pins: config groups identical in `ALGO_ID`, `TILE_ID`,
    `CUSTOM_OPTION`, `STAGES_ID` and `REDUCTION_SCHEME` still differ in it. Reading the name
    costs about 2 ms per shape against 0.1 ms for the GEMM alone, but unlike a byte-compare it
    is a direct observation of what cuBLAS did rather than an inference from the output."""
    from torch.profiler import ProfilerActivity, profile
    try:
        cublas_matmul(a, b, out_dtype)  # warm up: the first call loads the kernel
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            cublas_matmul(a, b, out_dtype)
            torch.cuda.synchronize()
        return [e.key for e in prof.key_averages() if e.self_device_time_total > 0]
    except Exception:
        return []


def _parse_launched(names):
    """Pull the reconstruction knobs out of the launched kernel names.

    CUTLASS 2.x profiler names read `cutlass_<arch>_<op>_s<MMA>gemm_<dt>_<tbM>x<tbN>_<tbK>x<stages>
    _<layout>_align<n>`, so the MMA gives the k-elements per dot (`s1688` -> 8, the k16 subtypes
    -> 16) and the second tile field gives the threadblock k-tile. nvjet names read
    `nvjet_<arch>_<dt>_<T1>x<T2>_<BK>x<STG>_<CxxCy>_...`, where the second field's first number is
    the k-slice grain. Returns None for anything else -- notably cuBLAS's SIMT `gemv` fallbacks,
    which are not CUTLASS at all and have no such structure."""
    raw = next((n for n in names if "cutlass_" in n or "nvjet_" in n), None)
    if raw is None:
        return None
    # A demangled symbol repeats the kernel name (template argument, then `::Params`), so parse
    # the FIRST occurrence only -- scanning the whole string finds the M x N tile twice and
    # mistakes the repeat for the k-tile field.
    core = re.search(r"(?:cutlass|nvjet)_[A-Za-z0-9_]+", raw).group(0)
    tiles = [t for t in core.split("_") if re.fullmatch(r"\d+x\d+", t)]
    if core.startswith("nvjet"):
        # nvjet_<arch>_<dtypes>_<T1xT2>_<BKxSTAGES>_<CxxCy>_...: the second field holds the grain.
        return {"family": "nvjet", "k_per_dot": None,
                "block_k": int(tiles[1].split("x")[0]) if len(tiles) >= 2 else None}
    m = re.search(r"_s(\d+)gemm", core)
    if m is None:
        return None
    k_per_dot = 8 if m.group(1) == "1688" else 16  # s1688 is m16n8k8, the rest here are k16 MMAs
    # cutlass_<arch>_<op>_[<dt>_]s<MMA>gemm_<dt>_<tbMxtbN>[_<tbKxstages>]_<layout>_align<n>.
    # Two-stage sm_75 instances omit the k-tile field; those were measured to use 32 (1264/1264).
    block_k = int(tiles[1].split("x")[0]) if len(tiles) >= 2 else _platform().ktile_default
    return {"family": "cutlass", "k_per_dot": k_per_dot, "block_k": block_k}


def _plan_from_launched(a, b, kind, out_dtype, nsplit, reduction):
    """Derive the reconstruction from what cuBLAS actually launched, with no byte-compare.

    Every knob is read rather than searched: the family and k-tile from the kernel name, the
    k-elements per dot from the MMA in that name, and the split-K merge scheme from
    `REDUCTION_SCHEME`. Returns None when the launch is something we do not model (a SIMT gemv
    fallback, an atomic reduction, an unparsable name), which sends the shape to the search."""
    info = _parse_launched(_launched_kernel_names(a, b, out_dtype))
    if info is None:
        return None
    M, K = a.shape
    if info["family"] == "nvjet":  # cuBLAS's own kernels: the aligned rule, grain from the name
        grain = info["block_k"] or _grain(kind)
        if nsplit <= 1:
            return ("plain", None)
        chunk = _chunk_from_ns(K, nsplit, grain)
        return ("split", chunk) if grain <= chunk < K else None
    if kind == "fp8":  # fp8 never reaches CUTLASS (cuBLAS needs every dim %16, i.e. aligned)
        return None
    k_per_dot, block_k = info["k_per_dot"], info["block_k"]
    if nsplit <= 1:
        return ("k_per_dot", (k_per_dot, K % block_k))
    cmode = dict(_platform().reduction_to_cmode).get(reduction)
    if cmode is None:  # e.g. an atomic (INPLACE) reduction, which is not deterministic for us
        return None
    # CUTLASS's split-K partition grain: `params_universal_base.h:52-59` takes kAlignK = 64 when
    # K divides evenly by it and falls back to 128 bits / 16 bits = 8 otherwise.
    big, small = max(_platform().splitk_grains), min(_platform().splitk_grains)
    grain = big if K % big == 0 else small
    chunk = _chunk_from_ns(K, nsplit, grain)
    return ("splitk_groups", (chunk, k_per_dot, block_k, cmode)) if grain <= chunk <= K else None


def _candidate_plans(K, nsplit, kind):
    """Every reconstruction worth trying for a shape, in try-order.

    The CUTLASS families are fp16/bf16 only. cuBLAS falls back to CUTLASS when a contiguous dim
    is not 8-element aligned, but it refuses fp8 altogether unless every dim is a multiple of
    16, so a runnable fp8 shape is always aligned and always stays on nvjet. (Their k-tile of 16
    is also below the K>=32 an fp8 `tl.dot` needs.)"""
    cutlass = kind != "fp8"
    if nsplit <= 1:
        # A plain GEMM if cuBLAS stayed on nvjet; otherwise it is on a CUTLASS kernel that
        # rounds its accumulator once per MMA, and neither the k per dot nor where the partial
        # group sits can be read off the shape.
        plans = [("plain", None)]
        if cutlass:
            plans += [("k_per_dot", c) for c in _k_per_dot_candidates(K)]
        return plans
    # The nvjet split-K rule first, then the CUTLASS split-K path, which differs in the
    # partition grain, has a per-slice residue tile, and has three possible merge schemes.
    plans = [("split", c) for c in _splitk_candidate_chunks(K, nsplit, kind)]
    if cutlass:
        plans += [("splitk_groups", c) for c in _splitk_group_candidates(K, nsplit)]
    return plans


def _plan_matches(M, N, K, kind, out_dtype, plan, seeds):
    """Byte-check `plan` against cuBLAS on `seeds`, one seed resident at a time.

    Holding every reference at once would need `len(seeds)` copies of A, B and D, which runs a
    large shape out of memory (a 8192x300000 fp16 operand is 4.9 GiB before the fp32 temporaries
    that build it), so each seed is generated, compared and freed."""
    for seed in seeds:
        a, b = _make_inputs(M, N, K, kind, seed)
        d = _cublas_direct(a, b, kind, out_dtype)[0]  # (out, nsplit, reduction)
        c = _reconstruct(a, b, out_dtype, plan)
        ok = c is not None and _bits_eq(c, d)
        del a, b, c, d
        if not ok:
            return False
    return True


def _calibrate(M, N, K, kind, out_dtype):
    """RUNTIME match: find the reconstruction that bit-matches cuBLAS run directly. Raises
    CublasUnsupportedShape if none does. cuBLAS's choice is a function of the shape, not of the
    data, so one calibration serves every future input of that shape.

    Three stages. A seed-0 prefilter kills nearly every candidate against a single reference.
    Survivors are checked on the rest of `_SEEDS`. The winner is then re-checked on
    `_CONFIRM_SEEDS`, which took no part in choosing it -- picking the best of ~36 candidates on
    32 seeds is a selection, and without a hold-out a candidate that fits by luck slips through
    (measured: 4 such shapes in 1,497 on the skinny+deep corner)."""
    a0, b0 = _make_inputs(M, N, K, kind, _SEEDS[0])
    d0, nsplit, _red = _cublas_direct(a0, b0, kind, out_dtype)
    survivors = []
    for plan in _candidate_plans(K, nsplit, kind):
        c = _reconstruct(a0, b0, out_dtype, plan)
        if c is not None and _bits_eq(c, d0):
            survivors.append(plan)
        del c
    del a0, b0, d0

    for plan in survivors:
        if _plan_matches(M, N, K, kind, out_dtype, plan, _SEEDS[1:] + _CONFIRM_SEEDS):
            return plan
    raise CublasUnsupportedShape(f"{M}x{N}x{K} {kind}: no reconstruction matches cuBLAS (nsplit={nsplit}, "
                                 f"{len(survivors)} candidates passed the first seed)")


def plan_origin(M, N, K, kind, out_dtype=torch.float16):
    """Which tier settled this shape: "static" | "pseudo-static" | "runtime" | "unsupported",
    or "?" if it has not been resolved yet."""
    return _PLAN.get((torch.cuda.get_device_capability(), _cublaslt_version(), M, N, K, kind, out_dtype),
                     ("?", None))[0]


def _resolve(a, b, kind, out_dtype, enable_runtime_match, enable_pseudo_static=True):
    """Reconstruction plan for this shape, cached after the first call. Three tiers, tried in
    order, each stricter about what it costs and weaker about what it proves:

      static        the heuristic alone decides it, no cuBLAS execution at all.
      pseudo-static one profiled cuBLAS launch; the recipe is READ off the launched kernel name
                    and `REDUCTION_SCHEME` rather than searched. Costs ~2 ms for the profile but
                    it is a direct observation of what cuBLAS did, so it cannot pick a wrong
                    recipe that happens to agree on the seeds it was checked against. It also
                    does not prove the result: nothing is byte-compared on this path.
      runtime       search the candidate recipes and keep the one that byte-matches cuBLAS on
                    32 selection seeds plus 8 hold-out seeds. Proves the match on those inputs,
                    but choosing the best of ~36 candidates is a selection and can still admit a
                    recipe that fits by luck.
    """
    M, K = a.shape
    N = b.shape[1]
    key = (torch.cuda.get_device_capability(), _cublaslt_version(), M, N, K, kind, out_dtype)
    if key in _PLAN:
        origin, plan = _PLAN[key]  # "static" | "pseudo-static" | "runtime" | "unsupported"
        if origin == "unsupported":
            raise CublasUnsupportedShape(f"{M}x{N}x{K} {kind}: unsupported (cached)")
        if origin == "runtime" and not enable_runtime_match:
            raise CublasNeedRuntimeMatch(f"{M}x{N}x{K} {kind}: needs a runtime match (cached)")
        return plan

    _, nsplit, reduction = _cublas_direct(a, b, kind, out_dtype, execute=False)  # heuristic only, no GEMM run
    if _is_static_exact(kind, nsplit, _aligned(M, N, K, kind)):
        plan = ("plain", None) if nsplit <= 1 else ("split", _static_chunk(K, nsplit, kind))
        _PLAN[key] = ("static", plan)
        return plan

    if enable_pseudo_static:
        plan = _plan_from_launched(a, b, kind, out_dtype, nsplit, reduction)
        if plan is not None:
            _PLAN[key] = ("pseudo-static", plan)
            return plan

    if not enable_runtime_match:
        raise CublasNeedRuntimeMatch(f"{M}x{N}x{K} {kind}: not statically guaranteed; "
                                     f"re-call with enable_runtime_match=True")  # not cached: unknown until runtime
    try:
        plan = _calibrate(M, N, K, kind, out_dtype)
    except CublasUnsupportedShape:
        _PLAN[key] = ("unsupported", None)  # even a runtime match found nothing; cache the negative
        raise
    _PLAN[key] = ("runtime", plan)
    return plan


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def cublas_equivalent_gemm(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype | None = None,
                           enable_runtime_match: bool = False, enable_pseudo_static: bool = True,
                           cublaslt: str | None = None) -> torch.Tensor:
    """fp16/bf16 GEMM, bit-identical to cuBLAS. `a` is [M,K], `b` is [K,N].

    Static by default: the reconstruction is planned from the cuBLAS heuristic alone (no GEMM
    run). Aligned fp16/bf16 is static-exact. A shape that is not statically guaranteed
    (non-aligned) raises CublasNeedRuntimeMatch -- unless `enable_runtime_match=True`, which
    byte-compares candidates against cuBLAS and returns the match or raises
    CublasUnsupportedShape.

    `cublaslt` picks which cuBLAS to match for this call -- a version prefix ("12.8") or a
    path -- overriding `set_cublaslt`. Default is the process-wide choice.
    """
    kind = _kind_of(a)
    if kind == "fp8":
        raise ValueError("use cublas_equivalent_scaled_mm for fp8")
    out_dtype = out_dtype or a.dtype
    with _using_cublaslt(cublaslt):
        plan = _resolve(a, b, kind, out_dtype, enable_runtime_match, enable_pseudo_static)
        return _reconstruct(a, b, out_dtype, plan)


def cublas_equivalent_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: float = 1.0, scale_b: float = 1.0,
                                out_dtype: torch.dtype = torch.float16, enable_runtime_match: bool = False,
                                enable_pseudo_static: bool = True,
                                cublaslt: str | None = None) -> torch.Tensor:
    """fp8 (e4m3) GEMM, bit-identical to cuBLAS. `a` is [M,K], `b` is [K,N] column-major
    (i.e. from `w.t()`); scales are scalars.

    fp8 plain is static-exact. fp8 split-K is NOT statically guaranteed (the vertical/cluster
    kernel is not bit-reproducible) -> raises CublasNeedRuntimeMatch unless
    `enable_runtime_match=True`, which byte-compares against cuBLAS and returns the match or
    raises CublasUnsupportedShape (vertical split-K).

    `cublaslt` picks which cuBLAS to match for this call -- a version prefix ("12.8") or a
    path -- overriding `set_cublaslt`. Default is the process-wide choice.
    """
    with _using_cublaslt(cublaslt):
        plan = _resolve(a, b, "fp8", out_dtype, enable_runtime_match, enable_pseudo_static)
        return _reconstruct(a, b, out_dtype, plan, sa=scale_a, sb=scale_b)


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def _check_one(kind, M, N, K):
    """Verify one shape and report which tier resolved it, read from the plan cache rather than
    guessed from which call succeeded. Returns (tier, bit_ok)."""
    a, b = _make_inputs(M, N, K, kind, 7)
    ref = cublas_matmul(a, b, torch.float16 if kind == "fp8" else None)
    call = (lambda rt: cublas_equivalent_scaled_mm(a, b, 1.0, 1.0, torch.float16, enable_runtime_match=rt)) \
        if kind == "fp8" else (lambda rt: cublas_equivalent_gemm(a, b, enable_runtime_match=rt))
    try:
        try:
            out = call(False)
        except CublasNeedRuntimeMatch:
            out = call(True)
    except CublasUnsupportedShape:
        return "unsupported", None
    tier = plan_origin(M, N, K, kind)
    return tier, _bits_eq(out, ref)


def verify():
    if not torch.cuda.is_available():
        print("no CUDA GPU; verify() needs one.")
        return
    print(f"device: {torch.cuda.get_device_name()} | cap {torch.cuda.get_device_capability()} | "
          f"cuda {torch.version.cuda}")
    print("tier = static (heuristic only) / pseudo-static (kernel name) / runtime (byte-compare)\n")

    shapes = [("fp16", 4096, 4096, 4096), ("fp16", 2048, 2048, 512), ("fp16", 16, 16384, 16384),
              ("fp16", 64, 64, 32768), ("fp16", 128, 128, 16384), ("fp8", 4096, 4096, 4096),
              ("fp8", 8192, 8192, 8192), ("fp8", 64, 64, 65536), ("fp8", 128, 128, 65536), ("fp8", 16, 64, 131072)]
    n_ok, n_unsup = 0, 0
    tiers = {"static": 0, "pseudo-static": 0, "runtime": 0}
    for (kind, M, N, K) in shapes:
        cls, ok = _check_one(kind, M, N, K)
        if cls == "unsupported":
            n_unsup += 1
            verdict = "UNSUPPORTED"
        else:
            tiers[cls] = tiers.get(cls, 0) + 1
            n_ok += int(bool(ok))
            verdict = "BIT-IDENTICAL" if ok else "DIFFER"
        print(f"  {kind:4s} {M:5d}x{N:5d}x{K:6d}  {cls:13s}  {verdict}")

    print(f"\nbit-identical {n_ok}/{sum(tiers.values())} | static {tiers['static']} | "
          f"pseudo-static {tiers['pseudo-static']} | runtime {tiers['runtime']} | unsupported {n_unsup}")


if __name__ == "__main__":
    verify()
