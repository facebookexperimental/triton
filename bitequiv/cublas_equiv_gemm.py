"""Bit-for-bit cuBLAS-equivalent Triton GEMM, driven by the cuBLASLt heuristic.

Top-level API (a cuBLAS-like matmul):

    from bitequiv.cublas_equiv_gemm import cublas_equivalent_gemm
    c = cublas_equivalent_gemm(a_fp16, b_fp16)                            # == cuBLAS fp16/bf16, bit-for-bit
    c = cublas_equivalent_gemm(a_fp8, b_fp8_col, scale_a=sa, scale_b=sb)  # == cuBLAS fp8, bit-for-bit

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
  4. HOW THE RECIPE IS DECIDED -- from the heuristic config alone (see `_static_plan`):

       `ALGO_ID`          -> which kernel family cuBLAS will run (nvjet or CUTLASS)
       `STAGES_ID`        -> the threadblock k-tile and the k per dot
       `REDUCTION_SCHEME` -> which of the three split-K merge schemes
       `SPLITK_NUM`       -> the partition

     Nothing is executed to plan a shape and no bytes are compared.  A config outside
     the measured tables declines rather than guessing.

Known unsupported (raise `CublasUnsupportedShape`), all of them SIMT -- CUDA cores, no tensor
core -- and none of them CUTLASS:

  * `ALGO_ID` 11, 13, 14 -- cuBLAS's `gemv` fallbacks for very thin M or N (`gemv2T_kernel`,
    `dot_kernel` + `reduce_1Block_kernel`). They reduce over K with a warp shuffle whose tree
    the heuristic config does not describe, and measuring confirms it: over 120 shapes of these
    three, a third have NO plan in this file that matches, and no recipe is common to the rest.
  * `ALGO_ID` 16 -- `magma_sgemmEx_kernel`, which cuBLAS picks only for very shallow K (an
    observed example: 14x1929x25, `SPLITK_NUM` 1, `STAGES_ID` 0). It is NOT a gemv: it is a full
    blocked GEMM on CUDA cores, with its own tile and k-loop, so it is a separate problem from
    the vector kernels above and would need its own measurement. How often it turns up depends
    entirely on the shape distribution: 0 hits in 600,000 draws of M, N, K uniform up to 32768,
    but 17,640 in 200,000 draws with every dim under 64. Every hit had K between 2 and 45,
    `SPLITK_NUM` 1 and `STAGES_ID` 0.

They stay out of `algo_family` and decline.  fp8 needs `BM >= 64` (see `_tile`) or it silently
uses a different MMA.

Known residual, and a follow-up rather than a property of the approach: over 100,692
shapes on sm_100 the derivation covers 99.91% and 99.891% of those are byte-identical to
cuBLAS.  The 110 that are not are all nvjet split-K at very deep K, and no partition at
all reproduces cuBLAS there -- a brute-force sweep of every chunk found nothing, and
neither the config nor the launched kernel name separates them from the 25,808 nvjet
split-K shapes that do match.  The cause is still being
inspected; until it is settled these shapes return a mismatching result instead of
declining.

Second known residual, bf16: bf16 is supported in code but the CUTLASS split-K merge is wrong
for it.  `_splitk_combine_modes` rounds to fp16 for CMODE 2 and CMODE 3 whatever the output
dtype, while cuBLAS rounds to the output dtype, i.e. to bf16.  Measured on sm_100 over 563
CUTLASS split-K shapes spanning 11 stages ids: bf16 at `REDUCTION_SCHEME` 1 or 4 (the two
schemes that map to those CMODEs) is byte-identical 0/184, bf16 at scheme 2 (fp32 workspace, no
intermediate rounding) is 95/95, and fp16 is 284/284 on every scheme.  A 300 s bf16 fuzz over
638 random shapes lands at 27.4% bit-consistent (700/2552 input compares), and all 463 distinct
failing shapes are that one class -- nothing else about bf16 fails.  So bf16 answers wrong
rather than declining on that path.  The fix is to round to the output dtype in the merge
kernel; it has not been made, so bf16 must be treated as unvalidated for split-K.

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
import struct
import threading
import warnings

import torch
import triton
import triton.language as tl

DEVICE = "cuda"
# How much workspace cuBLAS is allowed. This is an INPUT to its algorithm choice, not an
# implementation detail: the same shape at 0, at 1 MiB and at 32 MiB can come back with a
# different SPLITK_NUM and so a different accumulation order. It therefore belongs to the
# equivalence claim exactly like the library version does, and is selectable the same way --
# `workspace_bytes=` for one call, `set_workspace_bytes()` for the process.
#
# The default is a fixed 32 MiB and is deliberately NOT read from torch. What torch would have
# passed is the caller's business; the reference here is cuBLAS itself.
_WS_BYTES_DEFAULT = 33554432  # 32 MiB


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

    # -- what the heuristic config means on this architecture -----------------------------
    # Every table below is MEASURED. An unknown key declines rather than guessing, so a new
    # cuBLAS or a new GPU loses coverage instead of returning wrong bits.
    #
    # CUBLASLT_ALGO_CONFIG_ID -> which kernel family cuBLAS will launch. Measure: profile the
    # launched kernel name over a shape sweep and cross-tabulate against attr 0.
    algo_family: tuple = ()          # sm_100: see `_SM100` below
    # (family, CUBLASLT_ALGO_CONFIG_STAGES_ID) -> (threadblock k-tile, k per dot). This is the
    # pair the reconstruction turns on, and the stages id is the only field that pins it.
    # Measure: same sweep, read `<tbK>x<stages>` and the `s<MMA>gemm` token from the name.
    stages_recipe: tuple = ()
    # Partition grains to try for CUTLASS split-K, largest first; cross-check against `kAlignK`
    # in `cutlass/.../params_universal_base.h`.
    splitk_grains: tuple = ()        # sm_100: (8, 64)
    # CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME -> our merge scheme. Measure: cross-tabulate attr 3
    # against the merge scheme that byte-matches.
    reduction_to_cmode: tuple = ()

    # -- Triton-side constraint -----------------------------------------------------------
    # Minimum BM below which Triton stops using the native fp8 tensor-core path. Measure: dump
    # PTX over a (BM, BN, BK, num_warps) grid and look for the MMA instruction changing.
    fp8_min_bm: int = 64


_SM100 = ArchProfile(
    name="sm_100 (NVIDIA GB200 / B200)",
    measured=True,
    cublaslt_versions=((13, 1), (12, 8)),
    measured_cublaslt="13.1.1 and 12.8.5",
    # 12 and 21 were added after the first sweep, both measured the same way as 23/24/66:
    #   12 `gemmk1_kernel` -- in 3,125 heuristic hits it appeared ONLY with K == 1. With K == 1
    #      every output is a single product, so there is no accumulation order to get wrong and
    #      any plan matches (all 7 plan forms were byte-identical on 40 shapes). The entry is
    #      therefore safe but UNTESTED for real accumulation: if cuBLAS ever picks 12 for K > 1,
    #      the recipe it inherits from `("cutlass", 0)` is a guess, not a measurement.
    #   21 `cutlass::Kernel2` -- behaves as CUTLASS. Its STAGES_ID 18 and 20 re-derive the k-tiles
    #      already in `stages_recipe` (64 and 128) on shapes where the k-tile changes the grouping:
    #      16/16 and 13/13 byte-identical over 8 seeds each, and the other two k-tiles lose. So
    #      adding 21 keeps the (family, STAGES_ID) key pure. Only its STAGES_ID 25 is new.
    algo_family=((12, "cutlass"), (21, "cutlass"), (23, "cutlass"), (24, "cutlass"), (66, "nvjet")),
    # The CUTLASS side of this table is now the COMPLETE set of stages ids the sm_100 algos
    # advertise. `cublasLtMatmulAlgoCapGetAttribute(CUBLASLT_ALGO_CAP_STAGES_IDS)` over every
    # algo id `cublasLtMatmulAlgoGetIds` returns gives, for the cutlass families, exactly
    # {0} + {7..20} + {25} -- 16 keys, all listed below -- so a CUTLASS shape can no longer
    # decline on an unmeasured STAGES_ID.
    #
    # The k-tile comes from the `cublasLtMatmulStages_t` enum NAME: the names read `<ktile>x<n>`,
    # so `ktile = 16 << ((id - 1) // 6)` for ids 1..24, 25 is `32x10`, and 32..37 are
    # `8xAUTO`..`256xAUTO`. That rule reproduces all 13 non-zero keys that were measured one at a
    # time, with 0 disagreements, which is why the four keys added from it are trusted.
    #
    # `k per dot` is 16 for EVERY named stage: those kernels run `s16816` / `s161616`. STAGES_ID
    # 0 is `UNDEFINED` -- it names no k-tile, so the rule says nothing about it -- and stays a
    # measured exception at (32, 8), because the kernel behind it is `s1688`.
    #
    # HOW FAR THE FOUR RULE-DERIVED KEYS (7, 8, 11, 17) ARE ACTUALLY EXERCISED. 58,414,630
    # heuristic queries on sm_100, of which one 23,839,756-shape sweep counted per key:
    #   7  picked constantly (2.06 M hits in that sweep), and it DOES take split-K: 183,776 of
    #      those hits have SPLITK_NUM > 1. Byte-verified: 24/24 single-pass and 296/296 split-K at
    #      REDUCTION_SCHEME 2. Its split-K at REDUCTION_SCHEME 4 is bf16-only and does NOT match
    #      (0/200) -- that is the pre-existing bf16 hole in `_splitk_combine_modes`, which rounds
    #      to fp16 for CMODE 2 and 3 whatever the output dtype. A control on the stages ids that
    #      were already in this table shows the same 0% on bf16 + scheme 4 or 1 and 100% on
    #      bf16 + scheme 2, with fp16 clean on every scheme, so it is not something key 7 brings.
    #   8  picked rarely (724 hits in that sweep) and NEVER with SPLITK_NUM > 1, so its split-K
    #      path is unverified. Byte-verified 13/13 single-pass.
    #   11 and 17 were never picked at all -- 0 hits in any of the 58 M queries, though algos 21
    #      and 23 do advertise them. Both entries rest on the enum-name rule alone: UNVERIFIED.
    stages_recipe=(
        (("cutlass", 0), (32, 8)),
        (("cutlass", 7), (32, 16)),     # 32x1
        (("cutlass", 8), (32, 16)),     # 32x2
        (("cutlass", 9), (32, 16)),     # 32x3
        (("cutlass", 10), (32, 16)),    # 32x4
        (("cutlass", 11), (32, 16)),    # 32x5
        (("cutlass", 12), (32, 16)),    # 32x6
        (("cutlass", 13), (64, 16)),    # 64x1
        (("cutlass", 14), (64, 16)),    # 64x2
        (("cutlass", 15), (64, 16)),    # 64x3
        (("cutlass", 16), (64, 16)),    # 64x4
        (("cutlass", 17), (64, 16)),    # 64x5
        (("cutlass", 18), (64, 16)),    # 64x6
        (("cutlass", 19), (128, 16)),   # 128x1
        (("cutlass", 20), (128, 16)),   # 128x2
        # ALGO_ID 21 (`cutlass::Kernel2`) only; no other ALGO_ID was ever seen with this STAGES_ID.
        # k per dot: 16 byte-matches 15/15 shapes, 8 only 9/15. k-tile: measured on 34 shapes
        # picked so that 32, 64 and 128 really do group the k-loop differently -- 32 byte-matches
        # 34/34, 64 21/34, 128 15/34, over 8 seeds each.
        (("cutlass", 25), (32, 16)),
        (("nvjet", 35), (64, None)),      # fp16/bf16; the k-tile doubles as the split-K grain
        (("nvjet", 36), (128, None)),     # fp8
    ),
    splitk_grains=(8, 64),
    # 1 is CUBLASLT_REDUCTION_SCHEME_INPLACE. It was once declined as "atomic, so not
    # deterministic for us", but every shape carrying it byte-matches the serial fp16 chain.
    reduction_to_cmode=((0, 3), (1, 3), (2, 0), (4, 2)),
    fp8_min_bm=64,
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
        # practice most shapes are unaffected, and a version that renumbered ALGO_ID or
        # STAGES_ID would fall out of the measured tables and decline rather than answer wrong.
        # Warn once per architecture: the cache below is filled exactly once, so a fuzzer loop
        # stays quiet.
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
def _plain_gemm(A, B, C, M, N, K, am, ak, bk, bn, cm, cn, s, BM: tl.constexpr, BN: tl.constexpr,
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
    acc = acc * s
    ocm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    ocn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    tl.store(C + cm * ocm[:, None] + cn * ocn[None, :], acc.to(C.dtype.element_ty),
             mask=(ocm[:, None] < M) & (ocn[None, :] < N))


@triton.jit
def _plain_gemm_k_per_dot(A, B, C, M, N, K, am, ak, bk, bn, cm, cn, s, KPD, RES, BM: tl.constexpr,
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


def _triton_splitk_groups(a, b, chunk, k_per_dot, ktile, cmode, out_dtype, scale=1.0, BK=16):
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
                                     nsplit, scale, CMODE=cmode, BM=BM, BN=BN, num_warps=4)
    return c


def _triton_splitk(a, b, chunk, out_dtype, scale=1.0, BK=64):
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
    _splitk_combine[(ntile, )](w, c, M, N, w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1), nsplit,
                               scale, BM=BM, BN=BN, num_warps=4)
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
_CFG_ID, _CFG_SPLITK, _CFG_REDUCTION, _CFG_STAGES = 0, 2, 3, 6
_CFG_INNER_SHAPE, _CFG_CLUSTER_SHAPE = 7, 8
_CFG_COUNT = 9  # algo-config attributes 0..8
# The two attributes cublasLt.h declares uint16; every other one is 32 bits. cuBLASLt checks the
# buffer size exactly, so reading these with 4 bytes returns INVALID_VALUE, i.e. silently None.
_CFG_U16 = (_CFG_INNER_SHAPE, _CFG_CLUSTER_SHAPE)
_LT_SPEC = None   # process-wide choice from set_cublaslt(); None = newest installed
_TLS = threading.local()   # per-CALL overrides live here, never in a global -- see _using_cublaslt
_UNSET = object()
_LT_LIBS: dict = {}   # resolved path -> loaded library, so switching back and forth is free
_LT_PATHS: dict = {}  # spec -> resolved path
_SCALE_A = None  # device fp32 [1] holding the A scale cuBLAS is given (fp8 only)
_SCALE_B = None  # same for B
_WS_SPEC = None   # workspace override set by set_workspace_bytes(); None = the default
_WS_BUFS: dict = {}   # size in bytes -> device scratch, so switching sizes is free


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


def set_workspace_bytes(nbytes=None):
    """Choose how much workspace cuBLAS may use, for the rest of the process.

    `None` restores the default. cuBLAS reads this when it picks an algorithm, so two callers
    that allow different amounts can get different `SPLITK_NUM` for the same shape and so
    different bits -- which is why it is selectable rather than fixed. Process-wide, as the name
    says; a single call overrides it with `workspace_bytes=`, and that override is thread-local."""
    global _WS_SPEC
    _WS_SPEC = nbytes


def _workspace_bytes():
    """The allowance in force for the call on THIS thread."""
    v = getattr(_TLS, "ws_bytes", _UNSET)
    if v is not _UNSET:
        return v
    return _WS_BYTES_DEFAULT if _WS_SPEC is None else _WS_SPEC


def _ws_buffer(nbytes):
    """Device scratch of exactly `nbytes`, cached, so alternating between two sizes is free.

    Shared between threads. Only the cuBLAS-direct reference writes it -- `cublas_equivalent_gemm`
    runs Triton and never touches it -- so two threads would have to be calling `cublas_matmul`
    on different streams at the same moment to collide."""
    if nbytes not in _WS_BUFS:
        _WS_BUFS[nbytes] = torch.empty(max(nbytes, 1), dtype=torch.uint8, device=DEVICE)
    return _WS_BUFS[nbytes]


@contextlib.contextmanager
def _using_workspace(nbytes):
    """Select an allowance for one call. Thread-local for the same reason as `_using_cublaslt`:
    two threads can be matching two different allowances at once, and a global would let one of
    them retarget the other. The plan cache keys on it, so a plan made under one allowance is
    never reused under another."""
    if nbytes is None:
        yield
        return
    prev = getattr(_TLS, "ws_bytes", _UNSET)
    _TLS.ws_bytes = nbytes
    try:
        yield
    finally:
        if prev is _UNSET:
            del _TLS.ws_bytes
        else:
            _TLS.ws_bytes = prev


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
    global _SCALE_A, _SCALE_B
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
    if _SCALE_A is None:
        _SCALE_A = torch.ones(1, dtype=torch.float32, device=DEVICE)
        _SCALE_B = torch.ones(1, dtype=torch.float32, device=DEVICE)
    _LT_LIBS[path] = lib
    return lib


def _ck(nm, rc):
    if rc != 0:
        raise RuntimeError(f"cublasLt {nm} failed (status {rc})")


def _cfg(algo_ptr, attr):
    """One algo-config attribute, read at the width `cublasLtMatmulAlgoConfigAttributes_t`
    declares for it. Nothing in the reconstruction reads attr 7 or 8 -- they are collected so
    the config tuple is complete, and because reading them at the wrong width used to hide
    them. `INNER_SHAPE_ID` (7) is `cublasLtMatmulInnerShape_t`, i.e. exactly the MMA shape, so
    it looks like it should hand us `k per dot` directly; it does not. It was read 0
    (`UNDEFINED`) on every one of 2,448,266 sm_100 shapes, fp16, bf16 and fp8, split-K and not,
    across every ALGO_ID the heuristic returned -- cuBLAS never populates it. Do not build on
    it. `CLUSTER_SHAPE_ID` (8) IS populated (AUTO plus 1x1x1, 2x1x1, 4x1x1, 1x2x1, 2x2x1, 4x2x1,
    1x4x1, 2x4x1 were all seen); nothing here reads it either, because the cluster shape does
    not change the k-loop grouping."""
    lib = _load_lt()
    o, n = (ctypes.c_uint16(0), 2) if attr in _CFG_U16 else (ctypes.c_int(-1), 4)
    w = ctypes.c_size_t(0)
    r = lib.cublasLtMatmulAlgoConfigGetAttribute(algo_ptr, attr, ctypes.byref(o), n, ctypes.byref(w))
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


def _make_layouts(lib, desc, M, N, K, kind, out_dtype, sa=1.0, sb=1.0):
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
        _SCALE_A.fill_(sa)
        _SCALE_B.fill_(sb)
        for attr, t in ((_DESC_A_SCALE_PTR, _SCALE_A), (_DESC_B_SCALE_PTR, _SCALE_B)):
            sp = ctypes.c_void_p(t.data_ptr())
            _ck("scale", lib.cublasLtMatmulDescSetAttribute(desc, attr, ctypes.byref(sp), 8))
    _ck("LC", lib.cublasLtMatrixLayoutCreate(ctypes.byref(C), cd, M, N, N))
    _set_row(lib, C)
    _ck("LD", lib.cublasLtMatrixLayoutCreate(ctypes.byref(D), cd, M, N, N))
    _set_row(lib, D)
    return A, B, C, D


def _cublas_direct(a, b, kind, out_dtype, execute=True, sa=1.0, sb=1.0):
    """Query cuBLAS's top-heuristic algo and (if execute) run it DIRECTLY via ctypes
    cublasLtMatmul. Returns (D, config): `config` is the algo-config of the top heuristic
    result, which is everything the reconstruction is derived from. With execute, D is cuBLAS's
    exact output; without, D is None and no GEMM runs at all. No torch."""
    lib = _load_lt()
    M, K = a.shape
    N = b.shape[1]
    handle, desc, pref = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    A = B = C = D = None
    try:
        _ck("create", lib.cublasLtCreate(ctypes.byref(handle)))
        _ck("desc", lib.cublasLtMatmulDescCreate(ctypes.byref(desc), _CUBLAS_COMPUTE_32F, _CUDA_R_32F))
        A, B, C, D = _make_layouts(lib, desc, M, N, K, kind, out_dtype, sa, sb)
        _ck("pref", lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)))
        ws = ctypes.c_size_t(_workspace_bytes())
        _ck("pref-set", lib.cublasLtMatmulPreferenceSetAttribute(pref, _PREF_MAX_WS, ctypes.byref(ws), 8))
        results = (_HeurResult * 16)()
        ret = ctypes.c_int(0)
        rc = lib.cublasLtMatmulAlgoGetHeuristic(handle, desc, A, B, C, D, pref, 16,
                                                ctypes.cast(results, ctypes.c_void_p), ctypes.byref(ret))
        if rc != 0 or ret.value == 0:
            raise CublasUnsupportedShape(f"no cuBLAS algo for {M}x{N}x{K} {kind} (rc={rc}, ret={ret.value})")
        algo_ptr = ctypes.cast(ctypes.byref(results[0]), ctypes.c_void_p)
        config = tuple(_cfg(algo_ptr, i) for i in range(_CFG_COUNT))
        if not execute:
            return None, config
        out = torch.empty(M, N, device=DEVICE, dtype=out_dtype)
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = lib.cublasLtMatmul(handle, desc, ctypes.cast(ctypes.pointer(alpha), ctypes.c_void_p),
                                ctypes.c_void_p(a.data_ptr()), A, ctypes.c_void_p(b.data_ptr()), B,
                                ctypes.cast(ctypes.pointer(beta), ctypes.c_void_p), ctypes.c_void_p(out.data_ptr()), C,
                                ctypes.c_void_p(out.data_ptr()), D, algo_ptr,
                                ctypes.c_void_p(_ws_buffer(_workspace_bytes()).data_ptr()),
                                ctypes.c_size_t(_workspace_bytes()), stream)
        _ck("matmul", rc)
        torch.cuda.synchronize()
        return out, config
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


def cublas_matmul(a, b, out_dtype=None, scale_a=None, scale_b=None, cublaslt=None,
                  workspace_bytes=None):
    """cuBLAS's OWN output (run directly via cublasLtMatmul), the reference our
    reconstruction must match. fp16/bf16: `b` is [K,N]. fp8: `b` is [K,N] column-major.
    `cublaslt` and `workspace_bytes` pick which cuBLAS and which allowance, as in
    `cublas_equivalent_gemm`."""
    kind = _kind_of(a)
    out_dtype = out_dtype or (torch.float16 if kind == "fp8" else a.dtype)
    with _using_cublaslt(cublaslt), _using_workspace(workspace_bytes):
        return _cublas_direct(a, b, kind, out_dtype, sa=scale_a if scale_a is not None else 1.0,
                              sb=scale_b if scale_b is not None else 1.0)[0]


# --------------------------------------------------------------------------- #
# Reconstruction planning (verify-then-use) + per-shape cache
# --------------------------------------------------------------------------- #
# (capability,cublaslt,M,N,K,kind,out_dtype) -> (origin, plan). The capability and the cuBLASLt
# version keep a process that switched either one from reusing the old recipe. origin is
# "static" or "unsupported".
_PLAN: dict[tuple, tuple] = {}


# The only operand dtypes this file has been measured on. `_kind_of` maps everything else to
# "fp16", so without an explicit check an fp32 or e5m2 operand would run the fp16 recipe and
# return a wrong answer silently instead of declining. `_check_operands` refuses them.
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)


def _kind_of(a) -> str:
    if a.dtype == torch.float8_e4m3fn:
        return "fp8"
    if a.dtype == torch.bfloat16:
        return "bf16"
    return "fp16"


def _chunk_from_ns(K, ns, G):
    total = (K + G - 1) // G
    return ((total + ns - 1) // ns) * G


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


def _f32(x):
    """Round a Python float to fp32, the width cuBLAS keeps its scales in."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _fold_scales(scale_a, scale_b):
    """The single fp32 factor cuBLAS applies to the accumulator for an fp8 GEMM.

    cuBLAS multiplies the two operand scales together in fp32 and does ONE multiply on the fp32
    accumulator, before the cast to the output dtype. Applying them as two multiplies rounds
    twice and differs in the last bits whenever neither is a power of two: measured over 10
    scale pairs on 3 shapes, two multiplies matched cuBLAS 15/30 and this folding 30/30."""
    return _f32(_f32(1.0 if scale_a is None else scale_a) * _f32(1.0 if scale_b is None else scale_b))


def _static_plan(K, kind, config):
    """The reconstruction, derived from the cuBLAS heuristic config alone.

    Nothing is executed and nothing is byte-compared: `ALGO_ID` gives the kernel family,
    `STAGES_ID` gives the threadblock k-tile and the k per dot, `REDUCTION_SCHEME` gives the
    split-K merge scheme, and `SPLITK_NUM` gives the partition. Returns (plan, reason); plan is
    None when the config falls outside what has been measured on this platform, which is a
    decline, not a guess.

    Measured over 100,692 shapes on sm_100 across six corners (fp16 random / aligned /
    non-aligned / skinny+deep, fp8 random / skinny+deep): a plan is derived for 99.91% of them
    and 99.891% of those are byte-identical to cuBLAS. The residual is 110 shapes, all nvjet
    split-K with very deep K, where the derived partition is wrong AND no partition at all
    reproduces cuBLAS -- a brute-force sweep of every chunk found nothing. Nothing in the
    config, and nothing in the launched kernel name either, separates them from the 25,808
    nvjet split-K shapes that do match, so they are a known residual rather than a case this
    function could decline. See the follow-up note in the module docstring."""
    prof = _platform()
    if config is None:
        return None, "no-heuristic"
    algo, nsplit, reduction, stages = config[_CFG_ID], config[_CFG_SPLITK], config[_CFG_REDUCTION], config[_CFG_STAGES]
    nsplit = nsplit if isinstance(nsplit, int) and nsplit >= 1 else 1
    family = dict(prof.algo_family).get(algo)
    if family is None:
        return None, f"unsupported ALGO_ID {algo}"
    recipe = dict(prof.stages_recipe).get((family, stages))
    if recipe is None:
        return None, f"unsupported STAGES_ID {stages} for {family}"
    block_k, k_per_dot = recipe

    if family == "nvjet":  # cuBLAS's own kernels: one fp32 accumulator, k-tile-grained split-K
        if nsplit <= 1:
            return ("plain", None), "ok"
        chunk = _chunk_from_ns(K, nsplit, block_k)
        return (("split", chunk), "ok") if block_k <= chunk < K else (None, "chunk out of range")
    if kind == "fp8":  # cuBLAS refuses fp8 unless every dim is a multiple of 16, so it never
        return None, "fp8 on a CUTLASS kernel"  # leaves nvjet; if it did, this is unmeasured
    if nsplit <= 1:
        return ("k_per_dot", (k_per_dot, K % block_k)), "ok"
    cmode = dict(prof.reduction_to_cmode).get(reduction)
    if cmode is None:
        return None, f"unsupported REDUCTION_SCHEME {reduction}"
    # CUTLASS's split-K partition grain: `params_universal_base.h:52-59` takes kAlignK = 64 when
    # K divides evenly by it and falls back to 128 bits / 16 bits = 8 otherwise.
    big, small = max(prof.splitk_grains), min(prof.splitk_grains)
    grain = big if K % big == 0 else small
    chunk = _chunk_from_ns(K, nsplit, grain)
    return ((("splitk_groups", (chunk, k_per_dot, block_k, cmode)), "ok")
            if grain <= chunk <= K else (None, "chunk out of range"))


def _reconstruct(a, b, out_dtype, plan, scale=1.0):
    mode, arg = plan
    if mode == "plain":
        return _triton_plain(a, b, out_dtype, scale=scale)
    if mode == "k_per_dot":
        return _triton_plain_k_per_dot(a, b, out_dtype, arg[0], arg[1], scale=scale)
    if mode == "splitk_groups":
        return _triton_splitk_groups(a, b, arg[0], arg[1], arg[2], arg[3], out_dtype, scale=scale)
    return _triton_splitk(a, b, arg, out_dtype, scale=scale)


def plan_origin(M, N, K, kind, out_dtype=torch.float16):
    """How this shape was settled: "static", "unsupported", or "?" if not resolved yet."""
    return _PLAN.get(_plan_key(M, N, K, kind, out_dtype), ("?", None))[0]


def _plan_key(M, N, K, kind, out_dtype):
    """What a plan is valid for. The workspace allowance is in here because cuBLAS reads it when
    it picks an algorithm, so the same shape under two allowances can need two plans.

    The cuBLASLt version is the full (major, minor, patch), deliberately finer than the
    (major, minor) the platform gate accepts. cuBLAS may change which algorithm it returns in a
    patch release, so reusing a plan across two patches would be reusing a claim nobody checked.
    The finer key costs one extra heuristic query per shape after a patch bump, and only in a
    process that loads both -- measured at 0.05 ms, since the search tiers this diff deletes are
    what used to make a cache miss expensive."""
    return (torch.cuda.get_device_capability(), _cublaslt_version(), _workspace_bytes(),
            M, N, K, kind, out_dtype)


def _resolve(a, b, kind, out_dtype):
    """The reconstruction plan for this shape, derived from the heuristic and cached.

    cuBLAS's choice is a function of the shape, not of the data, so the plan is decided once
    and every future input of that shape reuses it."""
    M, K = a.shape
    N = b.shape[1]
    key = _plan_key(M, N, K, kind, out_dtype)
    if key in _PLAN:
        origin, plan = _PLAN[key]
        if origin == "unsupported":
            raise CublasUnsupportedShape(f"{M}x{N}x{K} {kind}: unsupported (cached)")
        return plan

    config = _cublas_direct(a, b, kind, out_dtype, execute=False)[1]  # heuristic only, no GEMM run
    plan, reason = _static_plan(K, kind, config)
    if plan is None:
        _PLAN[key] = ("unsupported", None)
        raise CublasUnsupportedShape(f"{M}x{N}x{K} {kind}: {reason}")
    _PLAN[key] = ("static", plan)
    return plan

def _canonical_layout(t, want):
    """Is `t` laid out the way `_make_layouts` tells cuBLAS it is? A size-1 dim has no say."""
    r, c = t.shape
    s0, s1 = t.stride()
    e0, e1 = (c, 1) if want == "row-major" else (1, r)
    return (r == 1 or s0 == e0) and (c == 1 or s1 == e1)


def _check_operands(a, b, kind):
    """cuBLAS is told one fixed layout per dtype (see `_make_layouts`), not the tensor's actual
    strides. An operand laid out any other way therefore makes the reference read the same bytes
    as a different matrix -- and cuBLAS does not complain, it returns a different product, while
    the Triton side reads the real strides and returns the right one. That looks like "Triton
    does not match cuBLAS" when it is really a wrong call. Refuse it up front instead."""
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"expected 2-D operands, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"shapes do not match: a is {tuple(a.shape)}, b is {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise ValueError(f"operands must share a dtype, got {a.dtype} and {b.dtype}")
    if a.dtype not in _SUPPORTED_DTYPES:
        ok = ", ".join(str(d) for d in _SUPPORTED_DTYPES)
        raise ValueError(f"unsupported operand dtype {a.dtype}; supported: {ok}. Nothing in this "
                         f"file has been measured for it, and the recipe it would fall through to "
                         f"is the fp16 one, which returns a result that is not cuBLAS's.")
    want_b = "column-major" if kind == "fp8" else "row-major"
    for name, t, want in (("a", a, "row-major"), ("b", b, want_b)):
        if _canonical_layout(t, want):
            continue
        hint = " -- pass `w.t()` for a [N,K] weight" if (name == "b" and kind == "fp8") else ""
        raise ValueError(f"{kind} needs {name} {list(t.shape)} {want}, got strides {t.stride()}{hint}. "
                         f"cuBLAS is told this layout, so any other one makes the reference compute a "
                         f"different product and the comparison meaningless.")


def cublas_equivalent_gemm(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype | None = None,
                           scale_a: float | None = None, scale_b: float | None = None,
                           cublaslt: str | None = None,
                           workspace_bytes: int | None = None) -> torch.Tensor:
    """A GEMM bit-identical to cuBLAS, for fp16, bf16 and fp8 (e4m3). `a` is [M,K] row-major.

    `b` is [K,N], row-major for fp16/bf16 and column-major for fp8 (i.e. `w.t()` of an [N,K]
    weight) -- that is cuBLAS's own requirement for e4m3, not a choice here, and passing the
    other layout raises rather than silently returning something that does not match.

    `scale_a` and `scale_b` are the fp8 operand scales and may only be given for fp8. They are
    folded into one fp32 factor before the accumulator is scaled, which is what cuBLAS does --
    see `_fold_scales`. `out_dtype` defaults to the input dtype, or fp16 for fp8 input.

    The plan comes from the cuBLAS heuristic alone: no GEMM is run to decide it and no bytes
    are compared. A shape whose heuristic config falls outside what has been measured on this
    platform raises CublasUnsupportedShape rather than guessing.

    `cublaslt` picks which cuBLAS to match for this call -- a version prefix ("12.8") or a
    path -- overriding `set_cublaslt`. Default is the process-wide choice.

    `workspace_bytes` is how much workspace cuBLAS may use, overriding `set_workspace_bytes`;
    the default is 32 MiB. It is a parameter and not a constant because cuBLAS reads it when it
    chooses an algorithm: the same shape under two allowances can get two different
    `SPLITK_NUM`, and so two different results, both of them cuBLAS's. Match the allowance the
    caller you are comparing against uses, or the answer is bit-identical to a cuBLAS nobody
    ran.
    """
    kind = _kind_of(a)
    if kind != "fp8" and (scale_a is not None or scale_b is not None):
        raise ValueError(f"scale_a/scale_b are fp8-only, but the input is {a.dtype}")
    _check_operands(a, b, kind)
    out_dtype = out_dtype or (torch.float16 if kind == "fp8" else a.dtype)
    with _using_cublaslt(cublaslt), _using_workspace(workspace_bytes):
        plan = _resolve(a, b, kind, out_dtype)
        return _reconstruct(a, b, out_dtype, plan, scale=_fold_scales(scale_a, scale_b))


def cublas_equivalent_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: float = 1.0, scale_b: float = 1.0,
                                out_dtype: torch.dtype = torch.float16,
                                cublaslt: str | None = None,
                                workspace_bytes: int | None = None) -> torch.Tensor:
    """`cublas_equivalent_gemm` under the name torch uses for the scaled fp8 path."""
    return cublas_equivalent_gemm(a, b, out_dtype, scale_a, scale_b, cublaslt, workspace_bytes)


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def _check_one(kind, M, N, K):
    """Verify one shape. Returns (how it was settled, bit_ok)."""
    a, b = _make_inputs(M, N, K, kind, 7)
    ref = cublas_matmul(a, b, torch.float16 if kind == "fp8" else None)
    try:
        out = cublas_equivalent_gemm(a, b)
    except CublasUnsupportedShape:
        return "unsupported", None
    return plan_origin(M, N, K, kind), _bits_eq(out, ref)


def verify():
    if not torch.cuda.is_available():
        print("no CUDA GPU; verify() needs one.")
        return
    print(f"device: {torch.cuda.get_device_name()} | cap {torch.cuda.get_device_capability()} | "
          f"cuBLASLt {'.'.join(map(str, _cublaslt_version()))}\n")

    shapes = [("fp16", 4096, 4096, 4096), ("fp16", 2048, 2048, 512), ("fp16", 16, 16384, 16384),
              ("fp16", 64, 64, 32768), ("fp16", 128, 128, 16384), ("fp8", 4096, 4096, 4096),
              ("fp8", 8192, 8192, 8192), ("fp8", 64, 64, 65536), ("fp8", 128, 128, 65536), ("fp8", 16, 64, 131072)]
    n_ok, n_static, n_unsup = 0, 0, 0
    for (kind, M, N, K) in shapes:
        how, ok = _check_one(kind, M, N, K)
        if how == "unsupported":
            n_unsup += 1
            verdict = "UNSUPPORTED"
        else:
            n_static += 1
            n_ok += int(bool(ok))
            verdict = "BIT-IDENTICAL" if ok else "DIFFER"
        print(f"  {kind:4s} {M:5d}x{N:5d}x{K:6d}  {how:11s}  {verdict}")

    print(f"\nbit-identical {n_ok}/{n_static} | static {n_static} | unsupported {n_unsup}")


if __name__ == "__main__":
    verify()
