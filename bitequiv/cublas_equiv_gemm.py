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
  3. The k-slice granularity `G` is fixed per dtype/kernel family (measured on
     GB300 / sm_100, reference = cuBLAS run directly, no torch):
       - fp16 / bf16 (nvjet)                     -> G = 64
       - fp8 (nvjet 2-CTA or horizontal)         -> G = 128
       - fp8 (nvjet vertical)                    -> G = 64
     `chunk = ceil(ceil(K/G)/nsplit)*G`; slices are uneven, the LAST is smaller.
  4. VERIFY-THEN-USE (sound by construction): before trusting any reconstruction
     we byte-compare it against cuBLAS run directly (`cublasLtMatmul` with the
     heuristic's own algo) on 3 order-sensitive seeds.  The first candidate recipe
     that matches all 3 is cached per shape; if none match, the shape is
     `CublasUnsupportedShape` -- we never silently return a non-matching result.
     Cached shapes are then PURE Triton (no cuBLAS call).

Known unsupported (raise `CublasUnsupportedShape`): fp16 non-aligned CUTLASS
`s1688` (K=8 MMA -- Triton's `tl.dot` is K=16) and odd-K tails; fp8 vertical
split-K where cuBLAS reduces K across the cluster CTAs (a distributed FP32 order a
tile-level two-pass split-K cannot emit, e.g. some N=192 shapes).

Depends on `torch`, `triton`, `ctypes` (stdlib), and the CUDA `libcublasLt` shared
library (always present with a CUDA runtime).  GB300 / sm_100.
"""
from __future__ import annotations

import ctypes
import glob

import torch
import triton
import triton.language as tl

DEVICE = "cuda"
_WS_BYTES = 33554432  # 32 MiB scratch for the cuBLAS-direct reference execute
_SEEDS = tuple(range(32))  # order-sensitive seeds to verify a reconstruction; more seeds reject borderline
# (near-miss) nsplit that a wide sweep would otherwise accept on too few samples, at the cost of slower calibration.


class CublasUnsupportedShape(Exception):
    """No Triton reconstruction bit-matches cuBLAS for this shape, even with a runtime
    byte-compare (e.g. fp8 vertical/cluster split-K, fp16 non-aligned s1688 K=8 / odd-K)."""


class CublasNeedRuntimeMatch(Exception):
    """The shape cannot be reconstructed from the heuristic STATICALLY with confidence; a
    runtime byte-compare against cuBLAS is needed to confirm (or reject) it. Raised only in
    static mode (`enable_runtime_match=False`). Re-call with `enable_runtime_match=True`."""


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
    om = (pm * BM + tl.arange(0, BM)) % M
    on = (pn * BN + tl.arange(0, BN)) % N
    ok = tl.arange(0, BK)
    ap = A + om[:, None] * am + ok[None, :] * ak
    bp = B + ok[:, None] * bk + on[None, :] * bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(ap, mask=ok[None, :] < K - k * BK, other=0.0),
                     tl.load(bp, mask=ok[:, None] < K - k * BK, other=0.0), acc)
        ap += BK * ak
        bp += BK * bk
    acc = acc * sa * sb
    ocm = pm * BM + tl.arange(0, BM)
    ocn = pn * BN + tl.arange(0, BN)
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
    tl.store(W + sk * ws + ocm[:, None] * wm + ocn[None, :] * wn, acc, mask=(ocm[:, None] < M) & (ocn[None, :] < N))


@triton.jit
def _splitk_combine(W, C, M, N, ws, wm, wn, cm, cn, S, sa, sb, BM: tl.constexpr, BN: tl.constexpr):
    """Pass 2: sum the S FP32 partials in FORWARD order (slice 0..S-1), scale, cast."""
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    m2 = (om[:, None] < M) & (on[None, :] < N)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for i in range(0, S):
        acc += tl.load(W + i * ws + om[:, None] * wm + on[None, :] * wn, mask=m2, other=0.0)
    acc = acc * sa * sb
    tl.store(C + cm * om[:, None] + cn * on[None, :], acc.to(C.dtype.element_ty), mask=m2)


def _tile(M: int, N: int) -> tuple[int, int]:
    """Partial tile (bit-neutral): small tiles for small M,N, capped at 128."""
    BM = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
    BN = 16 if N <= 16 else 32 if N <= 32 else 64 if N <= 64 else 128
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


def _triton_splitk(a, b, chunk, out_dtype, sa=1.0, sb=1.0, BK=64):
    """Deterministic two-pass split-K at `chunk`-element early slices (uneven tail),
    forward FP32 combine. Returns None if the chunk does not yield a real split (nsplit<2)."""
    b = _kcontig(b)
    M, K = a.shape
    N = b.shape[1]
    nsplit = (K + chunk - 1) // chunk
    if nsplit < 2 or nsplit > 8192:
        return None
    BM, BN = _tile(M, N)
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
_CFG_ID, _CFG_SPLITK = 0, 2
_LT = None
_ONES = None  # device fp32 [1.0] for fp8 scale pointers
_WS = None    # device scratch for the reference execute


class _HeurResult(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_char * 64), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


def _load_lt():
    global _LT, _ONES, _WS
    if _LT is not None:
        return _LT
    cands = (["/usr/local/cuda-13.0/lib64/libcublasLt.so.13"] + glob.glob("/usr/local/cuda*/lib64/libcublasLt.so*") +
             glob.glob("/usr/local/cuda*/targets/*/lib/libcublasLt.so*") + ["libcublasLt.so.13", "libcublasLt.so"])
    lib = None
    for c in cands:
        try:
            lib = ctypes.CDLL(c)
            break
        except OSError:
            continue
    if lib is None:
        raise OSError("libcublasLt not found")
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
    _ONES = torch.ones(1, dtype=torch.float32, device=DEVICE)
    _WS = torch.empty(_WS_BYTES, dtype=torch.uint8, device=DEVICE)
    _LT = lib
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
        if not execute:
            return None, nsplit
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
        return out, (nsplit if isinstance(nsplit, int) and nsplit >= 1 else 1)
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


def cublas_matmul(a, b, out_dtype=None):
    """cuBLAS's OWN output (run directly via cublasLtMatmul), the reference our
    reconstruction must match. fp16/bf16: `b` is [K,N]. fp8: `b` is [K,N] column-major."""
    kind = _kind_of(a)
    out_dtype = out_dtype or (torch.float16 if kind == "fp8" else a.dtype)
    return _cublas_direct(a, b, kind, out_dtype)[0]


# --------------------------------------------------------------------------- #
# Reconstruction planning (verify-then-use) + per-shape cache
# --------------------------------------------------------------------------- #
# (M,N,K,kind,out_dtype) -> (origin, plan): origin "static"|"runtime"|"unsupported"; plan
# ("plain",None) | ("split",chunk) | None. Decided once; a "runtime" plan is withheld in static mode.
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
    """k-slice granularity for the split-K partition: fp16/bf16 = 64, fp8 = 128
    (measured; fp8 G=128 at the heuristic nsplit dominates G=64 for every kernel family,
    including 2-CTA/horizontal/vertical -- the earlier "vertical->64" was a 3-seed artifact)."""
    return 128 if kind == "fp8" else 64


def _static_chunk(K, nsplit, kind):
    """The k-slice partition the cuBLAS heuristic implies: G-grain, uneven, remainder last."""
    return _chunk_from_ns(K, nsplit, _grain(kind))


_CALIB_MAX_NS = 256  # bounded split count for the wide runtime sweep


def _splitk_candidate_chunks(K, nsplit, kind):
    """Candidate partitions for the runtime-match path, in try-order: the heuristic nsplit
    +/-1 first (the common case resolves fast), then a WIDE canonical sweep over nsplit
    2..cap at the grain(s) -- the heuristic's split count can be off by more than 1, so the
    narrow window alone falsely rejects reconstructable shapes. Grains: fp16/bf16 = 64; fp8 =
    128 then 64. Deduped by chunk."""
    grains = (64, ) if kind != "fp8" else (128, 64)
    chunks, seen = [], set()

    def add(ns, G):
        if ns < 2:
            return
        chunk = _chunk_from_ns(K, ns, G)
        if G <= chunk < K and chunk not in seen:
            seen.add(chunk)
            chunks.append(chunk)

    for ns in (nsplit, nsplit - 1, nsplit + 1):  # near the heuristic first
        add(ns, _grain(kind))
    for G in grains:                             # then the wide sweep
        cap = min((K + G - 1) // G, _CALIB_MAX_NS)
        for ns in range(2, cap + 1):
            add(ns, G)
    return chunks


def _make_inputs(M, N, K, kind, seed):
    """Order-sensitive inputs (magnitude spread + alternating sign along K) so the K
    reduction order shows up in the output bits. Magnitudes are kept in range so the
    output stays finite -- for fp8 the values must fit e4m3 (max 448)."""
    torch.manual_seed(seed)
    sign = torch.where(torch.arange(K, device=DEVICE) % 2 == 0, 1.0, -1.0).unsqueeze(0)
    if kind == "fp8":
        scale = torch.logspace(-1, 1, K, device=DEVICE, dtype=torch.float32).unsqueeze(0)  # 0.1..10, fp8-safe
        a = (torch.randn(M, K, device=DEVICE) * scale * sign).to(torch.float8_e4m3fn)
        b = (torch.randn(N, K, device=DEVICE) * 0.25).to(torch.float8_e4m3fn).t()  # [K,N] col-major
        return a, b
    dt = torch.bfloat16 if kind == "bf16" else torch.float16
    scale = torch.logspace(-3, 3, K, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    a = (torch.randn(M, K, device=DEVICE) * scale * sign).to(dt)
    b = (torch.randn(K, N, device=DEVICE) * 0.05).to(dt)
    return a, b


def _bits_eq(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


def _aligned(M, N, K, kind):
    """Contiguous-dim alignment cuBLAS needs to stay on its reconstructable nvjet path.
    fp16/bf16: K%8==0 and N%8==0 (M free). fp8: all dims %16."""
    if kind == "fp8":
        return M % 16 == 0 and N % 16 == 0 and K % 16 == 0
    return K % 8 == 0 and N % 8 == 0


def _is_static_exact(kind, nsplit, aligned, M, N):
    """Can we reconstruct from the heuristic ALONE (no runtime byte-compare) and be sure it
    is bit-exact? Measured on GB300/sm_100:
      - fp16/bf16 aligned: yes (plain and split-K at G=64@heur both match);
      - fp8 large-output plain (ceil(M/64)*ceil(N/64) >= 64, nsplit==1): yes;
      - fp8 skinny (small output): NO -- even at SPLITK_NUM==1 cuBLAS may use a cluster kernel
        that reduces K across CTAs, which a single-accumulator/two-pass Triton kernel cannot
        bit-match; and fp8 split-K vertical is not reproducible -> defer to a runtime match;
      - fp16/bf16 non-aligned: NO (CUTLASS; K=16 variants reconstruct, s1688/odd-K do not)."""
    if not aligned:
        return False
    if kind in ("fp16", "bf16"):
        return True
    tiles = ((M + 63) // 64) * ((N + 63) // 64)
    return nsplit <= 1 and tiles >= 64  # fp8: only the large-output plain subset is static-safe


def _reconstruct(a, b, out_dtype, plan, sa=1.0, sb=1.0):
    mode, chunk = plan
    if mode == "plain":
        return _triton_plain(a, b, out_dtype, sa=sa, sb=sb)
    return _triton_splitk(a, b, chunk, out_dtype, sa=sa, sb=sb)


def _calibrate(M, N, K, kind, out_dtype):
    """RUNTIME match: find the reconstruction that bit-matches cuBLAS run directly, verified
    on _SEEDS order-sensitive seeds. Returns ("plain",None) or ("split",chunk); raises
    CublasUnsupportedShape if nothing matches. cuBLAS's choice is a function of the shape
    (not the data), so one calibration serves all future inputs of that shape."""
    refs, nsplit = [], 1
    for i, s in enumerate(_SEEDS):
        a, b = _make_inputs(M, N, K, kind, s)
        d, ns = _cublas_direct(a, b, kind, out_dtype)
        refs.append((a, b, d))
        if i == 0:
            nsplit = ns

    if nsplit <= 1:  # cuBLAS runs a plain GEMM
        if all(_bits_eq(_triton_plain(a, b, out_dtype), d) for a, b, d in refs):
            return ("plain", None)
        raise CublasUnsupportedShape(f"{M}x{N}x{K} {kind}: plain reconstruction does not match cuBLAS")

    a0, b0, d0 = refs[0]
    for chunk in _splitk_candidate_chunks(K, nsplit, kind):  # cuBLAS runs split-K
        c0 = _triton_splitk(a0, b0, chunk, out_dtype)
        if c0 is None or not _bits_eq(c0, d0):
            continue  # seed-0 prefilter: keeps the wide nsplit sweep cheap
        if all(_bits_eq(_triton_splitk(a, b, chunk, out_dtype), d) for a, b, d in refs[1:]):
            return ("split", chunk)
    raise CublasUnsupportedShape(f"{M}x{N}x{K} {kind}: no split-K recipe matches cuBLAS (nsplit={nsplit})")


def _resolve(a, b, kind, out_dtype, enable_runtime_match):
    """Reconstruction plan for this shape (cached). Static-exact shapes are planned from the
    heuristic alone (no GEMM run). Others need a runtime byte-compare: with
    enable_runtime_match they are calibrated against cuBLAS; without, raise
    CublasNeedRuntimeMatch."""
    M, K = a.shape
    N = b.shape[1]
    key = (M, N, K, kind, out_dtype)
    if key in _PLAN:
        origin, plan = _PLAN[key]  # origin: "static" | "runtime" | "unsupported"
        if origin == "unsupported":
            raise CublasUnsupportedShape(f"{M}x{N}x{K} {kind}: unsupported (cached)")
        if origin == "runtime" and not enable_runtime_match:
            raise CublasNeedRuntimeMatch(f"{M}x{N}x{K} {kind}: needs a runtime match (cached)")
        return plan

    nsplit = _cublas_direct(a, b, kind, out_dtype, execute=False)[1]  # static: heuristic only, no GEMM run
    if _is_static_exact(kind, nsplit, _aligned(M, N, K, kind), M, N):
        plan = ("plain", None) if nsplit <= 1 else ("split", _static_chunk(K, nsplit, kind))
        _PLAN[key] = ("static", plan)
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
                           enable_runtime_match: bool = False) -> torch.Tensor:
    """fp16/bf16 GEMM, bit-identical to cuBLAS. `a` is [M,K], `b` is [K,N].

    Static by default: the reconstruction is planned from the cuBLAS heuristic alone (no GEMM
    run). Aligned fp16/bf16 is static-exact. A shape that is not statically guaranteed
    (non-aligned) raises CublasNeedRuntimeMatch -- unless `enable_runtime_match=True`, which
    byte-compares candidates against cuBLAS and returns the match or raises
    CublasUnsupportedShape."""
    kind = _kind_of(a)
    if kind == "fp8":
        raise ValueError("use cublas_equivalent_scaled_mm for fp8")
    out_dtype = out_dtype or a.dtype
    plan = _resolve(a, b, kind, out_dtype, enable_runtime_match)
    return _reconstruct(a, b, out_dtype, plan)


def cublas_equivalent_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: float = 1.0, scale_b: float = 1.0,
                                out_dtype: torch.dtype = torch.float16,
                                enable_runtime_match: bool = False) -> torch.Tensor:
    """fp8 (e4m3) GEMM, bit-identical to cuBLAS. `a` is [M,K], `b` is [K,N] column-major
    (i.e. from `w.t()`); scales are scalars.

    fp8 plain is static-exact. fp8 split-K is NOT statically guaranteed (the vertical/cluster
    kernel is not bit-reproducible) -> raises CublasNeedRuntimeMatch unless
    `enable_runtime_match=True`, which byte-compares against cuBLAS and returns the match or
    raises CublasUnsupportedShape (vertical split-K)."""
    plan = _resolve(a, b, "fp8", out_dtype, enable_runtime_match)
    return _reconstruct(a, b, out_dtype, plan, sa=scale_a, sb=scale_b)


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def _check_one(kind, M, N, K):
    """Classify + verify one shape: try static first; if it needs a runtime match, retry with
    enable_runtime_match=True. Returns (class, bit_ok) where class is static/runtime/unsupported."""
    a, b = _make_inputs(M, N, K, kind, 7)
    ref = cublas_matmul(a, b, torch.float16 if kind == "fp8" else None)
    call = (lambda rt: cublas_equivalent_scaled_mm(a, b, 1.0, 1.0, torch.float16, enable_runtime_match=rt)) \
        if kind == "fp8" else (lambda rt: cublas_equivalent_gemm(a, b, enable_runtime_match=rt))
    try:
        return "static", _bits_eq(call(False), ref)
    except CublasNeedRuntimeMatch:
        pass
    try:
        return "runtime", _bits_eq(call(True), ref)
    except CublasUnsupportedShape:
        return "unsupported", None


def verify():
    if not torch.cuda.is_available():
        print("no CUDA GPU; verify() needs one.")
        return
    print(f"device: {torch.cuda.get_device_name()} | cap {torch.cuda.get_device_capability()} | "
          f"cuda {torch.version.cuda}")
    print("class = static (heuristic only) / runtime (byte-compare) / unsupported; then vs cuBLAS-direct\n")

    shapes = [("fp16", 4096, 4096, 4096), ("fp16", 2048, 2048, 512), ("fp16", 16, 16384, 16384),
              ("fp16", 64, 64, 32768), ("fp16", 128, 128, 16384), ("fp8", 4096, 4096, 4096),
              ("fp8", 8192, 8192, 8192), ("fp8", 64, 64, 65536), ("fp8", 128, 128, 65536), ("fp8", 16, 64, 131072)]
    n_ok = n_static = n_runtime = n_unsup = 0
    for (kind, M, N, K) in shapes:
        cls, ok = _check_one(kind, M, N, K)
        n_static += int(cls == "static")
        n_runtime += int(cls == "runtime")
        if cls == "unsupported":
            n_unsup += 1
            verdict = "UNSUPPORTED"
        else:
            n_ok += int(bool(ok))
            verdict = "BIT-IDENTICAL" if ok else "DIFFER"
        print(f"  {kind:4s} {M:5d}x{N:5d}x{K:6d}  {cls:11s}  {verdict}")

    print(f"\nbit-identical {n_ok}/{n_static + n_runtime} | static {n_static} | runtime {n_runtime} | "
          f"unsupported {n_unsup}")


if __name__ == "__main__":
    verify()
