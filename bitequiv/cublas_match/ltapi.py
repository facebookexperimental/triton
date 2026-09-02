"""cuBLASLt through ctypes: pick a library, ask the heuristic, run the chosen algo.

Everything in here is architecture-independent -- it is the C API and nothing else.  Two things
the caller can select, because both are INPUTS to which kernel cuBLAS picks and therefore to
the bits it returns:

  * which `libcublasLt` to call (`set_cublaslt`, `cublaslt=`), and
  * how much workspace it is allowed (`set_workspace_bytes`, `workspace_bytes=`).

`cublas_matmul` is the reference the reconstruction is measured against: cuBLAS run directly on
the algo its own heuristic returned, never through torch, which reaches cuBLAS by a different
entry point and can pick a different algo.
"""
from __future__ import annotations

import contextlib
import ctypes
import glob
import os
import re
import threading

import torch

from .errors import CublasUnsupportedShape

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
_CFG_ID, _CFG_SPLITK, _CFG_REDUCTION, _CFG_CUSTOM, _CFG_STAGES = 0, 2, 3, 5, 6
_CFG_INNER_SHAPE, _CFG_CLUSTER_SHAPE = 7, 8
_CFG_COUNT = 9  # algo-config attributes 0..8
# The two attributes cublasLt.h declares uint16; every other one is 32 bits. cuBLASLt checks the
# buffer size exactly, so reading these with 4 bytes returns INVALID_VALUE, i.e. silently None.
_CFG_U16 = (_CFG_INNER_SHAPE, _CFG_CLUSTER_SHAPE)
_LT_SPEC = None  # process-wide choice from set_cublaslt(); None = newest installed
_TLS = threading.local()  # per-CALL overrides live here, never in a global -- see _using_cublaslt
_UNSET = object()
_LT_LIBS: dict = {}  # resolved path -> loaded library, so switching back and forth is free
_LT_PATHS: dict = {}  # spec -> resolved path
_SCALE_A = None  # device fp32 [1] holding the A scale cuBLAS is given (fp8 only)
_SCALE_B = None  # same for B
_WS_SPEC = None  # workspace override set by set_workspace_bytes(); None = the default
_WS_BUFS: dict = {}  # size in bytes -> device scratch, so switching sizes is free


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


def cublaslt_version():
    """(major, minor, patch) of the cuBLASLt we actually call.

    The library, not the CUDA toolkit torch was built against: this `.so` is what decides the
    kernels, and its own names carry the architecture (`nvjet_sm100_...`), so it is what the
    measured rules are pinned to.
    """
    return _lt_version_of(_load_lt())


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


def _kind_of(a) -> str:
    """Which measured dtype family an operand belongs to.

    Anything not fp8 or bf16 maps to "fp16"; `gemm._check_operands` refuses the dtypes that
    would otherwise fall through to the fp16 rules by accident.
    """
    if a.dtype == torch.float8_e4m3fn:
        return "fp8"
    if a.dtype == torch.bfloat16:
        return "bf16"
    return "fp16"


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


def cublas_matmul(a, b, out_dtype=None, scale_a=None, scale_b=None, cublaslt=None, workspace_bytes=None):
    """cuBLAS's OWN output (run directly via cublasLtMatmul), the reference our
    reconstruction must match. fp16/bf16: `b` is [K,N]. fp8: `b` is [K,N] column-major.
    `cublaslt` and `workspace_bytes` pick which cuBLAS and which allowance, as in
    `cublas_equivalent_gemm`."""
    kind = _kind_of(a)
    out_dtype = out_dtype or (torch.float16 if kind == "fp8" else a.dtype)
    with _using_cublaslt(cublaslt), _using_workspace(workspace_bytes):
        return _cublas_direct(a, b, kind, out_dtype, sa=scale_a if scale_a is not None else 1.0,
                              sb=scale_b if scale_b is not None else 1.0)[0]
