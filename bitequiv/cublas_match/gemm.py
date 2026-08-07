"""The public API: a cuBLAS-like matmul whose output is bit-identical to cuBLAS.

    from bitequiv.cublas_match import cublas_equivalent_gemm
    c = cublas_equivalent_gemm(a_fp16, b_fp16)                            # == cuBLAS fp16/bf16
    c = cublas_equivalent_gemm(a_fp8, b_fp8_col, scale_a=sa, scale_b=sb)  # == cuBLAS fp8

Three steps, and the first two happen once per shape:

  1. ask cuBLAS's own heuristic what it would run       -- `ltapi._cublas_direct(execute=False)`
  2. read the recipe off the config it returned         -- `plan.static_plan`
  3. launch the Triton kernel that recipe names         -- `kernels`

Nothing is executed to decide a shape and no bytes are compared.  A config the measured tables
do not carry declines (`CublasUnsupportedShape`) and the caller falls back to cuBLAS, so an
unfamiliar kernel costs coverage rather than returning wrong bits.

See README.md for what has been measured, on which GPUs and which cuBLASLt versions, and for
the known gaps.
"""
from __future__ import annotations

import struct

import torch

from .arch import platform
from .errors import CublasUnsupportedShape
from .kernels import (_triton_gemmsn, _triton_gemv13, _triton_gemv14, _triton_gemv_cslice, _triton_plain,
                      _triton_plain_k_per_dot, _triton_splitk, _triton_splitk_blocks, _triton_splitk_groups)
from .ltapi import (DEVICE, _cublas_direct, _kind_of, _using_cublaslt, _using_workspace, _workspace_bytes,
                    cublas_matmul, cublaslt_version)
from .plan import static_plan

# --------------------------------------------------------------------------- #
# Reconstruction planning (verify-then-use) + per-shape cache
# --------------------------------------------------------------------------- #
# (capability,cublaslt,M,N,K,kind,out_dtype) -> (origin, plan). The capability and the cuBLASLt
# version keep a process that switched either one from reusing the old recipe. origin is
# "static" or "unsupported".
_PLAN: dict[tuple, tuple] = {}

# The only operand dtypes this package has been measured on. `ltapi._kind_of` maps everything
# else to "fp16", so without an explicit check an fp32 or e5m2 operand would run the fp16 recipe
# and return a wrong answer silently instead of declining. `_check_operands` refuses them.
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)


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


def _reconstruct(a, b, out_dtype, plan, scale=1.0):
    """Launch the kernel a plan names. The only place plan fields are turned into arguments."""
    fp8_min_bm = platform().fp8_min_bm
    if plan.mode == "plain":
        return _triton_plain(a, b, out_dtype, scale=scale)
    if plan.mode == "k_per_dot":
        return _triton_plain_k_per_dot(a, b, out_dtype, plan.k_per_dot, plan.leading_group_k, scale=scale)
    if plan.mode == "splitk_groups":
        return _triton_splitk_groups(a, b, plan.k_chunk, plan.k_per_dot, plan.block_k, plan.merge_scheme, out_dtype,
                                     scale=scale, fp8_min_bm=fp8_min_bm)
    if plan.mode == "split_blocks":
        return _triton_splitk_blocks(a, b, plan.k_chunk, plan.block_k, out_dtype, scale=scale, fp8_min_bm=fp8_min_bm)
    # The three CUDA-core modes take no scale: they are fp16/bf16 only, and `scale` is the fp8
    # operand-scale fold, which the API refuses to accept for any other dtype.
    if plan.mode == "gemmsn":
        return _triton_gemmsn(a, b, out_dtype, plan.simt[0], plan.simt[1])
    if plan.mode == "gemv13":
        return _triton_gemv13(a, b, out_dtype, plan.gemv[0], plan.gemv[1])
    if plan.mode == "gemv_cslice":
        return _triton_gemv_cslice(a, b, out_dtype, plan.gemv[0], plan.gemv[1])
    if plan.mode == "gemv14":
        return _triton_gemv14(a, b, out_dtype, plan.gemv[0], plan.gemv[1])
    assert plan.mode == "split", plan.mode
    return _triton_splitk(a, b, plan.k_chunk, out_dtype, scale=scale, fp8_min_bm=fp8_min_bm)


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
    return (torch.cuda.get_device_capability(), cublaslt_version(), _workspace_bytes(), M, N, K, kind, out_dtype)


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
    plan, reason = static_plan(platform(), M, N, K, kind, config)
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
                           scale_a: float | None = None, scale_b: float | None = None, cublaslt: str | None = None,
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
                                out_dtype: torch.dtype = torch.float16, cublaslt: str | None = None,
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
          f"cuBLASLt {'.'.join(map(str, cublaslt_version()))}\n")

    shapes = [("fp16", 4096, 4096, 4096), ("fp16", 2048, 2048, 512), ("fp16", 16, 16384, 16384),
              ("fp16", 64, 64, 32768), ("fp16", 128, 128, 16384), ("fp8", 4096, 4096, 4096), ("fp8", 8192, 8192, 8192),
              ("fp8", 64, 64, 65536), ("fp8", 128, 128, 65536), ("fp8", 16, 64, 131072)]
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
