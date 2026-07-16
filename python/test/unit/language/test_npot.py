# NPOT (non-power-of-2) language tests.
#
# Gated on TRITON_ALLOW_NPOT=1, which lets a non-power-of-2 shape flow through
# the compiler and build a modular LinearLayout. With the flag OFF the frontend
# rejects NPOT shapes (arange must be pow2), so every test here skips.
#
# Coverage: elementwise lowering via the modular per-element index path
# (applyLinearLayout ADD+UREM).
#
# The shared clampVecSizeForNpot inline is exercised here only via the CUDA
# device/lit path (tritongpu_to_llvm_npot.mlir). The AMD getVectorSize
# call-site wiring (third_party/amd/.../Utility.cpp) is not directly tested;
# AMD numeric is a separate lane.

import pytest
import torch

import triton
import triton.language as tl
from triton import knobs

from triton._internal_testing import is_cuda


def _allow_npot():
    return bool(knobs.language.allow_npot)


requires_npot = pytest.mark.skipif(
    not _allow_npot(),
    reason="requires TRITON_ALLOW_NPOT=1",
)
requires_gpu = pytest.mark.skipif(not is_cuda(), reason="requires CUDA GPU")


def _rel_err(actual, ref, eps=1e-6):
    a = actual.float()
    r = ref.float()
    denom = max(r.abs().max().item(), eps)
    return (a - r).abs().max().item() / denom


@requires_gpu
@requires_npot
@pytest.mark.parametrize("size", [33, 48, 127, 768, 1023, 1536])
def test_npot_arange(size):
    """arange + store over an NPOT extent exercises the modular store index
    (applyLinearLayout modular ADD+UREM path)."""

    @triton.jit
    def kernel(Out, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Out + i, i.to(tl.float32))

    out = torch.empty(size, device="cuda", dtype=torch.float32)
    kernel[(1, )](out, N=size)
    ref = torch.arange(size, device="cuda", dtype=torch.float32)
    assert _rel_err(out, ref) < 1e-5, f"size={size}"


@requires_gpu
@requires_npot
@pytest.mark.parametrize("size", [48, 96, 192, 1023])
def test_npot_elementwise(size):

    @triton.jit
    def kernel(X, Y, Out, N: tl.constexpr):
        i = tl.arange(0, N)
        x = tl.load(X + i)
        y = tl.load(Y + i)
        tl.store(Out + i, x * y + x)

    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")
    out = torch.empty(size, device="cuda")
    kernel[(1, )](x, y, out, N=size)
    assert _rel_err(out, x * y + x) < 1e-4, f"size={size}"


@requires_gpu
@requires_npot
@pytest.mark.parametrize("size", [48, 96, 192, 768])
def test_npot_copy_one_warp(size):
    """1D copy at num_warps=1, so sizePerThread = ceil(size / 32). size=96 gives
    sizePerThread==3, where the vectorized load/store must not over-vectorize a
    non-power-of-2 per-thread span (else misaligned access)."""

    @triton.jit
    def kernel(X, Out, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Out + i, tl.load(X + i))

    x = torch.randn(size, device="cuda")
    out = torch.empty(size, device="cuda")
    kernel[(1, )](x, out, N=size, num_warps=1)
    assert _rel_err(out, x) < 1e-6, f"size={size}"


@requires_gpu
@requires_npot
@pytest.mark.parametrize("M, N", [(16, 17), (16, 48), (8, 17)])
def test_npot_copy_2d(M, N):
    """2D copy with a non-power-of-2 inner extent exercises the blocked ->
    LinearLayout construction for NPOT row/col tiles."""

    @triton.jit
    def kernel(X, Out, M: tl.constexpr, N: tl.constexpr):
        idx = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        tl.store(Out + idx, tl.load(X + idx))

    x = torch.randn(M, N, device="cuda")
    out = torch.empty(M, N, device="cuda")
    kernel[(1, )](x, out, M=M, N=N)
    assert _rel_err(out, x) < 1e-6, f"M={M},N={N}"


@requires_gpu
@pytest.mark.parametrize("size", [64, 128, 256, 1024])
def test_pow2_control_elementwise(size):
    """Pow2 control: exercises the same elementwise lowering path as the NPOT
    tests, but with a power-of-2 shape so the layout is never modular. This must
    pass with the flag OFF (no @requires_npot), guarding that the NPOT changes
    are a correctness no-op for pow2 shapes."""

    @triton.jit
    def kernel(X, Y, Out, N: tl.constexpr):
        i = tl.arange(0, N)
        x = tl.load(X + i)
        y = tl.load(Y + i)
        tl.store(Out + i, x * y + x)

    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")
    out = torch.empty(size, device="cuda")
    kernel[(1, )](x, y, out, N=size)
    assert _rel_err(out, x * y + x) < 1e-4, f"size={size}"
