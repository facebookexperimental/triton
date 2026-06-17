# NPOT (non-power-of-2) tests. Run with TRITON_ALLOW_NPOT=1 via the
# py_npot_tests buck target. Split out of test_core.py so the flag-on target is
# scoped to just these tests; test_core.py runs flag-off via py_unit_language_tests.
# ruff: noqa: F821,F841
import pytest
import torch

import triton
import triton.language as tl

from triton._internal_testing import is_cuda


@pytest.mark.skipif(not triton.knobs.language.allow_npot, reason='NPOT not enabled (set TRITON_ALLOW_NPOT=1)')
@pytest.mark.interpreter
@pytest.mark.parametrize("SIZE", [33, 48, 127, 768, 1023, 1536])
def test_npot_arange(SIZE, device):
    """1D NPOT: verifies modular codegen (ADD+UREM) for tl.arange."""

    @triton.jit
    def kernel(Out, SIZE: tl.constexpr):
        idx = tl.arange(0, SIZE)
        tl.store(Out + idx, idx)

    out = torch.empty(SIZE, dtype=torch.int32, device=device)
    kernel[(1, )](out, SIZE=SIZE)
    expected = torch.arange(SIZE, dtype=torch.int32, device=device)
    assert torch.equal(out, expected)


@pytest.mark.skipif(not triton.knobs.language.allow_npot, reason='NPOT not enabled (set TRITON_ALLOW_NPOT=1)')
@pytest.mark.interpreter
@pytest.mark.parametrize("COLS", [48, 127, 768, 1023, 1536])
def test_npot_reduction(COLS, device):
    """1D NPOT reduction: verifies tl.sum with NPOT block size."""

    @triton.jit
    def kernel(X, Out, N: tl.constexpr):
        row_idx = tl.program_id(0)
        col_idx = tl.arange(0, N)
        x = tl.load(X + row_idx * N + col_idx)
        tl.store(Out + row_idx, tl.sum(x))

    ROWS = 16
    x = torch.randn(ROWS, COLS, dtype=torch.float32, device=device)
    out = torch.empty(ROWS, dtype=torch.float32, device=device)
    kernel[(ROWS, )](x, out, N=COLS)
    torch.testing.assert_close(out, x.sum(dim=1), rtol=1e-5, atol=1e-5)


# Device-only NPOT reduction correctness sweep. Unlike the @interpreter test
# above, this exercises the real TritonGPU->LLVM reduction lowering on hardware,
# which is where phantom (pow2-rounded) register/lane/warp slots are masked with
# the reduction identity. The interpreter does not model lane/warp/register
# wrapping, so it cannot catch the bug class this test targets.
@pytest.mark.skipif(not triton.knobs.language.allow_npot, reason='NPOT not enabled (set TRITON_ALLOW_NPOT=1)')
@pytest.mark.parametrize("N", [3, 6, 17, 33, 48, 96, 192, 768, 1023, 1536])
@pytest.mark.parametrize("num_warps", [1, 4, 8])
@pytest.mark.parametrize("op", ["sum", "max"])
def test_npot_reduction_device(N, num_warps, op, device):
    """Device NPOT reduction sweep for tl.sum / tl.max across num_warps.

    Regression test for phantom-slot miscompute: pow2-rounding the NPOT axis
    creates wrapped register/lane/warp slots that must be identity-filled
    (0 for sum, -inf for max) before the reduction folds them in. Covers
    within-thread register wrapping, intra-warp lane wrapping, and inter-warp
    NPOT warp collapse (where >1 warp maps onto the same NPOT axis slot).
    """
    if not is_cuda():
        pytest.skip("device NPOT reduction test is CUDA-only")

    if op == "sum":

        @triton.jit
        def kernel(X, Out, N: tl.constexpr):
            row_idx = tl.program_id(0)
            col_idx = tl.arange(0, N)
            x = tl.load(X + row_idx * N + col_idx)
            tl.store(Out + row_idx, tl.sum(x))
    else:

        @triton.jit
        def kernel(X, Out, N: tl.constexpr):
            row_idx = tl.program_id(0)
            col_idx = tl.arange(0, N)
            x = tl.load(X + row_idx * N + col_idx)
            tl.store(Out + row_idx, tl.max(x))

    ROWS = 16
    x = torch.randn(ROWS, N, dtype=torch.float32, device=device)
    out = torch.empty(ROWS, dtype=torch.float32, device=device)
    kernel[(ROWS, )](x, out, N=N, num_warps=num_warps)
    ref = x.sum(dim=1) if op == "sum" else x.max(dim=1).values
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not triton.knobs.language.allow_npot, reason='NPOT not enabled (set TRITON_ALLOW_NPOT=1)')
@pytest.mark.parametrize("M, N", [(16, 96), (16, 192)])
def test_npot_reduction_device_2d_registers(M, N, device):
    """Device 2D NPOT reduction with the NPOT axis carried on registers.

    A single program at num_warps=1 reduces an [M, N] tile over the NPOT N axis.
    With one warp the N axis spans multiple registers per thread, exercising
    computeRegisterWrappingPreds (within-thread register wrapping) that the 1D
    device sweep lands on lane/warp instead. Pow2-rounding N creates phantom
    register slots that must be identity-filled (0 for sum) before the fold.
    """
    if not is_cuda():
        pytest.skip("device NPOT reduction test is CUDA-only")

    @triton.jit
    def kernel(X, Out, M: tl.constexpr, N: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, N)[None, :]
        x = tl.load(X + rows * N + cols)
        tl.store(Out + tl.arange(0, M), tl.sum(x, axis=1))

    x = torch.randn(M, N, dtype=torch.float32, device=device)
    out = torch.empty(M, dtype=torch.float32, device=device)
    kernel[(1, )](x, out, M=M, N=N, num_warps=1)
    torch.testing.assert_close(out, x.sum(dim=1), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not triton.knobs.language.allow_npot, reason='NPOT not enabled (set TRITON_ALLOW_NPOT=1)')
def test_npot_flag_is_on_sentinel():
    # Guards against the whole NPOT suite silently skipping in CI.
    assert triton.knobs.language.allow_npot is True


@pytest.mark.skipif(not triton.knobs.language.allow_npot, reason='NPOT not enabled (set TRITON_ALLOW_NPOT=1)')
@pytest.mark.interpreter
@pytest.mark.parametrize("M, N", [(3, 48), (33, 48), (3, 127), (3, 1023), (3, 1536)])
def test_npot_2d_softmax(M, N, device):
    """2D NPOT: verifies per-dim algebra (product semimodule)."""

    @triton.jit
    def kernel(X, Out, M: tl.constexpr, N: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, N)[None, :]
        offsets = rows * N + cols
        x = tl.load(X + offsets)
        row_max = tl.max(x, axis=1)[:, None]
        numerator = tl.exp(x - row_max)
        denominator = tl.sum(numerator, axis=1)[:, None]
        tl.store(Out + offsets, numerator / denominator)

    x = torch.randn(M, N, dtype=torch.float32, device=device)
    out = torch.empty(M, N, dtype=torch.float32, device=device)
    kernel[(1, )](x, out, M=M, N=N)
    torch.testing.assert_close(out, torch.softmax(x, dim=1), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not triton.knobs.language.allow_npot, reason='NPOT not enabled (set TRITON_ALLOW_NPOT=1)')
@pytest.mark.interpreter
@pytest.mark.parametrize("M, N", [(3, 48), (33, 48), (3, 127), (3, 1023)])
def test_npot_backward(M, N, device):
    """2D NPOT backward: verifies invertAndCompose/lstsqModular layout conversion codegen."""

    @triton.jit
    def softmax_fwd_kernel(X, Out, M: tl.constexpr, N: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, N)[None, :]
        offsets = rows * N + cols
        x = tl.load(X + offsets)
        row_max = tl.max(x, axis=1)[:, None]
        numerator = tl.exp(x - row_max)
        denominator = tl.sum(numerator, axis=1)[:, None]
        tl.store(Out + offsets, numerator / denominator)

    @triton.jit
    def softmax_bwd_kernel(Out, DOut, DX, M: tl.constexpr, N: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, N)[None, :]
        offsets = rows * N + cols
        s = tl.load(Out + offsets)
        dy = tl.load(DOut + offsets)
        s_dy = s * dy
        sum_s_dy = tl.sum(s_dy, axis=1)[:, None]
        dx = s * (dy - sum_s_dy)
        tl.store(DX + offsets, dx)

    class NpotSoftmax(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            out = torch.empty_like(x)
            softmax_fwd_kernel[(1, )](x, out, M=M, N=N)
            ctx.save_for_backward(out)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (out, ) = ctx.saved_tensors
            dx = torch.empty_like(out)
            # grad from .sum() is a stride-0 broadcast; the kernel indexes with
            # contiguous offsets, so materialize it contiguous before launch.
            grad_output = grad_output.contiguous()
            softmax_bwd_kernel[(1, )](out, grad_output, dx, M=M, N=N)
            return dx

    x = torch.randn(M, N, dtype=torch.float32, device=device, requires_grad=True)
    # Triton forward + backward
    triton_out = NpotSoftmax.apply(x)
    triton_out.sum().backward()
    triton_dx = x.grad.clone()

    # Reference: PyTorch autograd
    x.grad = None
    ref_out = torch.softmax(x, dim=1)
    ref_out.sum().backward()
    ref_dx = x.grad

    torch.testing.assert_close(triton_out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(triton_dx, ref_dx, rtol=1e-4, atol=1e-4)
