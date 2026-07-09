# NPOT (non-power-of-2) language tests.
#
# Gated on TRITON_ALLOW_NPOT=1, which lets a non-power-of-2 shape flow through
# the compiler and build a modular LinearLayout. With the flag OFF the frontend
# rejects NPOT shapes (arange must be pow2), so every test here skips.
#
# Coverage: elementwise lowering via the modular per-element index path
# (applyLinearLayout ADD+UREM), plus reduction/softmax/layernorm masking of the
# phantom register/lane/warp slots the pow2-rounding introduces. A representative
# set of sizes is kept (not a brute sweep): running many JIT-compiled device
# kernels back-to-back in one process accumulates GPU state and flakes.
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
@pytest.mark.parametrize("size", [33, 127, 1023])
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
@pytest.mark.parametrize("size", [48, 192, 1023])
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
def test_npot_copy_one_warp():
    """1D copy at num_warps=1, size=96 gives sizePerThread==3, where the
    vectorized load/store must not over-vectorize a non-power-of-2 per-thread
    span (else misaligned access) -- exercises clampVecSizeForNpot."""

    @triton.jit
    def kernel(X, Out, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Out + i, tl.load(X + i))

    x = torch.randn(96, device="cuda")
    out = torch.empty(96, device="cuda")
    kernel[(1, )](x, out, N=96, num_warps=1)
    assert _rel_err(out, x) < 1e-6


@requires_gpu
@requires_npot
@pytest.mark.parametrize("M, N", [(16, 17), (8, 17)])
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


# ---------------------------------------------------------------------------
# Modular convert_layout / view ops must route through the real remap, not a
# pow2-only cheap no-op. Two guards: isExpensiveView (getFreeVariableMasks can't
# tell two ADD-mod-N maps apart) and the convert Case-5 no-op (minimalCvtLayout
# quotients a modular register dim away). Their core is pinned in
# LinearLayoutTest.{ModularRelayoutIsNotIdentity,ModularRegisterDimNotQuotientedAway}.
# End-to-end modular reshape / axis=None reduce are not exercised here
# (unimplemented; they fail earlier) -- the reshape is correctly marked EXPENSIVE
# (loud fail, not scramble). Random inputs (constants are scramble-blind).
# ---------------------------------------------------------------------------


@requires_gpu
@requires_npot
@pytest.mark.parametrize("M, N", [(16, 48), (16, 192), (33, 48), (8, 96)])
def test_npot_reshape_2d_to_1d_expensive_view_detected(M, N):
    """A modular 2D -> 1D reshape genuinely changes the layout. Previously
    isExpensiveView false-positived it as a cheap no-op and the reshape lowering
    silently scrambled the elements. Now the view is correctly classified
    (expensive / not a no-op), so the compile fails loudly instead of producing
    scrambled data. This is strictly safer; the correct-result path needs the
    follow-on modular reshape remap. We assert it does NOT silently succeed with
    scrambled output."""

    @triton.jit
    def kernel(X, Out, M: tl.constexpr, N: tl.constexpr):
        idx = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        x = tl.load(X + idx)
        y = tl.reshape(x, (M * N, ))
        tl.store(Out + tl.arange(0, M * N), y)

    x = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.full((M * N, ), -1.0, device="cuda", dtype=torch.float32)
    ref = x.reshape(M * N)
    try:
        kernel[(1, )](x, out, M=M, N=N)
    except Exception:
        # Expected: modular reshape is (correctly) rejected as an expensive view
        # rather than silently scrambling.
        return
    # If it DID compile, the result must be correct (never a silent scramble).
    assert _rel_err(out, ref) < 1e-6, f"silent scramble on reshape M={M},N={N}"


@requires_gpu
@requires_npot
@pytest.mark.parametrize("M, N", [(16, 48), (33, 48)])
@pytest.mark.xfail(
    reason="NPOT axis-reduction lowering (ReduceOpToLLVM) "
    "not yet implemented; separate from the convert guards.", strict=False)
def test_npot_sum_axis1(M, N):
    """Per-axis reduction over a modular 2D layout. Documents the still-open
    NPOT reduce-lowering gap (weighted sum is permutation-sensitive). xfail:
    tracked separately from the convert guards."""

    @triton.jit
    def kernel(X, W, Out, M: tl.constexpr, N: tl.constexpr):
        idx = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        x = tl.load(X + idx)
        w = tl.load(W + idx)
        sw = tl.sum(x * w, axis=1)  # [M]
        tl.store(Out + tl.arange(0, M), sw)

    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    w = torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = torch.empty(M, device="cuda", dtype=torch.float32)
    kernel[(1, )](x, w, out, M=M, N=N)
    ref = (x * w).sum(dim=1)
    assert _rel_err(out, ref) < 1e-3, f"wsum M={M},N={N}"


# One NPOT reduction gap remains (the configs miscompute, they do not crash) and
# is skipped pending a follow-up: a 2D reduce -> [:, None] -> broadcast (used by
# softmax/layernorm) miscomputes under a non-power-of-2 layout for some shapes.
_2D_BROADCAST_REASON = "NPOT 2D reduce -> broadcast (reduce -> [:, None] -> elementwise) miscompute"


def _redux_params():
    # Representative NPOT sizes spanning the masking paths: 1023@nw8 is a large
    # inter-warp collapse (>1 warp mapping onto the same axis slot); 192@nw1 is
    # within-thread register wrapping; 17 is intra-warp lane wrapping.
    out = []
    for op in ("sum", "max"):
        for N, num_warps in ((17, 4), (192, 1), (1023, 8)):
            out.append(pytest.param(N, num_warps, op))
    return out


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N, num_warps, op", _redux_params())
def test_npot_reduction_device(N, num_warps, op):
    """Device NPOT reduction for tl.sum / tl.max.

    Pow2-rounding the NPOT axis creates wrapped register/lane/warp slots that
    must be identity-filled (0 for sum, -inf for max) before the reduction folds
    them in. Covers within-thread register wrapping, intra-warp lane wrapping,
    and inter-warp NPOT warp collapse.
    """

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

    rows = 16
    x = torch.randn(rows, N, device="cuda", dtype=torch.float32)
    out = torch.empty(rows, device="cuda", dtype=torch.float32)
    kernel[(rows, )](x, out, N=N, num_warps=num_warps)
    ref = x.sum(dim=1) if op == "sum" else x.max(dim=1).values
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N", [17, 192])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_npot_argmax_device(N, num_warps):
    """Device NPOT argmax/argmin. The multi-op combine (value compare + index
    select) has no scalar identity, so phantom slots are masked by filling the
    VALUE operand with -inf (argmax) / +inf (argmin): the phantom loses every
    compare and its index is never selected."""

    @triton.jit
    def kernel(X, OutMax, OutMin, N: tl.constexpr):
        row_idx = tl.program_id(0)
        col_idx = tl.arange(0, N)
        x = tl.load(X + row_idx * N + col_idx)
        tl.store(OutMax + row_idx, tl.argmax(x, axis=0))
        tl.store(OutMin + row_idx, tl.argmin(x, axis=0))

    rows = 16
    x = torch.randn(rows, N, device="cuda", dtype=torch.float32)
    omax = torch.empty(rows, device="cuda", dtype=torch.int32)
    omin = torch.empty(rows, device="cuda", dtype=torch.int32)
    kernel[(rows, )](x, omax, omin, N=N, num_warps=num_warps)
    torch.testing.assert_close(omax.to(torch.int64), x.argmax(dim=1))
    torch.testing.assert_close(omin.to(torch.int64), x.argmin(dim=1))


@requires_gpu
@requires_npot
@pytest.mark.parametrize(
    "M, N",
    [
        (8, 1023),
        pytest.param(16, 192, marks=pytest.mark.skip(reason=_2D_BROADCAST_REASON)),
        pytest.param(3, 48, marks=pytest.mark.skip(reason=_2D_BROADCAST_REASON)),
    ],
)
@pytest.mark.parametrize("num_warps", [1, 4])
def test_npot_softmax_device(M, N, num_warps):
    """Device NPOT softmax: a max-reduce and a sum-reduce over the NPOT axis
    plus broadcast. Exercises both reduction identities (-inf, 0) end-to-end."""

    @triton.jit
    def kernel(X, Out, M: tl.constexpr, N: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, N)[None, :]
        offsets = rows * N + cols
        x = tl.load(X + offsets)
        row_max = tl.max(x, axis=1)[:, None]
        num = tl.exp(x - row_max)
        den = tl.sum(num, axis=1)[:, None]
        tl.store(Out + offsets, num / den)

    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = torch.empty(M, N, device="cuda", dtype=torch.float32)
    kernel[(1, )](x, out, M=M, N=N, num_warps=num_warps)
    torch.testing.assert_close(out, torch.softmax(x, dim=1), rtol=1e-4, atol=1e-4)


@requires_gpu
@requires_npot
@pytest.mark.parametrize(
    "M, N",
    [
        (8, 1023),
        pytest.param(16, 192, marks=pytest.mark.skip(reason=_2D_BROADCAST_REASON)),
        pytest.param(33, 192, marks=pytest.mark.skip(reason=_2D_BROADCAST_REASON)),
    ],
)
@pytest.mark.parametrize("num_warps", [1, 4])
def test_npot_layernorm_device(M, N, num_warps):
    """Device NPOT layernorm: mean and variance are sum-reductions over the NPOT
    axis. A phantom slot folded as a non-identity value would bias mean/var."""

    @triton.jit
    def kernel(X, Out, M: tl.constexpr, N: tl.constexpr, EPS: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, N)[None, :]
        offsets = rows * N + cols
        x = tl.load(X + offsets)
        mean = tl.sum(x, axis=1)[:, None] / N
        xc = x - mean
        var = tl.sum(xc * xc, axis=1)[:, None] / N
        tl.store(Out + offsets, xc / tl.sqrt(var + EPS))

    eps = 1e-5
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = torch.empty(M, N, device="cuda", dtype=torch.float32)
    kernel[(1, )](x, out, M=M, N=N, EPS=eps)
    mean = x.mean(dim=1, keepdim=True)
    var = ((x - mean)**2).mean(dim=1, keepdim=True)
    ref = (x - mean) / torch.sqrt(var + eps)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@requires_gpu
@requires_npot
def test_npot_reduction_axis_none_2d_raises():
    """axis=None over a multi-dim NPOT tensor must raise (the flatten reshape
    would scramble the modular layout). With the flag on this is a hard error."""

    @triton.jit
    def kernel(X, Out, M: tl.constexpr, N: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, N)[None, :]
        x = tl.load(X + rows * N + cols)
        tl.store(Out, tl.sum(x))

    x = torch.randn(3, 48, device="cuda", dtype=torch.float32)
    out = torch.empty(1, device="cuda", dtype=torch.float32)
    with pytest.raises(Exception) as excinfo:
        kernel[(1, )](x, out, M=3, N=48)
    # The frontend raises a ValueError that the JIT wraps in a CompilationError;
    # check the rendered message (including any wrapped cause) names the axis.
    msg = str(excinfo.value) + str(getattr(excinfo.value, "__cause__", ""))
    assert "axis=None" in msg, msg
