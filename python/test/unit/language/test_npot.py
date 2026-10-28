# NPOT (non-power-of-2) language tests, gated on TRITON_ALLOW_NPOT=1 (else they skip).
# A TTIR pass pads each NPOT tensor to the next power of two before TTGIR, keeping pad
# lanes out of memory and live results; an op it cannot pad fails to compile with an error.

import pytest
import torch

import triton
import triton.language as tl
from triton import knobs

from triton._internal_testing import is_cuda, is_hip


def _allow_npot():
    return bool(knobs.language.allow_npot)


requires_npot = pytest.mark.skipif(
    not _allow_npot(),
    reason="requires TRITON_ALLOW_NPOT=1",
)
requires_gpu = pytest.mark.skipif(not (is_cuda() or is_hip()), reason="requires a CUDA or HIP GPU")


def _rel_err(actual, ref, eps=1e-6):
    a = actual.float()
    r = ref.float()
    denom = max(r.abs().max().item(), eps)
    return (a - r).abs().max().item() / denom


@requires_gpu
@requires_npot
@pytest.mark.parametrize("size", [33, 48, 127, 768, 1023, 1536])
def test_npot_arange(size):
    """arange + store over an NPOT extent: a padded make_range and a store
    masked to the live lanes."""

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
    """The copy runs padded: no NPOT tensor reaches TTGIR, and the masked store
    leaves the canary past `size` untouched."""

    @triton.jit
    def kernel(X, Out, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Out + i, tl.load(X + i))

    x = torch.randn(size, device="cuda")
    out = torch.full((triton.next_power_of_2(size), ), -1.0, device="cuda")
    h = kernel[(1, )](x, out, N=size, num_warps=1)
    assert f"tensor<{size}x" not in h.asm["ttgir"]
    assert torch.equal(out[:size], x) and bool((out[size:] == -1.0).all()), f"size={size}"


@requires_gpu
@requires_npot
@pytest.mark.parametrize("M, N", [(16, 17), (16, 48), (8, 17)])
def test_npot_copy_2d(M, N):
    """2D copy with an NPOT inner extent runs padded to the next power of two."""

    @triton.jit
    def kernel(X, Out, M: tl.constexpr, N: tl.constexpr):
        idx = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        tl.store(Out + idx, tl.load(X + idx))

    x = torch.randn(M, N, device="cuda")
    out = torch.empty(M, N, device="cuda")
    kernel[(1, )](x, out, M=M, N=N)
    assert _rel_err(out, x) < 1e-6, f"M={M},N={N}"


@requires_gpu
@requires_npot
@pytest.mark.parametrize("col_major, M, N, dtype, num_warps", [
    (False, 24, 32, torch.float16, 4),
    (False, 24, 32, torch.float32, 8),
    (True, 64, 24, torch.int8, 4),
], ids=["rm_24x32_f16_nw4", "rm_24x32_f32_nw8", "cm_64x24_i8_nw4"])
def test_npot_coalesce_pow2_size_per_thread(col_major, M, N, dtype, num_warps):
    """z starts as a sentinel so unwritten elements are counted. The NPOT dim is
    padded before Coalesce, so this checks the masked padded store."""

    @triton.jit
    def kernel(X, Y, Z, s, M: tl.constexpr, N: tl.constexpr, COL_MAJOR: tl.constexpr):
        r = tl.arange(0, M)[:, None]
        c = tl.arange(0, N)[None, :]
        if COL_MAJOR:
            off = r + c * s
        else:
            off = r * s + c
        tl.store(Z + off, tl.load(X + off) + tl.load(Y + off))

    g = torch.Generator(device="cuda").manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(M * N, device="cuda", dtype=dtype, generator=g)
        y = torch.randn(M * N, device="cuda", dtype=dtype, generator=g)
        sentinel = -77.0
    else:
        x = torch.randint(-128, 128, (M * N, ), device="cuda", dtype=dtype, generator=g)
        y = torch.randint(-128, 128, (M * N, ), device="cuda", dtype=dtype, generator=g)
        sentinel = 77
    z = torch.full((M * N, ), sentinel, device="cuda", dtype=dtype)
    ref = x + y
    kernel[(1, )](x, y, z, M if col_major else N, M=M, N=N, COL_MAJOR=col_major, num_warps=num_warps)
    unwritten = int(((z == sentinel) & (ref != sentinel)).sum().item())
    wrong = int((z != ref).sum().item())
    assert unwritten == 0 and wrong == 0, (f"{M}x{N} {dtype} nw{num_warps}: {wrong}/{M * N} wrong, "
                                           f"{unwritten} unwritten")


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


@requires_gpu
@requires_npot
@pytest.mark.parametrize("M, N", [(3, 32), (32, 3), (16, 32)])
def test_npot_reshape(M, N):
    """A padded reshape keeps row-major order: the live elements stay a prefix
    when every NPOT dim leads its group ([3,32] <-> [96]), else a gather moves
    them ([32,3] <-> [96]). (16, 32) is the pow2 control."""

    @triton.jit
    def kernel(X, Out, Back, M: tl.constexpr, N: tl.constexpr):
        idx = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        flat = tl.arange(0, M * N)
        tl.store(Out + flat, tl.reshape(tl.load(X + idx), (M * N, )))
        tl.store(Back + idx, tl.reshape(tl.load(X + flat), (M, N)))

    x = torch.randn(M, N, device="cuda")
    out, back = torch.empty(M * N, device="cuda"), torch.empty_like(x)
    h = kernel[(1, )](x, out, back, M=M, N=N)
    assert ("tt.gather" in h.asm["ttgir"]) == (N == 3)
    assert torch.equal(out, x.reshape(-1)) and torch.equal(back, x), f"M={M},N={N}"


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
