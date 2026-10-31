# NPOT (non-power-of-2) language tests, gated on TRITON_ALLOW_NPOT=1 (else they skip).
# A TTIR pass pads each NPOT tensor to the next power of two before TTGIR, keeping pad
# lanes out of memory and live results; an op it cannot pad fails to compile with an error.

import pytest
import torch

import triton
import triton.language as tl
from triton import knobs

from triton._internal_testing import (is_cuda, is_hip, is_hip_cdna3, is_hip_cdna4, is_hip_gfx1250, is_hip_rdna3,
                                      is_hip_rdna4, is_hip_rdna4m)


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
@pytest.mark.parametrize("M, N", [(3, 32), (4, 32)])
def test_npot_sum_axis_none(M, N):
    """axis=None flattens with an allow_reorder reshape, which pads as an ordered
    one. The +1 makes loaded pad lanes nonzero. (4, 32) is the pow2 control."""

    @triton.jit
    def kernel(X, Out, M: tl.constexpr, N: tl.constexpr):
        idx = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        tl.store(Out, tl.sum(tl.load(X + idx) + 1))

    torch.manual_seed(0)
    x = torch.randint(-50, 50, (M, N), device="cuda", dtype=torch.int32)
    out = torch.empty(1, device="cuda", dtype=torch.int32)
    kernel[(1, )](x, out, M=M, N=N)
    assert out.item() == (x + 1).sum().item(), f"M={M},N={N}"


@requires_gpu
@requires_npot
@pytest.mark.parametrize("M, N", [(16, 48), (33, 48)])
def test_npot_sum_axis1(M, N):
    """A sum over an NPOT axis: pad lanes hold the identity, -0.0."""

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


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N", [96, 128])
def test_npot_reduce_two_sums(N):
    """Two sums in one reduce pad each input with its own identity. The +1 makes
    loaded pad lanes nonzero. N=128 is the pow2 control."""

    @triton.jit
    def _sum2(a0, a1, b0, b1):
        return a0 + b0, a1 + b1

    @triton.jit
    def kernel(X, Y, Out, N: tl.constexpr):
        i = tl.arange(0, N)
        s, t = tl.reduce((tl.load(X + i) + 1, tl.load(Y + i) + 1), 0, _sum2)
        tl.store(Out, s)
        tl.store(Out + 1, t)

    torch.manual_seed(0)
    x, y = torch.randint(-50, 50, (2, N), device="cuda", dtype=torch.int32)
    out = torch.empty(2, device="cuda", dtype=torch.int32)
    kernel[(1, )](x, y, out, N=N)
    assert out.tolist() == [(x + 1).sum().item(), (y + 1).sum().item()], f"N={N}"


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N", [96, 128])
@pytest.mark.parametrize("op", ["argmax", "argmin"])
def test_npot_argmax_nan(op, N):
    """argmax/argmin pad value and index with a copy of the last live lane, as
    every live value loses to a pad 0. With a NaN the result depends on the
    reduction tree even at pow2, so only its range is checked."""

    @triton.jit
    def kernel(X, Out, N: tl.constexpr, MAX: tl.constexpr):
        x = tl.load(X + tl.arange(0, N))
        tl.store(Out, tl.argmax(x, 0) if MAX else tl.argmin(x, 0))

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(N, device="cuda", generator=g).abs() + 1
    x = -x if op == "argmax" else x
    out = torch.empty(2, device="cuda", dtype=torch.int32)
    kernel[(1, )](x, out, N=N, MAX=op == "argmax")
    ref = getattr(torch, op)(x).item()
    x[37] = float("nan")
    kernel[(1, )](x, out[1:], N=N, MAX=op == "argmax")
    idx, nan_idx = out.tolist()
    assert idx == ref and 0 <= nan_idx < N, f"{op} N={N}: {idx} vs {ref}, NaN {nan_idx}"


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N", [96, 128])
@pytest.mark.parametrize("op", ["sum", "max"])
def test_npot_reverse_scan(op, N):
    """A reverse scan reads the pad lanes first: a sum pads with 0, a max with
    INT_MIN. The +1 makes loaded pad lanes nonzero, and x[N-1] is the smallest
    value, so a pad above x[N-1] would win the last max."""

    @triton.jit
    def _max(a, b):
        return tl.maximum(a, b)

    @triton.jit
    def kernel(X, Out, N: tl.constexpr, SUM: tl.constexpr):
        i = tl.arange(0, N)
        x = tl.load(X + i) + 1
        if SUM:
            y = tl.cumsum(x, 0, reverse=True)
        else:
            y = tl.associative_scan(x, 0, _max, reverse=True)
        tl.store(Out + i, y)

    torch.manual_seed(0)
    x = torch.randint(-50, 50, (N, ), device="cuda", dtype=torch.int32)
    x[-1] = -100
    out = torch.empty_like(x)
    kernel[(1, )](x, out, N=N, SUM=op == "sum")
    flip = (x + 1).flip(0)
    ref = (flip.cumsum(0) if op == "sum" else flip.cummax(0).values).flip(0).to(torch.int32)
    assert torch.equal(out, ref), f"{op} N={N}"


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N", [96, 128])
def test_npot_histogram(N):
    """Pad lanes load 0, a bin-0 canary; the live mask keeps them uncounted."""

    @triton.jit
    def kernel(X, Out, N: tl.constexpr, B: tl.constexpr):
        tl.store(Out + tl.arange(0, B), tl.histogram(tl.load(X + tl.arange(0, N)), B))

    x = torch.randint(1, 16, (N, ), device="cuda", dtype=torch.int32)
    out = torch.empty(16, device="cuda", dtype=torch.int32)
    kernel[(1, )](x, out, N=N, B=16)
    assert torch.equal(out, torch.bincount(x, minlength=16).to(torch.int32))


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N", [96, 128])
def test_npot_gather(N):
    """Reverse a vector with tl.gather; idx = N-1-i is negative in the pad lanes,
    which gather index 0 instead."""

    @triton.jit
    def kernel(X, Out, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Out + i, tl.gather(tl.load(X + i), N - 1 - i, 0))

    x = torch.full((128, ), float("nan"), device="cuda")
    x[:N] = torch.randn(N, device="cuda")
    out = torch.full((128, ), -1.0, device="cuda")
    kernel[(1, )](x, out, N=N)
    assert torch.equal(out[:N], x[:N].flip(0)) and bool((out[N:] == -1.0).all())


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N", [96, 128])
def test_npot_atomics(N):
    """atomic_add masks the pad lanes, so the canary past N is untouched;
    atomic_cas has no mask and fails to compile with an error at N=96."""

    @triton.jit
    def add(Z, N: tl.constexpr):
        tl.atomic_add(Z + tl.arange(0, N), 1.0)

    @triton.jit
    def cas(Z, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.atomic_cas(Z + i, i, i + 1)

    z = torch.zeros(128, device="cuda")
    z[N:] = -1.0
    add[(1, )](z, N=N)
    assert bool((z[:N] == 1.0).all()) and bool((z[N:] == -1.0).all())
    c = torch.arange(N, device="cuda", dtype=torch.int32)
    if N == 96:
        with pytest.raises(RuntimeError, match="'tt.atomic_cas' op on an NPOT tensor is unsupported"):
            cas[(1, )](c, N=N)
    else:
        cas[(1, )](c, N=N)
        assert torch.equal(c, torch.arange(1, N + 1, device="cuda", dtype=torch.int32))


@requires_gpu
@requires_npot
@pytest.mark.parametrize("M, N, K", [(32, 32, 96), (48, 80, 32), (32, 32, 128)])
def test_npot_dot(M, N, K):
    """A padded K is zero-filled on both operands; the +1 makes the loaded pad
    lanes nonzero. A padded M or N only adds rows and columns that the store
    masks. (32, 32, 128) is the pow2 control."""

    @triton.jit
    def kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        k = tl.arange(0, K)
        a = tl.load(A + m[:, None] * K + k[None, :]) + 1.0
        b = tl.load(B + k[:, None] * N + n[None, :]) + 1.0
        tl.store(C + m[:, None] * N + n[None, :], tl.dot(a, b))

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.full((M, N), float("nan"), device="cuda")
    kernel[(1, )](a, b, c, M=M, N=N, K=K)
    torch.testing.assert_close(c, ((a + 1).double() @ (b + 1).double()).float(), atol=1e-2, rtol=1e-2)


@requires_gpu
@requires_npot
@pytest.mark.parametrize("K", [96, 128])
def test_npot_dot_scaled(K):
    """mxfp8 with K=96: scale pad lanes take 127 (1.0). The kernel inverts the
    stored scales, so a loaded pad lane would be 0xFF (NaN). K=128 is the pow2
    control."""
    if is_cuda():
        cc = torch.cuda.get_device_capability()
        if cc < (8, 9):
            pytest.skip("float8e4nv not supported on CUDA < 8.9")
    if is_hip() and not (is_hip_cdna3() or is_hip_cdna4() or is_hip_rdna3() or is_hip_rdna4m() or is_hip_rdna4()
                         or is_hip_gfx1250()):
        pytest.skip("scaled_dot(e4m3, e4m3) only implemented for CDNA3, CDNA4, RDNA3, RDNA4, and above")

    @triton.jit
    def kernel(A, SA, B, SB, C, K: tl.constexpr):
        m = tl.arange(0, 32)
        k = tl.arange(0, K)
        s = tl.arange(0, K // 32)
        a = tl.load(A + m[:, None] * K + k[None, :])
        b = tl.load(B + k[:, None] * 32 + m[None, :])
        sa = ~tl.load(SA + m[:, None] * (K // 32) + s[None, :])
        sb = ~tl.load(SB + m[:, None] * (K // 32) + s[None, :])
        tl.store(C + m[:, None] * 32 + m[None, :], tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3"))

    torch.manual_seed(0)
    a = torch.randn(32, K, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(K, 32, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.randint(120, 134, (32, K // 32), device="cuda", dtype=torch.uint8)
    sb = torch.randint(120, 134, (32, K // 32), device="cuda", dtype=torch.uint8)
    c = torch.empty(32, 32, device="cuda")
    kernel[(1, )](a, ~sa, b, ~sb, c, K=K)
    if K != 128:
        # The NPOT claim is equality with padding: the same kernel on the inputs
        # zero-extended to K=128 (pad scale 127 is 1.0) gives the same bits.
        F, n = torch.nn.functional, 128 - K
        a2, b2 = F.pad(a.float(), (0, n)).to(a.dtype), F.pad(b.float(), (0, 0, 0, n)).to(b.dtype)
        sa2, sb2 = F.pad(sa, (0, n // 32), value=127), F.pad(sb, (0, n // 32), value=127)
        c2 = torch.empty_like(c)
        kernel[(1, )](a2, ~sa2, b2, ~sb2, c2, K=128)
        torch.testing.assert_close(c, c2, atol=0, rtol=0)
        return
    scale = lambda s: torch.exp2(s.double() - 127).repeat_interleave(32, dim=1)
    ref = (a.double() * scale(sa)) @ (b.double() * scale(sb).T)
    # On HIP, test_core.py test_scaled_dot's tolerance for e4m3.
    atol, rtol = (1e-5, 1e-2) if is_hip() else (1e-3, 1e-3)
    torch.testing.assert_close(c, ref.float(), atol=atol, rtol=rtol)
