# NPOT (non-power-of-2) language tests, gated on TRITON_ALLOW_NPOT=1 (else they skip).
# The NPOT tensors connected through their ops (a connected component of the def-use graph) keep
# their shape if their ops lower a mixed-radix layout, each memory access keeps padding's vector and
# the connected component takes a smaller total in-dim size in sum; a TTIR pass pads every other NPOT tensor or fails.

import re

import pytest
import torch

import triton
import triton.language as tl
from triton import knobs

from triton._internal_testing import (is_cuda, is_hip, is_hip_cdna3, is_hip_cdna4, is_hip_gfx1250, is_hip_rdna3,
                                      is_hip_rdna4, is_hip_rdna4m)
from triton.compiler.errors import CompilationError


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
    """arange + store over an NPOT extent."""

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
@pytest.mark.parametrize("size", [48, 96])
def test_npot_copy_one_warp(size):
    """The pad pass pads the copy (96 would lose padded's vector width), and
    the store leaves the canary past `size` untouched."""

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
@pytest.mark.parametrize("N, num_warps", [(768, 1), (3072, 4), (8448, 1)])
@pytest.mark.parametrize("op", ["load_store", "make_range", "splat", "elementwise"])
def test_npot_op_class(op, N, num_warps):
    """Each op class keeps its NPOT shape and matches torch and, bitwise, its
    padded counterpart, whose inner_tree scan is where the padded tensor's pad elements begin."""

    @triton.jit
    def kernel(X, S, Out, Scan, N: tl.constexpr, OP: tl.constexpr, SCAN: tl.constexpr):
        i = tl.arange(0, N)
        if OP == "load_store":
            v = tl.load(X + i, mask=i % 3 != 0, other=-1.0)
        elif OP == "make_range":
            v = i.to(tl.float32)
        elif OP == "splat":
            v = tl.zeros([N], tl.float32) + tl.load(S)
        else:
            x = tl.load(X + i)
            v = tl.where(x > 0, tl.exp(x), x.to(tl.int32).to(tl.float32))
        tl.store(Out + i, v)
        if SCAN:
            tl.store(Scan + i, tl.cumsum(v, 0, reduction_ordering=tl.ReductionOrdering.INNER_TREE))

    x, s = torch.randn(N, device="cuda"), torch.randn(1, device="cuda")
    i = torch.arange(N, device="cuda")
    ref = {
        "load_store": torch.where(i % 3 != 0, x, -1.0),
        "make_range": i.float(),
        "splat": s.expand(N),
        "elementwise": torch.where(x > 0, x.exp(),
                                   x.int().float()),
    }[op]
    outs = []
    for scan in (False, True):
        out, cum = torch.empty_like(x), torch.empty_like(x)
        h = kernel[(1, )](x, s, out, cum, N=N, OP=op, SCAN=scan, num_warps=num_warps)
        assert (f"tensor<{N}x" in h.asm["ttgir"]) != scan
        outs.append(out)
    assert torch.equal(outs[0], outs[1]) and _rel_err(outs[0], ref) < 1e-5, f"{op} N={N}"


@requires_gpu
@requires_npot
def test_npot_coalesced_copy():
    """An exact 768-element one-warp copy loads with the 1024 control's 128-bit
    vector width, the odd 3 outermost."""

    @triton.jit
    def kernel(X, Out, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Out + i, tl.load(X + i))

    def widths(h):
        if is_hip():
            return set(re.findall(r"(?:global|buffer)_load_dword\w*", h.asm["amdgcn"]))
        return set(re.findall(r"ld\.global\S*", h.asm["ptx"]))

    x, y = torch.randn(768, device="cuda"), torch.randn(1024, device="cuda")
    out = torch.empty_like(x)
    h = kernel[(1, )](x, out, N=768, num_warps=1)
    control = kernel[(1, )](y, torch.empty_like(y), N=1024, num_warps=1)
    assert "tensor<768x" in h.asm["ttgir"]
    assert widths(h) == widths(control) and all(re.search(r"(v4\.b32|dwordx4)$", w) for w in widths(h))
    assert torch.equal(out, x)


@requires_gpu
@requires_npot
@pytest.mark.parametrize("offset", [1, 0])
def test_npot_unaligned_copy(offset):
    """x[1:] is only 4-byte aligned, so padding's 512 copy loads v1 too and the
    384 copy over 128 threads, in a smaller total in-dim size, keeps its shape. The aligned
    control pads: padding loads v4 where 384 gets v1."""

    @triton.jit
    def kernel(X, Out, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Out + i, tl.load(X + i))

    x, out = torch.randn(385, device="cuda"), torch.full((385, ), -1.0, device="cuda")
    warps = 128 // triton.runtime.driver.active.get_current_target().warp_size
    h = kernel[(1, )](x[offset:], out[offset:], N=384, num_warps=warps)
    ref = torch.full_like(out, -1.0)
    ref[offset:offset + 384] = x[offset:offset + 384]
    assert ("tensor<384x" in h.asm["ttgir"]) == (offset == 1)
    assert torch.equal(out, ref), f"offset={offset}"


@requires_gpu
@requires_npot
def test_npot_padded_meets_exact():
    """The sum of a padded 48-element load is splat into a 768-element connected
    component that keeps its shape."""

    @triton.jit
    def kernel(X, Y, Out, M: tl.constexpr, N: tl.constexpr):
        s = tl.sum(tl.load(X + tl.arange(0, M)))
        i = tl.arange(0, N)
        tl.store(Out + i, tl.load(Y + i) * s)

    x = torch.randn(48, device="cuda")
    y = torch.randn(768, device="cuda")
    out = torch.empty(768, device="cuda")
    h = kernel[(1, )](x, y, out, M=48, N=768, num_warps=1)
    assert "tensor<64x" in h.asm["ttgir"] and "tensor<768x" in h.asm["ttgir"]
    assert _rel_err(out, y * x.sum()) < 1e-5


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
@pytest.mark.parametrize("M, N, trans", [(96, 64, False), (64, 96, False), (96, 64, True), (64, 64, False)])
def test_npot_tile(M, N, trans):
    """A 2-D tile addressed through expand_dims and broadcast keeps its NPOT
    shape on four warps, also through a transpose. (64, 64) is the pow2 control."""

    @triton.jit
    def kernel(X, Out, M: tl.constexpr, N: tl.constexpr, TRANS: tl.constexpr):
        m = tl.arange(0, M)
        n = tl.arange(0, N)
        x = tl.load(X + m[:, None] * N + n[None, :])
        if TRANS:
            tl.store(Out + n[:, None] * M + m[None, :], tl.trans(x))
        else:
            tl.store(Out + m[:, None] * N + n[None, :], x)

    x = torch.randn(M, N, device="cuda")
    out = torch.empty(N, M, device="cuda") if trans else torch.empty_like(x)
    h = kernel[(1, )](x, out, M=M, N=N, TRANS=trans, num_warps=4)
    assert f"tensor<{M}x{N}x" in h.asm["ttgir"]
    assert torch.equal(out, x.T if trans else x), f"M={M},N={N},trans={trans}"


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
@pytest.mark.parametrize("M, N, axis, num_warps, op", [
    (96, 80, 1, 4, "cumsum"),
    (96, 80, 1, 4, "reverse"),
    (96, 80, 1, 4, "argmax"),
    (96, 80, 0, 4, "cumsum"),
    (96, 80, 0, 4, "argmax"),
    (64, 192, 0, 8, "cumsum"),
    (64, 192, 0, 8, "reverse"),
    (32, 128, 1, 4, "cumsum"),
])
def test_npot_scan(M, N, axis, num_warps, op):
    """A scan keeps a mixed-radix layout and matches torch: [96, 80] along both dims, [64, 192] carrying across
    eight warps, argmax as a two-operand scan, [32, 128] as the pow2 control. Each shape stays: its exact
    accesses, narrower on a 64-lane warp, are no more than the padded tile's."""

    @triton.jit
    def _argmax(a, i, b, j):
        return tl.where(b > a, b, a), tl.where(b > a, j, i)

    @triton.jit
    def kernel(X, Out, Idx, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr, OP: tl.constexpr):
        off = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        x = tl.load(X + off)
        if OP == "cumsum":
            y = tl.cumsum(x, AXIS)
        elif OP == "reverse":
            y = tl.cumsum(x, AXIS, reverse=True)
        else:
            y, i = tl.associative_scan((x, off), AXIS, _argmax)
            tl.store(Idx + off, i)
        tl.store(Out + off, y)

    torch.manual_seed(0)
    x = torch.randperm(M * N, device="cuda", dtype=torch.int32).view(M, N) - M * N // 2
    out, idx = torch.empty_like(x), torch.zeros_like(x)
    h = kernel[(1, )](x, out, idx, M=M, N=N, AXIS=axis, OP=op, num_warps=num_warps)
    assert f"tensor<{M}x{N}x" in h.asm["ttgir"]
    if op == "argmax":
        ref = x.cummax(axis)
        off = torch.arange(M * N, device="cuda", dtype=torch.int32).view(M, N)
        assert torch.equal(out, ref.values) and torch.equal(idx, off.gather(axis, ref.indices))
    else:
        flip = x.flip(axis) if op == "reverse" else x
        ref = flip.cumsum(axis, dtype=torch.int32)
        assert torch.equal(out, ref.flip(axis) if op == "reverse" else ref)


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N, num_warps, kept", [(768, 1, True), (96, 4, False), (128, 4, True)])
def test_npot_histogram(N, num_warps, kept):
    """[768] on one warp keeps its shape. [96] on four warps pads: pad lanes
    load 0, a bin-0 canary, and the live mask keeps them uncounted. N=128 is
    the pow2 control."""

    @triton.jit
    def kernel(X, Out, N: tl.constexpr, B: tl.constexpr):
        tl.store(Out + tl.arange(0, B), tl.histogram(tl.load(X + tl.arange(0, N)), B))

    x = torch.randint(1, 16, (N, ), device="cuda", dtype=torch.int32)
    out = torch.empty(16, device="cuda", dtype=torch.int32)
    h = kernel[(1, )](x, out, N=N, B=16, num_warps=num_warps)
    assert (f"tensor<{N}x" in h.asm["ttgir"]) == kept
    assert torch.equal(out, torch.bincount(x, minlength=16).to(torch.int32))


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N, num_warps, kept", [(768, 1, True), (96, 4, False)])
def test_npot_loop_carried(N, num_warps, kept):
    """A loop-carried sum keeps its NPOT shape unless the pad rule pads it."""

    @triton.jit
    def kernel(X, Out, T: tl.constexpr, N: tl.constexpr):
        i = tl.arange(0, N)
        acc = tl.zeros([N], dtype=tl.int32)
        for t in range(T):
            acc += tl.load(X + t * N + i)
        tl.store(Out + i, acc)

    x = torch.randint(-1000, 1000, (4, N), device="cuda", dtype=torch.int32)
    out = torch.empty(N, device="cuda", dtype=torch.int32)
    h = kernel[(1, )](x, out, T=4, N=N, num_warps=num_warps)
    assert (f"tensor<{N}x" in h.asm["ttgir"]) == kept
    assert torch.equal(out, x.sum(0, dtype=torch.int32))


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N, num_warps, kept", [(768, 1, True), (96, 4, False), (128, 4, True)])
def test_npot_gather(N, num_warps, kept):
    """Reverse a vector with tl.gather: exact at 768 on one warp, padded at 96
    on four, where idx = N-1-i is negative in the pad lanes, which gather index
    0 instead. N=128 is the pow2 control."""

    @triton.jit
    def kernel(X, Out, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Out + i, tl.gather(tl.load(X + i), N - 1 - i, 0))

    P = triton.next_power_of_2(N)
    x = torch.full((P, ), float("nan"), device="cuda")
    x[:N] = torch.randn(N, device="cuda")
    out = torch.full((P, ), -1.0, device="cuda")
    h = kernel[(1, )](x, out, N=N, num_warps=num_warps)
    assert (f"tensor<{N}x" in h.asm["ttgir"]) == kept
    assert torch.equal(out[:N], x[:N].flip(0)) and bool((out[N:] == -1.0).all())


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N", [96, 128])
def test_npot_atomics(N):
    """atomic_add masks the pad lanes, so the canary past N is untouched;
    eight programs race an atomic_cas on each [32, N] element and one wins."""

    @triton.jit
    def add(Z, N: tl.constexpr):
        tl.atomic_add(Z + tl.arange(0, N), 1.0)

    @triton.jit
    def cas(Z, W, N: tl.constexpr):
        i = tl.reshape(tl.arange(0, 32 * N), (32, N))
        old = tl.atomic_cas(Z + i, i * 0, i * 0 + tl.program_id(0) + 1)
        tl.atomic_add(W + i, (old == 0).to(tl.int32))

    z = torch.zeros(128, device="cuda")
    z[N:] = -1.0
    add[(1, )](z, N=N)
    assert bool((z[:N] == 1.0).all()) and bool((z[N:] == -1.0).all())
    c = torch.zeros(32 * N, device="cuda", dtype=torch.int32)
    w = torch.zeros(32 * N, device="cuda", dtype=torch.int32)
    cas[(8, )](c, w, N=N)
    assert bool((w == 1).all()) and bool(((c >= 1) & (c <= 8)).all())


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


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N", [96, 128])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.uint8])
def test_npot_sort(dtype, descending, N):
    """sort and topk pad an NPOT dim with the value that sorts last, so the
    extremes of the dtype (+-inf for floats) in live lanes stay in place. N=128
    is the pow2 control."""

    @triton.jit
    def kernel(X, Out, Top, N: tl.constexpr, DESC: tl.constexpr):
        x = tl.load(X + tl.arange(0, N))
        tl.store(Out + tl.arange(0, N), tl.sort(x, descending=DESC))
        tl.store(Top + tl.arange(0, 32), tl.topk(x, 32, descending=DESC))

    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(N, device="cuda")
        lo, hi = float("-inf"), float("inf")
    else:
        lo, hi = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        x = torch.randint(lo, hi, (N, ), device="cuda", dtype=dtype)
    x[5], x[7] = hi, lo
    out, top = torch.empty_like(x), torch.empty(32, device="cuda", dtype=dtype)
    kernel[(1, )](x, out, top, N=N, DESC=descending)
    ref = torch.sort(x, descending=descending).values
    assert torch.equal(out, ref) and torch.equal(top, ref[:32])


@requires_gpu
@requires_npot
@pytest.mark.parametrize("M", [3, 4])
def test_npot_sort_other_dim(M):
    """Sorting a tensor with an NPOT dim other than the sorted one fails to
    compile with an error, raised inside sort, so it is in the exception's cause
    chain. M=4 is the pow2 control."""

    @triton.jit
    def kernel(X, Out, M: tl.constexpr):
        idx = tl.arange(0, M)[:, None] * 32 + tl.arange(0, 32)[None, :]
        tl.store(Out + idx, tl.sort(tl.load(X + idx)))

    x = torch.randn(M, 32, device="cuda")
    out = torch.empty_like(x)
    if M & (M - 1):
        with pytest.raises(CompilationError) as e:
            kernel[(1, )](x, out, M=M)
        assert "sort over an NPOT dim other than the sorted one is unsupported" in str(e.getrepr())
    else:
        kernel[(1, )](x, out, M=M)
        assert torch.equal(out, torch.sort(x).values)


@requires_gpu
@requires_npot
@pytest.mark.parametrize("N", [96, 128])
def test_npot_reduce_no_pad_value_rejected(N):
    """A subtract combiner has no pad value, so an NPOT reduce over it fails to
    compile with an error. N=128 is the pow2 control and compiles."""

    @triton.jit
    def sub(a, b):
        return a - b

    @triton.jit
    def kernel(X, Out, N: tl.constexpr):
        tl.store(Out, tl.reduce(tl.load(X + tl.arange(0, N)), 0, sub))

    x = torch.ones(N, device="cuda")
    out = torch.empty(1, device="cuda")
    if N == 128:
        kernel[(1, )](x, out, N=N)
        return
    with pytest.raises(RuntimeError,
                       match="'tt.reduce' op over an NPOT axis with a combiner that has no pad value is unsupported"):
        kernel[(1, )](x, out, N=N)


@requires_gpu
@requires_npot
def test_npot_reduce_no_pad_value_exact():
    """The [32, 96] counterpart keeps its shape on four warps, so no pad element enters
    the reduce and a combiner without a pad value compiles."""

    @triton.jit
    def sub(a, b):
        return a - b

    @triton.jit
    def kernel(X, Out):
        m = tl.arange(0, 32)
        x = tl.load(X + m[:, None] * 96 + tl.arange(0, 96)[None, :])
        tl.store(Out + m, tl.reduce(x, 1, sub))

    x = torch.zeros(32, 96, device="cuda")
    out = torch.ones(32, device="cuda")
    h = kernel[(1, )](x, out, num_warps=4)
    assert "tensor<32x96x" in h.asm["ttgir"]
    assert torch.equal(out, torch.zeros_like(out))
