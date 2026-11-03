"""Shared test support for architecture-specific ``tlx.ops.mm`` suites."""

import time

import pytest
import torch
from triton.tlx.ops import InvalidInput, UnsupportedOp
from triton.tlx.ops.kernels.mm._shapes import CORRECTNESS_SHAPES, operand

torch.manual_seed(0)

MAX_SECONDS_PER_CASE = 60
REL_PRECISION = {torch.float16: 1e-3, torch.bfloat16: 8e-3}
REFERENCE_ROW_CHUNK = 1 << 16


def shapes():
    return list(CORRECTNESS_SHAPES)


def _assert_strides(tensor, wanted):
    """Check that the operand has the recorded layout."""
    for dim, (got, expected) in enumerate(zip(tensor.stride(), wanted)):
        # A dim of extent 1 addresses no additional bytes, so stride 0 and
        # torch's chosen stride describe the same layout.
        if tensor.shape[dim] != 1:
            assert got == expected, f"dim {dim}: stride {got}, recorded {expected}"


def _reference_mm(a, b, arch):
    if arch != "gfx942" or a.shape[0] <= REFERENCE_ROW_CHUNK:
        return torch.matmul(a, b)
    ref = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    for start in range(0, a.shape[0], REFERENCE_ROW_CHUNK):
        ref[start:start + REFERENCE_ROW_CHUNK] = torch.matmul(a[start:start + REFERENCE_ROW_CHUNK], b)
    return ref


def run_mm_invariance_case(
    arch,
    M,
    N,
    K,
    a_strides,
    b_strides,
    dtype_name,
    spec,
):
    """Check per-dimension bitwise invariance under `spec` + torch-closeness.

    Each dim named by `spec` is varied in turn: M and N by slicing the
    operands, K by appending a zero-padded tail (mathematically a no-op
    that must not change the reduction schedule). `all()` covers every
    dimension. Unlike `run_mm_case`, declined shapes fail loudly: the spec
    must work on every listed shape.
    """
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    from triton.tlx.ops import Space, mm as tlx_mm

    a = operand(M, K, a_strides, dtype)
    b = operand(K, N, b_strides, dtype)
    _assert_strides(a, a_strides)
    _assert_strides(b, b_strides)

    dims = {"M", "N", "K"} if spec.all_dims else set(spec.invariant_dims)
    space = Space("heuristic", invariance=spec)
    full = tlx_mm(a, b, space=space)

    if "M" in dims:
        sub = max(M // 2, 1)
        rerun = tlx_mm(a[:sub], b, space=space)
        assert torch.equal(full[:sub], rerun), (f"mm({M}x{N}x{K}, {dtype}) with {spec} is not "
                                                "batch invariant: rows differ between full-M and sliced runs")
    if "N" in dims:
        sub_n = max(N // 2, 1)
        # A strided slice is neither row- nor column-major, which the op
        # legitimately rejects; materialize the same columns contiguously.
        rerun = tlx_mm(a, b[:, :sub_n].contiguous(), space=space)
        assert torch.equal(full[:, :sub_n], rerun), (f"mm({M}x{N}x{K}, {dtype}) with {spec} is not "
                                                     "column invariant: columns differ between full-N and sliced runs")
    if "K" in dims:
        pad = 64
        a_pad = torch.zeros((M, K + pad), device=a.device, dtype=dtype)
        a_pad[:, :K] = a
        b_pad = torch.zeros((K + pad, N), device=b.device, dtype=dtype)
        b_pad[:K, :] = b
        rerun = tlx_mm(a_pad, b_pad, space=space)
        assert torch.equal(full, rerun), (f"mm({M}x{N}x{K}, {dtype}) with {spec} is not "
                                          "reduction invariant: outputs differ with zero-padded K tail")

    ref = _reference_mm(a, b, arch)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(full, ref, atol=precision * ref.abs().max().item(), rtol=precision)

    torch.cuda.empty_cache()


def run_mm_case(
    arch,
    M,
    N,
    K,
    a_strides,
    b_strides,
    dtype_name,
    *,
    allow_decline=True,
):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    from triton.tlx.ops import mm as tlx_mm

    a = operand(M, K, a_strides, dtype)
    b = operand(K, N, b_strides, dtype)
    _assert_strides(a, a_strides)
    _assert_strides(b, b_strides)

    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        out = tlx_mm(a, b, space="heuristic")
    except (InvalidInput, UnsupportedOp) as declined:
        if not allow_decline:
            raise
        pytest.skip(f"{arch} declines this shape: {declined}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert elapsed < MAX_SECONDS_PER_CASE, (f"mm({M}x{N}x{K}, {dtype}) took {elapsed:.1f}s, "
                                            f"over the {MAX_SECONDS_PER_CASE}s budget")

    ref = _reference_mm(a, b, arch)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)

    del a, b, out, ref
    torch.cuda.empty_cache()


def cpu_tn_inputs(m, n, k):
    """CPU fp16 TN operands for launch-logic tests (no kernel runs)."""
    a = torch.ones((m, k), dtype=torch.float16)
    storage = torch.ones((n * (n - 1) + k, ), dtype=torch.float16)
    return a, storage.as_strided((k, n), (1, n))


class GridRecorder:
    """Stand-in for a ``@triton.jit`` kernel capturing launch grids."""

    def __init__(self):
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(grid)
        return lambda *args, **kwargs: None
