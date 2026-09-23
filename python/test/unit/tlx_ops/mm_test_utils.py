"""Shared test support for architecture-specific ``tlx.ops.mm`` suites."""

import time

import pytest
import torch
from triton.tlx.ops import InvalidInput, UnsupportedOp
from triton.tlx.ops.kernels.mm._shapes import CORRECTNESS_SHAPES, operand

torch.manual_seed(0)

MAX_SECONDS_PER_CASE = 60
REL_PRECISION = {torch.float16: 1e-3, torch.bfloat16: 8e-3}


def shapes():
    return list(CORRECTNESS_SHAPES)


def _assert_strides(tensor, wanted):
    """Check that the operand has the recorded layout."""
    for dim, (got, expected) in enumerate(zip(tensor.stride(), wanted)):
        # A dim of extent 1 addresses no additional bytes, so stride 0 and
        # torch's chosen stride describe the same layout.
        if tensor.shape[dim] != 1:
            assert got == expected, f"dim {dim}: stride {got}, recorded {expected}"


def run_mm_case(arch, M, N, K, a_strides, b_strides, dtype_name):
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
        pytest.skip(f"{arch} declines this shape: {declined}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert elapsed < MAX_SECONDS_PER_CASE, (f"mm({M}x{N}x{K}, {dtype}) took {elapsed:.1f}s, "
                                            f"over the {MAX_SECONDS_PER_CASE}s budget")

    ref = torch.matmul(a, b)
    precision = REL_PRECISION[dtype]
    torch.testing.assert_close(out, ref, atol=precision * ref.abs().max().item(), rtol=precision)

    del a, b, out, ref
    torch.cuda.empty_cache()
