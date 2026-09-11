"""gfx950 correctness coverage for ``tlx.ops.mm``."""

import time

import pytest
import torch
from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import InvalidInput, UnsupportedOp
from triton.tlx.ops.kernels.mm._shapes import GFX950_FOCUS, operand


pytestmark = pytest.mark.skipif(
    not is_hip_cdna4(), reason="Requires gfx950"
)

MAX_SECONDS_PER_CASE = 60


def _assert_strides(tensor, wanted):
    for dim, (got, expected) in enumerate(zip(tensor.stride(), wanted)):
        if tensor.shape[dim] != 1:
            assert got == expected, (
                f"dim {dim}: stride {got}, recorded {expected}"
            )


@pytest.mark.parametrize(
    "m,n,k,a_strides,b_strides,dtype_name", GFX950_FOCUS
)
def test_mm(m, n, k, a_strides, b_strides, dtype_name):
    dtype = {"fp16": torch.float16}[dtype_name]
    from triton.tlx.ops import mm as tlx_mm

    a = operand(m, k, a_strides, dtype)
    b = operand(k, n, b_strides, dtype)
    _assert_strides(a, a_strides)
    _assert_strides(b, b_strides)

    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        out = tlx_mm(a, b, arch="gfx950", space="heuristic")
    except (InvalidInput, UnsupportedOp) as declined:
        pytest.fail(f"gfx950 declines its focus shape: {declined}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert elapsed < MAX_SECONDS_PER_CASE, (
        f"mm({m}x{n}x{k}, {dtype}) took {elapsed:.1f}s, "
        f"over the {MAX_SECONDS_PER_CASE}s budget"
    )

    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        out,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "m,n,k,dtype",
    [
        (279, 256, 4096, torch.float16),
        (1024, 4096, 800, torch.bfloat16),
    ],
    ids=["intermediate-fp16", "full-grid-bf16"],
)
def test_mm_register_fallback(m, n, k, dtype):
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((n, k), device="cuda", dtype=dtype).T

    out = tlx_mm(a, b, arch="gfx950", space="heuristic")
    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        out,
        expected,
        atol=1e-2 * expected.abs().max().item(),
        rtol=1e-2,
    )


def test_mm_rejects_invalid_rank():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((16,), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 16), device="cuda", dtype=torch.float16)
    with pytest.raises(InvalidInput, match="rank-2"):
        tlx_mm(a, b, arch="gfx950")


def test_mm_rejects_mismatched_reduction_dimensions():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((17, 8), device="cuda", dtype=torch.float16)
    with pytest.raises(InvalidInput, match="reduction dimensions"):
        tlx_mm(a, b, arch="gfx950")


def test_mm_rejects_mismatched_dtype():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 8), device="cuda", dtype=torch.float32)
    with pytest.raises(InvalidInput, match="same dtype and device"):
        tlx_mm(a, b, arch="gfx950")


def test_mm_rejects_mismatched_device():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 8), device="cpu", dtype=torch.float16)
    with pytest.raises(InvalidInput, match="same dtype and device"):
        tlx_mm(a, b, arch="gfx950")


def test_mm_rejects_invalid_space():
    from triton.tlx.ops.kernels.mm.gfx950 import mm

    a = torch.randn((7, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((8192, 2048), device="cuda", dtype=torch.float16).T
    with pytest.raises(InvalidInput, match="space='heuristic'"):
        mm(a, b, space="full")


def test_mm_rejects_unsupported_operands():
    from triton.tlx.ops.kernels.mm.gfx950 import matmul, mm, supports

    a = torch.randn((7, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((8192, 2048), device="cuda", dtype=torch.float16).T
    unsupported_a = torch.randn(
        (17, 2048), device="cuda", dtype=torch.float16
    )
    assert supports(a, b)
    assert not supports(unsupported_a, b)
    assert not supports(a.to(torch.float32), b.to(torch.float32))
    assert not supports(a, b.contiguous())
    with pytest.raises(InvalidInput, match="does not support"):
        mm(unsupported_a, b)
    with pytest.raises(InvalidInput, match="does not support"):
        matmul(unsupported_a, b)


@pytest.mark.parametrize("failure", ["type", "shape", "dtype", "device"])
def test_mm_rejects_invalid_output(failure):
    from triton.tlx.ops.kernels.mm.gfx950 import matmul

    a = torch.randn((7, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((8192, 2048), device="cuda", dtype=torch.float16).T
    if failure == "type":
        out, match = object(), "torch.Tensor"
    elif failure == "shape":
        out = torch.empty((7, 8191), device="cuda", dtype=torch.float16)
        match = "output shape"
    elif failure == "dtype":
        out = torch.empty((7, 8192), device="cuda", dtype=torch.float32)
        match = "output dtype"
    else:
        out = torch.empty((7, 8192), device="cpu", dtype=torch.float16)
        match = "output device"
    with pytest.raises(InvalidInput, match=match):
        matmul(a, b, out=out)


def test_mm_rejects_plan_that_does_not_cover_m(monkeypatch):
    import triton.tlx.ops.kernels.mm.gfx950 as gfx950

    a = torch.randn((17, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 32), device="cuda", dtype=torch.float16).T
    monkeypatch.setitem(
        gfx950._KNOWN_PLANS,
        (17, 16, 32),
        gfx950._Plan(16, 16, 2, 16, 4),
    )
    with pytest.raises(InvalidInput, match="at most 16 rows"):
        gfx950.matmul(a, b)


def test_mm_supports_unaligned_contiguous_k_views():
    from triton.tlx.ops.kernels.mm.gfx950 import matmul

    a = torch.randn((7, 2049), device="cuda", dtype=torch.float16)[:, 1:]
    b = torch.randn((8192, 2049), device="cuda", dtype=torch.float16)[:, 1:].T
    actual = matmul(a, b)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "n,k,pattern_period,segment_k",
    [(8192, 2048, 512, 128), (2048, 4096, 1024, 256)],
)
def test_mm_matches_aten_for_cancellation(
    n, k, pattern_period, segment_k
):
    """Exercise cancellation-sensitive ordered partial reduction."""
    from triton.tlx.ops.kernels.mm.gfx950 import matmul

    values = torch.zeros(k, device="cuda", dtype=torch.float16)
    for base in range(0, k, pattern_period):
        values[base:base + segment_k] = 65504.0
        values[base + segment_k:base + 2 * segment_k] = 0.001
        values[base + 2 * segment_k:base + 3 * segment_k] = -65504.0
        values[base + 3 * segment_k:base + 4 * segment_k] = 0.001
    a = torch.ones((7, k), device="cuda", dtype=torch.float16)
    b = values[None, :].repeat(n, 1).contiguous().T

    actual = matmul(a, b)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
