import pytest
import torch

from triton._internal_testing import is_blackwell
from triton.tlx.ops import InvalidInput, mm
from triton.tlx.ops.kernels.mm._shapes import SM100_FOCUS
from triton.tlx.ops.kernels.mm.sm100 import PERF_SHAPES, UNSUPPORTED_SHAPES, shape_has_tma_compatible_strides

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.mm input layouts are sm100-specific")


def test_perf_shapes_satisfy_tma_layout_contract():
    assert PERF_SHAPES
    assert UNSUPPORTED_SHAPES
    assert len(PERF_SHAPES) + len(UNSUPPORTED_SHAPES) == len(SM100_FOCUS)
    assert {tuple(shape) for shape in PERF_SHAPES + UNSUPPORTED_SHAPES} == {tuple(shape) for shape in SM100_FOCUS}

    for M, N, K, a_strides, b_strides, dtype in PERF_SHAPES:
        element_size = {"fp16": 2, "bf16": 2}[dtype]
        assert shape_has_tma_compatible_strides(M, N, K, a_strides, b_strides, element_size)
    for M, N, K, a_strides, b_strides, dtype in UNSUPPORTED_SHAPES:
        element_size = {"fp16": 2, "bf16": 2}[dtype]
        assert not shape_has_tma_compatible_strides(M, N, K, a_strides, b_strides, element_size)


def test_padded_row_major_inputs():
    dtype = torch.bfloat16
    a = torch.randn((64, 136), device="cuda", dtype=dtype)[:, :128]
    b = torch.randn((128, 264), device="cuda", dtype=dtype)[:, :256]

    out = mm(a, b, arch="sm100")
    ref = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=8e-3 * ref.abs().max().item(), rtol=8e-3)


@pytest.mark.parametrize("which", ["a", "b"])
def test_broadcast_operand_is_rejected_before_launch(which):
    dtype = torch.bfloat16
    a = torch.randn((1, 128), device="cuda", dtype=dtype).expand(64, 128)
    b = torch.randn((128, 256), device="cuda", dtype=dtype)
    if which == "b":
        a = torch.randn((64, 128), device="cuda", dtype=dtype)
        b = torch.randn((1, 256), device="cuda", dtype=dtype).expand(128, 256)

    with pytest.raises(InvalidInput, match="broadcast or overlap"):
        mm(a, b, arch="sm100")


def test_unaligned_base_pointer_is_rejected_before_launch():
    dtype = torch.bfloat16
    a = torch.randn((64, 129), device="cuda", dtype=dtype)[:, 1:129]
    b = torch.randn((128, 256), device="cuda", dtype=dtype)

    with pytest.raises(InvalidInput, match="base pointer must be 16-byte aligned"):
        mm(a, b, arch="sm100")


def test_unaligned_descriptor_row_stride_is_rejected_before_launch():
    dtype = torch.bfloat16
    a = torch.randn((16, 7), device="cuda", dtype=dtype).T
    b = torch.randn((16, 8), device="cuda", dtype=dtype)

    with pytest.raises(InvalidInput, match="descriptor row stride 7 elements"):
        mm(a, b, arch="sm100")


def test_unaligned_output_row_stride_is_rejected_before_launch():
    dtype = torch.bfloat16
    a = torch.randn((64, 64), device="cuda", dtype=dtype)
    b = torch.randn((64, 16), device="cuda", dtype=dtype)[:, :12]

    with pytest.raises(InvalidInput, match="output row stride 12 elements"):
        mm(a, b, arch="sm100")


@pytest.mark.parametrize(
    "a, b, message",
    [
        (lambda: torch.randn((2, 3, 4), device="cuda", dtype=torch.bfloat16), lambda: torch.randn(
            (4, 5), device="cuda", dtype=torch.bfloat16), "expects two rank-2 tensors"),
        (lambda: torch.randn((4, 7), device="cuda", dtype=torch.bfloat16), lambda: torch.randn(
            (8, 5), device="cuda", dtype=torch.bfloat16), "reduction dimensions must match"),
        (lambda: torch.randn((4, 8), device="cuda", dtype=torch.bfloat16), lambda: torch.randn(
            (8, 5), device="cuda", dtype=torch.float16), "same dtype and device"),
    ],
)
def test_invalid_metadata_is_rejected_before_launch(a, b, message):
    with pytest.raises(InvalidInput, match=message):
        mm(a(), b(), arch="sm100")
