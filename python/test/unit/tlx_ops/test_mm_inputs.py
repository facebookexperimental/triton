import pytest
import torch

from triton._internal_testing import is_blackwell
from triton.tlx.ops import InvalidInput, mm

pytestmark = pytest.mark.skipif(not is_blackwell(), reason="tlx.ops.mm input layouts are sm100-specific")


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


@pytest.mark.parametrize(
    "a, b, message",
    [
        (lambda: torch.randn((2, 3, 4), device="cuda", dtype=torch.bfloat16), lambda: torch.randn(
            (4, 5), device="cuda", dtype=torch.bfloat16), "expects two 2D tensors"),
        (lambda: torch.randn((4, 7), device="cuda", dtype=torch.bfloat16), lambda: torch.randn(
            (8, 5), device="cuda", dtype=torch.bfloat16), "incompatible dimensions"),
        (lambda: torch.randn((4, 8), device="cuda", dtype=torch.bfloat16), lambda: torch.randn(
            (8, 5), device="cuda", dtype=torch.float16), "matching dtypes"),
    ],
)
def test_invalid_metadata_is_rejected_before_launch(a, b, message):
    with pytest.raises(InvalidInput, match=message):
        mm(a(), b(), arch="sm100")
