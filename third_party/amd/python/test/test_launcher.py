import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_hip



pytestmark = pytest.mark.skipif(not is_hip(), reason="Requires HIP backend")
@triton.jit
def _add_kernel(x, y, output, n_elements):
    BLOCK_SIZE: tl.constexpr = 256
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(
        output + offsets,
        tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask),
        mask=mask,
    )


@triton.jit
def _tuple_scale_kernel(inputs, output, SCALE: tl.constexpr):
    tl.store(output, (tl.load(inputs[0]) + tl.load(inputs[1])) * SCALE)
def test_flat_signature_launcher():
    n_elements = 1025
    x = torch.randn(n_elements, device="cuda")
    y = torch.randn(n_elements, device="cuda")
    output = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, 256), )

    compiled = _add_kernel.warmup(x, y, output, n_elements, grid=grid)
    assert compiled.run.arg_annotations is None

    _add_kernel[grid](x, y, output, n_elements)

    torch.testing.assert_close(output, x + y)
def test_structured_signature_launcher_fallback():
    x = torch.tensor([11.0], device="cuda")
    y = torch.tensor([31.0], device="cuda")
    output = torch.empty_like(x)

    compiled = _tuple_scale_kernel.warmup((x, y), output, SCALE=2.0, grid=(1, ))
    assert compiled.run.arg_annotations is not None

    _tuple_scale_kernel[(1, )]((x, y), output, SCALE=2.0)
    torch.testing.assert_close(output, torch.full_like(output, 84.0))