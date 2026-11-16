# fmt: off

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_cuda
from triton.language.extra import libdevice

pytestmark = pytest.mark.skipif(not is_cuda(), reason="Requires CUDA backend")


# -----------------------
# test extern functions
# -----------------------


@triton.jit
def tanh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    direct_import: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if direct_import:
        y = libdevice.tanh(x)
    else:
        y = tl.extra.libdevice.tanh(x)

    tl.store(y_ptr + offsets, y, mask=mask)

def test_cuda_f32_to_e2m1x2_satfinite():
    if not is_cuda() or torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("requires Blackwell")

    @triton.jit
    def kernel(Hi, Lo, Out):
        offsets = tl.arange(0, 32)
        hi = tl.load(Hi + offsets)
        lo = tl.load(Lo + offsets)
        tl.store(Out + offsets, tl.extra.cuda.f32_to_e2m1x2_satfinite(hi, lo))

    # Exact values, ties to even, saturation, and both signs of zero.
    values = torch.tensor([0., .25, .5, .75, 1., 1.25, 1.5, 1.75,
                           2., 2.5, 3., 3.5, 4., 5., 6., float("inf")], device="cuda")
    codes = torch.tensor([0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 7], device="cuda")
    hi = torch.cat((values, -values))
    codes = torch.cat((codes, codes | 8))
    lo = hi.flip(0)
    out = torch.empty(32, dtype=torch.uint16, device="cuda")
    h = kernel[(1,)](hi, lo, out)
    expected = (codes << 4) | codes.flip(0)
    assert torch.equal(out.to(torch.int64), expected)
    assert "cvt.rn.satfinite.e2m1x2.f32" in h.asm["ptx"]
    assert ".pzo" not in h.asm["ptx"]


@pytest.mark.parametrize("direct_import", [False, True])
@pytest.mark.parametrize("dtype_str", ['float32', 'float64'])
def test_math_extern(dtype_str, direct_import):

    if not torch.cuda.is_available():
        pytest.skip("Test requires CUDA target.")
        return

    torch.manual_seed(42)

    x = torch.randn((100,), dtype=getattr(torch, dtype_str), device="cuda")

    y_tri = torch.empty_like(x)
    tanh_kernel[(1, )](x, y_tri, x.shape[0], direct_import, BLOCK_SIZE=128)

    y_ref = torch.tanh(x)
    np.testing.assert_allclose(y_ref.cpu().numpy(), y_tri.cpu().numpy(), rtol=0, atol=1.0e-6)


@triton.jit
def builtin_libdevice_unary_kernel(
    x_ptr,
    builtin_ptr,
    libdevice_ptr,
    n_elements,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if OP == "exp":
        builtin = tl.exp(x)
        external = libdevice.exp(x)
    elif OP == "fast_exp":
        builtin = tl.exp(x)
        external = libdevice.fast_expf(x)
    elif OP == "sqrt_rn":
        builtin = tl.sqrt_rn(x)
        external = libdevice.sqrt_rn(x)
    elif OP == "exp2":
        builtin = tl.exp2(x)
        external = libdevice.exp2(x)
    elif OP == "log":
        builtin = tl.log(x)
        external = libdevice.log(x)
    elif OP == "log2":
        builtin = tl.log2(x)
        external = libdevice.log2(x)
    elif OP == "sin":
        builtin = tl.sin(x)
        external = libdevice.sin(x)
    elif OP == "cos":
        builtin = tl.cos(x)
        external = libdevice.cos(x)
    elif OP == "rsqrt":
        builtin = tl.rsqrt(x)
        external = libdevice.rsqrt(x)
    elif OP == "erf":
        builtin = tl.erf(x)
        external = libdevice.erf(x)
    elif OP == "floor":
        builtin = tl.floor(x)
        external = libdevice.floor(x)
    elif OP == "ceil":
        builtin = tl.ceil(x)
        external = libdevice.ceil(x)
    elif OP == "fast_sin":
        builtin = tl.sin(x)
        external = libdevice.fast_sinf(x)
    elif OP == "fast_cos":
        builtin = tl.cos(x)
        external = libdevice.fast_cosf(x)
    elif OP == "fast_log":
        builtin = tl.log(x)
        external = libdevice.fast_logf(x)
    elif OP == "fast_log2":
        builtin = tl.log2(x)
        external = libdevice.fast_log2f(x)

    tl.store(builtin_ptr + offsets, builtin, mask=mask)
    tl.store(libdevice_ptr + offsets, external, mask=mask)


@triton.jit
def builtin_libdevice_div_rn_kernel(
    x_ptr,
    y_ptr,
    builtin_ptr,
    libdevice_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    tl.store(builtin_ptr + offsets, tl.div_rn(x, y), mask=mask)
    tl.store(libdevice_ptr + offsets, libdevice.div_rn(x, y), mask=mask)


def _random_payloads(generator, n_elements):
    return torch.randint(-(2**31), 2**31 - 1, (n_elements,), dtype=torch.int32, device="cuda", generator=generator)


@pytest.mark.parametrize(
    "op", ["exp", "fast_exp", "exp2", "log", "log2", "sin", "cos", "sqrt_rn", "rsqrt", "erf", "floor", "ceil"]
)
def test_fpsan_libdevice_unary_equivalence(op, fresh_knobs):
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("test requires CUDA")

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    n_elements = 1024
    block_size = 256
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    x = _random_payloads(generator, n_elements)
    builtin = torch.empty_like(x)
    external = torch.empty_like(x)

    builtin_libdevice_unary_kernel[(triton.cdiv(n_elements, block_size),)](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(builtin, dtype=torch.float32),
        triton.TensorWrapper(external, dtype=torch.float32),
        n_elements,
        OP=op,
        BLOCK_SIZE=block_size,
    )

    assert torch.equal(builtin, external)


def test_fpsan_libdevice_div_rn_equivalence(fresh_knobs):
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("test requires CUDA")

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    n_elements = 1024
    block_size = 256
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    inputs = [_random_payloads(generator, n_elements) for _ in range(2)]
    builtin = torch.empty_like(inputs[0])
    external = torch.empty_like(inputs[0])

    builtin_libdevice_div_rn_kernel[(triton.cdiv(n_elements, block_size),)](
        *(triton.TensorWrapper(value, dtype=torch.float32) for value in inputs),
        triton.TensorWrapper(builtin, dtype=torch.float32),
        triton.TensorWrapper(external, dtype=torch.float32),
        n_elements,
        BLOCK_SIZE=block_size,
    )

    assert torch.equal(builtin, external)


@pytest.mark.parametrize("op", ["fast_sin", "fast_cos", "fast_log", "fast_log2"])
def test_fpsan_distinguishes_approximate_libdevice(op, fresh_knobs):
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("test requires CUDA")

    fresh_knobs.compilation.instrumentation_mode = "fpsan"
    n_elements = 1024
    block_size = 256
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    x = _random_payloads(generator, n_elements)
    builtin = torch.empty_like(x)
    external = torch.empty_like(x)

    builtin_libdevice_unary_kernel[(triton.cdiv(n_elements, block_size),)](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(builtin, dtype=torch.float32),
        triton.TensorWrapper(external, dtype=torch.float32),
        n_elements,
        OP=op,
        BLOCK_SIZE=block_size,
    )

    assert not torch.equal(builtin, external)
