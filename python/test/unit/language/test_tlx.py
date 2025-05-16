import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda
import triton.tlx.language as tlx


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires compute capability == 9 for NV",
)
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_async_tasks(BLOCK_SIZE, device):

    @triton.jit
    def add2_warp_specialized_kernel(
        x_ptr,
        y_ptr,
        z_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        with tlx.async_tasks():
            with tlx.async_task("default"):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                output = x + y
                tl.store(z_ptr + offsets, output, mask=mask)
            with tlx.async_task(num_warps=4):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
                output = a + b
                tl.store(c_ptr + offsets, output, mask=mask)

    def dual_add(x, y, a, b):
        return x + y, a + b

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    a = torch.rand(size, device=device)
    b = torch.rand(size, device=device)

    output1 = torch.empty_like(x)
    output2 = torch.empty_like(a)
    n_elements = output1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add2_warp_specialized_kernel[grid](x, y, output1, a, b, output2, n_elements, BLOCK_SIZE)

    ref_out1, ref_out2 = dual_add(x, y, a, b)
    torch.testing.assert_close(output1, ref_out1, check_dtype=False)
    torch.testing.assert_close(output2, ref_out2, check_dtype=False)


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires compute capability == 9 for NV",
)
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_alloc_barriers(BLOCK_SIZE, device):

    @triton.jit
    def add_with_mbarrier(
        x_ptr,
        y_ptr,
        z_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE

        num_bars = tl.constexpr(2)
        arrive_count = tl.constexpr(2)
        bars = tlx.alloc_barriers(num_bars, arrive_count)

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(z_ptr + offsets, output, mask=mask)

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)

    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_with_mbarrier[grid](x, y, output, n_elements, BLOCK_SIZE)

    torch.testing.assert_close(output, x + y, check_dtype=False)
