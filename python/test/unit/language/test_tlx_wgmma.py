
import pytest
import torch
import triton
import triton.language as tl
import triton.tlx.language as tlx
from triton._internal_testing import is_cuda

@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires compute capability == 9 for NV",
)
def test_async_dot(device):

    @triton.jit
    def async_dot_kernel(
        c_ptr,
        stride_cm,
        stride_cn,
        BLOCK_SIZE: tl.constexpr,
    ):
        buffers = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float16, 2)
        a = tlx.local_view(buffers, 0)
        b = tlx.local_view(buffers, 1)
        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        acc = tlx.async_dot(a, b, acc)
        offs_cm = tl.arange(0, BLOCK_SIZE)
        offs_cn = tl.arange(0, BLOCK_SIZE)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, acc)

    c = torch.empty((64, 64), device=device, dtype=torch.float32)
    grid = lambda META: (1,)
    async_dot_kernel[grid](
        c,  #
        c.stride(0),
        c.stride(1),  #
        64,
        num_warps=4,
    )
