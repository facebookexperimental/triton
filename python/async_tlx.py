import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

def test_async_tasks(BLOCK_SIZE):
    @triton.jit
    def test_warp_specialized_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        with tlx.async_tasks():
            with tlx.async_task("default"):
                tl.store(x_ptr + offsets, x)

    device='cuda'
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    test_warp_specialized_kernel[grid](x, BLOCK_SIZE)

test_async_tasks(BLOCK_SIZE=1024)
