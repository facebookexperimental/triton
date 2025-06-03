
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def dotmult_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)

def dotmult(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    dotmult_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return sum(output)

torch.manual_seed(0)
size = 100000
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_triton = dotmult(x, y)
print(x)
print(y)
print(output_triton)
