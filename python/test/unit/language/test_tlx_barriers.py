import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda
import triton.tlx.language as tlx


# Define triton kernels for unit tests
@triton.jit
def tlx_square_non_ws(
    x_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # prologue
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # mbarrier ops
    bars = tlx.alloc_barriers(num_barriers=1)  # create
    bar = tlx.local_view(bars, 0)
    tlx.barrier_arrive(bar=bar)  # Release
    tlx.barrier_wait(bar=bar, phase=0)  # Wait (proceed immediately)

    # Some arith ops TODO. add WS
    x = tl.load(x_ptr + offsets, mask=mask)
    z = x * x
    tl.store(z_ptr + offsets, z, mask=mask)

@triton.jit
def tlx_square_ws(
    x_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # prologue
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # mbarrier ops
    bars = tlx.alloc_barriers(num_barriers=1)  # create
    bar = tlx.local_view(bars, 0)

    with tlx.async_tasks():
        with tlx.async_task("default"):
            tlx.barrier_arrive(bar=bar)  # Release

        with tlx.async_task(num_warps=4):
            tlx.barrier_wait(bar=bar, phase=0)  # Wait (proceed immediately)

            # Some arith ops TODO. add WS
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            z = x * x
            tl.store(z_ptr + offsets, z, mask=mask)


# Unit test for arrive/wait
@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9 for NV",
)
@pytest.mark.parametrize("func, BLOCK_SIZE", [(f, 1024) for f in [tlx_square_non_ws, tlx_square_ws]])
# def test_mbarriers(BLOCK_SIZE, device):
def test_wait_arrive(func, BLOCK_SIZE, device):

    # prepare inputs
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    z = torch.empty_like(x)
    z_ref = torch.empty_like(x)

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    kernel = func[grid](x, z, n_elements, BLOCK_SIZE)

    # ASSERT in ttgir
    ttgir = kernel.asm["ttgir"]
    assert (ttgir.count("ttng.init_barrier") == 1) and (ttgir.count("ttng.wait_barrier") == 1) and (
        ttgir.count("ttng.barrier_expect") == 0) and (ttgir.count("ttng.arrive_barrier") == 1), f"TTGIR {ttgir}"

    z_ref = x * x

    torch.testing.assert_close(z, z_ref, check_dtype=False)

