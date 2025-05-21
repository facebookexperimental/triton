import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda
import triton.tlx.language as tlx


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9 for NV",
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
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    kernel = add2_warp_specialized_kernel[grid](x, y, output1, a, b, output2, n_elements, BLOCK_SIZE)
    ttgir = kernel.asm["ttgir"]
    assert "ttg.warp_specialize" in ttgir

    ref_out1, ref_out2 = dual_add(x, y, a, b)
    torch.testing.assert_close(output1, ref_out1, check_dtype=False)
    torch.testing.assert_close(output2, ref_out2, check_dtype=False)


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9 for NV",
)
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_mbarriers(BLOCK_SIZE, device):

    @triton.jit
    def tlx_square(
        x_ptr,
        z_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE

        # create mbarrier 
        bars = tlx.alloc_barriers(num_barriers=1)

        bar = tlx.local_view(bars, 0)

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        tlx.barrier_arrive(bar=bar)  # Release bar

        tlx.barrier_wait(bar=bar, phase=0)  # Wait bar

        x = tl.load(x_ptr + offsets, mask=mask)

        z = x * x
        tl.store(z_ptr + offsets, z, mask=mask)

    # prepare inputs
    torch.manual_seed(0)
    # size = 98432
    size = 4096
    x = torch.rand(size, device=device)
    z = torch.empty_like(x)
    z_ref = torch.empty_like(x)

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    kernel = tlx_square[grid](x, z, n_elements, BLOCK_SIZE)

    # ASSERT in ttgir
    ttgir = kernel.asm["ttgir"]
    assert (ttgir.count("ttng.init_barrier") == 1) and (ttgir.count("ttng.wait_barrier") == 1) and (
        ttgir.count("ttng.barrier_expect") == 0) and (ttgir.count("ttng.arrive_barrier") == 1), f"TTGIR {ttgir}"

    z_ref = x * x

    torch.testing.assert_close(z, z_ref, check_dtype=False)


@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires compute capability >= 9 for NV",
)
@pytest.mark.parametrize("BLOCK_SIZE", [(1024)])
def test_local_alloc_index(BLOCK_SIZE, device):

    @triton.jit
    def local_alloc_index(
        x_ptr,
        y_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        buffers = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(2))
        buffer0 = tlx.local_view(buffers, 0)
        buffer1 = tlx.local_view(buffers, 1)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = local_alloc_index[grid](x, y, n_elements, BLOCK_SIZE)
    # TODO(Arda): Once we have the loads, add checks here


def test_thread_id(device):

    @triton.jit
    def store_from_thread_0_kernel(output_ptr, value, n_elements, axis : tl.constexpr,
        BLOCK_SIZE: tl.constexpr,

    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tid = tlx.thread_id(axis)
        if tid == 0:
            tl.store(output_ptr + offsets, value, mask=mask)

    output = torch.zeros(32, dtype=torch.int32, device='cuda')
    n_elements = output.numel()
    value = 42
    store_from_thread_0_kernel[(1,)](output, value, n_elements, 0, 32, num_warps=1)
    torch.cuda.synchronize()
    expected_output = torch.zeros(32, dtype=torch.int32, device='cuda')
    expected_output[0] = value
    torch.testing.assert_close(output, expected_output)
