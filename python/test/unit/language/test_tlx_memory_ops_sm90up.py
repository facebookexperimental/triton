"""TLX memory ops tests -- Hopper and newer."""
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer
import triton.language.extra.tlx as tlx
from typing import Optional


@triton.jit
def local_gather_kernel(
    matrix_ptr,
    indices_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
):
    """Test lds gather using tlx.local_gather() with axis-based API."""
    indices_x = tl.arange(0, N)
    indices_y = tl.arange(0, M)
    offsets_2d = indices_x[:, None] * M + indices_y[None, :]
    matrix_regs = tl.load(matrix_ptr + offsets_2d)

    # Allocate 2D shared memory and store the matrix
    smem_1d_buffers = tlx.local_alloc((N * M, ), tlx.dtype_of(matrix_ptr), 1)
    smem_1d = tlx.local_view(smem_1d_buffers, 0)
    tlx.local_store(smem_1d, matrix_regs.reshape((N * M, )))

    # Load the gather indices
    offsets_1d = tl.arange(0, N)
    indices = tl.load(indices_ptr + offsets_1d)

    # Gather using axis-based API: result[i] = smem_1d[indices[i]]
    gathered = tlx.local_gather(smem_1d, indices, 0)

    # store result to global memory
    tl.store(output_ptr + offsets_1d, gathered)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_local_load(BLOCK_SIZE, device):

    @triton.jit
    def local_load(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_ptr_offsets = x_ptr + offsets
        y_ptr_offsets = y_ptr + offsets

        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 3)
        tlx.async_load(x_ptr_offsets, buffers[0], mask=mask)
        tlx.async_load(y_ptr_offsets, buffers[1], mask=mask)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        x_local = tlx.local_load(buffers[0])
        y_local = tlx.local_load(buffers[1])
        local_add = x_local + y_local
        tl.store(output_ptr + offsets, local_add, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = local_load[grid](x, y, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.local_alloc") == 1
    assert kernel.asm["ttgir"].count("ttg.memdesc_index") == 2
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 2
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1
    assert kernel.asm["ttgir"].count("ttg.async_wait") == 1
    assert kernel.asm["ttgir"].count("ttg.local_load") == 2
    torch.testing.assert_close(x + y, output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(4)])
def test_local_slice(BLOCK_SIZE, device):

    @triton.jit
    def local_load(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        x_ptr_offsets = x_ptr + offsets

        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
        tlx.async_load(x_ptr_offsets, buffers[0])
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        buffer_0 = tlx.local_slice(buffers[0], [0], [BLOCK_SIZE // 2])
        buffer_1 = tlx.local_slice(buffers[0], [BLOCK_SIZE // 2], [BLOCK_SIZE // 2])
        x_0 = tlx.local_load(buffer_0)
        x_1 = tlx.local_load(buffer_1)

        offsets = block_start + tl.arange(0, BLOCK_SIZE // 2)
        output_ptr_offsets = output_ptr + offsets
        tl.store(output_ptr_offsets, x_0)
        tl.store(output_ptr_offsets + BLOCK_SIZE // 2, x_1)

    torch.manual_seed(0)
    size = 4
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = local_load[grid](x, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.local_alloc") == 1
    assert kernel.asm["ttgir"].count("ttg.memdesc_index") == 1
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 1
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1
    assert kernel.asm["ttgir"].count("ttg.async_wait") == 1
    assert kernel.asm["ttgir"].count("ttg.local_load") == 2
    torch.testing.assert_close(x, output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_local_store(BLOCK_SIZE, device):

    @triton.jit
    def local_load_store(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_ptr_offsets = x_ptr + offsets
        y_ptr_offsets = y_ptr + offsets

        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, tl.constexpr(4))
        buffer0 = tlx.local_view(buffers, 0)
        buffer1 = tlx.local_view(buffers, 1)
        buffer2 = tlx.local_view(buffers, 2)
        tlx.async_load(x_ptr_offsets, buffer0, mask=mask)
        tlx.async_load(y_ptr_offsets, buffer1, mask=mask)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        x_local = tlx.local_load(buffer0)
        y_local = tlx.local_load(buffer1)
        local_add = x_local + y_local
        # store result into buffer2 and then load it
        tlx.local_store(buffer2, local_add)
        result = tlx.local_load(buffer2)
        tl.store(output_ptr + offsets, result, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = local_load_store[grid](x, y, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.local_alloc") == 1
    assert kernel.asm["ttgir"].count("ttg.memdesc_index") == 3
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 2
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1
    assert kernel.asm["ttgir"].count("ttg.async_wait") == 1
    assert kernel.asm["ttgir"].count("ttg.local_load") == 3
    assert kernel.asm["ttgir"].count("ttg.local_store") == 1
    torch.testing.assert_close(x + y, output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_async_wait(BLOCK_SIZE, device):

    @triton.jit
    def async_wait_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input_ptr_offsets = input_ptr + offsets
        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        tlx.async_load(input_ptr_offsets, buffer, mask=mask)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        x = tlx.local_load(buffer)
        tl.store(output_ptr + offsets, x, mask=mask)

    @triton.jit
    def async_wait_token_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input_ptr_offsets = input_ptr + offsets
        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, tl.constexpr(1))
        buffer = tlx.local_view(buffers, 0)
        token = tlx.async_load(input_ptr_offsets, buffer, mask=mask)
        token = tlx.async_load_commit_group([token])
        tlx.async_load_wait_group(tl.constexpr(0), [token])
        x = tlx.local_load(buffer)
        tl.store(output_ptr + offsets, x, mask=mask)

    torch.manual_seed(0)
    size = 64
    x = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = async_wait_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.async_copy_global_to_local") == 1
    assert kernel.asm["ttgir"].count("ttg.async_commit_group") == 1
    assert kernel.asm["ttgir"].count("ttg.async_wait") == 1
    torch.testing.assert_close(x, output)
    kernel = async_wait_token_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    torch.testing.assert_close(x, output)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_local_trans(device):

    @triton.jit
    def local_trans_kernel(
        input_ptr,
        output_ptr,
        M,
        N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Compute tile offset in global memory
        off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Compute global offsets
        input_offset = off_m[:, None] * N + off_n[None, :]
        output_offset = off_n[:, None] * M + off_m[None, :]

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1))
        buffer0 = tlx.local_view(buffers, 0)
        tlx.async_load(input_ptr + input_offset, buffer0)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))
        buffer1 = tlx.local_trans(buffer0)
        transposed = tlx.local_load(buffer1)
        tl.store(output_ptr + output_offset, transposed)

    torch.manual_seed(0)
    M, N = 32, 64
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 64
    x = torch.rand((M, N), dtype=torch.float32, device=device)
    y = torch.empty((N, M), dtype=torch.float32, device=device)
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    kernel = local_trans_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=1)
    assert kernel.asm["ttgir"].count("ttg.memdesc_trans") == 1
    torch.testing.assert_close(y, x.T)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
def test_local_gather(device):

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def local_gather_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        desc_in = tl.make_tensor_descriptor(
            input_ptr,
            shape=[1, M * N],
            strides=[M * N, 1],
            block_shape=[1, BLOCK_SIZE_M * BLOCK_SIZE_N],
        )

        desc_out = tl.make_tensor_descriptor(
            output_ptr,
            shape=[1, M * N],
            strides=[M * N, 1],
            block_shape=[1, BLOCK_SIZE_M * BLOCK_SIZE_N],
        )

        buffers_in = tlx.local_alloc((1, BLOCK_SIZE_N), tl.int16, BLOCK_SIZE_M)
        buffers_out = tlx.local_alloc((1, BLOCK_SIZE_N), tl.int16, BLOCK_SIZE_M)

        bars = tlx.alloc_barriers(tl.constexpr(1))
        bar = tlx.local_view(bars, 0)
        off_m = pid_m * BLOCK_SIZE_M
        off_n = pid_n * BLOCK_SIZE_N

        # Gather once
        tlx.barrier_expect_bytes(bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 2)
        reinterpreted = tlx.local_reinterpret(buffers_in, tl.int16, [1, BLOCK_SIZE_M * BLOCK_SIZE_N])
        tlx.async_descriptor_load(desc_in, reinterpreted, [0, off_m * N + off_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)

        # Use sub tiles separately
        for k in range(0, BLOCK_SIZE_M):
            buffer_in = tlx.local_view(buffers_in, k)
            buffer_out = tlx.local_view(buffers_out, k)
            in_local = tlx.local_load(buffer_in)
            tlx.local_store(buffer_out, in_local)

        reinterpreted = tlx.local_reinterpret(buffers_out, tl.int16, [1, BLOCK_SIZE_M * BLOCK_SIZE_N])
        tlx.async_descriptor_store(desc_out, reinterpreted, [0, off_m * N + off_n])

    triton.set_allocator(alloc_fn)
    M, N = 256, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = 64, 128
    x = torch.ones((M, N), dtype=torch.int16, device=device)
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    kernel = local_gather_kernel[grid](x, y, M, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    assert kernel.asm["ttgir"].count("ttng.async_tma_copy_global_to_local") == 1
    assert kernel.asm["ttgir"].count("ttng.async_tma_copy_local_to_global") == 1
    torch.testing.assert_close(x, y)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Need Hopper or newer")
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_local_index(BLOCK_SIZE, device):

    @triton.jit
    def local_index(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_ptr_offsets = x_ptr + offsets
        buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
        tlx.async_load(x_ptr_offsets, buffers[0], mask=mask)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        s = tl.zeros((1, ), dtype=tl.float32)
        for i in range(0, BLOCK_SIZE):
            s += tlx.local_load(buffers[0][i])

        # tl.store(output_ptr, s)
        # Store using block addressing - broadcast the sum to all elements in the block
        output_offsets = output_ptr + offsets
        s_broadcasted = tl.broadcast_to(s, (BLOCK_SIZE, ))
        tl.store(output_offsets, s_broadcasted, mask=mask)

    torch.manual_seed(0)
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    local_index[grid](x, output, n_elements, BLOCK_SIZE)
    y = torch.tensor([10.0, 10.0, 10.0, 10.0], device="cuda:0")
    torch.testing.assert_close(y, output)
