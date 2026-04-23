import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hopper_or_newer, is_blackwell
import triton.language.extra.tlx as tlx
from typing import Optional


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


# Tests tl.load->tlx_local_store->tlx_local_load
# This is a smem load/store test variant that does not use
# async_load, so this test can be run on platforms where
# async_load has no/limited support
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_load_store_smem_with_tl_load(BLOCK_SIZE, device):

    @triton.jit
    def smem_reg_store_load(
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

        smem_buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 3)
        x_smem = tlx.local_view(smem_buffers, 0)
        y_smem = tlx.local_view(smem_buffers, 1)

        x_tile = tl.load(x_ptr + offsets, mask=mask)
        y_tile = tl.load(y_ptr + offsets, mask=mask)

        tlx.local_store(x_smem, x_tile)
        tlx.local_store(y_smem, y_tile)

        x_reg = tlx.local_load(x_smem)
        y_reg = tlx.local_load(y_smem)
        local_add = x_reg + y_reg
        tl.store(output_ptr + offsets, local_add, mask=mask)

    torch.manual_seed(0)
    size = 256
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    kernel = smem_reg_store_load[grid](x, y, output, n_elements, BLOCK_SIZE)
    assert kernel.asm["ttgir"].count("ttg.local_alloc") == 1
    assert kernel.asm["ttgir"].count("ttg.memdesc_index") == 2
    assert kernel.asm["ttgir"].count("ttg.local_load") == 2
    assert kernel.asm["ttgir"].count("ttg.local_store") == 2
    torch.testing.assert_close(x + y, output)


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


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_local_reinterpret(device):

    @triton.jit
    def local_reinterpret_kernel(
        x32_ptr,
        y32_ptr,
        x16_ptr,
        y16_ptr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Compute tile offset in global memory
        off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Compute global offsets
        input_offset = off_m[:, None] * BLOCK_SIZE_N + off_n[None, :]
        output_offset = off_m[:, None] * BLOCK_SIZE_N + off_n[None, :]

        tmem_buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        tmem_buffer_0 = tlx.local_view(tmem_buffers, 0)

        # x32 GMEM -> x32 SMEM -> x32 Reg -> x32 TMEM -> x32 Reg -> y32 GMEM
        smem_buffers32 = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1),
                                         tlx.storage_kind.smem)
        smem_buffer_32_0 = tlx.local_view(smem_buffers32, 0)
        tlx.async_load(x32_ptr + input_offset, smem_buffer_32_0)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        x32_reg = tlx.local_load(smem_buffer_32_0)
        tlx.local_store(tmem_buffer_0, x32_reg)
        x32_reg_from_tmem = tlx.local_load(tmem_buffer_0)
        tl.store(y32_ptr + output_offset, x32_reg_from_tmem)

        # x16 GMEM -> x16 SMEM -> x16 Reg -> x16 TMEM -> x16 Reg -> y16 GMEM
        smem_buffers16 = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float16, tl.constexpr(1),
                                         tlx.storage_kind.smem)
        smem_buffer_16_0 = tlx.local_view(smem_buffers16, 0)
        tlx.async_load(x16_ptr + input_offset, smem_buffer_16_0)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        reinterpreted = tlx.local_reinterpret(tmem_buffer_0, tl.float16)

        x16_reg = tlx.local_load(smem_buffer_16_0)
        tlx.local_store(reinterpreted, x16_reg)
        x16_reg_from_tmem = tlx.local_load(reinterpreted)
        tl.store(y16_ptr + output_offset, x16_reg_from_tmem)

    torch.manual_seed(0)
    M, N = 64, 128
    BLOCK_SIZE_M, BLOCK_SIZE_N = M, N
    x32 = torch.rand((M, N), dtype=torch.float32, device=device)
    y32 = torch.zeros((M, N), dtype=torch.float32, device=device)
    x16 = torch.rand((M, N), dtype=torch.float16, device=device)
    y16 = torch.zeros((M, N), dtype=torch.float16, device=device)
    grid = lambda meta: (1, )
    kernel = local_reinterpret_kernel[grid](x32, y32, x16, y16, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    assert kernel.asm["ttgir"].count("ttg.memdesc_reinterpret") == 1
    assert kernel.asm["ttgir"].count("ttng.tmem_store") == 2
    assert kernel.asm["ttgir"].count("ttng.tmem_alloc") == 1

    torch.testing.assert_close(x32, y32)
    torch.testing.assert_close(x16, y16)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_local_reinterpret_swizzled(device):

    @triton.jit
    def local_reinterpret_swizzled_kernel(
        a_ptr,
        stride_am,
        stride_ak,
        b_ptr,
        stride_bk,
        stride_bn,
        c_ptr,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (tl.arange(0, BLOCK_M // 2)[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_ptrs2 = a_ptr + (tl.arange(BLOCK_M // 2, BLOCK_M)[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # async load a and b into SMEM
        buf_alloc_a = tlx.local_alloc((BLOCK_M // 2, BLOCK_K), tl.float16, tl.constexpr(2))
        buf_alloc_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, tl.constexpr(1))
        b_smem = tlx.local_view(buf_alloc_b, 0)
        # load half of a each time
        tlx.async_load(a_ptrs, buf_alloc_a[0])
        tlx.async_load(a_ptrs2, buf_alloc_a[1])
        tlx.async_load(b_ptrs, b_smem)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(tl.constexpr(0))

        buffers = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        acc_tmem = tlx.local_view(buffers, 0)

        # reinterpret a into one big tensor
        a_reinterpreted = tlx.local_reinterpret(buf_alloc_a, tl.float16, [BLOCK_M, BLOCK_K])
        # no barrier, tcgen5 mma synchronous semantic, compiler auto inserts barrier and wait
        tlx.async_dot(a_reinterpreted, b_smem, acc_tmem, use_acc=False, mBarriers=[], out_dtype=OUT_DTYPE)

        result = tlx.local_load(acc_tmem)

        c = result.to(tl.float16)
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c)

    torch.manual_seed(0)
    M, N, K = (64, 64, 32)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    y = torch.randn((K, N), device=device, dtype=torch.float16)
    z = torch.zeros((M, N), device=device, dtype=torch.float16)

    kern_kwargs = {"BLOCK_M": M, "BLOCK_K": K, "BLOCK_N": N, "OUT_DTYPE": tl.float32}
    kernel = local_reinterpret_swizzled_kernel[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z,
                                                       z.stride(0), z.stride(1), **kern_kwargs)

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttg.memdesc_reinterpret") == 1

    ref_out = torch.matmul(x, y)
    torch.testing.assert_close(z, ref_out)


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
        buffer_in = tlx.local_view(buffers_in, 0)
        tlx.barrier_expect_bytes(bar, BLOCK_SIZE_M * BLOCK_SIZE_N * 2)
        reinterpreted = tlx.local_reinterpret(buffer_in, tl.int16, [1, BLOCK_SIZE_M * BLOCK_SIZE_N])
        tlx.async_descriptor_load(desc_in, reinterpreted, [0, off_m * N + off_n], bar)
        tlx.barrier_wait(bar=bar, phase=0)

        # Use sub tiles separately
        for k in range(0, BLOCK_SIZE_M):
            buffer_in = tlx.local_view(buffers_in, k)
            buffer_out = tlx.local_view(buffers_out, k)
            in_local = tlx.local_load(buffer_in)
            tlx.local_store(buffer_out, in_local)

        buffer_out = tlx.local_view(buffers_out, 0)
        reinterpreted = tlx.local_reinterpret(buffer_out, tl.int16, [1, BLOCK_SIZE_M * BLOCK_SIZE_N])
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


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_tmem_alloc_index(BLOCK_SIZE, device):

    @triton.jit
    def kernel(BLOCK_SIZE: tl.constexpr, ):
        buffers = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(2), tlx.storage_kind.tmem)
        buffer0 = tlx.local_view(buffers, 0)  # noqa: F841
        buffer1 = tlx.local_view(buffers, 1)  # noqa: F841

    grid = lambda meta: (1, )
    kerenl_info = kernel[grid](BLOCK_SIZE)
    # TODO: check numerics once tmem load/store is ready
    kerenl_info.asm["ttgir"]
    assert kerenl_info.asm["ttgir"].count("kernel") == 1


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N", [(64, 64), (64, 8), (128, 16)])
def test_tmem_load_store(BLOCK_SIZE_M, BLOCK_SIZE_N, device):

    @triton.jit
    def tmem_load_store_kernel(
        x_ptr,
        stride_m,
        stride_n,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        offs_m = tl.arange(0, BLOCK_SIZE_M)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        x_ptr_offsets = x_ptr + (offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)

        a = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 1.0, tl.float32)

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        buffer1 = tlx.local_view(buffers, 0)
        tlx.local_store(buffer1, a)
        b = tlx.local_load(buffer1)
        # b == a == tensor of 1.0
        tl.store(x_ptr_offsets, b + 2)

    x = torch.rand((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32, device=device)
    grid = lambda meta: (1, )
    kerenl_info = tmem_load_store_kernel[grid](x, x.stride(0), x.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N)

    assert kerenl_info.asm["ttir"].count("ttng.tmem_store") == 1
    assert kerenl_info.asm["ttir"].count("ttng.tmem_load") == 1

    assert kerenl_info.asm["ttgir"].count("kernel") == 1
    assert kerenl_info.asm["ttgir"].count("ttng.tmem_alloc") == 1
    assert kerenl_info.asm["ttgir"].count("ttng.tmem_store") == 1
    assert kerenl_info.asm["ttgir"].count("ttng.tmem_load") == 1

    ref_out = torch.ones_like(x) + 2
    torch.testing.assert_close(x, ref_out)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N", [(128, 64)])
def test_tmem_subslice(BLOCK_SIZE_M, BLOCK_SIZE_N, device):

    @triton.jit
    def tmem_subslice_kernel(
        x_ptr,
        stride_m,
        stride_n,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        offs_m = tl.arange(0, BLOCK_SIZE_M)
        offs_n1 = tl.arange(0, BLOCK_SIZE_N // 4)
        offs_n2 = tl.arange(BLOCK_SIZE_N // 4, BLOCK_SIZE_N // 2)
        offs_n3 = tl.arange(BLOCK_SIZE_N // 2, 3 * BLOCK_SIZE_N // 4)
        offs_n4 = tl.arange(3 * BLOCK_SIZE_N // 4, BLOCK_SIZE_N)
        x_ptr_offsets1 = x_ptr + (offs_m[:, None] * stride_m + offs_n1[None, :] * stride_n)
        x_ptr_offsets2 = x_ptr + (offs_m[:, None] * stride_m + offs_n2[None, :] * stride_n)
        x_ptr_offsets3 = x_ptr + (offs_m[:, None] * stride_m + offs_n3[None, :] * stride_n)
        x_ptr_offsets4 = x_ptr + (offs_m[:, None] * stride_m + offs_n4[None, :] * stride_n)

        a = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 1.0, tl.float32)

        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        buffer1 = tlx.local_view(buffers, 0)
        tlx.local_store(buffer1, a)

        subslice1 = tlx.subslice(buffer1, 0, BLOCK_SIZE_N // 4)
        subslice2 = tlx.subslice(buffer1, BLOCK_SIZE_N // 4, BLOCK_SIZE_N // 4)
        subslice3 = tlx.subslice(buffer1, BLOCK_SIZE_N // 2, BLOCK_SIZE_N // 4)
        subslice4 = tlx.local_slice(buffer1, [0, 3 * BLOCK_SIZE_N // 4], [BLOCK_SIZE_M, BLOCK_SIZE_N // 4])

        b1 = tlx.local_load(subslice1)
        b2 = tlx.local_load(subslice2)
        b3 = tlx.local_load(subslice3)
        b4 = tlx.local_load(subslice4)
        # b == a == tensor of 1.0
        tl.store(x_ptr_offsets1, b1 + 2)
        tl.store(x_ptr_offsets2, b2 + 2)
        tl.store(x_ptr_offsets3, b3 + 2)
        tl.store(x_ptr_offsets4, b4 + 2)

    x = torch.rand((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32, device=device)
    grid = lambda meta: (1, )
    kerenl_info = tmem_subslice_kernel[grid](x, x.stride(0), x.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N)

    assert kerenl_info.asm["ttir"].count("ttng.tmem_store") == 1
    assert kerenl_info.asm["ttir"].count("ttng.tmem_load") == 4

    assert kerenl_info.asm["ttgir"].count("kernel") == 1
    assert kerenl_info.asm["ttgir"].count("ttng.tmem_alloc") == 1
    assert kerenl_info.asm["ttgir"].count("ttng.tmem_store") == 1
    assert kerenl_info.asm["ttgir"].count("ttng.tmem_load") == 4

    ref_out = torch.ones_like(x) + 2
    torch.testing.assert_close(x, ref_out)


@triton.jit
def _global_tmem_func(
    buffers,
    x_ptr,
    stride_m,
    stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    x_ptr_offsets = x_ptr + (offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)

    ones = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 1.0, tl.float32)
    buffer1 = tlx.local_view(buffers, 0)
    tlx.local_store(buffer1, ones)
    b = tlx.local_load(buffer1)

    tl.store(x_ptr_offsets, b)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N", [(64, 64)])
def test_tmem_op_func(BLOCK_SIZE_M, BLOCK_SIZE_N, device):

    @triton.jit
    def tmem_op_func_kernel(
        x_ptr,
        stride_m,
        stride_n,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        # init tmem buffers here
        buffers = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        # pass buffers to another func to do actual processing
        _global_tmem_func(buffers, x_ptr, stride_m, stride_n, BLOCK_SIZE_M, BLOCK_SIZE_N)

    x = torch.rand((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32, device=device)
    grid = lambda meta: (1, )
    tmem_op_func_kernel[grid](x, x.stride(0), x.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N)

    ref_out = torch.ones_like(x)
    torch.testing.assert_close(x, ref_out)


@triton.jit
def math_kernel(x):
    return x * 0.5 * (1 + (0.7978845608 * x * (1.0 + 0.044715 * x * x)))


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("BLOCK_SIZE", [(64)])
def test_inline_tmem(BLOCK_SIZE, device):

    @triton.jit
    def kernel(y_ptr, BLOCK_SIZE: tl.constexpr):
        buffers = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, tl.constexpr(4), tlx.storage_kind.tmem)
        buffer0 = buffers[0]
        x = tlx.local_load(buffer0)
        offsets_i = tl.arange(0, BLOCK_SIZE)[:, None]
        offsets_j = tl.arange(0, BLOCK_SIZE)[None, :]
        offsets = offsets_i * BLOCK_SIZE + offsets_j
        y = math_kernel(x)
        tl.store(y_ptr + offsets, y)

    y = torch.rand((64, 64), dtype=torch.float32, device=device)
    grid = lambda meta: (1, )
    kerenl_info = kernel[grid](y, BLOCK_SIZE)
    assert kerenl_info.asm["ttir"].count("store") == 1


# 1D gather test
@triton.jit
def local_gather_kernel(
    matrix_ptr, indices_ptr, output_ptr, 
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


@pytest.mark.parametrize("N,M", [(32, 32), (64, 64), (128, 128)])
def test_local_gather(N, M):
    """Test gathering from 1D reshaped shared memory (diagonal of 2D matrix)."""
    device = torch.device("cuda")

    # Create a test matrix with known values
    matrix = torch.arange(N * M, dtype=torch.float32, device=device).reshape(N, M)

    # Create gather indices for diagonal elements: 0, M+1, 2*(M+1), ...
    indices = torch.arange(N, dtype=torch.int32, device=device) * (M + 1)

    output = torch.zeros(N, dtype=torch.float32, device=device)

    # Compute expected result: diagonal elements
    expected = matrix.flatten()[indices]

    # Launch kernel
    local_gather_kernel[(1, )](
        matrix,
        indices,
        output,
        N=N,
        M=M,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


@triton.jit
def local_scatter_kernel(
    indices_ptr, values_ptr, output_ptr, 
    N: tl.constexpr, 
    M: tl.constexpr,
):
    """Test lds scatter using tlx.local_scatter() with axis-based API."""
    # Allocate 2D shared memory and store the matrix
    smem_buffers = tlx.local_alloc((N * M, ), tlx.dtype_of(values_ptr), 1)
    smem = tlx.local_view(smem_buffers, 0)

    indices_x = tl.arange(0, N)
    indices_y = tl.arange(0, M)
    offsets_2d = indices_x[:, None] * M + indices_y[None, :]
    zeros = tl.zeros([N * M], tl.float32)
    tlx.local_store(smem, zeros)

    # Load the scatter indices and values from input
    offsets_1d = tl.arange(0, N)
    indices = tl.load(indices_ptr + offsets_1d)
    values = tl.load(values_ptr + offsets_1d)

    # Scatter using axis-based API: smem_1d[indices[i]] = values[i]
    tlx.local_scatter(smem, values, indices, 0)

    # Read back data from shared memory
    smem_values = tlx.local_load(smem)

    # store result to global memory
    tl.store(output_ptr + offsets_2d, smem_values.reshape([N, M]))


# 1-warp test
@pytest.mark.parametrize("N,M", [(32, 32), (64, 64), (128, 128)])
def test_local_scatter(N, M):
    """Test scattering to 1D reshaped shared memory (diagonal of 2D matrix)."""
    device = torch.device("cuda")

    # Create scatter indices for diagonal elements: 0, M+1, 2*(M+1), ...
    indices = torch.arange(N, dtype=torch.int32, device=device) * (M + 1)

    # Create values to scatter
    values = torch.arange(N, dtype=torch.float32, device=device) + 100.0

    output = torch.zeros((N, M), dtype=torch.float32, device=device)

    # Compute expected result: matrix starts at zero, then diagonal gets values
    expected = torch.zeros((N, M), dtype=torch.float32, device=device)
    for i in range(N):
        expected[i, i] = values[i]

    # Launch kernel
    local_scatter_kernel[(1, )](
        indices,
        values,
        output,
        N=N,
        M=M,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


# multi-warp test
@pytest.mark.parametrize("N,M,num_warps", [(64, 64, 2), (128, 128, 4)])
def test_scatter_gather_multiwarp(N, M, num_warps):
    """Test scatter and gather with multiple warps."""
    device = torch.device("cuda")

    # Test gather
    matrix = torch.arange(N * M, dtype=torch.float32, device=device).reshape(N, M)
    gather_indices = torch.arange(N, dtype=torch.int32, device=device) * (M + 1)
    gather_output = torch.zeros(N, dtype=torch.float32, device=device)
    gather_expected = matrix.flatten()[gather_indices]

    local_gather_kernel[(1, )](
        matrix,
        gather_indices,
        gather_output,
        N=N,
        M=M,
        num_warps=num_warps,
    )

    torch.testing.assert_close(gather_output, gather_expected)

    # Test scatter
    scatter_indices = torch.arange(N, dtype=torch.int32, device=device) * (M + 1)
    scatter_values = torch.arange(N, dtype=torch.float32, device=device) + 100.0
    scatter_output = torch.zeros((N, M), dtype=torch.float32, device=device)
    scatter_expected = torch.zeros((N, M), dtype=torch.float32, device=device)
    for i in range(N):
        scatter_expected[i, i] = scatter_values[i]

    local_scatter_kernel[(1, )](
        scatter_indices,
        scatter_values,
        scatter_output,
        N=N,
        M=M,
        num_warps=num_warps,
    )

    torch.testing.assert_close(scatter_output, scatter_expected)


# ============================================================================
# 2D Native Gather/Scatter Tests
# ============================================================================


@triton.jit
def local_gather_2d_kernel(
    matrix_ptr,
    indices_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    axis: tl.constexpr,
):
    """Test 2D gather along specified axis."""
    # Load the matrix from global memory [N, M]
    indices_x = tl.arange(0, N)
    indices_y = tl.arange(0, M)
    offsets_2d = indices_x[:, None] * M + indices_y[None, :]
    matrix_data = tl.load(matrix_ptr + offsets_2d)

    # Store in shared memory
    smem_2d_array = tlx.local_alloc((N, M), tl.float32, 1)
    smem_2d = tlx.local_view(smem_2d_array, 0)
    tlx.local_store(smem_2d, matrix_data)

    # Load indices [N, M] - same rank as source
    indices = tl.load(indices_ptr + offsets_2d)

    # Gather along specified axis
    gathered = tlx.local_gather(smem_2d, indices, axis=axis)

    # Store result
    tl.store(output_ptr + offsets_2d, gathered)


@pytest.mark.parametrize("N,M,axis", [(32, 32, 0), (32, 32, 1), (64, 64, 0), (64, 64, 1)])
def test_local_gather_2d_native(N, M, axis):
    """Test 2D gather along different axes."""
    device = torch.device("cuda")

    # Create a test matrix [N, M]
    matrix = torch.arange(N * M, dtype=torch.float32, device=device).reshape(N, M)

    # Create indices [N, M] - each position specifies where to gather from along the axis
    if axis == 0:
        # Each column gathers from a shifted row pattern
        indices = torch.arange(M, dtype=torch.int32, device=device)[None, :].expand(N, M)
        indices = (indices + torch.arange(N, dtype=torch.int32, device=device)[:, None]) % N
        # Expected: result[i, j] = matrix[indices[i, j], j]
        expected = torch.gather(matrix, 0, indices.long())
    else:  # axis == 1
        # Each row gathers from a shifted column pattern
        indices = torch.arange(N, dtype=torch.int32, device=device)[:, None].expand(N, M)
        indices = (indices + torch.arange(M, dtype=torch.int32, device=device)[None, :]) % M
        # Expected: result[i, j] = matrix[i, indices[i, j]]
        expected = torch.gather(matrix, 1, indices.long())

    output = torch.zeros((N, M), dtype=torch.float32, device=device)

    local_gather_2d_kernel[(1, )](
        matrix,
        indices,
        output,
        N=N,
        M=M,
        axis=axis,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


@triton.jit
def local_scatter_2d_kernel(
    indices_ptr,
    values_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    axis: tl.constexpr,
):
    """Test 2D scatter along specified axis."""
    # Initialize shared memory to zero
    smem_2d_array = tlx.local_alloc((N, M), tl.float32, 1)
    smem_2d = tlx.local_view(smem_2d_array, 0)

    indices_x = tl.arange(0, N)
    indices_y = tl.arange(0, M)
    offsets_2d = indices_x[:, None] * M + indices_y[None, :]
    zeros = tl.zeros([N, M], tl.float32)
    tlx.local_store(smem_2d, zeros)

    # Load indices [N, M] and values [N, M]
    indices = tl.load(indices_ptr + offsets_2d)
    values = tl.load(values_ptr + offsets_2d)

    # Scatter along specified axis
    tlx.local_scatter(smem_2d, values, indices, axis=axis)

    # Read back the result
    result = tlx.local_load(smem_2d)
    tl.store(output_ptr + offsets_2d, result)


@pytest.mark.parametrize("N,M,axis", [(32, 32, 0), (32, 32, 1)])
def test_local_scatter_2d_native(N, M, axis):
    """Test 2D scatter along different axes."""
    device = torch.device("cuda")

    # Create indices [N, M] - reverse pattern for scatter
    if axis == 0:
        indices = torch.arange(M, dtype=torch.int32, device=device)[None, :].expand(N, M)
        indices = (N - 1 - indices - torch.arange(N, dtype=torch.int32, device=device)[:, None]) % N
    else:  # axis == 1
        indices = torch.arange(N, dtype=torch.int32, device=device)[:, None].expand(N, M)
        indices = (M - 1 - indices - torch.arange(M, dtype=torch.int32, device=device)[None, :]) % M

    # Create values to scatter
    values = torch.arange(N * M, dtype=torch.float32, device=device).reshape(N, M) + 100.0

    output = torch.zeros((N, M), dtype=torch.float32, device=device)

    # Expected: scatter values according to indices
    expected = torch.zeros((N, M), dtype=torch.float32, device=device)
    expected.scatter_(axis, indices.long(), values)

    local_scatter_2d_kernel[(1, )](
        indices,
        values,
        output,
        N=N,
        M=M,
        axis=axis,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


# ============================================================================
# 3D Gather/Scatter Tests
# ============================================================================


@triton.jit
def local_gather_3d_kernel(
    tensor_ptr,
    indices_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    axis: tl.constexpr,
):
    """Test 3D gather along specified axis."""
    # Load the tensor from global memory [N, M, P]
    idx_n = tl.arange(0, N)[:, None, None]
    idx_m = tl.arange(0, M)[None, :, None]
    idx_p = tl.arange(0, P)[None, None, :]

    offsets_3d = idx_n * (M * P) + idx_m * P + idx_p
    tensor_data = tl.load(tensor_ptr + offsets_3d)

    # Store in shared memory
    smem_3d_array = tlx.local_alloc((N, M, P), tl.float32, 1)
    smem_3d = tlx.local_view(smem_3d_array, 0)
    tlx.local_store(smem_3d, tensor_data)

    # Load indices [N, M, P] - same rank as source
    indices_data = tl.load(indices_ptr + offsets_3d)

    # Gather along specified axis
    gathered = tlx.local_gather(smem_3d, indices_data, axis=axis)

    # Store result
    tl.store(output_ptr + offsets_3d, gathered)


@pytest.mark.parametrize("N,M,P,axis", [(16, 8, 4, 0), (16, 8, 4, 1), (16, 8, 4, 2)])
def test_local_gather_3d_native(N, M, P, axis):
    """Test 3D gather along different axes."""
    device = torch.device("cuda")

    # Create a test tensor [N, M, P]
    tensor = torch.arange(N * M * P, dtype=torch.float32, device=device).reshape(N, M, P)

    # Create indices [N, M, P] - each position specifies where to gather from along the axis
    if axis == 0:
        # Pattern for gathering along first dimension
        base = torch.arange(M * P, dtype=torch.int32, device=device).reshape(1, M, P)
        offset = torch.arange(N, dtype=torch.int32, device=device).reshape(N, 1, 1)
        indices = (base + offset) % N
    elif axis == 1:
        # Pattern for gathering along second dimension
        base = torch.arange(N, dtype=torch.int32, device=device).reshape(N, 1, 1)
        offset = torch.arange(P, dtype=torch.int32, device=device).reshape(1, 1, P)
        indices = ((base + offset) % M).expand(N, M, P).contiguous()
    else:  # axis == 2
        # Pattern for gathering along third dimension
        base = torch.arange(N * M, dtype=torch.int32, device=device).reshape(N, M, 1)
        indices = (base % P).expand(N, M, P).contiguous()

    # Ensure indices is contiguous in C-style layout
    indices = indices.contiguous()

    # Compute expected result using torch.gather
    expected = torch.gather(tensor, axis, indices.long())

    output = torch.zeros((N, M, P), dtype=torch.float32, device=device)

    local_gather_3d_kernel[(1, )](
        tensor,
        indices,
        output,
        N=N,
        M=M,
        P=P,
        axis=axis,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)


@triton.jit
def local_scatter_3d_kernel(
    indices_ptr,
    values_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    axis: tl.constexpr,
):
    """Test 3D scatter along specified axis."""
    idx_n = tl.arange(0, N)[:, None, None]
    idx_m = tl.arange(0, M)[None, :, None]
    idx_p = tl.arange(0, P)[None, None, :]

    offsets_3d = idx_n * (M * P) + idx_m * P + idx_p

    # Initialize shared memory to zero
    smem_3d_array = tlx.local_alloc((N, M, P), tl.float32, 1)
    smem_3d = tlx.local_view(smem_3d_array, 0)

    zeros = tl.full([N, M, P], 0.0, tl.float32)
    tlx.local_store(smem_3d, zeros)

    # Load indices [N, M, P] and values [N, M, P]
    indices_data = tl.load(indices_ptr + offsets_3d)
    values_data = tl.load(values_ptr + offsets_3d)

    # Scatter along specified axis
    tlx.local_scatter(smem_3d, values_data, indices_data, axis=axis)

    # Read back the result
    result = tlx.local_load(smem_3d)
    tl.store(output_ptr + offsets_3d, result)


@pytest.mark.parametrize("N,M,P,axis", [(16, 8, 4, 0), (16, 8, 4, 1), (16, 8, 4, 2)])
def test_scatter_3d_native(N, M, P, axis):
    """Test 3D scatter along different axes."""
    device = torch.device("cuda")

    # Create indices [N, M, P] that form a permutation along the scatter axis
    if axis == 0:
        # For axis 0: permute N dimension, keeping (M, P) coordinates fixed
        # Each (j, k) position has a unique permutation of N indices
        base = torch.arange(M * P, dtype=torch.int32, device=device).reshape(1, M, P)
        offset = torch.arange(N, dtype=torch.int32, device=device).reshape(N, 1, 1)
        indices = ((N - 1 - base - offset) % N).contiguous()
    elif axis == 1:
        # For axis 1: permute M dimension, keeping (N, P) coordinates fixed
        # Each (i, k) position has a unique permutation of M indices
        base = torch.arange(N * P, dtype=torch.int32, device=device).reshape(N, 1, P)
        offset = torch.arange(M, dtype=torch.int32, device=device).reshape(1, M, 1)
        indices = ((M - 1 - base - offset) % M).contiguous()
    else:  # axis == 2
        # For axis 2: permute P dimension, keeping (N, M) coordinates fixed
        # Each (i, j) position has a unique permutation of P indices
        base = torch.arange(N * M, dtype=torch.int32, device=device).reshape(N, M, 1)
        offset = torch.arange(P, dtype=torch.int32, device=device).reshape(1, 1, P)
        indices = ((P - 1 - base - offset) % P).contiguous()

    # Ensure indices is contiguous
    indices = indices.contiguous()

    # Create values to scatter
    values = (torch.arange(N * M * P, dtype=torch.float32, device=device).reshape(N, M, P) + 200.0).contiguous()

    output = torch.zeros((N, M, P), dtype=torch.float32, device=device)

    # Expected: scatter values according to indices
    expected = torch.zeros((N, M, P), dtype=torch.float32, device=device)
    expected.scatter_(axis, indices.long(), values)

    local_scatter_3d_kernel[(1, )](
        indices,
        values,
        output,
        N=N,
        M=M,
        P=P,
        axis=axis,
        num_warps=1,
    )

    torch.testing.assert_close(output, expected)
