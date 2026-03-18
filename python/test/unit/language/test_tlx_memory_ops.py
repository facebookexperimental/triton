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
