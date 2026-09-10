"""TLX memory ops tests -- Blackwell-only."""
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
import triton.language.extra.tlx as tlx


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


@triton.jit
def math_kernel(x):
    return x * 0.5 * (1 + (0.7978845608 * x * (1.0 + 0.044715 * x * x)))


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
def test_tmem_copy_rejects_logical_rank2_scale_smem(device):

    @triton.jit
    def kernel(dummy):
        smem = tlx.local_alloc((128, 4), tl.uint8, tl.constexpr(1))
        tmem = tlx.local_alloc((128, 4), tl.uint8, tl.constexpr(1), tlx.storage_kind.tmem)
        tlx.tmem_copy(smem[0], tmem[0])
        tl.store(dummy, tl.full((), 0, tl.int32))

    dummy = torch.empty((), dtype=torch.int32, device=device)
    with pytest.raises(triton.CompilationError) as exc_info:
        kernel[(1, )](dummy)
    assert "scale tmem_copy requires an explicit packed i8 SMEM shape" in str(exc_info.value)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
@pytest.mark.parametrize("num_blocks", [1, 2])
def test_tmem_copy_accepts_packed_rank2_scale_smem(num_blocks, device):

    @triton.jit
    def kernel(NUM_BLOCKS: tl.constexpr):
        smem = tlx.local_alloc((32 * NUM_BLOCKS, 16), tl.uint8, tl.constexpr(1))
        tmem = tlx.local_alloc((128, 16 * NUM_BLOCKS), tl.uint8, tl.constexpr(1), tlx.storage_kind.tmem)
        tlx.tmem_copy(smem[0], tmem[0])

    compiled = kernel.warmup(num_blocks, grid=(1, ))
    assert compiled.asm["ttgir"].count("ttng.tmem_copy") == 1


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
@pytest.mark.parametrize("SCALE_OFFSET", [0, 8])
def test_tmem_scale_subslice_compile(SCALE_OFFSET):
    SCALE_BLOCK_N = 16
    SCALE_SLICE_N = 8
    BLOCK_M = 128
    BLOCK_K = SCALE_BLOCK_N * 32
    BLOCK_N = 64

    @triton.jit
    def tmem_scale_subslice_compile_kernel(BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
                                           SCALE_BLOCK_N: tl.constexpr, SCALE_SLICE_N: tl.constexpr,
                                           SCALE_OFFSET: tl.constexpr):
        a_smem = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float8e4nv, tl.constexpr(1))
        b_smem = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float8e4nv, tl.constexpr(1))
        acc_tmem = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        a_scale_tmem = tlx.local_alloc((BLOCK_M, SCALE_BLOCK_N), tl.uint8, tl.constexpr(1), tlx.storage_kind.tmem)
        b_scale_tmem = tlx.local_alloc((BLOCK_N, SCALE_BLOCK_N), tl.uint8, tl.constexpr(1), tlx.storage_kind.tmem)
        scale_smem = tlx.local_alloc((1, 1, 2, 2, 256), tl.uint8, tl.constexpr(1))
        scale_slice = tlx.subslice(a_scale_tmem[0], SCALE_OFFSET, SCALE_SLICE_N)
        tlx.tmem_copy(scale_smem[0], scale_slice)
        tlx.async_dot_scaled(a_smem[0], b_smem[0], acc_tmem[0], a_scale_tmem[0], "e4m3", b_scale_tmem[0], "e4m3",
                             use_acc=tl.constexpr(False))

    kernel = tmem_scale_subslice_compile_kernel.warmup(BLOCK_M, BLOCK_K, BLOCK_N, SCALE_BLOCK_N, SCALE_SLICE_N,
                                                       SCALE_OFFSET, grid=(1, ))
    ttgir = kernel.asm["ttgir"]
    subslice_ops = [line.strip() for line in ttgir.splitlines() if "ttng.tmem_subslice" in line]
    copy_ops = [line.strip() for line in ttgir.splitlines() if "ttng.tmem_copy" in line]

    assert "tensor_memory_scales_encoding" in ttgir
    assert len(subslice_ops) == 1
    assert len(copy_ops) == 1
    assert f"offset = {SCALE_OFFSET} : i32" in subslice_ops[0]
    assert f"!ttg.memdesc<128x{SCALE_BLOCK_N}xi8" in subslice_ops[0]
    assert f"!ttg.memdesc<128x{SCALE_SLICE_N}xi8" in subslice_ops[0]
    assert f"!ttg.memdesc<128x{SCALE_SLICE_N}xi8" in copy_ops[0]
    assert "tcgen05.cp" in kernel.asm["ptx"]


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
