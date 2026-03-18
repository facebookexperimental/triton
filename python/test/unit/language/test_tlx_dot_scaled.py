import math
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell
import triton.language.extra.tlx as tlx
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor

from test.unit.language.conftest_tlx import _swizzle_scale_to_5d


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_scaled_2cta(device):
    """
    Test 2-CTA scaled MMA generates tcgen05.mma.cta_group::2 instruction.
    Also verifies numerical correctness against reference implementation.
    """

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device=device)

    @triton.jit
    def tcgen5_dot_scaled_2cta_kernel(
        a_ptr,
        stride_am,
        stride_ak,
        b_ptr,
        stride_bk,
        stride_bn,
        a_scale_ptr,
        b_scale_ptr,
        c_ptr,
        stride_cm,
        stride_cn,
        A_format: tl.constexpr,
        B_format: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
    ):
        # difference from 1cta
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_cta0 = cluster_cta_rank == 0
        cta_bars = tlx.alloc_barriers(num_barriers=1, arrive_count=2)  # CTA0 waits for signals from both CTAs

        desc_a = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_M, BLOCK_K],
        )

        # difference from 1cta: B is split across 2 CTAs
        desc_b = tl.make_tensor_descriptor(
            b_ptr,
            shape=[K, N],
            strides=[stride_bk, stride_bn],
            block_shape=[BLOCK_K, BLOCK_N // 2],
        )

        desc_a_scale = tl.make_tensor_descriptor(
            a_scale_ptr,
            shape=[M // 128, K // 32 // 4, 2, 2 * 128],
            strides=[K // 32 // 4 * 2 * 2 * 128, 2 * 2 * 128, 2 * 128, 1],
            block_shape=[BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128],
        )

        # B scale is NOT split across CTAs - full scale needed for MMA
        desc_b_scale = tl.make_tensor_descriptor(
            b_scale_ptr,
            shape=[N // 128, K // 32 // 4, 2, 2 * 128],
            strides=[K // 32 // 4 * 2 * 2 * 128, 2 * 2 * 128, 2 * 128, 1],
            block_shape=[BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128],
        )

        # async load a and b into SMEM
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float8e4nv, tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_K, BLOCK_N // 2), tl.float8e4nv, tl.constexpr(1))  # difference from 1cta
        a_scale_tile = tlx.local_alloc((BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128), tl.uint8, tl.constexpr(1))
        # B scale tile is NOT halved - full scale for MMA
        b_scale_tile = tlx.local_alloc((BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128), tl.uint8, tl.constexpr(1))

        bars = tlx.alloc_barriers(tl.constexpr(4))
        bar_a = tlx.local_view(bars, 0)
        bar_b = tlx.local_view(bars, 1)
        bar_a_scale = tlx.local_view(bars, 2)
        bar_b_scale = tlx.local_view(bars, 3)
        tlx.barrier_expect_bytes(bar_a, BLOCK_M * BLOCK_K * 1)  # fp8
        tlx.barrier_expect_bytes(bar_b, BLOCK_K * (BLOCK_N // 2) * 1)  # difference from 1cta: B is half
        tlx.barrier_expect_bytes(bar_a_scale, BLOCK_M // 128 * BLOCK_K // 32 // 4 * 2 * 2 * 128)
        tlx.barrier_expect_bytes(bar_b_scale, BLOCK_N // 128 * BLOCK_K // 32 // 4 * 2 * 2 * 128)  # full B scale

        # difference from 1cta: A offset by CTA rank, B offset by CTA rank
        tlx.async_descriptor_load(desc_a, a_tile[0], [cluster_cta_rank * BLOCK_M, 0], bar_a)
        tlx.async_descriptor_load(desc_b, b_tile[0], [0, cluster_cta_rank * BLOCK_N // 2], bar_b)
        tlx.async_descriptor_load(desc_a_scale, a_scale_tile[0], [cluster_cta_rank * BLOCK_M // 128, 0, 0, 0],
                                  bar_a_scale)
        tlx.async_descriptor_load(desc_b_scale, b_scale_tile[0], [0, 0, 0, 0], bar_b_scale)  # full B scale

        tlx.barrier_wait(bar_a, tl.constexpr(0))
        tlx.barrier_wait(bar_b, tl.constexpr(0))
        tlx.barrier_wait(bar_a_scale, tl.constexpr(0))
        tlx.barrier_wait(bar_b_scale, tl.constexpr(0))

        # difference from 1cta: CTA0 waits for both CTAs before issuing MMA op
        # "Arrive Remote, Wait Local" pattern: all CTAs signal CTA 0's barrier, only CTA 0 waits
        tlx.barrier_arrive(cta_bars[0], 1, remote_cta_rank=0)
        tlx.barrier_wait(cta_bars[0], phase=0, pred=pred_cta0)

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)

        # Allocate barrier for MMA completion
        mma_done_bars = tlx.alloc_barriers(tl.constexpr(1))
        mma_done_bar = tlx.local_view(mma_done_bars, 0)

        # difference from 1cta: set two_ctas. Compiler auto generates pred to issue mma only from CTA0
        # Pass mma_done_bar directly to async_dot_scaled for MMA completion signaling
        tlx.async_dot_scaled(
            a_tile[0],
            b_tile[0],
            c_tile[0],
            a_scale_tile[0],
            A_format,
            b_scale_tile[0],
            B_format,
            use_acc=False,
            two_ctas=True,
            mBarriers=[mma_done_bar],
        )

        # Wait for MMA completion
        tlx.barrier_wait(mma_done_bar, tl.constexpr(0))

        result = tlx.local_load(c_tile[0])

        c = result.to(tl.float16)
        offs_m = cluster_cta_rank * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        tl.store(c_ptrs, c)

    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    # M=256 so BLOCK_M=128 per CTA, N=256 so BLOCK_N=256 total (128 per CTA for B data)
    M, N, K = (256, 256, 128)

    DTYPE_MAP = {
        "e5m2": torch.float8_e5m2,
        "e4m3": torch.float8_e4m3fn,
    }

    A_DATA_TYPE = "e4m3"
    B_DATA_TYPE = "e4m3"

    a = torch.randint(20, 40, (M, K), dtype=torch.uint8).to(DTYPE_MAP[A_DATA_TYPE]).to(device)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8).to(DTYPE_MAP[B_DATA_TYPE]).to(device)
    c = torch.zeros((M, N), device=device, dtype=torch.float16)

    a_scale = torch.randint(124, 130, (M, K // 32), dtype=torch.uint8, device=device)
    b_scale = torch.randint(124, 130, (N, K // 32), dtype=torch.uint8, device=device)
    a_scale_4d = _swizzle_scale_to_5d(a_scale.reshape(1, M, K // 32), M // 128, K // 32 // 4).squeeze(0)
    b_scale_4d = _swizzle_scale_to_5d(b_scale.reshape(1, N, K // 32), N // 128, K // 32 // 4).squeeze(0)

    BLOCK_M = M // 2  # 128 per CTA
    BLOCK_N = N  # 256 total, 128 per CTA for B data
    BLOCK_K = K
    kern_kwargs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_K": BLOCK_K,
        "BLOCK_N": BLOCK_N,
        "M": M,
        "N": N,
        "K": K,
    }
    kernel = tcgen5_dot_scaled_2cta_kernel[(M // BLOCK_M, N // BLOCK_N)](
        a,
        a.stride(0),
        a.stride(1),
        b,
        b.stride(0),
        b.stride(1),
        a_scale_4d,
        b_scale_4d,
        c,
        c.stride(0),
        c.stride(1),
        A_DATA_TYPE,
        B_DATA_TYPE,
        ctas_per_cga=(2, 1, 1),  # TLX way: explicitly set cluster dims
        **kern_kwargs,
    )

    # verify kernel launch cluster
    assert kernel.metadata.cluster_dims == (2, 1, 1), (
        f"expecting cluster dim to be (2, 1, 1), got {kernel.metadata.cluster_dims}")
    assert kernel.metadata.num_ctas == 1, (
        f"expecting num_ctas to be 1 when using ctas_per_cga, got {kernel.metadata.num_ctas}")

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("nvgpu.cluster_id") == 1
    assert ttgir.count("ttng.map_to_remote_buffer") == 1
    assert ttgir.count("ttng.tc_gen5_mma_scaled") >= 1

    ptx = kernel.asm["ptx"]
    # The key assertion: with two_ctas=True, should generate cta_group::2 for scaled MMA
    assert ptx.count("tcgen05.mma.cta_group::2") > 0, (
        f"Expected tcgen05.mma.cta_group::2 for 2-CTA scaled MMA, but found: "
        f"cta_group::1 count={ptx.count('tcgen05.mma.cta_group::1')}, "
        f"cta_group::2 count={ptx.count('tcgen05.mma.cta_group::2')}")

    # Numeric verification: compute reference and compare
    def fp8e8m0_to_float32(scale):
        """Convert FP8 E8M0 scale values to float32."""
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    # Compute reference: D = (A * A_scale) @ (B * B_scale)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    # Repeat each scale value 32 times along K dimension
    a_scale_f32 = a_scale_f32.repeat_interleave(32, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(32, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float16)

    atol = 1e-2 * math.sqrt(K / 32)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


@pytest.mark.parametrize("A_DATA_TYPE", ["e5m2", "e4m3"])
@pytest.mark.parametrize("B_DATA_TYPE", ["e5m2", "e4m3"])
@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_scaled(A_DATA_TYPE, B_DATA_TYPE, device):
    """
    Test D = (A * A_scale)  * (B * B_scale) with mxfp8 format for both A and B.

    Scale layout uses 5D TMA descriptor [1, rep_m, rep_k, 2, 256] with uint8 elements,
    matching cuBLAS block scaling layout.
    """

    VEC_SIZE = 32  # mxfp8 uses 32 elements per scale factor

    @triton.jit
    def tcgen5_dot_scaled_kernel(
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        A_format: tl.constexpr,
        B_format: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Scale tile dimensions for 5D TMA (per cuBLAS block scaling layout)
        REP_M: tl.constexpr = triton.cdiv(BLOCK_M, 128)
        REP_N: tl.constexpr = triton.cdiv(BLOCK_N, 128)
        REP_K: tl.constexpr = triton.cdiv(BLOCK_K, 128)

        # Allocate SMEM buffers
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), tl.constexpr(1))
        # 5D scale buffers: [1, REP_M/N, REP_K, 2, 256] for cuBLAS block scaling layout
        a_scale_tile = tlx.local_alloc((1, REP_M, REP_K, 2, 256), tlx.dtype_of(a_scale_desc), tl.constexpr(1))
        b_scale_tile = tlx.local_alloc((1, REP_N, REP_K, 2, 256), tlx.dtype_of(b_scale_desc), tl.constexpr(1))

        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        DATA_BYTES: tl.constexpr = BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N
        SCALE_BYTES: tl.constexpr = (REP_M + REP_N) * REP_K * 2 * 256
        tlx.barrier_expect_bytes(load_bar[0], DATA_BYTES + SCALE_BYTES)
        tlx.async_descriptor_load(a_desc, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(b_desc, b_tile[0], [0, 0], load_bar)
        # 5D offset with leading 0
        tlx.async_descriptor_load(a_scale_desc, a_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.async_descriptor_load(b_scale_desc, b_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        tlx.async_dot_scaled(a_tile[0], b_tile[0], c_tile[0], a_scale_tile[0], A_format, b_scale_tile[0], B_format,
                             use_acc=False)

        result = tlx.local_load(c_tile[0])
        c = result.to(tlx.dtype_of(c_desc))
        c_desc.store([0, 0], c)

    torch.manual_seed(0)
    M, N, K = (128, 128, 256)
    BLOCK_M, BLOCK_N, BLOCK_K = (M, N, K)

    DTYPE_MAP = {
        "e5m2": torch.float8_e5m2,
        "e4m3": torch.float8_e4m3fn,
    }

    a = torch.randint(20, 40, (M, K), dtype=torch.uint8).to(DTYPE_MAP[A_DATA_TYPE]).to(device)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8).to(DTYPE_MAP[B_DATA_TYPE]).to(device)
    c = torch.zeros((M, N), device=device, dtype=torch.float16)
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])
    c_desc = TensorDescriptor.from_tensor(c, block_shape=[BLOCK_M, BLOCK_N])

    # Create E8M0 scale tensors using 5D TMA layout: [1, rep_m, rep_k, 2, 256]
    a_scale = torch.randint(124, 130, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(124, 130, (N, K // VEC_SIZE), dtype=torch.uint8, device=device)

    # Swizzle to 5D cuBLAS block scaling layout for TMA: [1, rep_m, rep_k, 2, 256]
    a_scale_5d = _swizzle_scale_to_5d(a_scale.reshape(1, M, K // VEC_SIZE), M // 128, K // VEC_SIZE // 4)
    b_scale_5d = _swizzle_scale_to_5d(b_scale.reshape(1, N, K // VEC_SIZE), N // 128, K // VEC_SIZE // 4)

    a_scale_block_shape = [1, BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    b_scale_block_shape = [1, BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    a_scale_desc = TensorDescriptor.from_tensor(a_scale_5d, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale_5d, block_shape=b_scale_block_shape)

    kern_kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_K": BLOCK_K, "BLOCK_N": BLOCK_N}
    kernel = tcgen5_dot_scaled_kernel[(1, 1)](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        A_DATA_TYPE,
        B_DATA_TYPE,
        **kern_kwargs,
    )

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.async_tma_copy_global_to_local") == 4
    assert ttgir.count("ttng.tc_gen5_mma_scaled") == 1

    # Converts E8M0 format scale values to float32 by bit-shifting the exponent bits
    # into the correct position for IEEE 754 float32 representation
    def fp8e8m0_to_float32(scale):
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    # Compute reference (use original 2D scales, not swizzled 5D)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    # Repeats each scale value VEC_SIZE times along dimension 1.
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / VEC_SIZE)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_scaled_tmem_scales(device):
    """
    Test D = (A * A_scale) * (B * B_scale) with mxfp8 format and TMEM scales.

    This test verifies that scales can be stored in tensor memory (TMEM) instead
    of shared memory (SMEM). The scales are first loaded to SMEM via TMA, then
    copied to TMEM for use in the scaled MMA operation.
    """

    VEC_SIZE = 32  # mxfp8 uses 32 elements per scale factor

    @triton.jit
    def tcgen5_dot_scaled_tmem_scales_kernel(
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        A_format: tl.constexpr,
        B_format: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Scale tile dimensions for 5D TMA (per cuBLAS block scaling layout)
        REP_M: tl.constexpr = BLOCK_M // 128
        REP_N: tl.constexpr = BLOCK_N // 128
        REP_K: tl.constexpr = triton.cdiv(BLOCK_K // 32, 4)

        # Allocate SMEM buffers for A, B, and scales
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), tl.constexpr(1))
        # 5D scale buffers in SMEM: [1, REP_M/N, REP_K, 2, 256]
        a_scale_smem = tlx.local_alloc((1, REP_M, REP_K, 2, 256), tlx.dtype_of(a_scale_desc), tl.constexpr(1))
        b_scale_smem = tlx.local_alloc((1, REP_N, REP_K, 2, 256), tlx.dtype_of(b_scale_desc), tl.constexpr(1))

        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        DATA_BYTES: tl.constexpr = BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N
        SCALE_BYTES: tl.constexpr = (REP_M + REP_N) * REP_K * 2 * 256
        tlx.barrier_expect_bytes(load_bar[0], DATA_BYTES + SCALE_BYTES)
        tlx.async_descriptor_load(a_desc, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(b_desc, b_tile[0], [0, 0], load_bar)
        # Load scales to SMEM via TMA
        tlx.async_descriptor_load(a_scale_desc, a_scale_smem[0], [0, 0, 0, 0, 0], load_bar)
        tlx.async_descriptor_load(b_scale_desc, b_scale_smem[0], [0, 0, 0, 0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        # Allocate TMEM for scales and accumulator
        # Scale shape in TMEM: flatten 5D to 2D for TMEM storage
        SCALE_K: tl.constexpr = BLOCK_K // 32
        SCALE_N: tl.constexpr = BLOCK_N // 32
        a_scale_tmem = tlx.local_alloc((BLOCK_M, SCALE_K), tl.uint8, tl.constexpr(1), tlx.storage_kind.tmem)
        b_scale_tmem = tlx.local_alloc((BLOCK_K, SCALE_N), tl.uint8, tl.constexpr(1), tlx.storage_kind.tmem)

        # Copy scales from SMEM to TMEM directly using tmem_copy
        tlx.tmem_copy(a_scale_smem[0], a_scale_tmem[0])
        tlx.tmem_copy(b_scale_smem[0], b_scale_tmem[0])

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        # Use TMEM scales in async_dot_scaled
        tlx.async_dot_scaled(a_tile[0], b_tile[0], c_tile[0], a_scale_tmem[0], A_format, b_scale_tmem[0], B_format,
                             use_acc=False)

        result = tlx.local_load(c_tile[0])
        c = result.to(tlx.dtype_of(c_desc))
        c_desc.store([0, 0], c)

    torch.manual_seed(0)
    M, N, K = (128, 128, 256)
    BLOCK_M, BLOCK_N, BLOCK_K = (M, N, K)

    A_DATA_TYPE = "e4m3"
    B_DATA_TYPE = "e4m3"

    DTYPE_MAP = {
        "e5m2": torch.float8_e5m2,
        "e4m3": torch.float8_e4m3fn,
    }

    a = torch.randint(20, 40, (M, K), dtype=torch.uint8).to(DTYPE_MAP[A_DATA_TYPE]).to(device)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8).to(DTYPE_MAP[B_DATA_TYPE]).to(device)
    c = torch.zeros((M, N), device=device, dtype=torch.float16)
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])
    c_desc = TensorDescriptor.from_tensor(c, block_shape=[BLOCK_M, BLOCK_N])

    # Create E8M0 scale tensors using 5D TMA layout: [1, rep_m, rep_k, 2, 256]
    a_scale = torch.randint(124, 130, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(124, 130, (N, K // VEC_SIZE), dtype=torch.uint8, device=device)

    # Swizzle to 5D cuBLAS block scaling layout for TMA: [1, rep_m, rep_k, 2, 256]
    a_scale_5d = _swizzle_scale_to_5d(a_scale.reshape(1, M, K // VEC_SIZE), M // 128, K // VEC_SIZE // 4)
    b_scale_5d = _swizzle_scale_to_5d(b_scale.reshape(1, N, K // VEC_SIZE), N // 128, K // VEC_SIZE // 4)

    a_scale_block_shape = [1, BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    b_scale_block_shape = [1, BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    a_scale_desc = TensorDescriptor.from_tensor(a_scale_5d, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale_5d, block_shape=b_scale_block_shape)

    kern_kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_K": BLOCK_K, "BLOCK_N": BLOCK_N}
    kernel = tcgen5_dot_scaled_tmem_scales_kernel[(1, 1)](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        A_DATA_TYPE,
        B_DATA_TYPE,
        **kern_kwargs,
    )

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.async_tma_copy_global_to_local") == 4
    assert ttgir.count("ttng.tc_gen5_mma_scaled") == 1
    # Verify TMEM scales encoding is used
    assert "tensor_memory_scales_encoding" in ttgir
    # Verify tmem_copy is used for SMEM->TMEM transfer
    assert ttgir.count("ttng.tmem_copy") == 2

    # Converts E8M0 format scale values to float32
    def fp8e8m0_to_float32(scale):
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    # Compute reference (use original 2D scales, not swizzled 5D)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / VEC_SIZE)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_tmem_buffer_scales_two_entries(device):
    """
    Test storing to a TMEM buffer for scales with 2 entries.
    Stores all 0s (uint8) to entry 0 and all 127s (uint8) to entry 1,
    then verifies correctness by using each entry as scales in a
    separate scaled MMA operation.

    In E8M0 encoding, byte 0 maps to float 0.0 (so MMA result is zero)
    and byte 127 maps to 2^(127-127) = 1.0 (so MMA result equals the
    unscaled matmul).
    """

    @triton.jit
    def kernel(
        a_desc,
        b_desc,
        c0_desc,
        c1_desc,
        A_format: tl.constexpr,
        B_format: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        SCALE_K: tl.constexpr = BLOCK_K // 32
        SCALE_N: tl.constexpr = BLOCK_N // 32

        # Load A, B to SMEM via TMA
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), tl.constexpr(1))
        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        DATA_BYTES: tl.constexpr = BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N
        tlx.barrier_expect_bytes(load_bar[0], DATA_BYTES)
        tlx.async_descriptor_load(a_desc, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(b_desc, b_tile[0], [0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        # Allocate TMEM scale buffers with 2 entries
        a_scale_tmem = tlx.local_alloc((BLOCK_M, SCALE_K), tl.uint8, tl.constexpr(2), tlx.storage_kind.tmem)
        b_scale_tmem = tlx.local_alloc((BLOCK_K, SCALE_N), tl.uint8, tl.constexpr(2), tlx.storage_kind.tmem)

        # Entry 0: store all 0s
        tlx.local_store(a_scale_tmem[0], tl.full((BLOCK_M, SCALE_K), 0, tl.uint8))
        tlx.local_store(b_scale_tmem[0], tl.full((BLOCK_K, SCALE_N), 0, tl.uint8))

        # Entry 1: store all 127s
        tlx.local_store(a_scale_tmem[1], tl.full((BLOCK_M, SCALE_K), 127, tl.uint8))
        tlx.local_store(b_scale_tmem[1], tl.full((BLOCK_K, SCALE_N), 127, tl.uint8))

        # Accumulator in TMEM
        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)

        # MMA with entry 0 scales
        tlx.async_dot_scaled(a_tile[0], b_tile[0], c_tile[0], a_scale_tmem[0], A_format, b_scale_tmem[0], B_format,
                             use_acc=False)
        result0 = tlx.local_load(c_tile[0])
        c0_desc.store([0, 0], result0.to(tlx.dtype_of(c0_desc)))

        # MMA with entry 1 scales
        tlx.async_dot_scaled(a_tile[0], b_tile[0], c_tile[0], a_scale_tmem[1], A_format, b_scale_tmem[1], B_format,
                             use_acc=False)
        result1 = tlx.local_load(c_tile[0])
        c1_desc.store([0, 0], result1.to(tlx.dtype_of(c1_desc)))

    torch.manual_seed(0)
    M, N, K = 128, 128, 256
    BLOCK_M, BLOCK_N, BLOCK_K = M, N, K

    A_DATA_TYPE = "e4m3"
    B_DATA_TYPE = "e4m3"

    a = torch.randint(20, 40, (M, K), dtype=torch.uint8).to(torch.float8_e4m3fn).to(device)
    b = torch.randint(20, 40, (K, N), dtype=torch.uint8).to(torch.float8_e4m3fn).to(device)
    c0 = torch.zeros((M, N), device=device, dtype=torch.float16)
    c1 = torch.zeros((M, N), device=device, dtype=torch.float16)

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])
    c0_desc = TensorDescriptor.from_tensor(c0, block_shape=[BLOCK_M, BLOCK_N])
    c1_desc = TensorDescriptor.from_tensor(c1, block_shape=[BLOCK_M, BLOCK_N])

    kernel[(1, 1)](
        a_desc,
        b_desc,
        c0_desc,
        c1_desc,
        A_DATA_TYPE,
        B_DATA_TYPE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    VEC_SIZE = 32

    # E8M0 byte 0 → float 0.0, so result is exactly 0
    torch.testing.assert_close(c0, torch.zeros_like(c0), atol=0, rtol=0)

    # E8M0 byte 127 → float 2^(127-127) = 1.0, so result equals unscaled matmul
    ref_c1 = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / VEC_SIZE)
    torch.testing.assert_close(c1, ref_c1, atol=atol, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_scaled_mxfp4(device):
    """
    Test D = (A * A_scale) * (B * B_scale) with mxfp4 (e2m1) format for both A and B.

    For mxfp4 format:
    - Two fp4 (e2m1) elements are packed into a single uint8
    - A has logical shape (M, K), packed along K to get physical shape (M, K//2)
    - B is stored in transposed layout (N, K), packed along K to get (N, K//2)
    - B is transposed in SMEM before being passed to MMA to get (K//2, N)

    Scale layout uses 5D TMA descriptor [1, rep_m, rep_k, 2, 256] with uint8 elements,
    matching cuBLAS block scaling layout.
    """
    from triton.tools.mxfp import MXFP4Tensor

    VEC_SIZE = 32  # mxfp4 uses 32 elements per scale factor

    @triton.jit
    def tcgen5_dot_scaled_mxfp4_kernel(
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Scale tile dimensions for 5D TMA (per cuBLAS block scaling layout)
        REP_M: tl.constexpr = triton.cdiv(BLOCK_M, 128)
        REP_N: tl.constexpr = triton.cdiv(BLOCK_N, 128)
        REP_K: tl.constexpr = triton.cdiv(BLOCK_K, 128)

        # Allocate SMEM buffers
        # A: (M, K//2) - packed along K
        # B: (N, K//2) - stored in transposed layout, packed along K
        a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K // 2), tl.uint8, tl.constexpr(1))
        b_tile = tlx.local_alloc((BLOCK_N, BLOCK_K // 2), tl.uint8, tl.constexpr(1))
        # 5D scale buffers: [1, REP_M/N, REP_K, 2, 256] for cuBLAS block scaling layout
        a_scale_tile = tlx.local_alloc((1, REP_M, REP_K, 2, 256), tl.uint8, tl.constexpr(1))
        b_scale_tile = tlx.local_alloc((1, REP_N, REP_K, 2, 256), tl.uint8, tl.constexpr(1))

        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        DATA_BYTES: tl.constexpr = BLOCK_M * BLOCK_K // 2 + BLOCK_N * BLOCK_K // 2
        SCALE_BYTES: tl.constexpr = (REP_M + REP_N) * REP_K * 2 * 256
        tlx.barrier_expect_bytes(load_bar[0], DATA_BYTES + SCALE_BYTES)
        tlx.async_descriptor_load(a_desc, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(b_desc, b_tile[0], [0, 0], load_bar)
        # 5D offset with leading 0
        tlx.async_descriptor_load(a_scale_desc, a_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.async_descriptor_load(b_scale_desc, b_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        # Transpose B from (N, K//2) to (K//2, N) for MMA
        b_tile_T = tlx.local_trans(b_tile[0])

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        tlx.async_dot_scaled(a_tile[0], b_tile_T, c_tile[0], a_scale_tile[0], "e2m1", b_scale_tile[0], "e2m1",
                             use_acc=False)

        result = tlx.local_load(c_tile[0])
        c = result.to(tlx.dtype_of(c_desc))
        c_desc.store([0, 0], c)

    torch.manual_seed(0)
    M, N, K = (128, 128, 128)
    BLOCK_M, BLOCK_N, BLOCK_K = (M, N, K)

    # Create mxfp4 tensors and pack them
    # A has logical shape (M, K), packed along K to get physical shape (M, K//2)

    A = torch.full((M, K), 2, dtype=torch.float32, device=device)
    B = torch.full((N, K), 2, dtype=torch.float32, device=device)
    AMXFP4 = MXFP4Tensor(data=A, device=device)
    BMXFP4 = MXFP4Tensor(data=B, device=device)
    APACKED = AMXFP4.to_packed_tensor(dim=1)
    BPACKED = BMXFP4.to_packed_tensor(dim=1)

    a_ref = AMXFP4.to(torch.float32)

    # B is stored in transposed layout (N, K), packed along K to get (N, K//2)
    # This matches the hardware expectation for mxfp4
    b_ref = BMXFP4.to(torch.float32).T  # Transpose for reference matmul -> (K, N)

    c = torch.zeros((M, N), device=device, dtype=torch.float16)

    # TMA descriptors for packed mxfp4 data
    a_desc = TensorDescriptor.from_tensor(APACKED, [BLOCK_M, BLOCK_K // 2])
    b_desc = TensorDescriptor.from_tensor(BPACKED, [BLOCK_N, BLOCK_K // 2])  # B stored as (N, K//2)
    c_desc = TensorDescriptor.from_tensor(c, block_shape=[BLOCK_M, BLOCK_N])

    # Create E8M0 scale tensors using 5D TMA layout: [1, rep_m, rep_k, 2, 256]
    # This matches cuBLAS block scaling layout used by tcgen5_mma_scaled
    a_scale = torch.randint(127, 128, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(127, 128, (N, K // VEC_SIZE), dtype=torch.uint8, device=device)

    # Swizzle to 5D cuBLAS block scaling layout for TMA: [1, rep_m, rep_k, 2, 256]
    a_scale_5d = _swizzle_scale_to_5d(a_scale.reshape(1, M, K // VEC_SIZE), M // 128, K // VEC_SIZE // 4)
    b_scale_5d = _swizzle_scale_to_5d(b_scale.reshape(1, N, K // VEC_SIZE), N // 128, K // VEC_SIZE // 4)

    a_scale_block_shape = [1, BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    b_scale_block_shape = [1, BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    a_scale_desc = TensorDescriptor.from_tensor(a_scale_5d, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale_5d, block_shape=b_scale_block_shape)

    kern_kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_K": BLOCK_K, "BLOCK_N": BLOCK_N}
    kernel = tcgen5_dot_scaled_mxfp4_kernel[(1, 1)](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        **kern_kwargs,
    )

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.async_tma_copy_global_to_local") == 4
    assert ttgir.count("ttng.tc_gen5_mma_scaled") == 1

    # Converts E8M0 format scale values to float32 by bit-shifting the exponent bits
    # into the correct position for IEEE 754 float32 representation
    def fp8e8m0_to_float32(scale):
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    # Compute reference (use original 2D scales, not swizzled 5D)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    # Repeat each scale value VEC_SIZE times along dim 1
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a_ref * a_scale_f32, b_ref * b_scale_f32).to(torch.float16)
    atol = 1e-2 * math.sqrt(K / 32)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


@pytest.mark.parametrize(
    "A_format,B_format",
    [("e4m3", "e2m1"),  # A is mxfp8, B is mxfp4
     ("e2m1", "e4m3"),  # A is mxfp4, B is mxfp8
     ],
)
@pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
def test_async_dot_scaled_mixed_mxfp8_mxfp4(A_format, B_format, device):
    """
    Test D = (A * A_scale) * (B * B_scale) with mixed mxfp8 (e4m3) and mxfp4 (e2m1) formats.

    This test exercises the fp4Padded logic in TLX's async_dot_scaled:
    - When A is mxfp4 and B is mxfp8: A_fp4Padded=True, B_fp4Padded=False
    - When A is mxfp8 and B is mxfp4: A_fp4Padded=False, B_fp4Padded=True

    For mxfp4 format:
    - Two fp4 (e2m1) elements are packed into a single uint8
    - Tensor is packed along K dimension, so shape (M, K) becomes (M, K//2)
    - B is stored transposed as (N, K//2) and transposed in SMEM to (K//2, N)

    For mxfp8 format:
    - Standard fp8 e4m3 layout with shape (M, K) or (K, N)

    Scale layout uses 5D TMA descriptor [1, rep_m, rep_k, 2, 256] with uint8 elements (cuBLAS block scaling layout).
    """
    from triton.tools.mxfp import MXFP4Tensor

    VEC_SIZE = 32  # mxfp uses 32 elements per scale factor

    @triton.jit
    def tcgen5_dot_scaled_mixed_kernel(
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        A_format: tl.constexpr,
        B_format: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        A_IS_FP4: tl.constexpr,
        B_IS_FP4: tl.constexpr,
    ):
        # Scale tile dimensions for 5D TMA
        REP_M: tl.constexpr = triton.cdiv(BLOCK_M, 128)
        REP_N: tl.constexpr = triton.cdiv(BLOCK_N, 128)
        REP_K: tl.constexpr = triton.cdiv(BLOCK_K, 128)

        # Allocate SMEM buffers
        # For FP4: packed along K, so (M, K//2) or (N, K//2)
        # For FP8: full size (M, K) or (K, N)
        if A_IS_FP4:
            a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K // 2), tl.uint8, tl.constexpr(1))
        else:
            a_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_desc), tl.constexpr(1))

        if B_IS_FP4:
            # B is stored transposed as (N, K//2) for FP4
            b_tile = tlx.local_alloc((BLOCK_N, BLOCK_K // 2), tl.uint8, tl.constexpr(1))
        else:
            # B is (K, N) for FP8
            b_tile = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_desc), tl.constexpr(1))

        # 5D scale buffers: [1, REP_M/N, REP_K, 2, 256]
        a_scale_tile = tlx.local_alloc((1, REP_M, REP_K, 2, 256), tl.uint8, tl.constexpr(1))
        b_scale_tile = tlx.local_alloc((1, REP_N, REP_K, 2, 256), tl.uint8, tl.constexpr(1))

        # Calculate expected bytes for barrier
        if A_IS_FP4:
            A_BYTES: tl.constexpr = BLOCK_M * BLOCK_K // 2
        else:
            A_BYTES: tl.constexpr = BLOCK_M * BLOCK_K  # FP8 is 1 byte per element

        if B_IS_FP4:
            B_BYTES: tl.constexpr = BLOCK_N * BLOCK_K // 2
        else:
            B_BYTES: tl.constexpr = BLOCK_K * BLOCK_N  # FP8 is 1 byte per element

        SCALE_BYTES: tl.constexpr = (REP_M + REP_N) * REP_K * 2 * 256

        load_bar = tlx.alloc_barriers(tl.constexpr(1))
        tlx.barrier_expect_bytes(load_bar[0], A_BYTES + B_BYTES + SCALE_BYTES)
        tlx.async_descriptor_load(a_desc, a_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(b_desc, b_tile[0], [0, 0], load_bar)
        tlx.async_descriptor_load(a_scale_desc, a_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.async_descriptor_load(b_scale_desc, b_scale_tile[0], [0, 0, 0, 0, 0], load_bar)
        tlx.barrier_wait(load_bar[0], 0)

        # Transpose B from (N, K//2) to (K//2, N) for FP4, or use as-is for FP8
        if B_IS_FP4:
            b_tile_for_mma = tlx.local_trans(b_tile[0])
        else:
            b_tile_for_mma = b_tile[0]

        c_tile = tlx.local_alloc((BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem)
        tlx.async_dot_scaled(a_tile[0], b_tile_for_mma, c_tile[0], a_scale_tile[0], A_format, b_scale_tile[0], B_format,
                             use_acc=False)

        result = tlx.local_load(c_tile[0])
        c = result.to(tlx.dtype_of(c_desc))
        c_desc.store([0, 0], c)

    torch.manual_seed(0)
    M, N, K = (128, 128, 128)
    BLOCK_M, BLOCK_N, BLOCK_K = (M, N, K)

    A_IS_FP4 = A_format == "e2m1"
    B_IS_FP4 = B_format == "e2m1"

    # Create input tensors based on format
    if A_IS_FP4:
        # mxfp4: Create packed tensor (M, K//2)
        a_mxfp4 = MXFP4Tensor(data=torch.full((M, K), 2, dtype=torch.float32, device=device), device=device)
        a = a_mxfp4.to_packed_tensor(dim=1)  # Pack along K -> (M, K//2)
        a_ref = a_mxfp4.to(torch.float32)
        a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // 2])
    else:
        # mxfp8: Standard fp8 tensor (M, K)
        a = torch.randint(20, 40, (M, K), dtype=torch.uint8).to(torch.float8_e4m3fn).to(device)
        a_ref = a.to(torch.float32)
        a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])

    if B_IS_FP4:
        # mxfp4: Create packed tensor stored as (N, K//2), will be transposed in SMEM
        b_mxfp4 = MXFP4Tensor(data=torch.full((N, K), 2, dtype=torch.float32, device=device), device=device)
        b = b_mxfp4.to_packed_tensor(dim=1)  # Pack along K -> (N, K//2)
        b_ref = b_mxfp4.to(torch.float32).T  # Transpose for reference matmul -> (K, N)
        b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // 2])
    else:
        # mxfp8: Standard fp8 tensor (K, N)
        b = torch.randint(20, 40, (K, N), dtype=torch.uint8).to(torch.float8_e4m3fn).to(device)
        b_ref = b.to(torch.float32)
        b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])

    c = torch.zeros((M, N), device=device, dtype=torch.float16)
    c_desc = TensorDescriptor.from_tensor(c, block_shape=[BLOCK_M, BLOCK_N])

    # Create E8M0 scale tensors using 5D TMA layout: [1, rep_m, rep_k, 2, 256]
    a_scale = torch.randint(127, 128, (M, K // VEC_SIZE), dtype=torch.uint8, device=device)
    b_scale = torch.randint(127, 128, (N, K // VEC_SIZE), dtype=torch.uint8, device=device)

    # Swizzle to 5D cuBLAS block scaling layout for TMA
    a_scale_5d = _swizzle_scale_to_5d(a_scale.reshape(1, M, K // VEC_SIZE), M // 128, K // VEC_SIZE // 4)
    b_scale_5d = _swizzle_scale_to_5d(b_scale.reshape(1, N, K // VEC_SIZE), N // 128, K // VEC_SIZE // 4)

    a_scale_block_shape = [1, BLOCK_M // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    b_scale_block_shape = [1, BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 2 * 128]
    a_scale_desc = TensorDescriptor.from_tensor(a_scale_5d, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale_5d, block_shape=b_scale_block_shape)

    kern_kwargs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_K": BLOCK_K,
        "BLOCK_N": BLOCK_N,
        "A_IS_FP4": A_IS_FP4,
        "B_IS_FP4": B_IS_FP4,
    }
    kernel = tcgen5_dot_scaled_mixed_kernel[(1, 1)](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        A_format,
        B_format,
        **kern_kwargs,
    )

    ttgir = kernel.asm["ttgir"]
    assert ttgir.count("ttng.async_tma_copy_global_to_local") == 4
    assert ttgir.count("ttng.tc_gen5_mma_scaled") == 1

    # Check that fp4Padded is set correctly in the IR
    # When A is FP4 (mixed precision), A should have fp4Padded = true
    # When B is FP4 (mixed precision), B should have fp4Padded = true
    if A_IS_FP4:
        # First nvmma_shared (for A) should have fp4Padded = true
        assert "fp4Padded = true" in ttgir, "A should have fp4Padded=true when A is mxfp4 in mixed precision"
    if B_IS_FP4:
        # B's nvmma_shared should have fp4Padded = true
        assert "fp4Padded = true" in ttgir, "B should have fp4Padded=true when B is mxfp4 in mixed precision"

    # Converts E8M0 format scale values to float32
    def fp8e8m0_to_float32(scale):
        scale = scale.view(torch.uint8)
        scale = scale.to(torch.int32)
        scale = scale << 23
        scale = scale.view(torch.float32)
        return scale

    # Compute reference (use original 2D scales, not swizzled 5D)
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)
    # Repeat each scale value VEC_SIZE times along dim 1
    a_scale_f32 = a_scale_f32.repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_out = torch.matmul(a_ref * a_scale_f32, b_ref * b_scale_f32).to(torch.float16)

    atol = 1e-2 * math.sqrt(K / 32)
    torch.testing.assert_close(ref_out, c, atol=atol, rtol=0)


class TestToMxfp8:
    """Tests for the _to_mxfp8_block library function callable from JIT code with VEC_SIZE=32."""

    @staticmethod
    def _reference_mxfp8_quantize(data, vec_size, torch_dtype):
        """Python reference for MXFP8 quantization matching _compute_scale_and_quantize.

        Note: These tests store the data in SMEM without appropriate prescale swizzling to
        match the assumptions of TMEM. We do not test TMEM directly because we cannot provide
        enough information for an accurate layout.

        Returns:
            scale_e8m0: uint8 tensor [M, K // vec_size]
            data_fp8: fp8 tensor [M, K]
        """
        fp8_max = torch.finfo(torch_dtype).max
        M, K = data.shape
        num_scales = K // vec_size
        data_f32 = data.float()
        data_reshaped = data_f32.reshape(M, num_scales, vec_size)
        max_abs = data_reshaped.abs().amax(dim=2)
        descale = max_abs / fp8_max
        log2_descale = torch.log2(descale)
        ceil_log2 = torch.ceil(log2_descale)
        clamped_exp = torch.clamp(ceil_log2, -127.0, 127.0)
        is_zero = descale < 1e-38
        biased_exp = torch.where(is_zero, torch.zeros_like(clamped_exp), clamped_exp + 127)
        scale_e8m0 = biased_exp.to(torch.uint8)
        descale_fp = torch.where(
            biased_exp == 0,
            torch.ones_like(biased_exp),
            torch.exp2(127 - biased_exp),
        )
        scaled_data = data_reshaped * descale_fp.unsqueeze(2)
        scaled_data = torch.clamp(scaled_data, -fp8_max, fp8_max)
        data_flat = scaled_data.reshape(M, K)
        data_fp8 = data_flat.to(torch_dtype)
        return scale_e8m0, data_fp8

    @staticmethod
    def _run_to_mxfp8_block(input_data, elem_dtype, device):
        """Run _to_mxfp8_block in a JIT kernel and return FP8 data and scales."""
        torch_dtype = torch.float8_e4m3fn if elem_dtype == "e4m3" else torch.float8_e5m2
        M, K, VEC_SIZE = 128, 128, 32

        @triton.jit
        def kernel(
            input_ptr,
            data_out_ptr,
            scale_out_ptr,
            BLOCK_M: tl.constexpr,
            BLOCK_K: tl.constexpr,
            VEC_SIZE: tl.constexpr,
            ELEM_DTYPE: tl.constexpr,
        ):
            offs_m = tl.arange(0, BLOCK_M)
            offs_k = tl.arange(0, BLOCK_K)
            data = tl.load(input_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
            if ELEM_DTYPE == "e4m3":
                fp8_type: tl.constexpr = tl.float8e4nv
            else:
                fp8_type: tl.constexpr = tl.float8e5
            NUM_SCALES: tl.constexpr = BLOCK_K // VEC_SIZE
            data_tile = tlx.local_alloc((BLOCK_M, BLOCK_K), fp8_type, tl.constexpr(1))
            scale_tile = tlx.local_alloc((BLOCK_M, NUM_SCALES), tl.uint8, tl.constexpr(1))
            tlx._to_mxfp8_block(data, data_tile[0], scale_tile[0], VEC_SIZE, fp8_type)
            data_fp8 = tlx.local_load(data_tile[0])
            tl.store(data_out_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :], data_fp8)
            scale_loaded = tlx.local_load(scale_tile[0])
            scale_flat = tl.reshape(scale_loaded, [BLOCK_M * NUM_SCALES])
            tl.store(scale_out_ptr + tl.arange(0, BLOCK_M * NUM_SCALES), scale_flat)

        data_out = torch.empty(M, K, dtype=torch_dtype, device=device)
        scale_out = torch.empty(M * (K // VEC_SIZE), dtype=torch.uint8, device=device)
        kernel[(1, )](input_data, data_out, scale_out, M, K, VEC_SIZE, elem_dtype)
        return data_out, scale_out

    @pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
    @pytest.mark.parametrize("elem_dtype", ["e4m3", "e5m2"])
    def test_to_mxfp8_block_uniform(self, elem_dtype, device):
        """Test _to_mxfp8_block with uniform 1.0 input and VEC_SIZE=32."""
        torch_dtype = torch.float8_e4m3fn if elem_dtype == "e4m3" else torch.float8_e5m2
        M, K, VEC = 128, 128, 32
        input_data = torch.ones(M, K, dtype=torch.float32, device=device)

        data_out, scale_out = self._run_to_mxfp8_block(input_data, elem_dtype, device)

        ref_scale, ref_data = self._reference_mxfp8_quantize(input_data.cpu(), VEC, torch_dtype)
        torch.testing.assert_close(data_out.float().cpu(), ref_data.float())
        assert torch.equal(scale_out.cpu(), ref_scale.reshape(-1))

    @pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
    @pytest.mark.parametrize("elem_dtype", ["e4m3", "e5m2"])
    def test_to_mxfp8_block_zeros(self, elem_dtype, device):
        """Test _to_mxfp8_block with all-zero input."""
        M, K = 128, 128
        input_data = torch.zeros(M, K, dtype=torch.float32, device=device)

        data_out, scale_out = self._run_to_mxfp8_block(input_data, elem_dtype, device)

        assert torch.all(data_out.float() == 0)
        assert torch.all(scale_out == 0)

    @pytest.mark.skipif(not is_blackwell(), reason="Need Blackwell")
    @pytest.mark.parametrize("elem_dtype", ["e4m3", "e5m2"])
    def test_to_mxfp8_block_random(self, elem_dtype, device):
        """Test _to_mxfp8_block with random data against Python reference."""
        torch_dtype = torch.float8_e4m3fn if elem_dtype == "e4m3" else torch.float8_e5m2
        M, K, VEC = 128, 128, 32
        torch.manual_seed(42)
        input_data = torch.randn(M, K, dtype=torch.float32, device=device) * 100

        data_out, scale_out = self._run_to_mxfp8_block(input_data, elem_dtype, device)

        ref_scale, ref_data = self._reference_mxfp8_quantize(input_data.cpu(), VEC, torch_dtype)
        torch.testing.assert_close(data_out.float().cpu(), ref_data.float())
        assert torch.equal(scale_out.cpu(), ref_scale.reshape(-1))
