
import pytest
import torch
import triton
import triton.language as tl
import triton.tlx.language as tlx
from triton._internal_testing import is_cuda

"""
Below is TTGIR snippets of tl.dot:

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 8]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
...
    %cst = arith.constant dense<1.000000e-01> : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> loc(#loc1)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma> loc(#loc1)
    %cst_1 = arith.constant dense<2.000000e-01> : tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> loc(#loc2)
    %0 = ttg.local_alloc %cst_1 : (tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> !ttg.memdesc<64x64xf32, #shared, #smem> loc(#loc2)
    ttng.fence_async_shared {bCluster = false} loc(#loc3)
    %1 = ttng.warp_group_dot %cst, %0, %cst_0 {inputPrecision = 0 : i32} : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * !ttg.memdesc<64x64xf32, #shared, #smem> -> tensor<64x64xf32, #mma> loc(#loc3)

ttng.warp_group_dot 
    TTG_TensorOrMemDesc:$a,
        tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>

    TTG_TensorOrMemDesc:$b,
        tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
        => ttg.local_alloc -> !ttg.memdesc<64x64xf32, #shared, #smem>

    TT_FpIntTensor:$c,
        tensor<64x64xf32, #mma>

Now: 
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %1 = ttg.memdesc_subview %0[%c0_i32, %c0_i32_0, %c0_i32_0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %2 = ttg.memdesc_subview %0[%c1_i32, %c0_i32_1, %c0_i32_1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %3 = tt.call @"zeros____(0, 0)cconstexpr_64__(0, 1)cconstexpr_64__(1,)cconstexpr_fp32_"() : () -> tensor<64x64xf32>
    %4 = tlx.require_layout %1 : <64x64xf32, #shared, #smem, mutable> -> <64x64xf32, #shared1, #smem, mutable>
    %5 = tlx.require_layout %2 : <64x64xf32, #shared, #smem, mutable> -> <64x64xf32, #shared1, #smem, mutable>

Below are what we expect to see in TTIR:
%1 = tlx.require_layout %q_tile : ttg.memdesc<64x64xf16, #shared>  -> ttg.memdesc<64x64xf16, #sharedMMA>
%2 = tlx.require_layout %k_tile : ttg.memdesc<64x64xf16, #shared>  -> ttg.memdesc<64x64xf16, #sharedMMA>
%qk = ttng.warp_group_dot %1, %2
"""
@pytest.mark.skipif(
    not is_cuda() or torch.cuda.get_device_capability()[0] != 9,
    reason="Requires compute capability == 9 for NV",
)
def test_async_dot(device):
    # Define a unit test with similar schema with tl.dot

    @triton.jit
    def ref_kernel(X, stride_xm, stride_xk, Y, stride_yk, stride_yn, Z, stride_zm, stride_zn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, INPUT_PRECISION: tl.constexpr, out_dtype: tl.constexpr = tl.float32):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        x = tl.load(Xs)
        y = tl.load(Ys)
        z = tl.dot(x, y, input_precision=INPUT_PRECISION, out_dtype=out_dtype)
        tl.store(Zs, z)

    @triton.jit
    def tgt_kernel(
        X, stride_xm, stride_xk, 
        Y, stride_yk, stride_yn, 
        Z, stride_zm, stride_zn,
        BLOCK_M: tl.constexpr, 
        BLOCK_N: tl.constexpr, 
        BLOCK_K: tl.constexpr, 
        INPUT_PRECISION: tl.constexpr, 
        out_dtype: tl.constexpr = tl.float32
        # a_ptr,
        # b_ptr,
        # stride_cm,
        # stride_cn,
        # BLOCK_SIZE: tl.constexpr,
    ):
        dummy_output = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)

        buf_alloc_x = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float32, 1)
        buf_alloc_y = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float32, 1)
        a_smem = tlx.local_view(buf_alloc_x, 0)
        b_smem = tlx.local_view(buf_alloc_y, 0)

        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn

        # TODO. initialize values or async load

        z = tlx.async_dot(a_smem, b_smem, dummy_output, input_precision=INPUT_PRECISION, out_dtype=out_dtype)
        tl.store(Zs, z)

        # offs_cm = tl.arange(0, BLOCK_SIZE)
        # offs_cn = tl.arange(0, BLOCK_SIZE)
        # c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        # tl.store(c_ptrs, acc)

    # a_gmem = torch.ones((64, 64), device=device, dtype=torch.float32)
    # b_gmem = torch.ones((64, 64), device=device, dtype=torch.float32)
    # c = torch.ones((64, 64), device=device, dtype=torch.float32)
    # grid = lambda META: (1,)
    # async_dot_kernel[grid](
    #     # a_gmem,
    #     # b_gmem,
    #     c,  #
    #     c.stride(0),
    #     c.stride(1),  #
    #     64,
    #     num_warps=4,
    # )

    # print("\nc = ", c)
    # assert False

    M,N,K = (64,64,64)
    x = torch.ones((M, K), device=device, dtype=torch.float32)
    y = torch.ones((K, N), device=device, dtype=torch.float32)
    z = torch.empty_like(x, device=device, dtype=torch.float32)

    kern_kwargs = {
        'BLOCK_M': M, 'BLOCK_K': K, 'BLOCK_N': N, 'INPUT_PRECISION': "tf32", 'out_dtype': tl.float32
    }
    pgm = tgt_kernel[(1, 1)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), z, z.stride(0), z.stride(1), **kern_kwargs)
