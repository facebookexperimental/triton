
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

    @triton.jit
    def ref_dot_kernel(
        c_ptr,
        stride_cm,
        stride_cn,
        BLOCK_SIZE: tl.constexpr,
    ):
        a = tl.full((BLOCK_SIZE, BLOCK_SIZE), 0.1, dtype=tl.float32)
        b = tl.full((BLOCK_SIZE, BLOCK_SIZE), 0.2, dtype=tl.float32)
        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        acc = tl.dot(a, b, acc, "tf32")
        offs_cm = tl.arange(0, BLOCK_SIZE)
        offs_cn = tl.arange(0, BLOCK_SIZE)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, acc)

    @triton.jit
    def async_dot_kernel(
        c_ptr,
        stride_cm,
        stride_cn,
        BLOCK_SIZE: tl.constexpr,
    ):
        buffers = tlx.local_alloc((BLOCK_SIZE, BLOCK_SIZE), tl.float32, 2)
        a = tlx.local_view(buffers, 0)
        b = tlx.local_view(buffers, 1)
        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

        acc = tlx.async_dot(a, b, acc)

        offs_cm = tl.arange(0, BLOCK_SIZE)
        offs_cn = tl.arange(0, BLOCK_SIZE)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, acc)

    # torch.manual_seed(2025)
    # a = torch.rand((64, 64), device=device, dtype=torch.float32)
    # b = torch.rand((64, 64), device=device, dtype=torch.float32)
    c = torch.ones((64, 64), device=device, dtype=torch.float32)
    grid = lambda META: (1,)
    async_dot_kernel[grid](
        c,  #
        c.stride(0),
        c.stride(1),  #
        64,
        num_warps=4,
    )

    print("c = ", c)
    assert False
