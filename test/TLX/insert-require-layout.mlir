// RUN: triton-opt -split-input-file --tlx-insert-require-layout %s| FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#loc = loc("/home/kmanivannan/fb-triton/python/test/unit/language/test_tlx.py":158:0)
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
// CHECK-DAG: #[[shared2:.*]] = #ttg.swizzled_shared<{{.*}}>
// CHECK-DAG: #[[shared3:.*]] = #ttg.swizzled_shared<{{.*}}>
#smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @local_store_local_load_dot(%arg0: !tt.ptr<f16>, %arg1: tensor<64x32x!tt.ptr<f16>, #blocked>, %arg2: tensor<32x64x!tt.ptr<f16>, #blocked>) -> tensor<64x64xf32, #mma> {
    %24 = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable>
    %25 = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared1, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    // CHECK: %[[mem_desc1:.*]] = ttg.memdesc_subview %{{.*}}
    %26 = ttg.memdesc_subview %24[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    // CHECK: %[[mem_desc2:.*]] = ttg.memdesc_subview %{{.*}}
    %27 = ttg.memdesc_subview %25[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x32x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    %28 = tt.load %arg1 : tensor<64x32x!tt.ptr<f16>, #blocked>
    %29 = tt.load %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked>
    ttg.local_store %28, %26 : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    ttg.local_store %29, %27 : tensor<32x64xf16, #blocked> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    // CHECK: %[[req_layout_1:.*]] = tlx.require_layout %[[mem_desc1]] : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #[[shared2]], #smem, mutable>
    // CHECK: ttg.local_load %[[req_layout_1]]
    %30 = ttg.local_load %26 : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // CHECK: %[[req_layout_2:.*]] = tlx.require_layout %[[mem_desc2]] : !ttg.memdesc<32x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf16, #[[shared3]], #smem, mutable>
    // CHECK: ttg.local_load %[[req_layout_2]]
    %31 = ttg.local_load %27 : !ttg.memdesc<32x64xf16, #shared1, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %32 = ttg.convert_layout %cst : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #mma>
    %33 = ttg.convert_layout %30 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %34 = ttg.convert_layout %31 : tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %35 = tt.dot %33, %34, %32, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return %35 : tensor<64x64xf32, #mma>
  }
}
