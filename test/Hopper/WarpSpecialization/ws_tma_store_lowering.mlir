// RUN: triton-opt %s -split-input-file --nvgpu-ws-tma-store-lowering | FileCheck %s

#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttg.early_tma_store_lowering" = true} {
// CHECK-LABEL: tma_store_basic
//       CHECK: ttg.local_alloc %arg2
//   CHECK-NOT: ttng.fence_async_shared
//       CHECK: %[[TOKEN:.*]] = ttng.async_tma_copy_local_to_global
//  CHECK-SAME: -> !ttg.async.token
//       CHECK: ttng.async_tma_store_token_wait %[[TOKEN]] : !ttg.async.token
  tt.func public @tma_store_basic(%arg0: !tt.tensordesc<tensor<128x256xf32, #nvmma_128>>, %arg1: i32, %arg2: tensor<128x256xf32, #blocked>) {
    tt.descriptor_store %arg0[%arg1, %arg1], %arg2 : !tt.tensordesc<tensor<128x256xf32, #nvmma_128>>, tensor<128x256xf32, #blocked>
    tt.return
  }
}

// -----

#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_store_reduce_skipped
//       CHECK: tt.descriptor_store
//   CHECK-NOT: ttng.async_tma_copy_local_to_global
//   CHECK-NOT: ttng.async_tma_store_token_wait
  tt.func public @tma_store_reduce_skipped(%arg0: !tt.tensordesc<tensor<128x256xf32, #nvmma_128>>, %arg1: i32, %arg2: tensor<128x256xf32, #blocked>) {
    tt.descriptor_store %arg0[%arg1, %arg1], %arg2 reduce_kind = add : !tt.tensordesc<tensor<128x256xf32, #nvmma_128>>, tensor<128x256xf32, #blocked>
    tt.return
  }
}
