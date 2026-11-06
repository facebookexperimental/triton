// RUN: triton-opt %s -split-input-file --nvgpu-ws-tma-store-lowering=only-warp-specialized=true | FileCheck %s

// With only-warp-specialized (the Meta AutoWS path), only functions AutoWS
// warp-specializes have their stores lowered early.

// The warp-specialized loop marks the function for AutoWS; its epilogue store
// is lowered early.
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttg.early_tma_store_lowering" = true} {
// CHECK-LABEL: tma_store_after_ws_loop
//       CHECK: scf.for
//       CHECK: ttg.local_alloc
//       CHECK: ttng.async_tma_copy_local_to_global
//       CHECK: ttng.async_tma_store_token_wait
  tt.func public @tma_store_after_ws_loop(%arg0: !tt.tensordesc<128x256xf32, #nvmma_128>, %arg1: i32, %arg2: tensor<128x256xf32, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %acc = scf.for %i = %c0 to %arg1 step %c1 iter_args(%a = %arg2) -> (tensor<128x256xf32, #blocked>) : i32 {
      %n = arith.addf %a, %a : tensor<128x256xf32, #blocked>
      scf.yield %n : tensor<128x256xf32, #blocked>
    } {tt.warp_specialize}
    tt.descriptor_store %arg0[%arg1, %arg1], %acc : !tt.tensordesc<128x256xf32, #nvmma_128>, tensor<128x256xf32, #blocked>
    tt.return
  }
}

// -----

// A function AutoWS does not warp-specialize keeps its stores for the regular
// software pipeliner.
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttg.early_tma_store_lowering" = true} {
// CHECK-LABEL: tma_store_not_warp_specialized
//       CHECK: tt.descriptor_store
//   CHECK-NOT: ttng.async_tma_copy_local_to_global
  tt.func public @tma_store_not_warp_specialized(%arg0: !tt.tensordesc<128x256xf32, #nvmma_128>, %arg1: i32, %arg2: tensor<128x256xf32, #blocked>) {
    tt.descriptor_store %arg0[%arg1, %arg1], %arg2 : !tt.tensordesc<128x256xf32, #nvmma_128>, tensor<128x256xf32, #blocked>
    tt.return
  }
}
