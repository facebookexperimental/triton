// RUN: TRITON_USE_META_WS=1 triton-opt %s -split-input-file --triton-nvidia-gpu-test-generate-subtiled-region | FileCheck %s

// Test: SAME-TASK epilogue subtiling (separate_epilogue_store=False).
// The per-tile truncf -> local_store -> async_tma_copy_local_to_global ->
// async_tma_store_token_wait chain is entirely in ONE async task, so the
// producer store and the consumer TMA copy must be wrapped into a SINGLE
// interleaved SubtiledRegionOp (store_t -> copy_t -> token_wait per tile),
// NOT two sequential regions. With two sequential same-task regions the staging
// slot is reused before its TMA drains when numTiles > buffer.copy, silently
// corrupting the output (the EPILOGUE_SUBTILE > num_stages bug).

#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#blocked3d2 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm2 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked_full2 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2d2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem2 = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @same_task_epilogue_interleave
  // Exactly ONE subtiled region whose tile body holds BOTH the producer store
  // and the consumer TMA copy + its drain (interleaved), not two regions.
  // CHECK: ttng.subtiled_region
  // CHECK:        arith.truncf
  // CHECK-NEXT:   ttg.local_store
  // CHECK-NEXT:   ttng.async_tma_copy_local_to_global
  // CHECK-NEXT:   ttng.async_tma_store_token_wait
  // CHECK-NEXT:   ttng.subtiled_region_yield
  // CHECK-NOT: ttng.subtiled_region
  tt.func @same_task_epilogue_interleave(
      %tmem_buf: !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token,
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared2>>,
      %off0: i32, %off1: i32, %off2: i32) {
    // Pre-hoisted SMEM staging allocations (one per subtile).
    %smem0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>
    %smem1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>

    %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] : !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_full2>
    %reshaped = tt.reshape %loaded#0 : tensor<128x128xf32, #blocked_full2> -> tensor<128x2x64xf32, #blocked3d2>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3d2> -> tensor<128x64x2xf32, #blocked3d_perm2>
    %lhs, %rhs = tt.split %transposed : tensor<128x64x2xf32, #blocked3d_perm2> -> tensor<128x64xf32, #blocked2d2>

    // Tile 0: truncf -> store -> copy -> drain, all task 3.
    %trunc0 = arith.truncf %lhs {ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2d2> to tensor<128x64xf16, #blocked2d2>
    ttg.local_store %trunc0, %smem0 {ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked2d2> -> !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>
    %tok0 = ttng.async_tma_copy_local_to_global %desc[%off0, %off1] %smem0 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared2>>, !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok0 {ttg.partition = array<i32: 3>} : !ttg.async.token

    // Tile 1: same chain, all task 3.
    %trunc1 = arith.truncf %rhs {ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2d2> to tensor<128x64xf16, #blocked2d2>
    ttg.local_store %trunc1, %smem1 {ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked2d2> -> !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>
    %tok1 = ttng.async_tma_copy_local_to_global %desc[%off0, %off2] %smem1 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared2>>, !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok1 {ttg.partition = array<i32: 3>} : !ttg.async.token

    tt.return
  }
}
