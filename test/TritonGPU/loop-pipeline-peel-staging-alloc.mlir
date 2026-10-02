// RUN: env TRITON_USE_META_WS=1 triton-opt %s -tritongpu-pipeline | FileCheck %s

// With K == BLOCK_K the K loop folds away and the persistent tile loop is
// pipelined directly. Early TMA store lowering has already turned the epilogue
// store into local_alloc + async_tma_copy_local_to_global + token wait in the
// last stage, so the peeled epilogue must predicate that sequence. The TMA copy
// and wait are wrapped in scf.if; the staging local_alloc is left unpredicated
// because nothing but its own (predicated) users reads the buffer it creates.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @early_tma_store_staging_alloc
  // CHECK: scf.for
  // CHECK:   ttg.local_alloc
  // CHECK:   ttng.async_tma_copy_local_to_global
  // CHECK:   ttng.async_tma_store_token_wait
  // CHECK: scf.yield
  // CHECK: %[[ALLOC:.*]] = ttg.local_alloc %{{.*}} : (tensor<128x64xbf16, #blocked>)
  // CHECK-NEXT: %[[TOK:.*]] = scf.if %[[PRED:.*]] -> (!ttg.async.token) {
  // CHECK-NEXT: ttng.async_tma_copy_local_to_global %{{.*}} %[[ALLOC]]
  // CHECK: scf.if %[[PRED]] {
  // CHECK-NEXT: ttng.async_tma_store_token_wait %[[TOK]]
  tt.func @early_tma_store_staging_alloc(
      %lb: i32, %ub: i32, %in: !tt.tensordesc<128x64xbf16, #shared>, %out: !tt.tensordesc<128x64xbf16, #shared>) {
    %c0 = arith.constant 0 : i32
    %c148 = arith.constant 148 : i32
    scf.for %iv = %lb to %ub step %c148 : i32 {
      %v = tt.descriptor_load %in[%iv, %c0] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
      %w = arith.addf %v, %v {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xbf16, #blocked>
      %a = ttg.local_alloc %w {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
      %t = ttng.async_tma_copy_local_to_global %out[%iv, %c0] %a {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %t {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32}
    tt.return
  }
}
