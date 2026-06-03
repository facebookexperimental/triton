// RUN: triton-opt %s -split-input-file --nvgpu-test-tma-store-token-wait-reorder | FileCheck %s
// XFAIL: *

// Regression test for B-22-F1 / T273504812.
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// A wait_barrier before the producer is unrelated to the TMA store token.
// The reorder must not use it as the insertion target and assign the token
// wait a schedule position before the producing TMA store.
// CHECK-LABEL: @earlier_barrier_does_not_precede_producer
// CHECK: scf.for
// CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32}
// CHECK: [[TOK0:%.*]] = ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait [[TOK0]]
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 3 : i32, loop.stage = 0 : i32}
  tt.func public @earlier_barrier_does_not_precede_producer(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %lb: index, %ub: index, %step: index) {
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttng.wait_barrier %barrier, %c0 {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src0 {"loop.stage" = 0 : i32, "loop.cluster" = 1 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      %dummy = arith.addi %c0, %c0 {"loop.stage" = 0 : i32, "loop.cluster" = 2 : i32} : i32
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%dummy, %c0] %src1 {"loop.stage" = 0 : i32, "loop.cluster" = 3 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok0 {"can_rotate_by_buffer_count" = 1 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 4 : i32} : !ttg.async.token
    } {"tt.scheduled_max_stage" = 1 : i32}
    tt.return
  }
}
