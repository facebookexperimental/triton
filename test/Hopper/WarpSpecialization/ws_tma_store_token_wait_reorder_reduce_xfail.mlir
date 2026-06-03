// RUN: triton-opt %s -split-input-file --nvgpu-test-tma-store-token-wait-reorder | FileCheck %s
// XFAIL: *

// Regression test for B-22-F2 / T273504813.
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// TMA reduce tokens use the same wait op as TMA stores. The reorder pass
// should consume can_rotate_by_buffer_count and rotate the wait by K stores.
// CHECK-LABEL: @single_buffer_reduce_token_k1
// CHECK: scf.for
// CHECK: ttg.local_store {{.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_reduce {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 1 : i32, loop.stage = 1 : i32}
  tt.func public @single_buffer_reduce_token_k1(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_reduce add, %desc[%c0, %c0] %buf {"loop.stage" = 0 : i32, "loop.cluster" = 1 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok {"can_rotate_by_buffer_count" = 1 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 2 : i32} : !ttg.async.token
    } {"tt.scheduled_max_stage" = 1 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Mixed reduce/copy coverage makes the K-th target a TMA copy after the
// defining TMA reduce, exercising the store-like linearization path.
// CHECK-LABEL: @mixed_reduce_to_copy_k1
// CHECK: scf.for
// CHECK: [[REDTOK:%.*]] = ttng.async_tma_reduce {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait [[REDTOK]]
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 3 : i32, loop.stage = 0 : i32}
  tt.func public @mixed_reduce_to_copy_k1(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %reduce_src: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %copy_src: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %lb: index, %ub: index, %step: index) {
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      %reduce_tok = ttng.async_tma_reduce add, %desc[%c0, %c0] %reduce_src {"loop.stage" = 0 : i32, "loop.cluster" = 1 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      %dummy = arith.addi %c0, %c0 {"loop.stage" = 0 : i32, "loop.cluster" = 2 : i32} : i32
      %copy_tok = ttng.async_tma_copy_local_to_global %desc[%dummy, %c0] %copy_src {"loop.stage" = 0 : i32, "loop.cluster" = 3 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %reduce_tok {"can_rotate_by_buffer_count" = 1 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 4 : i32} : !ttg.async.token
    } {"tt.scheduled_max_stage" = 1 : i32}
    tt.return
  }
}
