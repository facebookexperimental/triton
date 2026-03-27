// RUN: triton-opt %s -split-input-file --nvgpu-tma-store-token-wait-reorder | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Single-buffered (K=1). local_store and tma_copy at stage 0, wait starts at
// stage 0. The iterator wraps once from tma_copy back to local_store, so
// currStage = 1. The wait is placed at stage 1 in a new cluster before the
// local_store's (split) cluster.
// CHECK-LABEL: single_buffer_k1
// CHECK: scf.for
// CHECK: ttg.local_store {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  tt.func public @single_buffer_k1(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buf {"loop.stage" = 0 : i32, "loop.cluster" = 1 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok {"can_rotate_by_buffer_count" = 1 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 2 : i32} : !ttg.async.token
    } {"tt.scheduled_max_stage" = 1 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Double-buffered (K=2). local_store at stage 0, tma_copy + wait at stage 1.
// The iterator wraps twice from tma_copy (stage 1) to find the 2nd
// local_store, so currStage = 3. The wait is placed at stage 3 in a new
// cluster before the local_store's (split) cluster.
// CHECK-LABEL: double_buffer_k2
// CHECK: scf.for
// CHECK: ttg.local_store {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 0 : i32, loop.stage = 3 : i32}
  tt.func public @double_buffer_k2(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buf {"loop.stage" = 1 : i32, "loop.cluster" = 1 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok {"can_rotate_by_buffer_count" = 2 : i32, "loop.stage" = 1 : i32, "loop.cluster" = 2 : i32} : !ttg.async.token
    } {"tt.scheduled_max_stage" = 2 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Without can_rotate_by_buffer_count attribute → schedule stays unchanged.
// CHECK-LABEL: no_attribute_no_change
// CHECK: scf.for
// CHECK: ttng.async_tma_store_token_wait {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
  tt.func public @no_attribute_no_change(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buf {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok {"loop.stage" = 0 : i32, "loop.cluster" = 1 : i32} : !ttg.async.token
    } {"tt.scheduled_max_stage" = 1 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// No SWP schedule on the loop → pass creates a basic schedule and still
// reorders. The wait gets placed before the 1st local_store (wrapping around).
// CHECK-LABEL: no_schedule_creates_basic
// CHECK: scf.for
// CHECK: ttg.local_store {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 0 : i32, loop.stage = 0 : i32}
  tt.func public @no_schedule_creates_basic(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buf : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok {"can_rotate_by_buffer_count" = 1 : i32} : !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Outside a loop → pass doesn't touch it, attribute preserved.
// CHECK-LABEL: outside_loop_no_op
// CHECK: can_rotate_by_buffer_count
  tt.func public @outside_loop_no_op(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32) {
    %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok0 {"can_rotate_by_buffer_count" = 1 : i32} : !ttg.async.token
    tt.return
  }
}
