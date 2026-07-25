// RUN: triton-opt %s -split-input-file --nvgpu-test-tma-store-token-wait-reorder | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Single-buffered (K=1). One TMA copy in the loop. Counting 1 copy forward
// wraps to the next iteration's copy, so the wait lands at stage 1.
// CHECK-LABEL: single_buffer_k1
// CHECK: scf.for
// CHECK: ttg.local_store {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  tt.func public @single_buffer_k1(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buf {"loop.stage" = 0 : i32, "loop.cluster" = 1 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok {"can_rotate_by_buffer_count" = 1 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 2 : i32} : !ttg.async.token
    } {"tt.scheduled_max_stage" = 1 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Double-buffered (K=2). One TMA copy at stage 1. Counting 2 copies forward
// wraps twice to the copy at stage 1 + 2*numStages = stage 3 (with numStages=1
// per wrap). Wait lands at stage 3.
// CHECK-LABEL: double_buffer_k2
// CHECK: scf.for
// CHECK: ttg.local_store {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 0 : i32, loop.stage = 3 : i32}
  tt.func public @double_buffer_k2(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buf {"loop.stage" = 1 : i32, "loop.cluster" = 1 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
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
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buf {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
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
// reorders. With K=1 and one copy, the wait wraps to stage 1.
// CHECK-LABEL: no_schedule_creates_basic
// CHECK: scf.for
// CHECK: ttg.local_store {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  tt.func public @no_schedule_creates_basic(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buf : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok {"can_rotate_by_buffer_count" = 1 : i32} : !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Cross-partition case: after code partitioning the local_store ops are in a
// different partition. The loop body only has memdesc_index + tma_copy + wait.
// With K=1 and one copy, the wait wraps to stage 1.
// CHECK-LABEL: cross_partition_memdesc_index
// CHECK: scf.for
// CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 1 : i32, loop.stage = 1 : i32}
  tt.func public @cross_partition_memdesc_index(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %multibuf: !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>,
      %barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %lb: index, %ub: index, %step: index) {
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      %slot = ttg.memdesc_index %multibuf[%c0] {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttng.wait_barrier %barrier, %c0 {"loop.stage" = 0 : i32, "loop.cluster" = 1 : i32} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %slot {"loop.stage" = 0 : i32, "loop.cluster" = 2 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok {"can_rotate_by_buffer_count" = 1 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 3 : i32} : !ttg.async.token
    } {"tt.scheduled_max_stage" = 1 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// If K covers every TMA store in the loop, the next producer is in the next
// iteration. A wait_barrier before the producer is part of that wraparound
// interval and must be considered as the insertion target.
// CHECK-LABEL: wraparound_considers_barrier_before_producer
// CHECK: scf.for
// CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  tt.func public @wraparound_considers_barrier_before_producer(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %lb: index, %ub: index, %step: index) {
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttng.wait_barrier %barrier, %c0 {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src {"loop.stage" = 0 : i32, "loop.cluster" = 1 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok {"can_rotate_by_buffer_count" = 1 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 2 : i32} : !ttg.async.token
    } {"tt.scheduled_max_stage" = 1 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Triple-buffered store staging with two TMA stores in the loop. For the
// second store, counting K=3 stores forward wraps to the first store, so its
// wait must be scheduled before the first store's wait_barrier.
// CHECK-LABEL: k3_two_stores_wraparound_uses_first_wait
// CHECK: scf.for
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK: ttng.wait_barrier
// CHECK: ttng.async_tma_copy_local_to_global
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 1 : i32, loop.stage = 3 : i32}
  tt.func public @k3_two_stores_wraparound_uses_first_wait(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %bar0: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %lb: index, %ub: index, %step: index) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    scf.for %iv = %lb to %ub step %step {
      ttng.wait_barrier %bar0, %c0 {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src0 {"loop.stage" = 0 : i32, "loop.cluster" = 1 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok0 {"can_rotate_by_buffer_count" = 3 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 2 : i32} : !ttg.async.token
      ttng.wait_barrier %bar1, %c0 {"loop.stage" = 1 : i32, "loop.cluster" = 3 : i32} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%c0, %c64] %src1 {"loop.stage" = 1 : i32, "loop.cluster" = 4 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok1 {"can_rotate_by_buffer_count" = 3 : i32, "loop.stage" = 1 : i32, "loop.cluster" = 5 : i32} : !ttg.async.token
    } {"tt.scheduled_max_stage" = 1 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Four TMA stores with K=3. The last three waits all wrap to earlier stores in
// the next iteration. They must be scheduled before the wrapped store's
// wait_barrier, not merely before the wrapped store.
// CHECK-LABEL: k3_four_stores_all_waits_wraparound
// CHECK: scf.for
// CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 12 : i32, loop.stage = 0 : i32}
// CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 5 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 6 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 0 : i32, loop.stage = 1 : i32}
// CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 9 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 10 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 4 : i32, loop.stage = 1 : i32}
// CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 13 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 14 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 8 : i32, loop.stage = 1 : i32}
  tt.func public @k3_four_stores_all_waits_wraparound(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src2: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %src3: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %bar0: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %bar1: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %bar2: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %bar3: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %lb: index, %ub: index, %step: index) {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %c96 = arith.constant 96 : i32
    scf.for %iv = %lb to %ub step %step {
      ttng.wait_barrier %bar0, %c0 {"loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      %tok0 = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src0 {"loop.stage" = 0 : i32, "loop.cluster" = 1 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok0 {"can_rotate_by_buffer_count" = 3 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 2 : i32} : !ttg.async.token
      ttng.wait_barrier %bar1, %c0 {"loop.stage" = 0 : i32, "loop.cluster" = 3 : i32} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      %tok1 = ttng.async_tma_copy_local_to_global %desc[%c0, %c32] %src1 {"loop.stage" = 0 : i32, "loop.cluster" = 4 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok1 {"can_rotate_by_buffer_count" = 3 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 5 : i32} : !ttg.async.token
      ttng.wait_barrier %bar2, %c0 {"loop.stage" = 0 : i32, "loop.cluster" = 6 : i32} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      %tok2 = ttng.async_tma_copy_local_to_global %desc[%c0, %c64] %src2 {"loop.stage" = 0 : i32, "loop.cluster" = 7 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok2 {"can_rotate_by_buffer_count" = 3 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 8 : i32} : !ttg.async.token
      ttng.wait_barrier %bar3, %c0 {"loop.stage" = 0 : i32, "loop.cluster" = 9 : i32} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      %tok3 = ttng.async_tma_copy_local_to_global %desc[%c0, %c96] %src3 {"loop.stage" = 0 : i32, "loop.cluster" = 10 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok3 {"can_rotate_by_buffer_count" = 3 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 11 : i32} : !ttg.async.token
    } {"tt.scheduled_max_stage" = 1 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Loop-carried token after an unrelated loop-carried value. The token wait
// should still trace to the yielded TMA store token and get rescheduled.
// CHECK-LABEL: loop_carried_token_with_dead_iter_arg
// CHECK: scf.for
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
// CHECK-SAME: {loop.cluster = 1 : i32, loop.stage = 1 : i32}
// CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
  tt.func public @loop_carried_token_with_dead_iter_arg(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %lb: index, %ub: index, %step: index, %i: i32) {
    %init_tok = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    %result:2 = scf.for %iv = %lb to %ub step %step iter_args(%dead = %i, %carried = %init_tok) -> (i32, !ttg.async.token) {
      ttng.async_tma_store_token_wait %carried {"can_rotate_by_buffer_count" = 1 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !ttg.async.token
      ttng.wait_barrier %barrier, %i {"loop.stage" = 0 : i32, "loop.cluster" = 1 : i32} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src {"loop.stage" = 0 : i32, "loop.cluster" = 2 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      scf.yield %dead, %tok : i32, !ttg.async.token
    } {"tt.scheduled_max_stage" = 1 : i32}
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
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32) {
    %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok0 {"can_rotate_by_buffer_count" = 1 : i32} : !ttg.async.token
    tt.return
  }
}
