// RUN: triton-opt %s -split-input-file --nvgpu-test-annotate-tma-store-waits | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Triple-buffered (buffer.copy = 3). K = 3.
// CHECK-LABEL: triple_buffer
// CHECK: ttng.async_tma_store_token_wait
// CHECK-SAME: can_rotate_by_buffer_count = 3
// CHECK-NOT: planned_pending_count
  tt.func public @triple_buffer(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc {"buffer.copy" = 3 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buf : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok : !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// CLC epilogues are carried by scf.while rather than scf.for.
// CHECK-LABEL: while_store_wait
// CHECK: scf.while
// CHECK: ttng.async_tma_store_token_wait
// CHECK-SAME: can_rotate_by_buffer_count = 2
// CHECK-NOT: planned_pending_count
// CHECK: } attributes {ttg.clc_persistent}
  tt.func public @while_store_wait(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: tensor<128x64xf16>, %i: i32) {
    %buf = ttg.local_alloc {"buffer.copy" = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %true = arith.constant true
    %false = arith.constant false
    scf.while (%valid = %true) : (i1) -> () {
      scf.condition(%valid)
    } do {
      %next_valid, %x, %y, %z = ttng.clc_advance : i1, i32, i32, i32
      ttg.local_store %src, %buf : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%i, %i] %buf : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok : !ttg.async.token
      scf.yield %next_valid : i1
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// An atomic dynamic-persistent scheduler is also an scf.while, but rotating its
// epilogue wait across unrelated scheduler iterations races staging-buffer reuse.
// CHECK-LABEL: dynamic_while_store_wait
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
  tt.func public @dynamic_while_store_wait(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: tensor<128x64xf16>, %i: i32) {
    %buf = ttg.local_alloc {"buffer.copy" = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %true = arith.constant true
    %false = arith.constant false
    scf.while (%valid = %true) : (i1) -> () {
      scf.condition(%valid)
    } do {
      ttg.local_store %src, %buf : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%i, %i] %buf : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok : !ttg.async.token
      scf.yield %false : i1
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// One-CTA reductions use the same two-slot rotation as TLX. The reorder pass
// is responsible for materializing the final queue drain.
// CHECK-LABEL: one_cta_reduce_rotates
// CHECK: ttng.async_tma_store_token_wait
// CHECK-SAME: can_rotate_by_buffer_count = 2
// CHECK-NOT: planned_pending_count
  tt.func public @one_cta_reduce_rotates(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc {"buffer.copy" = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_reduce add, %desc[%c0, %c0] %buf : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok : !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Single-buffered (buffer.copy = 1). K = 1 → annotated.
// CHECK-LABEL: single_buffer
// CHECK: ttng.async_tma_store_token_wait
// CHECK-SAME: can_rotate_by_buffer_count = 1
// CHECK-NOT: planned_pending_count
  tt.func public @single_buffer(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc {"buffer.copy" = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buf : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok : !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// No buffer.copy attribute → no annotation.
// CHECK-LABEL: no_buffer_copy
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
  tt.func public @no_buffer_copy(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: tensor<128x64xf16>,
      %lb: index, %ub: index, %step: index) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      ttg.local_store %src, %buf : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buf : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok : !ttg.async.token
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// Outside loop → no annotation (pass only annotates waits inside scf.for).
// CHECK-LABEL: outside_loop
// CHECK: ttng.async_tma_store_token_wait
// CHECK-NOT: can_rotate_by_buffer_count
  tt.func public @outside_loop(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %i: i32) {
    %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src0 : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok0 : !ttg.async.token
    tt.return
  }
}
