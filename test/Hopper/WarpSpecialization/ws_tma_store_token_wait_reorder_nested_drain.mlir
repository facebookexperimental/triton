// RUN: triton-opt %s --nvgpu-test-tma-store-token-wait-reorder | FileCheck %s
// RUN: triton-opt %s --nvgpu-test-tma-store-token-wait-reorder --tritongpu-pipeline=num-stages=2 | FileCheck %s --check-prefix=PIPE

// Reduced from MRS e59: each staging writer gets a queue-wide wait in its own
// stage, and the queue is drained once after the outer loop.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: nested_epilogue_loop_writer_waits
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: ttng.async_tma_store_wait {{.*}}loop.stage = 0 : i32, pendings = 1 : i32
// CHECK-NEXT: ttg.local_store
// CHECK: ttng.async_tma_store_token_wait {{.*}}planned_pending_count = 1
// CHECK-NEXT: ttng.async_tma_store_wait {{.*}}loop.stage = 0 : i32, pendings = 1 : i32
// CHECK-NEXT: ttg.local_store
// CHECK: ttng.async_tma_store_token_wait {{.*}}planned_pending_count = 1
// CHECK: } {tt.scheduled_max_stage
// CHECK-NEXT: {{^ *}}}
// CHECK-NEXT: ttng.async_tma_store_wait {{.*}}pendings = 0
// CHECK-NEXT: tt.return
// After pipelining, the inner loop's prologue writers keep their waits.
// PIPE-LABEL: nested_epilogue_loop_writer_waits
// PIPE: scf.for
// PIPE: ttng.async_tma_store_wait {{.*}}pendings = 1 : i32
// PIPE-NEXT: ttg.local_store
// PIPE: ttng.async_tma_store_wait {{.*}}pendings = 1 : i32
// PIPE-NEXT: ttg.local_store
// PIPE: scf.for
  tt.func public @nested_epilogue_loop_writer_waits(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: tensor<128x64xf16>,
      %buf0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %buf1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %lb: index, %ub: index, %step: index, %i: i32) {
    scf.for %t = %lb to %ub step %step {
      scf.for %j = %lb to %ub step %step {
        ttg.local_store %src, %buf0 {async_task_id = array<i32: 0>, "loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %tok0 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %buf0 {async_task_id = array<i32: 0>, "loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %tok0 {async_task_id = array<i32: 0>, "can_rotate_by_buffer_count" = 2 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !ttg.async.token
        ttg.local_store %src, %buf1 {async_task_id = array<i32: 0>, "loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : tensor<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %tok1 = ttng.async_tma_copy_local_to_global %desc[%i, %i] %buf1 {async_task_id = array<i32: 0>, "loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %tok1 {async_task_id = array<i32: 0>, "can_rotate_by_buffer_count" = 2 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !ttg.async.token
      } {"tt.scheduled_max_stage" = 1 : i32}
    }
    tt.return
  }
}
