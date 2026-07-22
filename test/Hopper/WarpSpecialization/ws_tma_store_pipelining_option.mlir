// RUN: triton-opt %s --nvgpu-warp-specialization="num-stages=2 capability=90" | FileCheck %s --check-prefix=DEFAULT
// RUN: triton-opt %s --nvgpu-warp-specialization="num-stages=2 capability=90 tma-store-pipelining=false" | FileCheck %s --check-prefix=DISABLED

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // DEFAULT-LABEL: @preannotated_wait
  // DEFAULT: ttng.async_tma_store_token_wait
  // DEFAULT-NOT: can_rotate_by_buffer_count
  // DISABLED-LABEL: @preannotated_wait
  // DISABLED: ttng.async_tma_store_token_wait
  // DISABLED-SAME: can_rotate_by_buffer_count = 1
  tt.func public @preannotated_wait(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %src: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %lb: index, %ub: index, %step: index, %i: i32) {
    %seed = arith.constant {async_task_id = array<i32: 0>} 0 : i32
    scf.for %iv = %lb to %ub step %step {
      %tok = ttng.async_tma_copy_local_to_global %desc[%i, %i] %src {async_task_id = array<i32: 0>, "loop.stage" = 0 : i32, "loop.cluster" = 0 : i32} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tok {async_task_id = array<i32: 0>, "can_rotate_by_buffer_count" = 1 : i32, "loop.stage" = 0 : i32, "loop.cluster" = 1 : i32} : !ttg.async.token
    } {"tt.scheduled_max_stage" = 1 : i32}
    tt.return
  }
}
