// RUN: triton-opt %s --nvws-meta-to-nvws-convert | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @direct_tma_reduce_wait
  tt.func @direct_tma_reduce_wait(
      %desc: !tt.tensordesc<tensor<128x128xf32, #shared>>,
      %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    scf.for %i = %lb to %ub step %step : i32 {
      %staging = ttg.local_alloc {async_task_id = array<i32: 0>} :
          () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>

      // CHECK: %[[TOKEN:.*]] = ttng.async_tma_reduce add
      // CHECK-SAME: {ttg.partition = array<i32: 0>}
      %token = ttng.async_tma_reduce add, %desc[%c0, %c0] %staging
          {async_task_id = array<i32: 0>} :
          !tt.tensordesc<tensor<128x128xf32, #shared>>,
          !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
          -> !ttg.async.token

      // The Meta schedule places this wait in an epilogue partition. The NVWS
      // conversion must co-locate it with its direct token producer.
      // CHECK: ttng.async_tma_store_token_wait %[[TOKEN]]
      // CHECK-SAME: {can_rotate_by_buffer_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttng.async_tma_store_token_wait %token
          {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} :
          !ttg.async.token
      scf.yield {async_task_id = array<i32: 0, 4>}
    } {async_task_id = array<i32: 0, 4>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["reduction", "unused", "unused", "unused", "epilogue"],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
