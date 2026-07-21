// RUN: triton-opt %s --nvgpu-test-ws-buffer-allocation | FileCheck %s

// createBuffer sorts channels by producer program order before hoisting their
// buffers. Hoisting must insert each new allocation after the previous hoisted
// allocation; otherwise every allocation is inserted at function entry and the
// final order is reversed.

// CHECK-LABEL: @hoist_preserves_producer_order
// CHECK: %[[A_BUF:.*]] = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16
// CHECK: %[[B_BUF:.*]] = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16
// CHECK: ttg.local_store {{.*}}, %[[A_BUF]]
// CHECK: ttg.local_store {{.*}}, %[[B_BUF]]

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hoist_preserves_producer_order(
      %a_desc: !tt.tensordesc<128x64xf16, #shared_a>,
      %b_desc: !tt.tensordesc<64x128xf16, #shared_b>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %a = tt.descriptor_load %a_desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared_a> -> tensor<128x64xf16, #blocked_a>
    %b = tt.descriptor_load %b_desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<64x128xf16, #shared_b> -> tensor<64x128xf16, #blocked_b>
    %a_use = arith.addf %a, %a {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked_a>
    %b_use = arith.addf %b, %b {async_task_id = array<i32: 0>} : tensor<64x128xf16, #blocked_b>
    tt.return
  }
}
