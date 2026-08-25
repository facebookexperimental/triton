// RUN: triton-opt %s --nvgpu-test-ws-buffer-allocation | FileCheck %s

// Each register-consumed descriptor load is converted to an nvws.descriptor_load
// that writes its own SMEM buffer, in producer program order. The A load's
// buffer/load must be emitted before the B load's; otherwise the producer order
// is reversed (e.g. every allocation hoisted to function entry in reverse).

// CHECK-LABEL: @hoist_preserves_producer_order
// CHECK: %[[A_BUF:.*]] = ttg.local_alloc {{.*}}: () -> !ttg.memdesc<128x64xf16
// CHECK: nvws.descriptor_load {{.*}} %[[A_BUF]]
// CHECK: %[[B_BUF:.*]] = ttg.local_alloc {{.*}}: () -> !ttg.memdesc<64x128xf16
// CHECK: nvws.descriptor_load {{.*}} %[[B_BUF]]

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#offsets = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_transposed = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
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

  // CHECK-LABEL: @gather_i16_register_consumed
  // CHECK: %[[GATHER_BUFFER:.*]] = ttg.local_alloc
  // CHECK: %[[OFFSETS32:.*]] = arith.extsi %{{.*}} {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK: nvws.descriptor_gather {{.*}}[%[[OFFSETS32]], {{.*}}] 16384 %[[GATHER_BUFFER]] {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK: %[[GATHER_VALUE:.*]] = ttg.local_load %[[GATHER_BUFFER]] {async_task_id = array<i32: 0>}
  // CHECK: arith.addf %[[GATHER_VALUE]], %[[GATHER_VALUE]]
  // CHECK-NOT: tt.descriptor_gather
  tt.func public @gather_i16_register_consumed(
      %desc: !tt.tensordesc<1x64xf16, #shared_a>,
      %x_offsets: tensor<128xi16, #offsets>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %gather = tt.descriptor_gather %desc[%x_offsets, %c0] {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : (!tt.tensordesc<1x64xf16, #shared_a>, tensor<128xi16, #offsets>, i32) -> tensor<128x64xf16, #blocked_a>
    %use = arith.addf %gather, %gather {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked_a>
    tt.return
  }

  // CHECK-LABEL: @gather_invalid_destination_falls_back
  // CHECK: %[[LANDING:.*]] = ttg.local_alloc {{.*}}!ttg.memdesc<128x64xf16, #shared,
  // CHECK: nvws.descriptor_gather {{.*}} 16384 %[[LANDING]]
  // CHECK: %[[RELOADED:.*]] = ttg.local_load %[[LANDING]]
  // CHECK: ttg.local_store %[[RELOADED]], %{{.*}}
  tt.func public @gather_invalid_destination_falls_back(
      %desc: !tt.tensordesc<1x64xf16, #shared_a>,
      %x_offsets: tensor<128xi32, #offsets>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %gather = tt.descriptor_gather %desc[%x_offsets, %c0] {async_task_id = array<i32: 1>} : (!tt.tensordesc<1x64xf16, #shared_a>, tensor<128xi32, #offsets>, i32) -> tensor<128x64xf16, #blocked_a>
    %invalid = ttg.local_alloc %gather {async_task_id = array<i32: 0>} : (tensor<128x64xf16, #blocked_a>) -> !ttg.memdesc<128x64xf16, #shared_transposed, #ttg.shared_memory>
    tt.return
  }
}
