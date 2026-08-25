// RUN: triton-opt %s --nvgpu-convert-descriptor-loads-to-nvws | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#offsets = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @descriptor_load
  // CHECK: %[[BUFFER:.*]] = ttg.local_alloc
  // CHECK: nvws.descriptor_load %arg0[%arg1, %arg1] 16384 %[[BUFFER]]
  // CHECK-SAME: {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // CHECK: %[[VALUE:.*]] = ttg.local_load %[[BUFFER]] {async_task_id = array<i32: 0>}
  // CHECK: arith.addf %[[VALUE]], %[[VALUE]]
  // CHECK-NOT: tt.descriptor_load
  tt.func public @descriptor_load(
      %desc: !tt.tensordesc<128x64xf16, #shared>, %i: i32) {
    %value = tt.descriptor_load %desc[%i, %i]
        {async_task_id = array<i32: 1>, loop.cluster = 2 : i32,
         loop.stage = 0 : i32, ttg.partition = array<i32: 1>} :
        !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    %use = arith.addf %value, %value {async_task_id = array<i32: 0>} :
        tensor<128x64xf16, #blocked>
    tt.return
  }

  // CHECK-LABEL: @descriptor_gather_i16
  // CHECK: %[[GATHER_BUFFER:.*]] = ttg.local_alloc
  // CHECK: %[[OFFSETS32:.*]] = arith.extsi %arg1
  // CHECK-SAME: {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 1 : i32}
  // CHECK: nvws.descriptor_gather %arg0[%[[OFFSETS32]], %arg2] 16384 %[[GATHER_BUFFER]]
  // CHECK-SAME: {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>}
  // CHECK: %[[GATHER_VALUE:.*]] = ttg.local_load %[[GATHER_BUFFER]] {async_task_id = array<i32: 0>}
  // CHECK: arith.addf %[[GATHER_VALUE]], %[[GATHER_VALUE]]
  // CHECK-NOT: tt.descriptor_gather
  tt.func public @descriptor_gather_i16(
      %desc: !tt.tensordesc<1x64xf16, #shared>,
      %x_offsets: tensor<128xi16, #offsets>, %y_offset: i32) {
    %value = tt.descriptor_gather %desc[%x_offsets, %y_offset]
        {async_task_id = array<i32: 2>, loop.cluster = 4 : i32,
         loop.stage = 1 : i32, ttg.partition = array<i32: 2>} :
        (!tt.tensordesc<1x64xf16, #shared>, tensor<128xi16, #offsets>, i32)
        -> tensor<128x64xf16, #blocked>
    %use = arith.addf %value, %value {async_task_id = array<i32: 0>} :
        tensor<128x64xf16, #blocked>
    tt.return
  }
}
