// RUN: triton-opt %s --nvws-meta-to-nvws-convert | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#offsets = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @descriptor_to_planned_buffer
  tt.func @descriptor_to_planned_buffer(
      %load_desc: !tt.tensordesc<tensor<64x64xf16, #shared>>,
      %gather_desc: !tt.tensordesc<tensor<1x64xf16, #shared>>,
      %offsets: tensor<128xi32, #offsets>, %lb: i32, %ub: i32, %step: i32) {
    // CHECK: %[[LOAD_BUFFER:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32}
    %load_buffer = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: %[[GATHER_BUFFER:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32}
    %gather_buffer = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: %[[PREHEADER_BUFFER:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32}
    %preheader_buffer = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK-NOT: tt.descriptor_load
    // CHECK: nvws.descriptor_load {{.*}} 8192 %[[PREHEADER_BUFFER]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NOT: tt.descriptor_load
    // CHECK-NOT: ttg.local_store
    %preheader = tt.descriptor_load %load_desc[%lb, %lb] {async_task_id = array<i32: 2>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
    ttg.local_store %preheader, %preheader_buffer {async_task_id = array<i32: 2>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: nvws.descriptor_load {{.*}} 8192 %[[LOAD_BUFFER]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      %load = tt.descriptor_load %load_desc[%i, %i] {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
      // CHECK-NOT: ttg.local_store
      ttg.local_store %load, %load_buffer {async_task_id = array<i32: 2>, loop.cluster = 9 : i32, loop.stage = 1 : i32} : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      // CHECK-NOT: tt.descriptor_gather
      // CHECK: nvws.descriptor_gather {{.*}} 16384 %[[GATHER_BUFFER]] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      %gather = tt.descriptor_gather %gather_desc[%offsets, %i] {async_task_id = array<i32: 2>, loop.cluster = 5 : i32, loop.stage = 0 : i32} : (!tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #offsets>, i32) -> tensor<128x64xf16, #blocked>
      // CHECK-NOT: ttg.local_store
      ttg.local_store %gather, %gather_buffer {async_task_id = array<i32: 2>, loop.cluster = 9 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK-NOT: async_task_id
      scf.yield {async_task_id = array<i32: 0, 2>}
    } {async_task_id = array<i32: 0, 2>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["default", "unused", "load"],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
