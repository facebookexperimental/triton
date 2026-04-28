// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The first MMA operand gives member A its exact view. Member B is cached at
  // the same time as a generic view, whose allocShape includes the three-stage
  // backing. Replaying B's transpose must infer its result type from that view.
  // CHECK-LABEL: @memdesc_trans_preserves_staged_alloc_shape
  tt.func @memdesc_trans_preserves_staged_alloc_shape(
      %desc_a: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %desc_b: !tt.tensordesc<tensor<256x64xf16, #shared>>,
      %acc: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
      %tok: !ttg.async.token) {
    %false = arith.constant false
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 700 : i32} : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %a = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 700 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 700 : i32} : () -> !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 700 : i32} : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]], [[V2]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]], [[V2]] {pending_count = 1 : i32} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]>
    %r = scf.for %iv = %c0 to %c1 step %c1 iter_args(%flag = %false) -> (i1) : i32 {
      // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V3]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V6:%.*]]:2 = nvws.semaphore.buffer [[V3]], [[V5]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64>
      // CHECK: nvws.descriptor_load %{{[-A-Za-z0-9_.$#]+}}[%{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}] 16384 [[V6]]#0 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %desc_a[%c0, %c0] 16384 %a {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: nvws.descriptor_load %{{[-A-Za-z0-9_.$#]+}}[%{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}] 32768 [[V6]]#1 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<256x64xf16, #shared>>, i32, i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64>
      // CHECK: nvws.semaphore.release [[V4]], [[V5]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.descriptor_load %desc_b[%c0, %c0] 32768 %b {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<256x64xf16, #shared>>, i32, i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      // CHECK: [[V7:%.*]] = nvws.semaphore.acquire [[V4]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V8:%.*]]:2 = nvws.semaphore.buffer [[V4]], [[V7]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64>
      // CHECK-NEXT: [[V9:%.*]] = ttg.memdesc_trans [[V8]]#1 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256>
      %bt = ttg.memdesc_trans %b {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared_t, #smem, mutable>
      // CHECK-NEXT: [[V10:%.*]] = ttng.tc_gen5_mma [[V8]]#0, [[V9]], %{{[-A-Za-z0-9_.$#]+}}[%{{[-A-Za-z0-9_.$#]+}}], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: nvws.semaphore.release [[V3]], [[V7]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %mma = ttng.tc_gen5_mma %a, %bt, %acc[%tok], %flag, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared_t, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: "use_token"([[V10]]) {ttg.partition = array<i32: 0>}
      "use_token"(%mma) {ttg.partition = array<i32: 0>} : (!ttg.async.token) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %true : i1
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i1"(%r) : (i1) -> ()
    tt.return
  }
}
