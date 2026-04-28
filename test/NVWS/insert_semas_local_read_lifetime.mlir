// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @outer_produced_inner_consumed
  tt.func @outer_produced_inner_consumed(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 200 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 200 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] released = -1 {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
    scf.for %outer = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 2>} : () -> tensor<128x64xf16, #blocked>
      // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V5:%.*]] = nvws.semaphore.buffer [[V2]], [[V4]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]] {ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

      // CHECK-NEXT: nvws.semaphore.release [[V3]], [[V4]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // The inner loop acquires once before entry. Every read uses that token,
      // and one release after the loop returns the buffer to owner {2}.
      // CHECK-NEXT: [[V6:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
      scf.for %inner = %lb to %ub step %step : i32 {
        // CHECK-NEXT: [[V7:%.*]] = nvws.semaphore.buffer [[V3]], [[V6]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[V8:%.*]] = ttg.local_load [[V7]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK-NEXT: "use_tensor"([[V8]]) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
        %loaded = ttg.local_load %alloc {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
        "use_tensor"(%loaded) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
      // CHECK-NEXT: } {ttg.partition = array<i32: 1>}
      } {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[V2]], [[V6]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
