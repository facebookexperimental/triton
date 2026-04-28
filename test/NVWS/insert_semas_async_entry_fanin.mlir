// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas \
// RUN:   --nvws-lower-semaphore -cse -o /dev/null

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @async_entry_fanin
  tt.func @async_entry_fanin(
      %desc: !tt.tensordesc<tensor<128x128xf16, #shared>>,
      %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 1701 : i32}
    // CHECK: [[OUTER_EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] released = -1 {pending_count = 1 : i32}
    // CHECK: [[INNER_READY:%.*]] = nvws.semaphore.create [[ALLOC]] {pending_count = 2 : i32}
    %alloc = ttg.local_alloc {buffer.id = 1701 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[OUTER_TOKEN:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: [[OUTER_BUFFER:%.*]] = nvws.semaphore.buffer [[OUTER_EMPTY]], [[OUTER_TOKEN]] {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 32768 [[OUTER_BUFFER]] {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: nvws.semaphore.release [[INNER_READY]], [[OUTER_TOKEN]] [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>}
      // CHECK-NEXT: nvws.semaphore.release [[INNER_READY]], [[OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>}
      nvws.descriptor_load %desc[%i, %i] 32768 %alloc {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NEXT: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
      scf.for %j = %lb to %ub step %step : i32 {
        // CHECK-NEXT: [[INNER_TOKEN:%.*]] = nvws.semaphore.acquire [[INNER_READY]] {ttg.partition = array<i32: 2>}
        %l2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
        "use2"(%l2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
        %l1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
        %corrected = "correct"(%l1) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
        ttg.local_store %corrected, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %l0 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
        "use0"(%l0) {ttg.partition = array<i32: 0>} : (!ty) -> ()
      } {ttg.partition = array<i32: 0, 1, 2>}
      // CHECK: [[DONE:%.*]] = nvws.semaphore.acquire [[INNER_READY]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[OUTER_EMPTY]], [[DONE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
