// RUN: triton-opt %s --allow-unregistered-dialect --nvws-semaphore-optimize=num-stages=3 --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore | FileCheck %s --implicit-check-not=nvws.semaphore

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @semaphore_pipeline
  tt.func @semaphore_pipeline(%desc: !tt.tensordesc<128x64xf16, #shared>, %lb: i32, %ub: i32, %step: i32) {
    // Optimize: depth 1 becomes depth 3.
    // CHECK: [[BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16
    // Lower: each logical semaphore becomes a three-slot mbarrier.
    // CHECK: [[EMPTY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK-COUNT-3: ttng.init_barrier {{%.*}}, 1
    // CHECK: [[FULL:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK-COUNT-3: ttng.init_barrier {{%.*}}, 1
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %empty = nvws.semaphore.create %buf released = 1 {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %buf {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // Assign: one stage plus two phase values are threaded through the loop.
    // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (i32, i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: ttng.wait_barrier {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK: [[PBUF:%.*]] = ttg.memdesc_index [[BUF]][{{%.*}}] {ttg.partition = array<i32: 0>}
      // CHECK: [[FULL_STAGE:%.*]] = ttg.memdesc_index [[FULL]][{{%.*}}] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.barrier_expect [[FULL_STAGE]], 16384 {ttg.partition = array<i32: 0>}
      // CHECK: ttng.async_tma_copy_global_to_local {{.*}} [[PBUF]], [[FULL_STAGE]], {{.*}} {ttg.partition = array<i32: 0>}
      %ptok = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %pbuf = nvws.semaphore.buffer %empty, %ptok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %desc[%i, %i] 16384 %pbuf {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %full, %ptok [#nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: ttng.wait_barrier {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK: [[CBUF:%.*]] = ttg.memdesc_index [[BUF]][{{%.*}}] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_load [[CBUF]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.fence_async_shared {bCluster = false, ttg.partition = array<i32: 1>}
      // CHECK: ttng.arrive_barrier {{.*}}, 1 {ttg.partition = array<i32: 1>}
      %ctok = nvws.semaphore.acquire %full {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %cbuf = nvws.semaphore.buffer %full, %ctok {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %v = ttg.local_load %cbuf {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      nvws.semaphore.release %empty, %ctok [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      "use"(%v) {ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>}
    ttg.local_dealloc %buf : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
