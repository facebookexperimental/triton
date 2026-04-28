// RUN: triton-opt %s --nvws-insert-semaphore -allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @loop_result_tensor
  tt.func @loop_result_tensor(%lb: i32, %ub: i32, %step: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #blocked>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create {{.*}} true
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create {{.*}} false
    // CHECK: [[LOOP:%.*]] = scf.for
    %loop = scf.for %i = %lb to %ub step %step iter_args(%arg = %zero) -> (tensor<32xf32, #blocked>) : i32 {
      %producer = "make_tensor"() {ttg.partition = array<i32: 1>} : () -> tensor<32xf32, #blocked>
      scf.yield {ttg.partition = array<i32: 1>} %producer : tensor<32xf32, #blocked>
    } {tt.num_stages = 1 : i32, tt.scheduled_max_stage = 0 : i32,
       tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.partition.outputs = [array<i32: 1>],
       ttg.partition.stages = [0 : i32, 0 : i32],
       ttg.warp_specialize.tag = 0 : i32}
    // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: ttg.local_store [[LOOP]], [[PBUF]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: [[CBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: [[LOAD:%.*]] = ttg.local_load [[CBUF]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: "use"([[LOAD]]) {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    "use"(%loop) {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : (tensor<32xf32, #blocked>) -> ()
    tt.return
  }
}
