// RUN: triton-opt %s --nvws-insert-semaphore -allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @loop_carried_tensor
  tt.func @loop_carried_tensor(%lb: i32, %ub: i32, %step: i32) {
    %zero = arith.constant dense<0> : tensor<1xi32, #blocked>
    scf.for %i = %lb to %ub step %step iter_args(%arg = %zero) -> (tensor<1xi32, #blocked>) : i32 {
      %producer = "make_tensor"() {ttg.partition = array<i32: 1>} : () -> tensor<1xi32, #blocked>
      // CHECK: nvws.semaphore.create
      // CHECK: nvws.semaphore.create
      // CHECK: ttg.local_load
      // CHECK: scf.yield
      scf.yield {ttg.partition = array<i32: 0, 1>} %producer : tensor<1xi32, #blocked>
    } {tt.num_stages = 1 : i32, tt.scheduled_max_stage = 0 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
