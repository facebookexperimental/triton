// RUN: triton-opt %s --nvws-partition-scheduling-meta -allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-warps" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @yield_output_uses_producer_partition
  // CHECK: %[[LOOP:.*]] = scf.for
  // CHECK: "test.producer"() {ttg.partition = array<i32: 1>}
  // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>}
  // CHECK: } {tt.warp_specialize
  // CHECK-SAME: ttg.partition.outputs = [array<i32: 1>]
  // CHECK-SAME: ttg.partition.stages = [0 : i32, 0 : i32]
  // CHECK: "test.consumer"(%[[LOOP]]) {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
  tt.func public @yield_output_uses_producer_partition(%n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<32xf32, #blocked>

    %loop_out = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(
      %acc = %zero
    ) -> (tensor<32xf32, #blocked>) : i32 {
      %producer = "test.producer"() {ttg.partition = array<i32: 1>} : () -> tensor<32xf32, #blocked>
      scf.yield %producer : tensor<32xf32, #blocked>
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32],
       ttg.partition.types = ["consumer", "producer"],
       ttg.warp_specialize.tag = 0 : i32}

    "test.consumer"(%loop_out) {ttg.partition = array<i32: 0>} : (tensor<32xf32, #blocked>) -> ()
    tt.return
  }
}
