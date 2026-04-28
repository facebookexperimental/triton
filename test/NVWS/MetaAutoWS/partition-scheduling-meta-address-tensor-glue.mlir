// RUN: triton-opt %s --nvws-partition-scheduling-meta -allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-warps" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @address_tensor_glue_spans_consumers
  // CHECK: arith.addi {{.*}} {ttg.partition = array<i32: 0, 2, 3>} : i32
  // CHECK: %[[SPLAT:.*]] = tt.splat {{.*}} {ttg.partition = array<i32: 0, 2, 3>} : i32 -> tensor<128xi32, #blocked>
  // CHECK: arith.addi %[[SPLAT]], {{.*}} {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
  tt.func public @address_tensor_glue_spans_consumers(%n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    scf.for %i = %c0_i32 to %n step %c1_i32 : i32 {
      %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
      %base2 = arith.addi %i, %c1_i32 {ttg.partition = array<i32: 2>} : i32
      %base3 = arith.addi %i, %c1_i32 {ttg.partition = array<i32: 3>} : i32
      %idx = arith.addi %base2, %base3 {ttg.partition = array<i32: 2, 3>} : i32
      %splat = tt.splat %idx {ttg.partition = array<i32: 2, 3>} : i32 -> tensor<128xi32, #blocked>
      %offsets = arith.addi %splat, %range {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
      "use_offsets"(%offsets) {ttg.partition = array<i32: 0>} : (tensor<128xi32, #blocked>) -> ()
      "use_idx2"(%idx) {ttg.partition = array<i32: 2>} : (i32) -> ()
      "use_idx3"(%idx) {ttg.partition = array<i32: 3>} : (i32) -> ()
      scf.yield
    } {tt.warp_specialize, ttg.warp_specialize.tag = 0 : i32,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["address", "unused", "producer0", "producer1"]}

    tt.return
  }
}
