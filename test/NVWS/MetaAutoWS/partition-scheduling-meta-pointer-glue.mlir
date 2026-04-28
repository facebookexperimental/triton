// RUN: triton-opt %s --nvws-partition-scheduling-meta -allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-warps" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @pointer_glue_spans_consumers
  // CHECK: tt.addptr {{.*}}ttg.partition = array<i32: [[P0:[0-9]+]], [[P1:[0-9]+]], [[P2:[0-9]+]]>
  // CHECK: tt.load {{.*}}ttg.partition = array<i32: [[P0]], [[P1]], [[P2]]>
  // CHECK: tt.splat {{.*}}ttg.partition = array<i32: [[P1]]>
  // CHECK: tt.splat {{.*}}ttg.partition = array<i32: [[P2]]>
  tt.func public @pointer_glue_spans_consumers(%idx_ptr: !tt.ptr<f32>, %n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    scf.for %i = %c0_i32 to %n step %c1_i32 : i32 {
      %ptr = tt.addptr %idx_ptr, %i {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      %idx = tt.load %ptr {ttg.partition = array<i32: 0, 1, 2>} : !tt.ptr<f32>
      %bias0 = tt.splat %idx {ttg.partition = array<i32: 1>} : f32 -> tensor<32xf32, #blocked>
      %bias1 = tt.splat %idx {ttg.partition = array<i32: 2>} : f32 -> tensor<32xf32, #blocked>
      "use_bias0"(%bias0) {ttg.partition = array<i32: 1>} : (tensor<32xf32, #blocked>) -> ()
      "use_bias1"(%bias1) {ttg.partition = array<i32: 2>} : (tensor<32xf32, #blocked>) -> ()
      scf.yield
    } {tt.warp_specialize, ttg.warp_specialize.tag = 0 : i32,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["producer", "consumer0", "consumer1"]}

    tt.return
  }
}
