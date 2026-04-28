// RUN: triton-opt %s --nvws-partition-scheduling-meta -allow-unregistered-dialect | FileCheck %s

module attributes {"ttg.num-warps" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @scalar_iter_arg_nested_loop
  // CHECK: %[[BASE0:.*]] = arith.addi {{.*}} {ttg.partition = array<i32: 0, 1, 2>} : i32
  // CHECK: %[[BASE:.*]] = arith.addi %[[BASE0]], {{.*}} {ttg.partition = array<i32: 0, 1, 2>} : i32
  // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[BASE]])
  tt.func public @scalar_iter_arg_nested_loop(%n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    scf.for %i = %c0_i32 to %n step %c1_i32 : i32 {
      %base0 = arith.addi %i, %c1_i32 {ttg.partition = array<i32: 1>} : i32
      %base = arith.addi %base0, %c1_i32 {ttg.partition = array<i32: 1>} : i32
      %inner = scf.for %j = %c0_i32 to %n step %c1_i32
          iter_args(%idx = %base) -> (i32) : i32 {
        %use0 = arith.addi %idx, %c1_i32 {ttg.partition = array<i32: 0>} : i32
        %use2 = arith.addi %idx, %c1_i32 {ttg.partition = array<i32: 2>} : i32
        %next = arith.addi %use0, %use2 {ttg.partition = array<i32: 0, 2>} : i32
        scf.yield %next : i32
      } {ttg.partition = array<i32: 0, 1, 2>,
         ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
         ttg.partition.types = ["consumer0", "producer", "consumer1"]}
      "use_inner"(%inner) {ttg.partition = array<i32: 0, 2>} : (i32) -> ()
      scf.yield
    } {tt.warp_specialize, ttg.warp_specialize.tag = 0 : i32,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["consumer0", "producer", "consumer1"]}

    tt.return
  }
}
