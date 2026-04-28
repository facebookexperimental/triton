// RUN: triton-opt %s --nvws-strip-partition-attrs-outside-ws | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @strip_partition_attrs_outside_ws(%arg0: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %outside = arith.addi %arg0, %c1_i32 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : i32
    %loop = scf.for %iv = %c0_i32 to %outside step %c1_i32 iter_args(%arg1 = %outside) -> (i32) : i32 {
      %inside = arith.addi %arg1, %c1_i32 {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : i32
      scf.yield %inside : i32
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    %after = arith.muli %loop, %c1_i32 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : i32
    tt.return
  }
}

// CHECK-LABEL: @strip_partition_attrs_outside_ws
// CHECK: arith.constant 0 : i32
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: arith.addi %arg0, %{{.*}} : i32
// CHECK: arith.addi
// CHECK-SAME: ttg.partition = array<i32: 1>
// CHECK-SAME: ttg.warp_specialize.tag = 0 : i32
// CHECK: } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
// CHECK-NEXT: arith.muli %{{.*}}, %{{.*}} : i32
