// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-optimize-partition-warps | FileCheck %s

// Test that non-default partitions are capped at the base warp group size (4)
// when the module's num_warps is greater than 4. Only the default partition
// should use the user's num_warps setting.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#shared_1d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK: module attributes {{.*}}"ttg.num-warps" = 8
module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 8 : i32} {

// CHECK-LABEL: @non_default_partitions_capped_to_base_warps
tt.func @non_default_partitions_capped_to_base_warps(%arg0: i32) {
  ttg.warp_specialize(%arg0)
    attributes {"ttg.partition.types" = ["default", "gemm", "load", "computation"]}
  default {
    ttg.warp_yield
  }
  // Partitions initialized at 8 warps should be shrunk.
  // gemm: scalar-only, shrinks to 1
  // CHECK: partition0({{.*}}) num_warps(1)
  partition0(%arg1: i32) num_warps(8) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  }
  // load: scalar-only, shrinks to 1
  // CHECK: partition1({{.*}}) num_warps(1)
  partition1(%arg1: i32) num_warps(8) {
    %0 = arith.muli %arg1, %arg1 : i32
    ttg.warp_return
  }
  // computation: scalar-only, shrinks to 1
  // CHECK: partition2({{.*}}) num_warps(1)
  partition2(%arg1: i32) num_warps(8) {
    %0 = arith.subi %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

// Verify that num_warps=4 behaves the same as before (no regression).
// CHECK-LABEL: @num_warps_4_unchanged
tt.func @num_warps_4_unchanged(%arg0: i32) {
  ttg.warp_specialize(%arg0)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(1)
  partition0(%arg1: i32) num_warps(4) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

}
