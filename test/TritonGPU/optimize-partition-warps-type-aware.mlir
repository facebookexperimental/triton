// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-optimize-partition-warps | FileCheck %s

// Tests for type-aware warp assignment in OptimizePartitionWarps pass.
// When partition types are specified via ttg.partition.types attribute:
// - For bwd FA (has reduction + computation): last partition gets 8 warps

#blocked8 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared_1d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 8 : i32} {

// Test 1: BWD FA pattern - computation (last partition) gets 8 warps
// CHECK-LABEL: @bwd_fa_computation_gets_8_warps
tt.func @bwd_fa_computation_gets_8_warps(%arg0: i32) {
  ttg.warp_specialize(%arg0) attributes {"ttg.partition.types" = ["reduction", "gemm", "load", "computation"]}
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(1)
  partition0(%arg1: i32) num_warps(8) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  }
  // CHECK: partition1({{.*}}) num_warps(1)
  partition1(%arg1: i32) num_warps(8) {
    %0 = arith.muli %arg1, %arg1 : i32
    ttg.warp_return
  }
  // CHECK: partition2({{.*}}) num_warps(8)
  // computation (last partition) gets 8 warps
  partition2(%arg1: i32) num_warps(4) {
    %0 = arith.subi %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

// Test 2: Without partition types attribute, normal optimization applies
// CHECK-LABEL: @no_partition_types_normal_optimization
tt.func @no_partition_types_normal_optimization(%arg0: i32) {
  ttg.warp_specialize(%arg0)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(1)
  partition0(%arg1: i32) num_warps(8) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  }
  // CHECK: partition1({{.*}}) num_warps(1)
  partition1(%arg1: i32) num_warps(8) {
    %0 = arith.subi %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

// Test 3: Without reduction, computation does not get override
// CHECK-LABEL: @no_reduction_no_override
tt.func @no_reduction_no_override(%arg0: i32) {
  ttg.warp_specialize(%arg0) attributes {"ttg.partition.types" = ["gemm", "load", "computation"]}
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(1)
  partition0(%arg1: i32) num_warps(8) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  }
  // CHECK: partition1({{.*}}) num_warps(1)
  partition1(%arg1: i32) num_warps(8) {
    %0 = arith.muli %arg1, %arg1 : i32
    ttg.warp_return
  }
  // CHECK: partition2({{.*}}) num_warps(1)
  partition2(%arg1: i32) num_warps(4) {
    %0 = arith.subi %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

// Test 4: Empty partition types array - should behave like no attribute
// CHECK-LABEL: @empty_partition_types
tt.func @empty_partition_types(%arg0: i32) {
  ttg.warp_specialize(%arg0) attributes {"ttg.partition.types" = []}
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(1)
  partition0(%arg1: i32) num_warps(8) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

// Test 5: A truncated type array must not be treated as if it described the
// specialized partitions. In particular, do not apply the computation warp
// override when role alignment is unknown.
// CHECK-LABEL: @truncated_partition_types_do_not_mislabel
// CHECK: ttg.warp_specialize({{.*}}) attributes {requestedRegisters = array<i32: 24, 24, 24>, ttg.partition.types = ["reduction", "computation"]}
// CHECK: partition0({{.*}}) num_warps(1)
// CHECK: partition1({{.*}}) num_warps(1)
// CHECK: partition2({{.*}}) num_warps(1)
tt.func @truncated_partition_types_do_not_mislabel(%arg0: i32) {
  ttg.warp_specialize(%arg0) attributes {"ttg.partition.types" = ["reduction", "computation"]}
  default {
    ttg.warp_yield
  }
  partition0(%arg1: i32) num_warps(4) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  }
  partition1(%arg1: i32) num_warps(4) {
    %0 = arith.subi %arg1, %arg1 : i32
    ttg.warp_return
  }
  partition2(%arg1: i32) num_warps(4) {
    %0 = arith.muli %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

}

// -----

// The 2-CTA backward schedule keeps computation in the default region. The
// four specialized scalar partitions map to reduction, GEMM, load, and relay;
// all shrink through the normal heuristic and use min_reg_auto_ws.
// CHECK-LABEL: @two_cta_bwd_specialized_partitions
// CHECK: ttg.warp_specialize() attributes {requestedRegisters = array<i32: 88, 88, 88, 88>, ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"]}
// CHECK: partition0() num_warps(1)
// CHECK: partition1() num_warps(1)
// CHECK: partition2() num_warps(1)
// CHECK: partition3() num_warps(1)
module attributes {ttg.max_reg_auto_ws = 88 : i32, ttg.min_reg_auto_ws = 88 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
tt.func @two_cta_bwd_specialized_partitions() {
  ttg.warp_specialize() attributes {"ttg.partition.types" = ["computation", "reduction", "gemm", "load", "relay"]}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(8) {
    ttg.warp_return
  }
  partition1() num_warps(8) {
    ttg.warp_return
  }
  partition2() num_warps(8) {
    ttg.warp_return
  }
  partition3() num_warps(8) {
    ttg.warp_return
  } : () -> ()
  tt.return
}
}
