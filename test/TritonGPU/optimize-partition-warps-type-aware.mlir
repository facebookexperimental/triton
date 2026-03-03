// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-optimize-partition-warps | FileCheck %s

// Tests for type-aware warp assignment in OptimizePartitionWarps pass.
// When partition types are specified via ttg.partition.types attribute:
// - For bwd FA (has reduction): reduction at index 0 gets 4 warps
// - For bwd FA (has reduction): computation partition gets 8 warps

#blocked8 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared_1d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 8 : i32} {

// Test 1: bwd FA pattern - reduction at index 0 gets 4 warps
// When partition has "reduction" type at index 0, it gets 4 warps (bwd FA default)
// CHECK-LABEL: @bwd_fa_reduction_at_index_zero
tt.func @bwd_fa_reduction_at_index_zero(%arg0: i32) {
  // The partition types attribute marks partition index 0 as "reduction" (bwd FA pattern)
  ttg.warp_specialize(%arg0) attributes {"ttg.partition.types" = ["reduction", "gemm"]}
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(4)
  // Reduction at index 0 in bwd FA pattern gets 4 warps
  partition0(%arg1: i32) num_warps(8) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  }
  // CHECK: partition1({{.*}}) num_warps(1)
  // Other partitions get normal optimization (no tensor ops → 1 warp)
  partition1(%arg1: i32) num_warps(8) {
    %0 = arith.subi %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

// Test 2: Full bwd FA pattern - reduction / gemm / load / computation
// reduction at index 0 gets 4 warps, computation gets 8 warps
// CHECK-LABEL: @bwd_fa_full_pattern
tt.func @bwd_fa_full_pattern(%arg0: i32) {
  ttg.warp_specialize(%arg0) attributes {"ttg.partition.types" = ["reduction", "gemm", "load", "computation"]}
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(4)
  // reduction at index 0 gets 4 warps
  partition0(%arg1: i32) num_warps(8) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  }
  // CHECK: partition1({{.*}}) num_warps(1)
  // gemm partition with no tensor ops gets optimized to 1 warp
  partition1(%arg1: i32) num_warps(8) {
    %0 = arith.muli %arg1, %arg1 : i32
    ttg.warp_return
  }
  // CHECK: partition2({{.*}}) num_warps(1)
  // load partition with no tensor ops gets optimized to 1 warp
  partition2(%arg1: i32) num_warps(8) {
    %0 = arith.subi %arg1, %arg1 : i32
    ttg.warp_return
  }
  // CHECK: partition3({{.*}}) num_warps(8)
  // computation partition gets 8 warps
  partition3(%arg1: i32) num_warps(4) {
    %0 = arith.divsi %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

// Test 3: Without partition types attribute, normal optimization applies
// This verifies that the type-aware logic only activates when the attribute is present.
// CHECK-LABEL: @no_partition_types_normal_optimization
tt.func @no_partition_types_normal_optimization(%arg0: i32) {
  // No ttg.partition.types attribute - normal optimization applies
  ttg.warp_specialize(%arg0)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(1)
  // Without type attribute, partition with no tensor ops gets optimized to 1 warp
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

// Test 4: Mixed scenario - reduction at index 0 with tensor ops in gemm
// CHECK-LABEL: @bwd_fa_mixed_with_tensor_ops
tt.func @bwd_fa_mixed_with_tensor_ops(%arg0: i32) {
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared_1d, #smem, mutable>
  ttg.warp_specialize(%arg0, %alloc) attributes {"ttg.partition.types" = ["reduction", "gemm", "load"]}
  default {
    ttg.warp_yield
  }
  // CHECK: partition0({{.*}}) num_warps(4)
  // reduction at index 0 gets 4 warps even with no tensor computation
  partition0(%arg1: i32, %arg2: !ttg.memdesc<128xi32, #shared_1d, #smem, mutable>) num_warps(8) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  }
  // CHECK: partition1({{.*}}) num_warps(1)
  // gemm partition with small tensor computation can be optimized
  partition1(%arg1: i32, %arg2: !ttg.memdesc<128xi32, #shared_1d, #smem, mutable>) num_warps(8) {
    %0 = tt.splat %arg1 : i32 -> tensor<128xi32, #blocked8>
    ttg.local_store %0, %arg2 : tensor<128xi32, #blocked8> -> !ttg.memdesc<128xi32, #shared_1d, #smem, mutable>
    ttg.warp_return
  }
  // CHECK: partition2({{.*}}) num_warps(1)
  // load partition with no tensor ops gets optimized
  partition2(%arg1: i32, %arg2: !ttg.memdesc<128xi32, #shared_1d, #smem, mutable>) num_warps(4) {
    %0 = arith.subi %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32, !ttg.memdesc<128xi32, #shared_1d, #smem, mutable>) -> ()
  tt.return
}

// Test 5: Partition types with different number of partitions (not 4)
// The "last partition = 8 warps" rule only applies to exactly 4 partitions.
// CHECK-LABEL: @three_partitions_no_special_last
tt.func @three_partitions_no_special_last(%arg0: i32) {
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
  // With 3 partitions (not 4), the last partition does NOT get special treatment
  partition2(%arg1: i32) num_warps(4) {
    %0 = arith.subi %arg1, %arg1 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

// Test 6: Empty partition types array - should behave like no attribute
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

}
