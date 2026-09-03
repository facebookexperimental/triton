// RUN: triton-opt -split-input-file --tlx-insert-require-layout --tlx-propagate-layout %s | FileCheck %s

// Test that when the combined insert+propagate pipeline cannot make all tensor
// carrier predecessors agree on a concrete layout, it stays valid by keeping an
// explicit layout conversion instead of retagging the carrier.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @conflicting_scf_if_result_types
  tt.func public @conflicting_scf_if_result_types(%cond: i1) -> tensor<64x64xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %zero_a = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared, #smem, mutable>
    %buf_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    %buf_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    // CHECK: scf.if {{.*}} -> (tensor<64x32xf16, #blocked>)
    %if_a = scf.if %cond -> (tensor<64x32xf16, #blocked>) {
      // CHECK: ttg.local_load {{.*}} -> tensor<64x32xf16, #blocked>
      %a = ttg.local_load %buf_a : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #blocked>
      scf.yield %a : tensor<64x32xf16, #blocked>
    } else {
      scf.yield %zero_a : tensor<64x32xf16, #blocked>
    }
    %b = ttg.local_load %buf_b : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> tensor<32x64xf16, #blocked>
    // CHECK: %[[A_DOT:.*]] = ttg.convert_layout %{{.*}} : tensor<64x32xf16, #{{.*}}> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %a_dot = ttg.convert_layout %if_a : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %b_dot = ttg.convert_layout %b : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    // CHECK: tt.dot %[[A_DOT]], %{{.*}}, %{{.*}}
    %dot = tt.dot %a_dot, %b_dot, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return %dot : tensor<64x64xf32, #mma>
  }
}

// -----
#blocked_nested = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_nested = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#dot_nested = #ttg.dot_op<{opIdx = 0, parent = #mma_nested, kWidth = 4}>
#shared_nested = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_nested = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-DAG: #[[$NESTED_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-DAG: #[[$NESTED_MMA:.*]] = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
  // CHECK-LABEL: @nested_non_retaggable_edge
  tt.func public @nested_non_retaggable_edge(
      %cond: i1,
      %x: tensor<64x32xf32, #blocked_nested>) -> tensor<64x32xbf16, #dot_nested> {
    %trunc = arith.truncf %x : tensor<64x32xf32, #blocked_nested> to tensor<64x32xbf16, #blocked_nested>
    %alloc = ttg.local_alloc %trunc : (tensor<64x32xbf16, #blocked_nested>) -> !ttg.memdesc<64x32xbf16, #shared_nested, #smem_nested>
    // CHECK: ttg.local_load {{.*}} -> tensor<64x32xbf16, #[[$NESTED_BLOCKED]]>
    %load = ttg.local_load %alloc : !ttg.memdesc<64x32xbf16, #shared_nested, #smem_nested> -> tensor<64x32xbf16, #blocked_nested>
    // CHECK: scf.if {{.*}} -> (tensor<64x32xbf16, #[[$NESTED_BLOCKED]]>)
    %selected = scf.if %cond -> (tensor<64x32xbf16, #blocked_nested>) {
      %inner = scf.if %cond -> (tensor<64x32xbf16, #blocked_nested>) {
        scf.yield %trunc : tensor<64x32xbf16, #blocked_nested>
      } else {
        scf.yield %trunc : tensor<64x32xbf16, #blocked_nested>
      }
      scf.yield %inner : tensor<64x32xbf16, #blocked_nested>
    } else {
      scf.yield %load : tensor<64x32xbf16, #blocked_nested>
    }
    %required = tlx.require_layout %selected : tensor<64x32xbf16, #blocked_nested> -> tensor<64x32xbf16, #dot_nested>
    tt.return %required : tensor<64x32xbf16, #dot_nested>
  }

  // CHECK-LABEL: @nested_blocked_inward
  tt.func public @nested_blocked_inward(
      %cond: i1,
      %x: tensor<64x32xf32, #blocked_nested>,
      %buf: !ttg.memdesc<64x32xbf16, #shared_nested, #smem_nested>) -> tensor<64x32xbf16, #dot_nested> {
    %trunc = arith.truncf %x : tensor<64x32xf32, #blocked_nested> to tensor<64x32xbf16, #blocked_nested>
    // CHECK: ttg.local_load {{.*}} -> tensor<64x32xbf16, #[[$NESTED_BLOCKED]]>
    %load = ttg.local_load %buf : !ttg.memdesc<64x32xbf16, #shared_nested, #smem_nested> -> tensor<64x32xbf16, #blocked_nested>
    // CHECK: scf.if {{.*}} -> (tensor<64x32xbf16, #[[$NESTED_BLOCKED]]>)
    %selected = scf.if %cond -> (tensor<64x32xbf16, #blocked_nested>) {
      %inner = scf.if %cond -> (tensor<64x32xbf16, #blocked_nested>) {
        scf.yield %load : tensor<64x32xbf16, #blocked_nested>
      } else {
        scf.yield %load : tensor<64x32xbf16, #blocked_nested>
      }
      scf.yield %inner : tensor<64x32xbf16, #blocked_nested>
    } else {
      scf.yield %trunc : tensor<64x32xbf16, #blocked_nested>
    }
    %required = tlx.require_layout %selected : tensor<64x32xbf16, #blocked_nested> -> tensor<64x32xbf16, #dot_nested>
    tt.return %required : tensor<64x32xbf16, #dot_nested>
  }

  // CHECK-LABEL: @nested_retaggable_edges
  tt.func public @nested_retaggable_edges(
      %cond: i1,
      %buf: !ttg.memdesc<64x32xbf16, #shared_nested, #smem_nested>) -> tensor<64x32xbf16, #dot_nested> {
    // CHECK: ttg.local_load {{.*}} -> tensor<64x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$NESTED_MMA]], kWidth = 4}>>
    %load = ttg.local_load %buf : !ttg.memdesc<64x32xbf16, #shared_nested, #smem_nested> -> tensor<64x32xbf16, #blocked_nested>
    // CHECK: scf.if {{.*}} -> (tensor<64x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$NESTED_MMA]], kWidth = 4}>>)
    %selected = scf.if %cond -> (tensor<64x32xbf16, #blocked_nested>) {
      // CHECK: scf.if {{.*}} -> (tensor<64x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$NESTED_MMA]], kWidth = 4}>>)
      %inner = scf.if %cond -> (tensor<64x32xbf16, #blocked_nested>) {
        scf.yield %load : tensor<64x32xbf16, #blocked_nested>
      } else {
        scf.yield %load : tensor<64x32xbf16, #blocked_nested>
      }
      scf.yield %inner : tensor<64x32xbf16, #blocked_nested>
    } else {
      scf.yield %load : tensor<64x32xbf16, #blocked_nested>
    }
    %required = tlx.require_layout %selected : tensor<64x32xbf16, #blocked_nested> -> tensor<64x32xbf16, #dot_nested>
    tt.return %required : tensor<64x32xbf16, #dot_nested>
  }
}

// -----
// Test that when tensor propagation through an scf.for carrier cannot make the
// init value and the backedge agree, the pipeline keeps an explicit layout
// conversion instead of retagging the loop-carried tensor.

#blocked_for = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_for = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared_for = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_for = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @conflicting_scf_for_iter_arg_types
  tt.func public @conflicting_scf_for_iter_arg_types() -> tensor<64x64xf32, #mma_for> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %zero_a = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #blocked_for>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_for>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared_for, #smem_for, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared_for, #smem_for, mutable>
    %buf_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared_for, #smem_for, mutable> -> !ttg.memdesc<64x32xf16, #shared_for, #smem_for, mutable>
    %buf_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared_for, #smem_for, mutable> -> !ttg.memdesc<32x64xf16, #shared_for, #smem_for, mutable>
    // CHECK: scf.for {{.*}} iter_args(%[[A_ARG:.*]] = %{{.*}}) -> (tensor<64x32xf16, #blocked>)
    %loop_a = scf.for %i = %c0 to %c2 step %c1 iter_args(%a_reg = %zero_a) -> (tensor<64x32xf16, #blocked_for>) {
      // CHECK: ttg.local_load {{.*}} -> tensor<64x32xf16, #blocked>
      %a_next = ttg.local_load %buf_a : !ttg.memdesc<64x32xf16, #shared_for, #smem_for, mutable> -> tensor<64x32xf16, #blocked_for>
      scf.yield %a_next : tensor<64x32xf16, #blocked_for>
    }
    // CHECK: %[[B_LOAD:.*]] = ttg.local_load {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %b = ttg.local_load %buf_b : !ttg.memdesc<32x64xf16, #shared_for, #smem_for, mutable> -> tensor<32x64xf16, #blocked_for>
    // CHECK: %[[A_DOT:.*]] = ttg.convert_layout %{{.*}} : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %a_dot = ttg.convert_layout %loop_a : tensor<64x32xf16, #blocked_for> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_for, kWidth = 4}>>
    %b_dot = ttg.convert_layout %b : tensor<32x64xf16, #blocked_for> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_for, kWidth = 4}>>
    // CHECK: tt.dot %[[A_DOT]], %[[B_LOAD]], %{{.*}}
    %dot = tt.dot %a_dot, %b_dot, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_for, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_for, kWidth = 4}>> -> tensor<64x64xf32, #mma_for>
    tt.return %dot : tensor<64x64xf32, #mma_for>
  }
}
