// RUN: triton-opt -split-input-file --tlx-propagate-layout %s | FileCheck %s

// Test that residual tensor require/release ops lower to convert_layout ops
// after propagation.

#blocked_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @lower_residual_tensor_constraints
  tt.func public @lower_residual_tensor_constraints(%arg: tensor<8x8xf16, #blocked_a>) -> tensor<8x8xf16, #blocked_a> {
    // CHECK: %[[REQ:.*]] = ttg.convert_layout %{{.*}} : tensor<8x8xf16, #{{.*}}> -> tensor<8x8xf16, #{{.*}}>
    %req = tlx.require_layout %arg : tensor<8x8xf16, #blocked_a> -> tensor<8x8xf16, #blocked_b>
    // CHECK: %[[REL:.*]] = ttg.convert_layout %[[REQ]] : tensor<8x8xf16, #{{.*}}> -> tensor<8x8xf16, #{{.*}}>
    %rel = tlx.release_layout %req : tensor<8x8xf16, #blocked_b> -> tensor<8x8xf16, #blocked_a>
    // CHECK: tt.return %[[REL]]
    tt.return %rel : tensor<8x8xf16, #blocked_a>
  }
}

// -----
// Test that an identity tensor require_layout folds away.

#blocked_req_id = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @erase_identity_require_layout
  tt.func public @erase_identity_require_layout(%arg: tensor<8x8xf16, #blocked_req_id>) -> tensor<8x8xf16, #blocked_req_id> {
    // CHECK-NOT: tlx.require_layout
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.return %{{.*}} : tensor<8x8xf16, #{{.*}}>
    %req = tlx.require_layout %arg : tensor<8x8xf16, #blocked_req_id> -> tensor<8x8xf16, #blocked_req_id>
    tt.return %req : tensor<8x8xf16, #blocked_req_id>
  }
}

// -----
// Test that an identity tensor release_layout folds away.

#blocked_id = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @erase_identity_release_layout
  tt.func public @erase_identity_release_layout(%arg: tensor<8x8xf16, #blocked_id>) -> tensor<8x8xf16, #blocked_id> {
    // CHECK-NOT: tlx.release_layout
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.return %{{.*}} : tensor<8x8xf16, #{{.*}}>
    %rel = tlx.release_layout %arg : tensor<8x8xf16, #blocked_id> -> tensor<8x8xf16, #blocked_id>
    tt.return %rel : tensor<8x8xf16, #blocked_id>
  }
}

// -----
// Test that a late dot-operand fallback through local_alloc/local_load is folded
// after the loop-carried tensor is retagged through RegionBranchOpInterface.

#blocked_loop = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_loop = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared_loop = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_loop = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @fold_loop_carried_dot_local_alloc
  tt.func public @fold_loop_carried_dot_local_alloc() -> tensor<64x64xf32, #mma_loop> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_loop>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #shared_loop, #smem_loop, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x32x64xf16, #shared_loop, #smem_loop, mutable>
    %buf_a0 = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<2x64x32xf16, #shared_loop, #smem_loop, mutable> -> !ttg.memdesc<64x32xf16, #shared_loop, #smem_loop, mutable>
    %buf_b0 = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<2x32x64xf16, #shared_loop, #smem_loop, mutable> -> !ttg.memdesc<32x64xf16, #shared_loop, #smem_loop, mutable>
    // CHECK: %[[A_INIT:.*]] = ttg.local_load {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 4}>>
    %a_init = ttg.local_load %buf_a0 : !ttg.memdesc<64x32xf16, #shared_loop, #smem_loop, mutable> -> tensor<64x32xf16, #blocked_loop>
    // CHECK: %[[B_INIT:.*]] = ttg.local_load {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 4}>>
    %b_init = ttg.local_load %buf_b0 : !ttg.memdesc<32x64xf16, #shared_loop, #smem_loop, mutable> -> tensor<32x64xf16, #blocked_loop>
    // CHECK: scf.for {{.*}} iter_args(%[[ACC_ARG:.*]] = {{.*}}, %[[A_ARG:.*]] = %[[A_INIT]], %[[B_ARG:.*]] = %[[B_INIT]]) -> (tensor<64x64xf32, #{{.*}}>, tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 4}>>, tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 4}>>)
    %result:3 = scf.for %i = %c0_i32 to %c2_i32 step %c1_i32
        iter_args(%acc = %cst, %a_reg = %a_init, %b_reg = %b_init)
        -> (tensor<64x64xf32, #mma_loop>, tensor<64x32xf16, #blocked_loop>, tensor<32x64xf16, #blocked_loop>) : i32 {
      %a_tmp = ttg.local_alloc %a_reg : (tensor<64x32xf16, #blocked_loop>) -> !ttg.memdesc<64x32xf16, #shared_loop, #smem_loop>
      %a_dot = ttg.local_load %a_tmp : !ttg.memdesc<64x32xf16, #shared_loop, #smem_loop> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_loop, kWidth = 4}>>
      %b_tmp = ttg.local_alloc %b_reg : (tensor<32x64xf16, #blocked_loop>) -> !ttg.memdesc<32x64xf16, #shared_loop, #smem_loop>
      %b_dot = ttg.local_load %b_tmp : !ttg.memdesc<32x64xf16, #shared_loop, #smem_loop> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_loop, kWidth = 4}>>
      // CHECK-NOT: ttg.local_alloc %
      // CHECK: tt.dot %[[A_ARG]], %[[B_ARG]], %[[ACC_ARG]]
      %dot = tt.dot %a_dot, %b_dot, %acc : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_loop, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_loop, kWidth = 4}>> -> tensor<64x64xf32, #mma_loop>
      %buf_a1 = ttg.memdesc_index %alloc_a[%c1_i32] : !ttg.memdesc<2x64x32xf16, #shared_loop, #smem_loop, mutable> -> !ttg.memdesc<64x32xf16, #shared_loop, #smem_loop, mutable>
      %buf_b1 = ttg.memdesc_index %alloc_b[%c1_i32] : !ttg.memdesc<2x32x64xf16, #shared_loop, #smem_loop, mutable> -> !ttg.memdesc<32x64xf16, #shared_loop, #smem_loop, mutable>
      %a_next = ttg.local_load %buf_a1 : !ttg.memdesc<64x32xf16, #shared_loop, #smem_loop, mutable> -> tensor<64x32xf16, #blocked_loop>
      %b_next = ttg.local_load %buf_b1 : !ttg.memdesc<32x64xf16, #shared_loop, #smem_loop, mutable> -> tensor<32x64xf16, #blocked_loop>
      scf.yield %dot, %a_next, %b_next : tensor<64x64xf32, #mma_loop>, tensor<64x32xf16, #blocked_loop>, tensor<32x64xf16, #blocked_loop>
    }
    tt.return %result#0 : tensor<64x64xf32, #mma_loop>
  }
}

// -----
// Test that a single fallback local_alloc feeding multiple compatible dot
// local_load users is folded one load at a time. Dead alloc cleanup is left to
// canonicalization/DCE.

#blocked_multi = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_multi = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared_multi = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_multi = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @fold_multi_use_dot_local_alloc
  tt.func public @fold_multi_use_dot_local_alloc() -> tensor<64x64xf32, #mma_multi> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_multi>
    %b_arg = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_multi, kWidth = 4}>>
    %src_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared_multi, #smem_multi, mutable>
    %src_buf = ttg.memdesc_index %src_alloc[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared_multi, #smem_multi, mutable> -> !ttg.memdesc<64x32xf16, #shared_multi, #smem_multi, mutable>
    %src = ttg.local_load %src_buf : !ttg.memdesc<64x32xf16, #shared_multi, #smem_multi, mutable> -> tensor<64x32xf16, #blocked_multi>
    %fallback = ttg.local_alloc %src : (tensor<64x32xf16, #blocked_multi>) -> !ttg.memdesc<64x32xf16, #shared_multi, #smem_multi>
    %a0 = ttg.local_load %fallback : !ttg.memdesc<64x32xf16, #shared_multi, #smem_multi> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_multi, kWidth = 4}>>
    %a1 = ttg.local_load %fallback : !ttg.memdesc<64x32xf16, #shared_multi, #smem_multi> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_multi, kWidth = 4}>>
    // CHECK: %[[SRC:.*]] = ttg.local_load {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 4}>>
    // CHECK-NOT: ttg.local_alloc %
    // CHECK-NOT: ttg.local_load %{{.*}} : !ttg.memdesc<64x32xf16
    // CHECK: %[[DOT0:.*]] = tt.dot %[[SRC]], %{{.*}}, %{{.*}} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 4}>>
    %dot0 = tt.dot %a0, %b_arg, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_multi, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_multi, kWidth = 4}>> -> tensor<64x64xf32, #mma_multi>
    // CHECK: %[[DOT1:.*]] = tt.dot %[[SRC]], %{{.*}}, %[[DOT0]] : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 4}>>
    %dot1 = tt.dot %a1, %b_arg, %dot0 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_multi, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_multi, kWidth = 4}>> -> tensor<64x64xf32, #mma_multi>
    tt.return %dot1 : tensor<64x64xf32, #mma_multi>
  }
}
