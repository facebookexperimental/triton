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
