// RUN: triton-opt %s --tlx-resolve-placeholder-layouts | FileCheck %s

// Removing no_verify changes both the arith.constant result and the shaped
// DenseElementsAttr stored in its value attribute. Keep those types aligned so
// the operation remains valid when early placeholder resolution verifies IR.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#deferred = #tlx.no_verify_layout<#blocked>

// CHECK-DAG: #[[$BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
// CHECK-NOT: #tlx.no_verify_layout
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @realign_dense_constant_type
  tt.func @realign_dense_constant_type() -> tensor<256xi32, #deferred> {
    // CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<256xi32, #[[$BLOCKED]]>
    %cst = arith.constant dense<0> : tensor<256xi32, #deferred>
    // CHECK: tt.return %[[CST]] : tensor<256xi32, #[[$BLOCKED]]>
    tt.return %cst : tensor<256xi32, #deferred>
  }
}
