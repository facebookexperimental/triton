// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="smem-budget=1" | FileCheck %s

// Test that the budget-aware convert elimination does not mutate a local_load
// backward chain in place when a value in the chain has users outside the
// rewrite set. Instead, the pass should fall back to inserting a convert_layout
// for the affected operand, keeping the external user's encoding intact.
//
// Setup: a function argument anchored at #blocked_a (low-score layout) feeds
// through a convert_layout to #blocked_b (high-score layout). A local_load
// in #blocked_b feeds through arith.extf into arith.subf (which also consumes
// the convert result). The extf result also feeds arith.negf (an external
// user not in the rewrite set). The main pass (steps 1-4) cannot eliminate the
// convert because the function argument's encoding is immutable. The
// budget-aware step 5 then tries to eliminate this over-budget convert by
// propagating #blocked_a forward. The backward chain from arith.subf
// operand 1 reaches the local_load through arith.extf, but arith.extf's
// result also feeds arith.negf (external user). Mutating arith.extf's type
// in place would create an encoding mismatch on arith.negf.

// CHECK-DAG: #[[$BLOCKED_A:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK-DAG: #[[$BLOCKED_B:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: @local_load_chain_external_user
// The local_load and extf must keep #blocked_b (not mutated to #blocked_a)
// CHECK: ttg.local_load %{{.*}} -> tensor<128x64xbf16, #[[$BLOCKED_B]]>
// CHECK: arith.extf {{.*}} : tensor<128x64xbf16, #[[$BLOCKED_B]]> to tensor<128x64xf32, #[[$BLOCKED_B]]>
// A convert_layout is inserted as a fallback for the subf operand
// CHECK: ttg.convert_layout {{.*}} : tensor<128x64xf32, #[[$BLOCKED_B]]> -> tensor<128x64xf32, #[[$BLOCKED_A]]>
// The subf operates in #blocked_a (the propagated source encoding)
// CHECK: arith.subf {{.*}} : tensor<128x64xf32, #[[$BLOCKED_A]]>
// The external user (negf) keeps #blocked_b — no encoding mismatch
// CHECK: arith.negf {{.*}} : tensor<128x64xf32, #[[$BLOCKED_B]]>

#blocked_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @local_load_chain_external_user(
      %arg0: tensor<128x64xf32, #blocked_a>,
      %smem: !ttg.memdesc<128x64xbf16, #shared, #smem>,
      %out1: !ttg.memdesc<128x64xf32, #shared, #smem, mutable>,
      %out2: !ttg.memdesc<128x64xf32, #shared, #smem, mutable>) {
    // Convert from immutable anchor #blocked_a to #blocked_b — survives steps 1-4
    %cvt = ttg.convert_layout %arg0 : tensor<128x64xf32, #blocked_a> -> tensor<128x64xf32, #blocked_b>
    // local_load with high-score layout (sizePerThread=[1,8], score=8)
    %load = ttg.local_load %smem : !ttg.memdesc<128x64xbf16, #shared, #smem> -> tensor<128x64xbf16, #blocked_b>
    // Elementwise chain from load
    %ext = arith.extf %load : tensor<128x64xbf16, #blocked_b> to tensor<128x64xf32, #blocked_b>
    // Use 1: subf consumes convert result and local_load chain (in rewrite set)
    %sub = arith.subf %cvt, %ext : tensor<128x64xf32, #blocked_b>
    ttg.local_store %sub, %out1 : tensor<128x64xf32, #blocked_b> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    // Use 2: negf consumes same chain value (external user, NOT in rewrite set)
    %neg = arith.negf %ext : tensor<128x64xf32, #blocked_b>
    ttg.local_store %neg, %out2 : tensor<128x64xf32, #blocked_b> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    tt.return
  }
}
