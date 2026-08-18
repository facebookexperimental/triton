// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="smem-budget=1" | FileCheck %s

// Test that a nonzero budget does not force elimination of a conversion that
// needs no shared-memory scratch.
//
// Setup: a function argument anchored at #blocked_a feeds through a
// convert_layout to #blocked_b. A local_load in #blocked_b feeds through
// arith.extf into arith.subf (which also consumes the convert result) and
// arith.negf. The selected conversion is register-only, so it must retain the
// normal profitability behavior even though the numeric budget is tiny.

// CHECK-DAG: #[[$BLOCKED_A:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK-DAG: #[[$BLOCKED_B:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: @local_load_chain_external_user
// CHECK: ttg.local_load %{{.*}} -> tensor<128x64xbf16, #[[$BLOCKED_B]]>
// CHECK: arith.extf {{.*}} : tensor<128x64xbf16, #[[$BLOCKED_B]]> to tensor<128x64xf32, #[[$BLOCKED_B]]>
// CHECK: ttg.convert_layout {{.*}} : tensor<128x64xf32, #[[$BLOCKED_B]]> -> tensor<128x64xf32, #[[$BLOCKED_A]]>
// CHECK: arith.subf {{.*}} : tensor<128x64xf32, #[[$BLOCKED_A]]>
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

// -----

// A predicated accumulator update is a per-result layout carrier. Keep the
// MFMA layout selected by the surrounding dot chain instead of inserting an
// MFMA -> blocked -> MFMA redistribution around the EXEC-restricted region.
// The module marker identifies the deliberately provisional predicate/carried
// ownership before layout-conversion cleanup selects the MFMA layout.

#wp_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#wp_dot0 = #ttg.dot_op<{opIdx = 0, parent = #wp_mma, kWidth = 4}>
#wp_dot1 = #ttg.dot_op<{opIdx = 1, parent = #wp_mma, kWidth = 4}>
#wp_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#wp_row = #ttg.slice<{dim = 1, parent = #wp_mma}>

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-DAG: #[[$WP_MMA:.*]] = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
  // CHECK-LABEL: tt.func @warp_predicate_keeps_mfma_accumulator
  tt.func @warp_predicate_keeps_mfma_accumulator(
      %lhs: tensor<32x32xf16, #wp_dot0>,
      %rhs: tensor<32x32xf16, #wp_dot1>,
      %acc: tensor<32x32xf32, #wp_mma>,
      %predicate: tensor<32xi1, #wp_row>)
      -> tensor<32x32xf32, #wp_mma> {
    %dot0 = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #wp_dot0> * tensor<32x32xf16, #wp_dot1> -> tensor<32x32xf32, #wp_mma>
    %blocked = ttg.convert_layout %dot0 : tensor<32x32xf32, #wp_mma> -> tensor<32x32xf32, #wp_blocked>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: %[[PREDICATED:.*]] = ttg.warp_predicate %{{.*}}(%[[DOT0:.*]]) {
    %predicated = ttg.warp_predicate %predicate (%blocked) {
      // CHECK: %[[SCALED:.*]] = arith.addf %[[DOT0]], %[[DOT0]] : tensor<32x32xf32, #[[$WP_MMA]]>
      %scaled = arith.addf %blocked, %blocked : tensor<32x32xf32, #wp_blocked>
      // CHECK: ttg.predicate_yield %[[SCALED]] : tensor<32x32xf32, #[[$WP_MMA]]>
      ttg.predicate_yield %scaled : tensor<32x32xf32, #wp_blocked>
    } : (tensor<32xi1, #wp_row>, tensor<32x32xf32, #wp_blocked>) -> tensor<32x32xf32, #wp_blocked>
    %mma = ttg.convert_layout %predicated : tensor<32x32xf32, #wp_blocked> -> tensor<32x32xf32, #wp_mma>
    // CHECK: %[[DOT1:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[PREDICATED]]
    %dot1 = tt.dot %lhs, %rhs, %mma : tensor<32x32xf16, #wp_dot0> * tensor<32x32xf16, #wp_dot1> -> tensor<32x32xf32, #wp_mma>
    tt.return %dot1 : tensor<32x32xf32, #wp_mma>
  }
}

// -----

// A hard register-layout pin may be nested under a deferred-verification
// wrapper.  It must win conflict resolution even when the producer's linear
// layout has a higher vectorization score.  Otherwise the FP16 reduction is
// silently retagged and changes from the MFMA-local reduction tree to a much
// longer per-thread chain.

#pin_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [8, 1], instrShape = [32, 32, 16], isTransposed = true}>
#pin_user = #tlx.user_layout<#pin_mma>
#pin = #tlx.no_verify_layout<#pin_user>
#pin_row = #ttg.slice<{dim = 1, parent = #pin}>
#pin_linear = #ttg.linear<{register = [[0, 32], [0, 16], [0, 8], [0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0], [128, 0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @nested_pin_keeps_mfma_reduction
  tt.func @nested_pin_keeps_mfma_reduction(
      %values: tensor<256x64xf16, #pin_linear>)
      -> tensor<256xf16, #pin_row> {
    // CHECK: %[[PINNED:.*]] = ttg.convert_layout %{{.*}} : tensor<256x64xf16, #{{.*}}> -> tensor<256x64xf16, #tlx.no_verify_layout<{{.*}}>>
    %pinned = ttg.convert_layout %values : tensor<256x64xf16, #pin_linear> -> tensor<256x64xf16, #pin>
    // CHECK: %[[SUM:.*]] = "tt.reduce"(%[[PINNED]])
    // CHECK: }) : (tensor<256x64xf16, #tlx.no_verify_layout<{{.*}}>>) -> tensor<256xf16, #ttg.slice<{{.*}}>>
    %sum = "tt.reduce"(%pinned) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
    ^bb0(%lhs: f16, %rhs: f16):
      %next = arith.addf %lhs, %rhs : f16
      tt.reduce.return %next : f16
    }) : (tensor<256x64xf16, #pin>) -> tensor<256xf16, #pin_row>
    tt.return %sum : tensor<256xf16, #pin_row>
  }
}
