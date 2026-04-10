// RUN: triton-opt -split-input-file --tlx-propagate-layout %s | FileCheck %s

// Test that tlx-propagate-layout can propagate a swizzled_shared constraint
// backward through ttg.memdesc_trans when the swizzle is effectively
// unswizzled. This exercises the guarded SwizzledSharedEncodingAttr path in
// LayoutBackwardPropagation.

#shared_src = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_trans = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared_req = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
// CHECK: #shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @propagate_swizzled_shared_through_memdesc_trans
  tt.func public @propagate_swizzled_shared_through_memdesc_trans() -> tensor<128x64xf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared_src, #smem, mutable>
    // CHECK: %[[SLICE:.*]] = ttg.memdesc_index %{{.*}}[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %slice = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared_src, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared_src, #smem, mutable>
    // CHECK: %[[TRANS:.*]] = ttg.memdesc_trans %[[SLICE]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    %trans = ttg.memdesc_trans %slice {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared_src, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared_trans, #smem, mutable>
    // CHECK-NOT: tlx.require_layout
    %req = tlx.require_layout %trans : !ttg.memdesc<128x64xf16, #shared_trans, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared_req, #smem, mutable>
    // CHECK: ttg.local_load %[[TRANS]] : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #blocked>
    %val = ttg.local_load %req : !ttg.memdesc<128x64xf16, #shared_req, #smem, mutable> -> tensor<128x64xf16, #blocked>
    tt.return %val : tensor<128x64xf16, #blocked>
  }
}
