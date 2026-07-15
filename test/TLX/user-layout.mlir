// RUN: triton-opt -split-input-file --tlx-propagate-layout %s | FileCheck %s

// A user-pinned layout (#tlx.user_layout<...>) is honored by layout propagation:
// the value is never retagged, and the wrapper is unwrapped back to the concrete
// layout the user asked for. The wrapper is general (wraps distributed or shared
// layouts); here it wraps a shared layout on a local_alloc.

#swizzled = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#user = #tlx.user_layout<#swizzled>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-DAG: #[[$SWIZZLED:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0]}>
// CHECK-NOT: #tlx.user_layout
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @user_pinned
  tt.func @user_pinned(%arg0: tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #blocked> {
    // The alloc keeps the user's swizzled layout (unwrapped from the marker),
    // not some consumer-inferred encoding.
    // CHECK: ttg.local_alloc {{.*}} -> !ttg.memdesc<128x64xf16, #[[$SWIZZLED]], #smem, mutable>
    %0 = ttg.local_alloc %arg0 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #user, #smem, mutable>
    %1 = ttg.local_load %0 : !ttg.memdesc<128x64xf16, #user, #smem, mutable> -> tensor<128x64xf16, #blocked>
    tt.return %1 : tensor<128x64xf16, #blocked>
  }
}

// -----

// A user-pinned buffer feeding a dot must NOT be folded away. Normally a
// local_alloc(src) -> local_load(dot_operand) pair is an LDS round-trip that
// tlx-propagate-layout eliminates (see the next case); the wrapper anchors it, so
// the buffer and its swizzle survive.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#swizzled = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#user = #tlx.user_layout<#swizzled>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#smem = #ttg.shared_memory

// CHECK-DAG: #[[$SWZ:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @pinned_dot_not_folded
  tt.func @pinned_dot_not_folded(%src: tensor<64x32xf16, #blocked>) -> tensor<64x32xf16, #dot0> {
    // CHECK: ttg.local_alloc {{.*}} -> !ttg.memdesc<64x32xf16, #[[$SWZ]], #smem, mutable>
    // CHECK: ttg.local_load
    // CHECK-NOT: #tlx.user_layout
    %buf = ttg.local_alloc %src : (tensor<64x32xf16, #blocked>) -> !ttg.memdesc<64x32xf16, #user, #smem, mutable>
    %a = ttg.local_load %buf : !ttg.memdesc<64x32xf16, #user, #smem, mutable> -> tensor<64x32xf16, #dot0>
    tt.return %a : tensor<64x32xf16, #dot0>
  }
}

// -----

// Baseline (no wrapper): the same local_alloc(src) -> local_load(dot_operand) is
// folded to a convert_layout and the LDS buffer is eliminated. This is exactly
// what the wrapper above prevents.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#swizzled = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @unpinned_dot_folded
  tt.func @unpinned_dot_folded(%src: tensor<64x32xf16, #blocked>) -> tensor<64x32xf16, #dot0> {
    // CHECK: ttg.convert_layout
    // CHECK-NOT: ttg.local_alloc
    %buf = ttg.local_alloc %src : (tensor<64x32xf16, #blocked>) -> !ttg.memdesc<64x32xf16, #swizzled, #smem, mutable>
    %a = ttg.local_load %buf : !ttg.memdesc<64x32xf16, #swizzled, #smem, mutable> -> tensor<64x32xf16, #dot0>
    tt.return %a : tensor<64x32xf16, #dot0>
  }
}
