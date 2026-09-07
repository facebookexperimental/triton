// RUN: split-file %s %t
// RUN: triton-opt -split-input-file --tlx-insert-require-layout %t/valid.mlir | FileCheck %s
// RUN: triton-opt -split-input-file --tlx-insert-require-layout --tlx-propagate-layout --canonicalize --tritonamdgpu-optimize-descriptor-encoding %t/valid.mlir | FileCheck %s --check-prefix=PROP --implicit-check-not=tlx.require_layout --implicit-check-not=tlx.user_layout
// RUN: not triton-opt --tlx-insert-require-layout %t/invalid.mlir 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: not triton-opt --tlx-insert-require-layout %t/invalid-fused.mlir 2>&1 | FileCheck %s --check-prefix=FUSED-ERROR
//
// Tests for the AMD TDM extension of TLXInsertRequireLayout.
//
// The pass anchors a `tlx.require_layout` on every TDM op's buffer operand
// so `tlx-propagate-layout` can rewrite the source `local_alloc` to a
// descriptor-compatible padded encoding from `buildDefaultTDMDescriptorEncoding`.

//--- valid.mlir

// Fused TDM loads constrain every destination independently.
// CHECK-DAG: #[[$FUSED_A:.*]] = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [64, 32]}>
// CHECK-DAG: #[[$FUSED_B:.*]] = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [32, 64]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @fused_tdm_constraints
  tt.func public @fused_tdm_constraints(
      %a: !tt.tensordesc<64x32xf16>, %b: !tt.tensordesc<32x64xf16>) {
    %da = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    %db = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    // CHECK: %[[RA:.*]] = tlx.require_layout %{{.*}} : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #[[$FUSED_A]], #smem, mutable>
    // CHECK: %[[RB:.*]] = tlx.require_layout %{{.*}} : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x64xf16, #[[$FUSED_B]], #smem, mutable>
    // CHECK: amdg.async_tdm_fused_copy_global_to_local %{{.*}}, %{{.*}} into %[[RA]], %[[RB]]
    %token = amdg.async_tdm_fused_copy_global_to_local %a, %b into %da, %db {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x32xf16>, !tt.tensordesc<32x64xf16> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>, !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Fused TDM loads preserve each member's explicit padded layout independently.
// CHECK-DAG: #[[$FUSED_EXPLICIT_A:.*]] = #ttg.padded_shared<[256:+16] {order = [1, 0], shape = [128, 128]}>
// CHECK-DAG: #[[$FUSED_EXPLICIT_B:.*]] = #ttg.padded_shared<[512:+32] {order = [1, 0], shape = [128, 128]}>
#padded_a = #ttg.padded_shared<[256:+16] {order = [1, 0], shape = [128, 128]}>
#padded_b = #ttg.padded_shared<[512:+32] {order = [1, 0], shape = [128, 128]}>
#pinned_a = #tlx.user_layout<#padded_a>
#pinned_b = #tlx.user_layout<#padded_b>
#smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @fused_tdm_preserves_pinned_padding
  tt.func public @fused_tdm_preserves_pinned_padding(
      %a: !tt.tensordesc<128x128xf16>, %b: !tt.tensordesc<128x128xf16>) {
    %da = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #pinned_a, #smem, mutable>
    %db = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #pinned_b, #smem, mutable>
    // CHECK: %[[RA:.*]] = tlx.require_layout %{{.*}} : !ttg.memdesc<128x128xf16, #{{.*}}, #smem, mutable> -> !ttg.memdesc<128x128xf16, #[[$FUSED_EXPLICIT_A]], #smem, mutable>
    // CHECK: %[[RB:.*]] = tlx.require_layout %{{.*}} : !ttg.memdesc<128x128xf16, #{{.*}}, #smem, mutable> -> !ttg.memdesc<128x128xf16, #[[$FUSED_EXPLICIT_B]], #smem, mutable>
    // CHECK: amdg.async_tdm_fused_copy_global_to_local %{{.*}}, %{{.*}} into %[[RA]], %[[RB]]
    %token = amdg.async_tdm_fused_copy_global_to_local %a, %b into %da, %db {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<128x128xf16>, !tt.tensordesc<128x128xf16> -> !ttg.memdesc<128x128xf16, #pinned_a, #smem, mutable>, !ttg.memdesc<128x128xf16, #pinned_b, #smem, mutable>
    tt.return
  }
}

// -----

// =============================================================================
// 1. TDM copy with no consumer. Default fallback fires.
// For block_shape [128, 32] fp16: pad_interval=32, pad_amount=128/16=8.
// =============================================================================

// CHECK-DAG: #[[$PADDED32:.*]] = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_no_consumer
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #[[$PADDED32]], #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  tt.func public @tdm_no_consumer(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %k] pred = %p : !tt.tensordesc<128x32xf16>
    %tok = amdg.async_tdm_copy_global_to_local %positioned into %buf : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// Explicit pinned padding is preserved instead of being replaced by the
// descriptor-default TDM layout.
// =============================================================================

#padded_explicit = #ttg.padded_shared<[256:+16] {order = [1, 0], shape = [128, 128]}>
#pinned_explicit = #tlx.user_layout<#padded_explicit>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$EXPLICIT:.*]] = #ttg.padded_shared<[256:+16] {order = [1, 0], shape = [128, 128]}>
  // CHECK-LABEL: @tdm_preserves_pinned_padding
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x128xf16, #[[$EXPLICIT]], #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  tt.func public @tdm_preserves_pinned_padding(%desc: !tt.tensordesc<128x128xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #pinned_explicit, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x128xf16, #pinned_explicit, #smem, mutable> -> !ttg.memdesc<128x128xf16, #pinned_explicit, #smem, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %k] pred = %p : !tt.tensordesc<128x128xf16>
    %tok = amdg.async_tdm_copy_global_to_local %positioned into %buf : !tt.tensordesc<128x128xf16> -> !ttg.memdesc<128x128xf16, #pinned_explicit, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 2. TDM copy + plain local_load (no dot consumer). Default fallback fires.
// =============================================================================

// CHECK-DAG: #[[$PADDED32:.*]] = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_local_load_no_dot
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #[[$PADDED32]], #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  tt.func public @tdm_local_load_no_dot(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %k] pred = %p : !tt.tensordesc<128x32xf16>
    %tok = amdg.async_tdm_copy_global_to_local %positioned into %buf : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %val = ttg.local_load %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #blocked>
    tt.return
  }
}

// -----
// =============================================================================
// 3. Idempotency: TDM op already wrapped in require_layout is left untouched.
// =============================================================================

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#padded = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_already_wrapped
  // CHECK-COUNT-1: tlx.require_layout
  // CHECK-NOT: tlx.require_layout
  tt.func public @tdm_already_wrapped(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #padded, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #padded, #smem, mutable> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable>
    %req = tlx.require_layout %buf : !ttg.memdesc<128x32xf16, #padded, #smem, mutable> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %k] pred = %p : !tt.tensordesc<128x32xf16>
    %tok = amdg.async_tdm_copy_global_to_local %positioned into %req : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 4. Dot-path skip on TDM-fed buffers: the dot-path walk is suppressed.
// =============================================================================

#dot0 = #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_dot_path_skip
  // Only the TDM anchor should fire; the dot-path anchor should NOT
  // because isFedByTDM returns true.
  // CHECK: tlx.require_layout
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  // CHECK-NOT: tlx.require_layout {{.*}} -> !ttg.memdesc<{{.*}}, #ttg.swizzled_shared
  tt.func public @tdm_dot_path_skip(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %k] pred = %p : !tt.tensordesc<128x32xf16>
    %tok = amdg.async_tdm_copy_global_to_local %positioned into %buf : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %val = ttg.local_load %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #dot0>
    tt.return
  }
}

// -----
// =============================================================================
// 5. bf16 default fallback: pad_amount = 128 / 16 = 8.
// =============================================================================

// CHECK-DAG: #[[$PADDED32BF16:.*]] = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_bf16_default
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xbf16, #[[$PADDED32BF16]], #smem, mutable>
  tt.func public @tdm_bf16_default(%desc: !tt.tensordesc<128x32xbf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xbf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %k] pred = %p : !tt.tensordesc<128x32xbf16>
    %tok = amdg.async_tdm_copy_global_to_local %positioned into %buf : !tt.tensordesc<128x32xbf16> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 6. fp32 default fallback: pad_amount = 128 / 32 = 4.
// =============================================================================

// CHECK-DAG: #[[$PADDED32FP32:.*]] = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [128, 32]}>

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_fp32_default
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf32, #[[$PADDED32FP32]], #smem, mutable>
  tt.func public @tdm_fp32_default(%desc: !tt.tensordesc<128x32xf32>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %k] pred = %p : !tt.tensordesc<128x32xf32>
    %tok = amdg.async_tdm_copy_global_to_local %positioned into %buf : !tt.tensordesc<128x32xf32> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 7. The descriptor update carrying the predicate is preserved when the TDM
// op is rewrapped.
// =============================================================================

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_pred_preserved
  // CHECK: %[[POSITIONED:.*]] = amdg.update_tensor_descriptor %arg0 add_offsets = [%arg1, %arg2] pred = %arg3
  // CHECK: amdg.async_tdm_copy_global_to_local %[[POSITIONED]] into
  tt.func public @tdm_pred_preserved(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %k] pred = %p : !tt.tensordesc<128x32xf16>
    %tok = amdg.async_tdm_copy_global_to_local %positioned into %buf : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 8. TDM store: default encoding is anchored on the source memdesc.
// =============================================================================

// A pre-encoded store descriptor keeps its existing shared layout.
// CHECK-DAG: #[[$SWIZZLED_STORE:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_store_anchor
  // CHECK: tlx.require_layout
  // CHECK-NEXT: amdg.async_tdm_copy_local_to_global
  tt.func public @tdm_store_anchor(%desc: !tt.tensordesc<128x128xf16, #shared>, %m: i32, %n: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %n] : !tt.tensordesc<128x128xf16, #shared>
    amdg.async_tdm_copy_local_to_global %positioned from %buf : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<128x128xf16, #shared>
    tt.return
  }
}

// -----
// =============================================================================
// A store descriptor without a shared layout receives the padded TDM default.
// =============================================================================

// CHECK-DAG: #[[$PADDED_STORE:.*]] = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 128]}>

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_store_assigns_padded_default
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x128xf16, #[[$PADDED_STORE]], #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_local_to_global
  tt.func public @tdm_store_assigns_padded_default(%desc: !tt.tensordesc<128x128xf16>, %m: i32, %n: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %n] : !tt.tensordesc<128x128xf16>
    amdg.async_tdm_copy_local_to_global %positioned from %buf : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<128x128xf16>
    tt.return
  }
}

// -----
// =============================================================================
// 9. TDM store + dot reader: store anchor always uses default encoding
//    (allowDotAware=false), not the WMMA-tuned form.
// =============================================================================

#dot0 = #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // The store anchor should use the default encoding regardless of the
  // dot consumer.
  // CHECK-LABEL: @tdm_store_with_dot_reader
  // CHECK: tlx.require_layout
  // CHECK-NEXT: amdg.async_tdm_copy_local_to_global
  tt.func public @tdm_store_with_dot_reader(%desc: !tt.tensordesc<128x32xf16, #shared>, %m: i32, %k: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %val = ttg.local_load %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #dot0>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %k] : !tt.tensordesc<128x32xf16, #shared>
    amdg.async_tdm_copy_local_to_global %positioned from %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> !tt.tensordesc<128x32xf16, #shared>
    tt.return
  }
}

// -----
// =============================================================================
// 21. Dot consumer reached through memdesc_subslice.
// This is the single-warp-per-SIMD GEMM shape: TDM loads the full
// BLOCK_K=128 tile, while local_load consumes 32-wide LDS subtiles.
// The TDM anchor must still discover the dot consumer through
// memdesc_subslice and choose the WMMA-tuned full-tile encoding `[128:+8]`.
// The dot-path walk should also recognize that the subslice is TDM-fed and
// avoid inserting a sibling swizzled anchor on the local_load operand.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [32, 128]}>

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_dot_consumer_through_subslice
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<32x128xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  // CHECK-NOT: tlx.require_layout
  tt.func public @tdm_dot_consumer_through_subslice(%desc: !tt.tensordesc<32x128xf16>, %m: i32, %k: i32, %p: i32)
      -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %k] pred = %p : !tt.tensordesc<32x128xf16>
    %tok = amdg.async_tdm_copy_global_to_local %positioned into %buf : !tt.tensordesc<32x128xf16> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %sub = ttg.memdesc_subslice %buf[0, 0] : !ttg.memdesc<32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable, 32x128>
    %t = ttg.local_load %sub : !ttg.memdesc<32x32xf16, #shared, #smem, mutable, 32x128> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.return %t : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
  }
}

// -----
// =============================================================================
// 22. Dot consumer reached through memdesc_subslice + memdesc_trans.
// This is the transposed-B single-warp-per-SIMD GEMM shape: TDM loads a
// full 32x128 tile, slices a 32-wide K subtile, transposes the memdesc view,
// and only then performs the dot-operand local_load.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[128:+16] {order = [1, 0], shape = [32, 128]}>

#mma_t = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared_t = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_t_trans = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem_t = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_dot_consumer_through_subslice_trans
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<32x128xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  // CHECK-NOT: tlx.require_layout
  tt.func public @tdm_dot_consumer_through_subslice_trans(%desc: !tt.tensordesc<32x128xf16>, %m: i32, %k: i32, %p: i32)
      -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_t, kWidth = 8}>> {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared_t, #smem_t, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x32x128xf16, #shared_t, #smem_t, mutable> -> !ttg.memdesc<32x128xf16, #shared_t, #smem_t, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %k] pred = %p : !tt.tensordesc<32x128xf16>
    %tok = amdg.async_tdm_copy_global_to_local %positioned into %buf : !tt.tensordesc<32x128xf16> -> !ttg.memdesc<32x128xf16, #shared_t, #smem_t, mutable>
    %sub = ttg.memdesc_subslice %buf[0, 32] : !ttg.memdesc<32x128xf16, #shared_t, #smem_t, mutable> -> !ttg.memdesc<32x32xf16, #shared_t, #smem_t, mutable, 32x128>
    %trans = ttg.memdesc_trans %sub {order = array<i32: 1, 0>} : !ttg.memdesc<32x32xf16, #shared_t, #smem_t, mutable, 32x128> -> !ttg.memdesc<32x32xf16, #shared_t_trans, #smem_t, mutable, 128x32>
    %t = ttg.local_load %trans : !ttg.memdesc<32x32xf16, #shared_t_trans, #smem_t, mutable, 128x32> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_t, kWidth = 8}>>
    tt.return %t : tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_t, kWidth = 8}>>
  }
}

// -----
// =============================================================================
// 23. Dot consumer reached through memdesc_reshape.
// A TDM load writes the full [128, 32] tile, while the dot operand consumes a
// reshaped [32, 128] view. The TDM anchor must still discover the dot consumer
// through memdesc_reshape and choose the WMMA-tuned source-tile encoding.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 32]}>

#mma_r = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared_r = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_r = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_dot_consumer_through_reshape
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  // CHECK-NOT: tlx.require_layout
  tt.func public @tdm_dot_consumer_through_reshape(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32)
      -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_r, kWidth = 8}>> {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared_r, #smem_r, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared_r, #smem_r, mutable> -> !ttg.memdesc<128x32xf16, #shared_r, #smem_r, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%m, %k] pred = %p : !tt.tensordesc<128x32xf16>
    %tok = amdg.async_tdm_copy_global_to_local %positioned into %buf : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared_r, #smem_r, mutable>
    %reshape = ttg.memdesc_reshape %buf : !ttg.memdesc<128x32xf16, #shared_r, #smem_r, mutable> -> !ttg.memdesc<32x128xf16, #shared_r, #smem_r, mutable>
    %t = ttg.local_load %reshape : !ttg.memdesc<32x128xf16, #shared_r, #smem_r, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_r, kWidth = 8}>>
    tt.return %t : tensor<32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_r, kWidth = 8}>>
  }
}

// -----

// Preserve the operand view's padded encoding, not the allocation's shape or
// order. The transposed view is row-major and descriptor-compatible even
// though its pinned allocation is column-major.
// CHECK-DAG: #[[$TRANS_VIEW:.*]] = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 64]}>
// PROP-DAG: #[[$TRANS_BASE:.*]] = #ttg.padded_shared<[128:+8] {order = [0, 1], shape = [64, 128]}>
// PROP-DAG: #[[$TRANS_VIEW:.*]] = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 64]}>
#base = #ttg.padded_shared<[128:+8] {order = [0, 1], shape = [64, 128]}>
#view = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 64]}>
#pinned_base = #tlx.user_layout<#base>
#smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_preserves_pinned_transpose
  // PROP-LABEL: @tdm_preserves_pinned_transpose
  // PROP-SAME: %[[DESC:.*]]: !tt.tensordesc<128x64xf16, #[[$TRANS_VIEW]]>
  tt.func public @tdm_preserves_pinned_transpose(%desc: !tt.tensordesc<128x64xf16>) {
    // PROP: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #[[$TRANS_BASE]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #pinned_base, #smem, mutable>
    // PROP: %[[VIEW:.*]] = ttg.memdesc_trans %[[ALLOC]] {{.*}} -> !ttg.memdesc<128x64xf16, #[[$TRANS_VIEW]], #smem, mutable>
    %trans = ttg.memdesc_trans %alloc {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #pinned_base, #smem, mutable> -> !ttg.memdesc<128x64xf16, #view, #smem, mutable>
    // CHECK: %[[REQ:.*]] = tlx.require_layout {{.*}} -> !ttg.memdesc<128x64xf16, #[[$TRANS_VIEW]], #smem, mutable>
    // CHECK-NEXT: amdg.async_tdm_copy_global_to_local %{{.*}} into %[[REQ]]
    // PROP: amdg.async_tdm_copy_global_to_local %[[DESC]] into %[[VIEW]]
    %tok = amdg.async_tdm_copy_global_to_local %desc into %trans : !tt.tensordesc<128x64xf16> -> !ttg.memdesc<128x64xf16, #view, #smem, mutable>
    tt.return
  }
}

// -----

// Stores through reshaped views must also preserve the view's shape. Padding
// remains every 64 elements, which is legal for the store's inner dimension.
// CHECK-DAG: #[[$RESHAPE_VIEW:.*]] = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [128, 64]}>
// PROP-DAG: #[[$RESHAPE_BASE:.*]] = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [64, 128]}>
// PROP-DAG: #[[$RESHAPE_VIEW:.*]] = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [128, 64]}>
#base = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [64, 128]}>
#view = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [128, 64]}>
#pinned_base = #tlx.user_layout<#base>
#smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_store_preserves_pinned_reshape
  // PROP-LABEL: @tdm_store_preserves_pinned_reshape
  // PROP-SAME: %[[DESC:.*]]: !tt.tensordesc<128x64xf16, #[[$RESHAPE_VIEW]]>
  tt.func public @tdm_store_preserves_pinned_reshape(%desc: !tt.tensordesc<128x64xf16>) {
    // PROP: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #[[$RESHAPE_BASE]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #pinned_base, #smem, mutable>
    // Match materializeConcreteMemDesc in the TLX frontend: keep the root pin
    // but expose the concrete encoding to reshape inference.
    %concrete = tlx.require_layout %alloc : !ttg.memdesc<64x128xf16, #pinned_base, #smem, mutable> -> !ttg.memdesc<64x128xf16, #base, #smem, mutable>
    // PROP: %[[VIEW:.*]] = ttg.memdesc_reshape %[[ALLOC]] {{.*}} -> !ttg.memdesc<128x64xf16, #[[$RESHAPE_VIEW]], #smem, mutable>
    %reshape = ttg.memdesc_reshape %concrete : !ttg.memdesc<64x128xf16, #base, #smem, mutable> -> !ttg.memdesc<128x64xf16, #view, #smem, mutable>
    // CHECK: %[[REQ:.*]] = tlx.require_layout {{.*}} -> !ttg.memdesc<128x64xf16, #[[$RESHAPE_VIEW]], #smem, mutable>
    // CHECK-NEXT: amdg.async_tdm_copy_local_to_global %{{.*}} from %[[REQ]]
    // PROP: amdg.async_tdm_copy_local_to_global %[[DESC]] from %[[VIEW]]
    amdg.async_tdm_copy_local_to_global %desc from %reshape : !ttg.memdesc<128x64xf16, #view, #smem, mutable> -> !tt.tensordesc<128x64xf16>
    tt.return
  }
}

// -----

// The fused path preserves each view independently, also for unpinned padded
// allocations. Both views differ in shape from their respective roots.
// CHECK-DAG: #[[$FUSED_TRANS:.*]] = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 64]}>
// CHECK-DAG: #[[$FUSED_RESHAPE:.*]] = #ttg.padded_shared<[256:+16] {order = [1, 0], shape = [128, 64]}>
// PROP-DAG: #[[$FUSED_TRANS:.*]] = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 64]}>
// PROP-DAG: #[[$FUSED_RESHAPE:.*]] = #ttg.padded_shared<[256:+16] {order = [1, 0], shape = [128, 64]}>
#base_a = #ttg.padded_shared<[128:+8] {order = [0, 1], shape = [64, 128]}>
#view_a = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 64]}>
#base_b = #ttg.padded_shared<[256:+16] {order = [1, 0], shape = [64, 128]}>
#view_b = #ttg.padded_shared<[256:+16] {order = [1, 0], shape = [128, 64]}>
#smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @fused_tdm_preserves_padded_views
  // PROP-LABEL: @fused_tdm_preserves_padded_views
  // PROP-SAME: %[[A:.*]]: !tt.tensordesc<128x64xf16, #[[$FUSED_TRANS]]>, %[[B:.*]]: !tt.tensordesc<128x64xf16, #[[$FUSED_RESHAPE]]>
  tt.func public @fused_tdm_preserves_padded_views(%a: !tt.tensordesc<128x64xf16>, %b: !tt.tensordesc<128x64xf16>) {
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #base_a, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #base_b, #smem, mutable>
    // PROP: %[[TRANS:.*]] = ttg.memdesc_trans {{.*}} -> !ttg.memdesc<128x64xf16, #[[$FUSED_TRANS]], #smem, mutable>
    %trans = ttg.memdesc_trans %alloc_a {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #base_a, #smem, mutable> -> !ttg.memdesc<128x64xf16, #view_a, #smem, mutable>
    // PROP: %[[RESHAPE:.*]] = ttg.memdesc_reshape {{.*}} -> !ttg.memdesc<128x64xf16, #[[$FUSED_RESHAPE]], #smem, mutable>
    %reshape = ttg.memdesc_reshape %alloc_b : !ttg.memdesc<64x128xf16, #base_b, #smem, mutable> -> !ttg.memdesc<128x64xf16, #view_b, #smem, mutable>
    // CHECK: %[[RA:.*]] = tlx.require_layout {{.*}} -> !ttg.memdesc<128x64xf16, #[[$FUSED_TRANS]], #smem, mutable>
    // CHECK: %[[RB:.*]] = tlx.require_layout {{.*}} -> !ttg.memdesc<128x64xf16, #[[$FUSED_RESHAPE]], #smem, mutable>
    // CHECK-NEXT: amdg.async_tdm_fused_copy_global_to_local %{{.*}}, %{{.*}} into %[[RA]], %[[RB]]
    // PROP: amdg.async_tdm_fused_copy_global_to_local %[[A]], %[[B]] into %[[TRANS]], %[[RESHAPE]]
    %tok = amdg.async_tdm_fused_copy_global_to_local %a, %b into %trans, %reshape {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<128x64xf16>, !tt.tensordesc<128x64xf16> -> !ttg.memdesc<128x64xf16, #view_a, #smem, mutable>, !ttg.memdesc<128x64xf16, #view_b, #smem, mutable>
    tt.return
  }
}

// -----

// An explicit maxPhase=1 layout is unpadded, not swizzled. Preserve it for
// loads, stores, and individual fused members, even with unencoded descriptors.
// CHECK-DAG: #[[$UNPADDED:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// CHECK-DAG: #[[$PADDED:.*]] = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [32, 32]}>
// PROP-DAG: #[[$UNPADDED:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// PROP-DAG: #[[$PADDED:.*]] = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [32, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#padded = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [32, 32]}>
#pinned_shared = #tlx.user_layout<#shared>
#pinned_padded = #tlx.user_layout<#padded>
#smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_preserves_pinned_unpadded
  // PROP-LABEL: @tdm_preserves_pinned_unpadded
  // PROP-SAME: %[[DESC:.*]]: !tt.tensordesc<32x32xf16, #[[$UNPADDED]]>, %[[OTHER:.*]]: !tt.tensordesc<32x32xf16, #[[$PADDED]]>
  tt.func public @tdm_preserves_pinned_unpadded(%desc: !tt.tensordesc<32x32xf16>, %other: !tt.tensordesc<32x32xf16>) {
    // PROP: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #[[$UNPADDED]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #pinned_shared, #smem, mutable>
    // PROP: %[[PADDED_ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #[[$PADDED]], #smem, mutable>
    %padded_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #pinned_padded, #smem, mutable>
    // CHECK: %[[LOAD:.*]] = tlx.require_layout {{.*}} -> !ttg.memdesc<32x32xf16, #[[$UNPADDED]], #smem, mutable>
    // CHECK-NEXT: amdg.async_tdm_copy_global_to_local %{{.*}} into %[[LOAD]]
    // PROP: amdg.async_tdm_copy_global_to_local %[[DESC]] into %[[ALLOC]]
    %tok = amdg.async_tdm_copy_global_to_local %desc into %alloc : !tt.tensordesc<32x32xf16> -> !ttg.memdesc<32x32xf16, #pinned_shared, #smem, mutable>
    // CHECK: %[[STORE:.*]] = tlx.require_layout {{.*}} -> !ttg.memdesc<32x32xf16, #[[$UNPADDED]], #smem, mutable>
    // CHECK-NEXT: amdg.async_tdm_copy_local_to_global %{{.*}} from %[[STORE]]
    // PROP: amdg.async_tdm_copy_local_to_global %[[DESC]] from %[[ALLOC]]
    amdg.async_tdm_copy_local_to_global %desc from %alloc : !ttg.memdesc<32x32xf16, #pinned_shared, #smem, mutable> -> !tt.tensordesc<32x32xf16>
    // CHECK: %[[RA:.*]] = tlx.require_layout {{.*}} -> !ttg.memdesc<32x32xf16, #[[$UNPADDED]], #smem, mutable>
    // CHECK: %[[RB:.*]] = tlx.require_layout {{.*}} -> !ttg.memdesc<32x32xf16, #[[$PADDED]], #smem, mutable>
    // CHECK-NEXT: amdg.async_tdm_fused_copy_global_to_local %{{.*}}, %{{.*}} into %[[RA]], %[[RB]]
    // PROP: amdg.async_tdm_fused_copy_global_to_local %[[DESC]], %[[OTHER]] into %[[ALLOC]], %[[PADDED_ALLOC]]
    %fused = amdg.async_tdm_fused_copy_global_to_local %desc, %other into %alloc, %padded_alloc {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<32x32xf16>, !tt.tensordesc<32x32xf16> -> !ttg.memdesc<32x32xf16, #pinned_shared, #smem, mutable>, !ttg.memdesc<32x32xf16, #pinned_padded, #smem, mutable>
    tt.return
  }
}

// -----

// A wide tile uses the unpadded descriptor fallback (1024 f16 elements exceed
// the 512-element padding interval limit). Explicitly selecting it is valid.
// CHECK-DAG: #[[$WIDE:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// PROP-DAG: #[[$WIDE:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#pinned = #tlx.user_layout<#shared>
#smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_store_preserves_pinned_unpadded_wide
  // PROP-LABEL: @tdm_store_preserves_pinned_unpadded_wide
  // PROP-SAME: %[[DESC:.*]]: !tt.tensordesc<4x1024xf16, #[[$WIDE]]>
  tt.func public @tdm_store_preserves_pinned_unpadded_wide(%desc: !tt.tensordesc<4x1024xf16, #shared>) {
    %c0 = arith.constant 0 : i32
    // PROP: ttg.local_alloc : () -> !ttg.memdesc<2x4x1024xf16, #[[$WIDE]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x4x1024xf16, #pinned, #smem, mutable>
    // PROP: %[[BUF:.*]] = ttg.memdesc_index {{.*}} -> !ttg.memdesc<4x1024xf16, #[[$WIDE]], #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x4x1024xf16, #pinned, #smem, mutable> -> !ttg.memdesc<4x1024xf16, #pinned, #smem, mutable>
    // CHECK: %[[REQ:.*]] = tlx.require_layout {{.*}} -> !ttg.memdesc<4x1024xf16, #[[$WIDE]], #smem, mutable>
    // CHECK-NEXT: amdg.async_tdm_copy_local_to_global %{{.*}} from %[[REQ]]
    // PROP: amdg.async_tdm_copy_local_to_global %[[DESC]] from %[[BUF]]
    amdg.async_tdm_copy_local_to_global %desc from %buf : !ttg.memdesc<4x1024xf16, #pinned, #smem, mutable> -> !tt.tensordesc<4x1024xf16, #shared>
    tt.return
  }
}

//--- invalid.mlir

// Genuine swizzling remains unsupported and is rejected by the TDM verifier.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>
#pinned = #tlx.user_layout<#shared>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // ERROR: TDM does not support swizzling
  tt.func public @tdm_rejects_explicit_swizzled(%desc: !tt.tensordesc<128x128xf16>) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #pinned, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x128xf16, #pinned, #smem, mutable> -> !ttg.memdesc<128x128xf16, #pinned, #smem, mutable>
    %positioned = amdg.update_tensor_descriptor %desc add_offsets = [%c0, %c0] : !tt.tensordesc<128x128xf16>
    %tok = amdg.async_tdm_copy_global_to_local %positioned into %buf : !tt.tensordesc<128x128xf16> -> !ttg.memdesc<128x128xf16, #pinned, #smem, mutable>
    tt.return
  }
}

//--- invalid-fused.mlir

#padded = #ttg.padded_shared<[256:+16] {order = [1, 0], shape = [128, 128]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>
#pinned_padded = #tlx.user_layout<#padded>
#pinned_shared = #tlx.user_layout<#shared>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // FUSED-ERROR: TDM does not support swizzling
  tt.func public @fused_tdm_rejects_invalid_explicit_member(
      %a: !tt.tensordesc<128x128xf16>, %b: !tt.tensordesc<128x128xf16>) {
    %da = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #pinned_padded, #smem, mutable>
    %db = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #pinned_shared, #smem, mutable>
    %token = amdg.async_tdm_fused_copy_global_to_local %a, %b into %da, %db {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<128x128xf16>, !tt.tensordesc<128x128xf16> -> !ttg.memdesc<128x128xf16, #pinned_padded, #smem, mutable>, !ttg.memdesc<128x128xf16, #pinned_shared, #smem, mutable>
    tt.return
  }
}
