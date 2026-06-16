// RUN: triton-opt -split-input-file --tlx-insert-require-layout %s | FileCheck %s
//
// Tests for the AMD TDM extension of TLXInsertRequireLayout.
//
// The pass anchors a `tlx.require_layout` on every TDM op's buffer operand
// so `tlx-propagate-layout` can rewrite the source `local_alloc` to a
// descriptor-compatible padded encoding from `buildDefaultTDMDescriptorEncoding`.

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
  tt.func public @tdm_no_consumer(%desc: !tt.tensordesc<tensor<128x32xf16, #shared>>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, %p : !tt.tensordesc<tensor<128x32xf16, #shared>> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
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
  tt.func public @tdm_local_load_no_dot(%desc: !tt.tensordesc<tensor<128x32xf16, #shared>>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, %p : !tt.tensordesc<tensor<128x32xf16, #shared>> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
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
  tt.func public @tdm_already_wrapped(%desc: !tt.tensordesc<tensor<128x32xf16, #shared>>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #padded, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #padded, #smem, mutable> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable>
    %req = tlx.require_layout %buf : !ttg.memdesc<128x32xf16, #padded, #smem, mutable> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %req, %p : !tt.tensordesc<tensor<128x32xf16, #shared>> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable>
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
  tt.func public @tdm_dot_path_skip(%desc: !tt.tensordesc<tensor<128x32xf16, #shared>>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, %p : !tt.tensordesc<tensor<128x32xf16, #shared>> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
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
  tt.func public @tdm_bf16_default(%desc: !tt.tensordesc<tensor<128x32xbf16, #shared>>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xbf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, %p : !tt.tensordesc<tensor<128x32xbf16, #shared>> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
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
  tt.func public @tdm_fp32_default(%desc: !tt.tensordesc<tensor<128x32xf32, #shared>>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, %p : !tt.tensordesc<tensor<128x32xf32, #shared>> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 7. Pred operand is preserved when the TDM op is rewrapped.
// =============================================================================

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_pred_preserved
  // CHECK: amdg.async_tdm_copy_global_to_local {{.*}}, %arg3
  tt.func public @tdm_pred_preserved(%desc: !tt.tensordesc<tensor<128x32xf16, #shared>>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, %p : !tt.tensordesc<tensor<128x32xf16, #shared>> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 8. TDM store: default encoding is anchored on the source memdesc.
// =============================================================================

// Store path uses swizzled (not padded) until alignTDMDescriptorEncodings is ported.
// CHECK-DAG: #[[SWIZZLED:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_store_anchor
  // CHECK: tlx.require_layout
  // CHECK-NEXT: amdg.async_tdm_copy_local_to_global
  tt.func public @tdm_store_anchor(%desc: !tt.tensordesc<tensor<128x128xf16, #shared>>, %m: i32, %n: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    amdg.async_tdm_copy_local_to_global %desc[%m, %n] from %buf : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<tensor<128x128xf16, #shared>>
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
  tt.func public @tdm_store_with_dot_reader(%desc: !tt.tensordesc<tensor<128x32xf16, #shared>>, %m: i32, %k: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %val = ttg.local_load %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #dot0>
    amdg.async_tdm_copy_local_to_global %desc[%m, %k] from %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> !tt.tensordesc<tensor<128x32xf16, #shared>>
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
  tt.func public @tdm_dot_consumer_through_subslice(%desc: !tt.tensordesc<tensor<32x128xf16>>, %m: i32, %k: i32, %p: i32)
      -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, %p : !tt.tensordesc<tensor<32x128xf16>> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
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

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [32, 128]}>

#mma_t = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared_t = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_t_trans = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem_t = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_dot_consumer_through_subslice_trans
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<32x128xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  // CHECK-NOT: tlx.require_layout
  tt.func public @tdm_dot_consumer_through_subslice_trans(%desc: !tt.tensordesc<tensor<32x128xf16>>, %m: i32, %k: i32, %p: i32)
      -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_t, kWidth = 8}>> {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared_t, #smem_t, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x32x128xf16, #shared_t, #smem_t, mutable> -> !ttg.memdesc<32x128xf16, #shared_t, #smem_t, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, %p : !tt.tensordesc<tensor<32x128xf16>> -> !ttg.memdesc<32x128xf16, #shared_t, #smem_t, mutable>
    %sub = ttg.memdesc_subslice %buf[0, 32] : !ttg.memdesc<32x128xf16, #shared_t, #smem_t, mutable> -> !ttg.memdesc<32x32xf16, #shared_t, #smem_t, mutable, 32x128>
    %trans = ttg.memdesc_trans %sub {order = array<i32: 1, 0>} : !ttg.memdesc<32x32xf16, #shared_t, #smem_t, mutable, 32x128> -> !ttg.memdesc<32x32xf16, #shared_t_trans, #smem_t, mutable, 128x32>
    %t = ttg.local_load %trans : !ttg.memdesc<32x32xf16, #shared_t_trans, #smem_t, mutable, 128x32> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_t, kWidth = 8}>>
    tt.return %t : tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_t, kWidth = 8}>>
  }
}
