// RUN: triton-opt -split-input-file --tlx-insert-require-layout %s | FileCheck %s
//
// Tests for the AMD TDM extension of TLXInsertRequireLayout.
//
// The pass anchors a `tlx.require_layout` on every
// `amdgpu.async_tdm_copy_global_to_local` op's buffer operand so
// `tlx-propagate-layout` can rewrite the source `local_alloc` to a
// descriptor-compatible padded encoding. When the buffer is consumed by
// a `local_load -> tt.dot` chain the WMMA-tuned padded layout is used
// (`composePaddedLayoutWMMA`); otherwise the descriptor-shape-only
// fallback is used (`buildDefaultTDMDescriptorEncoding`). The
// downstream AMD `OptimizeDescriptorEncoding` pass propagates the
// chosen encoding back to the descriptor's `TensorDescType` so the
// hardware lowering and the alloc agree.
// The dot-path walk in the same pass skips TDM-fed buffers so the two
// anchors don't conflict.

// =============================================================================
// 1. Smallest case: TDM copy with no consumer. Default fallback fires.
// For block_shape [128, 32] fp16: pad_interval=block_shape[order[0]]=32,
// pad_amount=128/16=8 -> `padded_shared<[32:+8], shape=[128, 32]>`.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_no_consumer
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  tt.func public @tdm_no_consumer(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 2. TDM copy + plain local_load (no dot consumer). Default fallback fires.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_plain_local_load
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  tt.func public @tdm_plain_local_load(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) -> tensor<128x32xf16, #blocked> {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %t = ttg.local_load %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #blocked>
    tt.return %t : tensor<128x32xf16, #blocked>
  }
}

// -----
// =============================================================================
// 3. TDM copy feeding tt.dot operand A (opIdx=0). The WMMA-tuned padded
// encoding from `composePaddedLayoutWMMA` is selected:
//   non-transposed (order[0]=1, 1-opIdx=1), padAmount=128/16=8, padInterval
//   = max(innerDim=32, bankWrapInterval=128) = 128 -> `[128:+8]`.
// `OptimizeDescriptorEncoding` propagates the same encoding back to the
// descriptor type so the hardware lowering and the alloc agree.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 32]}>

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_dot_operand_a
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  tt.func public @tdm_dot_operand_a(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32)
      -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %t = ttg.local_load %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.return %t : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
  }
}

// -----
// =============================================================================
// 4. TDM copy feeding tt.dot operand B (opIdx=1). WMMA-tuned encoding from
// `composePaddedLayoutWMMA`:
//   transposed (order[0]=1, 1-opIdx=0), padAmount = 2*instBitWidth/elemBits
//   = 2*128/16 = 16 (gfx1250 LDS-trans for fp16 has instBitWidth=128),
//   padInterval = max(innerDim=128, bankWrapInterval=128) = 128
//   -> `[128:+16]`.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[128:+16] {order = [1, 0], shape = [32, 128]}>

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_dot_operand_b
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<32x128xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  tt.func public @tdm_dot_operand_b(%desc: !tt.tensordesc<32x128xf16>, %k: i32, %n: i32, %p: i32)
      -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%k, %n] into %buf, pred = %p : !tt.tensordesc<32x128xf16> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %t = ttg.local_load %buf : !ttg.memdesc<32x128xf16, #shared, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    tt.return %t : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
  }
}

// -----
// =============================================================================
// 5. Conflicting dot consumers on the same TDM-fed buffer.
// `DotConsumerBackward` widens to `Conflict` (opIdx=0 and opIdx=1 disagree),
// so `findDotConsumer` returns nullopt and the anchor falls back to the
// descriptor-shape-only default `[32:+8]` instead of either WMMA-tuned variant.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>
// CHECK-NOT: #ttg.padded_shared<[128

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_conflicting_dot_consumers
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  tt.func public @tdm_conflicting_dot_consumers(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32)
      -> (tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
          tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %ta = ttg.local_load %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %tb = ttg.local_load %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    tt.return %ta, %tb : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
  }
}

// -----
// =============================================================================
// 6. Idempotency: TDM op already wrapped in tlx.require_layout is left alone.
// =============================================================================

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_p = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_already_wrapped
  // CHECK-COUNT-1: tlx.require_layout
  // CHECK-NOT: tlx.require_layout
  tt.func public @tdm_already_wrapped(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %req = tlx.require_layout %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared_p, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %req, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared_p, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 7. Multiple TDM copies on the same alloc each get their own anchor.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_multiple_copies
  // CHECK-COUNT-2: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NOT: tlx.require_layout
  tt.func public @tdm_multiple_copies(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf0 = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok0 = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf0, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %buf1 = ttg.memdesc_index %alloc[%c1] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok1 = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf1, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 8. Dot-path walk skips TDM-fed buffers.
// The TDM-fed buffer is also consumed by `local_load -> tt.dot`. Without
// the isFedByTDM check the dot-path walk would insert a swizzled-shared
// require_layout that conflicts with the TDM padded encoding. With the
// check, only the TDM padded anchor is inserted; no swizzled anchor.
// Encoding is the WMMA-tuned `[128:+8]` (opIdx=0, [128,32] fp16).
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 32]}>

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @dot_path_skips_tdm_buffer
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NOT: tlx.require_layout
  tt.func public @dot_path_skips_tdm_buffer(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32)
      -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %t = ttg.local_load %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.return %t : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
  }
}

// -----
// =============================================================================
// 9. Plain (non-TDM) dot-fed local_load still gets the swizzled-with-dot
//    anchor from the dot-path walk. Documents that the dot-path skip
//    applies only to TDM-fed buffers. Buffer is filled by local_store
//    (cp.async style) and consumed by a tt.DotOp; the dot-path backward
//    dataflow propagates the dot operand encoding back to the local_load.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.swizzled_shared<{vec = {{[1-9][0-9]*}}, perPhase = {{[1-9][0-9]*}}, maxPhase = {{[1-9][0-9]*}}, order = [1, 0]}>

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @non_tdm_dot_path_still_fires
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: ttg.local_load
  tt.func public @non_tdm_dot_path_still_fires(%a: tensor<64x32xf16, #blocked>, %b: tensor<32x64xf16, #blocked>) -> tensor<64x64xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared_b, #smem, mutable>
    %buf_a = ttg.memdesc_index %alloc_a[%c0] : !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    %buf_b = ttg.memdesc_index %alloc_b[%c0] : !ttg.memdesc<1x32x64xf16, #shared_b, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared_b, #smem, mutable>
    ttg.local_store %a, %buf_a : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    ttg.local_store %b, %buf_b : tensor<32x64xf16, #blocked> -> !ttg.memdesc<32x64xf16, #shared_b, #smem, mutable>
    %la = ttg.local_load %buf_a : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %lb = ttg.local_load %buf_b : !ttg.memdesc<32x64xf16, #shared_b, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %da = ttg.convert_layout %la : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %db = ttg.convert_layout %lb : tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %acc = ttg.convert_layout %cst : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #mma>
    %d = tt.dot %da, %db, %acc, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    %r = ttg.convert_layout %d : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
    tt.return %r : tensor<64x64xf32, #blocked>
  }
}

// -----
// =============================================================================
// 10. bf16: pad_amount = 128 / 16 = 8 (default fallback).
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_bf16_pad
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xbf16, #{{.*}}, #smem, mutable>
  tt.func public @tdm_bf16_pad(%desc: !tt.tensordesc<128x32xbf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xbf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<128x32xbf16> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 11. fp32: pad_amount = 128 / 32 = 4 (default fallback).
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [128, 32]}>

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_fp32_pad
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf32, #{{.*}}, #smem, mutable>
  tt.func public @tdm_fp32_pad(%desc: !tt.tensordesc<128x32xf32>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf32, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<128x32xf32> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 12. Wide block triggers swizzled fallback.
// max_pad_interval_elements = 256 * 32 / elem_width = 256 * 32 / 16 = 512 for fp16.
// A block_shape with innermost dim > 512 falls back to non-padded swizzled.
// =============================================================================

// CHECK-NOT: ttg.padded_shared

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_wide_falls_back_to_swizzled
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<8x1024xf16, #{{.*}}, #smem, mutable>
  tt.func public @tdm_wide_falls_back_to_swizzled(%desc: !tt.tensordesc<8x1024xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x8x1024xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x8x1024xf16, #shared, #smem, mutable> -> !ttg.memdesc<8x1024xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<8x1024xf16> -> !ttg.memdesc<8x1024xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 13. Subview chain through memdesc_reinterpret is followed.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_through_reinterpret
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  tt.func public @tdm_through_reinterpret(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable>
    %buf_3d = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_reinterpret %buf_3d : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 14. Pred operand is preserved and the wrapped op uses the new memdesc.
// =============================================================================

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_preserves_pred
  // CHECK: %[[REQ:.+]] = tlx.require_layout
  // CHECK: amdg.async_tdm_copy_global_to_local %{{.*}} into %[[REQ]], pred = %{{.*}}
  tt.func public @tdm_preserves_pred(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----
// =============================================================================
// 15. Sibling-subview pattern (the real GEMM software-pipeline shape):
//     the TDM op writes to slot N of a multi-buffer alloc while the
//     `local_load` reads from slot M (different `memdesc_index` op on the
//     same alloc). `findDotConsumer` must walk *up* to the alloc and back
//     *down* to find the load — a downstream-only walk from the TDM op's
//     buffer would miss it and silently fall back to the default encoding.
//     With WMMA-tuned encoding propagation, the anchor uses `[128:+8]`.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 32]}>

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_sibling_subviews_dot
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  tt.func public @tdm_sibling_subviews_dot(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32, %slot_w: i32, %slot_r: i32)
      -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf_w = ttg.memdesc_index %alloc[%slot_w] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf_w, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %buf_r = ttg.memdesc_index %alloc[%slot_r] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %t = ttg.local_load %buf_r : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.return %t : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
  }
}

// -----
// =============================================================================
// 16. scf.for iter-arg carrier: the buffer's subview is loop-carried. The
//     proper sparse backward dataflow (DotConsumerBackward) handles the
//     iter-arg via SparseBackwardDataFlowAnalysis's region-branch support;
//     the previous hand-rolled walk would have stopped at the iter-arg
//     boundary and missed the dot consumer. WMMA-tuned encoding `[128:+8]`.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 32]}>

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_loop_carried_buffer
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  tt.func public @tdm_loop_carried_buffer(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k_init: i32, %p: i32, %lo: i32, %hi: i32, %step: i32)
      -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf_w = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k_init] into %buf_w, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %buf_r0 = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    // The buffer subview is carried through iter_args of an scf.for loop;
    // the local_load consuming it lives inside the loop body.
    %r = scf.for %i = %lo to %hi step %step iter_args(%buf_iter = %buf_r0)
        -> (!ttg.memdesc<128x32xf16, #shared, #smem, mutable>) : i32 {
      scf.yield %buf_iter : !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    }
    %t = ttg.local_load %r : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.return %t : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
  }
}

// -----
// =============================================================================
// 17. End-to-end GEMM-shaped pattern: A and B descriptors, two TDM copies,
//     dot consumer. Both TDM ops anchor with the WMMA-tuned padded encoding
//     (A: opIdx=0 non-transposed -> `[128:+8]`; B: opIdx=1 transposed ->
//     `[128:+16]`); no swizzled-shared anchors from the dot-path walk on
//     TDM-fed buffers.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 32]}>
// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[128:+16] {order = [1, 0], shape = [32, 128]}>

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @gemm_pattern
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK: amdg.async_tdm_copy_global_to_local
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<32x128xf16, #{{.*}}, #smem, mutable>
  // CHECK: amdg.async_tdm_copy_global_to_local
  // CHECK-NOT: tlx.require_layout
  tt.func public @gemm_pattern(%desc_a: !tt.tensordesc<128x32xf16>, %desc_b: !tt.tensordesc<32x128xf16>, %m: i32, %n: i32, %k: i32, %p: i32)
      -> (tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
          tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>) {
    %c0 = arith.constant 0 : i32
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable>
    %buf_a = ttg.memdesc_index %alloc_a[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %buf_b = ttg.memdesc_index %alloc_b[%c0] : !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %tok_a = amdg.async_tdm_copy_global_to_local %desc_a[%m, %k] into %buf_a, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok_b = amdg.async_tdm_copy_global_to_local %desc_b[%k, %n] into %buf_b, pred = %p : !tt.tensordesc<32x128xf16> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %a = ttg.local_load %buf_a : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %b = ttg.local_load %buf_b : !ttg.memdesc<32x128xf16, #shared, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    tt.return %a, %b : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>, tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
  }
}

// -----
// =============================================================================
// 18. TDM store: anchor a `tlx.require_layout` on the source memdesc with
//     the descriptor-shape default encoding `[32:+8]`. No dot consumer
//     (the buffer is filled by `local_store`, not by a `local_load -> dot`),
//     so the WMMA-tuned path doesn't fire.
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_store_anchor
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_local_to_global
  tt.func public @tdm_store_anchor(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %src: tensor<128x32xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    ttg.local_store %src, %buf : tensor<128x32xf16, #blocked> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    amdg.async_tdm_copy_local_to_global %desc[%m, %k] from %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> !tt.tensordesc<128x32xf16>
    tt.return
  }
}

// -----
// =============================================================================
// 19. TDM store with a dot-fed source memdesc (read-modify-write pattern):
//     even though a `local_load -> tt.dot` reads the buffer, the store
//     anchor uses the default `[32:+8]` because the TDM store hardware
//     verifier requires `padInterval == innermost` (rejects WMMA-tuned).
//     `isFedByTDM` returns true for store-touched buffers so the
//     dot-path walk hands off to the TDM anchor (no swizzled anchor).
// =============================================================================

// CHECK-DAG: #{{.*}} = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>
// CHECK-NOT: #ttg.padded_shared<[128

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_store_with_dot_reader
  // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x32xf16, #{{.*}}, #smem, mutable>
  // CHECK-NEXT: amdg.async_tdm_copy_local_to_global
  tt.func public @tdm_store_with_dot_reader(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %src: tensor<128x32xf16, #blocked>)
      -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    ttg.local_store %src, %buf : tensor<128x32xf16, #blocked> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %t = ttg.local_load %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    amdg.async_tdm_copy_local_to_global %desc[%m, %k] from %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> !tt.tensordesc<128x32xf16>
    tt.return %t : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
  }
}

// -----
// =============================================================================
// 20. TDM prefetch is a fire-and-forget hint with no memdesc; the pass
//     leaves it untouched and emits no anchor.
// =============================================================================

// CHECK-NOT: tlx.require_layout

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tdm_prefetch_only
  // CHECK: amdg.tdm_prefetch
  // CHECK-NOT: tlx.require_layout
  tt.func public @tdm_prefetch_only(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i1) {
    amdg.tdm_prefetch %desc[%m, %k], %p, speculative = false : !tt.tensordesc<128x32xf16>
    tt.return
  }
}
