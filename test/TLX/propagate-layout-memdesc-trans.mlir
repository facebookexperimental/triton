// RUN: triton-opt -split-input-file --tlx-propagate-layout %s | FileCheck %s

// Test that tlx-propagate-layout can propagate a swizzled_shared constraint
// backward through ttg.memdesc_trans. Swizzle parameters (vec/perPhase/
// maxPhase) describe an XOR pattern over the contiguous dim and are invariant
// under transpose; only `order` is permuted by the inverse transpose order.

#shared_src = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_trans = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared_req = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// CHECK-DAG: #[[$SHARED_TRANS:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
// CHECK-DAG: #[[$SHARED_REQ:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @propagate_swizzled_shared_through_memdesc_trans
  tt.func public @propagate_swizzled_shared_through_memdesc_trans() -> tensor<128x64xf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #[[$SHARED_TRANS]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared_src, #smem, mutable>
    // CHECK: %[[SLICE:.*]] = ttg.memdesc_index %{{.*}}[%c0_i32] : !ttg.memdesc<1x64x128xf16, #[[$SHARED_TRANS]], #smem, mutable> -> !ttg.memdesc<64x128xf16, #[[$SHARED_TRANS]], #smem, mutable>
    %slice = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared_src, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared_src, #smem, mutable>
    // CHECK: %[[TRANS:.*]] = ttg.memdesc_trans %[[SLICE]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #[[$SHARED_TRANS]], #smem, mutable> -> !ttg.memdesc<128x64xf16, #[[$SHARED_REQ]], #smem, mutable>
    %trans = ttg.memdesc_trans %slice {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared_src, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared_trans, #smem, mutable>
    %req = tlx.require_layout %trans : !ttg.memdesc<128x64xf16, #shared_trans, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared_req, #smem, mutable>
    // CHECK: ttg.local_load %[[TRANS]] : !ttg.memdesc<128x64xf16, #[[$SHARED_REQ]], #smem, mutable> -> tensor<128x64xf16, #blocked>
    %val = ttg.local_load %req : !ttg.memdesc<128x64xf16, #shared_req, #smem, mutable> -> tensor<128x64xf16, #blocked>
    tt.return %val : tensor<128x64xf16, #blocked>
  }
}

// -----
// Test that tlx-propagate-layout can also propagate an nvmma_shared constraint
// backward through ttg.memdesc_trans. This exercises the dedicated
// NVMMASharedEncodingAttr branch in LayoutBackwardPropagation.

#nvmma_src = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#nvmma_trans = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#nvmma_req = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
// CHECK-DAG: #[[$NVMMA_SRC:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
// CHECK-DAG: #[[$NVMMA_DST:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#blocked_nv = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#smem_nv = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @propagate_nvmma_shared_through_memdesc_trans
  tt.func public @propagate_nvmma_shared_through_memdesc_trans() -> tensor<128x64xf16, #blocked_nv> {
    %c0_i32 = arith.constant 0 : i32
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #[[$NVMMA_SRC]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #nvmma_src, #smem_nv, mutable>
    // CHECK: %[[SLICE:.*]] = ttg.memdesc_index %{{.*}}[%c0_i32] : !ttg.memdesc<1x64x128xf16, #[[$NVMMA_SRC]], #smem, mutable> -> !ttg.memdesc<64x128xf16, #[[$NVMMA_SRC]], #smem, mutable>
    %slice = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<1x64x128xf16, #nvmma_src, #smem_nv, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_src, #smem_nv, mutable>
    // CHECK: %[[TRANS:.*]] = ttg.memdesc_trans %[[SLICE]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #[[$NVMMA_SRC]], #smem, mutable> -> !ttg.memdesc<128x64xf16, #[[$NVMMA_DST]], #smem, mutable>
    %trans = ttg.memdesc_trans %slice {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #nvmma_src, #smem_nv, mutable> -> !ttg.memdesc<128x64xf16, #nvmma_trans, #smem_nv, mutable>
    %req = tlx.require_layout %trans : !ttg.memdesc<128x64xf16, #nvmma_trans, #smem_nv, mutable> -> !ttg.memdesc<128x64xf16, #nvmma_req, #smem_nv, mutable>
    // CHECK: ttg.local_load %[[TRANS]] : !ttg.memdesc<128x64xf16, #[[$NVMMA_DST]], #smem, mutable> -> tensor<128x64xf16, #blocked>
    %val = ttg.local_load %req : !ttg.memdesc<128x64xf16, #nvmma_req, #smem_nv, mutable> -> tensor<128x64xf16, #blocked_nv>
    tt.return %val : tensor<128x64xf16, #blocked_nv>
  }
}

// -----
// Test that tlx-propagate-layout supports nontrivially-swizzled encodings
// through ttg.memdesc_trans (the previously-rejected case). The required
// swizzle (vec=4, perPhase=2, maxPhase=8, order=[1,0]) propagates back to the
// source alloc with the same vec/perPhase/maxPhase but the inverse-permuted
// order=[0,1].

#shared_src_sw = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_trans_sw = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared_req_sw = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
// CHECK-DAG: #[[$SHARED_SRC_SW:.*]] = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1]}>
// CHECK-DAG: #[[$SHARED_REQ_SW:.*]] = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#blocked_sw = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#smem_sw = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @propagate_nontrivial_swizzled_through_memdesc_trans
  tt.func public @propagate_nontrivial_swizzled_through_memdesc_trans() -> tensor<128x64xf16, #blocked_sw> {
    %c0_i32 = arith.constant 0 : i32
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #[[$SHARED_SRC_SW]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared_src_sw, #smem_sw, mutable>
    // CHECK: %[[SLICE:.*]] = ttg.memdesc_index %{{.*}}[%c0_i32] : !ttg.memdesc<1x64x128xf16, #[[$SHARED_SRC_SW]], #smem, mutable> -> !ttg.memdesc<64x128xf16, #[[$SHARED_SRC_SW]], #smem, mutable>
    %slice = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared_src_sw, #smem_sw, mutable> -> !ttg.memdesc<64x128xf16, #shared_src_sw, #smem_sw, mutable>
    // CHECK: %[[TRANS:.*]] = ttg.memdesc_trans %[[SLICE]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #[[$SHARED_SRC_SW]], #smem, mutable> -> !ttg.memdesc<128x64xf16, #[[$SHARED_REQ_SW]], #smem, mutable>
    %trans = ttg.memdesc_trans %slice {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared_src_sw, #smem_sw, mutable> -> !ttg.memdesc<128x64xf16, #shared_trans_sw, #smem_sw, mutable>
    %req = tlx.require_layout %trans : !ttg.memdesc<128x64xf16, #shared_trans_sw, #smem_sw, mutable> -> !ttg.memdesc<128x64xf16, #shared_req_sw, #smem_sw, mutable>
    // CHECK: ttg.local_load %[[TRANS]] : !ttg.memdesc<128x64xf16, #[[$SHARED_REQ_SW]], #smem, mutable> -> tensor<128x64xf16, #blocked>
    %val = ttg.local_load %req : !ttg.memdesc<128x64xf16, #shared_req_sw, #smem_sw, mutable> -> tensor<128x64xf16, #blocked_sw>
    tt.return %val : tensor<128x64xf16, #blocked_sw>
  }
}
