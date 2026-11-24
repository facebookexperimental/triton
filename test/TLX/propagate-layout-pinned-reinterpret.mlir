// RUN: triton-opt -split-input-file --tlx-propagate-layout %s | FileCheck %s

// A rank-changing local_reinterpret with an explicit layout is an independent
// view of the same bytes: the 3D TMA tile keeps its unswizzled NVMMA layout,
// and the pinned 2D shared_linear view feeds the MMA unchanged.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16, rank = 3}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 8], [0, 16], [0, 32]]}, alignment = 16>
#shared4 = #tlx.user_layout<#shared3>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {tlx.has_explicit_local_mem_access = true, tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$NVMMA3D:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16, rank = 3}>
  // CHECK-DAG: #[[$LINEAR:.*]] = #ttg.shared_linear<{offset = {{\[\[}}0, 1], [0, 2], [0, 4], [1, 0]
  // CHECK-LABEL: @pinned_rank_changing_view
  tt.func public @pinned_rank_changing_view(%a_desc: !tt.tensordesc<8x128x8xf16>) {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x8x128x8xf16, #[[$NVMMA3D]]
    %a_box = ttg.local_alloc : () -> !ttg.memdesc<1x8x128x8xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %bar_0 = ttg.memdesc_index %bar[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar_0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %bar_0, 16384, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: %[[TILE:.*]] = ttg.memdesc_index {{.*}} -> !ttg.memdesc<8x128x8xf16, #[[$NVMMA3D]]
    %0 = ttg.memdesc_index %a_box[%c0_i32] : !ttg.memdesc<1x8x128x8xf16, #shared, #smem, mutable> -> !ttg.memdesc<8x128x8xf16, #shared, #smem, mutable>
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}} -> !ttg.memdesc<8x128x8xf16, #[[$NVMMA3D]]
    ttng.async_tma_copy_global_to_local %a_desc[%c0_i32, %c0_i32, %c0_i32] %0, %bar_0, %true : !tt.tensordesc<8x128x8xf16>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<8x128x8xf16, #shared, #smem, mutable>
    %b_s = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared2, #smem, mutable>
    %1 = ttg.memdesc_index %b_s[%c0_i32] : !ttg.memdesc<1x64x64xf16, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>
    ttng.wait_barrier %bar_0, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttg.memdesc_reinterpret %[[TILE]] : !ttg.memdesc<8x128x8xf16, #[[$NVMMA3D]], #smem, mutable> -> !ttg.memdesc<128x64xf16, #[[$LINEAR]], #smem, mutable>
    %a = ttg.memdesc_reinterpret %0 : !ttg.memdesc<8x128x8xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>
    %2 = ttg.memdesc_index %acc[%c0_i32] : !ttg.memdesc<1x128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %3 = tlx.require_layout %a : !ttg.memdesc<128x64xf16, #shared4, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared3, #smem, mutable>
    // CHECK: ttng.tc_gen5_mma {{.*}} : !ttg.memdesc<128x64xf16, #[[$LINEAR]]
    %4 = ttng.tc_gen5_mma %3, %1, %2[], %false, %true : !ttg.memdesc<128x64xf16, #shared3, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
