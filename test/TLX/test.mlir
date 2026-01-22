// RUN: triton-opt -split-input-file --tlx-propagate-layout %s| FileCheck %s

// -----

// Test that multibuffering is cancelled for TMEM scales allocations.
// When a TMEMAllocOp with a 3D shape (1xMxK) receives TensorMemoryScalesEncodingAttr,
// the shape should be flattened to 2D (MxK) and memdesc_index ops should be removed.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared_scales = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, CTAsPerCGA = [1, 1, 1, 1, 1], CTASplitNum = [1, 1, 1, 1, 1], CTAOrder = [4, 3, 2, 1, 0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#dummy_tmem_layout = #tlx.dummy_tmem_layout<>
#scales_encoding = #ttng.tensor_memory_scales_encoding<>

// CHECK-DAG: #[[$TMEM_SCALES:.*]] = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @cancel_multibuffering_for_tmem_scales
  tt.func public @cancel_multibuffering_for_tmem_scales(
      %a_smem: !ttg.memdesc<128x256xf8E4M3FN, #shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem, mutable>,
      %a_scale_smem: !ttg.memdesc<1x1x2x2x256xi8, #shared_scales, #smem, mutable>,
      %b_scale_smem: !ttg.memdesc<1x1x2x2x256xi8, #shared_scales, #smem, mutable>) {
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true

    // Accumulator in TMEM
    %c_tile = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem_acc, #tmem, mutable>

    // CHECK: ttng.tmem_alloc : () -> !ttg.memdesc<128x8xi8, #[[$TMEM_SCALES]], #ttng.tensor_memory, mutable>
    %a_scale_tmem = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x8xi8, #dummy_tmem_layout, #tmem, mutable>
    // CHECK: ttng.tmem_alloc : () -> !ttg.memdesc<256x4xi8, #[[$TMEM_SCALES]], #ttng.tensor_memory, mutable>
    %b_scale_tmem = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x4xi8, #dummy_tmem_layout, #tmem, mutable>

    // CHECK-NOT: ttg.memdesc_index %{{.*}} : !ttg.memdesc<1x128x8xi8
    // CHECK-NOT: ttg.memdesc_index %{{.*}} : !ttg.memdesc<1x256x4xi8
    %a_scale_indexed = ttg.memdesc_index %a_scale_tmem[%c0_i32] : !ttg.memdesc<1x128x8xi8, #dummy_tmem_layout, #tmem, mutable> -> !ttg.memdesc<128x8xi8, #dummy_tmem_layout, #tmem, mutable>
    %b_scale_indexed = ttg.memdesc_index %b_scale_tmem[%c0_i32] : !ttg.memdesc<1x256x4xi8, #dummy_tmem_layout, #tmem, mutable> -> !ttg.memdesc<256x4xi8, #dummy_tmem_layout, #tmem, mutable>

    // Copy scales from SMEM to TMEM
    ttng.tmem_copy %a_scale_smem, %a_scale_indexed : !ttg.memdesc<1x1x2x2x256xi8, #shared_scales, #smem, mutable>, !ttg.memdesc<128x8xi8, #dummy_tmem_layout, #tmem, mutable>
    ttng.tmem_copy %b_scale_smem, %b_scale_indexed : !ttg.memdesc<1x1x2x2x256xi8, #shared_scales, #smem, mutable>, !ttg.memdesc<256x4xi8, #dummy_tmem_layout, #tmem, mutable>

    // Require scales layout for the MMA op
    %a_scale_req = tlx.require_layout %a_scale_indexed : !ttg.memdesc<128x8xi8, #dummy_tmem_layout, #tmem, mutable> -> !ttg.memdesc<128x8xi8, #scales_encoding, #tmem, mutable>
    %b_scale_req = tlx.require_layout %b_scale_indexed : !ttg.memdesc<256x4xi8, #dummy_tmem_layout, #tmem, mutable> -> !ttg.memdesc<256x4xi8, #scales_encoding, #tmem, mutable>

    // CHECK: ttng.tc_gen5_mma_scaled
    %0 = ttng.tc_gen5_mma_scaled %a_smem, %b_smem, %c_tile[], %a_scale_req, %b_scale_req, %false, %true lhs = e4m3 rhs = e4m3 : !ttg.memdesc<128x256xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem_acc, #tmem, mutable>, !ttg.memdesc<128x8xi8, #scales_encoding, #tmem, mutable>, !ttg.memdesc<256x4xi8, #scales_encoding, #tmem, mutable>
    tt.return
  }
}
