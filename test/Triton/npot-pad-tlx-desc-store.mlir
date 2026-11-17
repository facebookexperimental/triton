// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad='target=cuda:90 num-warps=4' -verify-diagnostics | FileCheck %s

// A descriptor store whose TMA box is legal at the NPOT shape
// (whole box, every dim at most 256, inner extent a whole number of
// 16B atoms) keeps its exact box.
#s0 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem0 = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @desc_store_box192
  tt.func @desc_store_box192(%do: !tt.tensordesc<64x192xf16>) {
    %c0 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x192xf16, #s0, #smem0, mutable>
    %v = ttg.local_load %buf : !ttg.memdesc<64x192xf16, #s0, #smem0, mutable> -> tensor<64x192xf16>
    // CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<64x192xf16>, tensor<64x192xf16>
    tt.descriptor_store %do[%c0, %c0], %v : !tt.tensordesc<64x192xf16>, tensor<64x192xf16>
    tt.return
  }
}

// -----

// Control: a box dim over the TMA 256 limit is not a legal box, so
// the store rejects.
#s1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem1 = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @desc_store_box264(%pa: !tt.ptr<f16>, %do: !tt.tensordesc<64x264xf16>) {
    %c0 = arith.constant 0 : i32
    %sp = tt.splat %pa : !tt.ptr<f16> -> tensor<64x264x!tt.ptr<f16>>
    %v = tt.load %sp : tensor<64x264x!tt.ptr<f16>>
    // expected-remark @+2 {{NPOT TMA box 256 does not divide extent 264}}
    // expected-error @+1 {{'tt.descriptor_store' op on an NPOT tensor is unsupported}}
    tt.descriptor_store %do[%c0, %c0], %v : !tt.tensordesc<64x264xf16>, tensor<64x264xf16>
    tt.return
  }
}
