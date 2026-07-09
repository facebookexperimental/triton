// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s -allow-unregistered-dialect -test-print-allocation -verify-diagnostics -o /dev/null

// NPOT nvmma_shared allocation (TRITON_ALLOW_NPOT). Verifies getAllocationShapePerCTA
// rounds ONLY the trailing operand-tile dims to pow2, never the leading
// multi-buffer/pipeline-stage dim (rounding it would inflate SMEM and OOM).

#NVMMA_SHARED_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {

// A non-power-of-two leading multi-buffer (pipeline-stage) dim must NOT be
// rounded up: 6 stages cost exactly 6, not 8.
//   6 * 64 * 128 * 2 bytes = 98304   (NOT 8 * 64 * 128 * 2 = 131072)
// expected-remark @below {{nvmma_npot_multibuffer}}
// expected-remark @below {{size = 98304}}
tt.func @nvmma_npot_multibuffer() {
  // expected-remark @below {{offset = 0, size = 98304}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<6x64x128xf16, #NVMMA_SHARED_128, #ttg.shared_memory, mutable>
  tt.return
}

// The leading multi-buffer dim is kept as-is, but NPOT *tile* dims are still
// rounded to pow2: leading 6 preserved, 144 row dim rounds to 256.
//   6 * 256 * 64 * 2 bytes = 196608
// expected-remark @below {{nvmma_npot_tile_dim_still_rounded}}
// expected-remark @below {{size = 196608}}
tt.func @nvmma_npot_tile_dim_still_rounded() {
  // expected-remark @below {{offset = 0, size = 196608}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<6x144x64xf16, #NVMMA_SHARED_128, #ttg.shared_memory, mutable>
  tt.return
}

}
