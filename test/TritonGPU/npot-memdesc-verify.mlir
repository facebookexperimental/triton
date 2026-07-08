// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics

// verifyAllocOp (Ops.cpp) under TRITON_ALLOW_NPOT: an alloc may have
// shape != allocShape ONLY when allocShape is the pow2-ceil rounding of shape
// (getAllocationShapePerCTA rounds NPOT tile dims up to the next power of two).
// The gate is a per-dim pow2-ceil match, NOT a "shape is NPOT" test, so pow2
// dims keep the strict guarantee and single-buffer NPOT allocs are accepted.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// ACCEPT: single-buffer NPOT tile rounded to pow2 (144 -> 256). The leading dim
// is itself a tile here; a drop_front(1) rule would wrongly reject this.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @npot_single_buffer_rounded() {
  %a = ttg.local_alloc : () -> !ttg.memdesc<144x64xf16, #shared, #smem, mutable, 256x64>
  tt.return
}
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// ACCEPT: multi-buffered NPOT (leading pipeline-stage dim 6 preserved exactly,
// trailing tile dim 144 -> 256).
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @npot_multibuffer_rounded() {
  %a = ttg.local_alloc : () -> !ttg.memdesc<6x144x64xf16, #shared, #smem, mutable, 6x256x64>
  tt.return
}
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// REJECT: over-alloc -- 144 grown to 512, not its pow2-ceil 256.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @npot_over_alloc() {
  // expected-error @+1 {{result shape and its alloc shape must match}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<144x64xf16, #shared, #smem, mutable, 512x64>
  tt.return
}
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// REJECT: a pow2 shape keeps the full strict guarantee even flag-ON --
// PowerOf2Ceil(128) == 128, so 128 != 256 cannot be a rounding.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @pow2_mismatch_still_strict() {
  // expected-error @+1 {{result shape and its alloc shape must match}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64>
  tt.return
}
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

// REJECT: the NPOT exemption relaxes only the pow2 requirement, never the
// non-zero one -- a zero trailing dim is rejected even flag-ON for an exempt
// (NVMMAShared) encoding.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @exempt_zero_trailing_dim_rejected() {
  // expected-error @+1 {{allocShape must have power-of-2 and non-zero dimensions}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<6x0x64xf16, #nvmma, #smem, mutable>
  tt.return
}
}
