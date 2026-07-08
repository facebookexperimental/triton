// RUN: TRITON_ALLOW_NPOT=0 triton-opt %s -verify-diagnostics

// Flag-gating half of the relaxation: with TRITON_ALLOW_NPOT OFF, the exact
// same alloc that npot-memdesc-verify.mlir accepts flag-ON is rejected. Proves
// verifyAllocOp is byte-identical to the pre-NPOT strict check when the flag is
// off (an alloc is never born a subview).

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @flag_off_rejects_rounded() {
  // expected-error @+1 {{result shape and its alloc shape must match}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<144x64xf16, #shared, #smem, mutable, 256x64>
  tt.return
}
}
