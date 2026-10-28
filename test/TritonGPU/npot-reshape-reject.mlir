// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -verify-diagnostics

// Under TRITON_ALLOW_NPOT, memdesc reshape inference rejects an NPOT source or
// result.

#sw = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#sw1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nv4 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, rank = 4}>
#nv2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @memdesc_3x64(%a: !ttg.memdesc<3x64xf16, #sw, #smem>) {
    // expected-error @+1 {{NPOT reshape is not yet supported}}
    %b = ttg.memdesc_reshape %a : !ttg.memdesc<3x64xf16, #sw, #smem> -> !ttg.memdesc<192xf16, #sw1, #smem>
    tt.return
  }

  tt.func @memdesc_pot_control(%d: !ttg.memdesc<32x1x4x64xf16, #nv4, #smem>) {
    %b = ttg.memdesc_reshape %d : !ttg.memdesc<32x1x4x64xf16, #nv4, #smem> -> !ttg.memdesc<128x64xf16, #nv2, #smem>
    tt.return
  }
}
