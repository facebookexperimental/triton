// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad='target=cuda:90 num-warps=4' -verify-diagnostics | FileCheck %s

// A memdesc index into a legal shared tile and an async TMA copy
// into it are views of the alloc layout: both keep the exact shape.
#s0 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#b0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem0 = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @index_copy_192
  tt.func @index_copy_192(%desc: !tt.tensordesc<64x192xf16>, %bar: !ttg.memdesc<1xi64, #b0, #smem0>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x64x192xf16, #s0, #smem0, mutable>
    // CHECK: ttg.memdesc_index {{.*}} -> !ttg.memdesc<64x192xf16
    %tile = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<2x64x192xf16, #s0, #smem0, mutable> -> !ttg.memdesc<64x192xf16, #s0, #smem0, mutable>
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}} -> !ttg.memdesc<64x192xf16
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %tile, %bar, %true : !tt.tensordesc<64x192xf16>, !ttg.memdesc<1xi64, #b0, #smem0> -> !ttg.memdesc<64x192xf16, #s0, #smem0, mutable>
    %v = ttg.local_load %tile : !ttg.memdesc<64x192xf16, #s0, #smem0, mutable> -> tensor<64x192xf16>
    tt.return
  }
}

// -----

// Control: an index into a tile whose rows do not tile whole
// swizzle atoms (192B rows in a 128B swizzle) rejects.
#s1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem1 = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @index_100(%buf: !ttg.memdesc<2x64x96xf16, #s1, #smem1, mutable>) {
    %c0 = arith.constant 0 : i32
    // expected-remark @+2 {{NPOT mixed-radix layout is not yet supported on ttg.memdesc_index}}
    // expected-error @+1 {{'ttg.memdesc_index' op on an NPOT pointer-to-tensor, descriptor, memory descriptor or encoded tensor is unsupported}}
    %tile = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<2x64x96xf16, #s1, #smem1, mutable> -> !ttg.memdesc<64x96xf16, #s1, #smem1, mutable>
    tt.return
  }
}

// -----

// Control: an async TMA copy into the same illegal tile rejects
// in the op verifier, which holds the TMA box rule.
#s2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#b2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem2 = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @copy_100(%desc: !tt.tensordesc<64x100xf16>, %bar: !ttg.memdesc<1xi64, #b2, #smem2>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x100xf16, #s2, #smem2, mutable>
    // expected-error @+1 {{NPOT TMA box 64 does not divide extent 100}}
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true : !tt.tensordesc<64x100xf16>, !ttg.memdesc<1xi64, #b2, #smem2> -> !ttg.memdesc<64x100xf16, #s2, #smem2, mutable>
    tt.return
  }
}

// -----

// A dot on a shared-memory tile whose rows do not tile whole
// swizzle atoms is not exact: the component pads and the dot
// rejects at its gated operand.
#s3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#mma3 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#smem3 = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @dot_illegal_tile(%a: !ttg.memdesc<64x64xf16, #s3, #smem3>, %b: !ttg.memdesc<64x96xf16, #s3, #smem3>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x96xf32, #mma3>
    // expected-remark @+2 {{NPOT mixed-radix layout is not yet supported for !ttg.memdesc<64x96xf16}}
    // expected-error @+1 {{'ttng.warp_group_dot' op on an NPOT pointer-to-tensor, descriptor, memory descriptor or encoded tensor is unsupported}}
    %acc = ttng.warp_group_dot %a, %b, %cst : !ttg.memdesc<64x64xf16, #s3, #smem3> * !ttg.memdesc<64x96xf16, #s3, #smem3> -> tensor<64x96xf32, #mma3>
    tt.return
  }
}
