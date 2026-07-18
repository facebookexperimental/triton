// RUN: triton-opt -split-input-file %s | FileCheck %s

// Verifier relaxation for TLX placeholder (deferred) layouts. A user-pinned
// layout is wrapped as #tlx.no_verify_layout<#tlx.user_layout<...>> at the load
// and only resolved in make_ttgir. Until then it can legitimately meet
// concrete/absent encodings on ops whose verifiers would otherwise reject the
// mix; those verifiers treat a TLX placeholder wrapper as "no layout" (defer).
//
// These modules only need to parse+verify (that is where the relaxed op
// verifiers run) -- no pass is applied.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ph = #tlx.no_verify_layout<#tlx.user_layout<#linear>>
#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  // ReshapeOp::verify defers when the src carries a placeholder, even though the
  // dst is a concrete (blocked) 3D layout the split path chose -- exactly the
  // P-tail case. Without the relaxation this fails with "src and dst both have
  // encodings, or ... neither".
  // CHECK-LABEL: @reshape_placeholder_src_concrete_dst
  tt.func @reshape_placeholder_src_concrete_dst(%x: tensor<128x128xbf16, #ph>) -> tensor<128x2x64xbf16, #blocked3d> {
    // CHECK: tt.reshape
    %y = tt.reshape %x allow_reorder : tensor<128x128xbf16, #ph> -> tensor<128x2x64xbf16, #blocked3d>
    tt.return %y : tensor<128x2x64xbf16, #blocked3d>
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ph = #tlx.no_verify_layout<#tlx.user_layout<#linear>>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  // ReshapeOp::verify also defers when the dst has no encoding (the placeholder
  // src alone defers the whole check).
  // CHECK-LABEL: @reshape_placeholder_src_null_dst
  tt.func @reshape_placeholder_src_null_dst(%x: tensor<128x128xbf16, #ph>) -> tensor<128x2x64xbf16> {
    // CHECK: tt.reshape
    %y = tt.reshape %x allow_reorder : tensor<128x128xbf16, #ph> -> tensor<128x2x64xbf16>
    tt.return %y : tensor<128x2x64xbf16>
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ph = #tlx.no_verify_layout<#tlx.user_layout<#linear>>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  // SameOperandsAndResultEncoding (verifySameEncoding) peels the TLX placeholder
  // wrapper off the operand and verifies the underlying concrete layout: here the
  // placeholder wraps #linear and the result is that same concrete #linear, so a
  // triton elementwise-inline-asm op with a placeholder operand and concrete
  // result verifies.
  // CHECK-LABEL: @same_encoding_placeholder_operand
  tt.func @same_encoding_placeholder_operand(%a: tensor<128x128xf32, #ph>) -> tensor<128x128xf32, #linear> {
    // CHECK: tt.elementwise_inline_asm
    %r = tt.elementwise_inline_asm "mov.b32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %a : tensor<128x128xf32, #ph> -> tensor<128x128xf32, #linear>
    tt.return %r : tensor<128x128xf32, #linear>
  }
}
