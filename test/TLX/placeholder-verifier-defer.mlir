// RUN: triton-opt -split-input-file %s --verify-diagnostics | FileCheck %s

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

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [8, 1], instrShape = [32, 32, 16], isTransposed = true}>
#result = #tlx.no_verify_layout<#tlx.user_layout<#mma>>
#operand_a = #tlx.no_verify_layout<#tlx.user_layout<#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  // Helper ABI reconciliation can temporarily expose a pinned operand and
  // accumulator beside an unresolved operand. The no_verify result defers dot
  // compatibility until placeholder resolution gives all three concrete types.
  // CHECK-LABEL: @dot_placeholder_with_unresolved_operand
  tt.func @dot_placeholder_with_unresolved_operand(
      %a: tensor<256x64xbf16, #operand_a>,
      %b: tensor<64x64xbf16>,
      %c: tensor<256x64xf32, #result>) -> tensor<256x64xf32, #result> {
    // CHECK: tt.dot
    %dot = tt.dot %a, %b, %c : tensor<256x64xbf16, #operand_a> * tensor<64x64xbf16> -> tensor<256x64xf32, #result>
    tt.return %dot : tensor<256x64xf32, #result>
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

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#ph = #tlx.no_verify_layout<#tlx.user_layout<#linear>>
#slice = #ttg.slice<{dim = 1, parent = #ph}>
module {
  // The nested placeholder keeps canonical slice-parent inference stable and
  // recursively defers verification while frontend IR has no ttg.num-warps
  // context.
  // CHECK-LABEL: @nested_placeholder_slice
  tt.func @nested_placeholder_slice(%x: tensor<128x128xf32, #ph>) -> tensor<128xf32, #slice> {
    // CHECK: "tt.reduce"
    %m = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %max = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %max : f32
    }) : (tensor<128x128xf32, #ph>) -> tensor<128xf32, #slice>
    tt.return %m : tensor<128xf32, #slice>
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#user = #tlx.user_layout<#shared>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @user_layout_does_not_defer_verification(%ptr: !tt.ptr<f32>) {
    // expected-error @+1 {{Non-distributed layout is not allowed in tensor type}}
    %tensor = tt.splat %ptr : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #user>
    tt.return
  }
}
