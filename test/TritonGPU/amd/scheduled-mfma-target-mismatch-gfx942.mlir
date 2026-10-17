// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942" --verify-diagnostics
// The second run checks that the diagnostic also fails the pass. Emitting an
// error is not enough on its own: while the op stayed legal, the conversion
// succeeded and handed back unconverted IR.
// RUN: not triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942" 2>&1 | FileCheck %s

// Target-free IR skips the verifier's encoding/target check, so the lowering
// has to make it: without this, a CDNA4 layout compiled for gfx942 quietly
// emits the gfx950-only v_mfma_f32_16x16x32_bf16.

// CHECK: error: 'amdg.scheduled_mfma' op carries a version 4 MFMA layout, which does not match the CDNA3 target
// CHECK: error: failed to legalize operation 'amdg.scheduled_mfma'

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scheduled_mfma_v4_layout_on_cdna3(
      %a: tensor<16x32xbf16, #lhs>,
      %b: tensor<32x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // expected-error @+2 {{carries a version 4 MFMA layout, which does not match the CDNA3 target}}
    // expected-error @+1 {{failed to legalize operation 'amdg.scheduled_mfma'}}
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>,
          tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// The commit boundary sizes its hazard padding from the target while reading
// the fragment shapes from the encoding, so it rejects the mismatch too.

// CHECK: error: 'amdg.mfma_commit' op carries a version 4 MFMA layout, which does not match the CDNA3 target
// CHECK: error: failed to legalize operation 'amdg.mfma_commit'

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_commit_v4_layout_on_cdna3(
      %acc: tensor<16x16xf32, #mma>,
      %b: tensor<32x16xbf16, #rhs>) {
    // expected-error @+2 {{carries a version 4 MFMA layout, which does not match the CDNA3 target}}
    // expected-error @+1 {{failed to legalize operation 'amdg.mfma_commit'}}
    %committed, %preserved = amdg.mfma_commit %acc, %b
        : tensor<16x16xf32, #mma>,
          tensor<32x16xbf16, #rhs>
    tt.return
  }
}

// -----

// Target-free scaled groups still require the selected gfx950 lowering target.

// CHECK: error: 'amdg.scaled_mfma_group' op is supported only on gfx950
// CHECK: error: failed to legalize operation 'amdg.scaled_mfma_group'
// CHECK-NOT: v_mfma_scale_f32_32x32x64_f8f6f4

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 64], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
#as = #ttg.linear<{register = [[64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[0, 0], [32, 0]], block = []}>
#bs = #ttg.linear<{register = [[64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[32, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scaled_mfma_group_on_cdna3(
      %a: tensor<128x64xf8E4M3FN, #lhs>, %as: tensor<128x2xi8, #as>,
      %b: tensor<64x128xf8E4M3FN, #rhs>, %bs: tensor<128x2xi8, #bs>) {
    %c = arith.constant dense<0.0> : tensor<128x128xf32, #mma>
    // expected-error @+2 {{is supported only on gfx950}}
    // expected-error @+1 {{failed to legalize operation 'amdg.scaled_mfma_group'}}
    %result = amdg.scaled_mfma_group %a scale %as, %b scale %bs, %c full_wave true
        : tensor<128x64xf8E4M3FN, #lhs>, tensor<128x2xi8, #as>,
          tensor<64x128xf8E4M3FN, #rhs>, tensor<128x2xi8, #bs>,
          tensor<128x128xf32, #mma> -> tensor<128x128xf32, #mma>
    tt.return
  }
}
