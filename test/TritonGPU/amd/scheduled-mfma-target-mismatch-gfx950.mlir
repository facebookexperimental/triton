// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --verify-diagnostics
// RUN: not triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" 2>&1 | FileCheck %s

// The other direction: a CDNA3 layout compiled for gfx950. The instruction
// still exists there, so nothing else in the pipeline catches it, but the
// wait states the lowering pads for are the wrong generation's.

// CHECK: error: 'amdg.scheduled_mfma' op carries a version 3 MFMA layout, which does not match the CDNA4 target
// CHECK: error: failed to legalize operation 'amdg.scheduled_mfma'

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scheduled_mfma_v3_layout_on_cdna4(
      %a: tensor<16x16xbf16, #lhs>,
      %b: tensor<16x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // expected-error @+2 {{carries a version 3 MFMA layout, which does not match the CDNA4 target}}
    // expected-error @+1 {{failed to legalize operation 'amdg.scheduled_mfma'}}
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>,
          tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// CHECK: error: 'amdg.mfma_commit' op carries a version 3 MFMA layout, which does not match the CDNA4 target
// CHECK: error: failed to legalize operation 'amdg.mfma_commit'

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_commit_v3_layout_on_cdna4(
      %acc: tensor<16x16xf32, #mma>,
      %b: tensor<16x16xbf16, #rhs>) {
    // expected-error @+2 {{carries a version 3 MFMA layout, which does not match the CDNA4 target}}
    // expected-error @+1 {{failed to legalize operation 'amdg.mfma_commit'}}
    %committed, %preserved = amdg.mfma_commit %acc, %b
        : tensor<16x16xf32, #mma>,
          tensor<16x16xbf16, #rhs>
    tt.return
  }
}
