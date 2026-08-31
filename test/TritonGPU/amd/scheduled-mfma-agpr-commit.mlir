// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942" --verify-diagnostics

// CDNA3 refuses an explicit AGPR accumulator outright, before any commit
// boundary is considered. The MFMA and its hazard padding are inline assembly,
// so LLVM schedules the compiler-generated v_accvgpr_read into the shadow of
// the MFMA that wrote the register. The commit-time diagnostic below cannot
// cover that: it needs the producing scheduled_mfma to be visible from an
// mfma_commit, and an epilogue with no commit -- or one behind an
// amd_register_handoff -- silently miscompiles instead.
//
// The commit-time AGPR/live-operand interaction is still covered on CDNA4, in
// scheduled-mfma-agpr-commit-gfx950.mlir, where the explicit class is legal.

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @agpr_accumulator_with_live_operand(
      %a: tensor<16x16xbf16, #lhs>,
      %b: tensor<16x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // expected-error @+1 {{accumulator_register_class "agpr" is not yet supported on CDNA3}}
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<16x16xbf16, #lhs>,
          tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %committed, %preserved = amdg.mfma_commit %result, %b
        : tensor<16x16xf32, #mma>,
          tensor<16x16xbf16, #rhs>
    tt.return
  }
}

// -----

// The refusal does not depend on the commit boundary: with no live dot operand
// at all, an explicit AGPR accumulator is still rejected on CDNA3. This is the
// case the commit-time diagnostic could never see.

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @agpr_accumulator_without_live_operand(
      %a: tensor<16x16xbf16, #lhs>,
      %b: tensor<16x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // expected-error @+1 {{accumulator_register_class "agpr" is not yet supported on CDNA3}}
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<16x16xbf16, #lhs>,
          tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %committed = amdg.mfma_commit %result
        : tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// `auto` names AGPRs for a persistent chain, so CDNA3 rejects it too: no
// silent fallback to VGPRs.

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @auto_persistent_accumulator_with_live_operand(
      %a: tensor<16x16xbf16, #lhs>,
      %b: tensor<16x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // expected-error @+1 {{accumulator_register_class "auto" is not yet supported on CDNA3 for a "persistent" accumulator}}
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x16xbf16, #lhs>,
          tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %committed, %preserved = amdg.mfma_commit %result, %b
        : tensor<16x16xf32, #mma>,
          tensor<16x16xbf16, #rhs>
    tt.return
  }
}

// -----

// The explicit VGPR class is how a persistent accumulator is carried on CDNA3,
// and it lowers cleanly across a commit that also carries a live dot operand.

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @vgpr_accumulator_with_live_operand(
      %a: tensor<16x16xbf16, #lhs>,
      %b: tensor<16x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>,
          tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %committed, %preserved = amdg.mfma_commit %result, %b
        : tensor<16x16xf32, #mma>,
          tensor<16x16xbf16, #rhs>
    tt.return
  }
}
