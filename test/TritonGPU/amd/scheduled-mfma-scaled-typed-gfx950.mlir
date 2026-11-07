// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --verify-diagnostics | FileCheck %s

// Typed FP8 operands take the scaled-dot lowering while packed I32 scale words
// flow directly to the native byte selectors.
// CHECK-LABEL: llvm.func @typed_transient
// CHECK-NOT: llvm.lshr
// CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4
// CHECK-SAME: fp8_e4m3, fp8_e4m3

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 64], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
#scale = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @typed_transient(
      %a: tensor<32x64xf8E4M3FN, #lhs>,
      %b: tensor<64x32xf8E4M3FN, #rhs>,
      %sa: tensor<32x2xi32, #scale>,
      %sb: tensor<32x2xi32, #scale>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e4m3", "e4m3" resident "none" accumulator "transient"
        register_class "auto" initialize true
        {a_scale_selector = 0 : i32, b_scale_selector = 1 : i32}
        : tensor<32x64xf8E4M3FN, #lhs>,
          tensor<64x32xf8E4M3FN, #rhs>,
          tensor<32x32xf32, #mma>,
          tensor<32x2xi32, #scale>,
          tensor<32x2xi32, #scale>
          -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// A persistent chain keeps one AGPR-pinned accumulator across consecutive
// native scaled MFMAs.
// CHECK-LABEL: llvm.func @typed_persistent_agpr
// CHECK-COUNT-2: rocdl.mfma.scale.f32.32x32x64.f8f6f4
// CHECK: llvm.inline_asm
// CHECK-SAME: "=a,0"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 64], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
#scale = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @typed_persistent_agpr(
      %a: tensor<32x64xf8E4M3FN, #lhs>,
      %b: tensor<64x32xf8E4M3FN, #rhs>,
      %sa: tensor<32x2xi32, #scale>,
      %sb: tensor<32x2xi32, #scale>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %first = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e4m3", "e4m3" resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        {amdg.cd_regclass = "a", a_scale_selector = 0 : i32, b_scale_selector = 1 : i32}
        : tensor<32x64xf8E4M3FN, #lhs>,
          tensor<64x32xf8E4M3FN, #rhs>,
          tensor<32x32xf32, #mma>,
          tensor<32x2xi32, #scale>,
          tensor<32x2xi32, #scale>
          -> tensor<32x32xf32, #mma>
    %second = amdg.scheduled_mfma_scaled %a, %b, %first scales %sa, %sb
        formats "e4m3", "e4m3" resident "none" accumulator "persistent"
        register_class "agpr" initialize false
        {amdg.cd_regclass = "a", a_scale_selector = 2 : i32, b_scale_selector = 3 : i32}
        : tensor<32x64xf8E4M3FN, #lhs>,
          tensor<64x32xf8E4M3FN, #rhs>,
          tensor<32x32xf32, #mma>,
          tensor<32x2xi32, #scale>,
          tensor<32x2xi32, #scale>
          -> tensor<32x32xf32, #mma>
    tt.return
  }
}
