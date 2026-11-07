// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --verify-diagnostics | FileCheck %s

// CHECK-LABEL: llvm.func @packed_scale_word
// CHECK: %[[SA:[0-9]+]] = llvm.extractvalue %arg2[0]
// CHECK-NEXT: %[[SB:[0-9]+]] = llvm.extractvalue %arg3[0]
// CHECK-NOT: llvm.lshr
// CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4
// CHECK-SAME: fp8_e4m3, fp8_e5m2, 3, %[[SA]], 2, %[[SB]]

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 64], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @packed_scale_word(
      %a: tensor<32x64xi8, #lhs>,
      %b: tensor<64x32xi8, #rhs>,
      %sa: tensor<32x2xi32, #scale>,
      %sb: tensor<32x2xi32, #scale>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e4m3", "e5m2" resident "none" accumulator "transient"
        register_class "auto" initialize true
        {a_scale_selector = 3 : i32, b_scale_selector = 2 : i32}
        : tensor<32x64xi8, #lhs>,
          tensor<64x32xi8, #rhs>,
          tensor<32x32xf32, #mma>,
          tensor<32x2xi32, #scale>,
          tensor<32x2xi32, #scale>
          -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: llvm.func @packed_scale_word_two_steps
// CHECK: %[[SA2:[0-9]+]] = llvm.extractvalue %arg2[0]
// CHECK-NEXT: %[[SB2:[0-9]+]] = llvm.extractvalue %arg3[0]
// CHECK-NOT: llvm.lshr
// CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4
// CHECK-SAME: fp8_e4m3, fp8_e5m2, 0, %[[SA2]], 1, %[[SB2]]
// CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4
// CHECK-SAME: fp8_e4m3, fp8_e5m2, 2, %[[SA2]], 3, %[[SB2]]
// CHECK-NOT: llvm.lshr

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 64], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @packed_scale_word_two_steps(
      %a: tensor<32x128xi8, #lhs>,
      %b: tensor<128x32xi8, #rhs>,
      %sa: tensor<32x2xi32, #scale>,
      %sb: tensor<32x2xi32, #scale>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e4m3", "e5m2" resident "none" accumulator "transient"
        register_class "auto" initialize true
        {a_scale_selector = 0 : i32, b_scale_selector = 1 : i32}
        : tensor<32x128xi8, #lhs>,
          tensor<128x32xi8, #rhs>,
          tensor<32x32xf32, #mma>,
          tensor<32x2xi32, #scale>,
          tensor<32x2xi32, #scale>
          -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 64], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @missing_selectors(
      %a: tensor<32x64xi8, #lhs>,
      %b: tensor<64x32xi8, #rhs>,
      %sa: tensor<32x2xi32, #scale>,
      %sb: tensor<32x2xi32, #scale>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // expected-error @+1 {{packed i32 scales require both byte selectors in [0, 3]}}
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e4m3", "e5m2" resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<32x64xi8, #lhs>,
          tensor<64x32xi8, #rhs>,
          tensor<32x32xf32, #mma>,
          tensor<32x2xi32, #scale>,
          tensor<32x2xi32, #scale>
          -> tensor<32x32xf32, #mma>
    tt.return
  }
}
