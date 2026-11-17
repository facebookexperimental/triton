// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --verify-diagnostics | FileCheck %s

// Packed FP6 scheduled MFMA: one native 16x16x128 scaled step over
// 128 logical values (first 24 of 32 per-lane bytes packed, rest zero
// padding), one E8M0 scale byte per 32 values.
// CHECK-LABEL: llvm.func @scaled_fp6_step
// CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4
// CHECK-SAME: fp6_e2m3, fp6_e3m2

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 128], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scaled_fp6_step(
      %a: tensor<16x128xi8, #lhs>,
      %b: tensor<128x16xi8, #rhs>,
      %sa: tensor<16x4xi8, #scale>,
      %sb: tensor<16x4xi8, #scale>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e2m3", "e3m2" resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x128xi8, #lhs>,
          tensor<128x16xi8, #rhs>,
          tensor<16x16xf32, #mma>,
          tensor<16x4xi8, #scale>,
          tensor<16x4xi8, #scale>
          -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// Same tile in FP8: all 32 per-lane bytes are packed codes.
// CHECK-LABEL: llvm.func @scaled_fp8_step
// CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4
// CHECK-SAME: fp8_e4m3, fp8_e5m2

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 128], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scaled_fp8_step(
      %a: tensor<16x128xi8, #lhs>,
      %b: tensor<128x16xi8, #rhs>,
      %sa: tensor<16x4xi8, #scale>,
      %sb: tensor<16x4xi8, #scale>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e4m3", "e5m2" resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x128xi8, #lhs>,
          tensor<128x16xi8, #rhs>,
          tensor<16x16xf32, #mma>,
          tensor<16x4xi8, #scale>,
          tensor<16x4xi8, #scale>
          -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// Same tile in FP4: first 16 per-lane bytes hold 32 codes, two per byte.
// CHECK-LABEL: llvm.func @scaled_fp4_step
// CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4
// CHECK-SAME: fp4_e2m1, fp4_e2m1

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 128], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scaled_fp4_step(
      %a: tensor<16x128xi8, #lhs>,
      %b: tensor<128x16xi8, #rhs>,
      %sa: tensor<16x4xi8, #scale>,
      %sb: tensor<16x4xi8, #scale>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e2m1", "e2m1" resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x128xi8, #lhs>,
          tensor<128x16xi8, #rhs>,
          tensor<16x16xf32, #mma>,
          tensor<16x4xi8, #scale>,
          tensor<16x4xi8, #scale>
          -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// Logical K must fill whole native steps: 96 is not a multiple of 128.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 128], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @bad_k_bytes(
      %a: tensor<16x96xi8, #lhs>,
      %b: tensor<96x16xi8, #rhs>,
      %sa: tensor<16x4xi8, #scale>,
      %sb: tensor<16x4xi8, #scale>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // expected-error @+1 {{operand K dimension must be a positive multiple of the native instruction K}}
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e2m3", "e3m2" resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x96xi8, #lhs>,
          tensor<96x16xi8, #rhs>,
          tensor<16x16xf32, #mma>,
          tensor<16x4xi8, #scale>,
          tensor<16x4xi8, #scale>
          -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// Completion of a scaled result goes through `mfma_commit`: the scaled
// 16x16x128 drain is 20 wait states (`s_nop 15` + `s_nop 3`).
// CHECK-LABEL: llvm.func @scaled_commit_fp8_step
// CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4
// CHECK-SAME: fp8_e4m3, fp8_e5m2
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 15\0As_nop 3", "=a,0,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 128], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scaled_commit_fp8_step(
      %a: tensor<16x128xi8, #lhs>,
      %b: tensor<128x16xi8, #rhs>,
      %sa: tensor<16x4xi8, #scale>,
      %sb: tensor<16x4xi8, #scale>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e4m3", "e5m2" resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x128xi8, #lhs>,
          tensor<128x16xi8, #rhs>,
          tensor<16x16xf32, #mma>,
          tensor<16x4xi8, #scale>,
          tensor<16x4xi8, #scale>
          -> tensor<16x16xf32, #mma>
    %committed = amdg.mfma_commit %result : tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// Two native steps (logical K 256): one intrinsic per step, scales held in
// the MFMA scale layout whose register jumps by one step (4 scale groups).
// CHECK-LABEL: llvm.func @scaled_multistep_fp8
// CHECK-COUNT-2: rocdl.mfma.scale.f32.16x16x128.f8f6f4

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 128], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#scale_ms = #ttg.linear<{register = [[0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scaled_multistep_fp8(
      %a: tensor<16x256xi8, #lhs>,
      %b: tensor<256x16xi8, #rhs>,
      %sa: tensor<16x8xi8, #scale_ms>,
      %sb: tensor<16x8xi8, #scale_ms>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e4m3", "e5m2" resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x256xi8, #lhs>,
          tensor<256x16xi8, #rhs>,
          tensor<16x16xf32, #mma>,
          tensor<16x8xi8, #scale_ms>,
          tensor<16x8xi8, #scale_ms>
          -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// Multi-wave CTA ([2, 2] warps): operand ownership is replication-aware
// (A replicated over N warps, B over M warps); each wave still emits one
// native step for its own 16x16 tile.
// CHECK-LABEL: llvm.func @scaled_multiwave_fp8
// CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4
// CHECK-SAME: fp8_e4m3, fp8_e5m2

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 128], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#scale_a_mw = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[0, 0], [16, 0]], block = []}>
#scale_b_mw = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scaled_multiwave_fp8(
      %a: tensor<32x128xi8, #lhs>,
      %b: tensor<128x32xi8, #rhs>,
      %sa: tensor<32x4xi8, #scale_a_mw>,
      %sb: tensor<32x4xi8, #scale_b_mw>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e4m3", "e5m2" resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<32x128xi8, #lhs>,
          tensor<128x32xi8, #rhs>,
          tensor<32x32xf32, #mma>,
          tensor<32x4xi8, #scale_a_mw>,
          tensor<32x4xi8, #scale_b_mw>
          -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// Scaled 32x32x64 commit: 36 wait states (`s_nop 15` twice + `s_nop 3`).
// CHECK-LABEL: llvm.func @scaled_commit_32x32x64
// CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4
// CHECK-SAME: fp8_e4m3, fp8_e5m2
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 15\0As_nop 15\0As_nop 3", "=a,0,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 64], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#scale32 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scaled_commit_32x32x64(
      %a: tensor<32x64xi8, #lhs>,
      %b: tensor<64x32xi8, #rhs>,
      %sa: tensor<32x2xi8, #scale32>,
      %sb: tensor<32x2xi8, #scale32>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e4m3", "e5m2" resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<32x64xi8, #lhs>,
          tensor<64x32xi8, #rhs>,
          tensor<32x32xf32, #mma>,
          tensor<32x2xi8, #scale32>,
          tensor<32x2xi8, #scale32>
          -> tensor<32x32xf32, #mma>
    %committed = amdg.mfma_commit %result : tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// Same per-thread scale count as the multi-step case above, but registers
// advance within a step instead of across steps: rejected, since lowering
// would apply each E8M0 byte to the wrong K step.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 128], isTransposed = false}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#scale_bad = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#scale_ms = #ttg.linear<{register = [[0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @bad_scale_order(
      %a: tensor<16x256xi8, #lhs>,
      %b: tensor<256x16xi8, #rhs>,
      %sa: tensor<16x8xi8, #scale_bad>,
      %sb: tensor<16x8xi8, #scale_ms>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // expected-error @+1 {{a_scale layout does not match the scaled MFMA scale layout}}
    %result = amdg.scheduled_mfma_scaled %a, %b, %acc scales %sa, %sb
        formats "e4m3", "e5m2" resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x256xi8, #lhs>,
          tensor<256x16xi8, #rhs>,
          tensor<16x16xf32, #mma>,
          tensor<16x8xi8, #scale_bad>,
          tensor<16x8xi8, #scale_ms>
          -> tensor<16x16xf32, #mma>
    tt.return
  }
}
