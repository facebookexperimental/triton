// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942" --verify-diagnostics | FileCheck %s

// Lowering cases come first, then the accumulator register class refusals.
// The target/encoding mismatch lives in scheduled-mfma-target-mismatch-gfx942
// instead, because it also asserts a nonzero exit status, which one file
// cannot attribute to a particular case.

// CHECK-LABEL: llvm.func @scheduled_mfma_transient_bf16_16x16x16
// CHECK: llvm.bitcast %{{.*}} : vector<4xbf16> to vector<4xi16>
// CHECK: rocdl.mfma.f32.16x16x16bf16.1k
// CHECK-SAME: (vector<4xi16>, vector<4xi16>, vector<4xf32>) -> vector<4xf32>
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 10"
// CHECK-NOT: amdg.

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scheduled_mfma_transient_bf16_16x16x16(
      %a: tensor<16x16xbf16, #lhs>,
      %b: tensor<16x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "transient"
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

// CHECK-LABEL: llvm.func @scheduled_mfma_persistent_f16_32x32x8
// CHECK: rocdl.mfma.f32.32x32x8f16
// CHECK-SAME: (vector<4xf16>, vector<4xf16>, vector<16xf32>) -> vector<16xf32>
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK-NOT: "v_mfma
// CHECK-NOT: "s_nop
// CHECK-NOT: amdg.
// CHECK: llvm.return

#mma32 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = true}>
#lhs32 = #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 4}>
#rhs32 = #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scheduled_mfma_persistent_f16_32x32x8(
      %a: tensor<32x8xf16, #lhs32>,
      %b: tensor<8x32xf16, #rhs32>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma32>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x8xf16, #lhs32>,
          tensor<8x32xf16, #rhs32>,
          tensor<32x32xf32, #mma32>
          -> tensor<32x32xf32, #mma32>
    tt.return
  }
}

// -----

// No ttg.target: a v3 encoding must still verify and lower on its own.

// CHECK-LABEL: llvm.func @scheduled_mfma_persistent_f16_16x16x16_untargeted
// CHECK: rocdl.mfma.f32.16x16x16f16
// CHECK-SAME: (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK-NOT: "v_mfma
// CHECK-NOT: "s_nop
// CHECK-NOT: amdg.
// CHECK: llvm.return

#mma16p = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#lhs16p = #ttg.dot_op<{opIdx = 0, parent = #mma16p, kWidth = 4}>
#rhs16p = #ttg.dot_op<{opIdx = 1, parent = #mma16p, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scheduled_mfma_persistent_f16_16x16x16_untargeted(
      %a: tensor<16x16xf16, #lhs16p>,
      %b: tensor<16x16xf16, #rhs16p>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma16p>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xf16, #lhs16p>,
          tensor<16x16xf16, #rhs16p>,
          tensor<16x16xf32, #mma16p>
          -> tensor<16x16xf32, #mma16p>
    tt.return
  }
}

// -----

// The explicit VGPR class is how a persistent accumulator is carried on CDNA3,
// and it lowers cleanly across a commit that also carries a live dot operand.
// CHECK-LABEL: llvm.func @vgpr_accumulator_with_live_operand
// CHECK: rocdl.mfma.f32.16x16x16bf16.1k
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 10", "=v,=a,0,1,~{memory}"
// CHECK: llvm.return

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

// -----

// CDNA3 retains its explicit VGPR-only accumulator contract. Native arithmetic
// does not broaden the accepted register classes without separate validation.
// The rejection applies with or without an explicit commit boundary.
//
// The commit-time AGPR/live-operand interaction is still covered on CDNA4, in
// scheduled-mfma-gfx950.mlir, where the explicit class is legal.

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

// Opaque assembly needs completion even through value forwarding. Native MFMA
// arithmetic and tied VGPR placement remain visible to the backend.
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @opaque_consumer_direct
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 10", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "v_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_direct(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %result : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // CHECK-LABEL: llvm.func @opaque_consumer_full_commit
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 10", "=a,0,~{memory}"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "v_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_full_commit(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %ready = amdg.mfma_commit %result : tensor<16x16xf32, #mma>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %ready : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

}

// -----

// A 32x32 destination needs more than the maximum single s_nop delay.
// CHECK-LABEL: llvm.func @opaque_consumer_32x32
// CHECK: rocdl.mfma.f32.32x32x8bf16.1k
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 15\0As_nop 2", "=v,0"
// CHECK: llvm.inline_asm has_side_effects {{.*}} "v_add_f32 $0, $1, 1.0"
// CHECK: llvm.return
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @opaque_consumer_32x32(
      %a: tensor<32x8xbf16, #lhs>, %b: tensor<8x32xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<32x32xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x8xbf16, #lhs>, tensor<8x32xbf16, #rhs>,
          tensor<32x32xf32, #mma> -> tensor<32x32xf32, #mma>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %result : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #mma>
    tt.return
  }
}
