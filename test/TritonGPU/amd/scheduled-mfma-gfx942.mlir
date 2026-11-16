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
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 10\0Av_add_f32 $0, $1, 1.0"
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
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 10\0Av_add_f32 $0, $1, 1.0"
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

  // Promoting a private store/load must not hide the opaque destination read.
  // CHECK-LABEL: llvm.func @opaque_consumer_private_reload
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 10", "=v,0"
  // CHECK: llvm.store
  // CHECK: llvm.load
  // CHECK: llvm.inline_asm has_side_effects "s_nop 10\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_private_reload(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %pack = builtin.unrealized_conversion_cast %result : tensor<16x16xf32, #mma> to !llvm.struct<(f32, f32, f32, f32)>
    %x = llvm.extractvalue %pack[0] : !llvm.struct<(f32, f32, f32, f32)>
    %one = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %one x f32 : (i32) -> !llvm.ptr<5>
    llvm.store %x, %slot : f32, !llvm.ptr<5>
    %reloaded = llvm.load %slot : !llvm.ptr<5> -> f32
    %out = llvm.inline_asm has_side_effects "v_add_f32 $0, $1, 1.0", "=v,v" %reloaded : (f32) -> f32
    tt.return
  }

}

// -----

// A 32x32 destination needs more than the maximum single s_nop delay.
// CHECK-LABEL: llvm.func @opaque_consumer_32x32
// CHECK: rocdl.mfma.f32.32x32x8bf16.1k
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 15\0As_nop 2", "=v,0"
// CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 15\0As_nop 2\0Av_add_f32 $0, $1, 1.0"
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

// -----

// Opaque writers can reuse or explicitly clobber a dead MFMA destination even
// without an SSA use of that destination. The wait belongs inside each asm so
// it moves with the writer if LLVM schedules or rematerializes the asm.
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @opaque_writer_dead_result
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects "s_nop 10\0Av_mov_b32 $0, 7", "=v"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.return
  tt.func public @opaque_writer_dead_result(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %writer = llvm.inline_asm has_side_effects "v_mov_b32 $0, 7", "=v" : () -> i32
    tt.return
  }

  // Pure asm still writes its output register and may move across an MFMA.
  // CHECK-LABEL: llvm.func @opaque_writer_pure
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK: llvm.inline_asm "s_nop 10\0Av_mov_b32 $0, 7", "=v"
  // CHECK: llvm.store
  // CHECK: llvm.return
  tt.func public @opaque_writer_pure(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>, %dst: !tt.ptr<i32>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %writer = llvm.inline_asm "v_mov_b32 $0, 7", "=v" : () -> i32
    %ptr = builtin.unrealized_conversion_cast %dst : !tt.ptr<i32> to !llvm.ptr<1>
    llvm.store %writer, %ptr : i32, !llvm.ptr<1>
    tt.return
  }

  // An asm with no operands or results can still clobber a physical VGPR.
  // CHECK-LABEL: llvm.func @opaque_writer_vgpr_clobber
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects "s_nop 10\0Av_mov_b32 v0, 7", "~{v0}"
  // CHECK: llvm.return
  tt.func public @opaque_writer_vgpr_clobber(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    llvm.inline_asm has_side_effects "v_mov_b32 v0, 7", "~{v0}" : () -> ()
    tt.return
  }

  // A commit on one incoming path cannot protect an unrelated writer after
  // the merge. The compiler commit remains unchanged and the writer is guarded.
  // CHECK-LABEL: llvm.func @opaque_writer_bypassed_commit
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK: llvm.cond_br
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 10", "=a,0,~{memory}"
  // CHECK: llvm.br
  // CHECK: llvm.inline_asm has_side_effects "s_nop 10\0Av_mov_b32 $0, 7", "=v"
  // CHECK: llvm.return
  tt.func public @opaque_writer_bypassed_commit(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    cf.cond_br %condition, ^commit, ^write
  ^commit:
    %ready = amdg.mfma_commit %result : tensor<16x16xf32, #mma>
    cf.br ^write
  ^write:
    %writer = llvm.inline_asm has_side_effects "v_mov_b32 $0, 7", "=v" : () -> i32
    tt.return
  }

  // The writer precedes the MFMA textually but follows it on the backedge.
  // CHECK-LABEL: llvm.func @opaque_writer_earlier_backedge
  // CHECK: llvm.br
  // CHECK: llvm.inline_asm has_side_effects "s_nop 10\0Av_mov_b32 $0, 7", "=v"
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK: llvm.cond_br
  // CHECK: llvm.return
  tt.func public @opaque_writer_earlier_backedge(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    cf.br ^loop
  ^loop:
    %writer = llvm.inline_asm has_side_effects "v_mov_b32 $0, 7", "=v" : () -> i32
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    cf.cond_br %condition, ^loop, ^exit
  ^exit:
    tt.return
  }

  // Native readers and writers remain visible to LLVM's hazard recognizer.
  // CHECK-LABEL: llvm.func @opaque_writer_native_only
  // CHECK-NOT: "s_nop
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK-NOT: "s_nop
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.fadd
  // CHECK-NOT: "s_nop
  // CHECK: llvm.store
  // CHECK-NOT: "s_nop
  // CHECK: llvm.return
  tt.func public @opaque_writer_native_only(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>, %dst: !tt.ptr<f32>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %pack = builtin.unrealized_conversion_cast %result : tensor<16x16xf32, #mma> to !llvm.struct<(f32, f32, f32, f32)>
    %x = llvm.extractvalue %pack[0] : !llvm.struct<(f32, f32, f32, f32)>
    %one = llvm.mlir.constant(1.0 : f32) : f32
    %native = llvm.fadd %x, %one : f32
    %ptr = builtin.unrealized_conversion_cast %dst : !tt.ptr<f32> to !llvm.ptr<1>
    llvm.store %native, %ptr : f32, !llvm.ptr<1>
    tt.return
  }

  // A persistent MFMA in a sibling function must not affect this writer.
  // CHECK-LABEL: llvm.func @opaque_writer_without_mfma
  // CHECK-NOT: "s_nop
  // CHECK: llvm.inline_asm has_side_effects "v_mov_b32 $0, 7", "=v"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.return
  tt.func public @opaque_writer_without_mfma() {
    %writer = llvm.inline_asm has_side_effects "v_mov_b32 $0, 7", "=v" : () -> i32
    tt.return
  }
}

// -----

// Pending register writes must complete before a real call or helper return:
// the next function can contain an opaque writer with no shared SSA values.
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // The caller supplies completion; this writer-only helper stays unguarded.
  // CHECK-LABEL: llvm.func {{.*}}@opaque_writer_helper
  // CHECK-NOT: "s_nop
  // CHECK: llvm.inline_asm has_side_effects "v_mov_b32 v0, 7", "~{v0}"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.return
  tt.func private @opaque_writer_helper() attributes {noinline = true} {
    llvm.inline_asm has_side_effects "v_mov_b32 v0, 7", "~{v0}" : () -> ()
    tt.return
  }

  // CHECK-LABEL: llvm.func @opaque_writer_caller_dead_result
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 10", "=v,0"
  // CHECK: llvm.call @opaque_writer_helper
  // CHECK: llvm.return
  tt.func public @opaque_writer_caller_dead_result(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.call @opaque_writer_helper() : () -> ()
    tt.return
  }

  // A helper must drain even when its MFMA result is not returned or used.
  // CHECK-LABEL: llvm.func {{.*}}@opaque_writer_mfma_helper
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 10", "=v,0"
  // CHECK: llvm.return
  tt.func private @opaque_writer_mfma_helper() attributes {noinline = true} {
    %a = arith.constant dense<1.0> : tensor<16x16xbf16, #lhs>
    %b = arith.constant dense<1.0> : tensor<16x16xbf16, #rhs>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // The helper completed before returning, so the caller needs no asm prefix.
  // CHECK-LABEL: llvm.func @opaque_writer_after_helper_return
  // CHECK-NOT: "s_nop
  // CHECK: llvm.call @opaque_writer_mfma_helper
  // CHECK-NOT: "s_nop
  // CHECK: llvm.inline_asm has_side_effects "v_mov_b32 v0, 7", "~{v0}"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.return
  tt.func public @opaque_writer_after_helper_return() {
    tt.call @opaque_writer_mfma_helper() : () -> ()
    llvm.inline_asm has_side_effects "v_mov_b32 v0, 7", "~{v0}" : () -> ()
    tt.return
  }

  // A pure external call can still become an opaque register writer.
  // extern_elementwise creates its LLVM declaration during conversion.
  // CHECK-LABEL: llvm.func @opaque_writer_external_call
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 10", "=v,0"
  // CHECK: llvm.call @opaque_writer_external
  // CHECK: llvm.store
  // CHECK: llvm.return
  tt.func public @opaque_writer_external_call(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>, %x: f32, %dst: !tt.ptr<f32>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %value = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "opaque_writer_external"} : (f32) -> f32
    %ptr = builtin.unrealized_conversion_cast %dst : !tt.ptr<f32> to !llvm.ptr<1>
    llvm.store %value, %ptr : f32, !llvm.ptr<1>
    tt.return
  }

  // Treat unrecognized generic intrinsics conservatively as call boundaries.
  // CHECK-LABEL: llvm.func @opaque_writer_generic_intrinsic
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 10", "=v,0"
  // CHECK: llvm.call_intrinsic "llvm.sin.f32"
  // CHECK: llvm.store
  // CHECK: llvm.return
  tt.func public @opaque_writer_generic_intrinsic(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>, %x: f32, %dst: !tt.ptr<f32>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %value = llvm.call_intrinsic "llvm.sin.f32"(%x) : (f32) -> f32
    %ptr = builtin.unrealized_conversion_cast %dst : !tt.ptr<f32> to !llvm.ptr<1>
    llvm.store %value, %ptr : f32, !llvm.ptr<1>
    tt.return
  }

  // CoroEarly turns this dedicated op into an indirect call. It bypasses
  // CallIntrinsicOp, so it must independently require the completion boundary.
  // CHECK-LABEL: llvm.func @opaque_writer_coro_resume
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 10", "=v,0"
  // CHECK: llvm.intr.coro.resume %{{.*}} : !llvm.ptr
  // CHECK: llvm.return
  tt.func public @opaque_writer_coro_resume(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>, %handle: !llvm.ptr) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    llvm.intr.coro.resume %handle : !llvm.ptr
    tt.return
  }

  // exp2 lowers to a direct LLVM call naming a native instruction. Its
  // independent output must not force a full drain of a dead MFMA result.
  // CHECK-LABEL: llvm.func @opaque_writer_native_exp2
  // CHECK-NOT: "s_nop
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK-NOT: "s_nop
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.call @llvm{{.*}}exp2.f32
  // CHECK-NOT: "s_nop
  // CHECK: llvm.store
  // CHECK-NOT: "s_nop
  // CHECK: llvm.return
  tt.func public @opaque_writer_native_exp2(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>, %x: f32, %dst: !tt.ptr<f32>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %value = math.exp2 %x : f32
    %ptr = builtin.unrealized_conversion_cast %dst : !tt.ptr<f32> to !llvm.ptr<1>
    llvm.store %value, %ptr : f32, !llvm.ptr<1>
    tt.return
  }

  // This exact AMDGPU intrinsic remains a native instruction. Its unrelated
  // register output does not require an opaque-call completion boundary.
  // CHECK-LABEL: llvm.func @opaque_writer_native_perm_intrinsic
  // CHECK-NOT: "s_nop
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k
  // CHECK-NOT: "s_nop
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.perm"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.store
  // CHECK-NOT: "s_nop
  // CHECK: llvm.return
  tt.func public @opaque_writer_native_perm_intrinsic(
      %a: tensor<16x16xbf16, #lhs>, %b: tensor<16x16xbf16, #rhs>,
      %x: i32, %y: i32, %selector: i32, %dst: !tt.ptr<i32>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x16xbf16, #lhs>, tensor<16x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %value = llvm.call_intrinsic "llvm.amdgcn.perm"(%x, %y, %selector) : (i32, i32, i32) -> i32
    %ptr = builtin.unrealized_conversion_cast %dst : !tt.ptr<i32> to !llvm.ptr<1>
    llvm.store %value, %ptr : i32, !llvm.ptr<1>
    tt.return
  }
}

// -----

// kWidth=8 requires two native K fragments on CDNA3.
// CHECK-LABEL: llvm.func @scheduled_mfma_transient_bf16_16x16x16_kwidth8
// CHECK-COUNT-2: rocdl.mfma.f32.16x16x16bf16.1k
// CHECK-NOT: rocdl.mfma
// CHECK-NOT: amdg.
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scheduled_mfma_transient_bf16_16x16x16_kwidth8(
      %a: tensor<16x32xbf16, #lhs>,
      %b: tensor<32x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %committed, %preserved = amdg.mfma_commit %result, %b
        : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
    tt.return
  }
}



// -----

// CHECK-LABEL: llvm.func @scheduled_mfma_persistent_f16_32x32x8_kwidth8
// CHECK-COUNT-2: rocdl.mfma.f32.32x32x8f16
// CHECK-NOT: rocdl.mfma
// CHECK: llvm.inline_asm has_side_effects{{.*}} "", "=v,0"
// CHECK-NOT: "v_mfma
// CHECK-NOT: amdg.
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scheduled_mfma_persistent_f16_32x32x8_kwidth8(
      %a: tensor<32x16xf16, #lhs>,
      %b: tensor<16x32xf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xf16, #lhs>, tensor<16x32xf16, #rhs>,
          tensor<32x32xf32, #mma> -> tensor<32x32xf32, #mma>
    tt.return
  }
}



// -----

// Rank-three batches remain wave-local on CDNA3 as well.
// CHECK-LABEL: llvm.func @wave_batched_transient_gfx942
// CHECK: rocdl.mfma.f32.16x16x16bf16.1k
// CHECK: llvm.inline_asm has_side_effects{{.*}}s_nop 10
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @wave_batched_transient_gfx942(%a: tensor<2x16x16xbf16, #lhs>, %b: tensor<2x16x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<2x16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "transient" register_class "auto" initialize true
        : tensor<2x16x16xbf16, #lhs>, tensor<2x16x16xbf16, #rhs>, tensor<2x16x16xf32, #mma>
          -> tensor<2x16x16xf32, #mma>
    %committed, %preserved = amdg.mfma_commit %result, %b
        : tensor<2x16x16xf32, #mma>, tensor<2x16x16xbf16, #rhs>
    tt.return
  }
}
