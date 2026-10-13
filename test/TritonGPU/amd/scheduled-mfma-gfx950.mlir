// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --verify-diagnostics | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --canonicalize --cse --verify-diagnostics | FileCheck %s --check-prefix=OPERAND-CSE
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --canonicalize --cse --triton-amdgpu-finalize-scheduled-mfma-operands --triton-amdgpu-finalize-scheduled-mfma-operands --verify-diagnostics | FileCheck %s --check-prefix=OPERAND-FINAL --implicit-check-not=ttg.amdg.scheduled_mfma_operand_pin
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --canonicalize --cse --triton-amdgpu-finalize-scheduled-mfma-operands --verify-diagnostics | FileCheck %s --check-prefix=LOAD-ORIGIN --implicit-check-not=ttg.amdg.scheduled_mfma_operand_pin

// On CDNA4 `auto` resolves a persistent accumulator to AGPRs, so the unsafe
// commit is reachable through the default path with no explicit register class
// in the source. This is the case an override-only guard would miss.

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @auto_persistent_accumulator_with_live_operand(
      %a: tensor<16x32xbf16, #lhs>,
      %b: tensor<32x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>,
          tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    // The commit op is illegal after this pass, so refusing it fails the pass.
    // expected-error @+2 {{input 0 is an AGPR-resident accumulator committed alongside a live dot operand}}
    // expected-error @+1 {{failed to legalize operation 'amdg.mfma_commit'}}
    %committed, %preserved = amdg.mfma_commit %result, %b
        : tensor<16x16xf32, #mma>,
          tensor<32x16xbf16, #rhs>
    tt.return
  }
}

// -----

// Legacy hazard attributes cannot change the native arithmetic lowering or
// remove its persistent result pin.
//
// CHECK-LABEL: llvm.func @forged_hazard_attributes
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=a,0"
// CHECK-NOT: "v_mfma
// CHECK-NOT: "s_nop
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @forged_hazard_attributes(
      %a: tensor<16x32xbf16, #lhs>,
      %b: tensor<32x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "auto" initialize true {
          ttg.amdg.scheduled_mfma.defer_result_drain,
          ttg.amdg.scheduled_mfma.repair_hazards_after_ra
        }
        : tensor<16x32xbf16, #lhs>,
          tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// Pinning the accumulator to VGPRs is the documented remedy and must lower.
// Its live-operand commit takes the persistent result's full required wait,
// even when the committed outputs are unused.
// CHECK-LABEL: llvm.func @vgpr_pinned_accumulator_with_live_operand
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 11", "=v,=a,0,1,~{memory}"
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @vgpr_pinned_accumulator_with_live_operand(
      %a: tensor<16x32xbf16, #lhs>,
      %b: tensor<32x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>,
          tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %committed, %preserved = amdg.mfma_commit %result, %b
        : tensor<16x16xf32, #mma>,
          tensor<32x16xbf16, #rhs>
    tt.return
  }
}

// -----

// CDNA4 keeps the explicit AGPR class so two persistent accumulator sets may
// deliberately occupy complementary register files. CDNA3 retains its explicit
// VGPR-only contract (see invalid.mlir).
//
// CHECK-LABEL: llvm.func @explicit_agpr_accumulator
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=a,0"
// CHECK-NOT: "v_mfma
// CHECK-NOT: "s_nop
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @explicit_agpr_accumulator(
      %a: tensor<16x32xbf16, #lhs>,
      %b: tensor<32x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<16x32xbf16, #lhs>,
          tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// A block argument may merge a scheduled accumulator with an unrelated value.
// Native arithmetic keeps both sides visible to LLVM without a chain proof.
//
// CHECK-LABEL: llvm.func @mixed_lineage_block_argument
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=a,0"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mixed_lineage_block_argument(
      %a: tensor<16x32xbf16, #lhs>,
      %b: tensor<32x16xbf16, #rhs>,
      %condition: i1) {
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %root = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>,
          tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    cf.cond_br %condition, ^bb1(%root : tensor<16x16xf32, #mma>), ^bb2
  ^bb1(%chain: tensor<16x16xf32, #mma>):
    cf.br ^bb3(%chain : tensor<16x16xf32, #mma>)
  ^bb2:
    cf.br ^bb3(%zero : tensor<16x16xf32, #mma>)
  ^bb3(%merged: tensor<16x16xf32, #mma>):
    %result = amdg.scheduled_mfma %a, %b, %merged
        resident "none" accumulator "persistent"
        register_class "auto" initialize false
        : tensor<16x32xbf16, #lhs>,
          tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %committed = amdg.mfma_commit %result
        : tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// Opaque assembly needs completion even through value forwarding. Native MFMA
// arithmetic and tied VGPR placement remain visible to the backend.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @opaque_consumer_direct
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 11", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_direct(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %result : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // CHECK-LABEL: llvm.func @opaque_consumer_full_commit
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 11", "=a,0,~{memory}"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_full_commit(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %ready = amdg.mfma_commit %result : tensor<16x16xf32, #mma>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %ready : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // CHECK-LABEL: llvm.func @opaque_consumer_short_commit
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 11", "=v,=a,0,1,~{memory}"
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_short_commit(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %ready, %live = amdg.mfma_commit %result, %b : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %ready : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // Strengthening the commit protects only its incoming path. The other
  // successor carries the undrained result to the same opaque consumer.
  // CHECK-LABEL: llvm.func @opaque_consumer_bypassed_short_commit
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 11", "=v,0"
  // CHECK: llvm.cond_br
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 11", "=v,=a,0,1,~{memory}"
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_bypassed_short_commit(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    cf.cond_br %condition, ^commit(%result : tensor<16x16xf32, #mma>), ^bypass(%result : tensor<16x16xf32, #mma>)
  ^commit(%to_commit: tensor<16x16xf32, #mma>):
    %ready, %live = amdg.mfma_commit %to_commit, %b : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
    cf.br ^join(%ready : tensor<16x16xf32, #mma>)
  ^bypass(%undrained: tensor<16x16xf32, #mma>):
    cf.br ^join(%undrained : tensor<16x16xf32, #mma>)
  ^join(%merged: tensor<16x16xf32, #mma>):
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %merged : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // Following a loop-carried identity must terminate and move completion to
  // the shared exit commit, without adding a producer guard on the backedge.
  // CHECK-LABEL: llvm.func @opaque_consumer_loop_short_commit
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "", "=v,0"
  // CHECK-NOT: "s_nop 11", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 11", "=v,=a,0,1,~{memory}"
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_loop_short_commit(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    cf.br ^loop(%result : tensor<16x16xf32, #mma>)
  ^loop(%carried: tensor<16x16xf32, #mma>):
    %identity = arith.addf %carried, %zero : tensor<16x16xf32, #mma>
    cf.cond_br %condition, ^loop(%identity : tensor<16x16xf32, #mma>), ^exit(%identity : tensor<16x16xf32, #mma>)
  ^exit(%final: tensor<16x16xf32, #mma>):
    %ready, %live = amdg.mfma_commit %final, %b : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %ready : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // A transient-only live-operand handoff retains its six-state contract.
  // CHECK-LABEL: llvm.func @transient_live_operand_short_commit
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK-NOT: "s_nop 11"
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 5", "=v,=a,0,1,~{memory}"
  // CHECK-NOT: "s_nop 11"
  // CHECK: llvm.return
  tt.func public @transient_live_operand_short_commit(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %ready, %live = amdg.mfma_commit %result, %b : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
    tt.return
  }

  // CHECK-LABEL: llvm.func @opaque_consumer_cfg_backedge
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 11", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_cfg_backedge(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    cf.br ^loop(%result : tensor<16x16xf32, #mma>)
  ^loop(%carried: tensor<16x16xf32, #mma>):
    %pin = tt.elementwise_inline_asm ""
        {constraints = "=v,0", packed_element = 1 : i32, pure = false}
        %carried : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %identity = arith.addf %pin, %zero : tensor<16x16xf32, #mma>
    cf.cond_br %condition, ^loop(%identity : tensor<16x16xf32, #mma>), ^exit(%identity : tensor<16x16xf32, #mma>)
  ^exit(%forwarded: tensor<16x16xf32, #mma>):
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %forwarded : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // CHECK-LABEL: llvm.func @opaque_consumer_predicate_yield
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 11", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_predicate_yield(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %forwarded = ttg.warp_predicate %condition (%zero) {
      %result = amdg.scheduled_mfma %a, %b, %zero
          resident "none" accumulator "persistent"
          register_class "vgpr" initialize true
          : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
            tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
      ttg.predicate_yield %result : tensor<16x16xf32, #mma>
    } {wave_uniform} : (i1, tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %forwarded : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // CHECK-LABEL: llvm.func @opaque_consumer_predicate_init
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 11", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_predicate_init(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %forwarded = ttg.warp_predicate %condition (%result) {
      ttg.predicate_yield %zero : tensor<16x16xf32, #mma>
    } : (i1, tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %forwarded : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // A private store/load can disappear during LLVM memory promotion.
  // CHECK-LABEL: llvm.func @opaque_consumer_private_reload
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11", "=v,0"
  // CHECK: llvm.store
  // CHECK: llvm.load
  // CHECK: llvm.inline_asm has_side_effects "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_private_reload(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
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

  // A copy reads the stored bytes even though the final load uses another slot.
  // CHECK-LABEL: llvm.func @opaque_consumer_private_memcpy
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11", "=v,0"
  // CHECK: llvm.store
  // CHECK: llvm.intr.memcpy
  // CHECK: llvm.load
  // CHECK: llvm.inline_asm has_side_effects "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_private_memcpy(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %pack = builtin.unrealized_conversion_cast %result : tensor<16x16xf32, #mma> to !llvm.struct<(f32, f32, f32, f32)>
    %x = llvm.extractvalue %pack[0] : !llvm.struct<(f32, f32, f32, f32)>
    %one = llvm.mlir.constant(1 : i32) : i32
    %bytes = llvm.mlir.constant(4 : i64) : i64
    %src = llvm.alloca %one x f32 : (i32) -> !llvm.ptr<5>
    %dst = llvm.alloca %one x f32 : (i32) -> !llvm.ptr<5>
    llvm.store %x, %src : f32, !llvm.ptr<5>
    "llvm.intr.memcpy"(%dst, %src, %bytes) {isVolatile = false} : (!llvm.ptr<5>, !llvm.ptr<5>, i64) -> ()
    %reloaded = llvm.load %dst : !llvm.ptr<5> -> f32
    %out = llvm.inline_asm has_side_effects "v_add_f32 $0, $1, 1.0", "=v,v" %reloaded : (f32) -> f32
    tt.return
  }

  // An unmodeled writer must not terminate the destination's use chain.
  // CHECK-LABEL: llvm.func @opaque_consumer_private_masked_store
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11", "=v,0"
  // CHECK: llvm.intr.masked.store
  // CHECK: llvm.load
  // CHECK: llvm.inline_asm has_side_effects "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_private_masked_store(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %pack = builtin.unrealized_conversion_cast %result : tensor<16x16xf32, #mma> to !llvm.struct<(f32, f32, f32, f32)>
    %x = llvm.extractvalue %pack[0] : !llvm.struct<(f32, f32, f32, f32)>
    %one = llvm.mlir.constant(1 : i32) : i32
    %index = llvm.mlir.constant(0 : i32) : i32
    %mask = llvm.mlir.constant(dense<true> : vector<1xi1>) : vector<1xi1>
    %undef = llvm.mlir.undef : vector<1xf32>
    %vector = llvm.insertelement %x, %undef[%index : i32] : vector<1xf32>
    %slot = llvm.alloca %one x f32 : (i32) -> !llvm.ptr<5>
    llvm.intr.masked.store %vector, %slot, %mask {alignment = 4 : i32} : vector<1xf32>, vector<1xi1> into !llvm.ptr<5>
    %reloaded = llvm.load %slot : !llvm.ptr<5> -> f32
    %out = llvm.inline_asm has_side_effects "v_add_f32 $0, $1, 1.0", "=v,v" %reloaded : (f32) -> f32
    tt.return
  }

  // The load precedes the store in this block, but can read it next iteration.
  // CHECK-LABEL: llvm.func @opaque_consumer_private_backedge
  // CHECK: llvm.load
  // CHECK: llvm.inline_asm has_side_effects "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11", "=v,0"
  // CHECK: llvm.store
  // CHECK: llvm.cond_br
  // CHECK: llvm.return
  tt.func public @opaque_consumer_private_backedge(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %one = llvm.mlir.constant(1 : i32) : i32
    %initial = llvm.mlir.constant(0.0 : f32) : f32
    %slot = llvm.alloca %one x f32 : (i32) -> !llvm.ptr<5>
    llvm.store %initial, %slot : f32, !llvm.ptr<5>
    cf.br ^loop
  ^loop:
    %reloaded = llvm.load %slot : !llvm.ptr<5> -> f32
    %out = llvm.inline_asm has_side_effects "v_add_f32 $0, $1, 1.0", "=v,v" %reloaded : (f32) -> f32
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %pack = builtin.unrealized_conversion_cast %result : tensor<16x16xf32, #mma> to !llvm.struct<(f32, f32, f32, f32)>
    %x = llvm.extractvalue %pack[0] : !llvm.struct<(f32, f32, f32, f32)>
    llvm.store %x, %slot : f32, !llvm.ptr<5>
    cf.cond_br %condition, ^loop, ^exit
  ^exit:
    tt.return
  }

  // Without a backedge, an earlier load cannot expose a later stored result.
  // CHECK-LABEL: llvm.func @opaque_consumer_private_earlier_load
  // CHECK: llvm.load
  // CHECK: llvm.inline_asm has_side_effects "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.store
  // CHECK-NOT: "s_nop
  // CHECK: llvm.return
  tt.func public @opaque_consumer_private_earlier_load(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %one = llvm.mlir.constant(1 : i32) : i32
    %initial = llvm.mlir.constant(0.0 : f32) : f32
    %slot = llvm.alloca %one x f32 : (i32) -> !llvm.ptr<5>
    llvm.store %initial, %slot : f32, !llvm.ptr<5>
    %reloaded = llvm.load %slot : !llvm.ptr<5> -> f32
    %out = llvm.inline_asm has_side_effects "v_add_f32 $0, $1, 1.0", "=v,v" %reloaded : (f32) -> f32
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %pack = builtin.unrealized_conversion_cast %result : tensor<16x16xf32, #mma> to !llvm.struct<(f32, f32, f32, f32)>
    %x = llvm.extractvalue %pack[0] : !llvm.struct<(f32, f32, f32, f32)>
    llvm.store %x, %slot : f32, !llvm.ptr<5>
    tt.return
  }

  // A full commit already completes the result before memory forwarding.
  // CHECK-LABEL: llvm.func @opaque_consumer_private_committed
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11", "=a,0,~{memory}"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.store
  // CHECK: llvm.load
  // CHECK: llvm.inline_asm has_side_effects "s_nop 11\0Av_add_f32 $0, $1, 1.0"
  // CHECK: llvm.return
  tt.func public @opaque_consumer_private_committed(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %ready = amdg.mfma_commit %result : tensor<16x16xf32, #mma>
    %pack = builtin.unrealized_conversion_cast %ready : tensor<16x16xf32, #mma> to !llvm.struct<(f32, f32, f32, f32)>
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
// CHECK: rocdl.mfma.f32.32x32x16.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 15\0As_nop 3", "=v,0"
// CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 15\0As_nop 3\0Av_add_f32 $0, $1, 1.0"
// CHECK: llvm.return
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @opaque_consumer_32x32(
      %a: tensor<32x16xbf16, #lhs>, %b: tensor<16x32xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<32x32xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>, tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma> -> tensor<32x32xf32, #mma>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %result : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// A shared live-operand commit must take the maximum of different persistent
// native shapes: 12 states for 16x16 and 20 states for 32x32. Both result pins
// stay empty and the boundary retains the VGPR accumulator constraints.
// CHECK-LABEL: llvm.func @mixed_shape_persistent_short_commit
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK: rocdl.mfma.f32.32x32x16.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 15\0As_nop 3", "=v,=v,=a,0,1,2,~{memory}"
// CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 15\0As_nop 3\0Av_add_f32 $0, $1, 1.0"
// CHECK: llvm.return
#mma16 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs16 = #ttg.dot_op<{opIdx = 0, parent = #mma16, kWidth = 8}>
#rhs16 = #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 8}>
#mma32 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs32 = #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 8}>
#rhs32 = #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mixed_shape_persistent_short_commit(
      %a16: tensor<16x32xbf16, #lhs16>, %b16: tensor<32x16xbf16, #rhs16>,
      %a32: tensor<32x16xbf16, #lhs32>, %b32: tensor<16x32xbf16, #rhs32>) {
    %zero16 = arith.constant dense<0.0> : tensor<16x16xf32, #mma16>
    %zero32 = arith.constant dense<0.0> : tensor<32x32xf32, #mma32>
    %small = amdg.scheduled_mfma %a16, %b16, %zero16
        resident "none" accumulator "persistent" register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs16>, tensor<32x16xbf16, #rhs16>, tensor<16x16xf32, #mma16>
          -> tensor<16x16xf32, #mma16>
    %large = amdg.scheduled_mfma %a32, %b32, %zero32
        resident "none" accumulator "persistent" register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs32>, tensor<16x32xbf16, #rhs32>, tensor<32x32xf32, #mma32>
          -> tensor<32x32xf32, #mma32>
    %ready16, %ready32, %live = amdg.mfma_commit %small, %large, %b16
        : tensor<16x16xf32, #mma16>, tensor<32x32xf32, #mma32>, tensor<32x16xbf16, #rhs16>
    %out16 = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %ready16 : tensor<16x16xf32, #mma16> -> tensor<16x16xf32, #mma16>
    %out32 = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %ready32 : tensor<32x32xf32, #mma32> -> tensor<32x32xf32, #mma32>
    tt.return
  }
}

// -----

// Opaque writers can reuse or explicitly clobber a dead MFMA destination even
// without an SSA use of that destination. The wait belongs inside each asm so
// it moves with the writer if LLVM schedules or rematerializes the asm.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @opaque_writer_dead_result
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects "s_nop 11\0Av_mov_b32 $0, 7", "=v"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.return
  tt.func public @opaque_writer_dead_result(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %writer = llvm.inline_asm has_side_effects "v_mov_b32 $0, 7", "=v" : () -> i32
    tt.return
  }

  // Pure asm still writes its output register and may move across an MFMA.
  // CHECK-LABEL: llvm.func @opaque_writer_pure
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK: llvm.inline_asm "s_nop 11\0Av_mov_b32 $0, 7", "=v"
  // CHECK: llvm.store
  // CHECK: llvm.return
  tt.func public @opaque_writer_pure(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %dst: !tt.ptr<i32>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %writer = llvm.inline_asm "v_mov_b32 $0, 7", "=v" : () -> i32
    %ptr = builtin.unrealized_conversion_cast %dst : !tt.ptr<i32> to !llvm.ptr<1>
    llvm.store %writer, %ptr : i32, !llvm.ptr<1>
    tt.return
  }

  // An asm with no operands or results can still clobber a physical VGPR.
  // CHECK-LABEL: llvm.func @opaque_writer_vgpr_clobber
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects "s_nop 11\0Av_mov_b32 v0, 7", "~{v0}"
  // CHECK: llvm.return
  tt.func public @opaque_writer_vgpr_clobber(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    llvm.inline_asm has_side_effects "v_mov_b32 v0, 7", "~{v0}" : () -> ()
    tt.return
  }

  // A commit on one incoming path cannot protect an unrelated writer after
  // the merge. The compiler commit remains unchanged and the writer is guarded.
  // CHECK-LABEL: llvm.func @opaque_writer_bypassed_commit
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK: llvm.cond_br
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11", "=a,0,~{memory}"
  // CHECK: llvm.br
  // CHECK: llvm.inline_asm has_side_effects "s_nop 11\0Av_mov_b32 $0, 7", "=v"
  // CHECK: llvm.return
  tt.func public @opaque_writer_bypassed_commit(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
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
  // CHECK: llvm.inline_asm has_side_effects "s_nop 11\0Av_mov_b32 $0, 7", "=v"
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK: llvm.cond_br
  // CHECK: llvm.return
  tt.func public @opaque_writer_earlier_backedge(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    cf.br ^loop
  ^loop:
    %writer = llvm.inline_asm has_side_effects "v_mov_b32 $0, 7", "=v" : () -> i32
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    cf.cond_br %condition, ^loop, ^exit
  ^exit:
    tt.return
  }

  // Native readers and writers remain visible to LLVM's hazard recognizer.
  // CHECK-LABEL: llvm.func @opaque_writer_native_only
  // CHECK-NOT: "s_nop
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK-NOT: "s_nop
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.fadd
  // CHECK-NOT: "s_nop
  // CHECK: llvm.store
  // CHECK-NOT: "s_nop
  // CHECK: llvm.return
  tt.func public @opaque_writer_native_only(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %dst: !tt.ptr<f32>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
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
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
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
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11", "=v,0"
  // CHECK: llvm.call @opaque_writer_helper
  // CHECK: llvm.return
  tt.func public @opaque_writer_caller_dead_result(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.call @opaque_writer_helper() : () -> ()
    tt.return
  }

  // A helper must drain even when its MFMA result is not returned or used.
  // CHECK-LABEL: llvm.func {{.*}}@opaque_writer_mfma_helper
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11", "=v,0"
  // CHECK: llvm.return
  tt.func private @opaque_writer_mfma_helper() attributes {noinline = true} {
    %a = arith.constant dense<1.0> : tensor<16x32xbf16, #lhs>
    %b = arith.constant dense<1.0> : tensor<32x16xbf16, #rhs>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
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
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11", "=v,0"
  // CHECK: llvm.call @opaque_writer_external
  // CHECK: llvm.store
  // CHECK: llvm.return
  tt.func public @opaque_writer_external_call(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %x: f32, %dst: !tt.ptr<f32>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %value = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "opaque_writer_external"} : (f32) -> f32
    %ptr = builtin.unrealized_conversion_cast %dst : !tt.ptr<f32> to !llvm.ptr<1>
    llvm.store %value, %ptr : f32, !llvm.ptr<1>
    tt.return
  }

  // Treat unrecognized generic intrinsics conservatively as call boundaries.
  // CHECK-LABEL: llvm.func @opaque_writer_generic_intrinsic
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11", "=v,0"
  // CHECK: llvm.call_intrinsic "llvm.sin.f32"
  // CHECK: llvm.store
  // CHECK: llvm.return
  tt.func public @opaque_writer_generic_intrinsic(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %x: f32, %dst: !tt.ptr<f32>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %value = llvm.call_intrinsic "llvm.sin.f32"(%x) : (f32) -> f32
    %ptr = builtin.unrealized_conversion_cast %dst : !tt.ptr<f32> to !llvm.ptr<1>
    llvm.store %value, %ptr : f32, !llvm.ptr<1>
    tt.return
  }

  // CoroEarly turns this dedicated op into an indirect call. It bypasses
  // CallIntrinsicOp, so it must independently require the completion boundary.
  // CHECK-LABEL: llvm.func @opaque_writer_coro_resume
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "s_nop 11", "=v,0"
  // CHECK: llvm.intr.coro.resume %{{.*}} : !llvm.ptr
  // CHECK: llvm.return
  tt.func public @opaque_writer_coro_resume(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %handle: !llvm.ptr) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    llvm.intr.coro.resume %handle : !llvm.ptr
    tt.return
  }

  // exp2 lowers to a direct LLVM call naming a native instruction. Its
  // independent output must not force a full drain of a dead MFMA result.
  // CHECK-LABEL: llvm.func @opaque_writer_native_exp2
  // CHECK-NOT: "s_nop
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK-NOT: "s_nop
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.call @llvm{{.*}}exp2.f32
  // CHECK-NOT: "s_nop
  // CHECK: llvm.store
  // CHECK-NOT: "s_nop
  // CHECK: llvm.return
  tt.func public @opaque_writer_native_exp2(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %x: f32, %dst: !tt.ptr<f32>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
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
  // CHECK: rocdl.mfma.f32.16x16x32.bf16
  // CHECK-NOT: "s_nop
  // CHECK: llvm.inline_asm has_side_effects {{.*}} "", "=v,0"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.perm"
  // CHECK-NOT: "s_nop
  // CHECK: llvm.store
  // CHECK-NOT: "s_nop
  // CHECK: llvm.return
  tt.func public @opaque_writer_native_perm_intrinsic(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>,
      %x: i32, %y: i32, %selector: i32, %dst: !tt.ptr<i32>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %unused = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %value = llvm.call_intrinsic "llvm.amdgcn.perm"(%x, %y, %selector) : (i32, i32, i32) -> i32
    %ptr = builtin.unrealized_conversion_cast %dst : !tt.ptr<i32> to !llvm.ptr<1>
    llvm.store %value, %ptr : i32, !llvm.ptr<1>
    tt.return
  }
}

// -----

// Unknown and explicitly resident inputs keep pure class pins through cleanup
// and finalization. Repeated native updates still share constrained inputs;
// the standalone tagged fixtures below cover late side-effect promotion.
// OPERAND-CSE-LABEL: llvm.func @scheduled_operand_pins_repeated_inputs
// OPERAND-CSE: %[[CA:.*]] = llvm.inline_asm asm_dialect = att operand_attrs = [] "", "=a,0"
// OPERAND-CSE: %[[CAV:.*]] = llvm.bitcast %[[CA]] : vector<4xi32> to vector<8xbf16>
// OPERAND-CSE: %[[CB0:.*]] = llvm.inline_asm asm_dialect = att operand_attrs = [] "", "=v,0"
// OPERAND-CSE: %[[CB0V:.*]] = llvm.bitcast %[[CB0]] : vector<4xi32> to vector<8xbf16>
// OPERAND-CSE: rocdl.mfma.f32.16x16x32.bf16 %[[CB0V]], %[[CAV]],
// OPERAND-CSE-NOT: ttg.amdg.scheduled_mfma_operand_pin
// OPERAND-CSE: rocdl.mfma.f32.16x16x32.bf16 %[[CB0V]], %[[CAV]],
// OPERAND-CSE: %[[CB1:.*]] = llvm.inline_asm asm_dialect = att operand_attrs = [] "", "=v,0"
// OPERAND-CSE: %[[CB1V:.*]] = llvm.bitcast %[[CB1]] : vector<4xi32> to vector<8xbf16>
// OPERAND-CSE-NOT: ttg.amdg.scheduled_mfma_operand_pin
// OPERAND-CSE: rocdl.mfma.f32.16x16x32.bf16 %[[CB1V]], %[[CAV]],
// OPERAND-CSE-NOT: ttg.amdg.scheduled_mfma_operand_pin
// OPERAND-CSE: llvm.return
// OPERAND-FINAL-LABEL: llvm.func @scheduled_operand_pins_repeated_inputs
// OPERAND-FINAL: %[[FA:.*]] = llvm.inline_asm asm_dialect = att operand_attrs = [] "", "=a,0"
// OPERAND-FINAL: %[[FAV:.*]] = llvm.bitcast %[[FA]] : vector<4xi32> to vector<8xbf16>
// OPERAND-FINAL: %[[FB0:.*]] = llvm.inline_asm asm_dialect = att operand_attrs = [] "", "=v,0"
// OPERAND-FINAL: %[[FB0V:.*]] = llvm.bitcast %[[FB0]] : vector<4xi32> to vector<8xbf16>
// OPERAND-FINAL: rocdl.mfma.f32.16x16x32.bf16 %[[FB0V]], %[[FAV]],
// OPERAND-FINAL: rocdl.mfma.f32.16x16x32.bf16 %[[FB0V]], %[[FAV]],
// OPERAND-FINAL: %[[FB1:.*]] = llvm.inline_asm asm_dialect = att operand_attrs = [] "", "=v,0"
// OPERAND-FINAL: %[[FB1V:.*]] = llvm.bitcast %[[FB1]] : vector<4xi32> to vector<8xbf16>
// OPERAND-FINAL: rocdl.mfma.f32.16x16x32.bf16 %[[FB1V]], %[[FAV]],
// OPERAND-FINAL: llvm.return

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scheduled_operand_pins_repeated_inputs(
      %a: tensor<16x32xbf16, #lhs>,
      %b0: tensor<32x16xbf16, #rhs>, %b1: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %first = amdg.scheduled_mfma %a, %b0, %zero
        resident "lhs" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %second = amdg.scheduled_mfma %a, %b0, %first
        resident "lhs" accumulator "persistent"
        register_class "agpr" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %third = amdg.scheduled_mfma %a, %b1, %second
        resident "lhs" accumulator "persistent"
        register_class "agpr" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// Finalization is restricted to compiler-owned operand pins. Cleanup may merge
// identical tagged pins, but must distinguish a different input or bank. The
// returned tuple checks every use after CSE and after side-effect promotion.
// OPERAND-CSE-LABEL: llvm.func @finalize_operand_pins_cse(
// OPERAND-CSE-SAME: %[[CX:.*]]: vector<4xi32>, %[[CY:.*]]: vector<4xi32>
// OPERAND-CSE: %[[CVX:.*]] = llvm.inline_asm {ttg.amdg.scheduled_mfma_operand_pin} "", "=v,0" %[[CX]]
// OPERAND-CSE-NOT: llvm.inline_asm
// OPERAND-CSE: %[[CVY:.*]] = llvm.inline_asm {ttg.amdg.scheduled_mfma_operand_pin} "", "=v,0" %[[CY]]
// OPERAND-CSE-NOT: llvm.inline_asm
// OPERAND-CSE: %[[CAX:.*]] = llvm.inline_asm {ttg.amdg.scheduled_mfma_operand_pin} "", "=a,0" %[[CX]]
// OPERAND-CSE-NOT: llvm.inline_asm
// OPERAND-CSE: llvm.insertvalue %[[CVX]], %{{.*}}[0]
// OPERAND-CSE: llvm.insertvalue %[[CVX]], %{{.*}}[1]
// OPERAND-CSE: llvm.insertvalue %[[CVY]], %{{.*}}[2]
// OPERAND-CSE: llvm.insertvalue %[[CAX]], %{{.*}}[3]
// OPERAND-CSE: llvm.insertvalue %[[CAX]], %{{.*}}[4]
// OPERAND-CSE: llvm.return
// OPERAND-FINAL-LABEL: llvm.func @finalize_operand_pins_cse(
// OPERAND-FINAL-SAME: %[[FX:.*]]: vector<4xi32>, %[[FY:.*]]: vector<4xi32>
// OPERAND-FINAL: %[[FVX:.*]] = llvm.inline_asm has_side_effects "", "=v,0" %[[FX]]
// OPERAND-FINAL-NOT: llvm.inline_asm
// OPERAND-FINAL: %[[FVY:.*]] = llvm.inline_asm has_side_effects "", "=v,0" %[[FY]]
// OPERAND-FINAL-NOT: llvm.inline_asm
// OPERAND-FINAL: %[[FAX:.*]] = llvm.inline_asm has_side_effects "", "=a,0" %[[FX]]
// OPERAND-FINAL-NOT: llvm.inline_asm
// OPERAND-FINAL: llvm.insertvalue %[[FVX]], %{{.*}}[0]
// OPERAND-FINAL: llvm.insertvalue %[[FVX]], %{{.*}}[1]
// OPERAND-FINAL: llvm.insertvalue %[[FVY]], %{{.*}}[2]
// OPERAND-FINAL: llvm.insertvalue %[[FAX]], %{{.*}}[3]
// OPERAND-FINAL: llvm.insertvalue %[[FAX]], %{{.*}}[4]
// OPERAND-FINAL: llvm.return

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  llvm.func @finalize_operand_pins_cse(%x: vector<4xi32>, %y: vector<4xi32>) -> !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)> {
    %vx = llvm.inline_asm {ttg.amdg.scheduled_mfma_operand_pin} "", "=v,0" %x : (vector<4xi32>) -> vector<4xi32>
    %vx_again = llvm.inline_asm {ttg.amdg.scheduled_mfma_operand_pin} "", "=v,0" %x : (vector<4xi32>) -> vector<4xi32>
    %vy = llvm.inline_asm {ttg.amdg.scheduled_mfma_operand_pin} "", "=v,0" %y : (vector<4xi32>) -> vector<4xi32>
    %ax = llvm.inline_asm {ttg.amdg.scheduled_mfma_operand_pin} "", "=a,0" %x : (vector<4xi32>) -> vector<4xi32>
    %ax_again = llvm.inline_asm {ttg.amdg.scheduled_mfma_operand_pin} "", "=a,0" %x : (vector<4xi32>) -> vector<4xi32>
    %empty = llvm.mlir.undef : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
    %r0 = llvm.insertvalue %vx, %empty[0] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
    %r1 = llvm.insertvalue %vx_again, %r0[1] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
    %r2 = llvm.insertvalue %vy, %r1[2] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
    %r3 = llvm.insertvalue %ax, %r2[3] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
    %r4 = llvm.insertvalue %ax_again, %r3[4] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
    llvm.return %r4 : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
  }

  // OPERAND-CSE-LABEL: llvm.func @finalize_operand_pins_unmarked
  // OPERAND-CSE: %[[CU:.*]] = llvm.inline_asm "", "=v,0"
  // OPERAND-CSE: %[[CC:.*]] = llvm.inline_asm has_side_effects "", "=a,0"
  // OPERAND-CSE: %[[CD:.*]] = llvm.inline_asm has_side_effects "s_nop 11", "=a,0" %[[CC]]
  // OPERAND-CSE: %[[CO:.*]] = llvm.inline_asm "v_add_u32 $0, 1, $1", "=v,v" %[[CU]]
  // OPERAND-CSE: llvm.return
  // OPERAND-FINAL-LABEL: llvm.func @finalize_operand_pins_unmarked
  // OPERAND-FINAL: %[[FU:.*]] = llvm.inline_asm "", "=v,0"
  // OPERAND-FINAL: %[[FC:.*]] = llvm.inline_asm has_side_effects "", "=a,0"
  // OPERAND-FINAL: %[[FD:.*]] = llvm.inline_asm has_side_effects "s_nop 11", "=a,0" %[[FC]]
  // OPERAND-FINAL: %[[FO:.*]] = llvm.inline_asm "v_add_u32 $0, 1, $1", "=v,v" %[[FU]]
  // OPERAND-FINAL: llvm.return
  llvm.func @finalize_operand_pins_unmarked(%x: vector<4xi32>) -> !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>)> {
    %user = llvm.inline_asm "", "=v,0" %x : (vector<4xi32>) -> vector<4xi32>
    %acc = llvm.inline_asm has_side_effects "", "=a,0" %x : (vector<4xi32>) -> vector<4xi32>
    %drain = llvm.inline_asm has_side_effects "s_nop 11", "=a,0" %acc : (vector<4xi32>) -> vector<4xi32>
    %opaque = llvm.inline_asm "v_add_u32 $0, 1, $1", "=v,v" %user : (vector<4xi32>) -> vector<4xi32>
    %empty = llvm.mlir.undef : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>)>
    %r0 = llvm.insertvalue %user, %empty[0] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>)>
    %r1 = llvm.insertvalue %drain, %r0[1] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>)>
    %r2 = llvm.insertvalue %opaque, %r1[2] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>)>
    llvm.return %r2 : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>)>
  }
}

// -----

// Function arguments do not prove shared-load provenance. Their operand class
// pins remain pure; persistent result placement and native arithmetic remain.
// LOAD-ORIGIN-LABEL: llvm.func @operand_order_unknown_roots
// LOAD-ORIGIN-COUNT-2: llvm.inline_asm asm_dialect = att operand_attrs = [] "", "=v,0"
// LOAD-ORIGIN: rocdl.mfma.f32.16x16x32.bf16
// LOAD-ORIGIN: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=a,0"
// LOAD-ORIGIN: llvm.return

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @operand_order_unknown_roots(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>) {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// Ordering is a property of both operand origins, not merely persistent MFMA.
// Forwarding is transparent only when every reachable terminal is a local load.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // LOAD-ORIGIN-LABEL: llvm.func @operand_order_direct_loads
  // LOAD-ORIGIN-COUNT-2: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=v,0"
  // LOAD-ORIGIN: rocdl.mfma.f32.16x16x32.bf16
  // LOAD-ORIGIN: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=a,0"
  // LOAD-ORIGIN: llvm.return
  tt.func public @operand_order_direct_loads(
      %a_smem: !ttg.memdesc<16x32xbf16, #shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<32x16xbf16, #shared, #smem, mutable>) {
    %a = ttg.local_load %a_smem : !ttg.memdesc<16x32xbf16, #shared, #smem, mutable> -> tensor<16x32xbf16, #lhs>
    %b = ttg.local_load %b_smem : !ttg.memdesc<32x16xbf16, #shared, #smem, mutable> -> tensor<32x16xbf16, #rhs>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // LOAD-ORIGIN-LABEL: llvm.func @operand_order_layout_and_slice
  // LOAD-ORIGIN-COUNT-2: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=v,0"
  // LOAD-ORIGIN: rocdl.mfma.f32.16x16x32.bf16
  // LOAD-ORIGIN: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=a,0"
  // LOAD-ORIGIN: llvm.return
  tt.func public @operand_order_layout_and_slice(
      %a_smem: !ttg.memdesc<32x32xbf16, #shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<32x32xbf16, #shared, #smem, mutable>) {
    %a_loaded = ttg.local_load %a_smem : !ttg.memdesc<32x32xbf16, #shared, #smem, mutable> -> tensor<32x32xbf16, #blocked>
    %b_loaded = ttg.local_load %b_smem : !ttg.memdesc<32x32xbf16, #shared, #smem, mutable> -> tensor<32x32xbf16, #blocked>
    %a_layout = ttg.convert_layout %a_loaded {allocation.offset = 0 : i32} : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #lhs>
    %b_layout = ttg.convert_layout %b_loaded {allocation.offset = 0 : i32} : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #rhs>
    %a = amdg.extract_slice %a_layout [0, 0] : tensor<32x32xbf16, #lhs> to tensor<16x32xbf16, #lhs>
    %b = amdg.extract_slice %b_layout [0, 0] : tensor<32x32xbf16, #rhs> to tensor<32x16xbf16, #rhs>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // LOAD-ORIGIN-LABEL: llvm.func @operand_order_computed_and_load
  // LOAD-ORIGIN-COUNT-2: llvm.inline_asm asm_dialect = att operand_attrs = [] "", "=v,0"
  // LOAD-ORIGIN: rocdl.mfma.f32.16x16x32.bf16
  // LOAD-ORIGIN: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=a,0"
  // LOAD-ORIGIN: llvm.return
  tt.func public @operand_order_computed_and_load(
      %a_smem: !ttg.memdesc<16x32xbf16, #shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<32x16xbf16, #shared, #smem, mutable>) {
    %a = ttg.local_load %a_smem : !ttg.memdesc<16x32xbf16, #shared, #smem, mutable> -> tensor<16x32xbf16, #lhs>
    %b = ttg.local_load %b_smem : !ttg.memdesc<32x16xbf16, #shared, #smem, mutable> -> tensor<32x16xbf16, #rhs>
    %computed = arith.addf %a, %a : tensor<16x32xbf16, #lhs>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %computed, %b, %zero
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // LOAD-ORIGIN-LABEL: llvm.func @operand_order_explicit_residency
  // LOAD-ORIGIN-COUNT-2: llvm.inline_asm asm_dialect = att operand_attrs = [] "", "=v,0"
  // LOAD-ORIGIN: rocdl.mfma.f32.16x16x32.bf16
  // LOAD-ORIGIN: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=a,0"
  // LOAD-ORIGIN: llvm.return
  tt.func public @operand_order_explicit_residency(
      %a_smem: !ttg.memdesc<16x32xbf16, #shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<32x16xbf16, #shared, #smem, mutable>) {
    %a = ttg.local_load %a_smem : !ttg.memdesc<16x32xbf16, #shared, #smem, mutable> -> tensor<16x32xbf16, #lhs>
    %b = ttg.local_load %b_smem : !ttg.memdesc<32x16xbf16, #shared, #smem, mutable> -> tensor<32x16xbf16, #rhs>
    %resident = amdg.register_resident %a class "vgpr" groups 4 : tensor<16x32xbf16, #lhs>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %resident, %b, %zero
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // Even two loaded operands are excluded when the operation requests a
  // resident operand; the existing AGPR/VGPR placement contract stays pure.
  // LOAD-ORIGIN-LABEL: llvm.func @operand_order_resident_operand
  // LOAD-ORIGIN: llvm.inline_asm asm_dialect = att operand_attrs = [] "", "=a,0"
  // LOAD-ORIGIN: llvm.inline_asm asm_dialect = att operand_attrs = [] "", "=v,0"
  // LOAD-ORIGIN: rocdl.mfma.f32.16x16x32.bf16
  // LOAD-ORIGIN: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=a,0"
  // LOAD-ORIGIN: llvm.return
  tt.func public @operand_order_resident_operand(
      %a_smem: !ttg.memdesc<16x32xbf16, #shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<32x16xbf16, #shared, #smem, mutable>) {
    %a = ttg.local_load %a_smem : !ttg.memdesc<16x32xbf16, #shared, #smem, mutable> -> tensor<16x32xbf16, #lhs>
    %b = ttg.local_load %b_smem : !ttg.memdesc<32x16xbf16, #shared, #smem, mutable> -> tensor<32x16xbf16, #rhs>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %zero
        resident "lhs" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // Both successors name the same block. The proof must inspect both
  // successor operand lists, including the computed B value on the false edge.
  // LOAD-ORIGIN-LABEL: llvm.func @operand_order_mixed_merge
  // LOAD-ORIGIN-COUNT-2: llvm.inline_asm asm_dialect = att operand_attrs = [] "", "=v,0"
  // LOAD-ORIGIN: rocdl.mfma.f32.16x16x32.bf16
  // LOAD-ORIGIN: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=a,0"
  // LOAD-ORIGIN: llvm.return
  tt.func public @operand_order_mixed_merge(
      %a_smem: !ttg.memdesc<16x32xbf16, #shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<32x16xbf16, #shared, #smem, mutable>,
      %condition: i1) {
    %a = ttg.local_load %a_smem : !ttg.memdesc<16x32xbf16, #shared, #smem, mutable> -> tensor<16x32xbf16, #lhs>
    %b = ttg.local_load %b_smem : !ttg.memdesc<32x16xbf16, #shared, #smem, mutable> -> tensor<32x16xbf16, #rhs>
    %computed_b = arith.addf %b, %b : tensor<32x16xbf16, #rhs>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    cf.cond_br %condition, ^merge(%b : tensor<32x16xbf16, #rhs>), ^merge(%computed_b : tensor<32x16xbf16, #rhs>)
  ^merge(%merged_b: tensor<32x16xbf16, #rhs>):
    %result = amdg.scheduled_mfma %a, %merged_b, %zero
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // A loop may execute zero times. Both initial and backedge inputs must be
  // loads, including the operands merged into the final scheduled update.
  // LOAD-ORIGIN-LABEL: llvm.func @operand_order_cfg_loop_loads
  // LOAD-ORIGIN-COUNT-2: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=v,0"
  // LOAD-ORIGIN: rocdl.mfma.f32.16x16x32.bf16
  // LOAD-ORIGIN-COUNT-2: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=v,0"
  // LOAD-ORIGIN: rocdl.mfma.f32.16x16x32.bf16
  // LOAD-ORIGIN: llvm.return
  tt.func public @operand_order_cfg_loop_loads(
      %a_smem: !ttg.memdesc<16x32xbf16, #shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<32x16xbf16, #shared, #smem, mutable>,
      %enter: i1, %again: i1) {
    %a = ttg.local_load %a_smem : !ttg.memdesc<16x32xbf16, #shared, #smem, mutable> -> tensor<16x32xbf16, #lhs>
    %b = ttg.local_load %b_smem : !ttg.memdesc<32x16xbf16, #shared, #smem, mutable> -> tensor<32x16xbf16, #rhs>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    cf.cond_br %enter, ^loop(%a, %b : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>), ^exit(%a, %b : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>)
  ^loop(%current_a: tensor<16x32xbf16, #lhs>, %current_b: tensor<32x16xbf16, #rhs>):
    %step = amdg.scheduled_mfma %current_a, %current_b, %zero
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    %next_a = ttg.local_load %a_smem : !ttg.memdesc<16x32xbf16, #shared, #smem, mutable> -> tensor<16x32xbf16, #lhs>
    %next_b = ttg.local_load %b_smem : !ttg.memdesc<32x16xbf16, #shared, #smem, mutable> -> tensor<32x16xbf16, #rhs>
    cf.cond_br %again, ^loop(%next_a, %next_b : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>), ^exit(%next_a, %next_b : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>)
  ^exit(%final_a: tensor<16x32xbf16, #lhs>, %final_b: tensor<32x16xbf16, #rhs>):
    %result = amdg.scheduled_mfma %final_a, %final_b, %zero
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    tt.return
  }

  // A pass-through cycle is accepted only because its reachable roots are
  // the loads on the incoming edge.
  // LOAD-ORIGIN-LABEL: llvm.func @operand_order_cfg_seeded_cycle
  // LOAD-ORIGIN-COUNT-2: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=v,0"
  // LOAD-ORIGIN: rocdl.mfma.f32.16x16x32.bf16
  // LOAD-ORIGIN: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "", "=a,0"
  // LOAD-ORIGIN: llvm.return
  tt.func public @operand_order_cfg_seeded_cycle(
      %a_smem: !ttg.memdesc<16x32xbf16, #shared, #smem, mutable>,
      %b_smem: !ttg.memdesc<32x16xbf16, #shared, #smem, mutable>,
      %again: i1) {
    %a = ttg.local_load %a_smem : !ttg.memdesc<16x32xbf16, #shared, #smem, mutable> -> tensor<16x32xbf16, #lhs>
    %b = ttg.local_load %b_smem : !ttg.memdesc<32x16xbf16, #shared, #smem, mutable> -> tensor<32x16xbf16, #rhs>
    %zero = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    cf.br ^loop(%a, %b : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>)
  ^loop(%current_a: tensor<16x32xbf16, #lhs>, %current_b: tensor<32x16xbf16, #rhs>):
    %result = amdg.scheduled_mfma %current_a, %current_b, %zero
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma>
    cf.cond_br %again, ^loop(%current_a, %current_b : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>), ^exit
  ^exit:
    tt.return
  }

}

// -----

// A forged legacy drain-coverage marker must not remove the explicit native
// results-only commit wait.
//
// CHECK-LABEL: llvm.func @forged_commit_drain_coverage
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects{{.*}} "", "=a,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 11", "=a,0,~{memory}"
// CHECK-NOT: llvm.inline_asm has_side_effects "", "=a,0,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @forged_commit_drain_coverage(
      %a: tensor<16x32xbf16, #lhs>,
      %b: tensor<32x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<16x32xbf16, #lhs>,
          tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %committed = amdg.mfma_commit %result {
        ttg.amdg.mfma_commit.drain_covered
      } : tensor<16x16xf32, #mma>
    tt.return
  }
}





// -----

// The first boundary explicitly anchors both persistent tuples and the live
// AGPR operand. The second boundary retains its tuple and constraints, but its
// inputs are returned after the combined full drain, even with dead outputs.
//
// CHECK-LABEL: llvm.func @covered_adjacent_split_commit
// CHECK: rocdl.mfma.f32.32x32x16.bf16
// CHECK: llvm.inline_asm has_side_effects{{.*}} "", "=a,0"
// CHECK: rocdl.mfma.f32.32x32x16.bf16
// CHECK: llvm.inline_asm has_side_effects{{.*}} "", "=v,0"
// CHECK: %[[COMBINED:[0-9]+]] = llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 15\0As_nop 3", "=a,=v,=a,0,1,2,~{memory}"
// CHECK: %[[DV:[0-9]+]] = llvm.extractvalue %[[COMBINED]][1]
// CHECK: %[[LIVE:[0-9]+]] = llvm.extractvalue %[[COMBINED]][2]
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,=a,0,1,~{memory}" %[[DV]], %[[LIVE]]
// CHECK-NOT: s_nop

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @covered_adjacent_split_commit(
      %a: tensor<32x16xbf16, #lhs>,
      %b: tensor<16x32xbf16, #rhs>,
      %live: tensor<16x32xbf16, #rhs>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %resident = amdg.register_resident %live class "agpr" groups 4
        : tensor<16x32xbf16, #rhs>
    %dk = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %dv = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %committed_dk = amdg.mfma_commit %dk : tensor<32x32xf32, #mma>
    %committed_dv, %preserved = amdg.mfma_commit %dv, %resident
        : tensor<32x32xf32, #mma>, tensor<16x32xbf16, #rhs>
    tt.return
  }
}





// -----

// The persistent result strengthens the second commit even when its live dot
// operand has no explicit AGPR residency proof.
//
// CHECK-LABEL: llvm.func @unanchored_split_commit
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=a,0,~{memory}"
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=v,=a,0,1,~{memory}"
// CHECK-NOT: llvm.inline_asm has_side_effects "", "=v,=a,0,1,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @unanchored_split_commit(
      %a: tensor<32x16xbf16, #lhs>,
      %b: tensor<16x32xbf16, #rhs>,
      %live: tensor<16x32xbf16, #rhs>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %dk = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %dv = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %committed_dk = amdg.mfma_commit %dk : tensor<32x32xf32, #mma>
    %committed_dv, %preserved = amdg.mfma_commit %dv, %live
        : tensor<32x32xf32, #mma>, tensor<16x32xbf16, #rhs>
    tt.return
  }
}





// -----

// Explicit N subtiles own independent native accumulator chains. The final
// commit drains both updated subtiles together without widening either chain.
//
// CHECK-LABEL: llvm.func @independent_output_subtiles
// CHECK: %[[FIRST_MFMA:[0-9]+]] = rocdl.mfma.f32.16x16x32.bf16
// CHECK: %[[FIRST_PACK:[0-9]+]] = llvm.bitcast %[[FIRST_MFMA]] : vector<4xf32> to vector<4xi32>
// CHECK: %[[FIRST_PIN:[0-9]+]] = llvm.inline_asm has_side_effects{{.*}} "", "=a,0" %[[FIRST_PACK]]
// CHECK: %[[FIRST:[0-9]+]] = llvm.bitcast %[[FIRST_PIN]] : vector<4xi32> to vector<4xf32>
// CHECK: llvm.extractelement %[[FIRST]]
// CHECK: %[[SECOND_MFMA:[0-9]+]] = rocdl.mfma.f32.16x16x32.bf16
// CHECK: %[[SECOND_PACK:[0-9]+]] = llvm.bitcast %[[SECOND_MFMA]] : vector<4xf32> to vector<4xi32>
// CHECK: %[[SECOND_PIN:[0-9]+]] = llvm.inline_asm has_side_effects{{.*}} "", "=a,0" %[[SECOND_PACK]]
// CHECK: %[[SECOND:[0-9]+]] = llvm.bitcast %[[SECOND_PIN]] : vector<4xi32> to vector<4xf32>
// CHECK-NOT: rocdl.mfma
// CHECK: llvm.extractelement %[[SECOND]]
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 11", "=a,=a,0,1,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @independent_output_subtiles(
      %a: tensor<16x32xbf16, #lhs>,
      %b: tensor<32x32xbf16, #rhs>,
      %acc: tensor<16x32xf32, #mma>) -> (tensor<16x16xf32, #mma>, tensor<16x16xf32, #mma>) {
    %b0 = amdg.extract_slice %b [0, 0] : tensor<32x32xbf16, #rhs> to tensor<32x16xbf16, #rhs>
    %b1 = amdg.extract_slice %b [0, 16] : tensor<32x32xbf16, #rhs> to tensor<32x16xbf16, #rhs>
    %acc0 = amdg.extract_slice %acc [0, 0] : tensor<16x32xf32, #mma> to tensor<16x16xf32, #mma>
    %acc1 = amdg.extract_slice %acc [0, 16] : tensor<16x32xf32, #mma> to tensor<16x16xf32, #mma>
    %first = amdg.scheduled_mfma %a, %b0, %acc0
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>,
          tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %second = amdg.scheduled_mfma %a, %b1, %acc1
        resident "none" accumulator "persistent"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #lhs>,
          tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %committed0, %committed1 = amdg.mfma_commit %first, %second : tensor<16x16xf32, #mma>, tensor<16x16xf32, #mma>
    tt.return %committed0, %committed1 : tensor<16x16xf32, #mma>, tensor<16x16xf32, #mma>
  }
}





// -----

// Each four-wave group owns one batch; each wave has two native N fragments.
// CHECK-LABEL: llvm.func @wave_batched_transient
// CHECK-COUNT-2: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects{{.*}}s_nop 5
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1, 4], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @wave_batched_transient(%a: tensor<2x16x32xbf16, #lhs>, %b: tensor<2x32x128xbf16, #rhs>) {
    %acc = arith.constant dense<7.000000e+00> : tensor<2x16x128xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "transient" register_class "auto" initialize true
        : tensor<2x16x32xbf16, #lhs>, tensor<2x32x128xbf16, #rhs>, tensor<2x16x128xf32, #mma>
          -> tensor<2x16x128xf32, #mma>
    %committed, %preserved = amdg.mfma_commit %result, %b : tensor<2x16x128xf32, #mma>, tensor<2x32x128xbf16, #rhs>
    tt.return
  }
}





// -----

// Each four-wave group owns one batch; each wave has two native N fragments.
// CHECK-LABEL: llvm.func @wave_batched_persistent
// CHECK-COUNT-2: rocdl.mfma.f32.16x16x32.bf16
// CHECK-NOT: rocdl.mfma
// CHECK-COUNT-2: llvm.inline_asm has_side_effects{{.*}} "", "=a,0"
// CHECK: llvm.inline_asm has_side_effects{{.*}}s_nop 11
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1, 4], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @wave_batched_persistent(%a: tensor<2x16x32xbf16, #lhs>, %b: tensor<2x32x128xbf16, #rhs>) {
    %acc = arith.constant dense<7.000000e+00> : tensor<2x16x128xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent" register_class "agpr" initialize true
        : tensor<2x16x32xbf16, #lhs>, tensor<2x32x128xbf16, #rhs>, tensor<2x16x128xf32, #mma>
          -> tensor<2x16x128xf32, #mma>
    %committed = amdg.mfma_commit %result : tensor<2x16x128xf32, #mma>
    tt.return
  }
}





// -----

// Each four-wave group owns one batch and explicitly slices the second N64
// subtile. Each wave updates one native output fragment. Separate class pins
// preserve the resident RHS and AGPR result while native arithmetic exposes
// the accumulator and operand lifetimes to LLVM.
// CHECK-LABEL: llvm.func @wave_batched_column_subtile
// CHECK: %[[SUB_A_REG:[0-9]+]] = llvm.inline_asm{{.*}} "", "=v,0"
// CHECK: %[[SUB_A:[0-9]+]] = llvm.bitcast %[[SUB_A_REG]] : vector<4xi32> to vector<8xbf16>
// CHECK: %[[SUB_B_REG:[0-9]+]] = llvm.inline_asm{{.*}} "", "=a,0"
// CHECK: %[[SUB_B:[0-9]+]] = llvm.bitcast %[[SUB_B_REG]] : vector<4xi32> to vector<8xbf16>
// CHECK: rocdl.mfma.f32.16x16x32.bf16 %[[SUB_B]], %[[SUB_A]],
// CHECK-NOT: rocdl.mfma
// CHECK: llvm.inline_asm has_side_effects{{.*}} "", "=a,0"
// CHECK: llvm.inline_asm has_side_effects{{.*}}s_nop 11
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1, 4], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @wave_batched_column_subtile(%a: tensor<2x16x32xbf16, #lhs>, %b: tensor<2x32x128xbf16, #rhs>) {
    %acc = arith.constant dense<7.000000e+00> : tensor<2x16x128xf32, #mma>
    %b_subtile = amdg.extract_slice %b [0, 0, 64] : tensor<2x32x128xbf16, #rhs> to tensor<2x32x64xbf16, #rhs>
    %acc_subtile = amdg.extract_slice %acc [0, 0, 64] : tensor<2x16x128xf32, #mma> to tensor<2x16x64xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b_subtile, %acc_subtile
        resident "rhs" accumulator "persistent" register_class "agpr" initialize false
        : tensor<2x16x32xbf16, #lhs>, tensor<2x32x64xbf16, #rhs>, tensor<2x16x64xf32, #mma>
          -> tensor<2x16x64xf32, #mma>
    %committed = amdg.mfma_commit %result : tensor<2x16x64xf32, #mma>
    tt.return
  }
}





// -----

// Initializing ignores the supplied accumulator only on the first native K
// update. The second update consumes that result through its native SSA
// chain; both updates retain the LHS operand's AGPR placement.
// CHECK-LABEL: llvm.func @persistent_resident_lhs_k_updates
// CHECK: %[[K_ZERO:[0-9]+]] = llvm.mlir.constant(dense<0.000000e+00> : vector<16xf32>)
// CHECK: %[[K0_A_REG:[0-9]+]] = llvm.inline_asm{{.*}} "", "=a,0"
// CHECK: %[[K0_A:[0-9]+]] = llvm.bitcast %[[K0_A_REG]] : vector<4xi32> to vector<8xbf16>
// CHECK: %[[K0:[0-9]+]] = rocdl.mfma.f32.32x32x16.bf16 %{{[0-9]+}}, %[[K0_A]], %[[K_ZERO]]
// CHECK: %[[K1_A_REG:[0-9]+]] = llvm.inline_asm{{.*}} "", "=a,0"
// CHECK: %[[K1_A:[0-9]+]] = llvm.bitcast %[[K1_A_REG]] : vector<4xi32> to vector<8xbf16>
// CHECK: rocdl.mfma.f32.32x32x16.bf16 %{{[0-9]+}}, %[[K1_A]], %[[K0]]
// CHECK-NOT: rocdl.mfma
// CHECK: llvm.inline_asm has_side_effects{{.*}} "", "=a,0"
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @persistent_resident_lhs_k_updates(%a: tensor<32x32xbf16, #lhs>, %b: tensor<32x32xbf16, #rhs>) {
    %acc = arith.constant dense<7.000000e+00> : tensor<32x32xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "lhs" accumulator "persistent" register_class "agpr" initialize true
        : tensor<32x32xbf16, #lhs>, tensor<32x32xbf16, #rhs>, tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %committed = amdg.mfma_commit %result : tensor<32x32xf32, #mma>
    tt.return
  }
}





// -----

// The loop update and exit commit consume the header value on different
// iterations/exit paths. Keep the previously supported direct-successor case.
// CHECK-LABEL: llvm.func @persistent_loop_direct_successors
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=a,0"
// CHECK-NOT: "s_nop 11", "=a,0"
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=a,0"
// CHECK-NOT: "s_nop 11", "=a,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 11", "=a,0,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @persistent_loop_direct_successors(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %count: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %initial = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent" register_class "agpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    cf.br ^header(%c0, %initial : i32, tensor<16x16xf32, #mma>)
  ^header(%i: i32, %acc: tensor<16x16xf32, #mma>):
    %continue = arith.cmpi slt, %i, %count : i32
    cf.cond_br %continue, ^body, ^exit
  ^body:
    %updated = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent" register_class "agpr" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %next = arith.addi %i, %c1 : i32
    cf.br ^header(%next, %updated : i32, tensor<16x16xf32, #mma>)
  ^exit:
    %committed = amdg.mfma_commit %acc : tensor<16x16xf32, #mma>
    tt.return
  }
}





// -----

// An unrelated owner branch joins before the persistent update. Both owner
// paths retain native accumulator updates and drain only at the commit.
// CHECK-LABEL: llvm.func @persistent_agpr_loop_after_owner_join
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=a,0"
// CHECK-NOT: "s_nop 11", "=a,0"
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=a,0"
// CHECK-NOT: "s_nop 11", "=a,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 11", "=a,0,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @persistent_agpr_loop_after_owner_join(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %count: i32, %owner: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %initial = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent" register_class "agpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    cf.br ^header(%c0, %initial : i32, tensor<16x16xf32, #mma>)
  ^header(%i: i32, %acc: tensor<16x16xf32, #mma>):
    %continue = arith.cmpi slt, %i, %count : i32
    cf.cond_br %continue, ^body_entry, ^exit
  ^body_entry:
    cf.cond_br %owner, ^owner_body, ^join
  ^owner_body:
    cf.br ^join
  ^join:
    %updated = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent" register_class "agpr" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %next = arith.addi %i, %c1 : i32
    cf.br ^header(%next, %updated : i32, tensor<16x16xf32, #mma>)
  ^exit:
    %committed = amdg.mfma_commit %acc : tensor<16x16xf32, #mma>
    tt.return
  }
}





// -----

// Loop-carried persistent results strengthen the shared exit commit while
// retaining VGPR ownership across an owner-branch join.
// CHECK-LABEL: llvm.func @persistent_vgpr_loop_after_owner_join
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 11", "=v,=a,0,1,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @persistent_vgpr_loop_after_owner_join(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %count: i32, %owner: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %initial = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent" register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    cf.br ^header(%c0, %initial : i32, tensor<16x16xf32, #mma>)
  ^header(%i: i32, %acc: tensor<16x16xf32, #mma>):
    %continue = arith.cmpi slt, %i, %count : i32
    cf.cond_br %continue, ^body_entry, ^exit
  ^body_entry:
    cf.cond_br %owner, ^owner_body, ^join
  ^owner_body:
    cf.br ^join
  ^join:
    %updated = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent" register_class "vgpr" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %next = arith.addi %i, %c1 : i32
    cf.br ^header(%next, %updated : i32, tensor<16x16xf32, #mma>)
  ^exit:
    %committed, %preserved = amdg.mfma_commit %acc, %b
        : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
    tt.return
  }
}





// -----

// A peeled update captures an OpResult on one arm and forwards it unchanged on
// the other. The two paths remain one accumulator chain through the join.
// CHECK-LABEL: llvm.func @persistent_agpr_opresult_diamond
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=a,0"
// CHECK-NOT: "s_nop 11", "=a,0"
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=a,0"
// CHECK-NOT: "s_nop 11", "=a,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 11", "=a,0,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @persistent_agpr_opresult_diamond(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %initial = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent" register_class "agpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    cf.cond_br %condition, ^update, ^skip
  ^update:
    %updated = amdg.scheduled_mfma %a, %b, %initial
        resident "none" accumulator "persistent" register_class "agpr" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    cf.br ^join(%updated : tensor<16x16xf32, #mma>)
  ^skip:
    cf.br ^join(%initial : tensor<16x16xf32, #mma>)
  ^join(%acc: tensor<16x16xf32, #mma>):
    %committed = amdg.mfma_commit %acc : tensor<16x16xf32, #mma>
    tt.return
  }
}





// -----

// An acyclic owner diamond before the update must not prevent proving the
// OpResult's exclusive uses. Preserve the VGPR/live-operand commit boundary.
// CHECK-LABEL: llvm.func @persistent_vgpr_opresult_after_owner_join
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 11", "=v,=a,0,1,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @persistent_vgpr_opresult_after_owner_join(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %condition: i1, %owner: i1) {
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %initial = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent" register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    cf.cond_br %condition, ^prefix, ^skip
  ^prefix:
    cf.cond_br %owner, ^owner_body, ^update
  ^owner_body:
    cf.br ^update
  ^update:
    %updated = amdg.scheduled_mfma %a, %b, %initial
        resident "none" accumulator "persistent" register_class "vgpr" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    cf.br ^join(%updated : tensor<16x16xf32, #mma>)
  ^skip:
    cf.br ^join(%initial : tensor<16x16xf32, #mma>)
  ^join(%acc: tensor<16x16xf32, #mma>):
    %committed, %preserved = amdg.mfma_commit %acc, %b
        : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
    tt.return
  }
}





// -----

// CFG simplification folds the bypass into a header successor operand. The
// update can reach the common join without taking that forwarding edge.
// CHECK-LABEL: llvm.func @persistent_opresult_forwarded_false_edge
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=a,0"
// CHECK-NOT: "s_nop 11", "=a,0"
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=a,0"
// CHECK-NOT: "s_nop 11", "=a,0"
// CHECK: "s_nop 11", "=a,0,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @persistent_opresult_forwarded_false_edge(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %initial = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent" register_class "agpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    cf.cond_br %condition, ^update, ^join(%initial : tensor<16x16xf32, #mma>)
  ^update:
    %updated = amdg.scheduled_mfma %a, %b, %initial
        resident "none" accumulator "persistent" register_class "agpr" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    cf.br ^join(%updated : tensor<16x16xf32, #mma>)
  ^join(%acc: tensor<16x16xf32, #mma>):
    %committed = amdg.mfma_commit %acc : tensor<16x16xf32, #mma>
    tt.return
  }
}





// -----

// Exercise the true-successor operand segment, a loop-header argument and an
// unrelated acyclic diamond before the sole update-side consumer.
// CHECK-LABEL: llvm.func @persistent_block_argument_forwarded_true_edge
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: "s_nop 11", "=v,=a,0,1,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @persistent_block_argument_forwarded_true_edge(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %count: i32, %owner: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %initial = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent" register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    cf.br ^header(%c0, %initial : i32, tensor<16x16xf32, #mma>)
  ^header(%i: i32, %acc: tensor<16x16xf32, #mma>):
    %done = arith.cmpi sge, %i, %count : i32
    cf.cond_br %done, ^exit(%acc : tensor<16x16xf32, #mma>), ^prefix
  ^prefix:
    cf.cond_br %owner, ^owner_body, ^update
  ^owner_body:
    cf.br ^update
  ^update:
    %updated = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent" register_class "vgpr" initialize false
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %next = arith.addi %i, %c1 : i32
    cf.br ^header(%next, %updated : i32, tensor<16x16xf32, #mma>)
  ^exit(%result: tensor<16x16xf32, #mma>):
    %committed, %preserved = amdg.mfma_commit %result, %b
        : tensor<16x16xf32, #mma>, tensor<32x16xbf16, #rhs>
    tt.return
  }
}





// -----

// The commit verifier also sees two uses of %initial. One is forwarding on
// the true edge; the other is the completion boundary on the opposite arm.
// CHECK-LABEL: llvm.func @mfma_commit_opresult_forwarded_edge
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects{{.*}} "", "=a,0"
// CHECK-COUNT-2: "s_nop 11", "=a,0,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_commit_opresult_forwarded_edge(
      %a: tensor<16x32xbf16, #lhs>, %b: tensor<32x16xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %initial = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent" register_class "agpr" initialize true
        : tensor<16x32xbf16, #lhs>, tensor<32x16xbf16, #rhs>, tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    cf.cond_br %condition, ^forwarded(%initial : tensor<16x16xf32, #mma>), ^direct
  ^forwarded(%acc: tensor<16x16xf32, #mma>):
    %committed = amdg.mfma_commit %acc : tensor<16x16xf32, #mma>
    tt.return
  ^direct:
    %other_committed = amdg.mfma_commit %initial : tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

// An intervening MFMA prevents shared completion even if the tuples are pinned.
//
// CHECK-LABEL: llvm.func @intervening_split_commit
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=a,0,~{memory}"
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=v,=a,0,1,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @intervening_split_commit(
      %a: tensor<32x16xbf16, #lhs>,
      %b: tensor<16x32xbf16, #rhs>,
      %live: tensor<16x32xbf16, #rhs>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %resident = amdg.register_resident %live class "agpr" groups 4
        : tensor<16x32xbf16, #rhs>
    %dk = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %dv = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %committed_dk = amdg.mfma_commit %dk : tensor<32x32xf32, #mma>
    %intervening = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>, tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma> -> tensor<32x32xf32, #mma>
    %committed_dv, %preserved = amdg.mfma_commit %dv, %resident
        : tensor<32x32xf32, #mma>, tensor<16x32xbf16, #rhs>
    tt.return
  }
}

// -----

// An untracked transient second root keeps its explicit handoff wait.
//
// CHECK-LABEL: llvm.func @transient_split_commit
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=a,0,~{memory}"
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 5", "=v,=a,0,1,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @transient_split_commit(
      %a: tensor<32x16xbf16, #lhs>,
      %b: tensor<16x32xbf16, #rhs>,
      %live: tensor<16x32xbf16, #rhs>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %resident = amdg.register_resident %live class "agpr" groups 4
        : tensor<16x32xbf16, #rhs>
    %dk = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %dv = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "transient"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %committed_dk = amdg.mfma_commit %dk : tensor<32x32xf32, #mma>
    %committed_dv, %preserved = amdg.mfma_commit %dv, %resident
        : tensor<32x32xf32, #mma>, tensor<16x32xbf16, #rhs>
    tt.return
  }
}

// -----

// A loop epilogue phi is safe when every incoming accumulator ends at a native result pin.
//
// CHECK-LABEL: llvm.func @loop_phi_split_commit
// CHECK: %[[COMBINED:[0-9]+]] = llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=a,=v,=a,0,1,2,~{memory}"
// CHECK: %[[DV:[0-9]+]] = llvm.extractvalue %[[COMBINED]][1]
// CHECK: %[[LIVE:[0-9]+]] = llvm.extractvalue %[[COMBINED]][2]
// CHECK: llvm.inline_asm has_side_effects{{.*}}"", "=v,=a,0,1,~{memory}" %[[DV]], %[[LIVE]]

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @loop_phi_split_commit(
      %a: tensor<32x16xbf16, #lhs>,
      %b: tensor<16x32xbf16, #rhs>,
      %live: tensor<16x32xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %resident = amdg.register_resident %live class "agpr" groups 4
        : tensor<16x32xbf16, #rhs>
    %dk = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %dv = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    cf.br ^loop(%dk, %dv : tensor<32x32xf32, #mma>, tensor<32x32xf32, #mma>)
  ^loop(%carried_dk: tensor<32x32xf32, #mma>, %carried_dv: tensor<32x32xf32, #mma>):
    %next_dk = amdg.scheduled_mfma %a, %b, %carried_dk
        resident "none" accumulator "persistent" register_class "agpr" initialize false
        : tensor<32x16xbf16, #lhs>, tensor<16x32xbf16, #rhs>, tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %next_dv = amdg.scheduled_mfma %a, %b, %carried_dv
        resident "none" accumulator "persistent" register_class "vgpr" initialize false
        : tensor<32x16xbf16, #lhs>, tensor<16x32xbf16, #rhs>, tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    cf.cond_br %condition, ^loop(%next_dk, %next_dv : tensor<32x32xf32, #mma>, tensor<32x32xf32, #mma>), ^exit(%next_dk, %next_dv : tensor<32x32xf32, #mma>, tensor<32x32xf32, #mma>)
  ^exit(%exit_dk: tensor<32x32xf32, #mma>, %exit_dv: tensor<32x32xf32, #mma>):
    %committed_dk = amdg.mfma_commit %exit_dk : tensor<32x32xf32, #mma>
    %committed_dv, %preserved = amdg.mfma_commit %exit_dv, %resident
        : tensor<32x32xf32, #mma>, tensor<16x32xbf16, #rhs>
    tt.return
  }
}

// -----

// Native users consume the retained second tuple after the combined drain.
//
// CHECK-LABEL: llvm.func @native_consumer_split_commit
// CHECK: %[[COMBINED:[0-9]+]] = llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=a,=v,=a,0,1,2,~{memory}"
// CHECK: %[[DV:[0-9]+]] = llvm.extractvalue %[[COMBINED]][1]
// CHECK: %[[LIVE:[0-9]+]] = llvm.extractvalue %[[COMBINED]][2]
// CHECK: llvm.inline_asm has_side_effects{{.*}}"", "=v,=a,0,1,~{memory}" %[[DV]], %[[LIVE]]
// CHECK: llvm.fadd
// CHECK: rocdl.mfma.f32.32x32x16.bf16

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @native_consumer_split_commit(
      %a: tensor<32x16xbf16, #lhs>,
      %b: tensor<16x32xbf16, #rhs>,
      %live: tensor<16x32xbf16, #rhs>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %resident = amdg.register_resident %live class "agpr" groups 4
        : tensor<16x32xbf16, #rhs>
    %dk = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %dv = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %committed_dk = amdg.mfma_commit %dk : tensor<32x32xf32, #mma>
    %committed_dv, %preserved = amdg.mfma_commit %dv, %resident
        : tensor<32x32xf32, #mma>, tensor<16x32xbf16, #rhs>
    %updated = arith.addf %committed_dv, %zero : tensor<32x32xf32, #mma>
    %next = amdg.scheduled_mfma %a, %preserved, %updated
        resident "none" accumulator "transient" register_class "vgpr" initialize false
        : tensor<32x16xbf16, #lhs>, tensor<16x32xbf16, #rhs>, tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// Opaque consumer completion guards remain in place after sharing the explicit boundaries.
//
// CHECK-LABEL: llvm.func @opaque_consumer_split_commit
// CHECK: %[[COMBINED:[0-9]+]] = llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=a,=v,=a,0,1,2,~{memory}"
// CHECK: %[[DV:[0-9]+]] = llvm.extractvalue %[[COMBINED]][1]
// CHECK: %[[LIVE:[0-9]+]] = llvm.extractvalue %[[COMBINED]][2]
// CHECK: llvm.inline_asm has_side_effects{{.*}}"", "=v,=a,0,1,~{memory}" %[[DV]], %[[LIVE]]
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3\0Av_add_f32

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @opaque_consumer_split_commit(
      %a: tensor<32x16xbf16, #lhs>,
      %b: tensor<16x32xbf16, #rhs>,
      %live: tensor<16x32xbf16, #rhs>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %resident = amdg.register_resident %live class "agpr" groups 4
        : tensor<16x32xbf16, #rhs>
    %dk = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %dv = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %committed_dk = amdg.mfma_commit %dk : tensor<32x32xf32, #mma>
    %committed_dv, %preserved = amdg.mfma_commit %dv, %resident
        : tensor<32x32xf32, #mma>, tensor<16x32xbf16, #rhs>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %committed_dv : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// A twelve-state first boundary cannot cover a twenty-state second boundary.
//
// CHECK-LABEL: llvm.func @insufficient_first_split_commit
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 11", "=a,0,~{memory}"
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=v,=a,0,1,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#small = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#small_lhs = #ttg.dot_op<{opIdx = 0, parent = #small, kWidth = 8}>
#small_rhs = #ttg.dot_op<{opIdx = 1, parent = #small, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @insufficient_first_split_commit(
      %a: tensor<32x16xbf16, #lhs>,
      %b: tensor<16x32xbf16, #rhs>,
      %live: tensor<16x32xbf16, #rhs>,
      %small_a: tensor<16x32xbf16, #small_lhs>,
      %small_b: tensor<32x16xbf16, #small_rhs>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %small_zero = arith.constant dense<0.0> : tensor<16x16xf32, #small>
    %resident = amdg.register_resident %live class "agpr" groups 4
        : tensor<16x32xbf16, #rhs>
    %dk = amdg.scheduled_mfma %small_a, %small_b, %small_zero
        resident "none" accumulator "persistent" register_class "agpr" initialize true
        : tensor<16x32xbf16, #small_lhs>, tensor<32x16xbf16, #small_rhs>, tensor<16x16xf32, #small>
          -> tensor<16x16xf32, #small>
    %dv = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %committed_dk = amdg.mfma_commit %dk : tensor<16x16xf32, #small>
    %committed_dv, %preserved = amdg.mfma_commit %dv, %resident
        : tensor<32x32xf32, #mma>, tensor<16x32xbf16, #rhs>
    tt.return
  }
}

// -----

// Direct initialized and loop-updated inputs meet at a zero-trip epilogue.
//
// CHECK-LABEL: llvm.func @zero_trip_phi_split_commit
// CHECK: %[[COMBINED:[0-9]+]] = llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=a,=v,=a,0,1,2,~{memory}"
// CHECK: %[[DV:[0-9]+]] = llvm.extractvalue %[[COMBINED]][1]
// CHECK: %[[LIVE:[0-9]+]] = llvm.extractvalue %[[COMBINED]][2]
// CHECK: llvm.inline_asm has_side_effects{{.*}}"", "=v,=a,0,1,~{memory}" %[[DV]], %[[LIVE]]

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @zero_trip_phi_split_commit(
      %a: tensor<32x16xbf16, #lhs>,
      %b: tensor<16x32xbf16, #rhs>,
      %live: tensor<16x32xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %resident = amdg.register_resident %live class "agpr" groups 4
        : tensor<16x32xbf16, #rhs>
    %dk = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %dv = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    cf.cond_br %condition, ^loop(%dk, %dv : tensor<32x32xf32, #mma>, tensor<32x32xf32, #mma>), ^exit(%dk, %dv : tensor<32x32xf32, #mma>, tensor<32x32xf32, #mma>)
  ^loop(%carried_dk: tensor<32x32xf32, #mma>, %carried_dv: tensor<32x32xf32, #mma>):
    %next_dk = amdg.scheduled_mfma %a, %b, %carried_dk
        resident "none" accumulator "persistent" register_class "agpr" initialize false
        : tensor<32x16xbf16, #lhs>, tensor<16x32xbf16, #rhs>, tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %next_dv = amdg.scheduled_mfma %a, %b, %carried_dv
        resident "none" accumulator "persistent" register_class "vgpr" initialize false
        : tensor<32x16xbf16, #lhs>, tensor<16x32xbf16, #rhs>, tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    cf.cond_br %condition, ^loop(%next_dk, %next_dv : tensor<32x32xf32, #mma>, tensor<32x32xf32, #mma>), ^exit(%next_dk, %next_dv : tensor<32x32xf32, #mma>, tensor<32x32xf32, #mma>)
  ^exit(%exit_dk: tensor<32x32xf32, #mma>, %exit_dv: tensor<32x32xf32, #mma>):
    %committed_dk = amdg.mfma_commit %exit_dk : tensor<32x32xf32, #mma>
    %committed_dv, %preserved = amdg.mfma_commit %exit_dv, %resident
        : tensor<32x32xf32, #mma>, tensor<16x32xbf16, #rhs>
    tt.return
  }
}

// -----

// A mutually exclusive opaque bypass keeps the producer drain and its guard.
//
// CHECK-LABEL: llvm.func @fork_opaque_split_commit
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=v,0"
// CHECK: %[[COMBINED:[0-9]+]] = llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=a,=v,=a,0,1,2,~{memory}"
// CHECK: %[[DV:[0-9]+]] = llvm.extractvalue %[[COMBINED]][1]
// CHECK: %[[LIVE:[0-9]+]] = llvm.extractvalue %[[COMBINED]][2]
// CHECK: llvm.inline_asm has_side_effects{{.*}}"", "=v,=a,0,1,~{memory}" %[[DV]], %[[LIVE]]
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3\0Av_add_f32
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3\0Av_add_f32

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fork_opaque_split_commit(
      %a: tensor<32x16xbf16, #lhs>,
      %b: tensor<16x32xbf16, #rhs>,
      %live: tensor<16x32xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %resident = amdg.register_resident %live class "agpr" groups 4
        : tensor<16x32xbf16, #rhs>
    %dk = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %dv = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    cf.cond_br %condition, ^commit(%dv : tensor<32x32xf32, #mma>), ^bypass(%dv : tensor<32x32xf32, #mma>)
  ^commit(%commit_dv: tensor<32x32xf32, #mma>):
    %committed_dk = amdg.mfma_commit %dk : tensor<32x32xf32, #mma>
    %committed_dv, %preserved = amdg.mfma_commit %commit_dv, %resident
        : tensor<32x32xf32, #mma>, tensor<16x32xbf16, #rhs>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %committed_dv : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #mma>
    tt.return
  ^bypass(%escape_dv: tensor<32x32xf32, #mma>):
    %escape = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %escape_dv : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

// Multi-group inputs and two logical VGPR accumulators retain every suffix group.
//
// CHECK-LABEL: llvm.func @multi_group_split_commit
// CHECK: %[[COMBINED:[0-9]+]] = llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=a,=a,=v,=v,=v,=v,=a,=a,0,1,2,3,4,5,6,7,~{memory}"
// CHECK: %[[DV0:[0-9]+]] = llvm.extractvalue %[[COMBINED]][2]
// CHECK: %[[DV1:[0-9]+]] = llvm.extractvalue %[[COMBINED]][3]
// CHECK: %[[DV20:[0-9]+]] = llvm.extractvalue %[[COMBINED]][4]
// CHECK: %[[DV21:[0-9]+]] = llvm.extractvalue %[[COMBINED]][5]
// CHECK: %[[LIVE0:[0-9]+]] = llvm.extractvalue %[[COMBINED]][6]
// CHECK: %[[LIVE1:[0-9]+]] = llvm.extractvalue %[[COMBINED]][7]
// CHECK: %[[SECOND:[0-9]+]] = llvm.inline_asm has_side_effects{{.*}}"", "=v,=v,=v,=v,=a,=a,0,1,2,3,4,5,~{memory}" %[[DV0]], %[[DV1]], %[[DV20]], %[[DV21]], %[[LIVE0]], %[[LIVE1]]
// CHECK: llvm.extractvalue %[[SECOND]][2]
// CHECK: llvm.extractvalue %[[SECOND]][3]
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3\0Av_add_f32

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @multi_group_split_commit(
      %a: tensor<32x16xbf16, #lhs>,
      %b: tensor<16x64xbf16, #rhs>,
      %live: tensor<16x64xbf16, #rhs>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
    %resident = amdg.register_resident %live class "agpr" groups 4
        : tensor<16x64xbf16, #rhs>
    %dk = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x64xbf16, #rhs>,
          tensor<32x64xf32, #mma>
          -> tensor<32x64xf32, #mma>
    %dv = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x64xbf16, #rhs>,
          tensor<32x64xf32, #mma>
          -> tensor<32x64xf32, #mma>
    %dv2 = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent" register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>, tensor<16x64xbf16, #rhs>, tensor<32x64xf32, #mma>
          -> tensor<32x64xf32, #mma>
    %committed_dk = amdg.mfma_commit %dk : tensor<32x64xf32, #mma>
    %committed_dv, %committed_dv2, %preserved = amdg.mfma_commit %dv, %dv2, %resident
        : tensor<32x64xf32, #mma>, tensor<32x64xf32, #mma>, tensor<16x64xbf16, #rhs>
    %out = tt.elementwise_inline_asm "v_add_f32 $0, $1, 1.0"
        {constraints = "=v,v", packed_element = 1 : i32, pure = false}
        %committed_dv2 : tensor<32x64xf32, #mma> -> tensor<32x64xf32, #mma>
    tt.return
  }
}

// -----

// A second accumulator phi mixing AGPR and VGPR roots keeps both full drains.
//
// CHECK-LABEL: llvm.func @mixed_class_phi_split_commit
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=a,0,~{memory}"
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=v,=a,0,1,~{memory}"
// CHECK-NOT: "=a,=v,=a,0,1,2,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mixed_class_phi_split_commit(
      %a: tensor<32x16xbf16, #lhs>,
      %b: tensor<16x32xbf16, #rhs>,
      %live: tensor<16x32xbf16, #rhs>, %condition: i1) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %resident = amdg.register_resident %live class "agpr" groups 4
        : tensor<16x32xbf16, #rhs>
    %dk = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %dv = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<32x16xbf16, #lhs>,
          tensor<16x32xbf16, #rhs>,
          tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    %other = amdg.scheduled_mfma %a, %b, %zero
        resident "none" accumulator "persistent" register_class "agpr" initialize true
        : tensor<32x16xbf16, #lhs>, tensor<16x32xbf16, #rhs>, tensor<32x32xf32, #mma>
          -> tensor<32x32xf32, #mma>
    cf.cond_br %condition, ^exit(%dv : tensor<32x32xf32, #mma>), ^exit(%other : tensor<32x32xf32, #mma>)
  ^exit(%merged: tensor<32x32xf32, #mma>):
    %committed_dk = amdg.mfma_commit %dk : tensor<32x32xf32, #mma>
    %committed_dv, %preserved = amdg.mfma_commit %merged, %resident
        : tensor<32x32xf32, #mma>, tensor<16x32xbf16, #rhs>
    tt.return
  }
}
