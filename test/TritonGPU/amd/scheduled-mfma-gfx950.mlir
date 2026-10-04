// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --verify-diagnostics | FileCheck %s

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
// The explicit live-operand commit retains its required wait.
// CHECK-LABEL: llvm.func @vgpr_pinned_accumulator_with_live_operand
// CHECK: rocdl.mfma.f32.16x16x32.bf16
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 5", "=v,=a,0,1,~{memory}"
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
  // CHECK-SAME: "s_nop 11", "=v,0"
  // CHECK: llvm.inline_asm has_side_effects
  // CHECK-SAME: "s_nop 5", "=v,=a,0,1,~{memory}"
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
    } : (i1, tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma>
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
