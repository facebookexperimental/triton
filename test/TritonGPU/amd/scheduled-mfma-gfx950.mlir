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

// Hazard-inference attributes are compiler-owned. Input IR cannot use them to
// skip the conservative input padding or result drain on an uncommitted chain.
//
// CHECK-LABEL: llvm.func @forged_hazard_attributes
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 3\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=a,v,v"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 11", "=a,0"

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

// Commit drain coverage is also a compiler-owned fact. A forged marker on an
// uncovered results-only commit must be cleared before lowering.
//
// CHECK-LABEL: llvm.func @forged_commit_drain_coverage
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=a,v,v"
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

// A full dK drain covers an immediately adjacent dV commit when the result
// chains and preserved operand have the required register classes.
//
// CHECK-LABEL: llvm.func @covered_adjacent_split_commit
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_32x32x16_bf16 $0, $1, $2, 0", "=a,v,v"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_32x32x16_bf16 $0, $1, $2, 0", "=&v,v,v"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 15\0As_nop 3", "=a,0,~{memory}"
// CHECK-NOT: "s_nop 5", "=v,=a,0,1,~{memory}"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "", "=v,=a,0,1,~{memory}"

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

// A dot operand without an explicit AGPR residency proof keeps the second
// commit's live-dependency wait.
//
// CHECK-LABEL: llvm.func @unanchored_split_commit
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 15\0As_nop 3", "=a,0,~{memory}"
// CHECK: llvm.inline_asm has_side_effects{{.*}}"s_nop 5", "=v,=a,0,1,~{memory}"
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

// Pinning the accumulator to VGPRs is the documented remedy and must lower.
// A live BF16 dependency makes the commit emit the same VGPR class, so this
// complete chain can use post-RA hazard repair and defer its result drain.
//
// CHECK-LABEL: llvm.func @vgpr_pinned_accumulator_with_live_operand
// CHECK: %[[FIRST:[0-9]+]] = llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=&v,v,v"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: %[[SECOND:[0-9]+]] = llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=&v,v,v,0"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 5", "=v,=a,0,1,~{memory}"

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @vgpr_pinned_accumulator_with_live_operand(
      %a: tensor<16x32xbf16, #lhs>,
      %b: tensor<32x16xbf16, #rhs>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %first = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize true
        : tensor<16x32xbf16, #lhs>,
          tensor<32x16xbf16, #rhs>,
          tensor<16x16xf32, #mma>
          -> tensor<16x16xf32, #mma>
    %result = amdg.scheduled_mfma %a, %b, %first
        resident "none" accumulator "persistent"
        register_class "vgpr" initialize false
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

// CDNA4 keeps the explicit AGPR class: there the accumulator read is ordered
// against the drain, and two persistent accumulator sets may deliberately
// occupy complementary register files. Contrast CDNA3, where the same request
// is rejected outright (see invalid.mlir).
//
// CHECK-LABEL: llvm.func @explicit_agpr_accumulator
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 3\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=a,v,v"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 11", "=a,0"

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

// A block argument that merges a scheduled accumulator with an unrelated
// value is not one chain. Its producer must retain conservative hazards.
//
// CHECK-LABEL: llvm.func @mixed_lineage_block_argument
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 3\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=a,v,v"

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

// Explicit N subtiles own independent native accumulator chains. The final
// commit drains both updated subtiles together without widening either chain.
//
// CHECK-LABEL: llvm.func @independent_output_subtiles
// CHECK: %[[FIRST:[0-9]+]] = llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=a,v,v"
// CHECK: llvm.extractelement %[[FIRST]]
// CHECK: %[[SECOND:[0-9]+]] = llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=a,v,v"
// CHECK-NOT: triton_amd_scheduled_mfma
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
// CHECK-COUNT-2: llvm.inline_asm has_side_effects{{.*}}v_mfma_f32_16x16x32_bf16
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
// subtile. Each wave updates one native output fragment. The resident operand
// and tied accumulator must have distinct AGPRs, including in LLVM's inferred
// minimum AGPR allocation.
// CHECK-LABEL: llvm.func @wave_batched_column_subtile
// CHECK: llvm.inline_asm has_side_effects{{.*}}v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=&a,a,v,0"
// CHECK-NOT: v_mfma_f32_16x16x32_bf16
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

// Initializing from zero permits source/destination AGPR reuse. A later native
// K update has a tied accumulator and must reserve distinct AGPRs for the
// resident source, even when the source kernel requested initialize=true.
// CHECK-LABEL: llvm.func @persistent_resident_lhs_k_updates
// CHECK: llvm.inline_asm has_side_effects{{.*}}v_mfma_f32_32x32x16_bf16 $0, $1, $2, 0", "=a,v,a"
// CHECK: llvm.inline_asm has_side_effects{{.*}}v_mfma_f32_32x32x16_bf16 $0, $1, $2, $0", "=&a,v,a,0"
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
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=a,v,v"
// CHECK-NOT: "s_nop 11", "=a,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=a,v,v,0"
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
// paths must still infer one accumulator chain and drain only at the commit.
// CHECK-LABEL: llvm.func @persistent_agpr_loop_after_owner_join
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=a,v,v"
// CHECK-NOT: "s_nop 11", "=a,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=a,v,v,0"
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

// A VGPR accumulator and live BF16 dependency retain their commit wait when
// the loop's accumulator update sits after the same owner-branch join.
// CHECK-LABEL: llvm.func @persistent_vgpr_loop_after_owner_join
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=&v,v,v"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=&v,v,v,0"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 5", "=v,=a,0,1,~{memory}"

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
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=a,v,v"
// CHECK-NOT: "s_nop 11", "=a,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=a,v,v,0"
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
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=&v,v,v"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=&v,v,v,0"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "s_nop 5", "=v,=a,0,1,~{memory}"

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
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=a,v,v"
// CHECK-NOT: "s_nop 11", "=a,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=a,v,v,0"
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
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=&v,v,v"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: llvm.inline_asm has_side_effects
// CHECK-SAME: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=&v,v,v,0"
// CHECK-NOT: "s_nop 11", "=v,0"
// CHECK: "s_nop 5", "=v,=a,0,1,~{memory}"

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
// CHECK: "; triton_amd_scheduled_mfma\0Av_mfma_f32_16x16x32_bf16 $0, $1, $2, 0", "=a,v,v"
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
