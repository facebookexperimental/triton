// RUN: TRITON_ENABLE_TTGIR_SCHED=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule 2>&1 | FileCheck %s
//
// Phase 1a of the TTGIR-level SCHED pass: detect M-split partition plans
// per MFMA-typed tt.dot, but do not yet mutate the IR. Emits:
//   * a per-loop summary remark
//   * a per-dot remark naming the planned split factor (blockM / instrM)
//   * a module-level summary remark
//
// Subsequent phases (1b backward walker, 1c forward walker, 1d sliceOp)
// will turn the plan into an actual SSA rewrite. See
//   ~/AMD/triton/claude/llir_sched_at_ttgir_plan.md

// Case A: a v8/v10-shaped dot. Result tensor<256x128xf32, #mma>, instrShape
// [16,16,32]. Expected split factor: blockM / instrM = 256 / 16 = 16.

// CHECK: remark: ttgir-sched: would M-split this dot into 8 (blockM=256 / ctaTileM=32), co-partitioning 0 producer op(s) + 1 user op(s) (phase 2: plan only)
// CHECK: remark: ttgir-sched: candidate inner loop with 1 MFMA tt.dot op(s); plans 1, skipped 0, bwd-infeasible 0, fwd-infeasible 0, applied 0, co-partition producer-ops 0 + user-ops 1 (phase 2: plan only)

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @v8_like_dot_M256(
      %arg_a: tensor<256x64xf16, #dot0>,
      %arg_b: tensor<64x128xf16, #dot1>,
      %arg_c_init: tensor<256x128xf32, #mma>,
      %lb: index, %ub: index, %step: index) -> tensor<256x128xf32, #mma> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %arg_c_init)
        -> (tensor<256x128xf32, #mma>) {
      %d = tt.dot %arg_a, %arg_b, %acc :
          tensor<256x64xf16, #dot0> * tensor<64x128xf16, #dot1>
          -> tensor<256x128xf32, #mma>
      scf.yield %d : tensor<256x128xf32, #mma>
    }
    tt.return %res : tensor<256x128xf32, #mma>
  }
}

// -----

// Case B: scf.for with no dots is not a candidate.

// CHECK: remark: ttgir-sched: visited 1 scf.for op(s), 0 candidate(s), 0 MFMA tt.dot op(s), 0 planned M-split(s), 0 skipped, 0 bwd-infeasible, 0 fwd-infeasible, 0 applied (phase 2: plan only)

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @inner_loop_no_dot(%lb: index, %ub: index, %step: index) {
    scf.for %iv = %lb to %ub step %step {
      scf.yield
    }
    tt.return
  }
}

// -----

// Case C: M < instrM (numPartitions would be 0). Dot is MFMA-typed so the
// loop counts as a candidate, but the per-dot plan is skipped.
// Here tensor<16x128xf32, #mma>, instrM=16, so 16/16 = 1 < 2 → skip
// (single-partition split would be a no-op).

// CHECK: remark: ttgir-sched: candidate inner loop with 1 MFMA tt.dot op(s); plans 0, skipped 1, bwd-infeasible 0, fwd-infeasible 0, applied 0, co-partition producer-ops 0 + user-ops 0 (phase 2: plan only)
// CHECK: remark: ttgir-sched: visited 1 scf.for op(s), 1 candidate(s), 1 MFMA tt.dot op(s), 0 planned M-split(s), 1 skipped, 0 bwd-infeasible, 0 fwd-infeasible, 0 applied (phase 2: plan only)

#mma_c = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0_c = #ttg.dot_op<{opIdx = 0, parent = #mma_c, kWidth = 8}>
#dot1_c = #ttg.dot_op<{opIdx = 1, parent = #mma_c, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @too_small_dot_skipped(
      %arg_a: tensor<16x64xf16, #dot0_c>,
      %arg_b: tensor<64x128xf16, #dot1_c>,
      %arg_c_init: tensor<16x128xf32, #mma_c>,
      %lb: index, %ub: index, %step: index) -> tensor<16x128xf32, #mma_c> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %arg_c_init)
        -> (tensor<16x128xf32, #mma_c>) {
      %d = tt.dot %arg_a, %arg_b, %acc :
          tensor<16x64xf16, #dot0_c> * tensor<64x128xf16, #dot1_c>
          -> tensor<16x128xf32, #mma_c>
      scf.yield %d : tensor<16x128xf32, #mma_c>
    }
    tt.return %res : tensor<16x128xf32, #mma_c>
  }
}
