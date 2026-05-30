// RUN: TRITON_ENABLE_TTGIR_SCHED=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule 2>&1 | FileCheck %s
//
// Phase 0 of the TTGIR-level SCHED pass: walks the module, identifies inner
// scf.for loops containing MFMA-typed tt.dot ops, and emits a remark per
// candidate loop plus a module-level summary remark. The IR is unchanged.
//
// Subsequent phases will add: M-split (Phase 1), N-split (Phase 2),
// schedule recipe + sched_barrier (Phase 3). See
//   ~/AMD/triton/claude/llir_sched_at_ttgir_plan.md

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

// First module: an inner scf.for containing exactly one MFMA-typed tt.dot.
// CHECK: remark: ttgir-sched: would M-split this dot into 8 (blockM=256 / ctaTileM=32), co-partitioning 0 producer op(s) + 1 user op(s) (phase 1d: plan only)
// CHECK: remark: ttgir-sched: candidate inner loop with 1 MFMA tt.dot op(s); plans 1, skipped 0, bwd-infeasible 0, fwd-infeasible 0, applied 0, co-partition producer-ops 0 + user-ops 1 (phase 1d: plan only)
// CHECK: remark: ttgir-sched: visited 1 scf.for op(s), 1 candidate(s), 1 MFMA tt.dot op(s), 1 planned M-split(s), 0 skipped, 0 bwd-infeasible, 0 fwd-infeasible, 0 applied (phase 1d: plan only)
// CHECK-LABEL: tt.func @inner_loop_with_one_mfma_dot
// CHECK: scf.for
// CHECK:   tt.dot

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @inner_loop_with_one_mfma_dot(
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

// Second module: scf.for with no dots — pass should visit but not flag as a
// candidate. Summary remark should report 0 candidates / 0 MFMA dots.

// CHECK: remark: ttgir-sched: visited 1 scf.for op(s), 0 candidate(s), 0 MFMA tt.dot op(s), 0 planned M-split(s), 0 skipped, 0 bwd-infeasible, 0 fwd-infeasible, 0 applied (phase 1d: plan only)
// CHECK-LABEL: tt.func @inner_loop_no_dot
// CHECK: scf.for

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @inner_loop_no_dot(%lb: index, %ub: index, %step: index) {
    scf.for %iv = %lb to %ub step %step {
      scf.yield
    }
    tt.return
  }
}
