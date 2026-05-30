// RUN: TRITON_ENABLE_TTGIR_SCHED=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule 2>&1 | FileCheck %s
//
// Phase 1c of the TTGIR-level SCHED pass: forward walker reports how many
// *user* ops (downstream of the dot's result) would be co-partitioned. No
// IR mutation.
//
// Walked from: dot's result, with currentDim=0 (the M dim of the result
// tensor). Stops at:
//   * scf.yield (recorded; the iter-arg back-edge is closed by next-iter
//     backward walk on dot.getC)
//   * unknown user op (bail; reported as fwd-infeasible)
//   * value escaping the loop body region (skip; outer users picked up by
//     a subsequent backward walk on the iter-arg / result)
//
// Recognised user ops on top of producer allow-list:
//   arith.truncf, arith.trunci, arith.{si,ui}tofp, arith.fpto{si,ui},
//   tt.store, ttg.local_store, amdgpu.buffer_store, scf.yield.

// Case A: v8/v10-shaped main loop, dot result feeds only scf.yield.
// 0 producers + 1 user (the yield).

// CHECK: remark: ttgir-sched: would M-split this dot into 8 (blockM=256 / ctaTileM=32), co-partitioning 0 producer op(s) + 1 user op(s) (phase 1d: plan only)
// CHECK: remark: ttgir-sched: candidate inner loop with 1 MFMA tt.dot op(s); plans 1, skipped 0, bwd-infeasible 0, fwd-infeasible 0, applied 0, co-partition producer-ops 0 + user-ops 1 (phase 1d: plan only)

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @v8_like_dot_yield_only(
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

// Case B: dot result → arith.truncf → scf.yield. The backward walker also
// picks up arith.extf on acc (acc was f16, dot needs f32 accumulator).
// → 1 producer (extf) + 2 users (truncf, yield).

// CHECK: remark: ttgir-sched: would M-split this dot into 8 (blockM=256 / ctaTileM=32), co-partitioning 1 producer op(s) + 2 user op(s) (phase 1d: plan only)
// CHECK: remark: ttgir-sched: candidate inner loop with 1 MFMA tt.dot op(s); plans 1, skipped 0, bwd-infeasible 0, fwd-infeasible 0, applied 0, co-partition producer-ops 1 + user-ops 2 (phase 1d: plan only)

#mma_b = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0_b = #ttg.dot_op<{opIdx = 0, parent = #mma_b, kWidth = 8}>
#dot1_b = #ttg.dot_op<{opIdx = 1, parent = #mma_b, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @dot_with_truncf_chain(
      %arg_a: tensor<256x64xf16, #dot0_b>,
      %arg_b: tensor<64x128xf16, #dot1_b>,
      %arg_acc_init: tensor<256x128xf16, #mma_b>,
      %lb: index, %ub: index, %step: index) -> tensor<256x128xf16, #mma_b> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %arg_acc_init)
        -> (tensor<256x128xf16, #mma_b>) {
      %acc_f32 = arith.extf %acc : tensor<256x128xf16, #mma_b> to tensor<256x128xf32, #mma_b>
      %d = tt.dot %arg_a, %arg_b, %acc_f32 :
          tensor<256x64xf16, #dot0_b> * tensor<64x128xf16, #dot1_b>
          -> tensor<256x128xf32, #mma_b>
      %dh = arith.truncf %d : tensor<256x128xf32, #mma_b> to tensor<256x128xf16, #mma_b>
      scf.yield %dh : tensor<256x128xf16, #mma_b>
    }
    tt.return %res : tensor<256x128xf16, #mma_b>
  }
}

// -----

// (Phase 1c Case C dropped — `tt.reshape allow_reorder` is folded into the
// elementwise trait check on this branch, so we can't easily build a forward
// walker bail case with a single primitive. Phase 1b Case C already covers
// the bail path (via tt.broadcast). When the forward walker grows
// dim-flipping ops in Phase 2, we'll add proper bail-trigger cases here.)
