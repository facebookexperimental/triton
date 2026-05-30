// RUN: TRITON_ENABLE_TTGIR_SCHED=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule 2>&1 | FileCheck %s
//
// Phase 1b of the TTGIR-level SCHED pass: backward walker reports how many
// producer ops would be co-partitioned along the M dim. No IR mutation.
//
// Walked from: dot's accumulator C (output dim 0) and dot's operand A
// (LHS dim 0 — same M as the result). Operand B is NOT walked — M-split
// doesn't touch B.
//
// Recognised ops (subset of WSDataPartition's allowed list, stripped of
// Hopper-only branches; see DotDecomposeAndSchedule.cpp comments):
//   Elementwise trait, arith.constant, arith.ext{si,ui,f},
//   tt.splat, tt.addptr, tt.load,
//   ttg.convert_layout, ttg.local_alloc, ttg.local_load,
//   amdgpu.buffer_load_to_local.
//
// Unknown ops cause the backward walker to bail; the dot is reported as
// "bwd-infeasible" but the loop is still flagged as a candidate.

// Case A: a v8-shaped main loop. A operand is fed by ttg.local_load. The
// memdesc operand to local_load is a function arg (leaf), so the backward
// walker visits only ttg.local_load itself. Accumulator C comes from the
// scf.for iter-arg (block arg → leaf), no additional ops co-partitioned.
// Expected total: 1 producer op co-partitioned (the local_load).

// CHECK: remark: ttgir-sched: would M-split this dot into 8 (blockM=256 / ctaTileM=32), co-partitioning 1 producer op(s) + 1 user op(s) (phase 2: plan only)
// CHECK: remark: ttgir-sched: candidate inner loop with 1 MFMA tt.dot op(s); plans 1, skipped 0, bwd-infeasible 0, fwd-infeasible 0, applied 0, co-partition producer-ops 1 + user-ops 1 (phase 2: plan only)

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem   = #ttg.shared_memory
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @v8_like_dot_with_local_load(
      %a_smem: !ttg.memdesc<256x64xf16, #shared, #smem, mutable>,
      %arg_b: tensor<64x128xf16, #dot1>,
      %arg_c_init: tensor<256x128xf32, #mma>,
      %lb: index, %ub: index, %step: index) -> tensor<256x128xf32, #mma> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %arg_c_init)
        -> (tensor<256x128xf32, #mma>) {
      %a = ttg.local_load %a_smem :
          !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          -> tensor<256x64xf16, #dot0>
      %d = tt.dot %a, %arg_b, %acc :
          tensor<256x64xf16, #dot0> * tensor<64x128xf16, #dot1>
          -> tensor<256x128xf32, #mma>
      scf.yield %d : tensor<256x128xf32, #mma>
    }
    tt.return %res : tensor<256x128xf32, #mma>
  }
}

// -----

// Case B: a v10-shaped main loop where operand A is consumed directly from
// a block argument (modelled as a plain function arg here). Backward walker
// visits no producer ops — both A and C are block arguments / leaves.
// Expected total: 0 producer ops co-partitioned. Plan is still emitted.

// CHECK: remark: ttgir-sched: would M-split this dot into 8 (blockM=256 / ctaTileM=32), co-partitioning 0 producer op(s) + 1 user op(s) (phase 2: plan only)
// CHECK: remark: ttgir-sched: candidate inner loop with 1 MFMA tt.dot op(s); plans 1, skipped 0, bwd-infeasible 0, fwd-infeasible 0, applied 0, co-partition producer-ops 0 + user-ops 1 (phase 2: plan only)

#mma_b = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0_b = #ttg.dot_op<{opIdx = 0, parent = #mma_b, kWidth = 8}>
#dot1_b = #ttg.dot_op<{opIdx = 1, parent = #mma_b, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @v10_dot_no_producer_chain(
      %arg_a: tensor<256x64xf16, #dot0_b>,
      %arg_b: tensor<64x128xf16, #dot1_b>,
      %arg_c_init: tensor<256x128xf32, #mma_b>,
      %lb: index, %ub: index, %step: index) -> tensor<256x128xf32, #mma_b> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %arg_c_init)
        -> (tensor<256x128xf32, #mma_b>) {
      %d = tt.dot %arg_a, %arg_b, %acc :
          tensor<256x64xf16, #dot0_b> * tensor<64x128xf16, #dot1_b>
          -> tensor<256x128xf32, #mma_b>
      scf.yield %d : tensor<256x128xf32, #mma_b>
    }
    tt.return %res : tensor<256x128xf32, #mma_b>
  }
}

// -----

// Case C: backward walker bails because operand A is produced by an unknown
// op (we use tt.broadcast as a stand-in for "not on the allow-list" — its
// dim-flipping logic is deferred to Phase 1c+). The loop is still a
// candidate, but the per-dot plan is marked bwd-infeasible.

// CHECK: remark: ttgir-sched: would M-split this dot into 8 (blockM=256 / ctaTileM=32), but backward walker bailed (phase 2: plan only)
// CHECK: remark: ttgir-sched: candidate inner loop with 1 MFMA tt.dot op(s); plans 1, skipped 0, bwd-infeasible 1, fwd-infeasible 0, applied 0, co-partition producer-ops 0 + user-ops 0 (phase 2: plan only)

#mma_c = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0_c = #ttg.dot_op<{opIdx = 0, parent = #mma_c, kWidth = 8}>
#dot1_c = #ttg.dot_op<{opIdx = 1, parent = #mma_c, kWidth = 8}>
#dot0_c_slice = #ttg.slice<{dim = 1, parent = #dot0_c}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @bwd_walker_bails_on_unknown_op(
      %arg_a_col: tensor<256xf16, #dot0_c_slice>,
      %arg_b: tensor<64x128xf16, #dot1_c>,
      %arg_c_init: tensor<256x128xf32, #mma_c>,
      %lb: index, %ub: index, %step: index) -> tensor<256x128xf32, #mma_c> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %arg_c_init)
        -> (tensor<256x128xf32, #mma_c>) {
      %expanded = tt.expand_dims %arg_a_col {axis = 1 : i32} :
          tensor<256xf16, #dot0_c_slice> -> tensor<256x1xf16, #dot0_c>
      %a = tt.broadcast %expanded :
          tensor<256x1xf16, #dot0_c> -> tensor<256x64xf16, #dot0_c>
      %d = tt.dot %a, %arg_b, %acc :
          tensor<256x64xf16, #dot0_c> * tensor<64x128xf16, #dot1_c>
          -> tensor<256x128xf32, #mma_c>
      scf.yield %d : tensor<256x128xf32, #mma_c>
    }
    tt.return %res : tensor<256x128xf32, #mma_c>
  }
}
