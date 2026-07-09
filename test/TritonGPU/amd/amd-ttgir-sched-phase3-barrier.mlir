// RUN: TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 triton-opt %s \
// RUN:   -split-input-file -tritonamdgpu-dot-decompose-and-schedule \
// RUN:   2>&1 | FileCheck %s --check-prefix=DEFAULT
//
// RUN: TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 \
// RUN:   TRITON_TTGIR_SCHED_BARRIER_STRIDE=0 triton-opt %s \
// RUN:   -split-input-file -tritonamdgpu-dot-decompose-and-schedule \
// RUN:   2>&1 | FileCheck %s --check-prefix=DISABLED
//
// RUN: TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 \
// RUN:   TRITON_TTGIR_SCHED_BARRIER_STRIDE=1 triton-opt %s \
// RUN:   -split-input-file -tritonamdgpu-dot-decompose-and-schedule \
// RUN:   2>&1 | FileCheck %s --check-prefix=PERDOT
//
// Phase 3: insert `rocdl.sched.barrier 0` between scheduling regions to
// prevent LLVM misched from reordering sub-dots across boundaries.
// `TRITON_TTGIR_SCHED_BARRIER_STRIDE` controls the stride:
//   * default (env unset) → one barrier per M-row (stride = numPartitionsN)
//   * 0 → barriers fully disabled (matches Phase 2 IR)
//   * 1 → barrier after every sub-dot
//   * k → barrier after every k-th sub-dot
//
// For the v8-shape input (8 M × 4 N = 32 sub-dots):
//   * default: 7 barriers (after sub-dot #4, #8, ..., #28; not after #32)
//   * stride=0: 0 barriers
//   * stride=1: 31 barriers
// (always n-1 because we skip the trailing barrier — the concat is the
// implicit cap.)

// Default stride: 7 sched.barrier ops.
// DEFAULT-LABEL: tt.func @v8_like_dot_phase3
// DEFAULT-COUNT-32: tt.dot {{.*}} -> tensor<32x32xf32
// DEFAULT-COUNT-7: rocdl.sched.barrier 0
// DEFAULT: amdg.concat
// DEFAULT-NOT: rocdl.sched.barrier

// STRIDE=0: zero sched.barrier ops (= Phase 2 IR shape).
// DISABLED-LABEL: tt.func @v8_like_dot_phase3
// DISABLED-COUNT-32: tt.dot {{.*}} -> tensor<32x32xf32
// DISABLED-NOT: rocdl.sched.barrier

// STRIDE=1: 31 sched.barrier ops (one between every adjacent sub-dot pair).
// PERDOT-LABEL: tt.func @v8_like_dot_phase3
// PERDOT-COUNT-32: tt.dot {{.*}} -> tensor<32x32xf32
// PERDOT-COUNT-31: rocdl.sched.barrier 0
// PERDOT: amdg.concat
// PERDOT-NOT: rocdl.sched.barrier

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @v8_like_dot_phase3(
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
