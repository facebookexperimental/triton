// RUN: TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 triton-opt %s \
// RUN:   -split-input-file -tritonamdgpu-dot-decompose-and-schedule \
// RUN:   2>&1 | FileCheck %s
//
// Phase 1d of the TTGIR-level SCHED pass: actual SSA mutation. Each
// candidate dot is replaced by N small dots sliced along M, glued back via
// amdgpu.concat. The producer chain is NOT modified — extract_slice is a
// no-op at the CTA-tile level, so upstream tile layout is preserved.
//
// Gated behind a separate env var TRITON_TTGIR_SCHED_APPLY=1 (default off)
// so the existing TRITON_ENABLE_TTGIR_SCHED=1 keeps planning-only behavior.

// Case A: v8-shaped dot, blockM=256 blockN=128, instr=[16,16,32],
// warpsPerCTA=[2,2]. ctaTileM=ctaTileN=32. numPartitionsM=8, numPartitionsN=4.
// Total: 8 × 4 = 32 sub-dots each producing tensor<32x32xf32, #mma>.
// Slices: 8 of A, 4 of B, 32 of C → 44 extract_slice.
// 1 concat reassembles tensor<256x128xf32, #mma>.

// CHECK-LABEL: tt.func @v8_like_dot_applied
// CHECK:       scf.for
// CHECK-COUNT-44: amdg.extract_slice
// CHECK-COUNT-32: tt.dot {{.*}} -> tensor<32x32xf32
// CHECK:       amdg.concat
// CHECK-NOT:   tt.dot {{.*}} -> tensor<256x128xf32
// CHECK:       scf.yield

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @v8_like_dot_applied(
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
