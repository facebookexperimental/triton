// RUN: TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 TRITON_TTGIR_SCHED_SLICE_LOADS=1 \
// RUN:   triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not="tensor<64x64xf16" \
// RUN:                  --implicit-check-not="tt.dot {{.*}} -> tensor<64x64xf32"
//
// Note: triton-opt prints encodings expanded (#ttg.dot_op<{...}>), not the
// #dot0/#dot1 source aliases — CHECKs match the expanded prefix.
//
// Producer-chain slicing — the K/MN-split on LOADS that feeds the modulo plan.
// SPEC for the not-yet-implemented TRITON_TTGIR_SCHED_SLICE_LOADS=1 path
// (scheme-driven materializeSlice). See claude/ttgir_sched_modulo_plan.md.
//
// Difference vs phase 1d (amd-ttgir-sched-phase1d-apply.mlir): phase 1d only
// `extract_slice`s the dot's REGISTER operands and leaves the `local_load`
// whole. With TRITON_TTGIR_SCHED_SLICE_LOADS=1 the pass rebuilds the producer
// chain the backward walker recorded, PER PARTITION:
//   * the SMEM load is sliced via `memdesc_subslice` + `local_load` (the leaf)
//   * an interior op (here `convert_layout`) is CLONED with the sliced operand
//     and a sliced result type
// and the original monolithic load/convert/dot become dead and are erased.
//
// This is exactly the case the getDefiningOp<LocalLoadOp> shortcut MISSES: a
// `convert_layout` between the load and the dot (the common case when the load
// layout != the dot_op layout).

// Shape: blockM = blockN = K = 64, instr = [16,16,32], warpsPerCTA = [2,2].
// ctaTileM = ctaTileN = 32  ->  numPartitionsM = numPartitionsN = 2.
//   => 4 sub-dots, each tensor<32x32xf32, #mma>.
// A chain:  local_load(#blocked) -> convert_layout(#dot0) -> dot
//   => 2 along M:  2x [ memdesc_subslice<32x64> + local_load + convert_layout ]
// B chain:  local_load(#dot1) -> dot       (leaf only, no interior op)
//   => 2 along N:  2x [ memdesc_subslice<64x32> + local_load ]

// CHECK-LABEL: tt.func @chain_dot_sliced
// CHECK:       scf.for
// A operand: SMEM load sliced along M (2x) + convert_layout cloned (2x).
// CHECK-DAG:   ttg.memdesc_subslice {{.*}} -> !ttg.memdesc<32x64xf16
// CHECK-DAG:   ttg.memdesc_subslice {{.*}} -> !ttg.memdesc<32x64xf16
// CHECK-DAG:   ttg.convert_layout {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0
// CHECK-DAG:   ttg.convert_layout {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0
// B operand: SMEM load sliced along N (2x), no interior op.
// CHECK-DAG:   ttg.memdesc_subslice {{.*}} -> !ttg.memdesc<64x32xf16
// CHECK-DAG:   ttg.memdesc_subslice {{.*}} -> !ttg.memdesc<64x32xf16
// CHECK-DAG:   ttg.local_load {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1
// CHECK-DAG:   ttg.local_load {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1
// 4 sub-dots, glued by one concat.
// CHECK-DAG:   tt.dot {{.*}} -> tensor<32x32xf32
// CHECK-DAG:   tt.dot {{.*}} -> tensor<32x32xf32
// CHECK-DAG:   tt.dot {{.*}} -> tensor<32x32xf32
// CHECK-DAG:   tt.dot {{.*}} -> tensor<32x32xf32
// CHECK:       amdg.concat
// CHECK:       scf.yield
// (--implicit-check-not in RUN asserts the monolithic 64x64 f16 loads + full dot are gone)

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
// Trivial (un-swizzled) shared layout so memdesc_subslice offsets (32) are
// tile-aligned. Real kernels swizzle — that is the genuine wall for sub-tile
// K-slicing (see plan doc caveat #3), but it does not affect this M/N test.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @chain_dot_sliced(
      %a_mem: !ttg.memdesc<64x64xf16, #shared, #smem>,
      %b_mem: !ttg.memdesc<64x64xf16, #shared, #smem>,
      %c_init: tensor<64x64xf32, #mma>,
      %lb: index, %ub: index, %step: index) -> tensor<64x64xf32, #mma> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %c_init)
        -> (tensor<64x64xf32, #mma>) {
      // A: local_load -> convert_layout -> dot  (interior op in the chain)
      %a_ld = ttg.local_load %a_mem : !ttg.memdesc<64x64xf16, #shared, #smem>
              -> tensor<64x64xf16, #blocked>
      %a = ttg.convert_layout %a_ld : tensor<64x64xf16, #blocked>
              -> tensor<64x64xf16, #dot0>
      // B: local_load -> dot  (leaf only)
      %b = ttg.local_load %b_mem : !ttg.memdesc<64x64xf16, #shared, #smem>
              -> tensor<64x64xf16, #dot1>
      %d = tt.dot %a, %b, %acc :
          tensor<64x64xf16, #dot0> * tensor<64x64xf16, #dot1>
          -> tensor<64x64xf32, #mma>
      scf.yield %d : tensor<64x64xf32, #mma>
    }
    tt.return %res : tensor<64x64xf32, #mma>
  }
}
