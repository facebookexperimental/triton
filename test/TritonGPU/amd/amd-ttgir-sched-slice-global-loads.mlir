// RUN: TRITON_ENABLE_TTGIR_SCHED=1 TRITON_TTGIR_SCHED_APPLY=1 \
// RUN:   TRITON_TTGIR_SCHED_SLICE_LOADS=1 TRITON_TTGIR_SCHED_SLICE_GLOBAL_LOADS=1 \
// RUN:   triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not="tt.load {{.*}} tensor<64x64x"
//
// Global-load slicing — opt-in via TRITON_TTGIR_SCHED_SLICE_GLOBAL_LOADS.
// Distinct from TRITON_TTGIR_SCHED_SLICE_LOADS (which slices the SMEM
// `local_load` leaf): here the producer chain bottoms out at a *global* `tt.load`
// (no LDS staging). With the global-load knob set, materializeSlice CLONES the
// `tt.load` per partition with a sliced pointer operand, instead of leaving the
// load whole + `extract_slice` (the default — preserves wide/coalesced access).
//
// Without TRITON_TTGIR_SCHED_SLICE_GLOBAL_LOADS the monolithic 64x64 tt.load
// would survive (extract_slice fallback); the --implicit-check-not asserts it is
// gone, i.e. the load was actually sliced.

// Shape: blockM = blockN = K = 64, instr = [16,16,32], warpsPerCTA = [2,2].
// ctaTileM = ctaTileN = 32  ->  numPartitionsM = numPartitionsN = 2  =>  4 sub-dots.
// A chain:  tt.load(#blocked) -> convert_layout(#dot0) -> dot   (sliced along M)
// B chain:  tt.load(#dot1) -> dot                               (sliced along N)

// CHECK-LABEL: tt.func @global_chain_dot_sliced
// CHECK:       scf.for
// A operand: global load sliced along M (2x) + convert_layout cloned (2x).
// CHECK-DAG:   tt.load {{.*}} : tensor<32x64x!tt.ptr<f16>, #blocked
// CHECK-DAG:   tt.load {{.*}} : tensor<32x64x!tt.ptr<f16>, #blocked
// CHECK-DAG:   ttg.convert_layout {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0
// CHECK-DAG:   ttg.convert_layout {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0
// B operand: global load sliced along N (2x), no interior op.
// CHECK-DAG:   tt.load {{.*}} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1
// CHECK-DAG:   tt.load {{.*}} : tensor<64x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1
// 4 sub-dots, glued by one concat.
// CHECK-DAG:   tt.dot {{.*}} -> tensor<32x32xf32
// CHECK-DAG:   tt.dot {{.*}} -> tensor<32x32xf32
// CHECK-DAG:   tt.dot {{.*}} -> tensor<32x32xf32
// CHECK-DAG:   tt.dot {{.*}} -> tensor<32x32xf32
// CHECK:       amdg.concat
// CHECK:       scf.yield

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @global_chain_dot_sliced(
      %a_ptrs: tensor<64x64x!tt.ptr<f16>, #blocked>,
      %b_ptrs: tensor<64x64x!tt.ptr<f16>, #dot1>,
      %c_init: tensor<64x64xf32, #mma>,
      %lb: index, %ub: index, %step: index) -> tensor<64x64xf32, #mma> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %c_init)
        -> (tensor<64x64xf32, #mma>) {
      // A: global tt.load -> convert_layout -> dot  (interior op in the chain)
      %a_ld = tt.load %a_ptrs : tensor<64x64x!tt.ptr<f16>, #blocked>
      %a = ttg.convert_layout %a_ld : tensor<64x64xf16, #blocked>
              -> tensor<64x64xf16, #dot0>
      // B: global tt.load -> dot  (leaf only)
      %b = tt.load %b_ptrs : tensor<64x64x!tt.ptr<f16>, #dot1>
      %d = tt.dot %a, %b, %acc :
          tensor<64x64xf16, #dot0> * tensor<64x64xf16, #dot1>
          -> tensor<64x64xf32, #mma>
      scf.yield %d : tensor<64x64xf32, #mma>
    }
    tt.return %res : tensor<64x64xf32, #mma>
  }
}
