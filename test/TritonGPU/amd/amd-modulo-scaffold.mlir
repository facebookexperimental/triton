// RUN: TRITON_ENABLE_AMD_MODULO=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule 2>&1 | FileCheck %s
//
// Phase E0+E1 of the AMD modulo scheduler. E0: the scaffold builds the
// backend-neutral DDG (TritonGPUModuloCore) for each inner loop using
// AMDLatencyModel and reports the per-pipeline node classification — the runtime
// test gate for AMDLatencyModel (tt.dot+AMDMfmaEncodingAttr → MFMA,
// ttg.local_load → LDS). E1: it runs the core modulo scheduler (rau by default),
// reports the II, and annotates each op with ttg.modulo_stage / ttg.modulo_order.
// See claude/amd_modulo_scheduling_plan.md (Phase E0/E1).

// E0: one MFMA tt.dot fed by two LDS local_loads → MFMA=1, LDS=2, GLOBAL=0.
// E1: a valid II + max stage are reported. E2: the loop is reordered into modulo
// order with a sched.barrier at the stage boundary.
// (The remark is on stderr and may print before the IR, so no CHECK-LABEL.)
// CHECK: remark: amd-modulo: DDG {{[0-9]+}} nodes MFMA=1 LDS=2 GLOBAL=0{{.*}}II={{[0-9]+}} maxStage={{[0-9]+}}{{.*}}barriers={{[0-9]+}}
// E1/E2: ops carry the schedule, and the body is reordered with a barrier at the
// stage boundary — prefetch-stage loads, then the barrier, then the compute dot.
// CHECK:      ttg.local_load {{.*}}ttg.modulo_stage
// CHECK:      rocdl.sched.barrier
// CHECK:      tt.dot {{.*}}ttg.modulo_stage = {{[0-9]+}} : i32

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_mfma_loop(
      %a_mem: !ttg.memdesc<32x32xf16, #shared, #smem>,
      %b_mem: !ttg.memdesc<32x32xf16, #shared, #smem>,
      %c_init: tensor<32x32xf32, #mma>,
      %lb: index, %ub: index, %step: index) -> tensor<32x32xf32, #mma> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %c_init)
        -> (tensor<32x32xf32, #mma>) {
      %a = ttg.local_load %a_mem : !ttg.memdesc<32x32xf16, #shared, #smem>
           -> tensor<32x32xf16, #dot0>
      %b = ttg.local_load %b_mem : !ttg.memdesc<32x32xf16, #shared, #smem>
           -> tensor<32x32xf16, #dot1>
      %d = tt.dot %a, %b, %acc :
          tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1>
          -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    tt.return %res : tensor<32x32xf32, #mma>
  }
}
