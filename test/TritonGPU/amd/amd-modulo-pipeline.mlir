// RUN: TRITON_ENABLE_AMD_MODULO=1 TRITON_AMD_MODULO_SERIALIZE=1 triton-opt %s \
// RUN:   -split-input-file -tritonamdgpu-dot-decompose-and-schedule 2>&1 | FileCheck %s
//
// RUN: TRITON_ENABLE_AMD_MODULO=1 TRITON_AMD_MODULO_SERIALIZE=1 triton-opt %s \
// RUN:   -split-input-file -tritonamdgpu-dot-decompose-and-schedule 2>/dev/null \
// RUN:   | triton-opt -tritonamdgpu-pipeline | FileCheck %s --check-prefix=EXPAND
//
// Phase E3: the AMD modulo scaffold emits a serialized tt::CoarseSchedule
// (modulo stage → loop.stage; slot = order%II → loop.cluster) so the EXISTING
// AMD pipeline expander (tritonamdgpu-pipeline) does the multi-buffering + loop
// expansion — i.e. modulo replaces tritonamdgpu-schedule-loops, and DECIDES
// num_stages (= maxStage+1) and per-buffer depth (stage span).
// See claude/amd_modulo_scheduling_plan.md (Phase E3).

// (1) The schedule is serialized: ops carry loop.stage/loop.cluster, the loop
//     carries tt.scheduled_max_stage, and num_stages is reported.
// CHECK: remark: amd-modulo: {{.*}}serialized num_stages={{[0-9]+}}
// CHECK-DAG: ttg.local_load {{.*}}loop.cluster = {{[0-9]+}} : i32{{.*}}loop.stage = {{[0-9]+}} : i32
// CHECK-DAG: tt.dot {{.*}}loop.cluster = {{[0-9]+}} : i32{{.*}}loop.stage = {{[0-9]+}} : i32
// CHECK: tt.scheduled_max_stage = {{[0-9]+}} : i32

// (2) The existing expander consumes the modulo schedule without error and the
//     kernel survives expansion.
// EXPAND: tt.func @amd_mfma_loop
// EXPAND: tt.dot

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_mfma_loop(
      %a_mem: !ttg.memdesc<256x64xf16, #shared, #smem>,
      %b_mem: !ttg.memdesc<64x256xf16, #shared, #smem>,
      %c_init: tensor<256x256xf32, #mma>,
      %lb: index, %ub: index, %step: index) -> tensor<256x256xf32, #mma> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %c_init)
        -> (tensor<256x256xf32, #mma>) {
      %a = ttg.local_load %a_mem : !ttg.memdesc<256x64xf16, #shared, #smem>
           -> tensor<256x64xf16, #dot0>
      %b = ttg.local_load %b_mem : !ttg.memdesc<64x256xf16, #shared, #smem>
           -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc :
          tensor<256x64xf16, #dot0> * tensor<64x256xf16, #dot1>
          -> tensor<256x256xf32, #mma>
      scf.yield %d : tensor<256x256xf32, #mma>
    }
    tt.return %res : tensor<256x256xf32, #mma>
  }
}
