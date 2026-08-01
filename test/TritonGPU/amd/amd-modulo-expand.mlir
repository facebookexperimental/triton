// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule=mode=modulo 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SCHEDULE
// RUN: TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule=mode=modulo 2>/dev/null \
// RUN:   | triton-opt -split-input-file -tritonamdgpu-pipeline 2>&1 \
// RUN:   | FileCheck %s
//
// Modulo runs before the guarded legacy scheduler. A successful modulo schedule
// is preserved; the standard AMD pipeline lowers and expands it.

// SCHEDULE: remark: amd-modulo:{{.*}}II={{[0-9]+}} maxStage=1{{.*}}serialized num_stages=2
// SCHEDULE-NOT: triton.warp_pipeline.border
// CHECK-LABEL: tt.func @early_lower
// The standard pipeline may choose register pipelining when this single-load
// fixture is not profitable/legal for LDS async copy. Verify the load is peeled
// into the prologue and forwarded as a loop-carried tensor to the stage-1 dot.
// CHECK:       tt.load {{.*}}amd.pipeliner_part = "prologue"
// CHECK:       scf.for {{.*}}iter_args({{.*}}tensor<256x64xf16, #blocked>)
// CHECK:         tt.load
// CHECK:         ttg.convert_layout {{.*}}tensor<256x64xf16, #blocked>
// CHECK:         tt.dot

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @early_lower(
      %a_ptrs: tensor<256x64x!tt.ptr<f16>, #blocked>,
      %b: tensor<64x256xf16, #dot1>,
      %c_init: tensor<256x256xf32, #mma>,
      %lb: index, %ub: index, %step: index) -> tensor<256x256xf32, #mma> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %c_init)
        -> (tensor<256x256xf32, #mma>) {
      %a_ld = tt.load %a_ptrs : tensor<256x64x!tt.ptr<f16>, #blocked>
      %a = ttg.convert_layout %a_ld : tensor<256x64xf16, #blocked>
              -> tensor<256x64xf16, #dot0>
      %d = tt.dot %a, %b, %acc :
          tensor<256x64xf16, #dot0> * tensor<64x256xf16, #dot1>
          -> tensor<256x256xf32, #mma>
      scf.yield %d : tensor<256x256xf32, #mma>
    }
    tt.return %res : tensor<256x256xf32, #mma>
  }
}
