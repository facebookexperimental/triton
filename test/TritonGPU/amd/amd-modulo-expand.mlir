// RUN: triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule=mode=early-lower 2>/dev/null \
// RUN:   | TRITON_USE_MODULO_SCHEDULE=1 triton-opt -split-input-file \
// RUN:       -tritonamdgpu-dot-decompose-and-schedule=mode=modulo 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SCHEDULE
// RUN: triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule=mode=early-lower 2>/dev/null \
// RUN:   | TRITON_USE_MODULO_SCHEDULE=1 triton-opt -split-input-file \
// RUN:       -tritonamdgpu-dot-decompose-and-schedule=mode=modulo 2>/dev/null \
// RUN:   | triton-opt -split-input-file \
// RUN:       -tritonamdgpu-dot-decompose-and-schedule=mode=expand 2>&1 \
// RUN:   | FileCheck %s
//
// The primary modulo path lowers eligible loads and attempts to serialize the
// raw DDG/MRT schedule. The current cost model produces no cross-iteration
// stage for this loop, so expansion deliberately uses the double-buffer fallback.

// SCHEDULE: remark: amd-modulo:{{.*}}maxStage=0{{.*}}serialize=rejected(no-overlap)
// SCHEDULE-NOT: triton.warp_pipeline.border
// CHECK-LABEL: tt.func @early_lower
// double-buffered alloc:
// CHECK:       ttg.local_alloc {{.*}}memdesc<2x256x64xf16
// prologue load peeled out of the loop:
// CHECK:       ttg.async_copy_global_to_local
// CHECK:       scf.for
// steady-state load (prefetch next iter) is ring-indexed:
// CHECK:         ttg.memdesc_index {{.*}}[%
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
