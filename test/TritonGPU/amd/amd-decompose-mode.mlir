// RUN: TRITON_ENABLE_AMD_MODULO=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule="mode=decompose" 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not="amd-modulo: DDG"
//
// Change #2 (two-slot collision fix): an explicit `mode=` pass option overrides
// env-var dispatch. Here TRITON_ENABLE_AMD_MODULO is set — env dispatch would
// route to the modulo scaffold (and emit an "amd-modulo: DDG" remark) — but
// `mode=decompose` forces the M/N-split decompose instead. The
// --implicit-check-not asserts the modulo scaffold did NOT run; the CHECKs assert
// the decompose did (sub-dots + concat). This is what lets the same pass sit at
// both the modulo slot and the decompose slot of the decompose+modulo pipeline.

// CHECK-LABEL: tt.func @decompose_mode
// CHECK:       scf.for
// CHECK:       tt.dot {{.*}} -> tensor<32x32xf32
// CHECK:       amdg.concat

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @decompose_mode(
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
