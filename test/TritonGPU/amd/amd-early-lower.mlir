// RUN: TRITON_AMD_EARLY_LOWER=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not="tt.load"
//
// Early load lowering (change #1 of claude/amd_decomp_modulo_pipeline.md): a
// global tt.load feeding an MFMA dot is lowered, BEFORE modulo, to a SINGLE-buffer
// staged copy: local_alloc<1 x tile> -> async_copy_global_to_local -> commit ->
// wait -> local_load (replacing the load's uses). This exposes async_copy (GLOBAL
// latency) and local_load (LDS) as distinct ops for a later modulo pass.
// (--implicit-check-not asserts the original global tt.load is gone.)

// CHECK-LABEL: tt.func @early_lower
// CHECK:       ttg.local_alloc {{.*}}memdesc<1x256x64xf16
// CHECK:       scf.for
// CHECK-DAG:     ttg.async_copy_global_to_local
// CHECK-DAG:     ttg.async_commit_group
// CHECK-DAG:     ttg.async_wait
// CHECK-DAG:     ttg.local_load
// CHECK-DAG:     ttg.convert_layout {{.*}}-> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0
// CHECK:         tt.dot
// CHECK:         scf.yield

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
      // Global load feeding the MFMA dot (through a convert_layout).
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
