// RUN: TRITON_ENABLE_AMD_MODULO=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule 2>&1 | FileCheck %s
//
// AMDLatencyModel must classify the lowered staged global load
// (ttg.async_copy_global_to_local) as GLOBAL — same long round-trip latency as a
// tt.load — so when modulo runs on an already-lowered loop it prefetches the copy
// ahead of the consuming local_load instead of treating it as free. Verified via
// the scaffold's DDG classification remark.
//
// CHECK: remark: amd-modulo: DDG {{[0-9]+}} nodes {{.*}}GLOBAL=1

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @lowered_loop(
      %a_ptrs: tensor<256x64x!tt.ptr<f16>, #blocked>,
      %b: tensor<64x256xf16, #dot1>,
      %c_init: tensor<256x256xf32, #mma>,
      %lb: index, %ub: index, %step: index) -> tensor<256x256xf32, #mma> {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %c_init)
        -> (tensor<256x256xf32, #mma>) {
      // Lowered staged load: global -> LDS (async), then consume from LDS.
      %tok = ttg.async_copy_global_to_local %a_ptrs, %buf
          : tensor<256x64x!tt.ptr<f16>, #blocked> -> <256x64xf16, #shared, #smem, mutable>
      %ct = ttg.async_commit_group tokens %tok
      %wt = ttg.async_wait %ct {num = 0 : i32}
      %a = ttg.local_load %buf token %wt
          : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #dot0>
      %d = tt.dot %a, %b, %acc :
          tensor<256x64xf16, #dot0> * tensor<64x256xf16, #dot1>
          -> tensor<256x256xf32, #mma>
      scf.yield %d : tensor<256x256xf32, #mma>
    }
    tt.return %res : tensor<256x256xf32, #mma>
  }
}
