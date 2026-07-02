// RUN: TRITON_ENABLE_AMD_MODULO=1 TRITON_AMD_MODULO_SERIALIZE=1 \
// RUN:   TRITON_USE_MODULO_SCHEDULE=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule 2>&1 | FileCheck %s
//
// LDS-capacity support for the AMD modulo scheduler (change #3 of
// claude/amd_decomp_modulo_pipeline.md). The serialize path sets the pipeline
// depth to min(LDS-feasible, latency-needed):
//   LDS-feasible = floor(160KB / per-iter operand bytes) - 1   (gfx950)
//   needed       = max(1, modulo maxStage)
// computeLDSStageCap derives `ldsCap` purely from the dot's operand tile bytes,
// so it is deterministic; `num_stages` is then min(ldsCap, needed)+1.

// CASE 1: A=256x64, B=64x256 f16 -> 32768 + 32768 = 65536 B/iter
//         ldsCap = floor(160*1024 / 65536) - 1 = 2 - 1 = 1  -> num_stages = 2
// CHECK: remark: amd-modulo:{{.*}}ldsCap=1{{.*}}serialized num_stages=2

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @lds_cap_256(
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

// -----

// CASE 2: smaller tiles A=128x64, B=64x128 f16 -> 16384 + 16384 = 32768 B/iter
//         ldsCap = floor(160*1024 / 32768) - 1 = 5 - 1 = 4
//         (the budget allows a deeper pipeline; `needed` then clamps num_stages)
// CHECK: remark: amd-modulo:{{.*}}ldsCap=4

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @lds_cap_128(
      %a_mem: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b_mem: !ttg.memdesc<64x128xf16, #shared, #smem>,
      %c_init: tensor<128x128xf32, #mma>,
      %lb: index, %ub: index, %step: index) -> tensor<128x128xf32, #mma> {
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %c_init)
        -> (tensor<128x128xf32, #mma>) {
      %a = ttg.local_load %a_mem : !ttg.memdesc<128x64xf16, #shared, #smem>
              -> tensor<128x64xf16, #dot0>
      %b = ttg.local_load %b_mem : !ttg.memdesc<64x128xf16, #shared, #smem>
              -> tensor<64x128xf16, #dot1>
      %d = tt.dot %a, %b, %acc :
          tensor<128x64xf16, #dot0> * tensor<64x128xf16, #dot1>
          -> tensor<128x128xf32, #mma>
      scf.yield %d : tensor<128x128xf32, #mma>
    }
    tt.return %res : tensor<128x128xf32, #mma>
  }
}
