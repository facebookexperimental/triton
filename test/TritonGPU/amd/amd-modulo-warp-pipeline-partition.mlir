// RUN: TRITON_ENABLE_AMD_MODULO=1 triton-opt %s -split-input-file \
// RUN:   -tritonamdgpu-dot-decompose-and-schedule 2>&1 | FileCheck %s
//
// Steps 4.7 + 4.8: AMD warp-pipeline cluster partitioning and s_setprio
// derivation.

// Test 1: Two pipelines (LDS + MFMA).
// A minimal GEMM loop with LDS local_loads feeding a dot.
//
// CHECK: remark: amd-modulo: DDG {{[0-9]+}} nodes MFMA={{[0-9]+}} LDS={{[0-9]+}}
// CHECK-SAME: clusters=
// Step 4.7/4.8 annotate each op with its cluster ID and derived s_setprio: the
// LDS local_loads and the MFMA dot land in different clusters.
// CHECK:      ttg.local_load {{.*}}ttg.warp_pipeline_cluster = {{[0-9]+}}{{.*}}ttg.warp_pipeline_priority = {{[0-9]+}}
// CHECK:      tt.dot {{.*}}ttg.warp_pipeline_cluster = {{[0-9]+}}{{.*}}ttg.warp_pipeline_priority = {{[0-9]+}}

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @gemm_lds_mfma(
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

// Test 2: Three pipelines (GLOBAL + LDS + MFMA).
// A GEMM loop with global loads, local stores/loads, and a dot.
//
// CHECK: remark: amd-modulo: DDG {{[0-9]+}} nodes{{.*}}GLOBAL={{[1-9]}}
// CHECK-SAME: clusters=
// GLOBAL/LDS ops and the MFMA dot are annotated with cluster + priority.
// CHECK:      tt.load {{.*}}ttg.warp_pipeline_cluster = {{[0-9]+}}{{.*}}ttg.warp_pipeline_priority = {{[0-9]+}}
// CHECK:      tt.dot {{.*}}ttg.warp_pipeline_cluster = {{[0-9]+}}{{.*}}ttg.warp_pipeline_priority = {{[0-9]+}}

#mma2 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#dot0_2 = #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 4}>
#dot1_2 = #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 4}>
#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared2 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared2t = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem2 = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @gemm_global_lds_mfma(
      %a_ptr: tensor<128x64x!tt.ptr<f16>, #blocked_a>,
      %b_ptr: tensor<64x128x!tt.ptr<f16>, #blocked_b>,
      %a_buf: !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>,
      %b_buf: !ttg.memdesc<64x128xf16, #shared2t, #smem2, mutable>,
      %c_init: tensor<128x128xf32, #mma2>,
      %lb: index, %ub: index, %step: index) -> tensor<128x128xf32, #mma2> {
    %res:3 = scf.for %iv = %lb to %ub step %step
        iter_args(%acc = %c_init,
                  %pa = %a_ptr, %pb = %b_ptr)
        -> (tensor<128x128xf32, #mma2>,
            tensor<128x64x!tt.ptr<f16>, #blocked_a>,
            tensor<64x128x!tt.ptr<f16>, #blocked_b>) {
      // GLOBAL: async loads from HBM
      %a_data = tt.load %pa : tensor<128x64x!tt.ptr<f16>, #blocked_a>
      %b_data = tt.load %pb : tensor<64x128x!tt.ptr<f16>, #blocked_b>
      // LDS: store to shared, then load with dot_op layout
      ttg.local_store %a_data, %a_buf : tensor<128x64xf16, #blocked_a>
          -> !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>
      ttg.local_store %b_data, %b_buf : tensor<64x128xf16, #blocked_b>
          -> !ttg.memdesc<64x128xf16, #shared2t, #smem2, mutable>
      %a = ttg.local_load %a_buf : !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>
           -> tensor<128x64xf16, #dot0_2>
      %b = ttg.local_load %b_buf : !ttg.memdesc<64x128xf16, #shared2t, #smem2, mutable>
           -> tensor<64x128xf16, #dot1_2>
      // MFMA: matrix multiply
      %d = tt.dot %a, %b, %acc :
          tensor<128x64xf16, #dot0_2> * tensor<64x128xf16, #dot1_2>
          -> tensor<128x128xf32, #mma2>
      scf.yield %d, %pa, %pb : tensor<128x128xf32, #mma2>,
          tensor<128x64x!tt.ptr<f16>, #blocked_a>,
          tensor<64x128x!tt.ptr<f16>, #blocked_b>
    }
    tt.return %res#0 : tensor<128x128xf32, #mma2>
  }
}

// -----

// Test 3: Single pipeline (MFMA only).
// A loop with only a dot and loop-carried operands. Only one pipeline (MFMA)
// is active, so warp-pipelining is not applicable.
//
// With a single active pipeline there is no warp-pipelining: the remark reports
// clusters=1(no-warp-pipeline) and no cluster/priority attributes are emitted.
// CHECK: remark: amd-modulo: DDG {{[0-9]+}} nodes MFMA=1 LDS=0 GLOBAL=0
// CHECK-SAME: clusters=1(no-warp-pipeline)

#mma3 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0_3 = #ttg.dot_op<{opIdx = 0, parent = #mma3, kWidth = 8}>
#dot1_3 = #ttg.dot_op<{opIdx = 1, parent = #mma3, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @mfma_only(
      %a: tensor<256x64xf16, #dot0_3>,
      %b: tensor<64x256xf16, #dot1_3>,
      %c_init: tensor<256x256xf32, #mma3>,
      %lb: index, %ub: index, %step: index) -> tensor<256x256xf32, #mma3> {
    %res:3 = scf.for %iv = %lb to %ub step %step
        iter_args(%acc = %c_init, %ai = %a, %bi = %b)
        -> (tensor<256x256xf32, #mma3>,
            tensor<256x64xf16, #dot0_3>,
            tensor<64x256xf16, #dot1_3>) {
      %d = tt.dot %ai, %bi, %acc :
          tensor<256x64xf16, #dot0_3> * tensor<64x256xf16, #dot1_3>
          -> tensor<256x256xf32, #mma3>
      scf.yield %d, %ai, %bi : tensor<256x256xf32, #mma3>,
          tensor<256x64xf16, #dot0_3>,
          tensor<64x256xf16, #dot1_3>
    }
    tt.return %res#0 : tensor<256x256xf32, #mma3>
  }
}
