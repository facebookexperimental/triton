// RUN: triton-opt %s --triton-nvidia-optimize-tmem-layouts | FileCheck %s

// Test TMemSplitLoadPattern with M=64 (BWD attention dq accumulator case).
// A 64x128 TMEM load split into two 64x64 halves should be replaced with
// two tmem_subslice + tmem_load pairs.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_split_load_m64
  tt.func public @tmem_split_load_m64(%arg0: !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> (tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>) {
    // CHECK: %[[S0:.+]] = ttng.tmem_subslice %{{.+}} {N = 0 : i32}
    // CHECK: %[[L0:.+]] = ttng.tmem_load %[[S0]] : !ttg.memdesc<64x64xf32
    // CHECK: %[[C0:.+]] = ttg.convert_layout %[[L0]]
    // CHECK: %[[S1:.+]] = ttng.tmem_subslice %{{.+}} {N = 64 : i32}
    // CHECK: %[[L1:.+]] = ttng.tmem_load %[[S1]] : !ttg.memdesc<64x64xf32
    // CHECK: %[[C1:.+]] = ttg.convert_layout %[[L1]]
    // CHECK: tt.return %[[C0]], %[[C1]]
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked1>
    %1 = tt.reshape %0 : tensor<64x128xf32, #blocked1> -> tensor<64x2x64xf32, #blocked2>
    %2 = tt.trans %1 {order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked2> -> tensor<64x64x2xf32, #blocked3>
    %3 = ttg.convert_layout %2 : tensor<64x64x2xf32, #blocked3> -> tensor<64x64x2xf32, #blocked4>
    %outLHS, %outRHS = tt.split %3 : tensor<64x64x2xf32, #blocked4> -> tensor<64x64xf32, #blocked>
    tt.return %outLHS, %outRHS : tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>
  }
}
