// RUN: not triton-opt %s -convert-triton-to-tritongpu='target=cuda:100 num-warps=2' -relayout-tritongpu 2>&1 | FileCheck %s

// The TMEM layouts in this pass need a power-of-two warp count >= 4.
// With 2 warps the conversion must fail legalization cleanly instead
// of crashing.

// CHECK: error: failed to legalize operation 'ttng.tmem_load' that was explicitly marked illegal
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
tt.func @pot_two_warp_relayout(
    %arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
  %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32>
  tt.return
}
