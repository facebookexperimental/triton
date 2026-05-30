// RUN: TRITON_TRANSPOSE_PROPAGATE_DEBUG=1 triton-opt %s --tritongpu-transpose-propagate 2>&1 | FileCheck %s

// D2: an annotated dot whose result feeds an arith.mulf should yield
// plan ops=2 (the dot + the mulf), boundary=1 (the tt.return). Without
// the elementwise rule (D1 baseline), it was ops=1 + boundary=1.

// CHECK: plan roots=1
// CHECK-SAME: ops=2
// CHECK-SAME: boundary=1
// CHECK: tt.func @dot_then_mulf

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @dot_then_mulf(
      %a: tensor<16x16xf32, #dotA>,
      %b: tensor<16x16xf32, #dotB>,
      %c: tensor<16x16xf32, #blocked>,
      %s: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %dot = tt.dot %a, %b, %c {tt.transpose_propagate_root}
        : tensor<16x16xf32, #dotA> * tensor<16x16xf32, #dotB> -> tensor<16x16xf32, #blocked>
    %prod = arith.mulf %dot, %s : tensor<16x16xf32, #blocked>
    tt.return %prod : tensor<16x16xf32, #blocked>
  }
}
