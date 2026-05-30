// RUN: TRITON_TRANSPOSE_PROPAGATE_DEBUG=1 triton-opt %s --tritongpu-transpose-propagate 2>&1 | FileCheck %s

// D4: TransElide. dot -> tt.trans -> return. The trans of the
// transposed dot result is the original (un-transposed) dot. trans
// gets classified as TransElide (counted as a plan op) but engine does
// NOT recurse on it; the tt.return is therefore not a plan op or
// boundary -- because no DFS edge crosses the trans.

// CHECK: plan roots=1
// CHECK-SAME: ops=2
// CHECK-SAME: boundary=0
// CHECK: tt.func @dot_then_trans

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedT = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @dot_then_trans(
      %a: tensor<16x16xf32, #dotA>,
      %b: tensor<16x16xf32, #dotB>,
      %c: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blockedT> {
    %dot = tt.dot %a, %b, %c {tt.transpose_propagate_root}
        : tensor<16x16xf32, #dotA> * tensor<16x16xf32, #dotB> -> tensor<16x16xf32, #blocked>
    %t = tt.trans %dot {order = array<i32: 1, 0>} : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blockedT>
    tt.return %t : tensor<16x16xf32, #blockedT>
  }
}
