// RUN: triton-opt %s --tritongpu-transpose-propagate | FileCheck %s

// D9 commit: the simplest commit-able plan -- dot (root) -> truncf -> return.
// Expected IR transformations:
//   * tt.trans inserted right after the root dot (lazy root).
//   * arith.truncf rewritten on the trans'd value (now in transposed
//     orientation, encoding adjusted via convert_layout if needed).
//   * tt.trans inserted at the tt.return boundary to bring the value
//     back to original orientation.
//   * The annotation attribute is stripped from the root.

// CHECK-LABEL: tt.func @commit_simple_chain
// CHECK: tt.dot
// CHECK-NOT: tt.transpose_propagate_root
// CHECK: tt.trans
// CHECK: arith.truncf
// CHECK: tt.trans
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @commit_simple_chain(
      %a: tensor<16x16xf32, #dotA>,
      %b: tensor<16x16xf32, #dotB>,
      %c: tensor<16x16xf32, #blocked>) -> tensor<16x16xf16, #blocked> {
    %dot = tt.dot %a, %b, %c {tt.transpose_propagate_root}
        : tensor<16x16xf32, #dotA> * tensor<16x16xf32, #dotB> -> tensor<16x16xf32, #blocked>
    %p = arith.truncf %dot : tensor<16x16xf32, #blocked> to tensor<16x16xf16, #blocked>
    tt.return %p : tensor<16x16xf16, #blocked>
  }
}
