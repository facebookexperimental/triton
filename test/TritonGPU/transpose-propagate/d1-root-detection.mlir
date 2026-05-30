// RUN: TRITON_TRANSPOSE_PROPAGATE_DEBUG=1 triton-opt %s --tritongpu-transpose-propagate 2>&1 | FileCheck %s

// D1: annotated root triggers DFS. With no rules registered, every direct
// user of the root's result is classified as BoundaryInsert (the
// conservative default). Plan summary remark should fire.

// CHECK: plan roots=1
// CHECK-SAME: ops=1
// CHECK-SAME: boundary=1
// CHECK: tt.func @annotated_root_one_dot
// CHECK: tt.dot

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @annotated_root_one_dot(
      %a: tensor<16x16xf16, #dotA>,
      %b: tensor<16x16xf16, #dotB>,
      %c: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %r = tt.dot %a, %b, %c {tt.transpose_propagate_root} : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf32, #blocked>
    tt.return %r : tensor<16x16xf32, #blocked>
  }
}
