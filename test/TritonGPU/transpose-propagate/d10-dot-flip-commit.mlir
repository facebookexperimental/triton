// RUN: triton-opt %s --tritongpu-transpose-propagate | FileCheck %s

// D10: chained dot DotFlip. Two dots, first annotated as root, second's
// opC is the first's result. After commit:
//   * root dot kept unchanged, annotation stripped.
//   * tt.trans inserted on root's result (lazy root).
//   * SECOND dot rewritten with swapped operands per algebraic identity
//     out_t = dot(B^T, A^T, C^T). New A = trans(orig B) + convert to
//     DotOp<opIdx=0, parent=accEnc>; new B = trans(orig A) + convert to
//     DotOp<opIdx=1, parent=accEnc>; new C = trans(orig C) = the
//     in-closure value.
//   * Final tt.trans inserted at tt.return boundary.

// CHECK-LABEL: tt.func @chained_dots_via_acc
// CHECK-NOT: tt.transpose_propagate_root
// First dot kept, lazy root trans inserted after it.
// CHECK: tt.dot
// CHECK: tt.trans
// Second dot rewritten with swapped operand encodings.
// CHECK: tt.dot
// CHECK: tt.trans
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @chained_dots_via_acc(
      %a1: tensor<16x16xf16, #dotA>,
      %b1: tensor<16x16xf16, #dotB>,
      %c1: tensor<16x16xf32, #blocked>,
      %a2: tensor<16x16xf16, #dotA>,
      %b2: tensor<16x16xf16, #dotB>) -> tensor<16x16xf32, #blocked> {
    %d1 = tt.dot %a1, %b1, %c1 {tt.transpose_propagate_root}
        : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf32, #blocked>
    %d2 = tt.dot %a2, %b2, %d1 : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf32, #blocked>
    tt.return %d2 : tensor<16x16xf32, #blocked>
  }
}
