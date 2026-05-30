// RUN: TRITON_TRANSPOSE_PROPAGATE_DEBUG=1 triton-opt %s --tritongpu-transpose-propagate 2>&1 | FileCheck %s

// D5: chained dot via opC (accumulator) consumption. dot1 (root,
// annotated) -> dot2 (consuming dot1's result as opC). Both dots
// end up in the plan; tt.return on dot2's result becomes a boundary.
// (Chaining via opA or opB requires a convert_layout in between,
// which D6 will handle.)

// CHECK: plan roots=1
// CHECK-SAME: ops=2
// CHECK-SAME: boundary=1
// CHECK: tt.func @chained_dots_via_acc

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
