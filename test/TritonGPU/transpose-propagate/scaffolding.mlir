// RUN: triton-opt %s --tritongpu-transpose-propagate | FileCheck %s

// D0: scaffolding test. With no rules registered and no annotated dot,
// the pass is a structural no-op: IR round-trips byte-identical.

// CHECK-LABEL: tt.func @no_annotation
// CHECK: tt.dot

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @no_annotation(
      %a: tensor<16x16xf16, #dotA>,
      %b: tensor<16x16xf16, #dotB>,
      %c: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %r = tt.dot %a, %b, %c : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf32, #blocked>
    tt.return %r : tensor<16x16xf32, #blocked>
  }
}
