// RUN: triton-opt %s --tritongpu-transpose-propagate | FileCheck %s

// D11: scf.for with iter_arg in the closure. Annotated dot inside the
// loop produces the closure; arith.mulf in the loop consumes it;
// scf.yield carries the result back to acc. SCFCarryRetype commit
// inserts a back-trans before the yield so the loop carries a
// consistent type.

// CHECK-LABEL: tt.func @scf_for_loop
// CHECK-NOT: tt.transpose_propagate_root
// CHECK: scf.for
// CHECK: tt.dot
// CHECK: tt.trans
// CHECK: arith.mulf
// CHECK: tt.trans
// CHECK: scf.yield

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scf_for_loop(
      %a: tensor<16x16xf32, #dotA>,
      %b: tensor<16x16xf32, #dotB>,
      %s: tensor<16x16xf32, #blocked>,
      %lo: index, %hi: index, %step: index,
      %init: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %r = scf.for %i = %lo to %hi step %step iter_args(%acc = %init) -> tensor<16x16xf32, #blocked> {
      %dot = tt.dot %a, %b, %acc {tt.transpose_propagate_root}
          : tensor<16x16xf32, #dotA> * tensor<16x16xf32, #dotB> -> tensor<16x16xf32, #blocked>
      %prod = arith.mulf %dot, %s : tensor<16x16xf32, #blocked>
      scf.yield %prod : tensor<16x16xf32, #blocked>
    }
    tt.return %r : tensor<16x16xf32, #blocked>
  }
}
