// RUN: TRITON_TRANSPOSE_PROPAGATE_DEBUG=1 triton-opt %s --tritongpu-transpose-propagate 2>&1 | FileCheck %s

// D8: scf.for with annotated dot inside, accumulator carried as iter_arg.
// dot is root; mulf inside the loop uses dot.result; scf.yield carries
// mulf result back to acc. SCFCarryRetype classifies the yield and the
// iter_arg's downstream uses become in-closure.
//
// Plan:
//   ops = {dot, mulf, scf.yield}      = 3
//   boundary = {tt.return on scf.for.result, possibly mulf.users via iter_arg loop}

// CHECK: plan roots=1
// CHECK-SAME: ops=3
// CHECK: tt.func @scf_for_loop

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
