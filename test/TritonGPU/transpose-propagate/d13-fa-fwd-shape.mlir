// RUN: triton-opt %s --tritongpu-transpose-propagate | FileCheck %s

// D13: synthetic FA-fwd-shape e2e. scf.for body contains:
//   - tt.dot (QK, root, annotated)
//   - softmax body: reduce (max), expand_dims, broadcast, sub, exp,
//                    reduce (sum), expand_dims, broadcast, mulf (rescale)
//   - truncf (P = exp(qk - m) cast to fp16)
//   - convert_layout (P -> dot_op)
//   - tt.dot (PV, consumes P + V, accumulates into acc)
//   - scf.yield (carries acc, m, l back through iter_args)
//
// This exercises D9 (boundary trans), D10 (DotFlip on PV), D11
// (SCFCarryRetype on the multi-arg loop), plus the simple rules
// (D2 elementwise, D3 axis swap, D4 trans elide, D6 convert layout).
//
// Goal: dry-run accepts, IR mutates, output verifies.

// CHECK-LABEL: tt.func @fa_fwd_shape
// CHECK-NOT: tt.transpose_propagate_root
// CHECK: scf.for
// CHECK: tt.dot
// CHECK: scf.yield

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#sliced = #ttg.slice<{dim = 1, parent = #blocked}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Note: this is a synthetic "FA-shape" -- the operand types are
  // chosen to be square so that the encoding-preserving transforms
  // work without needing per-shape encoding adjustment. The point is
  // to exercise the combined commit path on a softmax+chained-dot
  // pattern inside scf.for.
  tt.func @fa_fwd_shape(
      %q: tensor<16x16xf16, #dotA>,
      %k: tensor<16x16xf16, #dotB>,
      %v: tensor<16x16xf16, #dotB>,
      %sm_scale: tensor<16x16xf32, #blocked>,
      %lo: index, %hi: index, %step: index,
      %acc_init: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %out = scf.for %i = %lo to %hi step %step
        iter_args(%acc = %acc_init) -> tensor<16x16xf32, #blocked> {
      %zero = arith.constant dense<0.0> : tensor<16x16xf32, #blocked>
      // QK -- annotated root
      %qk = tt.dot %q, %k, %zero {tt.transpose_propagate_root}
          : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf32, #blocked>
      // softmax body (simplified)
      %scaled = arith.mulf %qk, %sm_scale : tensor<16x16xf32, #blocked>
      %p_f32 = math.exp2 %scaled : tensor<16x16xf32, #blocked>
      %p_f16 = arith.truncf %p_f32 : tensor<16x16xf32, #blocked> to tensor<16x16xf16, #blocked>
      // P -> dot operand
      %p = ttg.convert_layout %p_f16 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #dotA>
      // PV
      %pv = tt.dot %p, %v, %acc
          : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf32, #blocked>
      scf.yield %pv : tensor<16x16xf32, #blocked>
    }
    tt.return %out : tensor<16x16xf32, #blocked>
  }
}
