// RUN: triton-opt %s --tritongpu-transpose-propagate | FileCheck %s

// D14: multiple roots + TransElide pattern (FA-bwd / HSTU-style).
// Two annotated tt.dot ops in the same func, each starting its own
// closure. TransElide fires on the tt.trans of silu(qk) (the user's
// FA-bwd example: trans(silu) -> silu_t naturally).

// CHECK-LABEL: tt.func @fa_bwd_shape
// CHECK-NOT: tt.transpose_propagate_root
// Two roots, both kept in place after commit.
// CHECK: tt.dot
// CHECK: tt.dot

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedT = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Mimics the user's FA-bwd example structure:
  //   qk = q . trans(k)            (annotated)
  //   silu = math.exp2 qk          (stand-in for silu)
  //   silu_t = trans(silu)         (TransElide -> the un-transposed silu)
  //   dv += silu_t . do            (sink for silu chain)
  //   dqk = do . trans(v)          (annotated)
  //   scaled = arith.mulf dqk, f   (elementwise on closure)
  //   dk += trans(scaled) . q      (sink for dqk chain)
  tt.func @fa_bwd_shape(
      %q: tensor<16x16xf16, #dotA>,
      %k_t: tensor<16x16xf16, #dotB>,
      %do_dotA: tensor<16x16xf16, #dotA>,
      %v_t: tensor<16x16xf16, #dotB>,
      %f_scalar: tensor<16x16xf32, #blocked>,
      %silu_t_dotA: tensor<16x16xf16, #dotA>,
      %do_dotB: tensor<16x16xf16, #dotB>,
      %scaled_dotA: tensor<16x16xf16, #dotA>,
      %q_dotB: tensor<16x16xf16, #dotB>,
      %dv_init: tensor<16x16xf32, #blocked>,
      %dk_init: tensor<16x16xf32, #blocked>)
        -> (tensor<16x16xf32, #blocked>, tensor<16x16xf32, #blocked>) {
    %z = arith.constant dense<0.0> : tensor<16x16xf32, #blocked>

    // Root 1: QK
    %qk = tt.dot %q, %k_t, %z {tt.transpose_propagate_root}
        : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf32, #blocked>

    // Root 2: DQK
    %dqk = tt.dot %do_dotA, %v_t, %z {tt.transpose_propagate_root}
        : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf32, #blocked>

    // Sink for QK chain: dv += silu_t . do.
    %dv = tt.dot %silu_t_dotA, %do_dotB, %dv_init
        : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf32, #blocked>

    // Sink for DQK chain: dk += trans(scaled) . q.
    %dk = tt.dot %scaled_dotA, %q_dotB, %dk_init
        : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf32, #blocked>

    tt.return %dv, %dk : tensor<16x16xf32, #blocked>, tensor<16x16xf32, #blocked>
  }
}
