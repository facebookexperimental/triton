// RUN: TRITON_TRANSPOSE_PROPAGATE_DEBUG=1 triton-opt %s --tritongpu-transpose-propagate 2>&1 | FileCheck %s

// D6: ConvertLayoutAdjust rule. Chain dot1 -> truncf -> convert_layout
// -> dot2. With the convert classified, the DFS reaches dot2 too.

// CHECK: plan roots=1
// CHECK-SAME: ops=4
// CHECK-SAME: boundary=1
// CHECK: tt.func @chained_dots_via_cvt

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @chained_dots_via_cvt(
      %a: tensor<16x16xf16, #dotA>,
      %k: tensor<16x16xf16, #dotB>,
      %c1: tensor<16x16xf32, #blocked>,
      %v: tensor<16x16xf16, #dotB>,
      %c2: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %qk = tt.dot %a, %k, %c1 {tt.transpose_propagate_root}
        : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf32, #blocked>
    %p_f16 = arith.truncf %qk : tensor<16x16xf32, #blocked> to tensor<16x16xf16, #blocked>
    %p = ttg.convert_layout %p_f16 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #dotA>
    %pv = tt.dot %p, %v, %c2 : tensor<16x16xf16, #dotA> * tensor<16x16xf16, #dotB> -> tensor<16x16xf32, #blocked>
    tt.return %pv : tensor<16x16xf32, #blocked>
  }
}
