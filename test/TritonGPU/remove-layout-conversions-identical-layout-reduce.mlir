// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s

// A non-associative float reduce whose result is also used outside the
// backward slice must not be duplicated, even when the downstream
// convert_layout is between physically identical layouts. Duplicated
// reduces may evaluate in a different order and disagree bitwise, so
// the convert must remain.

// CHECK-LABEL: @identical_layout_reduce_kept
// CHECK: ttg.convert_layout

#blocked_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
#slice_a = #ttg.slice<{dim = 1, parent = #blocked_a}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.target" = "cuda:90"} {
  tt.func @identical_layout_reduce_kept(%in: tensor<64x64xf32, #blocked_a>) -> (tensor<64xf32, #slice_a>, tensor<64x1xf32, #blocked_b>) {
    %sum = "tt.reduce"(%in) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %r = arith.addf %a, %b : f32
      tt.reduce.return %r : f32
    }) : (tensor<64x64xf32, #blocked_a>) -> tensor<64xf32, #slice_a>
    %exp = tt.expand_dims %sum {axis = 1 : i32} : tensor<64xf32, #slice_a> -> tensor<64x1xf32, #blocked_a>
    %cvt = ttg.convert_layout %exp : tensor<64x1xf32, #blocked_a> -> tensor<64x1xf32, #blocked_b>
    tt.return %sum, %cvt : tensor<64xf32, #slice_a>, tensor<64x1xf32, #blocked_b>
  }
}
