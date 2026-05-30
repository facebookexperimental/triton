// RUN: TRITON_TRANSPOSE_PROPAGATE_DEBUG=1 triton-opt %s --tritongpu-transpose-propagate 2>&1 | FileCheck %s

// D3: dot -> reduce -> expand_dims -> broadcast chain should reach all 4
// ops as plan ops; only the final consumer (tt.return) becomes a boundary.

// CHECK: plan roots=1
// CHECK-SAME: ops=5
// CHECK-SAME: boundary=1
// CHECK: tt.func @reduce_chain

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
#sliced = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @reduce_chain(
      %a: tensor<16x16xf32, #dotA>,
      %b: tensor<16x16xf32, #dotB>,
      %c: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %dot = tt.dot %a, %b, %c {tt.transpose_propagate_root}
        : tensor<16x16xf32, #dotA> * tensor<16x16xf32, #dotB> -> tensor<16x16xf32, #blocked>
    %r = "tt.reduce"(%dot) <{axis = 1 : i32}> ({
      ^bb0(%x: f32, %y: f32):
        %m = arith.maxnumf %x, %y : f32
        tt.reduce.return %m : f32
      }) : (tensor<16x16xf32, #blocked>) -> tensor<16xf32, #sliced>
    %ex = tt.expand_dims %r {axis = 1 : i32} : tensor<16xf32, #sliced> -> tensor<16x1xf32, #blocked>
    %bc = tt.broadcast %ex : tensor<16x1xf32, #blocked> -> tensor<16x16xf32, #blocked>
    %prod = arith.mulf %dot, %bc : tensor<16x16xf32, #blocked>
    tt.return %prod : tensor<16x16xf32, #blocked>
  }
}
