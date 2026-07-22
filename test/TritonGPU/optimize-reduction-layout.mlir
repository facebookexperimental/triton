// RUN: triton-opt %s -split-input-file --tritongpu-optimize-reduction-layout=min-underparallel=8 | FileCheck %s

// A reduce over the NON-contiguous axis (axis 0) is under-parallelized within the warp:
// only threadsPerWarp[0] = 4 lanes sit on the 128-long reduce axis, and the axis is split
// across 4 warps (warpsPerCTA[0] = 4), so it pays the cross-warp shared-memory stage. The
// pass rewrites the operand to put 32 lanes on the axis and the warps on the kept dim
// (threadsPerWarp = [32, 1], warpsPerCTA = [1, 4]) via a convert_layout. Bit-identical
// because inner_tree is layout-invariant.

// CHECK-DAG: #[[MOVED:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @colreduce
  tt.func public @colreduce(%X: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Out: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<128x1xi32, #blocked>
    %m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %m_0 = tt.expand_dims %m {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %c = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %c_1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c_2 = tt.expand_dims %c_1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %x_4 = arith.muli %m_0, %cst : tensor<128x1xi32, #blocked>
    %x_5 = tt.splat %X : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %x_6 = tt.addptr %x_5, %x_4 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
    %x_7 = tt.broadcast %x_6 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x32x!tt.ptr<f32>, #blocked>
    %x_8 = tt.broadcast %c_2 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %x_9 = tt.addptr %x_7, %x_8 : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>
    %x_10 = tt.load %x_9 : tensor<128x32x!tt.ptr<f32>, #blocked>
    // The pass converts the operand to the moved layout and keeps axis + ordering.
    // CHECK: ttg.convert_layout %{{[0-9]+}} : {{.*}} -> tensor<128x32xf32, #[[MOVED]]>
    // CHECK: "tt.reduce"({{.*}}) <{axis = 0 : i32, reduction_ordering = "inner_tree"}>
    // CHECK: (tensor<128x32xf32, #[[MOVED]]>)
    %s = "tt.reduce"(%x_10) <{axis = 0 : i32, reduction_ordering = "inner_tree"}> ({
    ^bb0(%s_11: f32, %s_12: f32):
      %s_13 = arith.addf %s_11, %s_12 : f32
      tt.reduce.return %s_13 : f32
    }) : (tensor<128x32xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.splat %Out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked1>
    %3 = tt.addptr %2, %c : tensor<32x!tt.ptr<f32>, #blocked1>, tensor<32xi32, #blocked1>
    %4 = ttg.convert_layout %s : tensor<32xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %3, %4 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
