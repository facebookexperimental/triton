// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s

// #p96 isn't exact at 64x192, nor #p192 at 64x96 (96 and 256, 192 and 128
// don't divide), so RLC must keep a convert between the reduce and the broadcast.
#p192 = #ttg.linear<{register = [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 64]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [1, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [64, 192]}>
#p96 = #ttg.linear<{register = [[4, 0], [8, 0], [16, 0], [32, 0], [0, 32]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[1, 0], [2, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [64, 96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @forward_onto_broadcast
  // CHECK: %[[R:.*]] = "tt.reduce"
  // CHECK: ttg.convert_layout %[[R]] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[P96:linear[0-9]*]]}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[P192:linear[0-9]*]]}>>
  // CHECK: tt.broadcast %{{.*}} : tensor<64x1xf32, #[[P192]]> -> tensor<64x192xf32, #[[P192]]>
  tt.func public @forward_onto_broadcast(%p: !tt.ptr<f32>) {
    %pa = tt.splat %p : !tt.ptr<f32> -> tensor<64x96x!tt.ptr<f32>, #p96>
    %pb = tt.splat %p : !tt.ptr<f32> -> tensor<64x192x!tt.ptr<f32>, #p192>
    %a = tt.load %pa : tensor<64x96x!tt.ptr<f32>, #p96>
    %r = "tt.reduce"(%a) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %m = arith.maxnumf %x, %y : f32
      tt.reduce.return %m : f32
    }) : (tensor<64x96xf32, #p96>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #p96}>>
    %e = tt.expand_dims %r {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #p96}>> -> tensor<64x1xf32, #p96>
    %c = ttg.convert_layout %e : tensor<64x1xf32, #p96> -> tensor<64x1xf32, #p192>
    %b = tt.broadcast %c : tensor<64x1xf32, #p192> -> tensor<64x192xf32, #p192>
    tt.store %pb, %b : tensor<64x192x!tt.ptr<f32>, #p192>
    tt.return
  }

  // CHECK-LABEL: @backward_into_reduce
  // CHECK: %[[R:.*]] = "tt.reduce"
  // CHECK: ttg.convert_layout %[[R]] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[P96:linear[0-9]*]]}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[P192:linear[0-9]*]]}>>
  tt.func public @backward_into_reduce(%p: !tt.ptr<f32>, %x: f32) {
    %pb = tt.splat %p : !tt.ptr<f32> -> tensor<64x192x!tt.ptr<f32>, #p192>
    %a0 = tt.splat %x : f32 -> tensor<64x96xf32, #p96>
    %a = arith.mulf %a0, %a0 : tensor<64x96xf32, #p96>
    %r = "tt.reduce"(%a) <{axis = 1 : i32}> ({
    ^bb0(%u: f32, %v: f32):
      %m = arith.maxnumf %u, %v : f32
      tt.reduce.return %m : f32
    }) : (tensor<64x96xf32, #p96>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #p96}>>
    %c = ttg.convert_layout %r : tensor<64xf32, #ttg.slice<{dim = 1, parent = #p96}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #p192}>>
    %e = tt.expand_dims %c {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #p192}>> -> tensor<64x1xf32, #p192>
    %b = tt.broadcast %e : tensor<64x1xf32, #p192> -> tensor<64x192xf32, #p192>
    tt.store %pb, %b : tensor<64x192x!tt.ptr<f32>, #p192>
    tt.return
  }
}
