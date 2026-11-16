// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-accelerate-matmul | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-coalesce-async-copy | FileCheck %s --check-prefix=ASYNC

#ba = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#sa = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#lb = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16], [0, 32]], lane = [[16, 0], [32, 0], [64, 0], [0, 1], [0, 2]], warp = [[0, 4], [0, 8]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#lc = #ttg.linear<{register = [[0, 1], [0, 2], [16, 0], [32, 0], [64, 0], [0, 32]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#ls = #ttg.linear<{register = [[32, 0]], lane = [[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [96, 4]}>
#ld = #ttg.linear<{register = [[32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 1], [0, 2]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [96, 4]}>
// A 96-row B scale tile loads one byte per thread in its own layout, so it
// is loaded in the TMEM store layout instead: four scales per thread. The
// convert after the load is looked through.
// CHECK-DAG: #[[$T:.*]] = #ttg.linear<{register = {{\[}}[0, 1], [0, 2], [32, 0]], lane = {{\[}}[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = {{\[}}[0, 0], [0, 0]]
// CHECK-LABEL: @npot_b_scale
// CHECK: %[[P:.*]] = ttg.convert_layout %{{.*}} : tensor<96x4x!tt.ptr<i8>, #{{.*}}> -> tensor<96x4x!tt.ptr<i8>, #[[$T]]>
// CHECK: %[[L:.*]] = tt.load %[[P]] : tensor<96x4x!tt.ptr<i8>, #[[$T]]>
// CHECK: ttng.tmem_alloc %[[L]] : (tensor<96x4xi8, #[[$T]]>)
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @npot_b_scale(%a: tensor<128x128xf8E4M3FN, #ba>, %sa: tensor<128x4xi8, #sa>, %pb: !tt.ptr<f8E4M3FN>, %psb: !tt.ptr<i8>, %pc: !tt.ptr<f32>) {
    %0 = tt.splat %pb : !tt.ptr<f8E4M3FN> -> tensor<128x96x!tt.ptr<f8E4M3FN>, #lb>
    %b = tt.load %0 : tensor<128x96x!tt.ptr<f8E4M3FN>, #lb>
    %1 = tt.splat %psb : !tt.ptr<i8> -> tensor<96x4x!tt.ptr<i8>, #ls>
    %sb = tt.load %1 : tensor<96x4x!tt.ptr<i8>, #ls>
    %sbc = ttg.convert_layout %sb : tensor<96x4xi8, #ls> -> tensor<96x4xi8, #ld>
    %c = arith.constant dense<0.0> : tensor<128x96xf32, #lc>
    %d = tt.dot_scaled %a scale %sa, %b scale %sbc, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x128xf8E4M3FN, #ba>, tensor<128x4xi8, #sa> * tensor<128x96xf8E4M3FN, #lb>, tensor<96x4xi8, #ld> -> tensor<128x96xf32, #lc>
    %2 = tt.splat %pc : !tt.ptr<f32> -> tensor<128x96x!tt.ptr<f32>, #lc>
    tt.store %2, %d : tensor<128x96x!tt.ptr<f32>, #lc>
    tt.return
  }
}

// -----

#ba = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#bb = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#bc = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#s1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// Control: a 128-row scale tile keeps its load layout and is converted.
// CHECK-LABEL: @pow2_b_scale
// CHECK: %[[L:.*]] = tt.load %{{.*}} : tensor<128x4x!tt.ptr<i8>, #[[$S:.*]]>
// CHECK: %[[C:.*]] = ttg.convert_layout %[[L]] : tensor<128x4xi8, #[[$S]]> -> tensor<128x4xi8, #{{.*}}>
// CHECK: ttng.tmem_alloc %[[C]] : (tensor<128x4xi8, #{{.*}}>)
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @pow2_b_scale(%a: tensor<128x128xf8E4M3FN, #ba>, %b: tensor<128x128xf8E4M3FN, #bb>, %sa: tensor<128x4xi8, #s1>, %psb: tensor<128x4x!tt.ptr<i8>, #s1>, %pc: tensor<128x128x!tt.ptr<f32>, #bc>) {
    %c = arith.constant dense<0.0> : tensor<128x128xf32, #bc>
    %sb = tt.load %psb : tensor<128x4x!tt.ptr<i8>, #s1>
    %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x128xf8E4M3FN, #ba>, tensor<128x4xi8, #s1> * tensor<128x128xf8E4M3FN, #bb>, tensor<128x4xi8, #s1> -> tensor<128x128xf32, #bc>
    tt.store %pc, %d : tensor<128x128x!tt.ptr<f32>, #bc>
    tt.return
  }
}

// -----

#t = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [96, 4]}>
#sh = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
// The pipelined copy of that tile holds 3 words per thread, which no
// power-of-two blocked layout covers; the pass leaves it alone.
// ASYNC-LABEL: @npot_scale_copy
// ASYNC-NOT: ttg.convert_layout
// ASYNC: ttg.async_copy_global_to_local %{{.*}}, %{{.*}} : tensor<96x4x!tt.ptr<i8>, #{{.*}}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @npot_scale_copy(%p: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #t}>>
    %e = tt.expand_dims %r {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #t}>> -> tensor<1x4xi32, #t>
    %b = tt.broadcast %e : tensor<1x4xi32, #t> -> tensor<96x4xi32, #t>
    %s = tt.splat %p : !tt.ptr<i8> -> tensor<96x4x!tt.ptr<i8>, #t>
    %q = tt.addptr %s, %b : tensor<96x4x!tt.ptr<i8>, #t>, tensor<96x4xi32, #t>
    %m = ttg.local_alloc : () -> !ttg.memdesc<96x4xi8, #sh, #smem, mutable>
    %tok = ttg.async_copy_global_to_local %q, %m : tensor<96x4x!tt.ptr<i8>, #t> -> <96x4xi8, #sh, #smem, mutable>
    tt.return
  }
}
