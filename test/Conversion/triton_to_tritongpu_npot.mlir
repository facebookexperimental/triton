// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=1' | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s --check-prefix=W4

// The pow2 part takes lanes, then warps, and the odd part is the slowest
// register digit; as for the pow2 default, Coalesce later sets the vector.
// CHECK-DAG: #[[$BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
// CHECK-DAG: #[[$MIXED_RADIX:.*]] = #ttg.linear<{register = {{\[\[}}32], [64], [128]], lane = {{\[\[}}1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [384]}>
// W4-DAG: #[[$MIXED_RADIX:.*]] = #ttg.linear<{register = {{\[\[}}2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], lane = {{\[\[}}0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = {{\[\[}}0, 32], [1, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [96, 64]}>
// CHECK-DAG: #[[$K96BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
// W4-DAG: #[[$K96BLOCKED_W4:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$DENSE:.*]] = #ttg.linear<{register = {{\[\[}}0, 1], [0, 2], [1, 0], [2, 0], [16, 0], [32, 0], [64, 0], [0, 32]], lane = {{\[\[}}0, 4], [0, 8], [0, 16], [4, 0], [8, 0]], warp = [], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [128, 96]}>
// W4-DAG: #[[$DENSE_W4:.*]] = #ttg.linear<{register = {{\[\[}}0, 1], [0, 2], [1, 0], [2, 0], [64, 0], [0, 32]], lane = {{\[\[}}0, 4], [0, 8], [0, 16], [4, 0], [8, 0]], warp = {{\[\[}}16, 0], [32, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>

// CHECK-LABEL: @copy_384
// CHECK: tt.load {{.*}} : tensor<384x!tt.ptr<f32>, #[[$MIXED_RADIX]]>
// CHECK: tt.store {{.*}} : tensor<384x!tt.ptr<f32>, #[[$MIXED_RADIX]]>
tt.func @copy_384(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %range = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %srcPtrs = tt.addptr %srcs, %range : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  %value = tt.load %srcPtrs : tensor<384x!tt.ptr<f32>>
  %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %dstPtrs = tt.addptr %dsts, %range : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  tt.store %dstPtrs, %value : tensor<384x!tt.ptr<f32>>
  tt.return
}

// W4-LABEL: @copy_96x64
// W4: tt.load {{.*}} : tensor<96x64x!tt.ptr<f32>, #[[$MIXED_RADIX]]>
// W4: tt.store {{.*}} : tensor<96x64x!tt.ptr<f32>, #[[$MIXED_RADIX]]>
tt.func @copy_96x64(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %range = tt.make_range {end = 6144 : i32, start = 0 : i32} : tensor<6144xi32>
  %offsets = tt.reshape %range : tensor<6144xi32> -> tensor<96x64xi32>
  %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<96x64x!tt.ptr<f32>>
  %srcPtrs = tt.addptr %srcs, %offsets : tensor<96x64x!tt.ptr<f32>>, tensor<96x64xi32>
  %value = tt.load %srcPtrs : tensor<96x64x!tt.ptr<f32>>
  %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<96x64x!tt.ptr<f32>>
  %dstPtrs = tt.addptr %dsts, %offsets : tensor<96x64x!tt.ptr<f32>>, tensor<96x64xi32>
  tt.store %dstPtrs, %value : tensor<96x64x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL: @copy_128
// CHECK: tt.load {{.*}} : tensor<128x!tt.ptr<f32>, #[[$BLOCKED]]>
tt.func @copy_128(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %srcPtrs = tt.addptr %srcs, %range : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %value = tt.load %srcPtrs : tensor<128x!tt.ptr<f32>>
  %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %dstPtrs = tt.addptr %dsts, %range : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  tt.store %dstPtrs, %value : tensor<128x!tt.ptr<f32>>
  tt.return
}

// A dot over a K the blocked layout cannot cover keeps the converter's
// operand layouts, but its power-of-two result still takes the dot's
// blocked layout, as a power-of-two dot's does.
// CHECK-LABEL: @dot_k96_pow2_result
// CHECK: tt.dot {{.*}} -> tensor<128x32xf32, #[[$K96BLOCKED]]>
// W4-LABEL: @dot_k96_pow2_result
// W4: tt.dot {{.*}} -> tensor<128x32xf32, #[[$K96BLOCKED_W4]]>
tt.func @dot_k96_pow2_result() {
  %a = arith.constant dense<1.0> : tensor<128x96xf16>
  %b = arith.constant dense<2.0> : tensor<96x32xf16>
  %c = arith.constant dense<0.0> : tensor<128x32xf32>
  %0 = tt.dot %a, %b, %c : tensor<128x96xf16> * tensor<96x32xf16> -> tensor<128x32xf32>
  tt.return
}

// Every mixed-radix dot result takes the dense layout, whatever its users:
// the dot's 4x4 sizePerThread on the pow2 parts, the odd factor slowest.
// CHECK-LABEL: @dot_reduce_n96
// CHECK: tt.dot {{.*}} -> tensor<128x96xf32, #[[$DENSE]]>
// W4-LABEL: @dot_reduce_n96
// W4: tt.dot {{.*}} -> tensor<128x96xf32, #[[$DENSE_W4]]>
tt.func @dot_reduce_n96() {
  %a = arith.constant dense<1.0> : tensor<128x96xf16>
  %b = arith.constant dense<2.0> : tensor<96x96xf16>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  %d = tt.dot %a, %b, %c : tensor<128x96xf16> * tensor<96x96xf16> -> tensor<128x96xf32>
  %e = "tt.reduce"(%d) <{axis = 1 : i32}> ({
  ^bb0(%x: f32, %y: f32):
    %m = arith.maximumf %x, %y : f32
    tt.reduce.return %m : f32
  }) : (tensor<128x96xf32>) -> tensor<128xf32>
  tt.return
}

// A chained dot consumes the dense layout directly.
// CHECK-LABEL: @dot_chain_n96
// CHECK-COUNT-2: tt.dot {{.*}} -> tensor<128x96xf32, #[[$DENSE]]>
// W4-LABEL: @dot_chain_n96
// W4-COUNT-2: tt.dot {{.*}} -> tensor<128x96xf32, #[[$DENSE_W4]]>
tt.func @dot_chain_n96() {
  %a = arith.constant dense<1.0> : tensor<128x96xf16>
  %b = arith.constant dense<2.0> : tensor<96x96xf16>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  %d = tt.dot %a, %b, %c : tensor<128x96xf16> * tensor<96x96xf16> -> tensor<128x96xf32>
  %b2 = arith.constant dense<2.0> : tensor<96x96xf32>
  %e = tt.dot %d, %b2, %c : tensor<128x96xf32> * tensor<96x96xf32> -> tensor<128x96xf32>
  tt.return
}
