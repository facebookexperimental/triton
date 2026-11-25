// RUN: triton-opt %s -tritongpu-coalesce='sector-bytes=32' | FileCheck %s --check-prefix=SECTOR
// RUN: triton-opt %s -tritongpu-coalesce | FileCheck %s --check-prefix=DEFAULT

// Scalar rank-2 loads whose AxisInfo contiguity is [1, 1] pick the blocked
// layout touching fewer 32-byte sectors. With sector scoring off (the
// default) every case keeps the contiguity-order layout.

// SECTOR: [[ORIG:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
// SECTOR: [[SWAPPED:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 1], order = [0, 1]}>
// DEFAULT: [[ORIG:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#sliceDim0 = #ttg.slice<{dim = 0, parent = #blocked}>
#sliceDim1 = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {

// A 32x64 FP16 tile with row stride 2 and column stride 64 touches 64
// sectors with threadsPerWarp [1, 64] but 8 sectors with [32, 2].
// SECTOR: tt.func @sector_swap(
// SECTOR: tt.load {{.*}} : tensor<32x64x!tt.ptr<f16>, [[SWAPPED]]>
// DEFAULT: tt.func @sector_swap(
// DEFAULT: tt.load {{.*}} : tensor<32x64x!tt.ptr<f16>, [[ORIG]]>
tt.func @sector_swap(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<32x64xf16, #blocked> {
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #sliceDim1>
  %re = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32, #sliceDim1> -> tensor<32x1xi32, #blocked>
  %c = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #sliceDim0>
  %ce = tt.expand_dims %c {axis = 0 : i32} : tensor<64xi32, #sliceDim0> -> tensor<1x64xi32, #blocked>
  %s0 = arith.constant dense<2> : tensor<32x1xi32, #blocked>
  %s1 = arith.constant dense<64> : tensor<1x64xi32, #blocked>
  %ro = arith.muli %re, %s0 : tensor<32x1xi32, #blocked>
  %co = arith.muli %ce, %s1 : tensor<1x64xi32, #blocked>
  %rob = tt.broadcast %ro : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
  %cob = tt.broadcast %co : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
  %off = arith.addi %rob, %cob : tensor<32x64xi32, #blocked>
  %bs = tt.splat %base : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
  %ptrs = tt.addptr %bs, %off : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
  %v = tt.load %ptrs : tensor<32x64x!tt.ptr<f16>, #blocked>
  tt.return %v : tensor<32x64xf16, #blocked>
}

// Fully scattered in both layouts (64 sectors each): the tie keeps the
// contiguity-order layout.
// SECTOR: tt.func @sector_tie(
// SECTOR: tt.load {{.*}} : tensor<32x64x!tt.ptr<f16>, [[ORIG]]>
// DEFAULT: tt.func @sector_tie(
// DEFAULT: tt.load {{.*}} : tensor<32x64x!tt.ptr<f16>, [[ORIG]]>
tt.func @sector_tie(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<32x64xf16, #blocked> {
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #sliceDim1>
  %re = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32, #sliceDim1> -> tensor<32x1xi32, #blocked>
  %c = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #sliceDim0>
  %ce = tt.expand_dims %c {axis = 0 : i32} : tensor<64xi32, #sliceDim0> -> tensor<1x64xi32, #blocked>
  %s0 = arith.constant dense<1024> : tensor<32x1xi32, #blocked>
  %s1 = arith.constant dense<1048576> : tensor<1x64xi32, #blocked>
  %ro = arith.muli %re, %s0 : tensor<32x1xi32, #blocked>
  %co = arith.muli %ce, %s1 : tensor<1x64xi32, #blocked>
  %rob = tt.broadcast %ro : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
  %cob = tt.broadcast %co : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
  %off = arith.addi %rob, %cob : tensor<32x64xi32, #blocked>
  %bs = tt.splat %base : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
  %ptrs = tt.addptr %bs, %off : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
  %v = tt.load %ptrs : tensor<32x64x!tt.ptr<f16>, #blocked>
  tt.return %v : tensor<32x64xf16, #blocked>
}

// Runtime strides cannot be scored, so the contiguity-order layout is kept.
// SECTOR: tt.func @sector_runtime_stride(
// SECTOR: tt.load {{.*}} : tensor<32x64x!tt.ptr<f16>, [[ORIG]]>
// DEFAULT: tt.func @sector_runtime_stride(
// DEFAULT: tt.load {{.*}} : tensor<32x64x!tt.ptr<f16>, [[ORIG]]>
tt.func @sector_runtime_stride(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %s0: i32, %s1: i32) -> tensor<32x64xf16, #blocked> {
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #sliceDim1>
  %re = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32, #sliceDim1> -> tensor<32x1xi32, #blocked>
  %c = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #sliceDim0>
  %ce = tt.expand_dims %c {axis = 0 : i32} : tensor<64xi32, #sliceDim0> -> tensor<1x64xi32, #blocked>
  %s0t = tt.splat %s0 : i32 -> tensor<32x1xi32, #blocked>
  %s1t = tt.splat %s1 : i32 -> tensor<1x64xi32, #blocked>
  %ro = arith.muli %re, %s0t : tensor<32x1xi32, #blocked>
  %co = arith.muli %ce, %s1t : tensor<1x64xi32, #blocked>
  %rob = tt.broadcast %ro : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
  %cob = tt.broadcast %co : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
  %off = arith.addi %rob, %cob : tensor<32x64xi32, #blocked>
  %bs = tt.splat %base : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
  %ptrs = tt.addptr %bs, %off : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
  %v = tt.load %ptrs : tensor<32x64x!tt.ptr<f16>, #blocked>
  tt.return %v : tensor<32x64xf16, #blocked>
}

}
