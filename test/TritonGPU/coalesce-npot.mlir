// The coalesce pass factors an NPOT contiguous dim into a VALID blocked encoding
// whose threadsPerWarp is pow2 and tiles the warp (product == 64); the NPOT
// remainder is carried by the tensor shape (the modular LinearLayout). Without
// the fix the factorization is threadsPerWarp = [10, 6] (= 60) / [1, 48] etc. --
// a non-pow2 factor that does not tile the 64-lane warp and is rejected by
// BlockedEncodingAttr::verify. warpSize is 64 (CDNA).
//
// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s

// N=48 contiguous, 8 elems/thread (dot A-operand style). This is the exact case
// that produced the invalid threadsPerWarp = [10, 6]: the NPOT lane factor
// (48/8 = 6) floors to a pow2 (4), the rest of the warp fills the strided dim
// (64/4 = 16), so threadsPerWarp = [16, 4] tiles the warp and sizePerThread = 8
// carries the vectorized load.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: [[COALESCED:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @npot_load_48_vec8
  tt.func public @npot_load_48_vec8(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 48 : i32, start = 0 : i32} : tensor<48xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<48xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x48xi32, #blocked>
    %c48 = arith.constant dense<48> : tensor<64x1xi32, #blocked>
    %4 = arith.muli %2, %c48 : tensor<64x1xi32, #blocked>
    %5 = tt.broadcast %4 : tensor<64x1xi32, #blocked> -> tensor<64x48xi32, #blocked>
    %6 = tt.broadcast %3 : tensor<1x48xi32, #blocked> -> tensor<64x48xi32, #blocked>
    %7 = arith.addi %5, %6 : tensor<64x48xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x48x!tt.ptr<f16>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<64x48x!tt.ptr<f16>, #blocked>, tensor<64x48xi32, #blocked>
    // CHECK: tt.load {{.*}} : tensor<64x48x!tt.ptr<f16>, [[COALESCED]]>
    %10 = tt.load %9 : tensor<64x48x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// N=48 contiguous, 1 elem/thread. The NPOT lane factor (48) floors to a pow2
// (32) and the strided dim takes the remaining 2 lanes: threadsPerWarp = [2, 32].
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: [[COALESCED:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @npot_load_48
  tt.func public @npot_load_48(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<64x48xi32, #blocked>) {
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x48x!tt.ptr<f16>, #blocked>
    %1 = tt.addptr %0, %arg1 : tensor<64x48x!tt.ptr<f16>, #blocked>, tensor<64x48xi32, #blocked>
    // CHECK: tt.load {{.*}} : tensor<64x48x!tt.ptr<f16>, [[COALESCED]]>
    %2 = tt.load %1 : tensor<64x48x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// N=192 = 3 * 64 is a multiple of the warp size, so the whole warp already tiles
// the contiguous dim (threadsPerWarp = [1, 64]); flooring is a no-op and the
// remaining warps spread over both dims. This case worked before the fix.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: [[COALESCED:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-LABEL: @npot_load_192
  tt.func public @npot_load_192(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<64x192xi32, #blocked>) {
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x192x!tt.ptr<f16>, #blocked>
    %1 = tt.addptr %0, %arg1 : tensor<64x192x!tt.ptr<f16>, #blocked>, tensor<64x192xi32, #blocked>
    // CHECK: tt.load {{.*}} : tensor<64x192x!tt.ptr<f16>, [[COALESCED]]>
    %2 = tt.load %1 : tensor<64x192x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// pow2 path is unchanged under the flag: a [64, 64] load with a fully-tiling
// input encoding stays [1, 64] (the coalesce convert is a no-op). flooring a
// pow2 factor is identity, so the pow2 factorization is byte-identical whether
// the flag is on or off (see coalesce.mlir for the flag-off pow2 suite).
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: [[POW2LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @pow2_load_64
  tt.func public @pow2_load_64(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<64x64xi32, #blocked>) {
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %1 = tt.addptr %0, %arg1 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    // CHECK: tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>, [[POW2LAYOUT]]>
    %2 = tt.load %1 : tensor<64x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// Account for the threads actually assigned after flooring. Dividing by the
// pre-floor factor (5) would leave only 51 apparent threads for dim1, floor its
// warp factor to 2, and spill 2 warps into the size-1 dim0. Those warps would
// duplicate work. The 4 assigned threads leave 64 for dim1, which uses all 4
// warps there.
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 64], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: [[RANK3:#.*]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 16, 4], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
  // CHECK-LABEL: @npot_rank3_uses_assigned_threads
  tt.func public @npot_rank3_uses_assigned_threads(%arg0: tensor<1x64x5x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 1, 1]> : tensor<3xi32>}) {
    // CHECK: tt.load {{.*}} : tensor<1x64x5x!tt.ptr<f16>, [[RANK3]]>
    %0 = tt.load %arg0 : tensor<1x64x5x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
