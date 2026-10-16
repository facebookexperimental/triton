// The coalesce pass factors an NPOT contiguous dim into a VALID blocked encoding
// whose threadsPerWarp is pow2 and tiles the warp (product == 64); the NPOT
// remainder is carried by the tensor shape (the modular LinearLayout). Without
// the fix the factorization is threadsPerWarp = [10, 6] (= 60) / [1, 48] etc. --
// a non-pow2 factor that does not tile the 64-lane warp and is rejected by
// BlockedEncodingAttr::verify. warpSize is 64 (CDNA).
//
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s

// N=48 contiguous, 8 elems/thread (dot A-operand style). This is the exact case
// that produced the invalid threadsPerWarp = [10, 6]: the NPOT lane factor
// (48/8 = 6) floors to a pow2 (4), the rest of the warp fills the strided dim
// (64/4 = 16), so threadsPerWarp = [16, 4] tiles the warp and sizePerThread = 8
// carries the vectorized load.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: [[$COALESCED:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
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
    // CHECK: tt.load {{.*}} : tensor<64x48x!tt.ptr<f16>, [[$COALESCED]]>
    %10 = tt.load %9 : tensor<64x48x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// N=48 contiguous, 1 elem/thread. The NPOT lane factor (48) floors to a pow2
// (32) and the strided dim takes the remaining 2 lanes: threadsPerWarp = [2, 32].
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: [[$COALESCED:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @npot_load_48
  tt.func public @npot_load_48(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<64x48xi32, #blocked>) {
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x48x!tt.ptr<f16>, #blocked>
    %1 = tt.addptr %0, %arg1 : tensor<64x48x!tt.ptr<f16>, #blocked>, tensor<64x48xi32, #blocked>
    // CHECK: tt.load {{.*}} : tensor<64x48x!tt.ptr<f16>, [[$COALESCED]]>
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
  // CHECK: [[$COALESCED:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-LABEL: @npot_load_192
  tt.func public @npot_load_192(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<64x192xi32, #blocked>) {
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x192x!tt.ptr<f16>, #blocked>
    %1 = tt.addptr %0, %arg1 : tensor<64x192x!tt.ptr<f16>, #blocked>, tensor<64x192xi32, #blocked>
    // CHECK: tt.load {{.*}} : tensor<64x192x!tt.ptr<f16>, [[$COALESCED]]>
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
  // CHECK: [[$POW2LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @pow2_load_64
  tt.func public @pow2_load_64(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<64x64xi32, #blocked>) {
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %1 = tt.addptr %0, %arg1 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    // CHECK: tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>, [[$POW2LAYOUT]]>
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
  // CHECK: [[$RANK3:#.*]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 16, 4], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
  // CHECK-LABEL: @npot_rank3_uses_assigned_threads
  tt.func public @npot_rank3_uses_assigned_threads(%arg0: tensor<1x64x5x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 1, 1]> : tensor<3xi32>}) {
    // CHECK: tt.load {{.*}} : tensor<1x64x5x!tt.ptr<f16>, [[$RANK3]]>
    %0 = tt.load %arg0 : tensor<1x64x5x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// sizePerThread is floored to a pow2 as well. 24x32 f16 over 4 warps of 32 lanes
// is 768 / 128 = 6 elements per thread, under the 8-wide vector limit. An NPOT
// sizePerThread (6) fails BlockedEncodingAttr::verify, and in NDEBUG builds its
// 24-wide CTA tile left columns 24-31 unwritten. It floors to 4, and the modular
// LinearLayout carries the NPOT remainder. warpSize is 32 here.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: [[$COALESCED:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @npot_load_24x32_per_thread_floor
  tt.func public @npot_load_24x32_per_thread_floor(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 24 : i32, start = 0 : i32} : tensor<24xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<24xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<24x1xi32, #blocked>
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %c32 = arith.constant dense<32> : tensor<24x1xi32, #blocked>
    %4 = arith.muli %2, %c32 : tensor<24x1xi32, #blocked>
    %5 = tt.broadcast %4 : tensor<24x1xi32, #blocked> -> tensor<24x32xi32, #blocked>
    %6 = tt.broadcast %3 : tensor<1x32xi32, #blocked> -> tensor<24x32xi32, #blocked>
    %7 = arith.addi %5, %6 : tensor<24x32xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<24x32x!tt.ptr<f16>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<24x32x!tt.ptr<f16>, #blocked>, tensor<24x32xi32, #blocked>
    // CHECK: tt.load {{.*}} : tensor<24x32x!tt.ptr<f16>, [[$COALESCED]]>
    %10 = tt.load %9 : tensor<24x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// Column-major counterpart: 64x24 i8 with dim 0 contiguous, 4 warps of 32
// lanes. 1536 / 128 = 12 elements per thread, under the 16-wide vector limit;
// 12 floors to 8 (unfloored, the 48-row CTA tile left rows 48-63 unwritten).
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: [[$COALESCED:#.*]] = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
  // CHECK-LABEL: @npot_load_64x24_colmajor_per_thread_floor
  tt.func public @npot_load_64x24_colmajor_per_thread_floor(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 24 : i32, start = 0 : i32} : tensor<24xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<24xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x24xi32, #blocked>
    %c64 = arith.constant dense<64> : tensor<1x24xi32, #blocked>
    %4 = arith.muli %3, %c64 : tensor<1x24xi32, #blocked>
    %5 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x24xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<1x24xi32, #blocked> -> tensor<64x24xi32, #blocked>
    %7 = arith.addi %5, %6 : tensor<64x24xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<64x24x!tt.ptr<i8>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<64x24x!tt.ptr<i8>, #blocked>, tensor<64x24xi32, #blocked>
    // CHECK: tt.load {{.*}} : tensor<64x24x!tt.ptr<i8>, [[$COALESCED]]>
    %10 = tt.load %9 : tensor<64x24x!tt.ptr<i8>, #blocked>
    tt.return
  }
}

// -----

// Descriptor loads get the same pow2 floor. 8x96 f16 over 4 warps of 32 lanes
// is 768 / 128 = 6 elements per thread, under the 8-wide vector limit. 6 fails
// BlockedEncodingAttr::verify; it floors to 4.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: [[$DESC:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @npot_descriptor_load_8x96_per_thread_floor
  tt.func public @npot_descriptor_load_8x96_per_thread_floor(%arg0: !tt.tensordesc<8x96xf16, #shared>) -> tensor<8x96xf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    // CHECK: tt.descriptor_load {{.*}} -> tensor<8x96xf16, [[$DESC]]>
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<8x96xf16, #shared> -> tensor<8x96xf16, #blocked>
    tt.return %0 : tensor<8x96xf16, #blocked>
  }
}

// -----

// local_load gets the same pow2 floor. A shared buffer's leading dim may be
// NPOT, so 24x16 f16 over 4 warps of 32 lanes is 384 / 128 = 3 elements per
// thread. 3 fails BlockedEncodingAttr::verify; it floors to 2.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: [[$LL:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @npot_local_load_24x16_per_thread_floor
  tt.func public @npot_local_load_24x16_per_thread_floor(%arg0: !ttg.memdesc<24x16xf16, #shared, #smem>) -> tensor<24x16xf16, #blocked> {
    // CHECK: ttg.local_load {{.*}} -> tensor<24x16xf16, [[$LL]]>
    %0 = ttg.local_load %arg0 : !ttg.memdesc<24x16xf16, #shared, #smem> -> tensor<24x16xf16, #blocked>
    tt.return %0 : tensor<24x16xf16, #blocked>
  }
}
