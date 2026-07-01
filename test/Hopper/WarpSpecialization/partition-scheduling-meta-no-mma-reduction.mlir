// RUN: triton-opt %s --nvgpu-partition-scheduling-meta | FileCheck %s

// No-MMA reduction kernels (RMS norm / LayerNorm / softmax-only) have no MMA
// and use a plain in-register tt.reduce (not a TMA-atomic reduce). The pass
// must create a 3-warp-group layout: reduction (default, index 0) / load /
// epilogue_store, route all tensor compute into the reduction partition, and
// annotate the scalar offset ops so the tt.warp_specialize verifier passes.
// This path is gated to mmas.empty(); MMA kernels are unaffected.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {

// CHECK-LABEL: @rmsnorm_no_mma
//
// Scalar offset feeds the load and the store: union of their partitions.
// CHECK: arith.muli {{.*}}ttg.partition = array<i32: 1, 2>
// Load -> load partition.
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: 2>
// Square / reduce / rsqrt / normalize -> reduction partition (0).
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: 0>
// CHECK: tt.reduce.return {{.*}}ttg.partition = array<i32: 0>
// CHECK: math.rsqrt {{.*}}ttg.partition = array<i32: 0>
// CHECK: tt.expand_dims {{.*}}ttg.partition = array<i32: 0>
// CHECK: tt.broadcast {{.*}}ttg.partition = array<i32: 0>
// Store-staging alloc stays in reduction (producer side of reduction->store).
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: 0>
// TMA store + its token wait -> epilogue_store partition (1).
// CHECK: ttng.async_tma_copy_local_to_global {{.*}}ttg.partition = array<i32: 1>
// CHECK: ttng.async_tma_store_token_wait {{.*}}ttg.partition = array<i32: 1>
//
// The reduction partition is created first => index 0 (default 4-warp group).
// CHECK: tt.warp_specialize
// CHECK-SAME: ttg.partition.types = ["reduction", "epilogue_store", "load"]
tt.func public @rmsnorm_no_mma(
  %x_desc: !tt.tensordesc<tensor<64x64xf32, #shared>>,
  %y_desc: !tt.tensordesc<tensor<64x64xf32, #shared>>,
  %n_tiles: i32
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %eps = arith.constant dense<1.000000e-06> : tensor<64xf32, #slice>
  scf.for %i = %c0_i32 to %n_tiles step %c1_i32 : i32 {
    %offs = arith.muli %i, %c64_i32 : i32
    %x = tt.descriptor_load %x_desc[%offs, %c0_i32] : !tt.tensordesc<tensor<64x64xf32, #shared>> -> tensor<64x64xf32, #blocked>
    %sq = arith.mulf %x, %x : tensor<64x64xf32, #blocked>
    %r = "tt.reduce"(%sq) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #slice>
    %denom = arith.addf %r, %eps : tensor<64xf32, #slice>
    %rinv = math.rsqrt %denom : tensor<64xf32, #slice>
    %rinv_e = tt.expand_dims %rinv {axis = 1 : i32} : tensor<64xf32, #slice> -> tensor<64x1xf32, #blocked>
    %rinv_b = tt.broadcast %rinv_e : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
    %norm = arith.mulf %x, %rinv_b : tensor<64x64xf32, #blocked>
    %stage = ttg.local_alloc %norm : (tensor<64x64xf32, #blocked>) -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %tok = ttng.async_tma_copy_local_to_global %y_desc[%offs, %c0_i32] %stage : !tt.tensordesc<tensor<64x64xf32, #shared>>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok : !ttg.async.token
    scf.yield
  } {tt.warp_specialize, tt.separate_epilogue_store = true, tt.merge_epilogue = true}
  tt.return
}

// Confirm any tt.reduce anchors the path: a max-reduction (tl.max) variant also
// gets the reduction-at-index-0 layout and verifies (all ops annotated).
//
// CHECK-LABEL: @maxnorm_no_mma
// CHECK: tt.reduce.return {{.*}}ttg.partition = array<i32: 0>
// CHECK: ttng.async_tma_copy_local_to_global {{.*}}ttg.partition = array<i32: 1>
// CHECK: ttng.async_tma_store_token_wait {{.*}}ttg.partition = array<i32: 1>
// CHECK: tt.warp_specialize
// CHECK-SAME: ttg.partition.types = ["reduction", "epilogue_store", "load"]
tt.func public @maxnorm_no_mma(
  %x_desc: !tt.tensordesc<tensor<64x64xf32, #shared>>,
  %y_desc: !tt.tensordesc<tensor<64x64xf32, #shared>>,
  %n_tiles: i32
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  scf.for %i = %c0_i32 to %n_tiles step %c1_i32 : i32 {
    %offs = arith.muli %i, %c64_i32 : i32
    %x = tt.descriptor_load %x_desc[%offs, %c0_i32] : !tt.tensordesc<tensor<64x64xf32, #shared>> -> tensor<64x64xf32, #blocked>
    %m = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %mx = arith.maxnumf %a, %b : f32
      tt.reduce.return %mx : f32
    }) : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #slice>
    %m_e = tt.expand_dims %m {axis = 1 : i32} : tensor<64xf32, #slice> -> tensor<64x1xf32, #blocked>
    %m_b = tt.broadcast %m_e : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
    %norm = arith.divf %x, %m_b : tensor<64x64xf32, #blocked>
    %stage = ttg.local_alloc %norm : (tensor<64x64xf32, #blocked>) -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %tok = ttng.async_tma_copy_local_to_global %y_desc[%offs, %c0_i32] %stage : !tt.tensordesc<tensor<64x64xf32, #shared>>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok : !ttg.async.token
    scf.yield
  } {tt.warp_specialize, tt.separate_epilogue_store = true, tt.merge_epilogue = true}
  tt.return
}

}
