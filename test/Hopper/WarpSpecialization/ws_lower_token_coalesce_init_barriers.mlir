// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-partition-scheduling-meta --nvgpu-warp-specialization | FileCheck %s

// Regression guard for token-initialization barrier coalescing in
// WSLowerToken.cpp. Lowering emits one CTA barrier after initializing each
// token's full/empty mbarriers. When consecutive initializers land in the same
// block with nothing but initialization scaffolding between them (constants,
// local_alloc, memdesc_index, init_barrier and the generated barriers), the
// earlier barriers are redundant: no token user can run before the last one.
//
// This kernel warp-specializes into three partitions and creates more than one
// token, so lowering emits more than one init barrier and they collapse to a
// single barrier. Without coalesceAdjacentTokenInitBarriers every initializer
// keeps its own barrier and the CHECK-NOT below fires.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {

// CHECK-LABEL: @coalesce_token_init_barriers
// Exactly one CTA barrier survives token initialization, and it still comes
// after every init_barrier so it continues to order all token users.
// CHECK: ttng.init_barrier
// CHECK: ttg.barrier
// CHECK-NOT: ttg.barrier
// CHECK: ttg.warp_specialize
tt.func public @coalesce_token_init_barriers(
  %x_desc: !tt.tensordesc<64x64xf32, #shared>,
  %y_desc: !tt.tensordesc<64x64xf32, #shared>,
  %n_tiles: i32
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %eps = arith.constant dense<1.000000e-06> : tensor<64xf32, #slice>
  scf.for %i = %c0_i32 to %n_tiles step %c1_i32 : i32 {
    %offs = arith.muli %i, %c64_i32 : i32
    %x = tt.descriptor_load %x_desc[%offs, %c0_i32] : !tt.tensordesc<64x64xf32, #shared> -> tensor<64x64xf32, #blocked>
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
    %tok = ttng.async_tma_copy_local_to_global %y_desc[%offs, %c0_i32] %stage : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok : !ttg.async.token
    scf.yield
  } {tt.warp_specialize, tt.separate_epilogue_store = true, tt.merge_epilogue = true, tt.smem_budget = 200000 : i32}
  tt.return
}

}
