// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-partition-scheduling-meta --nvgpu-warp-specialization | FileCheck %s

// Regression guard for the no-MMA reduction materialization crash: the
// `attr.size() == 1 && "expected exactly 1 partition element"` assertion in
// doTaskIdPropagate (WSTaskIdPropagate.cpp).
//
// PartitionSchedulingMeta annotates a no-MMA in-register reduction (RMS norm /
// LayerNorm / softmax-only). Its scalar-offset fixup gives an offset op that
// feeds BOTH the load and the store a multi-element partition array (union of
// the two partitions, e.g. `ttg.partition = array<i32: 1, 2>`). doTaskIdPropagate
// used to assume every op carried exactly one partition id and aborted on that
// union. It now materializes every id in the array, so the offset is replicated
// per-partition by code partitioning. The annotation side is covered by
// partition-scheduling-meta-no-mma-reduction.mlir; this drives the FULL
// warp-specialization pipeline so it reaches doTaskIdPropagate. Without the fix
// triton-opt aborts and FileCheck sees empty input, failing the test.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {

// CHECK-LABEL: @rmsnorm_no_mma
// The pass must lower without crashing into the reduction/epilogue_store/load
// 3-warp-group layout.
// CHECK: ttg.warp_specialize
// CHECK-SAME: ttg.partition.types = ["reduction", "epilogue_store", "load"]
// Reduction chain stays in the default (reduction) partition, task 0.
// CHECK: tt.reduce.return {{.*}}async_task_id = array<i32: 0>
// The scalar offset that fed both the load and the store is replicated: once in
// the epilogue-store partition (task 1) feeding the TMA store, and once in the
// load partition (task 2) feeding the TMA load. This is the multi-partition
// materialization the fix enables.
// CHECK: arith.muli %{{.*}}, %c64_i32 {async_task_id = array<i32: 1>
// CHECK: async_tma_copy_local_to_global
// CHECK: arith.muli %{{.*}}, %c64_i32 {async_task_id = array<i32: 2>
// CHECK: async_tma_copy_global_to_local
tt.func public @rmsnorm_no_mma(
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

// A max-reduction (tl.max) variant lowers through the same materialization path.
//
// CHECK-LABEL: @maxnorm_no_mma
// CHECK: ttg.warp_specialize
// CHECK-SAME: ttg.partition.types = ["reduction", "epilogue_store", "load"]
// CHECK: arith.muli %{{.*}}, %c64_i32 {async_task_id = array<i32: 1>
// CHECK: async_tma_copy_local_to_global
// CHECK: arith.muli %{{.*}}, %c64_i32 {async_task_id = array<i32: 2>
// CHECK: async_tma_copy_global_to_local
tt.func public @maxnorm_no_mma(
  %x_desc: !tt.tensordesc<64x64xf32, #shared>,
  %y_desc: !tt.tensordesc<64x64xf32, #shared>,
  %n_tiles: i32
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  scf.for %i = %c0_i32 to %n_tiles step %c1_i32 : i32 {
    %offs = arith.muli %i, %c64_i32 : i32
    %x = tt.descriptor_load %x_desc[%offs, %c0_i32] : !tt.tensordesc<64x64xf32, #shared> -> tensor<64x64xf32, #blocked>
    %m = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %mx = arith.maxnumf %a, %b : f32
      tt.reduce.return %mx : f32
    }) : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #slice>
    %m_e = tt.expand_dims %m {axis = 1 : i32} : tensor<64xf32, #slice> -> tensor<64x1xf32, #blocked>
    %m_b = tt.broadcast %m_e : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
    %norm = arith.divf %x, %m_b : tensor<64x64xf32, #blocked>
    %stage = ttg.local_alloc %norm : (tensor<64x64xf32, #blocked>) -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %tok = ttng.async_tma_copy_local_to_global %y_desc[%offs, %c0_i32] %stage : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %tok : !ttg.async.token
    scf.yield
  } {tt.warp_specialize, tt.separate_epilogue_store = true, tt.merge_epilogue = true, tt.smem_budget = 200000 : i32}
  tt.return
}

}
