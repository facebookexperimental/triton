// RUN: triton-opt %s --triton-nvidia-interleave-tmem --allow-unregistered-dialect | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// A shared_cluster_memory view must be treated as possibly aliasing the staging buffer.
// CHECK-LABEL: @plain_token_wait_before_cluster_arrive
// CHECK: ttng.async_tma_copy_local_to_global
// CHECK: ttng.async_tma_store_token_wait
// CHECK: ttng.arrive_barrier
// CHECK: ttg.local_store
tt.func @plain_token_wait_before_cluster_arrive(
    %desc: !tt.tensordesc<128x64xf16, #shared>,
    %buf: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
    %bar: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
    %v: tensor<128x64xf16, #blocked>, %peer: i32) {
  %c0 = arith.constant 0 : i32
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
  %acc = ttng.tmem_load %alloc : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear>
  %tok = ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %buf : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
  ttng.async_tma_store_token_wait %tok : !ttg.async.token
  %remote = ttng.map_to_remote_buffer %bar, %peer : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier_shared, #ttng.shared_cluster_memory, mutable>
  ttng.arrive_barrier %remote, 1 : !ttg.memdesc<1xi64, #barrier_shared, #ttng.shared_cluster_memory, mutable>
  ttg.local_store %v, %buf : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  "user"(%acc) : (tensor<128x64xf32, #linear>) -> ()
  tt.return
}
}
