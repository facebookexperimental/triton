// RUN: not triton-opt %s --nvws-memory-planner=num-buffers=1 -allow-unregistered-dialect 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @partial_view_store_is_not_full_overwrite(%arg0: tensor<128x128xf32, #blocked>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %slice = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: error: NVWS memory planner does not support partial-view TMEM producer modeling
    ttng.tmem_store %arg0, %slice, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
