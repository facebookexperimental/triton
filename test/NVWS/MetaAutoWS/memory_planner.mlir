// RUN: triton-opt %s --nvws-memory-planner="num-buffers=7" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_memory_planner(%arg0: tensor<128x128xf32, #blocked>) {
    // CHECK: ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.offset = 0 : i32}
    %alloc0, %token0 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK: ttng.tmem_alloc %arg0 {buffer.copy = 1 : i32, buffer.id = 1 : i32, buffer.offset = 0 : i32}
    %alloc1, %token1 = ttng.tmem_alloc %arg0 : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    tt.return
  }
}
