// RUN: triton-opt %s --nvws-memory-planner=num-buffers=0 -allow-unregistered-dialect | FileCheck %s --check-prefix=NOOP
// RUN: not triton-opt %s --nvws-memory-planner=num-buffers=1 -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=ERROR

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // NOOP-LABEL: @sourceful_alloc_requires_single_writer_partition
  // NOOP: ttng.tmem_alloc %arg0 {ttg.partition = array<i32: 4, 5>}
  // NOOP-NOT: buffer.id
  tt.func public @sourceful_alloc_requires_single_writer_partition(%arg0: tensor<128x128xf32, #blocked>) {
    // ERROR: error: NVWS memory planner expected sourceful ttng.tmem_alloc to have exactly one partition
    %alloc, %token = ttng.tmem_alloc %arg0 {ttg.partition = array<i32: 4, 5>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %out, %out_token = ttng.tmem_load %alloc[%token] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    "use"(%out) : (tensor<128x128xf32, #linear>) -> ()
    tt.return
  }
}
