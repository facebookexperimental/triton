// RUN: not triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=4 smem-alloc-algo=1 smem-budget=400000" 2>&1 | FileCheck %s

// A two-subtile ordinary 1CTA output ring cannot satisfy a three-stage
// correctness floor: neither three nor four divides the subtile count.
// CHECK: error: illegal ordinary TMA output-ring depth 3 for 2 subtiles

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_illegal_output_ring_floor(
      %c_desc: !tt.tensordesc<128x128xf16, #shared>) {
    %C0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %C1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %c128 = arith.constant {async_task_id = array<i32: 0, 1>} 128 : i32
    %value = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    ttg.local_store %value, %C0 {async_task_id = array<i32: 0>, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.local_store %value, %C1 {async_task_id = array<i32: 0>, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %t00 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c0] %C0 {async_task_id = array<i32: 1>, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t02 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c0] %C0 {async_task_id = array<i32: 1>, loop.stage = 2 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t10 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c128] %C1 {async_task_id = array<i32: 1>, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t12 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c128] %C1 {async_task_id = array<i32: 1>, loop.stage = 2 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    tt.return
  }
}
