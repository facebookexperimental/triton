// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=4 smem-alloc-algo=1 smem-budget=400000" | FileCheck %s

// A four-subtile ordinary 1CTA output ring with a three-stage correctness
// floor must be repaired upward to four copies.
// CHECK-LABEL: @repair_output_ring_floor
// CHECK-COUNT-4: ttg.local_alloc {buffer.copy = 4 : i32, buffer.id = [[ID:[0-9]+]] : i32

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @repair_output_ring_floor(
      %c_desc: !tt.tensordesc<128x128xf16, #shared>) {
    %C0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %C1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %C2 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %C3 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %c128 = arith.constant {async_task_id = array<i32: 0, 1>} 128 : i32
    %c256 = arith.constant {async_task_id = array<i32: 0, 1>} 256 : i32
    %c384 = arith.constant {async_task_id = array<i32: 0, 1>} 384 : i32
    %value = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    ttg.local_store %value, %C0 {async_task_id = array<i32: 0>, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.local_store %value, %C1 {async_task_id = array<i32: 0>, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.local_store %value, %C2 {async_task_id = array<i32: 0>, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.local_store %value, %C3 {async_task_id = array<i32: 0>, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %t00 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c0] %C0 {async_task_id = array<i32: 1>, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t02 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c0] %C0 {async_task_id = array<i32: 1>, loop.stage = 2 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t10 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c128] %C1 {async_task_id = array<i32: 1>, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t12 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c128] %C1 {async_task_id = array<i32: 1>, loop.stage = 2 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t20 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c256] %C2 {async_task_id = array<i32: 1>, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t22 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c256] %C2 {async_task_id = array<i32: 1>, loop.stage = 2 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t30 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c384] %C3 {async_task_id = array<i32: 1>, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t32 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c384] %C3 {async_task_id = array<i32: 1>, loop.stage = 2 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    tt.return
  }
}

// -----

// Ordinary 2CTA output staging is always one copy. Unlike 1CTA, the paired
// CTAs do not have a cluster-aware completion phase for a deeper ring.
// CHECK-LABEL: @keep_2cta_output_ring_at_one
// CHECK-COUNT-4: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[ID2:[0-9]+]] : i32

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @keep_2cta_output_ring_at_one(
      %c_desc: !tt.tensordesc<128x128xf16, #shared>) {
    %C0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %C1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %C2 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %C3 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %c128 = arith.constant {async_task_id = array<i32: 0, 1>} 128 : i32
    %c256 = arith.constant {async_task_id = array<i32: 0, 1>} 256 : i32
    %c384 = arith.constant {async_task_id = array<i32: 0, 1>} 384 : i32
    %value = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    ttg.local_store %value, %C0 {async_task_id = array<i32: 0>, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.local_store %value, %C1 {async_task_id = array<i32: 0>, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.local_store %value, %C2 {async_task_id = array<i32: 0>, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.local_store %value, %C3 {async_task_id = array<i32: 0>, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %t0 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c0] %C0 {async_task_id = array<i32: 1>, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t1 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c128] %C1 {async_task_id = array<i32: 1>, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t2 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c256] %C2 {async_task_id = array<i32: 1>, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    %t3 = ttng.async_tma_copy_local_to_global %c_desc[%c0, %c384] %C3 {async_task_id = array<i32: 1>, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    tt.return
  }
}
