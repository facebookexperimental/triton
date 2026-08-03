// RUN: triton-opt %s '--nvgpu-test-ws-code-partition=num-buffers=1' | FileCheck %s

// A TMA buffer shared by gen5 and scalar tasks needs both synchronization paths.

// CHECK-LABEL: @tma_mixed_consumers
// CHECK: ttng.wait_barrier {{.*}}async_task_id = array<i32: 0>{{.*}}direction = "backward"{{.*}}dstTask = 1
// CHECK: nvws.producer_acquire {{.*}}async_task_id = array<i32: 0>{{.*}}dstTask = 2
// CHECK-COUNT-1: ttng.async_tma_copy_global_to_local
// CHECK-DAG: ttng.tc_gen5_mma {{.*}}async_task_id = array<i32: 1>, is_async
// CHECK-DAG: ttng.wait_barrier {{.*}}async_task_id = array<i32: 2>
// CHECK-DAG: ttg.local_load {{.*}}async_task_id = array<i32: 2>
// CHECK-DAG: nvws.consumer_release {{.*}}async_task_id = array<i32: 2>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_mixed_consumers(%desc: !tt.tensordesc<128x64xf16, #shared>, %b: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %acc_tok: !ttg.async.token, %out: !tt.ptr<f16>, %lb: i32, %ub: i32, %step: i32) attributes {noinline = false} {
    %a = ttg.local_alloc {async_task_id = array<i32: 0, 1, 2>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 1>} true
    %ptrs = tt.splat %out {async_task_id = array<i32: 2>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    scf.for %iv = %lb to %ub step %step iter_args(%tok = %acc_tok) -> (!ttg.async.token) : i32 {
      %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      ttg.local_store %tile, %a {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %mma_tok = ttng.tc_gen5_mma %a, %b, %acc[%tok], %false, %true {async_task_id = array<i32: 1>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %loaded = ttg.local_load %a {async_task_id = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      tt.store %ptrs, %loaded {async_task_id = array<i32: 2>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      scf.yield {async_task_id = array<i32: 1>} %mma_tok : !ttg.async.token
    } {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }
}
