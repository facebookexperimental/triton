// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation '--nvgpu-test-ws-memory-planner=num-buffers=2 smem-budget=200000' '--nvgpu-test-ws-code-partition=num-buffers=2' | FileCheck %s
// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation '--nvgpu-test-ws-memory-planner=num-buffers=2 smem-budget=200000' '--nvgpu-test-ws-code-partition=num-buffers=2' | FileCheck %s --check-prefix=TASK0
// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation '--nvgpu-test-ws-memory-planner=num-buffers=2 smem-budget=200000' '--nvgpu-test-ws-code-partition=num-buffers=2' | FileCheck %s --check-prefix=TASK1
// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation '--nvgpu-test-ws-memory-planner=num-buffers=2 smem-budget=200000' '--nvgpu-test-ws-code-partition=num-buffers=2' | FileCheck %s --check-prefix=TASK2
// RUN: env TRITON_WS_DISABLE_BRIDGE_FOLD=1 triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation '--nvgpu-test-ws-memory-planner=num-buffers=2 smem-budget=200000' '--nvgpu-test-ws-code-partition=num-buffers=2' | FileCheck %s --check-prefix=NOFOLD

// Buffer creation should reuse the descriptor load's existing SMEM allocation.

// CHECK-LABEL: @post_tma_load_multi_consumer_allocs
// CHECK-NOT: tt.descriptor_load
// CHECK-COUNT-1: ttng.async_tma_copy_global_to_local
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.return

// TASK1-LABEL: @post_tma_load_multi_consumer_allocs
// TASK1: ttng.wait_barrier {{.*}}async_task_id = array<i32: 1>
// TASK1: ttg.local_load {{.*}}async_task_id = array<i32: 1>

// TASK2-LABEL: @post_tma_load_multi_consumer_allocs
// TASK2: ttng.wait_barrier {{.*}}async_task_id = array<i32: 2>
// TASK2: ttg.local_load {{.*}}async_task_id = array<i32: 2>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @post_tma_load_multi_consumer_allocs(%desc: !tt.tensordesc<128x64xf16, #shared>, %out0: !tt.ptr<f16>, %out1: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %ptrs0 = tt.splat %out0 {async_task_id = array<i32: 1>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptrs1 = tt.splat %out1 {async_task_id = array<i32: 2>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    %alloc0 = ttg.local_alloc %tile {async_task_id = array<i32: 0, 1>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %loaded0 = ttg.local_load %alloc0 {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
    %consumer0 = arith.addf %loaded0, %loaded0 {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked>
    %consumer1 = arith.subf %tile, %tile {async_task_id = array<i32: 2>} : tensor<128x64xf16, #blocked>
    tt.store %ptrs0, %consumer0 {async_task_id = array<i32: 1>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.store %ptrs1, %consumer1 {async_task_id = array<i32: 2>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// The folded descriptor destination has a register consumer and an MMAv5
// consumer. Code partitioning must retain both completion paths, and the
// register consumer's release must keep the proxy-fence request because the
// producer refills the slot with TMA.

// CHECK-LABEL: @post_tma_load_mma_bridge
// CHECK: ttg.local_alloc {buffer.copy = 2 : i32
// CHECK-NOT: ttg.local_store
// CHECK: ttng.tc_gen5_mma {{.*}}is_async
// CHECK-NOT: ttg.local_store
// CHECK-COUNT-1: ttng.async_tma_copy_global_to_local
// CHECK-NOT: ttg.local_store
// CHECK: tt.return

// NOFOLD-LABEL: @post_tma_load_mma_bridge
// NOFOLD: ttg.local_store
// NOFOLD: tt.return

// TASK0-LABEL: @post_tma_load_mma_bridge
// TASK0: ttng.wait_barrier {{.*}}async_task_id = array<i32: 0>{{.*}}direction = "forward", dstTask = 2 : i32
// TASK0: ttg.local_load {{.*}}async_task_id = array<i32: 0>
// TASK0: nvws.consumer_release {{.*}}async_task_id = array<i32: 0>{{.*}}dstTask = 2 : i32{{.*}}ttng.fence_generic_reads

// TASK1-LABEL: @post_tma_load_mma_bridge
// TASK1: ttng.wait_barrier {{.*}}async_task_id = array<i32: 1>{{.*}}direction = "forward", dstTask = 2 : i32
// TASK1: ttng.tc_gen5_mma {{.*}}is_async

// TASK2-LABEL: @post_tma_load_mma_bridge
// TASK2: ttng.wait_barrier {{.*}}async_task_id = array<i32: 2>{{.*}}direction = "backward", dstTask = 1 : i32
// TASK2: nvws.producer_acquire {{.*}}async_task_id = array<i32: 2>{{.*}}dstTask = 0 : i32
// TASK2-COUNT-1: ttng.async_tma_copy_global_to_local

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>
#blocked_out = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @post_tma_load_mma_bridge(
      %desc: !tt.tensordesc<64x128xbf16, #shared>,
      %b: !ttg.memdesc<128x64xbf16, #shared_t, #smem>,
      %out: !tt.ptr<bf16>, %rrms_out: !tt.ptr<f32>) attributes {noinline = false} {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c4 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 4 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 1>} true
    %ptrs = tt.splat %out {async_task_id = array<i32: 0>} : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #blocked_out>
    %rrms_ptrs = tt.splat %rrms_out {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked>
    %acc, %tok = ttng.tmem_alloc {async_task_id = array<i32: 0, 1>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc_tok = %tok) -> (!ttg.async.token) : i32 {
      %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x128xbf16, #shared> -> tensor<64x128xbf16, #blocked>
      %mma_a = ttg.local_alloc %tile {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
      %register_consumer = arith.extf %tile {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x128xbf16, #blocked> to tensor<64x128xf32, #blocked>
      %rrms_work = arith.mulf %register_consumer, %register_consumer {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x128xf32, #blocked>
      tt.store %rrms_ptrs, %rrms_work {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x128x!tt.ptr<f32>, #blocked>
      %mma_tok = ttng.tc_gen5_mma %mma_a, %b, %acc[%acc_tok], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x64xbf16, #shared_t, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %loaded, %load_tok = ttng.tmem_load %acc[%mma_tok] {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear>
      %truncated = arith.truncf %loaded {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x64xf32, #linear> to tensor<64x64xbf16, #linear>
      %output = ttg.convert_layout %truncated {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #linear> -> tensor<64x64xbf16, #blocked_out>
      tt.store %ptrs, %output {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x64x!tt.ptr<bf16>, #blocked_out>
      scf.yield {async_task_id = array<i32: 0>} %load_tok : !ttg.async.token
    } {async_task_id = array<i32: 0, 1, 2>, tt.data_partition_factor = 1 : i32, tt.scheduled_max_stage = 2 : i32}
    tt.return
  }
}

// -----

// An allocation in an ancestor block can serve a direct consumer in a nested
// region when it dominates that consumer.

// CHECK-LABEL: @post_tma_load_nested_consumer
// CHECK-NOT: tt.descriptor_load
// CHECK-COUNT-1: ttng.async_tma_copy_global_to_local
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @post_tma_load_nested_consumer(%desc: !tt.tensordesc<128x64xf16, #shared>, %out0: !tt.ptr<f16>, %out1: !tt.ptr<f16>, %lb: i32, %ub: i32, %step: i32) attributes {noinline = false} {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %ptrs0 = tt.splat %out0 {async_task_id = array<i32: 1>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptrs1 = tt.splat %out1 {async_task_id = array<i32: 2>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    %alloc = ttg.local_alloc %tile {async_task_id = array<i32: 0, 1>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %loaded = ttg.local_load %alloc {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked>
    tt.store %ptrs0, %loaded {async_task_id = array<i32: 1>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    scf.for %iv = %lb to %ub step %step : i32 {
      %consumer = arith.addf %tile, %tile {async_task_id = array<i32: 2>} : tensor<128x64xf16, #blocked>
      tt.store %ptrs1, %consumer {async_task_id = array<i32: 2>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    } {async_task_id = array<i32: 0, 2>}
    tt.return
  }
}
