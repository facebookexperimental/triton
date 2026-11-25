// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation | FileCheck %s
// RUN: env TRITON_WS_DISABLE_BRIDGE_FOLD=1 triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation | FileCheck %s --check-prefix=NOFOLD

// Each register-consumed descriptor load is converted to an nvws.descriptor_load
// that writes its own SMEM buffer, in producer program order. The A load's
// buffer/load must be emitted before the B load's; otherwise the producer order
// is reversed (e.g. every allocation hoisted to function entry in reverse).

// CHECK-LABEL: @hoist_preserves_producer_order
// CHECK: %[[A_BUF:.*]] = ttg.local_alloc {{.*}}: () -> !ttg.memdesc<128x64xf16
// CHECK: nvws.descriptor_load {{.*}} %[[A_BUF]]
// CHECK: %[[B_BUF:.*]] = ttg.local_alloc {{.*}}: () -> !ttg.memdesc<64x128xf16
// CHECK: nvws.descriptor_load {{.*}} %[[B_BUF]]

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hoist_preserves_producer_order(
      %a_desc: !tt.tensordesc<128x64xf16, #shared_a>,
      %b_desc: !tt.tensordesc<64x128xf16, #shared_b>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %a = tt.descriptor_load %a_desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared_a> -> tensor<128x64xf16, #blocked_a>
    %b = tt.descriptor_load %b_desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<64x128xf16, #shared_b> -> tensor<64x128xf16, #blocked_b>
    %a_use = arith.addf %a, %a {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked_a>
    %b_use = arith.addf %b, %b {async_task_id = array<i32: 0>} : tensor<64x128xf16, #blocked_b>
    tt.return
  }
}

// -----

// An exact-layout MMA allocation is redundant when a descriptor destination
// already holds the same tile. The descriptor buffer must directly serve both
// the register and MMA consumers before channel discovery.

// CHECK-LABEL: @fold_descriptor_mma_bridge
// CHECK: %[[A:.*]] = ttg.local_alloc {{.*}}: () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
// CHECK: nvws.descriptor_load {{.*}} %[[A]]
// CHECK: ttg.local_load %[[A]] {async_task_id = array<i32: 0>}
// CHECK-NOT: ttg.local_store
// CHECK: ttng.tc_gen5_mma %[[A]],
// CHECK: tt.return

// NOFOLD-LABEL: @fold_descriptor_mma_bridge
// NOFOLD: ttg.local_load
// NOFOLD: %[[MMA_A:.*]] = ttg.local_alloc
// NOFOLD: ttg.local_store {{.*}}, %[[MMA_A]]
// NOFOLD: ttng.tc_gen5_mma %[[MMA_A]],

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fold_descriptor_mma_bridge(
      %desc: !tt.tensordesc<64x128xbf16, #shared_a>,
      %b: !ttg.memdesc<128x64xbf16, #shared_b, #smem>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 1>} true
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 2>} : !tt.tensordesc<64x128xbf16, #shared_a> -> tensor<64x128xbf16, #blocked_a>
    %mma_a = ttg.local_alloc %tile {async_task_id = array<i32: 2>} : (tensor<64x128xbf16, #blocked_a>) -> !ttg.memdesc<64x128xbf16, #shared_a, #smem>
    %register_consumer = arith.extf %tile {async_task_id = array<i32: 0>} : tensor<64x128xbf16, #blocked_a> to tensor<64x128xf32, #blocked_a>
    %acc, %tok = ttng.tmem_alloc {async_task_id = array<i32: 1>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %mma_a, %b, %acc[%tok], %false, %true {async_task_id = array<i32: 1>, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared_a, #smem>, !ttg.memdesc<128x64xbf16, #shared_b, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// A physical cluster must retain the bridge until the direct-sharing protocol
// handles cross-CTA consumers.

// CHECK-LABEL: @keep_descriptor_mma_bridge_2cta
// CHECK: %[[A2:.*]] = ttg.local_alloc {{.*}}: () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
// CHECK: nvws.descriptor_load {{.*}} %[[A2]]
// CHECK: ttg.local_load %[[A2]]
// CHECK: %[[MMA_A:.*]] = ttg.local_alloc
// CHECK: ttg.local_store {{.*}}, %[[MMA_A]]
// CHECK: ttng.tc_gen5_mma %[[MMA_A]],
// CHECK: tt.return

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @keep_descriptor_mma_bridge_2cta(
      %desc: !tt.tensordesc<64x128xbf16, #shared_a>,
      %b: !ttg.memdesc<128x64xbf16, #shared_b, #smem>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 1>} true
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 2>} : !tt.tensordesc<64x128xbf16, #shared_a> -> tensor<64x128xbf16, #blocked_a>
    %mma_a = ttg.local_alloc %tile {async_task_id = array<i32: 2>} : (tensor<64x128xbf16, #blocked_a>) -> !ttg.memdesc<64x128xbf16, #shared_a, #smem>
    %register_consumer = arith.extf %tile {async_task_id = array<i32: 0>} : tensor<64x128xbf16, #blocked_a> to tensor<64x128xf32, #blocked_a>
    %acc, %tok = ttng.tmem_alloc {async_task_id = array<i32: 1>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %mma_a, %b, %acc[%tok], %false, %true {async_task_id = array<i32: 1>, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared_a, #smem>, !ttg.memdesc<128x64xbf16, #shared_b, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Data partitioning gives each slice its own descriptor destination, register
// load, bridge and MMA. Each slice's ring is folded independently, even when
// the two MMAs share one MMA partition and one B operand.

// CHECK-LABEL: @fold_descriptor_mma_bridge_data_partitioned
// CHECK: %[[A0:.*]] = ttg.local_alloc {{.*}}: () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
// CHECK: nvws.descriptor_load {{.*}} %[[A0]]
// CHECK: ttg.local_load %[[A0]] {async_task_id = array<i32: 0>}
// CHECK: %[[A1:.*]] = ttg.local_alloc {{.*}}: () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
// CHECK: nvws.descriptor_load {{.*}} %[[A1]]
// CHECK: ttg.local_load %[[A1]] {async_task_id = array<i32: 3>}
// CHECK-NOT: ttg.local_store
// CHECK: ttng.tc_gen5_mma %[[A0]],
// CHECK: ttng.tc_gen5_mma %[[A1]],
// CHECK: tt.return

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fold_descriptor_mma_bridge_data_partitioned(
      %desc: !tt.tensordesc<64x128xbf16, #shared_a>,
      %b: !ttg.memdesc<128x64xbf16, #shared_b, #smem>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c64 = arith.constant {async_task_id = array<i32: 2>} 64 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 1>} true
    %tile0 = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 2>} : !tt.tensordesc<64x128xbf16, #shared_a> -> tensor<64x128xbf16, #blocked_a>
    %tile1 = tt.descriptor_load %desc[%c64, %c0] {async_task_id = array<i32: 2>} : !tt.tensordesc<64x128xbf16, #shared_a> -> tensor<64x128xbf16, #blocked_a>
    %mma_a0 = ttg.local_alloc %tile0 {async_task_id = array<i32: 2>} : (tensor<64x128xbf16, #blocked_a>) -> !ttg.memdesc<64x128xbf16, #shared_a, #smem>
    %mma_a1 = ttg.local_alloc %tile1 {async_task_id = array<i32: 2>} : (tensor<64x128xbf16, #blocked_a>) -> !ttg.memdesc<64x128xbf16, #shared_a, #smem>
    %reg0 = arith.extf %tile0 {async_task_id = array<i32: 0>} : tensor<64x128xbf16, #blocked_a> to tensor<64x128xf32, #blocked_a>
    %reg1 = arith.extf %tile1 {async_task_id = array<i32: 3>} : tensor<64x128xbf16, #blocked_a> to tensor<64x128xf32, #blocked_a>
    %acc0, %tok0 = ttng.tmem_alloc {async_task_id = array<i32: 1>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc1, %tok1 = ttng.tmem_alloc {async_task_id = array<i32: 1>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mt0 = ttng.tc_gen5_mma %mma_a0, %b, %acc0[%tok0], %false, %true {async_task_id = array<i32: 1>, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared_a, #smem>, !ttg.memdesc<128x64xbf16, #shared_b, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %mt1 = ttng.tc_gen5_mma %mma_a1, %b, %acc1[%tok1], %false, %true {async_task_id = array<i32: 1>, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared_a, #smem>, !ttg.memdesc<128x64xbf16, #shared_b, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// A register consumer that uses the loaded value twice is still one consumer.

// CHECK-LABEL: @fold_descriptor_mma_bridge_reused_register_value
// CHECK: %[[R:.*]] = ttg.local_alloc {{.*}}: () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
// CHECK: ttg.local_load %[[R]] {async_task_id = array<i32: 0>}
// CHECK-NOT: ttg.local_store
// CHECK: ttng.tc_gen5_mma %[[R]],

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared_c = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fold_descriptor_mma_bridge_reused_register_value(
      %desc: !tt.tensordesc<64x128xbf16, #shared_a>,
      %b: !ttg.memdesc<128x64xbf16, #shared_b, #smem>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 1>} true
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 2>} : !tt.tensordesc<64x128xbf16, #shared_a> -> tensor<64x128xbf16, #blocked_a>
    %mma_a = ttg.local_alloc %tile {async_task_id = array<i32: 2>} : (tensor<64x128xbf16, #blocked_a>) -> !ttg.memdesc<64x128xbf16, #shared_a, #smem>
    %register_consumer = arith.mulf %tile, %tile {async_task_id = array<i32: 0>} : tensor<64x128xbf16, #blocked_a>
    %acc, %tok = ttng.tmem_alloc {async_task_id = array<i32: 1>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %mma_a, %b, %acc[%tok], %false, %true {async_task_id = array<i32: 1>, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared_a, #smem>, !ttg.memdesc<128x64xbf16, #shared_b, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// The MMA allocation uses a different swizzle than the descriptor destination.

// CHECK-LABEL: @keep_descriptor_mma_bridge_layout_mismatch
// CHECK: ttg.local_load
// CHECK: %[[MMA_K:.*]] = ttg.local_alloc
// CHECK: ttg.local_store {{.*}}, %[[MMA_K]]
// CHECK: ttng.tc_gen5_mma %[[MMA_K]],

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared_c = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @keep_descriptor_mma_bridge_layout_mismatch(
      %desc: !tt.tensordesc<64x128xbf16, #shared_a>,
      %b: !ttg.memdesc<128x64xbf16, #shared_b, #smem>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 1>} true
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 2>} : !tt.tensordesc<64x128xbf16, #shared_a> -> tensor<64x128xbf16, #blocked_a>
    %mma_a = ttg.local_alloc %tile {async_task_id = array<i32: 2>} : (tensor<64x128xbf16, #blocked_a>) -> !ttg.memdesc<64x128xbf16, #shared_c, #smem>
    %register_consumer = arith.extf %tile {async_task_id = array<i32: 0>} : tensor<64x128xbf16, #blocked_a> to tensor<64x128xf32, #blocked_a>
    %acc, %tok = ttng.tmem_alloc {async_task_id = array<i32: 1>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %mma_a, %b, %acc[%tok], %false, %true {async_task_id = array<i32: 1>, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared_c, #smem>, !ttg.memdesc<128x64xbf16, #shared_b, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// The register consumer and the MMA share a task, so the roles are not distinct.

// CHECK-LABEL: @keep_descriptor_mma_bridge_same_task
// CHECK: ttg.local_load
// CHECK: %[[MMA_K:.*]] = ttg.local_alloc
// CHECK: ttg.local_store {{.*}}, %[[MMA_K]]
// CHECK: ttng.tc_gen5_mma %[[MMA_K]],

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared_c = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @keep_descriptor_mma_bridge_same_task(
      %desc: !tt.tensordesc<64x128xbf16, #shared_a>,
      %b: !ttg.memdesc<128x64xbf16, #shared_b, #smem>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 1>} true
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 2>} : !tt.tensordesc<64x128xbf16, #shared_a> -> tensor<64x128xbf16, #blocked_a>
    %mma_a = ttg.local_alloc %tile {async_task_id = array<i32: 2>} : (tensor<64x128xbf16, #blocked_a>) -> !ttg.memdesc<64x128xbf16, #shared_a, #smem>
    %register_consumer = arith.extf %tile {async_task_id = array<i32: 1>} : tensor<64x128xbf16, #blocked_a> to tensor<64x128xf32, #blocked_a>
    %acc, %tok = ttng.tmem_alloc {async_task_id = array<i32: 1>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %mma_a, %b, %acc[%tok], %false, %true {async_task_id = array<i32: 1>, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared_a, #smem>, !ttg.memdesc<128x64xbf16, #shared_b, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// The bridge and the MMA consume different pipelined iterations.

// CHECK-LABEL: @keep_descriptor_mma_bridge_cadence_mismatch
// CHECK: ttg.local_load
// CHECK: %[[MMA_K:.*]] = ttg.local_alloc
// CHECK: ttg.local_store {{.*}}, %[[MMA_K]]
// CHECK: ttng.tc_gen5_mma %[[MMA_K]],

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared_c = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @keep_descriptor_mma_bridge_cadence_mismatch(
      %desc: !tt.tensordesc<64x128xbf16, #shared_a>,
      %b: !ttg.memdesc<128x64xbf16, #shared_b, #smem>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 1>} true
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 2>} : !tt.tensordesc<64x128xbf16, #shared_a> -> tensor<64x128xbf16, #blocked_a>
    %mma_a = ttg.local_alloc %tile {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<64x128xbf16, #blocked_a>) -> !ttg.memdesc<64x128xbf16, #shared_a, #smem>
    %register_consumer = arith.extf %tile {async_task_id = array<i32: 0>} : tensor<64x128xbf16, #blocked_a> to tensor<64x128xf32, #blocked_a>
    %acc, %tok = ttng.tmem_alloc {async_task_id = array<i32: 1>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %mma_a, %b, %acc[%tok], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared_a, #smem>, !ttg.memdesc<128x64xbf16, #shared_b, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
