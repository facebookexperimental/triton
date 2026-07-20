// RUN: triton-opt %s --nvgpu-test-ws-code-partition=num-buffers=1 | FileCheck %s

// A post-allocation TMA channel can feed a reduction epilogue whose final
// descriptor_store remains in the same consumer task.

// CHECK-LABEL: @reduce_epilogue_descriptor_store
// CHECK: ttg.warp_specialize
// CHECK: default
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: partition0
// CHECK: ttng.wait_barrier
// CHECK: [[TILE:%.*]] = ttg.local_load
// CHECK: [[SUM:%.*]] = "tt.reduce"([[TILE]])
// CHECK: tt.descriptor_store

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_epilogue_descriptor_store(%input: !tt.tensordesc<tensor<128x256xf16, #shared>>, %output: !tt.tensordesc<tensor<128x256xf16, #shared>>) attributes {noinline = false} {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %buffer = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
    %loaded = tt.descriptor_load %input[%c0, %c0] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x256xf16, #shared>> -> tensor<128x256xf16, #blocked>
    ttg.local_store %loaded, %buffer {async_task_id = array<i32: 0>} : tensor<128x256xf16, #blocked> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
    %tile = ttg.local_load %buffer {async_task_id = array<i32: 1>} : !ttg.memdesc<128x256xf16, #shared, #smem, mutable> -> tensor<128x256xf16, #blocked>
    %sum = "tt.reduce"(%tile) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f16, %rhs: f16):
      %next = arith.addf %lhs, %rhs {async_task_id = array<i32: 1>} : f16
      tt.reduce.return %next {async_task_id = array<i32: 1>} : f16
    }) {async_task_id = array<i32: 1>} : (tensor<128x256xf16, #blocked>) -> tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked}>>
    %expanded = tt.expand_dims %sum {async_task_id = array<i32: 1>, axis = 1 : i32} : tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf16, #blocked>
    %broadcast = tt.broadcast %expanded {async_task_id = array<i32: 1>} : tensor<128x1xf16, #blocked> -> tensor<128x256xf16, #blocked>
    %result = arith.addf %tile, %broadcast {async_task_id = array<i32: 1>} : tensor<128x256xf16, #blocked>
    tt.descriptor_store %output[%c0, %c0], %result {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x256xf16, #shared>>, tensor<128x256xf16, #blocked>
    tt.return
  }
}
