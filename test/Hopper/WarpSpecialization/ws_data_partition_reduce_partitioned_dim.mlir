// RUN: triton-opt %s --nvgpu-ws-data-partition=num-warp-groups=3 -verify-diagnostics | FileCheck %s

// A column sum of an M-partitioned accumulator combines the per-partition
// partials with the reduction's combiner.

// CHECK-LABEL: @col_reduce_combines_partials
// CHECK: [[P0:%.*]] = "tt.reduce"({{.*}}) <{axis = 0 : i32{{.*}}}>
// CHECK: (tensor<64x256xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
// CHECK: [[P1:%.*]] = "tt.reduce"({{.*}}) <{axis = 0 : i32{{.*}}}>
// CHECK: (tensor<64x256xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
// CHECK: [[SUM:%.*]] = arith.addf [[P0]], [[P1]] {{.*}}: tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
// CHECK: tt.store {{.*}}, [[SUM]] {{.*}}:
// CHECK-NOT: tensor<128x256xf32

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @col_reduce_combines_partials(%desc_a: !tt.tensordesc<128x64xf16>, %desc_b: !tt.tensordesc<64x256xf16>, %out: !tt.ptr<f32>) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %a = tt.descriptor_load %desc_a[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
    %a_smem = ttg.local_alloc %a {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b = tt.descriptor_load %desc_b[%c0_i32, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
    %b_smem = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %dot = ttng.warp_group_dot %a_smem, %b_smem, %acc {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
    %sum = "tt.reduce"(%dot) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) {async_task_id = array<i32: 1, 2>} : (tensor<128x256xf32, #mma>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #mma}>>
    %ptr = tt.splat %out {async_task_id = array<i32: 1, 2>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>
    tt.store %ptr, %sum {async_task_id = array<i32: 1, 2>} : tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #mma}>>
    tt.return
  }
}

