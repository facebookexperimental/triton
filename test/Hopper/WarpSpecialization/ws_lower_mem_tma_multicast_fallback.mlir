// RUN: triton-opt %s --nvgpu-test-ws-code-partition=num-buffers=2 | FileCheck %s --implicit-check-not=tt.multicast_axes --implicit-check-not=ctaMask

// A planned axis with physical extent one is not an actual multicast. Drop
// the plan before allocating multicast rendezvous barriers and lower an
// ordinary per-CTA TMA copy.

// CHECK-LABEL: @ineffective_multicast_axis
// CHECK: ttng.barrier_expect
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.wait_barrier

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {
  "ttg.cluster-dim-x" = 2 : i32,
  "ttg.cluster-dim-y" = 1 : i32,
  "ttg.cluster-dim-z" = 1 : i32,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  ttg.target = "cuda:100"
} {
  tt.func public @ineffective_multicast_axis(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %out: !tt.ptr<f16>) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %ptrs = tt.splat %out {async_task_id = array<i32: 1>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %i0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : index
    %i1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : index
    %i4 = arith.constant {async_task_id = array<i32: 0, 1>} 4 : index
    %buffer = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %iv = %i0 to %i4 step %i1 {
      %tile = tt.descriptor_load %desc[%pid, %c0] {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32, tt.multicast_axes = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      ttg.local_store %tile, %buffer {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %local = ttg.local_load %buffer {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %consumer = arith.addf %local, %local {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked>
      tt.store %ptrs, %consumer {async_task_id = array<i32: 1>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      scf.yield
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize}
    tt.return
  }

  // Blackwell AutoWS currently falls back to per-CTA loads because issuing a
  // multicast TMA from a worker partition does not complete remote barriers.

  // CHECK-LABEL: @blackwell_multicast_falls_back
  // CHECK: ttng.barrier_expect
  // CHECK: ttng.async_tma_copy_global_to_local
  // CHECK: ttng.wait_barrier
  tt.func public @blackwell_multicast_falls_back(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %out: !tt.ptr<f16>) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %ptrs = tt.splat %out {async_task_id = array<i32: 1>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %i0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : index
    %i1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : index
    %i4 = arith.constant {async_task_id = array<i32: 0, 1>} 4 : index
    %buffer = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %iv = %i0 to %i4 step %i1 {
      %tile = tt.descriptor_load %desc[%pid, %c0] {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32, tt.multicast_axes = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      ttg.local_store %tile, %buffer {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %local = ttg.local_load %buffer {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %consumer = arith.addf %local, %local {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked>
      tt.store %ptrs, %consumer {async_task_id = array<i32: 1>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      scf.yield
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize}
    tt.return
  }
}
