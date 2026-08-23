// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448" | FileCheck %s --implicit-check-not=async_task_id --implicit-check-not=ttg.partition --implicit-check-not=ttg.warp_specialize --implicit-check-not=atomic_bcast_slot --implicit-check-not=tt.unsplat

// An eligible loop-carried atomic is visited before a CLC read assigned to only
// a strict subset of the while's partitions. Rejecting the CLC path must not
// leave the earlier atomic broadcast transform in the function.
// CHECK-LABEL: @mixed_atomic_and_clustered_clc
// CHECK-COUNT-1: tt.atomic_rmw
// CHECK: ttng.clc_try_cancel_async
// CHECK: ttng.clc_read
// CHECK: scf.yield

// A clustered CLC function with an analyzed multicast load must reject AutoWS
// before the CLC fetch is rewritten. The standard clustered multicast protocol
// remains intact in the unspecialized function.
// CHECK-LABEL: @clustered_clc_multicast
// CHECK: tt.descriptor_load
// CHECK-SAME: tt.multicast_axes = array<i32: 0>
// CHECK: ttng.clc_try_cancel_async
// CHECK: ttng.clc_read

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {
  "ttg.cluster-dim-x" = 2 : i32,
  "ttg.cluster-dim-y" = 2 : i32,
  "ttg.cluster-dim-z" = 1 : i32,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  ttg.target = "cuda:100"
} {
  tt.func public @mixed_atomic_and_clustered_clc(%counter: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %results:5 = scf.while (%tile = %c0, %valid = %true, %x = %c0, %y = %c0, %z = %c0) : (i32, i1, i32, i32, i32) -> (i32, i1, i32, i32, i32) {
      scf.condition(%valid) %tile, %valid, %x, %y, %z : i32, i1, i32, i32, i32
    } do {
    ^bb0(%tile: i32, %valid: i1, %x: i32, %y: i32, %z: i32):
      %next_tile = tt.atomic_rmw add, acq_rel, gpu, %counter, %c1, %true {async_task_id = array<i32: 0, 1>} : (!tt.ptr<i32>, i32, i1) -> i32
      %token = ttng.clc_try_cancel_async {async_task_id = array<i32: 0, 1>} : !ttg.async.token
      %next_valid, %next_x, %next_y, %next_z = ttng.clc_read %token {async_task_id = array<i32: 0, 1>} : !ttg.async.token -> i1, i32, i32, i32
      scf.yield %next_tile, %next_valid, %next_x, %next_y, %next_z : i32, i1, i32, i32, i32
    } attributes {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }

  tt.func public @clustered_clc_multicast(
      %desc: !tt.tensordesc<128x64xf16, #shared>, %out: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 0>, tt.multicast_axes = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    %ptrs = tt.splat %out : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.store %ptrs, %tile {async_task_id = array<i32: 1>} : tensor<128x64x!tt.ptr<f16>, #blocked>
    %results:4 = scf.while (%valid = %true, %x = %c0, %y = %c0, %z = %c0) : (i1, i32, i32, i32) -> (i1, i32, i32, i32) {
      scf.condition(%valid) %valid, %x, %y, %z : i1, i32, i32, i32
    } do {
    ^bb0(%valid: i1, %x: i32, %y: i32, %z: i32):
      %token = ttng.clc_try_cancel_async {async_task_id = array<i32: 0>} : !ttg.async.token
      %next_valid, %next_x, %next_y, %next_z = ttng.clc_read %token {async_task_id = array<i32: 0>} : !ttg.async.token -> i1, i32, i32, i32
      scf.yield %next_valid, %next_x, %next_y, %next_z : i1, i32, i32, i32
    } attributes {async_task_id = array<i32: 0, 1>}
    tt.return
  }
}
