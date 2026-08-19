// RUN: triton-opt %s --nvgpu-partition-scheduling-meta --nvgpu-test-taskid-propagate=num-warp-groups=2 --nvgpu-test-ws-atomic-broadcast | FileCheck %s --check-prefix=BCAST
// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-partition-scheduling-meta '--nvgpu-warp-specialization=capability=100 num-stages=3 smem-budget=232448 tile-prefetch-depth=2' | FileCheck %s --check-prefix=FULL

// The initial CLC seed remains the program id tuple. Only the next tuple from
// clc_read is broadcast for the following iteration.
// BCAST-LABEL: @while_clc_broadcast
// BCAST-COUNT-4: ttg.local_alloc : () -> !ttg.memdesc<1xi32
// BCAST: %[[X:.*]] = tt.get_program_id x
// BCAST: %[[Y:.*]] = tt.get_program_id y
// BCAST: %[[Z:.*]] = tt.get_program_id z
// BCAST: scf.while (%[[VALID_ARG:.*]] = %true, %[[X_ARG:.*]] = %[[X]], %[[Y_ARG:.*]] = %[[Y]], %[[Z_ARG:.*]] = %[[Z]])
// BCAST: scf.condition(%[[VALID_ARG]]) {{.*}}%[[VALID_ARG]], %[[X_ARG]], %[[Y_ARG]], %[[Z_ARG]]
// BCAST: ttng.clc_try_cancel_async {{.*}}async_task_id = array<i32: [[OWNER:[0-9]+]]>
// BCAST: ttng.clc_read {{.*}}async_task_id = array<i32: [[OWNER]]>
// BCAST: tt.splat {{.*}}tensor<1xi32
// BCAST-NEXT: ttg.local_store
// BCAST-NEXT: {{.*}}ttg.local_load
// BCAST-NEXT: {{.*}}tt.unsplat
// BCAST: tt.splat {{.*}}tensor<1xi32
// BCAST-NEXT: ttg.local_store
// BCAST-NEXT: {{.*}}ttg.local_load
// BCAST-NEXT: {{.*}}tt.unsplat
// BCAST: tt.splat {{.*}}tensor<1xi32
// BCAST-NEXT: ttg.local_store
// BCAST-NEXT: {{.*}}ttg.local_load
// BCAST-NEXT: {{.*}}tt.unsplat
// BCAST: tt.splat {{.*}}tensor<1xi32
// BCAST-NEXT: ttg.local_store
// BCAST-NEXT: {{.*}}ttg.local_load
// BCAST-NEXT: {{.*}}tt.unsplat
// BCAST: scf.yield {{.*}}async_task_id = array<i32: 0, 1>

// FULL-LABEL: @while_clc_broadcast
// FULL-COUNT-4: buffer.id
// FULL: ttg.warp_specialize
// FULL: default {
// FULL: scf.while
// FULL: ttg.local_load
// FULL: partition0
// FULL: scf.while
// FULL-COUNT-1: ttng.clc_try_cancel_async
// FULL-COUNT-1: ttng.clc_read
// FULL: ttg.local_store

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @while_clc_broadcast(
      %a_desc: !tt.tensordesc<128x64xf16, #shared>,
      %b_desc: !tt.tensordesc<64x256xf16, #shared>,
      %out: !tt.ptr<f32>, %k_tiles: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %acc_init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %initial_x = tt.get_program_id x : i32
    %initial_y = tt.get_program_id y : i32
    %initial_z = tt.get_program_id z : i32
    %results:4 = scf.while (%valid = %true, %x = %initial_x, %y = %initial_y, %z = %initial_z) : (i1, i32, i32, i32) -> (i1, i32, i32, i32) {
      scf.condition(%valid) %valid, %x, %y, %z : i1, i32, i32, i32
    } do {
    ^bb0(%valid: i1, %x: i32, %y: i32, %z: i32):
      %inner = scf.for %ki = %c0 to %k_tiles step %c1 iter_args(%acc = %acc_init) -> (tensor<128x256xf32, #mma>) : i32 {
        %a = tt.descriptor_load %a_desc[%x, %ki] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %a_alloc = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 0 : i32} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%ki, %y] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
        %b_alloc = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 0 : i32} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        %dot = ttng.warp_group_dot %a_alloc, %b_alloc, %acc {inputPrecision = 0 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        scf.yield %dot : tensor<128x256xf32, #mma>
      } {tt.scheduled_max_stage = 2 : i32}
      %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked1>
      %cvt = ttg.convert_layout %inner : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
      tt.store %ptrs, %cvt : tensor<128x256x!tt.ptr<f32>, #blocked1>
      %token = ttng.clc_try_cancel_async : !ttg.async.token
      %next_valid, %next_x, %next_y, %next_z = ttng.clc_read %token : !ttg.async.token -> i1, i32, i32, i32
      scf.yield %next_valid, %next_x, %next_y, %next_z : i1, i32, i32, i32
    } attributes {tt.warp_specialize}
    tt.return
  }
}
