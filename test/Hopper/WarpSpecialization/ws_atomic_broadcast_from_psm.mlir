// RUN: triton-opt %s -split-input-file --nvgpu-partition-scheduling-meta --nvgpu-test-taskid-propagate=num-warp-groups=2 --nvgpu-test-ws-atomic-broadcast | FileCheck %s --check-prefix=BCAST
// RUN: triton-opt %s -split-input-file --nvgpu-partition-scheduling-meta --nvgpu-test-taskid-propagate=num-warp-groups=2 --nvgpu-test-ws-atomic-broadcast=tile-prefetch-depth=2 | FileCheck %s --check-prefix=DEPTH2

// Dynamic-persistent GEMM: an outer scf.while claims each tile with a scalar
// atomic, starting UNPARTITIONED. This exercises the full front half of the
// outer-while AutoWS path end to end -- PartitionSchedulingMeta assigns the
// partitions directly on the scf.while (no inner-loop annotation workaround),
// task-id propagation materializes async_task_ids, and the atomic-broadcast
// transform turns the loop-carried scalar tile claim into a run-once-in-owner +
// broadcast-to-all-partitions operation. Regression guard for the metadata
// contract between PSM/task-id propagation and doDynamicTileBroadcast: the
// atomic must reach the transform carrying the full partition union so it is
// classified as a broadcast (not passed through per-partition, which would
// deadlock).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // BCAST-LABEL: @while_atomic_broadcast
  // A single function-scope SMEM slot carries the broadcast tile id.
  // BCAST: ttg.local_alloc : () -> !ttg.memdesc<1xi32
  // The nested TMA loads pin the owner partition.
  // BCAST: tt.descriptor_load {{.*}}async_task_id = array<i32: [[LOAD:[0-9]+]]>
  // The run-once atomic executes in exactly one (the TMA-load) partition...
  // BCAST: tt.atomic_rmw {{.*}}async_task_id = array<i32: [[LOAD]]>
  // BCAST-NOT: tt.atomic_rmw
  // ...splats + stores its result into the slot in that same owner partition...
  // BCAST: tt.splat {{.*}}async_task_id = array<i32: [[LOAD]]>
  // BCAST: ttg.local_store {{.*}}async_task_id = array<i32: [[LOAD]]>
  // ...and every partition loads + unsplats the broadcast tile id.
  // BCAST: ttg.local_load {{.*}}async_task_id = array<i32: 0, 1>
  // BCAST: tt.unsplat {{.*}}async_task_id = array<i32: 0, 1>
  // BCAST: scf.yield {{.*}}async_task_id = array<i32: 0, 1>
  // BCAST: } attributes {
  // BCAST-SAME: tt.warp_specialize
  // BCAST-SAME: ttg.partition.types = ["computation", "load"]

  // DEPTH2-LABEL: @while_atomic_broadcast
  // tile-prefetch-depth=2 tags the slot for 2-deep multi-buffering (consumed by
  // the memory planner later in the full pipeline).
  // DEPTH2: ttg.local_alloc {ttg.atomic_broadcast_copies = 2 : i32}
  // DEPTH2: tt.atomic_rmw
  // DEPTH2-NOT: tt.atomic_rmw
  // DEPTH2: tt.unsplat
  tt.func public @while_atomic_broadcast(
      %a_desc: !tt.tensordesc<128x64xf16, #shared>,
      %b_desc: !tt.tensordesc<64x256xf16, #shared>,
      %out: !tt.ptr<f32>, %counter: !tt.ptr<i32>, %k_tiles: i32, %num_tiles: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %acc_init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %start = tt.get_program_id x : i32
    %result = scf.while (%tile = %start) : (i32) -> i32 {
      %valid = arith.cmpi slt, %tile, %num_tiles : i32
      scf.condition(%valid) %tile : i32
    } do {
    ^bb0(%tile: i32):
      %inner = scf.for %ki = %c0 to %k_tiles step %c1 iter_args(%acc = %acc_init) -> (tensor<128x256xf32, #mma>) : i32 {
        %a = tt.descriptor_load %a_desc[%tile, %ki] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %a_alloc = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%ki, %tile] : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
        %b_alloc = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        %dot = ttng.warp_group_dot %a_alloc, %b_alloc, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        scf.yield %dot : tensor<128x256xf32, #mma>
      }
      %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked1>
      %cvt = ttg.convert_layout %inner : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
      tt.store %ptrs, %cvt : tensor<128x256x!tt.ptr<f32>, #blocked1>
      %next = tt.atomic_rmw add, acq_rel, gpu, %counter, %c1, %true : (!tt.ptr<i32>, i32, i1) -> i32
      scf.yield %next : i32
    } attributes {tt.warp_specialize}
    tt.return
  }
}
