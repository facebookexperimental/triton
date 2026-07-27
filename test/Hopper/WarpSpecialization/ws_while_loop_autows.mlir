// RUN: triton-opt %s -split-input-file --nvgpu-test-taskid-propagate=num-warp-groups=2 | FileCheck %s --check-prefix=TASKID
// RUN: triton-opt %s -split-input-file --nvgpu-ws-data-partition=num-warp-groups=3 | FileCheck %s --check-prefix=DATAPART
// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-code-partition="num-buffers=1" | FileCheck %s --check-prefix=CODEPART
// RUN: triton-opt %s -split-input-file --nvgpu-ws-data-partition=num-warp-groups=3 --triton-simplify-single-trip-while --nvgpu-partition-scheduling-meta | FileCheck %s --check-prefix=DEFERRED
// RUN: triton-opt %s -split-input-file --nvgpu-partition-scheduling-meta | FileCheck %s --check-prefix=PSM
// RUN: triton-opt %s -split-input-file --nvgpu-partition-scheduling-meta --nvgpu-test-taskid-propagate=num-warp-groups=2 | FileCheck %s --check-prefix=PSM-TASKID
// RUN: triton-opt %s -split-input-file --nvgpu-partition-scheduling-meta --nvgpu-partition-scheduling-meta | FileCheck %s --check-prefix=PSM-ROUNDTRIP

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // TASKID-LABEL: @while_task_id_propagation
  // TASKID: scf.while
  // TASKID: scf.condition
  // TASKID-SAME: {async_task_id = array<i32: 0, 1>}
  // TASKID: scf.yield
  // TASKID-SAME: {async_task_id = array<i32: 0, 1>}
  // TASKID: attributes {async_task_id = array<i32: 0, 1>}
  tt.func public @while_task_id_propagation(%arg0: i32) {
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result = scf.while (%off = %c0, %cond = %true) : (i32, i1) -> i32 {
      scf.condition(%cond) %off : i32
    } do {
    ^bb0(%off: i32):
      %new_off = arith.addi %off, %c1 {async_task_id = array<i32: 0>} : i32
      %use = arith.addi %new_off, %arg0 {async_task_id = array<i32: 1>} : i32
      scf.yield %use, %false : i32, i1
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [16, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // PSM-LABEL: @while_partition_warp_budget_rejected
  // PSM-NOT: tt.warp_specialize
  // PSM-NOT: ttg.partition
  // PSM: tt.return
  tt.func public @while_partition_warp_budget_rejected(
      %a_ptr: !tt.ptr<f16>, %b_ptr: !tt.ptr<f16>, %num_tiles: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %acc_init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %result:2 = scf.while (%tile = %c0, %acc = %acc_init) : (i32, tensor<128x256xf32, #mma>) -> (i32, tensor<128x256xf32, #mma>) {
      %valid = arith.cmpi slt, %tile, %num_tiles : i32
      scf.condition(%valid) %tile, %acc : i32, tensor<128x256xf32, #mma>
    } do {
    ^bb0(%tile: i32, %acc: tensor<128x256xf32, #mma>):
      %a_ptrs = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
      %a = tt.load %a_ptrs : tensor<128x64x!tt.ptr<f16>, #blocked>
      %a_alloc = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_ptrs = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
      %b = tt.load %b_ptrs : tensor<64x256x!tt.ptr<f16>, #blocked1>
      %b_alloc = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_alloc, %b_alloc, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %next = arith.addi %tile, %c1 : i32
      scf.yield %next, %dot : i32, tensor<128x256xf32, #mma>
    } attributes {tt.warp_specialize}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // PSM-LABEL: @while_partition_scheduling
  // PSM: scf.while
  // PSM: arith.cmpi
  // PSM: scf.condition
  // PSM: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD:[0-9]+]]>
  // PSM: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD]]>
  // PSM: ttng.warp_group_dot {{.*}}ttg.partition = array<i32: [[COMPUTE:[0-9]+]]>
  // PSM: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[COMPUTE]]>
  // PSM: } attributes {
  // PSM-SAME: tt.warp_specialize
  // PSM-SAME: ttg.partition.stages
  // PSM-SAME: ttg.partition.types = ["computation", "load"]
  // PSM-SAME: ttg.warp_specialize.tag

  // PSM-ROUNDTRIP-LABEL: @while_partition_scheduling
  // PSM-ROUNDTRIP: tt.descriptor_load {{.*}}ttg.partition = array<i32: 1>
  // PSM-ROUNDTRIP: ttng.warp_group_dot {{.*}}ttg.partition = array<i32: 0>
  // PSM-ROUNDTRIP: } attributes {
  // PSM-ROUNDTRIP-SAME: tt.warp_specialize
  // PSM-ROUNDTRIP-SAME: ttg.partition.types = ["computation", "load"]

  // PSM-TASKID-LABEL: @while_partition_scheduling
  // PSM-TASKID: scf.while
  // PSM-TASKID: arith.cmpi
  // PSM-TASKID-SAME: async_task_id = array<i32: 0, 1>
  // PSM-TASKID: scf.condition
  // PSM-TASKID-SAME: async_task_id = array<i32: 0, 1>
  // PSM-TASKID: arith.addi
  // PSM-TASKID-SAME: async_task_id = array<i32: 0, 1>
  // PSM-TASKID-NEXT: scf.yield
  // PSM-TASKID-SAME: async_task_id = array<i32: 0, 1>
  tt.func public @while_partition_scheduling(
      %a_desc: !tt.tensordesc<128x64xf16, #shared>,
      %b_desc: !tt.tensordesc<64x256xf16, #shared>,
      %out: !tt.ptr<f32>, %k_tiles: i32, %num_tiles: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %acc_init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %result = scf.while (%tile = %c0) : (i32) -> i32 {
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
      %next = arith.addi %tile, %c1 : i32
      scf.yield %next : i32
    } attributes {tt.warp_specialize}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // PSM-LABEL: @while_partition_reordered_rejected
  // PSM-NOT: tt.warp_specialize
  // PSM-NOT: ttg.partition
  // PSM: tt.return
  tt.func public @while_partition_reordered_rejected() {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result:2 = scf.while (%i = %c0, %j = %c1) : (i32, i32) -> (i32, i32) {
      scf.condition(%true) %j, %i : i32, i32
    } do {
    ^bb0(%j: i32, %i: i32):
      %next = arith.addi %i, %j {ttg.partition = array<i32: 0>} : i32
      scf.yield %next, %j : i32, i32
    } attributes {tt.warp_specialize, ttg.partition.stages = [0 : i32], ttg.partition.types = ["computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // A condition-only slot may precede a forwarded slot, as it does for the
  // CLC scheduler's (valid, x) carry. The after-region argument must map back
  // to yield slot 1 rather than assuming after-argument index 0.
  // PSM-LABEL: @while_partition_ordered_subset
  // PSM: %[[RESULT:.*]] = scf.while
  // PSM: scf.condition
  // PSM: ^bb0(%[[J:.*]]: i32):
  // PSM: arith.addi %[[J]], {{.*}} {ttg.partition = array<i32: 0>}
  // PSM: scf.yield
  // PSM: attributes {
  // PSM-SAME: tt.warp_specialize
  // PSM-SAME: ttg.partition.stages = [0 : i32]
  // PSM-SAME: ttg.partition.types = ["computation"]
  // PSM-ROUNDTRIP-LABEL: @while_partition_ordered_subset
  // PSM-ROUNDTRIP: arith.addi {{.*}} {ttg.partition = array<i32: 0>}
  // PSM-ROUNDTRIP: attributes {
  // PSM-ROUNDTRIP-SAME: tt.warp_specialize
  // PSM: tt.return
  tt.func public @while_partition_ordered_subset() {
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result = scf.while (%valid = %true, %j = %c0) : (i1, i32) -> i32 {
      scf.condition(%valid) %j : i32
    } do {
    ^bb0(%j: i32):
      %next = arith.addi %j, %c1 {ttg.partition = array<i32: 0>} : i32
      scf.yield %false, %next : i1, i32
    } attributes {tt.warp_specialize, ttg.partition.stages = [0 : i32], ttg.partition.types = ["computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // PSM-LABEL: @while_partition_computed_forward_rejected
  // PSM-NOT: tt.warp_specialize
  // PSM-NOT: ttg.partition
  // PSM: tt.return
  tt.func public @while_partition_computed_forward_rejected() {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result = scf.while (%i = %c0, %j = %c1) : (i32, i32) -> i32 {
      %sum = arith.addi %i, %j : i32
      scf.condition(%true) %sum : i32
    } do {
    ^bb0(%sum: i32):
      %next = arith.addi %sum, %c1 {ttg.partition = array<i32: 0>} : i32
      scf.yield %next, %sum : i32, i32
    } attributes {tt.warp_specialize, ttg.partition.stages = [0 : i32], ttg.partition.types = ["computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Data partitioning must see the outer while's factor before simplification.
  // Partition scheduling must then see warp_specialize on the forwarded loop.
  // DEFERRED-LABEL: @deferred_single_trip_while
  // DEFERRED-NOT: scf.while
  // DEFERRED: scf.for
  // DEFERRED: ttng.warp_group_dot
  // DEFERRED-SAME: ttg.partition
  // DEFERRED: ttng.warp_group_dot
  // DEFERRED-SAME: ttg.partition
  // DEFERRED: } {tt.data_partition_factor = 2 : i32
  // DEFERRED-SAME: tt.warp_specialize
  tt.func public @deferred_single_trip_while(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %out: !tt.ptr<f32>, %k_tiles: index) {
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : i32
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    scf.while (%valid = %true) : (i1) -> () {
      scf.condition(%valid)
    } do {
      %acc_init = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
      %inner = scf.for %ki = %c0_idx to %k_tiles step %c1_idx iter_args(%iter_acc = %acc_init) -> (tensor<128x256xf32, #mma>) {
        %a_ptrs = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
        %a = tt.load %a_ptrs : tensor<128x64x!tt.ptr<f16>, #blocked>
        %a_alloc = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b_ptrs = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
        %b = tt.load %b_ptrs : tensor<64x256x!tt.ptr<f16>, #blocked1>
        %b_alloc = ttg.local_alloc %b : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        %dot = ttng.warp_group_dot %a_alloc, %b_alloc, %iter_acc {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        scf.yield %dot : tensor<128x256xf32, #mma>
      }
      %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked1>
      %cvt = ttg.convert_layout %inner : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
      tt.store %ptrs, %cvt : tensor<128x256x!tt.ptr<f32>, #blocked1>
      scf.yield %false : i1
    } attributes {tt.warp_specialize, tt.data_partition_factor = 2 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // TASKID-LABEL: @while_condition_region_propagation
  // TASKID: %[[WHILE:.*]] = scf.while
  // TASKID: arith.cmpi {{.*}} {async_task_id = array<i32: 0, 1>}
  // TASKID: scf.condition
  // TASKID-SAME: {async_task_id = array<i32: 0, 1>}
  // TASKID: scf.yield
  // TASKID-SAME: {async_task_id = array<i32: 0, 1>}
  // TASKID: arith.addi %[[WHILE]], {{.*}} {async_task_id = array<i32: 1>}
  tt.func public @while_condition_region_propagation(%bound: i32, %out: tensor<16x!tt.ptr<i32>, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero_vec = arith.constant dense<0> : tensor<16xi32, #blocked>
    %result = scf.while (%i = %c0) : (i32) -> i32 {
      %keep_going = arith.cmpi slt, %i, %bound {async_task_id = array<i32: 0>} : i32
      scf.condition(%keep_going) %i : i32
    } do {
    ^bb0(%i: i32):
      %next = arith.addi %i, %c1 {async_task_id = array<i32: 1>} : i32
      scf.yield %next : i32
    }
    %used = arith.addi %result, %c1 {async_task_id = array<i32: 1>} : i32
    %used_vec = tt.splat %used {async_task_id = array<i32: 1>} : i32 -> tensor<16xi32, #blocked>
    %mask = arith.cmpi sge, %used_vec, %zero_vec {async_task_id = array<i32: 1>} : tensor<16xi32, #blocked>
    tt.store %out, %used_vec, %mask {async_task_id = array<i32: 1>} : tensor<16x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // DATAPART-LABEL: @while_data_partition
  // DATAPART: tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>
  // DATAPART: tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>
  // DATAPART: ttng.warp_group_dot {{.*}} -> tensor<64x256xf32
  // DATAPART: ttng.warp_group_dot {{.*}} -> tensor<64x256xf32
  tt.func public @while_data_partition(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %out: !tt.ptr<f32>, %arg2: i32) {
    %true = arith.constant {async_task_id = array<i32: 0, 1, 2>} true
    %false = arith.constant {async_task_id = array<i32: 0, 1, 2>} false
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c64 = arith.constant {async_task_id = array<i32: 0>} 64 : i32
    %acc_init = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %result:2 = scf.while (%acc = %acc_init, %off = %c0, %cond = %true) : (tensor<128x256xf32, #mma>, i32, i1) -> (tensor<128x256xf32, #mma>, i32) {
      scf.condition(%cond) %acc, %off : tensor<128x256xf32, #mma>, i32
    } do {
    ^bb0(%acc: tensor<128x256xf32, #mma>, %off: i32):
      %a_ptrs = tt.splat %arg0 {async_task_id = array<i32: 0>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
      %a = tt.load %a_ptrs {async_task_id = array<i32: 0>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      %a_alloc = ttg.local_alloc %a {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_ptrs = tt.splat %arg1 {async_task_id = array<i32: 0>} : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
      %b = tt.load %b_ptrs {async_task_id = array<i32: 0>} : tensor<64x256x!tt.ptr<f16>, #blocked1>
      %b_alloc = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_alloc, %b_alloc, %acc {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %new_off = arith.addi %off, %c64 {async_task_id = array<i32: 0>} : i32
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %dot, %new_off, %false : tensor<128x256xf32, #mma>, i32, i1
    }
    %ptrs = tt.splat %out {async_task_id = array<i32: 1, 2>} : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked1>
    %cvt = ttg.convert_layout %result#0 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
    tt.store %ptrs, %cvt {async_task_id = array<i32: 1, 2>} : tensor<128x256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // DATAPART-LABEL: @while_descriptor_data_partition
  // DATAPART: scf.while
  // DATAPART: tt.descriptor_load {{.*}} : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16
  // DATAPART: arith.addi %{{.*}}, %{{.*}} : i32
  // DATAPART: tt.descriptor_load {{.*}} : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16
  // DATAPART: ttng.warp_group_dot {{.*}} -> tensor<64x256xf32
  // DATAPART: ttng.warp_group_dot {{.*}} -> tensor<64x256xf32
  tt.func public @while_descriptor_data_partition(%desc_storage: !tt.ptr<i8>, %out: !tt.ptr<f32>, %bound: i32) {
    %true = arith.constant {async_task_id = array<i32: 0, 1, 2>} true
    %false = arith.constant {async_task_id = array<i32: 0, 1, 2>} false
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c64 = arith.constant {async_task_id = array<i32: 0>} 64 : i32
    %acc_init = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %desc_a = ttng.reinterpret_tensor_descriptor %desc_storage {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<128x64xf16>
    %desc_b = ttng.reinterpret_tensor_descriptor %desc_storage {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<64x256xf16>
    %result:2 = scf.while (%acc = %acc_init, %off = %c0, %cond = %true) : (tensor<128x256xf32, #mma>, i32, i1) -> (tensor<128x256xf32, #mma>, i32) {
      scf.condition(%cond) %acc, %off : tensor<128x256xf32, #mma>, i32
    } do {
    ^bb0(%acc: tensor<128x256xf32, #mma>, %off: i32):
      %a = tt.descriptor_load %desc_a[%off, %c0] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %a_alloc = ttg.local_alloc %a {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %desc_b[%c0, %off] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
      %b_alloc = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_alloc, %b_alloc, %acc {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %new_off = arith.addi %off, %c64 {async_task_id = array<i32: 0>} : i32
      %next_cond = arith.cmpi slt, %new_off, %bound {async_task_id = array<i32: 0>} : i32
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %dot, %new_off, %next_cond : tensor<128x256xf32, #mma>, i32, i1
    }
    %ptrs = tt.splat %out {async_task_id = array<i32: 1, 2>} : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked1>
    %cvt = ttg.convert_layout %result#0 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
    tt.store %ptrs, %cvt {async_task_id = array<i32: 1, 2>} : tensor<128x256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CODEPART-LABEL: @while_result_used_by_task
  // CODEPART: ttg.warp_specialize
  // CODEPART: partition0
  // CODEPART: %[[WHILE:.*]] = scf.while
  // CODEPART: scf.condition
  // CODEPART: arith.addi
  // CODEPART: scf.yield
  // CODEPART: arith.addi %[[WHILE]],
  // CODEPART: tt.splat
  tt.func public @while_result_used_by_task(%src: tensor<16xf32, #blocked>, %dst: tensor<16x!tt.ptr<f32>, #blocked>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : index
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : index
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %false = arith.constant {async_task_id = array<i32: 0, 1>} false
    %zero_vec = arith.constant {async_task_id = array<i32: 1>} dense<0> : tensor<16xi32, #blocked>
    %alloc = ttg.local_alloc {async_task_id = array<i32: 0>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
    scf.for %iv = %c0 to %c1 step %c1 {
      %while_result = scf.while (%carry = %c0_i32, %keep = %true) : (i32, i1) -> i32 {
        scf.condition(%keep) {async_task_id = array<i32: 1>} %carry : i32
      } do {
      ^bb0(%carry: i32):
        %next = arith.addi %carry, %c1_i32 {async_task_id = array<i32: 1>} : i32
        scf.yield {async_task_id = array<i32: 1>} %next, %false : i32, i1
      } attributes {async_task_id = array<i32: 1>}
      %used = arith.addi %while_result, %c1_i32 {async_task_id = array<i32: 1>} : i32
      %used_vec = tt.splat %used {async_task_id = array<i32: 1>} : i32 -> tensor<16xi32, #blocked>
      %mask = arith.cmpi sge, %used_vec, %zero_vec {async_task_id = array<i32: 1>} : tensor<16xi32, #blocked>
      %stored = arith.addf %src, %src {async_task_id = array<i32: 0>} : tensor<16xf32, #blocked>
      ttg.local_store %stored, %alloc {async_task_id = array<i32: 0>} : tensor<16xf32, #blocked> -> !ttg.memdesc<16xf32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {async_task_id = array<i32: 1>} : !ttg.memdesc<16xf32, #shared, #smem, mutable> -> tensor<16xf32, #blocked>
      tt.store %dst, %loaded, %mask {async_task_id = array<i32: 1>} : tensor<16x!tt.ptr<f32>, #blocked>
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["compute", "compute"]}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // A CLC (Cluster Launch Control) persistent GEMM whose outer loop is an
  // scf.while: the loop-carried `is_valid` bit is refreshed each iteration by
  // `ttng.clc_advance` (hardware work-stealing), exactly as the ClcTileScheduler
  // emits (`while sched.is_valid(): ...; sched = sched.advance()`). This shows
  // that data partitioning splits the while-carried accumulator + warp_group_dot
  // along M across the two consumer warp groups (async_task_id 1, 2) regardless
  // of how the loop advances -- the `clc_advance` producer op (task 0) is left
  // intact. (Simplified: uses warp_group_dot to keep the accumulator in
  // registers; the data-partition mechanics are identical for tc_gen5_mma.)
  // DATAPART-LABEL: @while_clc_data_partition
  // DATAPART: scf.while
  // DATAPART: tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>
  // DATAPART: tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>
  // The two consumer warp groups each get an M/2 = 64 slice of the accumulator.
  // DATAPART: ttng.warp_group_dot {{.*}} -> tensor<64x256xf32
  // DATAPART: ttng.warp_group_dot {{.*}} -> tensor<64x256xf32
  // The CLC advance (loop-carried valid producer, task 0) is left intact.
  // DATAPART: ttng.clc_advance
  tt.func public @while_clc_data_partition(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %out: !tt.ptr<f32>) {
    %true = arith.constant {async_task_id = array<i32: 0, 1, 2>} true
    %acc_init = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %result = scf.while (%acc = %acc_init, %valid = %true) : (tensor<128x256xf32, #mma>, i1) -> tensor<128x256xf32, #mma> {
      scf.condition(%valid) %acc : tensor<128x256xf32, #mma>
    } do {
    ^bb0(%acc: tensor<128x256xf32, #mma>):
      %a_ptrs = tt.splat %arg0 {async_task_id = array<i32: 0>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
      %a = tt.load %a_ptrs {async_task_id = array<i32: 0>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      %a_alloc = ttg.local_alloc %a {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_ptrs = tt.splat %arg1 {async_task_id = array<i32: 0>} : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
      %b = tt.load %b_ptrs {async_task_id = array<i32: 0>} : tensor<64x256x!tt.ptr<f16>, #blocked1>
      %b_alloc = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
      %dot = ttng.warp_group_dot %a_alloc, %b_alloc, %acc {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
      %nvalid, %x, %y, %z = ttng.clc_advance {async_task_id = array<i32: 0>} : i1, i32, i32, i32
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %dot, %nvalid : tensor<128x256xf32, #mma>, i1
    }
    %ptrs = tt.splat %out {async_task_id = array<i32: 1, 2>} : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked1>
    %cvt = ttg.convert_layout %result {async_task_id = array<i32: 1, 2>} : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1>
    tt.store %ptrs, %cvt {async_task_id = array<i32: 1, 2>} : tensor<128x256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
