// RUN: triton-opt %s -split-input-file --nvgpu-test-taskid-propagate=num-warp-groups=2 | FileCheck %s --check-prefix=TASKID
// RUN: triton-opt %s -split-input-file --nvgpu-ws-data-partition=num-warp-groups=3 | FileCheck %s --check-prefix=DATAPART
// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-code-partition="num-buffers=1" | FileCheck %s --check-prefix=CODEPART

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
  // DATAPART: tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16
  // DATAPART: arith.addi %{{.*}}, %{{.*}} : i32
  // DATAPART: tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16
  // DATAPART: ttng.warp_group_dot {{.*}} -> tensor<64x256xf32
  // DATAPART: ttng.warp_group_dot {{.*}} -> tensor<64x256xf32
  tt.func public @while_descriptor_data_partition(%desc_storage: !tt.ptr<i8>, %out: !tt.ptr<f32>, %bound: i32) {
    %true = arith.constant {async_task_id = array<i32: 0, 1, 2>} true
    %false = arith.constant {async_task_id = array<i32: 0, 1, 2>} false
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c64 = arith.constant {async_task_id = array<i32: 0>} 64 : i32
    %acc_init = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %desc_a = ttng.reinterpret_tensor_descriptor %desc_storage {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<tensor<128x64xf16>>
    %desc_b = ttng.reinterpret_tensor_descriptor %desc_storage {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xf16>>
    %result:2 = scf.while (%acc = %acc_init, %off = %c0, %cond = %true) : (tensor<128x256xf32, #mma>, i32, i1) -> (tensor<128x256xf32, #mma>, i32) {
      scf.condition(%cond) %acc, %off : tensor<128x256xf32, #mma>, i32
    } do {
    ^bb0(%acc: tensor<128x256xf32, #mma>, %off: i32):
      %a = tt.descriptor_load %desc_a[%off, %c0] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked>
      %a_alloc = ttg.local_alloc %a {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %desc_b[%c0, %off] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #blocked1>
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
