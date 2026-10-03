// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory -test-tritonamdgpu-membar | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --test-tritonamdgpu-membar | FileCheck %s --check-prefix=SCF
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --test-tritonamdgpu-membar --test-tritonamdgpu-membar | FileCheck %s --check-prefix=SCF

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#A_SHARED_T = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [0, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// Check that we only get a single barrier when using AsyncWait
// CHECK-LABEL: pipelined_async_copy_local_to_global
tt.func @pipelined_async_copy_local_to_global(%A: !tt.ptr<f16>) {
  %index_0 = arith.constant 0 : i32
  %index_1 = arith.constant 1 : i32
  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_a = ttg.memdesc_index %alloc[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_b = ttg.memdesc_index %alloc[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Load TileA
  %1 = ttg.async_copy_global_to_local %a_ptr, %tile_a: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Wait for TileA
  %2 = ttg.async_wait %1 {num = 4 : i32}
  // Read TileA
  %4 = ttg.local_load %tile_a token %2 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  // Load into TileB
  %3 = ttg.async_copy_global_to_local %a_ptr, %tile_b : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // There should be a single barrier after async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.return
  tt.return
}
// Same as above but different order of ops
// CHECK-LABEL: pipelined_async_copy_local_to_global_2
tt.func @pipelined_async_copy_local_to_global_2(%A: !tt.ptr<f16>) {
  %index_0 = arith.constant 0 : i32
  %index_1 = arith.constant 1 : i32
  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_a = ttg.memdesc_index %alloc[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_b = ttg.memdesc_index %alloc[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Load Tile
  %1 = ttg.async_copy_global_to_local %a_ptr, %tile_a: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Wait for TileA
  %2 = ttg.async_wait %1 {num = 4 : i32}
  // Load into TileB
  %3 = ttg.async_copy_global_to_local %a_ptr, %tile_b : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Read TileA
  %4 = ttg.local_load %tile_a token %2 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  // There should be a single barrier after async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.return
  tt.return
}
// Check that multiple LocalLoads waiting on the same AsyncWait produce one barrier
// CHECK-LABEL: pipelined_async_copy_local_to_global_3
tt.func @pipelined_async_copy_local_to_global_3(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>) {
  %index_0 = arith.constant 0 : i32
  %index_1 = arith.constant 1 : i32
  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>
  %b_ptr = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>

  %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_a_1 = ttg.memdesc_index %alloc_a[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_a_2 = ttg.memdesc_index %alloc_a[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_b_1 = ttg.memdesc_index %alloc_b[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_b_2 = ttg.memdesc_index %alloc_b[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  // Load TileA_1
  %1 = ttg.async_copy_global_to_local %a_ptr, %tile_a_1: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Load TileB_1
  %2 = ttg.async_copy_global_to_local %b_ptr, %tile_b_1: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Wait for TileA
  %3 = ttg.async_wait %1, %2 {num = 4 : i32}
  // Read TileA_1
  %4 = ttg.local_load %tile_a_1 token %3 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  // Read TileB_1
  %5 = ttg.local_load %tile_b_1 token %3 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  // Load into TileA_2
  %6 = ttg.async_copy_global_to_local %a_ptr, %tile_a_2 : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Load into TileB_2
  %7 = ttg.async_copy_global_to_local %b_ptr, %tile_b_2 : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  // There should be a single barrier after async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.return
  tt.return
}

// A wait only orders the async producer before the local consumer. It does not
// release the consumed LDS slot for a later cooperative refill. Reusing the
// same ring slot therefore needs a second workgroup barrier after local_load.
// CHECK-LABEL: same_slot_refill_after_async_wait
tt.func @same_slot_refill_after_async_wait(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %slot0 = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %async0 = amdg.buffer_load_to_local %A[%offset] into %slot0 : <f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %commit0 = ttg.async_commit_group tokens %async0
  %wait0 = amdg.async_wait %commit0 {num_inst = 0 : i32}
  // CHECK: amdg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: amdg.buffer_load_to_local
  %value = ttg.local_load %slot0 {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<128x32xf16, #AL>
  %async1 = amdg.buffer_load_to_local %B[%offset] into %slot0 : <f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  tt.return
}

// A different constant ring slot is independently owned, so precise allocation
// slices must avoid a false release barrier after local_load.
// CHECK-LABEL: distinct_slot_prefetch_after_async_wait
tt.func @distinct_slot_prefetch_after_async_wait(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %slot0 = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %slot1 = ttg.memdesc_index %alloc[%c1_i32] : !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %slot0_t = ttg.memdesc_trans %slot0 {order=array<i32: 1,0>} : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x128xf16, #A_SHARED_T, #ttg.shared_memory, mutable>

  %async0 = amdg.buffer_load_to_local %A[%offset] into %slot0 : <f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %commit0 = ttg.async_commit_group tokens %async0
  %wait0 = amdg.async_wait %commit0 {num_inst = 0 : i32}
  // CHECK: amdg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  // CHECK-NEXT: amdg.buffer_load_to_local
  %value = ttg.local_load %slot0_t {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<32x128xf16, #A_SHARED_T, #ttg.shared_memory, mutable> -> tensor<32x128xf16, #AL>
  %async1 = amdg.buffer_load_to_local %B[%offset] into %slot1 : <f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  tt.return
}

// A loop-carried AsyncWait token orders the prior async producer, but reusing
// the same allocation still needs a release barrier after LocalLoad.
// CHECK-LABEL: async_wait_in_previous_loop_iteration
tt.func @async_wait_in_previous_loop_iteration(%a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}

  // CHECK: cf.br
  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    // CHECK: ttg.local_load
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttg.async_copy_global_to_local
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

    // CHECK: ttg.async_wait
    %8 = ttg.async_wait %7 {num = 4 : i32}
    // CHECK: ttg.barrier local
    // CHECK-NOT: ttg.barrier local
    scf.yield %8: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// Check we do get a barrier for LocalLoad if the initial loop token does not come from AsyncWait
// CHECK-LABEL: intial_loop_token_is_not_from_async_wait
tt.func @intial_loop_token_is_not_from_async_wait(%a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %1) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    // CHECK: ttg.local_load
    // CHECK: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    %8 = ttg.async_wait %7 {num = 4 : i32}
    scf.yield %8: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// Same as above but the loop carried token does not come from AsyncWait
// CHECK-LABEL: loop_carried_token_not_from_async_wait
tt.func @loop_carried_token_not_from_async_wait(%a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}
  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    // CHECK: ttg.local_load
    // CHECK: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    scf.yield %7: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}


// Tokens from either branch order the next producer-to-consumer edge, but the
// same-slot refill before the branch still needs a release barrier.
// CHECK-LABEL: async_wait_inside_if
tt.func @async_wait_inside_if(%cond: i1, %a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}

  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    // CHECK: ttg.local_load
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    %103 = scf.if %cond -> (!ttg.async.token) {
      %8 = ttg.async_wait %7 {num = 4 : i32}
      scf.yield %8 : !ttg.async.token
    } else {
      %9 = ttg.async_wait %7 {num = 4 : i32}
      scf.yield %9 : !ttg.async.token
    }
    scf.yield %103: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// Check that we do get a barrier for an if where one branch does not yield an token from AsyncWait
// CHECK-LABEL: non_async_wait_token_from_then
tt.func @non_async_wait_token_from_then(%cond: i1, %a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}

  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    // We should get a barrier because the then branch does not yield an token from AsyncWait
    // CHECK: ttg.local_load
    // CHECK: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    %103 = scf.if %cond -> (!ttg.async.token) {
      scf.yield %7 : !ttg.async.token
    } else {
      %8 = ttg.async_wait %7 {num = 4 : i32}
      scf.yield %8 : !ttg.async.token
    }
    scf.yield %103: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// See above
// CHECK-LABEL: non_async_wait_token_from_else
tt.func @non_async_wait_token_from_else(%cond: i1, %a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}

  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
    // We should get a barrier because the else branch does not yield an token from AsyncWait
    // CHECK: ttg.local_load
    // CHECK: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    %103 = scf.if %cond -> (!ttg.async.token) {
      %8 = ttg.async_wait %7 {num = 4 : i32}
      scf.yield %8 : !ttg.async.token
    } else {
      %9 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
      scf.yield %9 : !ttg.async.token
    }
    scf.yield %103: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// CHECK-LABEL: missing_barrier_reused_allocation
tt.func @missing_barrier_reused_allocation(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %alloc1 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>

  %slice1_0 = ttg.memdesc_index %alloc1[%c0_i32] : !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %async1 = amdg.buffer_load_to_local %A[%offset] into %slice1_0 : <f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %token1 = ttg.async_commit_group tokens %async1
  %wait1 = amdg.async_wait %token1 {num_inst = 0 : i32}
  // CHECK: ttg.barrier local
  // CHECK: ttg.local_load
  %local_load = ttg.local_load %slice1_0 token %wait1 {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<128x32xf16, #AL>
  ttg.local_dealloc %alloc1 : !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %alloc2 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %slice2_0 = ttg.memdesc_index %alloc2[%c0_i32] : !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // op2: Async load into alloc2 (overlapping with the dealloc'd alloc1 that is still being local_load'd from)
  // CHECK: ttg.barrier local
  // CHECK-NEXT: amdg.buffer_load_to_local
  %async2 = amdg.buffer_load_to_local %B[%offset] into %slice2_0 : <f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %token2 = ttg.async_commit_group tokens %async2
  %wait2 = amdg.async_wait %token2 {num_inst = 0 : i32}
  // CHECK: ttg.barrier local
  tt.return
}

// tlx.workgroup_barrier lowers to `rocdl.sched.barrier none; ttg.barrier local;
// rocdl.sched.barrier none`. When Membar scans forward from an async wait for a
// sync point it must look THROUGH the scheduling-only sched fences to reach the
// real ttg.barrier; otherwise it treats the first sched fence as a stopping
// point and inserts a redundant barrier right after the wait, doubling the
// workgroup barrier in a hand-rolled ping-pong. Here the wait is immediately
// followed by the sched/barrier bracket, so the forward scan actually reaches
// the sched fences (this fails without the look-through: an extra
// `ttg.barrier local` appears between the async_wait and the first sched fence).
// CHECK-LABEL: workgroup_barrier_after_async_wait_needs_no_barrier
tt.func @workgroup_barrier_after_async_wait_needs_no_barrier(%A: !tt.ptr<f16>) {
  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>
  %alloc = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %async1 = amdg.buffer_load_to_local %A[%offset] into %alloc : <f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %token1 = ttg.async_commit_group tokens %async1
  %wait1 = amdg.async_wait %token1 {num_inst = 0 : i32}
  // No barrier is inserted between the async wait and the workgroup barrier; the
  // wait's deferred barrier is satisfied by the real ttg.barrier below.
  // CHECK: amdg.async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: rocdl.sched.barrier
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: rocdl.sched.barrier
  // CHECK-NOT: ttg.barrier local
  rocdl.sched.barrier none
  ttg.barrier local
  rocdl.sched.barrier none
  %read = ttg.local_load %alloc token %wait1 {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<128x32xf16, #AL>
  tt.return
}

// A lone scheduling fence is NOT a sync point. After looking through the sched
// fence the forward scan reaches the local_load's memory effect with no real
// ttg.barrier in between, so Membar must still insert a barrier after the wait.
// This guards the look-through against over-suppressing genuinely needed
// barriers.
// CHECK-LABEL: lone_sched_fence_still_needs_barrier
tt.func @lone_sched_fence_still_needs_barrier(%A: !tt.ptr<f16>) {
  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>
  %alloc = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %async1 = amdg.buffer_load_to_local %A[%offset] into %alloc : <f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %token1 = ttg.async_commit_group tokens %async1
  %wait1 = amdg.async_wait %token1 {num_inst = 0 : i32}
  // The wait's deferred barrier is inserted before the lone sched fence.
  // CHECK: amdg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: rocdl.sched.barrier
  // CHECK-NEXT: ttg.local_load
  rocdl.sched.barrier none
  %read = ttg.local_load %alloc token %wait1 {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<128x32xf16, #AL>
  tt.return
}

}

// -----

#src = #ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[0, 32], [32, 0]], block = []}>
#dst = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [16, 0], [0, 64]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[0, 32], [32, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // Differently sized live buffers put the identical conversions at overlapping
  // but unequal scratch intervals. Do not infer physical wave ownership from
  // logical tensor types when the scratch bases differ.
  // CHECK-LABEL: @warp_local_scratch_reuse_reject_different_interval
  // CHECK: ttg.local_alloc{{.*}}allocation.offset = 0
  // CHECK: ttg.convert_layout{{.*}}allocation.offset = 24576
  // CHECK: ttg.local_dealloc
  // CHECK: ttg.local_alloc{{.*}}allocation.offset = 0
  // CHECK: ttg.barrier local
  // CHECK-NEXT: {{.*}}ttg.convert_layout{{.*}}allocation.offset = 16384
  // CHECK: tt.return
  tt.func @warp_local_scratch_reuse_reject_different_interval(%a: tensor<64x128xbf16, #src>, %b: tensor<64x128xbf16, #src>, %out: tensor<64x128x!tt.ptr<bf16>, #dst>) {
    %large = ttg.local_alloc : () -> !ttg.memdesc<3x64x64xbf16, #shared, #smem, mutable>
    %x = ttg.convert_layout %a : tensor<64x128xbf16, #src> -> tensor<64x128xbf16, #dst>
    tt.store %out, %x : tensor<64x128x!tt.ptr<bf16>, #dst>
    ttg.local_dealloc %large : !ttg.memdesc<3x64x64xbf16, #shared, #smem, mutable>
    %small = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
    %y = ttg.convert_layout %b : tensor<64x128xbf16, #src> -> tensor<64x128xbf16, #dst>
    tt.store %out, %y : tensor<64x128x!tt.ptr<bf16>, #dst>
    ttg.local_dealloc %small : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
    tt.return
  }

  // In structured IR the false path retains the same-block x -> z dependency,
  // which qualifies for a wave fence. The true path contributes y -> z from a
  // different block, which needs a CTA barrier. The CTA barrier must win over
  // the qualifying dependency, including when the analysis is run again.
  // CHECK-LABEL: @warp_local_scratch_reuse_mixed_control_flow
  // CHECK-NOT: llvm.amdgcn.wave.barrier
  // CHECK: ttg.convert_layout{{.*}}allocation.offset = 0
  // CHECK-NOT: llvm.amdgcn.wave.barrier
  // CHECK: ttg.barrier local
  // CHECK-NEXT: {{.*}}ttg.convert_layout{{.*}}allocation.offset = 0
  // CHECK-NOT: llvm.amdgcn.wave.barrier
  // CHECK: ttg.barrier local
  // CHECK-NEXT: {{.*}}ttg.convert_layout{{.*}}allocation.offset = 0
  // CHECK-NOT: llvm.amdgcn.wave.barrier
  // CHECK: tt.return
  // SCF-LABEL: @warp_local_scratch_reuse_mixed_control_flow
  // SCF-NOT: llvm.amdgcn.wave.barrier
  // SCF-NOT: ttg.barrier local
  // SCF: ttg.convert_layout{{.*}}allocation.offset = 0
  // SCF-NEXT: tt.store
  // SCF-NEXT: scf.if
  // SCF-NEXT: ttg.barrier local
  // SCF-NEXT: {{.*}}ttg.convert_layout{{.*}}allocation.offset = 0
  // SCF-NEXT: tt.store
  // SCF-NEXT: }
  // SCF-NEXT: ttg.barrier local
  // SCF-NEXT: {{.*}}ttg.convert_layout{{.*}}allocation.offset = 0
  // SCF-NEXT: tt.store
  // SCF-NEXT: tt.return
  tt.func @warp_local_scratch_reuse_mixed_control_flow(%a: tensor<64x128xbf16, #src>, %b: tensor<64x128xbf16, #src>, %c: tensor<64x128xbf16, #src>, %out: tensor<64x128x!tt.ptr<bf16>, #dst>, %cond: i1) {
    %x = ttg.convert_layout %a : tensor<64x128xbf16, #src> -> tensor<64x128xbf16, #dst>
    tt.store %out, %x : tensor<64x128x!tt.ptr<bf16>, #dst>
    scf.if %cond {
      %y = ttg.convert_layout %b : tensor<64x128xbf16, #src> -> tensor<64x128xbf16, #dst>
      tt.store %out, %y : tensor<64x128x!tt.ptr<bf16>, #dst>
    }
    %z = ttg.convert_layout %c : tensor<64x128xbf16, #src> -> tensor<64x128xbf16, #dst>
    tt.store %out, %z : tensor<64x128x!tt.ptr<bf16>, #dst>
    tt.return
  }
}
