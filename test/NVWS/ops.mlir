// RUN: triton-opt --split-input-file %s | FileCheck %s

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @semaphore_create
  // CHECK: nvws.semaphore.create {{.*}} released = -1
  // CHECK: nvws.semaphore.create {{.*}}
  tt.func @semaphore_create(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %0 = nvws.semaphore.create %d, %e released = -1 : !nvws.semaphore<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    %1 = nvws.semaphore.create %d : !nvws.semaphore<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>]>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @semaphore_acquire_buffer
  // CHECK: nvws.semaphore.create {{.*}} released = 5
  // CHECK: nvws.semaphore.acquire {{.*}} : <[{{.*}}]> -> !ttg.async.token
  // CHECK: nvws.semaphore.acquire {{.*}}[{{.*}}, {{.*}}] : <[{{.*}}]> -> !ttg.async.token
  // CHECK: nvws.semaphore.buffer {{.*}}, {{.*}} : <[{{.*}}]>, !ttg.async.token ->
  // CHECK: nvws.semaphore.buffer {{.*}}[{{.*}}], {{.*}} : <[{{.*}}]>, !ttg.async.token ->
  tt.func @semaphore_acquire_buffer(%d : !ttg.memdesc<3x64x16xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = nvws.semaphore.create %d released = -1 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>
    %mask = nvws.semaphore.create %d released = 5 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>
    %1 = nvws.semaphore.acquire %0 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]> -> !ttg.async.token
    %2 = nvws.semaphore.acquire %0[%c1_i32, %c0_i32] : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]> -> !ttg.async.token
    %3 = nvws.semaphore.buffer %0, %1 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>
    %4 = nvws.semaphore.buffer %0[%c1_i32], %2 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @semaphore_release
  // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
  // CHECK: nvws.semaphore.release {{.*}}[{{.*}}], {{.*}} [#nvws.async_op<tma_load>, #nvws.async_op<tc5mma>] : <[{{.*}}]>, !ttg.async.token
  tt.func @semaphore_release(%d : !ttg.memdesc<3x64x16xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = nvws.semaphore.create %d : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>
    %1 = nvws.semaphore.acquire %0 : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]> -> !ttg.async.token
    nvws.semaphore.release %0, %1 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>, !ttg.async.token
    nvws.semaphore.release %0[%c0_i32], %1 [#nvws.async_op<tma_load>, #nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>]>, !ttg.async.token
    tt.return
  }
}

// -----

// CHECK-LABEL: @warp_group_nothing
tt.func @warp_group_nothing() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  tt.return
}

// CHECK-LABEL: @warp_1_partition
tt.func @warp_1_partition() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  // CHECK-NEXT:  num_warps(4) {
  partition0  num_warps(4) {
  // CHECK-NEXT: nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  tt.return
}

// CHECK-LABEL: @warp_2_partition
tt.func @warp_2_partition() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  // CHECK-NEXT: partition0  num_warps(8) {
  partition0  num_warps(8) {
  // CHECK-NEXT: nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: partition1 num_warps(4) {
  partition1 num_warps(4) {
  // CHECK-NEXT:   nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  tt.return
}

// CHECK-LABEL: @warp_group_results
tt.func @warp_group_results(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK-NEXT: %[[WG:.*]]:2 = nvws.warp_group
  %0:2 = nvws.warp_group
  // CHECK-NEXT: partition0 num_warps(4) {
  partition0 num_warps(4) {
  // CHECK-NEXT: nvws.warp_group.yield %{{.*}}, %{{.*}} : i32, i32
    nvws.warp_group.yield %arg0, %arg1 : i32, i32
  // CHECK-NEXT: }
  }
  tt.return %0#0, %0#1 : i32, i32
}

// CHECK-LABEL: @token_producer_consumer
tt.func @token_producer_consumer() {

  // CHECK: nvws.create_token
  // CHECK: nvws.producer_acquire
  // CHECK: nvws.producer_commit
  // CHECK: nvws.consumer_wait
  // CHECK: nvws.consumer_release

  %0 = nvws.create_token {loadType = 1 : i32, numBuffers = 3 : i32} : tensor<3x!nvws.token>

  %c0_i32 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 0 : i32
  %false = arith.constant {async_task_id = dense<0> : vector<1xi32>} false

  nvws.producer_acquire %0, %c0_i32, %false {async_task_id = dense<0> : vector<1xi32>} : tensor<3x!nvws.token>, i32, i1
  nvws.producer_commit %0, %c0_i32 {async_task_id = dense<0> : vector<1xi32>} : tensor<3x!nvws.token>, i32
  nvws.consumer_wait %0, %c0_i32, %false {async_task_id = dense<1> : vector<1xi32>} : tensor<3x!nvws.token>, i32, i1
  nvws.consumer_release %0, %c0_i32 {async_task_id = dense<1> : vector<1xi32>} : tensor<3x!nvws.token>, i32
  tt.return
}

// CHECK-LABEL: @token_with_ws_constraints
tt.func @token_with_ws_constraints() {

  // CHECK: nvws.producer_acquire
  // CHECK-SAME: constraints = {WSBarrier = {dstTask = 1 : i32}}
  // CHECK: nvws.producer_commit
  // CHECK-SAME: constraints = {WSBarrier = {dstTask = 1 : i32}}
  // CHECK: nvws.consumer_wait
  // CHECK-SAME: constraints = {WSBarrier = {dstTask = 0 : i32}}
  // CHECK: nvws.consumer_release
  // CHECK-SAME: constraints = {WSBarrier = {dstTask = 0 : i32}}

  %0 = nvws.create_token {loadType = 1 : i32, numBuffers = 3 : i32} : tensor<3x!nvws.token>

  %c0_i32 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 0 : i32
  %false = arith.constant {async_task_id = dense<0> : vector<1xi32>} false

  nvws.producer_acquire %0, %c0_i32, %false {async_task_id = dense<0> : vector<1xi32>, constraints = {WSBarrier = {dstTask = 1 : i32}}} : tensor<3x!nvws.token>, i32, i1
  nvws.producer_commit %0, %c0_i32 {async_task_id = dense<0> : vector<1xi32>, constraints = {WSBarrier = {dstTask = 1 : i32}}} : tensor<3x!nvws.token>, i32
  nvws.consumer_wait %0, %c0_i32, %false {async_task_id = dense<1> : vector<1xi32>, constraints = {WSBarrier = {dstTask = 0 : i32}}} : tensor<3x!nvws.token>, i32, i1
  nvws.consumer_release %0, %c0_i32 {async_task_id = dense<1> : vector<1xi32>, constraints = {WSBarrier = {dstTask = 0 : i32}}} : tensor<3x!nvws.token>, i32
  tt.return
}
