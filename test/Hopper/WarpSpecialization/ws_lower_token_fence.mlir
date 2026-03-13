// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-lower-token | FileCheck %s

// Verify that a ProducerCommitOp with the `fenced` attribute lowers to a
// fence_async_shared followed by an arrive_barrier.  This fence replaces the
// one that was previously hardcoded in WSTMAStoreLowering: in the warp-
// specialized pipeline the fence is now emitted at the semantic level during
// token lowering rather than unconditionally at the TMA store lowering level.
//
// For the non-warp-specialized pipeline the fence is still emitted directly by
// TMAStoresPipeline (tested in test/TritonGPU/loop-pipeline-hopper.mlir).

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @fenced_producer_commit
//       CHECK: ttng.fence_async_shared {bCluster = false}
//       CHECK: ttng.arrive_barrier
  tt.func @fenced_producer_commit() {
    %0 = nvws.create_token {loadType = 3 : i32, numBuffers = 1 : i32} : tensor<1x!nvws.token>
    %c0_i32 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 0 : i32
    %false = arith.constant {async_task_id = dense<0> : vector<1xi32>} false
    nvws.producer_acquire %0, %c0_i32, %false {async_task_id = dense<0> : vector<1xi32>} : tensor<1x!nvws.token>, i32, i1
    nvws.producer_commit %0, %c0_i32 {async_task_id = dense<0> : vector<1xi32>, fenced} : tensor<1x!nvws.token>, i32
    nvws.consumer_wait %0, %c0_i32, %false {async_task_id = dense<1> : vector<1xi32>} : tensor<1x!nvws.token>, i32, i1
    nvws.consumer_release %0, %c0_i32 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x!nvws.token>, i32
    tt.return
  }
}

// -----

// Verify that a ProducerCommitOp WITHOUT the `fenced` attribute does NOT
// produce a fence_async_shared.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @unfenced_producer_commit
//   CHECK-NOT: ttng.fence_async_shared
//       CHECK: ttng.arrive_barrier
  tt.func @unfenced_producer_commit() {
    %0 = nvws.create_token {loadType = 3 : i32, numBuffers = 1 : i32} : tensor<1x!nvws.token>
    %c0_i32 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 0 : i32
    %false = arith.constant {async_task_id = dense<0> : vector<1xi32>} false
    nvws.producer_acquire %0, %c0_i32, %false {async_task_id = dense<0> : vector<1xi32>} : tensor<1x!nvws.token>, i32, i1
    nvws.producer_commit %0, %c0_i32 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x!nvws.token>, i32
    nvws.consumer_wait %0, %c0_i32, %false {async_task_id = dense<1> : vector<1xi32>} : tensor<1x!nvws.token>, i32, i1
    nvws.consumer_release %0, %c0_i32 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x!nvws.token>, i32
    tt.return
  }
}
