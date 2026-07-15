// RUN: triton-opt %s --nvgpu-warp-specialization | FileCheck %s
// XFAIL: *

// Regression test for B-17-F1 / T273495688.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @async_load_producer_commit
  // CHECK: ttng.async_copy_mbarrier_arrive
  // CHECK-NOT: nvws.producer_commit
  // CHECK: ttng.wait_barrier
  tt.func public @async_load_producer_commit() {
    %tok = nvws.create_token {loadType = 1 : i32, numBuffers = 1 : i32} : tensor<1x!nvws.token>
    %idx = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
    %phase = arith.constant {ttg.partition = array<i32: 1>} false

    nvws.producer_commit %tok, %idx {ttg.partition = array<i32: 0>} : tensor<1x!nvws.token>, i32
    nvws.consumer_wait %tok, %idx, %phase {ttg.partition = array<i32: 1>} : tensor<1x!nvws.token>, i32, i1
    tt.return
  }
}
