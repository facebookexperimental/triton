// RUN: triton-opt %s --nvgpu-warp-specialization | FileCheck %s
// XFAIL: *

// Regression test for B-17-F2 / T273495687.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tma_store_wait_two_nvws_tokens
  // CHECK: ttg.local_alloc
  // CHECK: %[[EMPTY0:.*]] = ttg.local_alloc
  // CHECK: ttg.local_alloc
  // CHECK: %[[EMPTY1:.*]] = ttg.local_alloc
  // CHECK: %[[BAR0:.*]] = ttg.memdesc_index %[[EMPTY0]][%{{.*}}]
  // CHECK: %[[BAR1:.*]] = ttg.memdesc_index %[[EMPTY1]][%{{.*}}]
  // CHECK: ttng.async_tma_store_token_wait %arg0, %[[BAR0]][%{{.*}}], %[[BAR1]][%{{.*}}]
  // CHECK-NOT: nvws_token
  tt.func public @tma_store_wait_two_nvws_tokens(%store_tok: !ttg.async.token) {
    %tok0 = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %tok1 = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %idx0 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
    %idx1 = arith.constant {ttg.partition = array<i32: 1>} 1 : i32

    "ttng.async_tma_store_token_wait"(%store_tok, %tok0, %tok1, %idx0, %idx1)
        <{operandSegmentSizes = array<i32: 1, 0, 0, 2, 2>, ttg.partition = array<i32: 1>}>
        : (!ttg.async.token, tensor<2x!nvws.token>, tensor<2x!nvws.token>, i32, i32) -> ()
    tt.return
  }
}
