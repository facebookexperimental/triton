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
  // CHECK: nvws.tma_store_wait %arg0, %[[BAR0]][%{{.*}}], %[[BAR1]][%{{.*}}]
  // CHECK-NOT: nvws_token
  tt.func public @tma_store_wait_two_nvws_tokens(
      %src: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>) {
    %tok0 = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %tok1 = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %idx0 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
    %idx1 = arith.constant {async_task_id = array<i32: 1>} 1 : i32

    "nvws.tma_store_wait"(%src, %tok0, %tok1, %idx0, %idx1)
        <{operandSegmentSizes = array<i32: 1, 0, 0, 2, 2>, async_task_id = array<i32: 1>}>
        : (!ttg.memdesc<128x64xf16, #shared, #smem, mutable>, tensor<2x!nvws.token>, tensor<2x!nvws.token>, i32, i32) -> ()
    tt.return
  }
}
