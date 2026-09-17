// RUN: triton-opt %s --nvgpu-test-ping-pong-sync="capability=100 num-warp-groups=3" | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: module attributes
  // CHECK-SAME: ttng.warp_specialize_barrier_ids = array<i32: 2, 5>
  // CHECK-LABEL: @pingpong_avoids_user_and_warp_specialize_ids
  tt.func @pingpong_avoids_user_and_warp_specialize_ids(
      %ping_ptr: !tt.ptr<i32>, %pong_ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %user3 = ttng.user_named_barrier_id %c3 : i32
    %user4 = ttng.user_named_barrier_id %c4 : i32

    ttg.warp_specialize(%ping_ptr, %pong_ptr, %c0, %c1, %c2) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %iv = %lb to %ub step %step : i32 {
        // CHECK: %[[PONG_ID:.+]] = arith.constant 7 : i32
        // CHECK-NEXT: %[[PONG:.+]] = ttng.compiler_named_barrier_id %[[PONG_ID]] : i32
        // CHECK: ttng.wait_barrier_named %[[PONG]], {{.*}} : !ttng.named_barrier_id, i32
        tt.store %arg0, %iv {async_task_id = array<i32: 1>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
        scf.yield
      }
      ttg.warp_return
    }
    partition1(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %lb: i32,
               %step: i32, %ub: i32) num_warps(4) {
      scf.for %iv = %lb to %ub step %step : i32 {
        // CHECK: %[[PING_ID:.+]] = arith.constant 6 : i32
        // CHECK-NEXT: %[[PING:.+]] = ttng.compiler_named_barrier_id %[[PING_ID]] : i32
        // CHECK: ttng.wait_barrier_named %[[PING]], {{.*}} : !ttng.named_barrier_id, i32
        tt.store %arg1, %iv {async_task_id = array<i32: 2>, pingpong_first_partition_id = 1 : i32, pingpong_id = 0 : i32} : !tt.ptr<i32>
        scf.yield
      }
      ttg.warp_return
    } : (!tt.ptr<i32>, !tt.ptr<i32>, i32, i32, i32) -> ()
    tt.return
  }
}
