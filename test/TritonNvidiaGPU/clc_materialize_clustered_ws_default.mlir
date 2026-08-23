// RUN: triton-opt %s -verify-each --triton-nvidia-gpu-clc-materialize | FileCheck %s --implicit-check-not=perThread

// CHECK-LABEL: @clustered_clc_ws_default
// CHECK: %[[RESP:.*]] = ttg.local_alloc
// CHECK: scf.while
// CHECK: %[[BARRIERS:.*]] = ttg.local_alloc
// CHECK: %[[DONE:.*]] = ttg.memdesc_index %[[BARRIERS]]
// CHECK-NEXT: ttng.init_barrier %[[DONE]], 1
// CHECK: %[[READY:.*]] = ttg.memdesc_index %[[BARRIERS]]
// CHECK-NEXT: ttng.init_barrier %[[READY]], 4
// CHECK: ttg.warp_specialize
// CHECK: default {
// CHECK: ttng.barrier_expect %[[DONE]], 16
// CHECK: ttng.fence_async_shared
// CHECK: ttng.arrive_barrier
// CHECK: ttng.wait_barrier %[[READY]]
// CHECK: ttng.clc_try_cancel %[[RESP]], %[[DONE]]
// CHECK: ttng.wait_barrier %[[DONE]]
// CHECK-NEXT: ttng.clc_load_result %[[RESP]]

// CHECK-LABEL: @clustered_clc_ws_partition
// CHECK: ttg.warp_specialize
// CHECK: partition0({{.*}}, {{.*}}, %[[PHASE:.*]]: i32)
// CHECK: ttng.wait_barrier {{.*}}, %[[PHASE]]

module attributes {
  "ttg.cluster-dim-x" = 2 : i32,
  "ttg.cluster-dim-y" = 2 : i32,
  "ttg.cluster-dim-z" = 1 : i32,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  ttg.target = "cuda:100"
} {
  tt.func public @clustered_clc_ws_default() {
    %true = arith.constant true
    %result = scf.while (%valid = %true) : (i1) -> i1 {
      scf.condition(%valid) %valid : i1
    } do {
    ^bb0(%valid: i1):
      ttg.warp_specialize()
      default {
        %tok = ttng.clc_try_cancel_async : !ttg.async.token
        %next_valid, %x, %y, %z = ttng.clc_read %tok : !ttg.async.token -> i1, i32, i32, i32
        ttg.warp_yield
      }
      partition0() num_warps(1) {
        ttg.warp_return
      } : () -> ()
      scf.yield %valid : i1
    }
    tt.return
  }

  tt.func public @clustered_clc_ws_partition() {
    %true = arith.constant true
    %result = scf.while (%valid = %true) : (i1) -> i1 {
      scf.condition(%valid) %valid : i1
    } do {
    ^bb0(%valid: i1):
      ttg.warp_specialize()
      default {
        ttg.warp_yield
      }
      partition0() num_warps(1) {
        %tok = ttng.clc_try_cancel_async : !ttg.async.token
        %next_valid, %x, %y, %z = ttng.clc_read %tok : !ttg.async.token -> i1, i32, i32, i32
        ttg.warp_return
      } : () -> ()
      scf.yield %valid : i1
    }
    tt.return
  }
}
