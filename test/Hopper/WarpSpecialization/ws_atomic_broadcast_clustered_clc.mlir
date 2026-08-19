// RUN: triton-opt %s | FileCheck %s
// RUN: triton-opt %s --nvgpu-test-ws-atomic-broadcast | FileCheck %s --check-prefix=BCAST
// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448" | FileCheck %s --check-prefix=FULL
// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448" --triton-nvidia-gpu-clc-materialize | FileCheck %s --check-prefix=MATERIALIZED

// The cluster-rank-zero CLC request is still run once per CTA partition set.
// Each CTA receives the hardware-multicast response, then its owner partition
// broadcasts the decoded tuple to the other warp partition through SMEM.
// CHECK-LABEL: @clustered_clc
// CHECK: ttng.clc_try_cancel_async
// CHECK: ttng.clc_read

// BCAST-LABEL: @clustered_clc
// BCAST-COUNT-4: ttg.local_alloc
// BCAST: ttng.clc_try_cancel_async {{.*}}async_task_id = array<i32: [[OWNER:[0-9]+]]>
// BCAST: ttng.clc_read {{.*}}async_task_id = array<i32: [[OWNER]]>
// BCAST: ttg.local_store
// BCAST-NEXT: {{.*}}ttg.local_load
// BCAST-NEXT: {{.*}}tt.unsplat
// BCAST: ttg.local_store
// BCAST-NEXT: {{.*}}ttg.local_load
// BCAST-NEXT: {{.*}}tt.unsplat
// BCAST: ttg.local_store
// BCAST-NEXT: {{.*}}ttg.local_load
// BCAST-NEXT: {{.*}}tt.unsplat
// BCAST: ttg.local_store
// BCAST-NEXT: {{.*}}ttg.local_load
// BCAST-NEXT: {{.*}}tt.unsplat
// BCAST: scf.yield
// BCAST: attributes {async_task_id = array<i32: 0, 1>}

// FULL-LABEL: @clustered_clc
// FULL: ttg.warp_specialize
// FULL-COUNT-1: ttng.clc_try_cancel_async
// FULL-COUNT-1: ttng.clc_read
// FULL: ttg.local_store
// FULL: ttg.local_load

// MATERIALIZED-LABEL: @clustered_clc
// Four CTAs times four owner warps times 32 threads.
// MATERIALIZED: ttng.init_barrier {{.*}}, 512
// MATERIALIZED: ttng.barrier_expect
// MATERIALIZED-NEXT: ttng.fence_async_shared
// MATERIALIZED: nvg.cluster_id
// MATERIALIZED: ttng.map_to_remote_buffer
// MATERIALIZED-NEXT: ttng.arrive_barrier {{.*}}perThread
// MATERIALIZED-NEXT: ttng.wait_barrier
// MATERIALIZED-NEXT: ttng.clc_try_cancel
// MATERIALIZED: ttng.wait_barrier
// MATERIALIZED-NEXT: ttng.clc_load_result
// MATERIALIZED-NEXT: ttng.clc_is_canceled
module attributes {
  "ttg.cluster-dim-x" = 2 : i32,
  "ttg.cluster-dim-y" = 2 : i32,
  "ttg.cluster-dim-z" = 1 : i32,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  ttg.target = "cuda:100"
} {
  tt.func public @clustered_clc() {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %r0, %r1, %r2, %r3 = scf.while (%valid = %true, %x = %c0_i32, %y = %c0_i32, %z = %c0_i32) : (i1, i32, i32, i32) -> (i1, i32, i32, i32) {
      scf.condition(%valid) %valid, %x, %y, %z : i1, i32, i32, i32
    } do {
    ^bb0(%valid: i1, %x: i32, %y: i32, %z: i32):
      %tok = ttng.clc_try_cancel_async {async_task_id = array<i32: 0>} : !ttg.async.token
      %next_valid, %next_x, %next_y, %next_z = ttng.clc_read %tok {async_task_id = array<i32: 0>} : !ttg.async.token -> i1, i32, i32, i32
      scf.yield %next_valid, %next_x, %next_y, %next_z : i1, i32, i32, i32
    } attributes {async_task_id = array<i32: 0, 1>}
    tt.return
  }
}
