// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-clc-materialize | FileCheck %s

// Stage 4 (clc-materialize) runs after physical AutoWS. When the CLC persistent
// scf.while has already been placed inside a ttg.warp_specialize worker
// partition AND the launch uses an explicit cluster, the clustered CLC
// allocations -- response buffer, completion barrier, cluster empty barrier --
// must be hoisted ABOVE the ttg.warp_specialize and threaded into the isolated
// worker region through the partition's explicit capture list.
//
// Two things break without the hoist:
//   1. A cluster rendezvous nested in one partition region is executed only by
//      that partition's warps, so barrier init would omit every other warp and
//      deadlock.
//   2. The allocations must outlive the specialized loop.
//
// Only the allocations move. The tile-id decode and the rest of the CLC
// sequence stay inside the loop, which is what the trailing checks pin.

// CHECK-LABEL: @clc_ws_cluster
// Response buffer, completion barrier and cluster empty barrier are all
// allocated before the warp_specialize. For each barrier the memdesc_index that
// is captured is the one AFTER init_barrier.
// CHECK:       %[[RESP:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
// CHECK:       %[[FULLA:.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
// CHECK:       ttng.init_barrier %{{.*}}, 1
// CHECK:       %[[FULL:.*]] = ttg.memdesc_index %[[FULLA]]
// CHECK:       %[[EMPTYA:.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
// The empty barrier's arrival count is the cluster size, not 1.
// CHECK:       ttng.init_barrier %{{.*}}, 2
// CHECK:       %[[EMPTY:.*]] = ttg.memdesc_index %[[EMPTYA]]
//
// All three reach the isolated region through the capture list.
// CHECK:       ttg.warp_specialize(%{{.*}}, %[[RESP]], %[[FULL]], %[[EMPTY]])
// CHECK:       partition0(%{{.*}}: i32, %[[P_RESP:.*]]: !ttg.memdesc<2xi64{{.*}}, %[[P_FULL:.*]]: !ttg.memdesc<1xi64{{.*}}, %[[P_EMPTY:.*]]: !ttg.memdesc<1xi64
//
// The CLC sequence stays inside the partition's loop and consumes the captured
// allocations rather than re-allocating in the isolated region.
// CHECK:       scf.while
// CHECK:       ttng.wait_barrier %[[P_EMPTY]]
// CHECK:       ttng.clc_try_cancel %[[P_RESP]], %[[P_FULL]]
// CHECK:       ttng.clc_load_result %[[P_RESP]]
// CHECK-NOT:   ttg.local_alloc

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @clc_ws_cluster(%ub: i32) {
    ttg.warp_specialize(%ub)
    default {
      ttg.warp_yield
    }
    partition0(%arg_ub: i32) num_warps(4) {
      // Defined inside the region: ttg.warp_specialize partitions are isolated
      // from above, which is exactly why the CLC allocations need capturing.
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %r:2 = scf.while (%i = %c0_i32, %v = %c1_i32) : (i32, i32) -> (i32, i32) {
        %cond = arith.cmpi slt, %i, %arg_ub : i32
        scf.condition(%cond) %i, %v : i32, i32
      } do {
      ^bb0(%i: i32, %v: i32):
        %tok = ttng.clc_try_cancel_async : !ttg.async.token
        %valid, %x, %y, %z = ttng.clc_read %tok : !ttg.async.token -> i1, i32, i32, i32
        %next = arith.addi %i, %c1_i32 : i32
        scf.yield %next, %x : i32, i32
      }
      ttg.warp_return
    } : (i32) -> ()
    tt.return
  }
}

// -----

// Without an explicit cluster (cluster size 1) there is nothing to rendezvous,
// so the allocations stay next to the loop inside the partition and no capture
// is added. This pins the hoist to the clustered case only.

// CHECK-LABEL: @clc_ws_no_cluster
// CHECK:       ttg.warp_specialize(%{{[^,)]*}})
// CHECK:       partition0(%{{.*}}: i32) num_warps
// CHECK:       ttg.local_alloc
// CHECK:       scf.while

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @clc_ws_no_cluster(%ub: i32) {
    ttg.warp_specialize(%ub)
    default {
      ttg.warp_yield
    }
    partition0(%arg_ub: i32) num_warps(4) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %r:2 = scf.while (%i = %c0_i32, %v = %c1_i32) : (i32, i32) -> (i32, i32) {
        %cond = arith.cmpi slt, %i, %arg_ub : i32
        scf.condition(%cond) %i, %v : i32, i32
      } do {
      ^bb0(%i: i32, %v: i32):
        %tok = ttng.clc_try_cancel_async : !ttg.async.token
        %valid, %x, %y, %z = ttng.clc_read %tok : !ttg.async.token -> i1, i32, i32, i32
        %next = arith.addi %i, %c1_i32 : i32
        scf.yield %next, %x : i32, i32
      }
      ttg.warp_return
    } : (i32) -> ()
    tt.return
  }
}
