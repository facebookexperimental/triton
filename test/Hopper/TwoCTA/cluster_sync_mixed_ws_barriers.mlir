// RUN: triton-opt -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm --verify-diagnostics %s | FileCheck %s

// Verify that 2-CTA kernels can combine entry-block cross-CTA barriers with
// local AutoWS pipeline barriers. Only entry-block barrier initialization should
// participate in the kernel-entry cluster sync; a later barrier init inside a
// ttg.warp_specialize partition is CTA-local and should lower normally.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  // CHECK-LABEL: llvm.func @entry_cluster_sync_with_ws_local_barrier
  tt.func public @entry_cluster_sync_with_ws_local_barrier() attributes {noinline = false} {
    %barriers = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    %entry_barrier = ttg.memdesc_index %barriers[%c0] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>

    // CHECK: mbarrier.init.shared::cta.b64
    // CHECK: fence.mbarrier_init.release.cluster
    // CHECK-NEXT: nvvm.cluster.arrive.relaxed {aligned}
    // CHECK-NEXT: nvvm.cluster.wait {aligned}
    // CHECK: nvvm.mapa
    ttng.init_barrier %entry_barrier, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %remote_entry_barrier = ttng.map_to_remote_buffer %entry_barrier, %c0 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %remote_entry_barrier, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>

    // CHECK: ttg.warp_specialize
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %local_barriers = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      %c0_partition = arith.constant 0 : i32
      %local_barrier = ttg.memdesc_index %local_barriers[%c0_partition] : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>

      // CHECK-NOT: nvvm.cluster.arrive
      // CHECK-NOT: nvvm.cluster.wait
      // CHECK: mbarrier.init.shared::cta.b64
      // CHECK-NOT: nvvm.cluster.arrive
      // CHECK-NOT: nvvm.cluster.wait
      // CHECK: mbarrier.arrive.shared::cta.b64
      // CHECK-NOT: nvvm.cluster.arrive
      // CHECK-NOT: nvvm.cluster.wait
      // CHECK: ttg.warp_return
      ttng.init_barrier %local_barrier, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttng.arrive_barrier %local_barrier, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_return
    } : () -> ()

    tt.return
  }
}
