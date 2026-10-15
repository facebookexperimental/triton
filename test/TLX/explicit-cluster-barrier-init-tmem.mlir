// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' | FileCheck %s

// Scope alone requests a bootstrap: no remote consumers or paired-MMA flag.
// The trailing CTA init and its allocation must move before TMEM allocation.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @scope_bootstrap_before_tmem
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: ttng.tcgen5_global_alloc
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: ttng.tcgen5_global_alloc
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: ttng.tcgen5_global_alloc
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: ttng.tcgen5_global_alloc
  // CHECK: fence.mbarrier_init.release.cluster
  // CHECK-NEXT: nvvm.cluster.arrive.relaxed aligned
  // CHECK-NEXT: nvvm.cluster.wait aligned
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK: ttng.tcgen5_global_alloc
  // CHECK-SAME: tensor_memory_size = 128
  // CHECK-NOT: two_ctas = true
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK: llvm.return
  tt.func public @scope_bootstrap_before_tmem() {
    %bar0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %bar1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar1, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.tcgen5_global_alloc {tensor_memory_size = 128 : i32, two_ctas = false}
    %bar2 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar2, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Explicit cluster scope and legacy paired/remote triggers share one bootstrap.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @scope_and_legacy_bootstrap_before_paired_tmem
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: ttng.tcgen5_global_alloc
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK: fence.mbarrier_init.release.cluster
  // CHECK-NEXT: nvvm.cluster.arrive.relaxed aligned
  // CHECK-NEXT: nvvm.cluster.wait aligned
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK: ttng.tcgen5_global_alloc
  // CHECK-SAME: two_ctas = true
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: llvm.return
  tt.func public @scope_and_legacy_bootstrap_before_paired_tmem() attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    %bar0 = ttg.memdesc_index %bars[%zero] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar0, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.tcgen5_global_alloc {tensor_memory_size = 128 : i32, two_ctas = true}
    %bar1 = ttg.memdesc_index %bars[%one] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %remote = ttng.map_to_remote_buffer %bar0, %zero : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %remote, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    tt.return
  }
}
