// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: single_cta_cluster_barrier
  tt.func @single_cta_cluster_barrier() {
    // CHECK-NOT: rocdl.s.barrier
    // CHECK: llvm.return
    amdg.cluster_barrier_arrive
    amdg.cluster_barrier_wait
    tt.return
  }
}
// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: single_physical_cta_cluster_barrier
  tt.func @single_physical_cta_cluster_barrier() {
    // CHECK-NOT: rocdl.s.barrier
    // CHECK: llvm.return
    amdg.cluster_barrier_arrive
    amdg.cluster_barrier_wait
    tt.return
  }
}
// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 4 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: independent_cta_cluster_barrier
  tt.func @independent_cta_cluster_barrier() {
    // CHECK: rocdl.s.barrier.signal id = -3
    // CHECK: rocdl.s.barrier.wait id = -3
    amdg.cluster_barrier_arrive
    amdg.cluster_barrier_wait
    tt.return
  }
}
// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: cluster_barrier_arrive
  tt.func @cluster_barrier_arrive() {
    // CHECK: rocdl.s.barrier.signal id = -3
    amdg.cluster_barrier_arrive
    tt.return
  }
}
// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: cluster_barrier_wait
  tt.func @cluster_barrier_wait() {
    // CHECK: rocdl.s.barrier.wait id = -3
    amdg.cluster_barrier_wait
    tt.return
  }
}
