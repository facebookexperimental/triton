// RUN: triton-opt -split-input-file %s -verify-diagnostics

// NOTE: beta intentionally does NOT reject ttng.cluster_arrive / ttng.cluster_wait
// / ttng.cluster_barrier inside ttg.warp_specialize -- beta's autows/TLX pipelines
// place them there (e.g. LoadMMASpecialization, WSLowerMem) and
// ClusterArriveOpConversion/ClusterWaitOpConversion/ClusterBarrierOpConversion lower
// them with all-warps wrapping (see lowerClusterSyncForAllWarps). The only remaining
// verifier constraint is that a cluster barrier requires a multi-CTA cluster.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.func @cluster_barrier_num_ctas_invalid() {
    // expected-error @below {{requires a multi-CTA cluster}}
    ttng.cluster_barrier
    llvm.return
  }
}
