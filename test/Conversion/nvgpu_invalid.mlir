// RUN: triton-opt -split-input-file %s -verify-diagnostics

// Meta Triton does not reject ttng.cluster_arrive, ttng.cluster_wait, or
// ttng.cluster_barrier inside ttg.warp_specialize. AutoWS and TLX place them
// there, and their conversions lower them with all-warps wrapping. A cluster
// barrier must still execute in a multi-CTA cluster.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.func @cluster_barrier_num_ctas_invalid() {
    // expected-error @below {{requires a multi-CTA cluster}}
    ttng.cluster_barrier
    llvm.return
  }
}
