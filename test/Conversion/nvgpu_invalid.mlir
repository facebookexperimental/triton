// RUN: triton-opt -split-input-file %s -verify-diagnostics

// Meta Triton permits cluster synchronization inside warp-specialized regions. TLX
// and automatic warp specialization rely on this behavior.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.func @cluster_barrier_num_ctas_invalid() {
    // expected-error @below {{requires more than one CTA per cluster}}
    ttng.cluster_barrier
    llvm.return
  }
}
