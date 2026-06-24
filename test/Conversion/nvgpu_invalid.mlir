// RUN: triton-opt -split-input-file %s -verify-diagnostics

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.func @cluster_barrier_in_default_region_invalid() {
    ttg.warp_specialize()
    default {
      // expected-error @below {{cannot be used inside `ttg.warp_specialize`}}
      ttng.cluster_barrier
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    llvm.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.func @cluster_barrier_in_partition_invalid() {
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      // expected-error @below {{cannot be used inside `ttg.warp_specialize`}}
      ttng.cluster_barrier
      ttg.warp_return
    } : () -> ()
    llvm.return
  }
}

// -----

// NOTE: beta intentionally does NOT reject ttng.cluster_arrive / ttng.cluster_wait
// inside ttg.warp_specialize -- beta's autows/TLX pipelines place them there and
// ClusterArriveOpConversion/ClusterWaitOpConversion lower them with all-warps
// wrapping. Only ttng.cluster_barrier (backend-only, unused in beta) is rejected.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.func @cluster_barrier_num_ctas_invalid() {
    // expected-error @below {{requires ttg.num-ctas > 1}}
    ttng.cluster_barrier
    llvm.return
  }
}
