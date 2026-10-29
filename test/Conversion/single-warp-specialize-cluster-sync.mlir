// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' | FileCheck %s --check-prefix=LOWER
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --initialize-ws-cluster-barriers='compute-capability=100 ptx-version=87' --convert-warp-specialize-to-llvm | FileCheck %s --check-prefix=WS

// Single-WS lowering supports exactly one warp_specialize region. The module
// keeps it unless lowering adds a synthetic region for worker warps to join a
// cluster sync, or is logical multi-CTA (where cluster barrier insertion after
// allocation may add one). An entry bootstrap alone adds none: workers join it
// from the WS header, so they are still dispatched by warp ID with no state
// switch.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, "ttg.single-warp-specialize" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // LOWER: module attributes {{{.*}}"ttg.single-warp-specialize" = true
  // LOWER-LABEL: llvm.func @explicit_bootstrap_keeps_single_ws
  // LOWER: nvvm.cluster.arrive.relaxed aligned
  // LOWER-NEXT: nvvm.cluster.wait aligned
  // LOWER-NEXT: ttg.warp_specialize
  // LOWER-NOT: ttg.warp_specialize
  // WS-LABEL: llvm.func @explicit_bootstrap_keeps_single_ws
  // WS: @!$0 barrier.cluster.arrive.relaxed.aligned
  // WS-NOT: llvm.switch
  tt.func public @explicit_bootstrap_keeps_single_ws() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// The legacy paired-MMA bootstrap of TLX 2-CTA kernels likewise keeps single-WS.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, "ttg.single-warp-specialize" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // LOWER: module attributes {{{.*}}"ttg.single-warp-specialize" = true
  // LOWER-LABEL: llvm.func @paired_mma_bootstrap_keeps_single_ws
  // LOWER: nvvm.cluster.arrive.relaxed aligned
  // LOWER-NEXT: nvvm.cluster.wait aligned
  // LOWER-NEXT: ttg.warp_specialize
  // LOWER-NOT: ttg.warp_specialize
  // WS-LABEL: llvm.func @paired_mma_bootstrap_keeps_single_ws
  // WS: @!$0 barrier.cluster.arrive.relaxed.aligned
  // WS-NOT: llvm.switch
  tt.func public @paired_mma_bootstrap_keeps_single_ws() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #shared, #smem, mutable>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %c0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

// In a logical multi-CTA module, cluster barrier insertion runs after
// allocation and may add a sync that needs a synthetic region, so single-WS is
// dropped conservatively even though nothing ends up being inserted here.
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, "ttg.single-warp-specialize" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // LOWER: module attributes
  // LOWER-NOT: ttg.single-warp-specialize
  // LOWER-LABEL: llvm.func @multi_cta_drops_single_ws
  // WS-LABEL: llvm.func @multi_cta_drops_single_ws
  // WS: llvm.switch
  tt.func public @multi_cta_drops_single_ws() {
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// Cross-CTA mbarrier init sync insertion places a cluster barrier before the
// WS region, which workers join through a synthetic region. Dropping single-WS
// up front lets both regions dispatch through the state switch.
#barrierEnc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, "ttg.single-warp-specialize" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // LOWER: module attributes
  // LOWER-NOT: ttg.single-warp-specialize
  // LOWER-LABEL: llvm.func @inserted_init_sync_drops_single_ws
  // LOWER: fence.mbarrier_init.release.cluster
  // LOWER-NEXT: ttg.warp_specialize
  // LOWER-NEXT: default {
  // LOWER-NEXT: nvvm.cluster.arrive.relaxed aligned
  // LOWER-NEXT: nvvm.cluster.wait aligned
  // LOWER-NEXT: ttg.warp_yield
  // LOWER: partition0() num_warps(4) {
  // LOWER-NEXT: nvvm.cluster.arrive.relaxed aligned
  // LOWER-NEXT: nvvm.cluster.wait aligned
  // LOWER-NEXT: ttg.warp_return
  // LOWER: ttg.warp_specialize
  // WS-LABEL: llvm.func @inserted_init_sync_drops_single_ws
  // WS: llvm.switch
  tt.func public @inserted_init_sync_drops_single_ws(%idx: i32) {
    %c0 = arith.constant 0 : i32
    %barriers = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrierEnc, #smem, mutable>
    %barrier0 = ttg.memdesc_index %barriers[%c0] : !ttg.memdesc<2x1xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>
    ttng.init_barrier %barrier0, 1 : !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      %barrier = ttg.memdesc_index %barriers[%idx] : !ttg.memdesc<2x1xi64, #barrierEnc, #smem, mutable> -> !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>
      ttng.wait_barrier %barrier, %c0 : !ttg.memdesc<1xi64, #barrierEnc, #smem, mutable>
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// A cluster barrier inside a WS region is backed by mbarriers that are
// initialized behind an all-warp rendezvous in a synthetic region at kernel
// entry, so single-WS is dropped even without logical multi-CTA.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, "ttg.single-warp-specialize" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // LOWER: module attributes
  // LOWER-NOT: ttg.single-warp-specialize
  // LOWER-SAME: ttg.ws_cluster_barrier_count = 2
  // LOWER-NOT: ttg.single-warp-specialize
  // LOWER-LABEL: llvm.func @ws_cluster_barrier_drops_single_ws
  // WS-LABEL: llvm.func @ws_cluster_barrier_drops_single_ws
  // WS: llvm.switch
  tt.func public @ws_cluster_barrier_drops_single_ws() {
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttng.cluster_barrier
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttng.cluster_barrier
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
