// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' | FileCheck %s --check-prefix=LOWER
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --convert-warp-specialize-to-llvm > %t
// RUN: FileCheck %s --check-prefix=HEADER < %t
// RUN: FileCheck %s --check-prefix=ARRIVES < %t
// RUN: FileCheck %s --check-prefix=WAITS < %t

// Bootstrap alone uses the worker-header arrival. All three ordinary sync ops
// after the user WS region must still dispatch workers. The single-WS fast path
// must be disabled before allocation because lowering creates more WS regions.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, "ttg.single-warp-specialize" = true, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // LOWER: module attributes {
  // LOWER-NOT: ttg.single-warp-specialize
  // LOWER-SAME: tlx.cluster_sync_kernel_init = true
  // LOWER-NOT: ttg.single-warp-specialize
  // LOWER-LABEL: llvm.func @explicit_bootstrap_then_ordinary_syncs
  // LOWER: mbarrier.init.shared::cta.b64
  // LOWER: nvvm.cluster.arrive.relaxed aligned
  // LOWER-NEXT: nvvm.cluster.wait aligned
  // LOWER: ttg.warp_specialize
  // LOWER: ttg.warp_return
  // LOWER: ttg.warp_specialize
  // LOWER: default {
  // LOWER-NEXT: nvvm.cluster.arrive.relaxed aligned
  // LOWER-NEXT: nvvm.cluster.wait aligned
  // LOWER-NEXT: ttg.warp_yield
  // LOWER: partition0() num_warps(4) {
  // LOWER-NEXT: nvvm.cluster.arrive.relaxed aligned
  // LOWER-NEXT: nvvm.cluster.wait aligned
  // LOWER-NEXT: ttg.warp_return
  // LOWER: ttg.warp_specialize
  // LOWER: default {
  // LOWER-NEXT: nvvm.cluster.arrive.relaxed aligned
  // LOWER-NEXT: ttg.warp_yield
  // LOWER: partition0() num_warps(4) {
  // LOWER-NEXT: nvvm.cluster.arrive.relaxed aligned
  // LOWER-NEXT: ttg.warp_return
  // LOWER: ttg.warp_specialize
  // LOWER: default {
  // LOWER-NEXT: nvvm.cluster.wait aligned
  // LOWER-NEXT: ttg.warp_yield
  // LOWER: partition0() num_warps(4) {
  // LOWER-NEXT: nvvm.cluster.wait aligned
  // LOWER-NEXT: ttg.warp_return
  // HEADER-LABEL: llvm.func @explicit_bootstrap_then_ordinary_syncs
  // HEADER: @!$0 barrier.cluster.arrive.relaxed.aligned
  // HEADER-NEXT: llvm.cond_br
  // HEADER-NOT: @!$0 barrier.cluster.arrive
  // ARRIVES-LABEL: llvm.func @explicit_bootstrap_then_ordinary_syncs
  // ARRIVES-COUNT-5: nvvm.cluster.arrive.relaxed aligned
  // ARRIVES-NOT: nvvm.cluster.arrive
  // WAITS-LABEL: llvm.func @explicit_bootstrap_then_ordinary_syncs
  // WAITS-COUNT-5: nvvm.cluster.wait aligned
  // WAITS-NOT: nvvm.cluster.wait
  tt.func public @explicit_bootstrap_then_ordinary_syncs() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    ttng.cluster_barrier {relaxed = true}
    ttng.cluster_arrive {relaxed = true}
    ttng.cluster_wait
    tt.return
  }
}

// -----

// The legacy remote-consumer trigger must mark its bootstrap too. Narrowing
// the exemption must not dispatch workers twice for that bootstrap.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // LOWER-LABEL: llvm.func @legacy_bootstrap_then_ordinary_sync
  // LOWER: mbarrier.init.shared::cta.b64
  // LOWER: nvvm.cluster.arrive.relaxed aligned
  // LOWER-NEXT: nvvm.cluster.wait aligned
  // LOWER: nvvm.mapa
  // LOWER: ttg.warp_specialize
  // LOWER: ttg.warp_return
  // LOWER: ttg.warp_specialize
  // LOWER: default {
  // LOWER-NEXT: nvvm.cluster.arrive.relaxed aligned
  // LOWER-NEXT: nvvm.cluster.wait aligned
  // LOWER-NEXT: ttg.warp_yield
  // LOWER: partition0() num_warps(4) {
  // LOWER-NEXT: nvvm.cluster.arrive.relaxed aligned
  // LOWER-NEXT: nvvm.cluster.wait aligned
  // LOWER-NEXT: ttg.warp_return
  // HEADER-LABEL: llvm.func @legacy_bootstrap_then_ordinary_sync
  // HEADER: @!$0 barrier.cluster.arrive.relaxed.aligned
  // HEADER-NEXT: llvm.cond_br
  // HEADER-NOT: @!$0 barrier.cluster.arrive
  // ARRIVES-LABEL: llvm.func @legacy_bootstrap_then_ordinary_sync
  // ARRIVES-COUNT-3: nvvm.cluster.arrive.relaxed aligned
  // ARRIVES-NOT: nvvm.cluster.arrive
  // WAITS-LABEL: llvm.func @legacy_bootstrap_then_ordinary_sync
  // WAITS-COUNT-3: nvvm.cluster.wait aligned
  // WAITS-NOT: nvvm.cluster.wait
  tt.func public @legacy_bootstrap_then_ordinary_sync() {
    %zero = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %remote = ttng.map_to_remote_buffer %bar, %zero : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %remote, 1 : !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    ttng.cluster_barrier {relaxed = true}
    tt.return
  }
}
