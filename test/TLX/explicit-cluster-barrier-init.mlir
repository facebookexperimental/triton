// RUN: triton-opt -split-input-file --inline --allocate-shared-memory-nv --convert-triton-gpu-to-llvm=compute-capability=90 %s | FileCheck %s
// RUN: triton-opt -split-input-file --inline --allocate-shared-memory-nv --convert-triton-gpu-to-llvm=compute-capability=100 %s | FileCheck %s

// No remote consumers or paired MMA: the explicit request alone bootstraps
// the physical cluster while retaining logical num-ctas=1.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @explicit_physical_cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK: mbarrier.init.shared::cta.b64 [$1], 1;
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK-NOT: nvvm.cluster.
  // CHECK: fence.mbarrier_init.release.cluster
  // CHECK-NEXT: nvvm.cluster.arrive.relaxed aligned
  // CHECK-NEXT: nvvm.cluster.wait aligned
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK: llvm.return
  tt.func public @explicit_physical_cluster() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Initialization is entry-only; ordinary consumers may be nested in loops
// and conditional regions without introducing a divergent bootstrap.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @explicit_nested_consumers
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK: fence.mbarrier_init.release.cluster
  // CHECK-NEXT: nvvm.cluster.arrive.relaxed aligned
  // CHECK-NEXT: nvvm.cluster.wait aligned
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: mbarrier.arrive.shared::cta.b64
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: llvm.return
  tt.func public @explicit_nested_consumers(%pred: i1) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c1 step %c1 : i32 {
      scf.if %pred {
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      }
    }
    tt.return
  }
}

// -----

// Multiple explicit slots plus a trailing default init share one bootstrap.
// CGALayout replicates each barrier across the two logical CTAs. The ordinary
// logical-cluster exit sync is non-relaxed and is not a second bootstrap.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @explicit_logical_cluster_mixed_slots
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-COUNT-3: mbarrier.init.shared::cta.b64
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK-NOT: nvvm.cluster.
  // CHECK: fence.mbarrier_init.release.cluster
  // CHECK-NEXT: nvvm.cluster.arrive.relaxed aligned
  // CHECK-NEXT: nvvm.cluster.wait aligned
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK-NOT: nvvm.cluster.
  // CHECK: nvvm.cluster.arrive aligned
  // CHECK-NEXT: nvvm.cluster.wait aligned
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK: llvm.return
  tt.func public @explicit_logical_cluster_mixed_slots() {
    %bars = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %bar0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<3xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar0, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<3xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar1, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %bar2 = ttg.memdesc_index %bars[%c2] : !ttg.memdesc<3xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar2, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A valid explicit request with only one physical CTA needs no fence or sync.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 1 : i32} {
  // CHECK-LABEL: @explicit_single_cta
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: mbarrier.init.shared::cta.b64 [$1], 1;
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK: llvm.return
  tt.func public @explicit_single_cta() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Omitted and explicit false preserve the default local-only behavior, even
// with a physical cluster. No consumer otherwise requests a bootstrap.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @default_cluster_init
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-COUNT-2: mbarrier.init.shared::cta.b64
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK: llvm.return
  tt.func public @default_cluster_init() {
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bar0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar1, 1 {bCluster = false} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Inlining makes the helper's explicit init unconditional kernel-entry code.
// No remote consumer or paired MMA supplies an alternative bootstrap trigger.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  tt.func private @init_helper(%bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) attributes {noinline = false} {
    ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
  // CHECK-LABEL: @inlined_explicit_init
  // CHECK-NOT: llvm.call
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: mbarrier.init.shared::cta.b64 [$1], 1;
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK-NOT: nvvm.cluster.
  // CHECK: fence.mbarrier_init.release.cluster
  // CHECK-NEXT: nvvm.cluster.arrive.relaxed aligned
  // CHECK-NEXT: nvvm.cluster.wait aligned
  // CHECK-NOT: llvm.call
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK: llvm.return
  tt.func public @inlined_explicit_init() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.call @init_helper(%bar) : (!ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

// A cluster request in any position covers the whole entry-init batch,
// including default-scope inits. There are no legacy consumer triggers.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @mixed_cluster_first
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK-NOT: nvvm.cluster.
  // CHECK: fence.mbarrier_init.release.cluster
  // CHECK-NEXT: nvvm.cluster.arrive.relaxed aligned
  // CHECK-NEXT: nvvm.cluster.wait aligned
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK: llvm.return
  tt.func public @mixed_cluster_first() {
    %bar0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar0, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %bar1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %bar2 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar2, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @mixed_cluster_middle
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK-NOT: nvvm.cluster.
  // CHECK: fence.mbarrier_init.release.cluster
  // CHECK-NEXT: nvvm.cluster.arrive.relaxed aligned
  // CHECK-NEXT: nvvm.cluster.wait aligned
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK: llvm.return
  tt.func public @mixed_cluster_middle() {
    %bar0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %bar1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar1, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %bar2 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar2, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @mixed_cluster_last
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK-NOT: nvvm.cluster.
  // CHECK: fence.mbarrier_init.release.cluster
  // CHECK-NEXT: nvvm.cluster.arrive.relaxed aligned
  // CHECK-NEXT: nvvm.cluster.wait aligned
  // CHECK-NOT: fence.mbarrier_init.release.cluster
  // CHECK-NOT: nvvm.cluster.
  // CHECK-NOT: mbarrier.init.shared::cta.b64
  // CHECK: llvm.return
  tt.func public @mixed_cluster_last() {
    %bar0 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %bar1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %bar2 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar2, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
