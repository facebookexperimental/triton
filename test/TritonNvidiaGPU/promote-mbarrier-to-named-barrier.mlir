// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-promote-mbarrier-to-named-barrier | FileCheck %s

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_uniform_loops
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttng.init_barrier
  // CHECK: partition0
  // CHECK: scf.for
  // CHECK: %[[ARRIVE_COUNT:.*]] = arith.constant 64 : i32
  // CHECK: ttng.arrive_barrier_named {{.*}}, %[[ARRIVE_COUNT]]
  // CHECK: partition1
  // CHECK: scf.for
  // CHECK: %[[WAIT_COUNT:.*]] = arith.constant 64 : i32
  // CHECK: ttng.wait_barrier_named {{.*}}, %[[WAIT_COUNT]]
  tt.func public @promote_uniform_loops(%n: i32) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar, %n) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %upper = arith.addi %end, %c1 : i32
      scf.for %i = %c0 to %upper step %c1 : i32 {
        ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %upper = arith.addi %end, %c1 : i32
      scf.for %i = %c0 to %upper step %c1 : i32 {
        %phase = arith.andi %i, %c1 : i32
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_nonuniform_arrive_loop
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttg.warp_id
  // CHECK: scf.for
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  tt.func public @reject_nonuniform_arrive_loop() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 6>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(2) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %warp = ttg.warp_id
      %upper = arith.addi %warp, %c1 : i32
      scf.for %i = %c0 to %upper step %c1 : i32 {
        ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(2) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_nonuniform_wait_loop
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK: partition1
  // CHECK: ttg.warp_id
  // CHECK: scf.for
  // CHECK: ttng.wait_barrier {{.*}} :
  tt.func public @reject_nonuniform_wait_loop() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 6>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(2) {
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(2) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %warp = ttg.warp_id
      %upper = arith.addi %warp, %c1 : i32
      scf.for %i = %c0 to %upper step %c1 : i32 {
        ttng.wait_barrier %arg0, %c0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttng.init_barrier
  // CHECK: partition0
  // CHECK: %[[ARRIVE_ID:.*]] = arith.constant 4 : i32
  // CHECK: %[[ARRIVE_HANDLE:.*]] = ttng.compiler_named_barrier_id %[[ARRIVE_ID]] : i32
  // CHECK: %[[ARRIVE_COUNT:.*]] = arith.constant 64 : i32
  // CHECK: ttng.arrive_barrier_named %[[ARRIVE_HANDLE]], %[[ARRIVE_COUNT]]
  // CHECK: partition1
  // CHECK: %[[WAIT_ID:.*]] = arith.constant 4 : i32
  // CHECK: %[[WAIT_HANDLE:.*]] = ttng.compiler_named_barrier_id %[[WAIT_ID]] : i32
  // CHECK: %[[WAIT_COUNT:.*]] = arith.constant 64 : i32
  // CHECK: ttng.wait_barrier_named %[[WAIT_HANDLE]], %[[WAIT_COUNT]]
  // CHECK-NOT: ttng.inval_barrier
  // CHECK-NOT: ttg.local_dealloc
  tt.func public @promote() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %bar : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_expected_bytes
  // CHECK: ttg.local_alloc
  // CHECK: ttng.barrier_expect
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_expected_bytes(%pred: i1) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.barrier_expect %bar, 128, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_same_partition
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_same_partition() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_per_thread
  // CHECK: ttng.arrive_barrier {{.*}} {perThread}
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_per_thread() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 32 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      ttng.arrive_barrier %arg0, 1 {perThread} : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_cross_cta_broadcast
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK: ttng.inval_barrier
  // CHECK: ttg.local_dealloc
  tt.func public @reject_cross_cta_broadcast() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.fence_mbarrier_init_release_cluster
    ttng.cluster_barrier {relaxed = true}
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %bar : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_repeated_wait
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK: ttng.inval_barrier
  // CHECK: ttg.local_dealloc
  tt.func public @reject_repeated_wait() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      scf.for %i = %c0 to %c2 step %c1 : i32 {
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %bar : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_mismatched_loop_bounds
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK: ttng.inval_barrier
  // CHECK: ttg.local_dealloc
  tt.func public @reject_mismatched_loop_bounds() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : i32
      %ub = arith.constant 1 : i32
      %step = arith.constant 1 : i32
      scf.for %i = %lb to %ub step %step : i32 {
        ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      scf.for %i = %c0 to %c2 step %c1 : i32 {
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %bar : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_balanced_count
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttng.init_barrier
  // CHECK: partition0
  // CHECK: %[[COUNT:.*]] = arith.constant 64 : i32
  // CHECK: ttng.arrive_barrier_named {{.*}}, %[[COUNT]]
  // CHECK: partition1
  // CHECK: %[[COUNT:.*]] = arith.constant 64 : i32
  // CHECK: ttng.wait_barrier_named {{.*}}, %[[COUNT]]
  tt.func public @promote_balanced_count() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 2 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      ttng.arrive_barrier %arg0, 2 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_mismatched_count
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}}
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @reject_mismatched_count() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 2 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @skip_dynamic_user_id
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}}
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @skip_dynamic_user_id(%id: i32) {
    %user = ttng.user_named_barrier_id %id : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @skip_exhausted_id_pool
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}}
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @skip_exhausted_id_pool() {
    %c3 = arith.constant 3 : i32
    %user3 = ttng.user_named_barrier_id %c3 : i32
    %c4 = arith.constant 4 : i32
    %user4 = ttng.user_named_barrier_id %c4 : i32
    %c5 = arith.constant 5 : i32
    %user5 = ttng.user_named_barrier_id %c5 : i32
    %c6 = arith.constant 6 : i32
    %user6 = ttng.user_named_barrier_id %c6 : i32
    %c7 = arith.constant 7 : i32
    %user7 = ttng.user_named_barrier_id %c7 : i32
    %c8 = arith.constant 8 : i32
    %user8 = ttng.user_named_barrier_id %c8 : i32
    %c9 = arith.constant 9 : i32
    %user9 = ttng.user_named_barrier_id %c9 : i32
    %c10 = arith.constant 10 : i32
    %user10 = ttng.user_named_barrier_id %c10 : i32
    %c11 = arith.constant 11 : i32
    %user11 = ttng.user_named_barrier_id %c11 : i32
    %c12 = arith.constant 12 : i32
    %user12 = ttng.user_named_barrier_id %c12 : i32
    %c13 = arith.constant 13 : i32
    %user13 = ttng.user_named_barrier_id %c13 : i32
    %c14 = arith.constant 14 : i32
    %user14 = ttng.user_named_barrier_id %c14 : i32
    %c15 = arith.constant 15 : i32
    %user15 = ttng.user_named_barrier_id %c15 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @reject_cluster_memory
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}}
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @reject_cluster_memory() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %cta = arith.constant 0 : i32
      %remote = ttng.map_to_remote_buffer %arg0, %cta : !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #ttng.shared_cluster_memory, mutable>
      ttng.arrive_barrier %remote, 1 : !ttg.memdesc<1xi64, #barrier, #ttng.shared_cluster_memory, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier_array = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_multicast
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}} {ctaMask = 1 : i32}
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @reject_multicast() {
    %c0 = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1x2xi64, #barrier_array, #smem, mutable>
    %slot = ttg.memdesc_index %bar[%c0] : !ttg.memdesc<1x2xi64, #barrier_array, #smem, mutable> -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    ttng.init_barrier %slot, 1 : !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%slot) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2xi64, #barrier, #smem, mutable>) num_warps(1) {
      ttng.arrive_barrier %arg0, 1 {ctaMask = 1 : i32} : !ttg.memdesc<2xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<2xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<2xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}
