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
  // CHECK-LABEL: @prefer_smaller_group
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK-NOT: ttg.local_alloc : () -> !ttg.memdesc<1xi64
  // CHECK: ttng.arrive_barrier
  // CHECK: arith.constant 4 : i32
  // CHECK: ttng.arrive_barrier_named
  // CHECK: ttng.wait_barrier %
  // CHECK: arith.constant 4 : i32
  // CHECK: ttng.wait_barrier_named
  tt.func public @prefer_smaller_group(%index: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c6 = arith.constant 6 : i32
    %c7 = arith.constant 7 : i32
    %c8 = arith.constant 8 : i32
    %c9 = arith.constant 9 : i32
    %c10 = arith.constant 10 : i32
    %c11 = arith.constant 11 : i32
    %c12 = arith.constant 12 : i32
    %c13 = arith.constant 13 : i32
    %c14 = arith.constant 14 : i32
    %c15 = arith.constant 15 : i32
    %u6 = ttng.user_named_barrier_id %c6 : i32
    %u7 = ttng.user_named_barrier_id %c7 : i32
    %u8 = ttng.user_named_barrier_id %c8 : i32
    %u9 = ttng.user_named_barrier_id %c9 : i32
    %u10 = ttng.user_named_barrier_id %c10 : i32
    %u11 = ttng.user_named_barrier_id %c11 : i32
    %u12 = ttng.user_named_barrier_id %c12 : i32
    %u13 = ttng.user_named_barrier_id %c13 : i32
    %u14 = ttng.user_named_barrier_id %c14 : i32
    %u15 = ttng.user_named_barrier_id %c15 : i32
    %large = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %large0 = ttg.memdesc_index %large[%c0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %large1 = ttg.memdesc_index %large[%c1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %small = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %large0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %large1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %small, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%large, %index, %small) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>, %arg1: i32, %arg2: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %bar = ttg.memdesc_index %arg0[%arg1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %arg2, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>, %arg1: i32, %arg2: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %bar = ttg.memdesc_index %arg0[%arg1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %arg2, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x1xi64, #barrier, #smem, mutable>, i32, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @prefer_deeper_loop
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: scf.for
  // CHECK: arith.constant 15 : i32
  // CHECK: ttng.arrive_barrier_named
  // CHECK: ttng.wait_barrier %
  // CHECK: scf.for
  // CHECK: arith.constant 15 : i32
  // CHECK: ttng.wait_barrier_named
  tt.func public @prefer_deeper_loop() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %c9_i32 = arith.constant 9 : i32
    %c10_i32 = arith.constant 10 : i32
    %c11_i32 = arith.constant 11 : i32
    %c12_i32 = arith.constant 12 : i32
    %c13_i32 = arith.constant 13 : i32
    %c14_i32 = arith.constant 14 : i32
    %u4 = ttng.user_named_barrier_id %c4_i32 : i32
    %u5 = ttng.user_named_barrier_id %c5_i32 : i32
    %u6 = ttng.user_named_barrier_id %c6_i32 : i32
    %u7 = ttng.user_named_barrier_id %c7_i32 : i32
    %u8 = ttng.user_named_barrier_id %c8_i32 : i32
    %u9 = ttng.user_named_barrier_id %c9_i32 : i32
    %u10 = ttng.user_named_barrier_id %c10_i32 : i32
    %u11 = ttng.user_named_barrier_id %c11_i32 : i32
    %u12 = ttng.user_named_barrier_id %c12_i32 : i32
    %u13 = ttng.user_named_barrier_id %c13_i32 : i32
    %u14 = ttng.user_named_barrier_id %c14_i32 : i32
    %shallow = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %deep = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %shallow, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %deep, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%shallow, %deep) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %arg1: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 1 : index
      %step = arith.constant 1 : index
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      scf.for %i = %lb to %ub step %step {
        ttng.arrive_barrier %arg1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %arg1: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %lb = arith.constant 0 : index
      %ub = arith.constant 1 : index
      %step = arith.constant 1 : index
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      scf.for %i = %lb to %ub step %step {
        ttng.wait_barrier %arg1, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_multibuffer
  // CHECK-NOT: ttg.local_alloc
  // CHECK: partition0
  // CHECK-DAG: arith.constant 4 : i32
  // CHECK-DAG: arith.constant 5 : i32
  // CHECK: arith.select
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK-DAG: arith.constant 4 : i32
  // CHECK-DAG: arith.constant 5 : i32
  // CHECK: arith.select
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.inval_barrier
  // CHECK-NOT: ttg.local_dealloc
  tt.func public @promote_multibuffer(%index: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %bar0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bars, %index) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>, %arg1: i32) num_warps(1) {
      %bar = ttg.memdesc_index %arg0[%arg1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>, %arg1: i32) num_warps(1) {
      %phase = arith.constant 0 : i32
      %bar = ttg.memdesc_index %arg0[%arg1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x1xi64, #barrier, #smem, mutable>, i32) -> ()
    ttng.inval_barrier %bar0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.inval_barrier %bar1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %bars : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// The other promotion path: every slot is reached by its own constant-indexed
// arrive and wait, so `getStaticSlot` resolves each use and the rewrite emits a
// direct ID per slot. `@promote_multibuffer` above only covers the dynamic
// selector, which is the branch that emits the `arith.select` chain.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_multibuffer_static_coverage
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: arith.select
  // CHECK: partition0
  // CHECK-DAG: ttng.compiler_named_barrier_id
  // CHECK-DAG: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK-DAG: ttng.compiler_named_barrier_id
  // CHECK-DAG: ttng.wait_barrier_named
  // CHECK-NOT: arith.select
  // CHECK-NOT: ttng.inval_barrier
  tt.func public @promote_multibuffer_static_coverage() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %bar0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bars) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %s0 = arith.constant 0 : i32
      %s1 = arith.constant 1 : i32
      %a0 = ttg.memdesc_index %arg0[%s0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %a1 = ttg.memdesc_index %arg0[%s1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %a0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %a1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %s0 = arith.constant 0 : i32
      %s1 = arith.constant 1 : i32
      %w0 = ttg.memdesc_index %arg0[%s0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %w1 = ttg.memdesc_index %arg0[%s1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %w0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %w1, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) -> ()
    ttng.inval_barrier %bar0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.inval_barrier %bar1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %bars : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Three slots selected by a dynamic index. The chain compares against slots 0
// and 1 and falls through to slot 2, i.e. N-1 selects for N barriers -- the
// last slot is the else-branch, not a catch-all. That is only sound because the
// index is in [0, N) by construction: it indexes the same N-slot allocation in
// the mbarrier program being rewritten, so an out-of-range value would already
// be out of bounds there. This pins the shape so the assumption stays visible.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_multibuffer_dynamic_chain
  // CHECK-NOT: ttg.local_alloc
  // CHECK: partition0
  // CHECK-COUNT-3: ttng.compiler_named_barrier_id
  // CHECK: arith.select
  // CHECK: arith.select
  // CHECK-NOT: arith.select
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK-COUNT-3: ttng.compiler_named_barrier_id
  // CHECK: arith.select
  // CHECK: arith.select
  // CHECK-NOT: arith.select
  // CHECK: ttng.wait_barrier_named
  tt.func public @promote_multibuffer_dynamic_chain(%index: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64, #barrier, #smem, mutable>
    %bar0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<3x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<3x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar2 = ttg.memdesc_index %bars[%c2] : !ttg.memdesc<3x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar2, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bars, %index) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<3x1xi64, #barrier, #smem, mutable>, %arg1: i32) num_warps(1) {
      %bar = ttg.memdesc_index %arg0[%arg1] : !ttg.memdesc<3x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<3x1xi64, #barrier, #smem, mutable>, %arg1: i32) num_warps(1) {
      %phase = arith.constant 0 : i32
      %bar = ttg.memdesc_index %arg0[%arg1] : !ttg.memdesc<3x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<3x1xi64, #barrier, #smem, mutable>, i32) -> ()
    ttng.inval_barrier %bar0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.inval_barrier %bar1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.inval_barrier %bar2 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %bars : !ttg.memdesc<3x1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_incomplete_multibuffer
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_incomplete_multibuffer() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %bar0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar0) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
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

// The participant count is derived from one arrive and one wait, so every
// arrive has to come from the same partition for that count to describe the
// group. Here slot 0 is arrived in partition0 and slot 1 in partition1, so the
// group spans two producers and must not be promoted.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 7 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_arrives_in_different_partitions
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK-NOT: ttng.arrive_barrier_named
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_arrives_in_different_partitions() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %bar0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bars) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5, 6>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %s0 = arith.constant 0 : i32
      %a0 = ttg.memdesc_index %arg0[%s0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %a0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %s1 = arith.constant 1 : i32
      %a1 = ttg.memdesc_index %arg0[%s1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %a1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition2(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %s0 = arith.constant 0 : i32
      %s1 = arith.constant 1 : i32
      %w0 = ttg.memdesc_index %arg0[%s0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %w1 = ttg.memdesc_index %arg0[%s1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %w0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %w1, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) -> ()
    ttng.inval_barrier %bar0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.inval_barrier %bar1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %bars : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
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
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
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

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// `%dead` is a barrier view captured into the specialize op but never used in
// any region. The first views sweep keeps it (the capture operand still uses
// it), capture pruning drops the operand, and only the second sweep erases
// the view itself. Without the second sweep a dead `memdesc_index` leaks.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_prunes_capture_only_view
  // CHECK-NOT: ttg.memdesc_index
  // CHECK: ttng.arrive_barrier_named
  // CHECK-NOT: ttg.memdesc_index
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttg.memdesc_index
  // CHECK-NOT: ttng.inval_barrier
  // CHECK-NOT: ttg.local_dealloc
  // CHECK: tt.return
  tt.func public @promote_prunes_capture_only_view(%index: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %bar0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %dead = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bars, %index, %dead) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>, %arg1: i32, %arg2: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %bar = ttg.memdesc_index %arg0[%arg1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>, %arg1: i32, %arg2: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %bar = ttg.memdesc_index %arg0[%arg1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x1xi64, #barrier, #smem, mutable>, i32, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    ttng.inval_barrier %bar0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.inval_barrier %bar1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %bars : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Indexing an already-indexed slot (`%d` of `%bar0`) is an unknown use: the
// pass cannot map the second index back to a slot, so the group must not be
// promoted. Otherwise identical to the static-coverage promote test, so the
// rejection pins this guard and nothing else.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_double_index
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.arrive_barrier_named
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_double_index() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %bar0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %d = ttg.memdesc_index %bar0[%c0] : !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bars) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %s0 = arith.constant 0 : i32
      %s1 = arith.constant 1 : i32
      %a0 = ttg.memdesc_index %arg0[%s0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %a1 = ttg.memdesc_index %arg0[%s1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %a0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %a1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %s0 = arith.constant 0 : i32
      %s1 = arith.constant 1 : i32
      %w0 = ttg.memdesc_index %arg0[%s0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %w1 = ttg.memdesc_index %arg0[%s1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %w0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %w1, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) -> ()
    ttng.inval_barrier %bar0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.inval_barrier %bar1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %bars : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Only slot 0 is invalidated. Partial invalidation coverage rejects the group:
// a promoted slot without its `inval_barrier` would lose the invalidation the
// mbarrier program performed.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_partial_invalidation
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.arrive_barrier_named
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_partial_invalidation() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %bar0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bars) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %s0 = arith.constant 0 : i32
      %s1 = arith.constant 1 : i32
      %a0 = ttg.memdesc_index %arg0[%s0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %a1 = ttg.memdesc_index %arg0[%s1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %a0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %a1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %s0 = arith.constant 0 : i32
      %s1 = arith.constant 1 : i32
      %w0 = ttg.memdesc_index %arg0[%s0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %w1 = ttg.memdesc_index %arg0[%s1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %w0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %w1, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) -> ()
    ttng.inval_barrier %bar0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %bars : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    tt.return
  }
}
