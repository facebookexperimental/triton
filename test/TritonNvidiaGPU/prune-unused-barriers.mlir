// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-prune-unused-barriers | FileCheck %s

#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Test 1: Barrier with only init (no waits) should be fully pruned.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @prune_init_only
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttng.init_barrier
  // CHECK: tt.return
  tt.func @prune_init_only() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    tt.return
  }
}

// -----

#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Test 2: Barrier with init + arrive (no waits) should be fully pruned.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @prune_init_arrive
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttng.init_barrier
  // CHECK-NOT: ttng.arrive_barrier
  // CHECK-NOT: ttng.inval_barrier
  // CHECK: tt.return
  tt.func @prune_init_arrive(%pred: i1) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.arrive_barrier %bar, 1, %pred : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    tt.return
  }
}

// -----

#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Test 3: Barrier with init + wait should NOT be pruned.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @keep_barrier_with_wait
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: ttng.wait_barrier
  // CHECK: ttng.inval_barrier
  // CHECK: tt.return
  tt.func @keep_barrier_with_wait() {
    %c0 = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    tt.return
  }
}
