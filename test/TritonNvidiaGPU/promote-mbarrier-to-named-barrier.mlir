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
        %offset = arith.subi %i, %c0 : i32
        %iteration = arith.divui %offset, %c1 : i32
        %phase = arith.andi %iteration, %c1 : i32
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
  // CHECK: ttng.warp_specialize_barrier_ids = array<i32: 2, 3, 4>
  // CHECK-LABEL: @promote
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttng.init_barrier
  // CHECK: partition0
  // CHECK: %[[ARRIVE_ID:.*]] = arith.constant 5 : i32
  // CHECK: %[[ARRIVE_HANDLE:.*]] = ttng.compiler_named_barrier_id %[[ARRIVE_ID]] : i32
  // CHECK: %[[ARRIVE_COUNT:.*]] = arith.constant 64 : i32
  // CHECK: ttng.arrive_barrier_named %[[ARRIVE_HANDLE]], %[[ARRIVE_COUNT]]
  // CHECK: partition1
  // CHECK: %[[WAIT_ID:.*]] = arith.constant 5 : i32
  // CHECK: %[[WAIT_HANDLE:.*]] = ttng.compiler_named_barrier_id %[[WAIT_ID]] : i32
  // CHECK: %[[WAIT_COUNT:.*]] = arith.constant 64 : i32
  // CHECK: ttng.wait_barrier_named %[[WAIT_HANDLE]], %[[WAIT_COUNT]]
  // CHECK-NOT: ttng.inval_barrier
  // CHECK-NOT: ttg.local_dealloc
  tt.func public @promote() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, ttg.partition.types = ["default", "producer", "consumer", "epilogue"], warpGroupStartIds = array<i32: 4, 5>}
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

// TLX fixup sets triton.skip_generic_pipeline before lowering even when the
// source contains only explicit warp specialization (no TLX dialect ops).
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {triton.skip_generic_pipeline, tlx.has_warp_spec_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: module attributes
  // CHECK-NOT: ttng.warp_specialize_barrier_ids
  // CHECK-LABEL: @preserve_tlx_mbarrier
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.arrive_barrier_named
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @preserve_tlx_mbarrier() {
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

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_initially_satisfied_wait
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_initially_satisfied_wait() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 1 : i32
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
  // CHECK-LABEL: @reject_dynamic_phase
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: arith.xori
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_dynamic_phase(%phase: i1) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar, %phase) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %arg1: i1) num_warps(1) {
      %true = arith.constant true
      %phase_bit = arith.xori %arg1, %true : i1
      %phase_i32 = arith.extui %phase_bit : i1 to i32
      ttng.wait_barrier %arg0, %phase_i32 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i1) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_dynamic_phase_with_initial_wait
  // CHECK-NOT: ttg.local_alloc
  // CHECK: default
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition0
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: scf.if
  tt.func public @promote_dynamic_phase_with_initial_wait() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %result = scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %result = scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        %parity = arith.andi %iteration, %c1 : i64
        %phase_bit = arith.trunci %parity : i64 to i1
        %phase = arith.extui %phase_bit : i1 to i32
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_dynamic_phase_with_initial_skip
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_dynamic_phase_with_initial_skip() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %result = scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %true = arith.constant true
      %result = scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        %parity = arith.andi %iteration, %c1 : i64
        %phase_bit = arith.trunci %parity : i64 to i1
        %initially_satisfied = arith.xori %phase_bit, %true : i1
        %phase = arith.extui %initially_satisfied : i1 to i32
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_loop_peeled_wait
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_loop_peeled_wait() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %result = scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i64
      %has_iterations = arith.cmpi ult, %lb, %ub : index
      ttng.wait_barrier %arg0, %c0, %has_iterations : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %peeled_ub = arith.subi %ub, %step : index
      %result = scf.for %i = %lb to %peeled_ub step %step iter_args(%iteration = %c1) -> (i64) {
        %parity = arith.andi %iteration, %c1 : i64
        %phase_bit = arith.trunci %parity : i64 to i1
        %phase = arith.extui %phase_bit : i1 to i32
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_loop_peeled_subtiles
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_loop_peeled_subtiles() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %bar0 = ttg.memdesc_index %bars[%c0_i32] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1_i32] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bars) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c3 = arith.constant 3 : i64
      %result = scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        %slot0 = arith.remui %iteration, %c2 : i64
        %slot0_i32 = arith.trunci %slot0 : i64 to i32
        %selected0 = ttg.memdesc_index %bars[%slot0_i32] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        ttng.arrive_barrier %selected0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %event1 = arith.addi %iteration, %c1 : i64
        %slot1 = arith.remui %event1, %c2 : i64
        %slot1_i32 = arith.trunci %slot1 : i64 to i32
        %selected1 = ttg.memdesc_index %bars[%slot1_i32] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        ttng.arrive_barrier %selected1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %event2 = arith.addi %iteration, %c2 : i64
        %slot2 = arith.remui %event2, %c2 : i64
        %slot2_i32 = arith.trunci %slot2 : i64 to i32
        %selected2 = ttg.memdesc_index %bars[%slot2_i32] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        ttng.arrive_barrier %selected2, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c3 : i64
        scf.yield %next : i64
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0_i32_local = arith.constant 0 : i32
      %c1_i32_local = arith.constant 1 : i32
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c3 = arith.constant 3 : i64
      %has_iterations = arith.cmpi ult, %lb, %ub : index
      %selected0 = ttg.memdesc_index %arg0[%c0_i32_local] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %selected0, %c0_i32_local, %has_iterations : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %selected1 = ttg.memdesc_index %arg0[%c1_i32_local] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %selected1, %c0_i32_local, %has_iterations : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %selected2 = ttg.memdesc_index %arg0[%c0_i32_local] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %selected2, %c1_i32_local, %has_iterations : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %peeled_ub = arith.subi %ub, %step : index
      %result = scf.for %i = %lb to %peeled_ub step %step iter_args(%iteration = %c3) -> (i64) {
        %slot0 = arith.remui %iteration, %c2 : i64
        %slot0_i32 = arith.trunci %slot0 : i64 to i32
        %generation0 = arith.divui %iteration, %c2 : i64
        %parity0 = arith.andi %generation0, %c1 : i64
        %phase0 = arith.trunci %parity0 : i64 to i32
        %selected3 = ttg.memdesc_index %arg0[%slot0_i32] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        ttng.wait_barrier %selected3, %phase0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %event1 = arith.addi %iteration, %c1 : i64
        %slot1 = arith.remui %event1, %c2 : i64
        %slot1_i32 = arith.trunci %slot1 : i64 to i32
        %generation1 = arith.divui %event1, %c2 : i64
        %parity1 = arith.andi %generation1, %c1 : i64
        %phase1 = arith.trunci %parity1 : i64 to i32
        %selected4 = ttg.memdesc_index %arg0[%slot1_i32] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        ttng.wait_barrier %selected4, %phase1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %event2 = arith.addi %iteration, %c2 : i64
        %slot2 = arith.remui %event2, %c2 : i64
        %slot2_i32 = arith.trunci %slot2 : i64 to i32
        %generation2 = arith.divui %event2, %c2 : i64
        %parity2 = arith.andi %generation2, %c1 : i64
        %phase2 = arith.trunci %parity2 : i64 to i32
        %selected5 = ttg.memdesc_index %arg0[%slot2_i32] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        ttng.wait_barrier %selected5, %phase2 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c3 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_misaligned_loop_peeled_wait
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_misaligned_loop_peeled_wait() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %result = scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %has_iterations = arith.cmpi ult, %lb, %ub : index
      ttng.wait_barrier %arg0, %c0, %has_iterations : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %peeled_ub = arith.subi %ub, %step : index
      %result = scf.for %i = %lb to %peeled_ub step %step iter_args(%iteration = %c2) -> (i64) {
        %parity = arith.andi %iteration, %c1 : i64
        %phase_bit = arith.trunci %parity : i64 to i1
        %phase = arith.extui %phase_bit : i1 to i32
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_dynamic_phase_multibuffer
  // CHECK-NOT: ttg.local_alloc
  // CHECK: default
  // CHECK-DAG: ttng.compiler_named_barrier_id
  // CHECK-DAG: ttng.compiler_named_barrier_id
  // CHECK: arith.select
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition0
  // CHECK-DAG: ttng.compiler_named_barrier_id
  // CHECK-DAG: ttng.compiler_named_barrier_id
  // CHECK: arith.select
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: scf.if
  tt.func public @promote_dynamic_phase_multibuffer() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %bar0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bars) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub = arith.constant 8 : index
      %step = arith.constant 1 : index
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %result = scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0_i64) -> (i64) {
        %slot_i64 = arith.remui %iteration, %c2_i64 : i64
        %slot = arith.trunci %slot_i64 : i64 to i32
        %bar = ttg.memdesc_index %bars[%slot] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1_i64 : i64
        scf.yield %next : i64
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 8 : index
      %step = arith.constant 1 : index
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %result = scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0_i64) -> (i64) {
        %slot_i64 = arith.remui %iteration, %c2_i64 : i64
        %slot = arith.trunci %slot_i64 : i64 to i32
        %bar = ttg.memdesc_index %arg0[%slot] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %generation = arith.divui %iteration, %c2_i64 : i64
        %parity = arith.andi %generation, %c1_i64 : i64
        %phase_bit = arith.trunci %parity : i64 to i1
        %phase = arith.extui %phase_bit : i1 to i32
        ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1_i64 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_mismatched_dynamic_phase_schedule
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_mismatched_dynamic_phase_schedule() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub = arith.constant 3 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %result = scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %result = scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        %parity = arith.andi %iteration, %c1 : i64
        %phase_bit = arith.trunci %parity : i64 to i1
        %phase = arith.extui %phase_bit : i1 to i32
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
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
  // CHECK-LABEL: @promote_default_to_partition
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttng.init_barrier
  // CHECK: default
  // CHECK: %[[ARRIVE_ID:.*]] = arith.constant 4 : i32
  // CHECK: %[[ARRIVE_HANDLE:.*]] = ttng.compiler_named_barrier_id %[[ARRIVE_ID]] : i32
  // CHECK: %[[ARRIVE_COUNT:.*]] = arith.constant 160 : i32
  // CHECK: ttng.arrive_barrier_named %[[ARRIVE_HANDLE]], %[[ARRIVE_COUNT]]
  // CHECK: partition0
  // CHECK: %[[WAIT_ID:.*]] = arith.constant 4 : i32
  // CHECK: %[[WAIT_HANDLE:.*]] = ttng.compiler_named_barrier_id %[[WAIT_ID]] : i32
  // CHECK: %[[WAIT_COUNT:.*]] = arith.constant 160 : i32
  // CHECK: ttng.wait_barrier_named %[[WAIT_HANDLE]], %[[WAIT_COUNT]]
  // CHECK-NOT: ttng.inval_barrier
  // CHECK-NOT: ttg.local_dealloc
  tt.func public @promote_default_to_partition() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %bar0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bars) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttng.arrive_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %bar = ttg.memdesc_index %arg0[%phase] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>) num_warps(1) {
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

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 7 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_multiple_partitions
  // CHECK-NOT: ttg.local_alloc
  // CHECK: default
  // CHECK: arith.constant 4 : i32
  // CHECK: arith.constant 224 : i32
  // CHECK: ttng.wait_barrier_named
  // CHECK: partition0
  // CHECK: arith.constant 4 : i32
  // CHECK: arith.constant 224 : i32
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK: arith.constant 4 : i32
  // CHECK: arith.constant 224 : i32
  // CHECK: ttng.arrive_barrier_named
  tt.func public @promote_multiple_partitions() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 2 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(2) {
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
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
    ttg.warp_specialize(%bar0, %bar1) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %arg1: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.arrive_barrier %arg1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %arg1: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
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

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Under dynamic phases the waits track every arrival, so a second arrive in a
// loop with a different trip count than the wait's loop would shift the
// observed phases and must be rejected. Constant-phase waits reject the same
// shape too: a named-barrier generation accumulates arrivals toward the
// thread count, so surplus arrivals skew later generations instead of
// harmlessly advancing an unobserved phase.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_unmatched_loop_arrive_dynamic_phase
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @reject_unmatched_loop_arrive_dynamic_phase() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c0i64 = arith.constant 0 : i64
      %c1i64 = arith.constant 1 : i64
      %r0 = scf.for %i = %c0 to %c4 step %c1 iter_args(%iteration = %c0i64) -> (i64) {
        ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1i64 : i64
        scf.yield %next : i64
      }
      scf.for %j = %c0 to %c8 step %c1 {
        ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0i64 = arith.constant 0 : i64
      %c1i64 = arith.constant 1 : i64
      %r1 = scf.for %i = %c0 to %c4 step %c1 iter_args(%iteration = %c0i64) -> (i64) {
        %parity = arith.andi %iteration, %c1i64 : i64
        %phase_bit = arith.trunci %parity : i64 to i1
        %phase = arith.extui %phase_bit : i1 to i32
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1i64 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_uniform_if_guarded
  // CHECK-NOT: ttg.local_alloc
  // CHECK: default
  // CHECK: arith.constant 160 : i32
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition0
  // CHECK: arith.constant 160 : i32
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.inval_barrier
  // CHECK-NOT: ttg.local_dealloc
  tt.func public @promote_uniform_if_guarded() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %pid = tt.get_program_id x : i32
      %c512 = arith.constant 512 : i32
      %cond = arith.cmpi slt, %pid, %c512 : i32
      scf.if %cond {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
          ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
          %next = arith.addi %iteration, %c1 : i64
          scf.yield %next : i64
        }
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %pid = tt.get_program_id x : i32
      %c512 = arith.constant 512 : i32
      %cond = arith.cmpi slt, %pid, %c512 : i32
      scf.if %cond {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
          %parity = arith.andi %iteration, %c1 : i64
          %phase_bit = arith.trunci %parity : i64 to i1
          %phase = arith.extui %phase_bit : i1 to i32
          ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
          %next = arith.addi %iteration, %c1 : i64
          scf.yield %next : i64
        }
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_divergent_if_bound
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_divergent_if_bound() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %pid = tt.get_program_id x : i32
      %c512 = arith.constant 512 : i32
      %cond = arith.cmpi slt, %pid, %c512 : i32
      scf.if %cond {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
          ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
          %next = arith.addi %iteration, %c1 : i64
          scf.yield %next : i64
        }
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %pid = tt.get_program_id x : i32
      %c1024 = arith.constant 1024 : i32
      %cond = arith.cmpi slt, %pid, %c1024 : i32
      scf.if %cond {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
          %parity = arith.andi %iteration, %c1 : i64
          %phase_bit = arith.trunci %parity : i64 to i1
          %phase = arith.extui %phase_bit : i1 to i32
          ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
          %next = arith.addi %iteration, %c1 : i64
          scf.yield %next : i64
        }
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_thread_dependent_if
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_thread_dependent_if() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub = arith.constant 2 : index
      %step = arith.constant 1 : index
      %false = arith.constant false
      scf.for %i = %lb to %ub step %step iter_args(%flag = %false) -> (i1) {
        scf.if %flag {
          %lb2 = arith.constant 0 : index
          %ub2 = arith.constant 4 : index
          %step2 = arith.constant 1 : index
          %c0 = arith.constant 0 : i64
          %c1 = arith.constant 1 : i64
          scf.for %j = %lb2 to %ub2 step %step2 iter_args(%iteration = %c0) -> (i64) {
            ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
            %next = arith.addi %iteration, %c1 : i64
            scf.yield %next : i64
          }
        }
        scf.yield %flag : i1
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 2 : index
      %step = arith.constant 1 : index
      %false = arith.constant false
      scf.for %i = %lb to %ub step %step iter_args(%flag = %false) -> (i1) {
        scf.if %flag {
          %lb2 = arith.constant 0 : index
          %ub2 = arith.constant 4 : index
          %step2 = arith.constant 1 : index
          %c0 = arith.constant 0 : i64
          %c1 = arith.constant 1 : i64
          scf.for %j = %lb2 to %ub2 step %step2 iter_args(%iteration = %c0) -> (i64) {
            %parity = arith.andi %iteration, %c1 : i64
            %phase_bit = arith.trunci %parity : i64 to i1
            %phase = arith.extui %phase_bit : i1 to i32
            ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
            %next = arith.addi %iteration, %c1 : i64
            scf.yield %next : i64
          }
        }
        scf.yield %flag : i1
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_uniform_while_guarded
  // CHECK-NOT: ttg.local_alloc
  // CHECK: default
  // CHECK: arith.constant 160 : i32
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition0
  // CHECK: arith.constant 160 : i32
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.inval_barrier
  // CHECK-NOT: ttg.local_dealloc
  tt.func public @promote_uniform_while_guarded() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c4 = arith.constant 4 : i32
      scf.while (%i = %c0) : (i32) -> (i32) {
          %cond = arith.cmpi slt, %i, %c4 : i32
          scf.condition(%cond) %i : i32
      } do {
        ^bb0(%ia: i32):
          %lb = arith.constant 0 : index
          %ub = arith.constant 4 : index
          %step = arith.constant 1 : index
          %c0_64 = arith.constant 0 : i64
          %c1_64 = arith.constant 1 : i64
          scf.for %j = %lb to %ub step %step iter_args(%iteration = %c0_64) -> (i64) {
            ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
            %next = arith.addi %iteration, %c1_64 : i64
            scf.yield %next : i64
          }
          %next = arith.addi %ia, %c1 : i32
          scf.yield %next : i32
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c4 = arith.constant 4 : i32
      scf.while (%i = %c0) : (i32) -> (i32) {
          %cond = arith.cmpi slt, %i, %c4 : i32
          scf.condition(%cond) %i : i32
      } do {
        ^bb0(%ia: i32):
          %lb = arith.constant 0 : index
          %ub = arith.constant 4 : index
          %step = arith.constant 1 : index
          %c0_64 = arith.constant 0 : i64
          %c1_64 = arith.constant 1 : i64
          scf.for %j = %lb to %ub step %step iter_args(%iteration = %c0_64) -> (i64) {
            %parity = arith.andi %iteration, %c1_64 : i64
            %phase_bit = arith.trunci %parity : i64 to i1
            %phase = arith.extui %phase_bit : i1 to i32
            ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
            %next = arith.addi %iteration, %c1_64 : i64
            scf.yield %next : i64
          }
          %next = arith.addi %ia, %c1 : i32
          scf.yield %next : i32
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_divergent_while_bound
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_divergent_while_bound() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c4 = arith.constant 4 : i32
      scf.while (%i = %c0) : (i32) -> (i32) {
          %cond = arith.cmpi slt, %i, %c4 : i32
          scf.condition(%cond) %i : i32
      } do {
        ^bb0(%ia: i32):
          %lb = arith.constant 0 : index
          %ub = arith.constant 4 : index
          %step = arith.constant 1 : index
          %c0_64 = arith.constant 0 : i64
          %c1_64 = arith.constant 1 : i64
          scf.for %j = %lb to %ub step %step iter_args(%iteration = %c0_64) -> (i64) {
            ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
            %next = arith.addi %iteration, %c1_64 : i64
            scf.yield %next : i64
          }
          %next = arith.addi %ia, %c1 : i32
          scf.yield %next : i32
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c8 = arith.constant 8 : i32
      scf.while (%i = %c0) : (i32) -> (i32) {
          %cond = arith.cmpi slt, %i, %c8 : i32
          scf.condition(%cond) %i : i32
      } do {
        ^bb0(%ia: i32):
          %lb = arith.constant 0 : index
          %ub = arith.constant 4 : index
          %step = arith.constant 1 : index
          %c0_64 = arith.constant 0 : i64
          %c1_64 = arith.constant 1 : i64
          scf.for %j = %lb to %ub step %step iter_args(%iteration = %c0_64) -> (i64) {
            %parity = arith.andi %iteration, %c1_64 : i64
            %phase_bit = arith.trunci %parity : i64 to i1
            %phase = arith.extui %phase_bit : i1 to i32
            ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
            %next = arith.addi %iteration, %c1_64 : i64
            scf.yield %next : i64
          }
          %next = arith.addi %ia, %c1 : i32
          scf.yield %next : i32
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_guarded_arrive_bare_wait
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_guarded_arrive_bare_wait() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %pid = tt.get_program_id x : i32
      %c512 = arith.constant 512 : i32
      %cond = arith.cmpi slt, %pid, %c512 : i32
      scf.if %cond {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
          ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
          %next = arith.addi %iteration, %c1 : i64
          scf.yield %next : i64
        }
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        %parity = arith.andi %iteration, %c1 : i64
        %phase_bit = arith.trunci %parity : i64 to i1
        %phase = arith.extui %phase_bit : i1 to i32
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// A bare arrive does not excuse a guarded wait. The named barrier counts the
// waiting task's threads, so when the guard is false those threads never
// reach `bar.sync`, the generation's arrival count is never met, and the
// producer's later arrivals skew into it. There is no unguarded-arrive
// exemption: every wait must agree with every arrive.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_bare_arrive_guarded_wait
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: default
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: partition0
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @reject_bare_arrive_guarded_wait() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %pid = tt.get_program_id x : i32
      %c512 = arith.constant 512 : i32
      %cond = arith.cmpi slt, %pid, %c512 : i32
      scf.if %cond {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
          %parity = arith.andi %iteration, %c1 : i64
          %phase_bit = arith.trunci %parity : i64 to i1
          %phase = arith.extui %phase_bit : i1 to i32
          ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
          %next = arith.addi %iteration, %c1 : i64
          scf.yield %next : i64
        }
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Same-task arrives share one accounting entry, so a second loop-nested arrive
// would contribute uncounted arrivals and over-release the named barrier.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_multiple_looped_arrives_same_task
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @reject_multiple_looped_arrives_same_task() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// All slots of one allocation must resolve to tasks under a single
// `warp_specialize` op, even when the per-slot thread counts coincide.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_cross_ws_slots
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @reject_cross_ws_slots() {
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
    ttg.warp_specialize(%bar1) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 6, 7>}
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

// A `(iv - lb) / step` phase over a negative-step loop is not a 0-based
// counter: `iv - lb` is negative and the unsigned division wraps instead of
// tracking the iteration. The DivUI counter form requires a provably positive
// step, so this group must not be promoted. Every other check passes -- the
// arrive loop carries an exact 0/1 counter and both loops share the same
// bounds -- so the rejection pins the step check.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_negative_step_divui_phase
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: arith.divui
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.arrive_barrier_named
  // CHECK-NOT: ttng.wait_barrier_named
  tt.func public @reject_negative_step_divui_phase() {
    %c8 = arith.constant 8 : i32
    %c0 = arith.constant 0 : i32
    %cneg1 = arith.constant -1 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar, %c8, %c0, %cneg1) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %lb: i32, %ub: i32, %step: i32) num_warps(1) {
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %ctr = scf.for %i = %lb to %ub step %step iter_args(%c = %zero) -> (i32) : i32 {
        ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %n = arith.addi %c, %one : i32
        scf.yield %n : i32
      }
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %lb: i32, %ub: i32, %step: i32) num_warps(1) {
      %one = arith.constant 1 : i32
      scf.for %i = %lb to %ub step %step : i32 {
        %d = arith.subi %i, %lb : i32
        %q = arith.divui %d, %step : i32
        %phase = arith.andi %q, %one : i32
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        scf.yield
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Waits must agree with each other, not just with the arrives. Here the
// arrive is unguarded while the two waits sit under different uniform guards
// (`pid < 512` vs `pid < 256`): when only one guard holds, its task blocks in
// `bar.sync` forever because the other task's threads never arrive. The bare
// arrive must not excuse the disagreement.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_disagreeing_wait_guards
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: default
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: partition0
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @reject_disagreeing_wait_guards() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %pid = tt.get_program_id x : i32
      %c512 = arith.constant 512 : i32
      %cond = arith.cmpi slt, %pid, %c512 : i32
      scf.if %cond {
        %phase = arith.constant 0 : i32
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %pid = tt.get_program_id x : i32
      %c256 = arith.constant 256 : i32
      %cond = arith.cmpi slt, %pid, %c256 : i32
      scf.if %cond {
        %phase = arith.constant 0 : i32
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Nested whiles are proven outermost first: the inner loop's inits may
// reference the outer loop's carriers (`%ia` seeds the inner loop here), and
// those carriers are only in the proven set after the outer loop is proven.
// Proving innermost first rejects this shape.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_nested_while_guarded
  // CHECK-NOT: ttg.local_alloc
  // CHECK: default
  // CHECK: arith.constant 160 : i32
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition0
  // CHECK: arith.constant 160 : i32
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.inval_barrier
  // CHECK-NOT: ttg.local_dealloc
  tt.func public @promote_nested_while_guarded() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      scf.while (%i = %c0) : (i32) -> (i32) {
          %cond = arith.cmpi slt, %i, %c4 : i32
          scf.condition(%cond) %i : i32
      } do {
        ^bb0(%ia: i32):
          %w = scf.while (%j = %ia) : (i32) -> (i32) {
              %cond2 = arith.cmpi slt, %j, %c8 : i32
              scf.condition(%cond2) %j : i32
          } do {
            ^bb0(%ja: i32):
              ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
              %next2 = arith.addi %ja, %c1 : i32
              scf.yield %next2 : i32
          }
          %next = arith.addi %ia, %c1 : i32
          scf.yield %next : i32
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      scf.while (%i = %c0) : (i32) -> (i32) {
          %cond = arith.cmpi slt, %i, %c4 : i32
          scf.condition(%cond) %i : i32
      } do {
        ^bb0(%ia: i32):
          %w = scf.while (%j = %ia) : (i32) -> (i32) {
              %cond2 = arith.cmpi slt, %j, %c8 : i32
              scf.condition(%cond2) %j : i32
          } do {
            ^bb0(%ja: i32):
              %phase = arith.constant 0 : i32
              ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
              %next2 = arith.addi %ja, %c1 : i32
              scf.yield %next2 : i32
          }
          %next = arith.addi %ia, %c1 : i32
          scf.yield %next : i32
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Same-task arrives share one accounting entry, so a second loop-nested
// arrive is rejected even when it sits in its own loop with a different trip
// count: its arrivals would be uncounted and would over-release the named
// barrier. Constant-phase waits do not excuse the surplus.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_second_looped_arrive_same_task_constant_phase
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @reject_second_looped_arrive_same_task_constant_phase() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c4 step %c1 {
        ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      scf.for %j = %c0 to %c8 step %c1 {
        ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Every producer task must contribute a loop-nest-matching arrive for each
// wait. The default task's arrive loop runs twice as many iterations as the
// wait's loop, so its surplus arrivals would skew later named-barrier
// generations. (The init count covers both producers, so the rejection pins
// the nest check rather than the count check.)
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_unmatched_producer_task_constant_phase
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: default
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @reject_unmatched_producer_task_constant_phase() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 2 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %phase = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_guards_on_different_loop_counters
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  // The arrive side guards twice on the outer counter k (fires 4*8 = 32
  // times) while the wait side guards on two different counters i and l
  // (fires 1*8 = 8 times). All four loops share bounds 0..4, so bounds-only
  // counter equivalence wrongly promotes this; the counters must correspond
  // positionally.
  tt.func public @reject_guards_on_different_loop_counters() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub4 = arith.constant 4 : index
      %ub8 = arith.constant 8 : index
      %step = arith.constant 1 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      scf.for %k = %lb to %ub4 step %step {
        scf.for %q = %lb to %ub4 step %step {
          %outer = arith.cmpi eq, %k, %c1 : index
          scf.if %outer {
            %inner = arith.cmpi eq, %k, %c1 : index
            scf.if %inner {
              scf.for %j = %lb to %ub8 step %step iter_args(%iteration = %c0) -> (i64) {
                ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
                %next = arith.addi %iteration, %c1_i64 : i64
                scf.yield %next : i64
              }
            }
          }
        }
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub4 = arith.constant 4 : index
      %ub8 = arith.constant 8 : index
      %step = arith.constant 1 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      scf.for %i = %lb to %ub4 step %step {
        scf.for %l = %lb to %ub4 step %step {
          %outer = arith.cmpi eq, %i, %c1 : index
          scf.if %outer {
            %inner = arith.cmpi eq, %l, %c1 : index
            scf.if %inner {
              scf.for %m = %lb to %ub8 step %step iter_args(%iteration = %c0) -> (i64) {
                %parity = arith.andi %iteration, %c1_i64 : i64
                %phase_bit = arith.trunci %parity : i64 to i1
                %phase = arith.extui %phase_bit : i1 to i32
                ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
                %next = arith.addi %iteration, %c1_i64 : i64
                scf.yield %next : i64
              }
            }
          }
        }
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_triangular_guard_counter_mismatch
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  // The guards agree textually (k == 1 vs i == 1 over 0..4 loops) but the
  // arrive side nests in a triangular loop (0..k) while the wait side nests
  // in a rectangular loop (0..4): the loop-nest check rejects the mismatch.
  tt.func public @reject_triangular_guard_counter_mismatch() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      scf.for %k = %lb to %ub step %step {
        %cond = arith.cmpi eq, %k, %c1 : index
        scf.if %cond {
          scf.for %j = %lb to %k step %step iter_args(%iteration = %c0) -> (i64) {
            ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
            %next = arith.addi %iteration, %c1_i64 : i64
            scf.yield %next : i64
          }
        }
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      scf.for %i = %lb to %ub step %step {
        %cond = arith.cmpi eq, %i, %c1 : index
        scf.if %cond {
          scf.for %l = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
            %parity = arith.andi %iteration, %c1_i64 : i64
            %phase_bit = arith.trunci %parity : i64 to i1
            %phase = arith.extui %phase_bit : i1 to i32
            ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
            %next = arith.addi %iteration, %c1_i64 : i64
            scf.yield %next : i64
          }
        }
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_matching_positional_guard_counters
  // CHECK-NOT: ttg.local_alloc
  // CHECK: default
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition0
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.inval_barrier
  // CHECK-NOT: ttg.local_dealloc
  // Guards on corresponding nest positions (outer k in 0..4 vs outer i in
  // 0..4) still promote: positional counter mapping is not over-strict.
  tt.func public @promote_matching_positional_guard_counters() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub4 = arith.constant 4 : index
      %ub8 = arith.constant 8 : index
      %step = arith.constant 1 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      scf.for %k = %lb to %ub4 step %step {
        scf.for %q = %lb to %ub4 step %step {
          %cond = arith.cmpi eq, %k, %c1 : index
          scf.if %cond {
            scf.for %j = %lb to %ub8 step %step iter_args(%iteration = %c0) -> (i64) {
              ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
              %next = arith.addi %iteration, %c1_i64 : i64
              scf.yield %next : i64
            }
          }
        }
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub4 = arith.constant 4 : index
      %ub8 = arith.constant 8 : index
      %step = arith.constant 1 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      scf.for %i = %lb to %ub4 step %step {
        scf.for %l = %lb to %ub4 step %step {
          %cond = arith.cmpi eq, %i, %c1 : index
          scf.if %cond {
            scf.for %m = %lb to %ub8 step %step iter_args(%iteration = %c0) -> (i64) {
              %parity = arith.andi %iteration, %c1_i64 : i64
              %phase_bit = arith.trunci %parity : i64 to i1
              %phase = arith.extui %phase_bit : i1 to i32
              ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
              %next = arith.addi %iteration, %c1_i64 : i64
              scf.yield %next : i64
            }
          }
        }
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @promote_matching_else_guards
  // CHECK-NOT: ttg.local_alloc
  // CHECK: default
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition0
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.inval_barrier
  // CHECK-NOT: ttg.local_dealloc
  // Both sides gate on the else branch of agreeing conditions.
  tt.func public @promote_matching_else_guards() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %pid = tt.get_program_id x : i32
      %c512 = arith.constant 512 : i32
      %cond = arith.cmpi slt, %pid, %c512 : i32
      scf.if %cond {
      } else {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
          ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
          %next = arith.addi %iteration, %c1 : i64
          scf.yield %next : i64
        }
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %pid = tt.get_program_id x : i32
      %c512 = arith.constant 512 : i32
      %cond = arith.cmpi slt, %pid, %c512 : i32
      scf.if %cond {
      } else {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
          %parity = arith.andi %iteration, %c1 : i64
          %phase_bit = arith.trunci %parity : i64 to i1
          %phase = arith.extui %phase_bit : i1 to i32
          ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
          %next = arith.addi %iteration, %c1 : i64
          scf.yield %next : i64
        }
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_then_vs_else_guard_mismatch
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  // Same condition, but the arrive sits in the then branch and the wait in
  // the else branch: region numbers differ, so the guards disagree.
  tt.func public @reject_then_vs_else_guard_mismatch() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %pid = tt.get_program_id x : i32
      %c512 = arith.constant 512 : i32
      %cond = arith.cmpi slt, %pid, %c512 : i32
      scf.if %cond {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
          ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
          %next = arith.addi %iteration, %c1 : i64
          scf.yield %next : i64
        }
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %pid = tt.get_program_id x : i32
      %c512 = arith.constant 512 : i32
      %cond = arith.cmpi slt, %pid, %c512 : i32
      scf.if %cond {
      } else {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
          %parity = arith.andi %iteration, %c1 : i64
          %phase_bit = arith.trunci %parity : i64 to i1
          %phase = arith.extui %phase_bit : i1 to i32
          ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
          %next = arith.addi %iteration, %c1 : i64
          scf.yield %next : i64
        }
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_guard_predicate_mismatch
  // CHECK: ttg.local_alloc
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK-NOT: ttng.wait_barrier_named
  // Identical operands with different predicates (slt vs sle) disagree.
  tt.func public @reject_guard_predicate_mismatch() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %pid = tt.get_program_id x : i32
      %c512 = arith.constant 512 : i32
      %cond = arith.cmpi slt, %pid, %c512 : i32
      scf.if %cond {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
          ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
          %next = arith.addi %iteration, %c1 : i64
          scf.yield %next : i64
        }
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %pid = tt.get_program_id x : i32
      %c512 = arith.constant 512 : i32
      %cond = arith.cmpi sle, %pid, %c512 : i32
      scf.if %cond {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
          %parity = arith.andi %iteration, %c1 : i64
          %phase_bit = arith.trunci %parity : i64 to i1
          %phase = arith.extui %phase_bit : i1 to i32
          ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
          %next = arith.addi %iteration, %c1 : i64
          scf.yield %next : i64
        }
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Both waits observe the same completed mbarrier phase. A named-barrier wait
// participates in a new generation each time, so the second one would hang.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_two_wait_sites_same_task_constant_phase
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: partition0
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: partition1
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @reject_two_wait_sites_same_task_constant_phase() {
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
      ttng.wait_barrier %arg0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Two different phase expressions can describe the same generation in a
// pipelined loop. Matching each wait independently to the producer loop does
// not prove that the consumer participates only once per generation.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_two_wait_sites_same_task_dynamic_phase
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: default
  // CHECK: ttng.arrive_barrier {{.*}} :
  // CHECK: partition0
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: ttng.wait_barrier_named
  // CHECK: tt.return
  tt.func public @reject_two_wait_sites_same_task_dynamic_phase() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) num_warps(1) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %step = arith.constant 1 : index
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      scf.for %i = %lb to %ub step %step iter_args(%iteration = %c0) -> (i64) {
        %parity0 = arith.andi %iteration, %c1 : i64
        %phase0 = arith.trunci %parity0 : i64 to i32
        ttng.wait_barrier %arg0, %phase0 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %parity1 = arith.remui %iteration, %c2 : i64
        %phase1 = arith.trunci %parity1 : i64 to i32
        ttng.wait_barrier %arg0, %phase1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %iteration, %c1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>) -> ()
    tt.return
  }
}
