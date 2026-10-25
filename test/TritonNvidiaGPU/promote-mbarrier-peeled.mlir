// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-promote-mbarrier-to-named-barrier --triton-nvidia-gpu-verify-named-barrier-counts | FileCheck %s

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peel_i32
  // CHECK-NOT: ttng.init_barrier
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK: %[[PRED:.*]] = arith.cmpi slt
  // CHECK: scf.if %[[PRED]] {
  // CHECK-NEXT: ttng.wait_barrier_named
  // CHECK-NEXT: }
  // CHECK: scf.for
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.wait_barrier {{.*}} :
  // CHECK: tt.return
  tt.func public @peel_i32(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peel_i64
  // CHECK-NOT: ttng.init_barrier
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK: %[[PRED:.*]] = arith.cmpi slt
  // CHECK: scf.if %[[PRED]] {
  // CHECK-NEXT: ttng.wait_barrier_named
  // CHECK-NEXT: }
  // CHECK: scf.for
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.wait_barrier {{.*}} :
  // CHECK: tt.return
  tt.func public @peel_i64(%lb: i64, %ub: i64) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i64, %end: i64) num_warps(1) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %step = arith.constant 3 : i64
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i64
      %upper = arith.addi %end, %c0 : i64
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i64 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i64, %end: i64) num_warps(1) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %step = arith.constant 3 : i64
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i64
      %upper = arith.addi %end, %c0 : i64
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i64
      %short = arith.subi %upper, %step : i64
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i64 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i64, i64) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peel_index
  // CHECK-NOT: ttng.init_barrier
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK: %[[PRED:.*]] = arith.cmpi slt
  // CHECK: scf.if %[[PRED]] {
  // CHECK-NEXT: ttng.wait_barrier_named
  // CHECK-NEXT: }
  // CHECK: scf.for
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.wait_barrier {{.*}} :
  // CHECK: tt.return
  tt.func public @peel_index(%lb: index, %ub: index) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: index, %end: index) num_warps(1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %step = arith.constant 3 : index
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : index
      %upper = arith.addi %end, %c0 : index
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : index {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: index, %end: index) num_warps(1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %step = arith.constant 3 : index
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : index
      %upper = arith.addi %end, %c0 : index
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : index
      %short = arith.subi %upper, %step : index
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : index {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, index, index) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peel_signed_widening
  // CHECK-NOT: ttng.init_barrier
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK: %[[PRED:.*]] = arith.cmpi slt
  // CHECK: scf.if %[[PRED]] {
  // CHECK-NEXT: ttng.wait_barrier_named
  // CHECK-NEXT: }
  // CHECK: scf.for
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.wait_barrier {{.*}} :
  // CHECK: tt.return
  tt.func public @peel_signed_widening(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %lower64 = arith.extsi %lower : i32 to i64
      %short64 = arith.extsi %short : i32 to i64
      %step64 = arith.extsi %step : i32 to i64
      %result = scf.for %i = %lower64 to %short64 step %step64 iter_args(%counter = %counter_init) -> (i64) : i64 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peel_index_cast
  // CHECK-NOT: ttng.init_barrier
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK: %[[PRED:.*]] = arith.cmpi slt
  // CHECK: scf.if %[[PRED]] {
  // CHECK-NEXT: ttng.wait_barrier_named
  // CHECK-NEXT: }
  // CHECK: scf.for
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.wait_barrier {{.*}} :
  // CHECK: tt.return
  tt.func public @peel_index_cast(%lb: index, %ub: index) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: index, %end: index) num_warps(1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %step = arith.constant 3 : index
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : index
      %upper = arith.addi %end, %c0 : index
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : index {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: index, %end: index) num_warps(1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %step = arith.constant 3 : index
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : index
      %upper = arith.addi %end, %c0 : index
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : index
      %short = arith.subi %upper, %step : index
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %lower64 = arith.index_cast %lower : index to i64
      %short64 = arith.index_cast %short : index to i64
      %step64 = arith.index_cast %step : index to i64
      %result = scf.for %i = %lower64 to %short64 step %step64 iter_args(%counter = %counter_init) -> (i64) : i64 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, index, index) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peel_empty
  // CHECK-NOT: ttng.init_barrier
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK: %[[PRED:.*]] = arith.cmpi slt
  // CHECK: scf.if %[[PRED]] {
  // CHECK-NEXT: ttng.wait_barrier_named
  // CHECK-NEXT: }
  // CHECK: scf.for
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.wait_barrier {{.*}} :
  // CHECK: tt.return
  tt.func public @peel_empty(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 0 : i32
      %upper = arith.constant 0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 0 : i32
      %upper = arith.constant 0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peel_one
  // CHECK-NOT: ttng.init_barrier
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK: %[[PRED:.*]] = arith.cmpi slt
  // CHECK: scf.if %[[PRED]] {
  // CHECK-NEXT: ttng.wait_barrier_named
  // CHECK-NEXT: }
  // CHECK: scf.for
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.wait_barrier {{.*}} :
  // CHECK: tt.return
  tt.func public @peel_one(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 0 : i32
      %upper = arith.constant 1 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 0 : i32
      %upper = arith.constant 1 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peel_nondivisible
  // CHECK-NOT: ttng.init_barrier
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK: %[[PRED:.*]] = arith.cmpi slt
  // CHECK: scf.if %[[PRED]] {
  // CHECK-NEXT: ttng.wait_barrier_named
  // CHECK-NEXT: }
  // CHECK: scf.for
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.wait_barrier {{.*}} :
  // CHECK: tt.return
  tt.func public @peel_nondivisible(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 2 : i32
      %upper = arith.constant 9 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 2 : i32
      %upper = arith.constant 9 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peel_unpredicated_nonempty
  // CHECK-NOT: ttng.init_barrier
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK: ttng.wait_barrier_named
  // CHECK: scf.for
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.wait_barrier {{.*}} :
  // CHECK: tt.return
  tt.func public @peel_unpredicated_nonempty(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 0 : i32
      %upper = arith.constant 1 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 0 : i32
      %upper = arith.constant 1 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_init_zero
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_init_zero(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 0 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_second_wait
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_second_wait(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_epilogue_wait
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_epilogue_wait(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttng.wait_barrier %b, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_phase_one
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_phase_one(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 1 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_unpredicated
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_unpredicated(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_arrival_count
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_arrival_count(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 2 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_multiple_slots
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_multiple_slots(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %bar = ttg.memdesc_index %storage[%zero] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %other = ttg.memdesc_index %storage[%one] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %other, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%bar, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_predicate
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_predicate(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi sle, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_wrong_upper
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_wrong_upper(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %c1 : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_negative_step
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_negative_step(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant -3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant -3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_offset
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_offset(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %offset = arith.addi %counter, %counter1 : i64
        %parity = arith.andi %offset, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_increment
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_increment(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 2 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_arrive_init
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_arrive_init(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 1 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 1 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_prologue_after_loop
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_prologue_after_loop(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

// The staging-buffer producer and TMA-store consumer from META WS GEMM.
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @gemm_epilogue_store
  // CHECK-NOT: ttng.init_barrier
  // CHECK: default
  // CHECK: ttg.local_store
  // CHECK: %[[COUNT:.*]] = arith.constant 160 : i32
  // CHECK: ttng.arrive_barrier_named {{.*}}, %[[COUNT]]
  // CHECK: partition0
  // CHECK: %[[PRED:.*]] = arith.cmpi slt
  // CHECK: scf.if %[[PRED]] {
  // CHECK-NEXT: ttng.wait_barrier_named
  // CHECK-NEXT: }
  // CHECK: scf.if %[[PRED]] {
  // CHECK-NEXT: ttng.async_tma_copy_local_to_global
  // CHECK: scf.for
  // CHECK: ttng.wait_barrier_named
  // CHECK-NEXT: ttng.async_tma_copy_local_to_global
  tt.func public @gemm_epilogue_store(%M: i32, %N: i32, %desc: !tt.tensordesc<128x128xf16, #shared>) {
    %root_zero = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #barrier, #smem, mutable>
    %root_slot = ttg.memdesc_index %bar[%root_zero] : !ttg.memdesc<1x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %root_slot, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %staging = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.warp_specialize(%M, %N, %bar, %staging, %desc) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
    default {
      %c127 = arith.constant 127 : i32
      %c128 = arith.constant 128 : i32
      %step = arith.constant 148 : i32
      %zero = arith.constant 0 : i64
      %one = arith.constant 1 : i64
      %tile = arith.constant dense<0.0> : tensor<128x128xf16, #blocked>
      %pid = tt.get_program_id x : i32
      %m = arith.addi %M, %c127 : i32
      %mt = arith.divsi %m, %c128 : i32
      %n = arith.addi %N, %c127 : i32
      %nt = arith.divsi %n, %c128 : i32
      %tiles = arith.muli %mt, %nt : i32
      %result = scf.for %i = %pid to %tiles step %step iter_args(%counter = %zero) -> (i64) : i32 {
        ttg.local_store %tile, %staging : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        ttng.arrive_barrier %root_slot, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %one : i64
        scf.yield %next : i64
      }
      ttg.warp_yield
    }
    partition0(%aM: i32, %aN: i32, %b: !ttg.memdesc<1x1xi64, #barrier, #smem, mutable>, %buffer: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %out: !tt.tensordesc<128x128xf16, #shared>) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c127 = arith.constant 127 : i32
      %c128 = arith.constant 128 : i32
      %step = arith.constant 148 : i32
      %one = arith.constant 1 : i64
      %pid = tt.get_program_id x : i32
      %m = arith.addi %aM, %c127 : i32
      %mt = arith.divsi %m, %c128 : i32
      %n = arith.addi %aN, %c127 : i32
      %nt = arith.divsi %n, %c128 : i32
      %tiles = arith.muli %mt, %nt : i32
      %pred = arith.cmpi slt, %pid, %tiles : i32
      %slot = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      ttng.wait_barrier %slot, %c0, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      scf.if %pred {
        %token = ttng.async_tma_copy_local_to_global %out[%c0, %c0] %buffer : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
      }
      %short = arith.subi %tiles, %step : i32
      %result = scf.for %i = %pid to %short step %step iter_args(%counter = %one) -> (i64) : i32 {
        %parity = arith.andi %counter, %one : i64
        %bit = arith.trunci %parity : i64 to i1
        %phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %slot, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %token = ttng.async_tma_copy_local_to_global %out[%c0, %c0] %buffer : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
        %next = arith.addi %counter, %one : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (i32, i32, !ttg.memdesc<1x1xi64, #barrier, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !tt.tensordesc<128x128xf16, #shared>) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_bound_underflow
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_bound_underflow(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 0 : i32
      %upper = arith.constant -2147483648 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 0 : i32
      %upper = arith.constant -2147483648 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @peel_remainder
  // CHECK-NOT: ttng.init_barrier
  // CHECK: ttng.arrive_barrier_named
  // CHECK: partition1
  // CHECK: %[[PRED:.*]] = arith.cmpi slt
  // CHECK: scf.if %[[PRED]] {
  // CHECK-NEXT: ttng.wait_barrier_named
  // CHECK-NEXT: }
  // CHECK: scf.for
  // CHECK: ttng.wait_barrier_named
  // CHECK-NOT: ttng.wait_barrier {{.*}} :
  // CHECK: tt.return
  tt.func public @peel_remainder(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %counter2 = arith.constant 2 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %counter2 = arith.constant 2 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.remui %counter, %counter2 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reject_peel_task_backedge
  // CHECK: ttng.init_barrier
  // CHECK-NOT: barrier_named
  // CHECK: ttng.arrive_barrier {{.*}}, 1 :
  // CHECK-NOT: barrier_named
  // CHECK: ttng.wait_barrier {{.*}} :
  // CHECK-NOT: barrier_named
  // CHECK: tt.return
  tt.func public @reject_peel_task_backedge(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      cf.br ^repeat
    ^repeat:
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      cf.cond_br %pred, ^repeat, ^exit
    ^exit:
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}
