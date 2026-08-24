// RUN: triton-opt --split-input-file %s -triton-uplift-while-to-for | FileCheck %s

// A countable loop: the before-region is a single `cmpi slt` of a loop-carried
// induction var against a dominating bound, and the after-region increments the
// induction var by a loop-invariant step. This is the static-persistent tile
// scheduler shape (`_x < num_tiles`, `_x += num_programs`) and uplifts to
// `scf.for`.
// CHECK-LABEL: @countable
// CHECK: scf.for
// CHECK-NOT: scf.while
tt.func @countable(%lb: i32, %ub: i32, %step: i32, %ptr: !tt.ptr<i32>) {
  scf.while (%i = %lb) : (i32) -> i32 {
    %c = arith.cmpi slt, %i, %ub : i32
    scf.condition(%c) %i : i32
  } do {
  ^bb0(%i: i32):
    tt.store %ptr, %i : !tt.ptr<i32>
    %n = arith.addi %i, %step : i32
    scf.yield %n : i32
  }
  tt.return
}

// -----

// Countable loop carrying an extra payload value alongside the induction var:
// the payload becomes an `scf.for` iter_arg and the loop still uplifts.
// CHECK-LABEL: @countable_with_carry
// CHECK: scf.for
// CHECK-NOT: scf.while
tt.func @countable_with_carry(%lb: i32, %ub: i32, %step: i32, %acc0: i32) -> i32 {
  %r:2 = scf.while (%i = %lb, %acc = %acc0) : (i32, i32) -> (i32, i32) {
    %c = arith.cmpi slt, %i, %ub : i32
    scf.condition(%c) %i, %acc : i32, i32
  } do {
  ^bb0(%i: i32, %acc: i32):
    %a = arith.addi %acc, %i : i32
    %n = arith.addi %i, %step : i32
    scf.yield %n, %a : i32, i32
  }
  tt.return %r#1 : i32
}

// -----

// The AutoWS / pipelining annotations attached to the while (by tl.condition)
// are transferred onto the uplifted scf.for, so a scheduler-driven persistent
// while loop gets the same warp-specialization / pipelining as a hand-written
// for loop.
// CHECK-LABEL: @countable_carries_attrs
// CHECK-NOT: scf.while
// CHECK: scf.for
// CHECK: tt.warp_specialize
tt.func @countable_carries_attrs(%lb: i32, %ub: i32, %step: i32, %ptr: !tt.ptr<i32>) {
  scf.while (%i = %lb) : (i32) -> i32 {
    %c = arith.cmpi slt, %i, %ub : i32
    scf.condition(%c) %i : i32
  } do {
  ^bb0(%i: i32):
    tt.store %ptr, %i : !tt.ptr<i32>
    %n = arith.addi %i, %step : i32
    scf.yield %n : i32
  } attributes {tt.warp_specialize, tt.num_stages = 3 : i32, tt.data_partition_factor = 2 : i32}
  tt.return
}

// -----

// Negative: the induction var is advanced by an atomic (the dynamic
// work-stealing scheduler), not a loop-invariant `addi` step, so the loop is not
// countable and must be preserved.
// CHECK-LABEL: @atomic_advance
// CHECK: scf.while
tt.func @atomic_advance(%init: i32, %ub: i32, %counter: !tt.ptr<i32>) {
  %c1 = arith.constant 1 : i32
  scf.while (%i = %init) : (i32) -> i32 {
    %c = arith.cmpi slt, %i, %ub : i32
    scf.condition(%c) %i : i32
  } do {
  ^bb0(%i: i32):
    %n = tt.atomic_rmw add, acq_rel, gpu, %counter, %c1 : (!tt.ptr<i32>, i32) -> i32
    scf.yield %n : i32
  }
  tt.return
}

// -----

// Negative: the bound is computed inside the before-region, so it does not
// dominate the loop (and the before-region has more than one op). The uplift
// pattern does not fire. (The body stores so the loop is not dead-code
// eliminated by the greedy driver.)
// CHECK-LABEL: @bound_in_loop
// CHECK: scf.while
tt.func @bound_in_loop(%init: i32, %step: i32, %m: i32, %ptr: !tt.ptr<i32>) {
  scf.while (%i = %init) : (i32) -> i32 {
    %ub = arith.muli %m, %m : i32
    %c = arith.cmpi slt, %i, %ub : i32
    scf.condition(%c) %i : i32
  } do {
  ^bb0(%i: i32):
    tt.store %ptr, %i : !tt.ptr<i32>
    %n = arith.addi %i, %step : i32
    scf.yield %n : i32
  }
  tt.return
}

// -----

// Negative: the condition is a plain loop-carried `i1` (the CLC hardware-valid /
// constant-flip shape), not a `cmpi`, so the loop is not countable. (The body
// stores so the loop is not dead-code eliminated by the greedy driver.)
// CHECK-LABEL: @non_cmpi_condition
// CHECK: scf.while
tt.func @non_cmpi_condition(%init: i32, %valid0: i1, %ptr: !tt.ptr<i32>) {
  %false = arith.constant false
  %c1 = arith.constant 1 : i32
  scf.while (%valid = %valid0, %i = %init) : (i1, i32) -> (i1, i32) {
    scf.condition(%valid) %valid, %i : i1, i32
  } do {
  ^bb0(%v: i1, %i: i32):
    tt.store %ptr, %i : !tt.ptr<i32>
    %n = arith.addi %i, %c1 : i32
    scf.yield %false, %n : i1, i32
  }
  tt.return
}
