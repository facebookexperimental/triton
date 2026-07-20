// RUN: triton-opt --split-input-file %s -triton-simplify-single-trip-while | FileCheck %s

// The "constant-flip" pattern: the loop-carried condition is a constant `true`
// on entry and a constant `false` after the body, and scf.condition forwards all
// before-block args unchanged. The loop provably runs once and is inlined.
// CHECK-LABEL: @single_trip
// CHECK-NOT: scf.while
// CHECK: arith.addi
// CHECK: tt.return
tt.func @single_trip(%arg0: i32) -> i32 {
  %true = arith.constant true
  %false = arith.constant false
  %c1 = arith.constant 1 : i32
  %r:2 = scf.while (%valid = %true, %acc = %arg0) : (i1, i32) -> (i1, i32) {
    scf.condition(%valid) %valid, %acc : i1, i32
  } do {
  ^bb0(%v: i1, %a: i32):
    %n = arith.addi %a, %c1 : i32
    scf.yield %false, %n : i1, i32
  }
  tt.return %r#1 : i32
}

// -----

// The canonicalized shape (as produced for the non-persistent tile scheduler):
// unused loop-carried values are dropped, so the loop forwards no args and has no
// results -- just the `i1` flag flipping true -> false. The body runs once.
// CHECK-LABEL: @single_trip_no_results
// CHECK-NOT: scf.while
// CHECK: tt.store
tt.func @single_trip_no_results(%arg0: !tt.ptr<i32>) {
  %true = arith.constant true
  %false = arith.constant false
  %c1 = arith.constant 1 : i32
  scf.while (%valid = %true) : (i1) -> () {
    scf.condition(%valid)
  } do {
    tt.store %arg0, %c1 : !tt.ptr<i32>
    scf.yield %false : i1
  }
  tt.return
}

// -----

// Negative: the init condition is not a constant `true` (it is a runtime arg),
// so the loop may run zero times and must be preserved.
// CHECK-LABEL: @nonconstant_init
// CHECK: scf.while
tt.func @nonconstant_init(%arg0: i32, %valid0: i1) -> i32 {
  %false = arith.constant false
  %c1 = arith.constant 1 : i32
  %r:2 = scf.while (%valid = %valid0, %acc = %arg0) : (i1, i32) -> (i1, i32) {
    scf.condition(%valid) %valid, %acc : i1, i32
  } do {
  ^bb0(%v: i1, %a: i32):
    %n = arith.addi %a, %c1 : i32
    scf.yield %false, %n : i1, i32
  }
  tt.return %r#1 : i32
}

// -----

// Negative: the after-region yields `true` for the condition, so the loop can
// iterate more than once and must be preserved.
// CHECK-LABEL: @yield_true
// CHECK: scf.while
tt.func @yield_true(%arg0: i32) -> i32 {
  %true = arith.constant true
  %c1 = arith.constant 1 : i32
  %r:2 = scf.while (%valid = %true, %acc = %arg0) : (i1, i32) -> (i1, i32) {
    scf.condition(%valid) %valid, %acc : i1, i32
  } do {
  ^bb0(%v: i1, %a: i32):
    %n = arith.addi %a, %c1 : i32
    scf.yield %true, %n : i1, i32
  }
  tt.return %r#1 : i32
}

// -----

// Negative: the condition value is a computed op, not a direct before-block
// argument, so the (deliberately narrow) matcher does not fire.
// CHECK-LABEL: @computed_condition
// CHECK: scf.while
tt.func @computed_condition(%arg0: i32) -> i32 {
  %true = arith.constant true
  %false = arith.constant false
  %c1 = arith.constant 1 : i32
  %r:2 = scf.while (%valid = %true, %acc = %arg0) : (i1, i32) -> (i1, i32) {
    %c = arith.andi %valid, %valid : i1
    scf.condition(%c) %valid, %acc : i1, i32
  } do {
  ^bb0(%v: i1, %a: i32):
    %n = arith.addi %a, %c1 : i32
    scf.yield %false, %n : i1, i32
  }
  tt.return %r#1 : i32
}
