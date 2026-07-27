// RUN: triton-opt --split-input-file %s -triton-simplify-single-trip-while | FileCheck %s
// RUN: triton-opt --split-input-file %s -triton-simplify-single-trip-while | FileCheck %s --check-prefix=NO-EARLY
// RUN: triton-opt --split-input-file %s -triton-simplify-single-trip-while | FileCheck %s --check-prefix=ONE-LEVEL

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

// -----

// An annotated scheduler while forwards AutoWS to the for loop exactly one
// loop-nesting level inside it, even with non-loop control flow outside the
// while and between it and the for. A nested second-level for does not receive
// the attributes, and explicit inner settings win over the outer loop's
// defaults.
// CHECK-LABEL: @forwards_autows
// CHECK-NOT: scf.while
// CHECK: scf.if
// CHECK: scf.for
// CHECK: } {tt.data_partition_factor = 2 : i32
// CHECK-SAME: tt.disallow_acc_multi_buffer
// CHECK-SAME: tt.mem_plan_pick = 3 : i32
// CHECK-SAME: tt.merge_correction = true
// CHECK-SAME: tt.merge_epilogue = true
// CHECK-SAME: tt.merge_epilogue_to_computation = true
// CHECK-SAME: tt.multi_cta
// CHECK-SAME: tt.num_stages = 5 : i32
// CHECK-SAME: tt.separate_epilogue_store = true
// CHECK-SAME: tt.smem_alloc_algo = 1 : i32
// CHECK-SAME: tt.smem_budget = 65536 : i32
// CHECK-SAME: tt.smem_circular_reuse = true
// CHECK-SAME: tt.tmem_alloc_algo = 2 : i32
// CHECK-SAME: tt.warp_specialize
// NO-EARLY-LABEL: @forwards_autows
// NO-EARLY-NOT: llvm.loop_annotation
// NO-EARLY-NOT: tt.flatten
// NO-EARLY-NOT: tt.list_schedule_pick
// NO-EARLY-NOT: tt.loop_unroll_factor
// NO-EARLY: tt.return
// ONE-LEVEL-LABEL: @forwards_autows
// ONE-LEVEL: tt.warp_specialize
// ONE-LEVEL-NOT: tt.warp_specialize
// ONE-LEVEL: tt.return
tt.func @forwards_autows(%arg0: i32) -> i32 {
  %true = arith.constant true
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.if %true -> (i32) {
    %while_result = scf.while (%valid = %true, %acc = %arg0) : (i1, i32) -> i32 {
      scf.condition(%valid) %acc : i32
    } do {
    ^bb0(%acc: i32):
      %inner = scf.if %true -> (i32) {
        %loop = scf.for %i = %c0 to %c1 step %c1 iter_args(%v = %acc) -> (i32) {
          %nested = scf.for %j = %c0 to %c1 step %c1 iter_args(%nv = %v) -> (i32) {
            %nested_next = arith.addi %nv, %nv : i32
            scf.yield %nested_next : i32
          }
          %next = arith.addi %nested, %nested : i32
          scf.yield %next : i32
        } {tt.num_stages = 5 : i32}
        scf.yield %loop : i32
      } else {
        scf.yield %acc : i32
      }
      scf.yield %false, %inner : i1, i32
    } attributes {tt.warp_specialize, tt.num_stages = 3 : i32, tt.loop_unroll_factor = 4 : i32, tt.disallow_acc_multi_buffer, tt.flatten, tt.multi_cta, llvm.loop_annotation = true, tt.data_partition_factor = 2 : i32, tt.list_schedule_pick = 1 : i32, tt.mem_plan_pick = 3 : i32, tt.merge_epilogue = true, tt.merge_epilogue_to_computation = true, tt.merge_correction = true, tt.separate_epilogue_store = true, tt.tmem_alloc_algo = 2 : i32, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 65536 : i32, tt.smem_circular_reuse = true}
    scf.yield %while_result : i32
  } else {
    scf.yield %arg0 : i32
  }
  tt.return %r : i32
}

// -----

// Do not silently discard a WS request when no first-level inner loop exists.
// CHECK-LABEL: @annotated_without_forward_target
// CHECK: scf.while
// CHECK: } attributes {tt.warp_specialize}
tt.func @annotated_without_forward_target(%arg0: !tt.ptr<i32>) {
  %true = arith.constant true
  %false = arith.constant false
  %c1 = arith.constant 1 : i32
  scf.while (%valid = %true) : (i1) -> () {
    scf.condition(%valid)
  } do {
    tt.store %arg0, %c1 : !tt.ptr<i32>
    scf.yield %false : i1
  } attributes {tt.warp_specialize}
  tt.return
}

// -----

// Matching nested whiles are processed from the inside out, so erasing the
// inner loop cannot leave a stale handle in the pass worklist.
// CHECK-LABEL: @nested_single_trip
// CHECK-NOT: scf.while
// CHECK-COUNT-2: arith.addi
tt.func @nested_single_trip(%arg0: i32) -> i32 {
  %true = arith.constant true
  %false = arith.constant false
  %outer = scf.while (%valid = %true, %acc = %arg0) : (i1, i32) -> i32 {
    scf.condition(%valid) %acc : i32
  } do {
  ^bb0(%acc: i32):
    %inner = scf.while (%inner_valid = %true, %inner_acc = %acc) : (i1, i32) -> i32 {
      scf.condition(%inner_valid) %inner_acc : i32
    } do {
    ^bb0(%inner_acc: i32):
      %next = arith.addi %inner_acc, %inner_acc : i32
      scf.yield %false, %next : i1, i32
    }
    %next = arith.addi %inner, %inner : i32
    scf.yield %false, %next : i1, i32
  }
  tt.return %outer : i32
}
