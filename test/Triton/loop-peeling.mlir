// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -triton-test-loop-peeling -canonicalize | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @simple_loop_i32
// CHECK: (%[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32) -> f32
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[NUB:.*]] = arith.subi %[[UB]], %[[STEP]]
// CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NUB]] step %[[STEP]]
// CHECK: scf.yield
// CHECK: %[[RANGE:.*]] = arith.subi %[[UB]], %[[LB]]
// CHECK: %[[RANGE_M1:.*]] = arith.subi %[[RANGE]], %[[ONE]]
// CHECK: %[[ITERS_M1:.*]] = arith.divsi %[[RANGE_M1]], %[[STEP]]
// CHECK: %[[DELTA:.*]] = arith.muli %[[ITERS_M1]], %[[STEP]]
// CHECK: %[[LAST_IV:.*]] = arith.addi %[[DELTA]], %[[LB]]
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[LB]], %[[UB]]
// CHECK: %[[IF:.*]] = scf.if %[[COND]]
// CHECK:   %[[DEF:.*]] = "def"(%[[LAST_IV]]) : (i32) -> f32
// CHECK:   %[[RES:.*]] = arith.addf %[[FOR]], %[[DEF]] : f32
// CHECK:   scf.yield %[[RES]] : f32
// CHECK: else
// CHECK:   scf.yield %[[FOR]] : f32
// CHECK: tt.return %[[IF]] : f32
tt.func @simple_loop_i32(%lb : i32, %ub : i32, %step : i32) -> f32 {
  %init = arith.constant 0.00e+00 : f32
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i32 {
    %a = "def"(%iv) : (i32) -> f32
    %res = arith.addf %acc, %a : f32
    scf.yield %res : f32
  } {__test_peel_epilogue}

  tt.return %loop#0 : f32
}
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @simple_loop_i32
// CHECK: (%[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32) -> f32
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[NUB:.*]] = arith.subi %[[UB]], %[[STEP]]
// CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NUB]] step %[[STEP]]
// CHECK: scf.yield
// CHECK: %[[RANGE:.*]] = arith.subi %[[UB]], %[[LB]]
// CHECK: %[[RANGE_M1:.*]] = arith.subi %[[RANGE]], %[[ONE]]
// CHECK: %[[ITERS_M1:.*]] = arith.divsi %[[RANGE_M1]], %[[STEP]]
// CHECK: %[[DELTA:.*]] = arith.muli %[[ITERS_M1]], %[[STEP]]
// CHECK: %[[LAST_IV:.*]] = arith.addi %[[DELTA]], %[[LB]]
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[LB]], %[[UB]]
// CHECK: %[[IF:.*]] = scf.if %[[COND]]
// CHECK:   %[[DEF:.*]] = "def"(%[[LAST_IV]]) : (i32) -> f32
// CHECK:   %[[RES:.*]] = arith.addf %[[FOR]], %[[DEF]] : f32
// CHECK:   scf.yield %[[RES]] : f32
// CHECK: else
// CHECK:   scf.yield %[[FOR]] : f32
// CHECK: tt.return %[[IF]] : f32
tt.func @simple_loop_i32(%lb : i32, %ub : i32, %step : i32) -> f32 {
  %init = arith.constant 0.00e+00 : f32
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i32 {
    %a = "def"(%iv) : (i32) -> f32
    %res = arith.addf %acc, %a : f32
    scf.yield %res : f32
  } {__test_peel_epilogue}

  tt.return %loop#0 : f32
}
}

// -----

// Test prologue peeling: the first iteration is extracted before the loop.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @prologue_peel_simple
// CHECK: (%[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32) -> f32
// Prologue: first iteration with iv = lb
// CHECK: %[[PDEF:.*]] = "def"(%[[LB]]) : (i32) -> f32
// CHECK: %[[PACC:.*]] = arith.addf %[[PDEF]],
// New lower bound = lb + step
// CHECK: %[[NLB:.*]] = arith.addi %[[LB]], %[[STEP]]
// Loop starts from lb + step with init arg from prologue
// CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[NLB]] to %[[UB]] step %[[STEP]] iter_args(%[[ACC:.*]] = %[[PACC]]) -> (f32)
// CHECK:   %[[DEF:.*]] = "def"(%[[IV]]) : (i32) -> f32
// CHECK:   %[[RES:.*]] = arith.addf %[[ACC]], %[[DEF]] : f32
// CHECK:   scf.yield %[[RES]] : f32
// CHECK: tt.return %[[FOR]] : f32
tt.func @prologue_peel_simple(%lb : i32, %ub : i32, %step : i32) -> f32 {
  %init = arith.constant 0.00e+00 : f32
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i32 {
    %a = "def"(%iv) : (i32) -> f32
    %res = arith.addf %acc, %a : f32
    scf.yield %res : f32
  } {__test_peel_prologue}

  tt.return %loop#0 : f32
}
}

// -----

// Test prologue peeling with multiple iter args.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @prologue_peel_multi_args
// CHECK: (%[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32) -> (f32, i32)
// Prologue ops with iv = lb
// CHECK: %[[PA:.*]] = "def"(%[[LB]]) : (i32) -> f32
// CHECK: %[[PACC:.*]] = arith.addf %[[PA]],
// New lower bound = lb + step
// CHECK: %[[NLB:.*]] = arith.addi %[[LB]], %[[STEP]]
// Loop with prologue outputs as init args (cnt init = step after canonicalize)
// CHECK: %[[FOR:.*]]:2 = scf.for %[[IV:.*]] = %[[NLB]] to %[[UB]] step %[[STEP]] iter_args(%[[ACC:.*]] = %[[PACC]], %[[CNT:.*]] = %[[STEP]]) -> (f32, i32)
// CHECK:   scf.yield
// CHECK: tt.return %[[FOR]]#0, %[[FOR]]#1 : f32, i32
tt.func @prologue_peel_multi_args(%lb : i32, %ub : i32, %step : i32) -> (f32, i32) {
  %finit = arith.constant 0.00e+00 : f32
  %iinit = arith.constant 0 : i32
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%acc = %finit, %cnt = %iinit) -> (f32, i32) : i32 {
    %a = "def"(%iv) : (i32) -> f32
    %res = arith.addf %acc, %a : f32
    %cnt_next = arith.addi %cnt, %step : i32
    scf.yield %res, %cnt_next : f32, i32
  } {__test_peel_prologue}

  tt.return %loop#0, %loop#1 : f32, i32
}
}
