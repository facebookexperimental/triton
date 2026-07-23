// RUN: triton-opt %s -split-input-file | FileCheck %s
// RUN: not triton-opt %s -split-input-file --nvgpu-test-ws-atomic-broadcast

// Deterministic coverage for the `classifyAtomic` / `classifyCLC` REJECT
// branches (WSAtomicBroadcast.cpp). The tile op already carries explicit
// `async_task_id` (post task-id-propagation shape), so the lit-only
// `--nvgpu-test-ws-atomic-broadcast` pass reads the partition sets directly (via
// getAsyncTaskIds) with no PSM/propagation needed and hits exactly one reject
// guard per module. That test pass `signalPassFailure`s on reject (it does NOT
// do the orchestrator's graceful teardown -- that shared teardown path is proven
// once, on the scatter case, in ws_atomic_broadcast_reject.mlir), so the second
// RUN uses `not` to assert the pass fails on every shape here.
//
// The first RUN (round-trip + FileCheck) guarantees each module is valid IR, so
// the `not` on the second RUN reflects a genuine reject and not a parse error.

// A scalar, loop-carried atomic replicated to a STRICT SUBSET of the while's
// partitions -> reject (taskIds.size() != allParts.size()), WSAtomicBroadcast.cpp
// :114-118. Atomic is {0,1}; the while spans {0,1,2}.
// CHECK-LABEL: @reject_subset_atomic
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_subset_atomic(%counter: !tt.ptr<i32>, %n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %r = scf.while (%arg0 = %c0_i32) : (i32) -> i32 {
      %cond = arith.cmpi slt, %arg0, %n : i32
      scf.condition(%cond) %arg0 : i32
    } do {
    ^bb0(%arg0: i32):
      %next = tt.atomic_rmw add, acq_rel, gpu, %counter, %c1_i32, %true {async_task_id = array<i32: 0, 1>} : (!tt.ptr<i32>, i32, i1) -> i32
      scf.yield %next : i32
    } attributes {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----

// A scalar atomic replicated to ALL partitions, but whose result is NOT the
// loop-carried yield of the while (the while yields an unrelated value) ->
// reject (getLoopCarryingWhile == nullptr), WSAtomicBroadcast.cpp :109-113. This
// is also the "unrelated replicated atomic" shape (same code check).
// CHECK-LABEL: @reject_non_carried_atomic
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_non_carried_atomic(%counter: !tt.ptr<i32>, %n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %r = scf.while (%arg0 = %c0_i32) : (i32) -> i32 {
      %cond = arith.cmpi slt, %arg0, %n : i32
      scf.condition(%cond) %arg0 : i32
    } do {
    ^bb0(%arg0: i32):
      %next = tt.atomic_rmw add, acq_rel, gpu, %counter, %c1_i32, %true {async_task_id = array<i32: 0, 1>} : (!tt.ptr<i32>, i32, i1) -> i32
      %other = arith.addi %arg0, %c1_i32 {async_task_id = array<i32: 0, 1>} : i32
      scf.yield %other : i32
    } attributes {async_task_id = array<i32: 0, 1>}
    tt.return
  }
}

// -----

// A CLC fetch (clc_read) replicated across partitions but NOT inside a while ->
// reject (classifyCLC not-in-while), WSAtomicBroadcast.cpp :267-271.
// CHECK-LABEL: @reject_clc_not_in_while
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_clc_not_in_while() {
    %tok = ttng.clc_try_cancel_async {async_task_id = array<i32: 0, 1>} : !ttg.async.token
    %valid, %x, %y, %z = ttng.clc_read %tok {async_task_id = array<i32: 0, 1>} : !ttg.async.token -> i1, i32, i32, i32
    tt.return
  }
}

// -----

// A CLC fetch inside a while, token from clc_try_cancel_async, but replicated to
// a STRICT SUBSET of the while's partitions -> reject (classifyCLC subset),
// WSAtomicBroadcast.cpp :276-280. clc_read is {0,1}; the while spans {0,1,2}.
// CHECK-LABEL: @reject_clc_subset
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_clc_subset(%n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %r = scf.while (%arg0 = %c0_i32) : (i32) -> i32 {
      %cond = arith.cmpi slt, %arg0, %n : i32
      scf.condition(%cond) %arg0 : i32
    } do {
    ^bb0(%arg0: i32):
      %tok = ttng.clc_try_cancel_async {async_task_id = array<i32: 0, 1, 2>} : !ttg.async.token
      %valid, %x, %y, %z = ttng.clc_read %tok {async_task_id = array<i32: 0, 1>} : !ttg.async.token -> i1, i32, i32, i32
      scf.yield %arg0 : i32
    } attributes {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }
}
