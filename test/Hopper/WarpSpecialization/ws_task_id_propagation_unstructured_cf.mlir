// RUN: triton-opt %s -split-input-file --nvgpu-test-taskid-propagate=num-warp-groups=2 | FileCheck %s

// Regression: partition-id backward propagation must handle UNSTRUCTURED control
// flow (`cf.cond_br` / `cf.br`) instead of aborting. Before the fix,
// visitBranchOperand asserted the branch was an scf.{if,for,while} and otherwise
// hit llvm_unreachable("Unknown branch operation"), crashing
// NVGPUWarpSpecialization on any kernel whose control flow lowers to cf ops.
// The common source is an early-exit `return` / bounds guard at the top of the
// kernel (both the HSTU ragged self-attention forward and the cross-attention
// backward reduce_dq crash on exactly this); a loop transform such as
// tritongpu-fuse-nested-loops can also produce it. With the fix the branch
// condition receives the union of the partition ids flowing into the successor blocks
// and the pass completes.

// Case 1: cf.cond_br whose successors take forwarded operands used by different
// partition-anchored ops -> the condition gets the union of both partition ids.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @unstructured_cond_br_task_id
  // CHECK: cf.cond_br %{{.*}}, ^bb1(%{{.*}} : i32), ^bb2(%{{.*}} : i32) {ttg.partition = array<i32: 0, 1>}
  tt.func public @unstructured_cond_br_task_id(%arg0: i32, %cond: i1) {
    cf.cond_br %cond, ^bb1(%arg0 : i32), ^bb2(%arg0 : i32)
  ^bb1(%x: i32):
    %a = arith.addi %x, %x {ttg.partition = array<i32: 0>} : i32
    cf.br ^bb3
  ^bb2(%y: i32):
    %b = arith.muli %y, %y {ttg.partition = array<i32: 1>} : i32
    cf.br ^bb3
  ^bb3:
    tt.return
  }
}

// -----

// Case 2: the real kernel pattern -- an early-exit guard `cf.cond_br` whose
// successors take NO forwarded operands (empty union). The handler must not
// crash on the empty union and must leave the branch in place; the guard
// condition simply carries no partition id (a scalar control op replicates across
// partitions). This is the shape both autoWS kernels actually hit.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @early_return_guard_task_id
  // CHECK: cf.cond_br
  // CHECK: arith.addi %{{.*}} {ttg.partition = array<i32: 0>}
  tt.func public @early_return_guard_task_id(%n: i32, %cond: i1) {
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:  // early exit
    tt.return
  ^bb2:  // body
    %a = arith.addi %n, %n {ttg.partition = array<i32: 0>} : i32
    tt.return
  }
}
