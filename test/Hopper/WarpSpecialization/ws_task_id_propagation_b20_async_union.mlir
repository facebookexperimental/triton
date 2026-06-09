// RUN: triton-opt %s -split-input-file --nvgpu-test-taskid-propagate=num-warp-groups=2 | FileCheck %s

// Regression tests for B-20-F1 / T273501459.
//
// Existing Hopper-style `async_task_id` anchors must contribute to the global
// allTasks union used to mark assumes, loops, and loop-bound constants. The
// second case is the equivalent `ttg.partition` control shape.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @async_anchors_mark_assume_loop_and_bounds
  // CHECK:       %[[C0:.*]] = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
  // CHECK-NEXT:  %[[C1:.*]] = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
  // CHECK-NEXT:  %[[C16:.*]] = arith.constant {async_task_id = array<i32: 0, 1, 2>} 16 : i32
  // CHECK:       llvm.intr.assume {{.*}} : i1 {async_task_id = array<i32: 0, 1, 2>}
  // CHECK:       scf.for %{{.*}} = %[[C0]] to %[[C16]] step %[[C1]]  : i32 {
  // CHECK:       } {async_task_id = array<i32: 0, 1, 2>}
  tt.func public @async_anchors_mark_assume_loop_and_bounds() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %pid = tt.get_program_id x {async_task_id = array<i32: 0>} : i32
    %cmp = arith.cmpi slt, %pid, %c16 : i32
    llvm.intr.assume %cmp : i1
    scf.for %iv = %c0 to %c16 step %c1 : i32 {
      %producer = tt.get_program_id y {async_task_id = array<i32: 1>} : i32
      %consumer = tt.get_program_id z {async_task_id = array<i32: 2>} : i32
      %sum = arith.addi %producer, %consumer : i32
    }
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @partition_anchors_mark_assume_loop_and_bounds
  // CHECK:       %[[C0:.*]] = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
  // CHECK-NEXT:  %[[C1:.*]] = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
  // CHECK-NEXT:  %[[C16:.*]] = arith.constant {async_task_id = array<i32: 0, 1, 2>} 16 : i32
  // CHECK:       llvm.intr.assume {{.*}} : i1 {async_task_id = array<i32: 0, 1, 2>}
  // CHECK:       scf.for %{{.*}} = %[[C0]] to %[[C16]] step %[[C1]]  : i32 {
  // CHECK:       } {async_task_id = array<i32: 0, 1, 2>}
  tt.func public @partition_anchors_mark_assume_loop_and_bounds() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %pid = tt.get_program_id x {"ttg.partition" = array<i32: 3>} : i32
    %cmp = arith.cmpi slt, %pid, %c16 : i32
    llvm.intr.assume %cmp : i1
    scf.for %iv = %c0 to %c16 step %c1 : i32 {
      %producer = tt.get_program_id y {"ttg.partition" = array<i32: 4>} : i32
      %consumer = tt.get_program_id z {"ttg.partition" = array<i32: 5>} : i32
      %sum = arith.addi %producer, %consumer : i32
    }
    tt.return
  }
}
