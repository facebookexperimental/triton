// RUN: triton-opt %s --nvgpu-test-taskid-propagate=num-warp-groups=2 | FileCheck %s
// XFAIL: *

// Regression test for B-9-F1 / T273481168.
//
// `getAsyncTaskIds` is documented to return sorted task IDs. The test task-id
// propagation pass normalizes pre-existing task annotations through that helper
// before propagation. A non-canonical spelling with a repeated, non-adjacent
// task ID should therefore become the sorted unique set `{0, 1}`.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @async_task_id_attribute_is_canonicalized
  // CHECK:       tt.get_program_id x {async_task_id = array<i32: 0, 1>} : i32
  // CHECK-NOT:   async_task_id = array<i32: 0, 1, 1>
  tt.func public @async_task_id_attribute_is_canonicalized() {
    %pid = tt.get_program_id x {async_task_id = array<i32: 1, 0, 1>} : i32
    tt.return
  }
}
