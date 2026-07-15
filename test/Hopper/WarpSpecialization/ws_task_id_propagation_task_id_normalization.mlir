// RUN: triton-opt %s --nvgpu-test-taskid-propagate=num-warp-groups=2 | FileCheck %s

// Regression test for B-9-F1 / T273481168.
//
// `getWSPartitionIds` is documented to return sorted partition IDs. The test partition-id
// propagation pass normalizes pre-existing task annotations through that helper
// before propagation. A non-zero-based partition set should therefore be
// remapped to the canonical contiguous set `{0, 1}`.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @ttg.partition_attribute_is_canonicalized
  // CHECK:       tt.get_program_id x {ttg.partition = array<i32: 0, 1>} : i32
  // CHECK-NOT:   ttg.partition = array<i32: 0, 1, 1>
  tt.func public @ttg.partition_attribute_is_canonicalized() {
    %pid = tt.get_program_id x {ttg.partition = array<i32: 1, 2>} : i32
    tt.return
  }
}
