// RUN: triton-opt %S/ws_memory_planner_annotation.mlir --nvgpu-test-ws-memory-planner="num-buffers=2 write-decision-file=%t.decisions" --mlir-print-debuginfo --mlir-use-nameloc-as-prefix -o /dev/null
// RUN: triton-opt %S/ws_memory_planner_annotation.mlir --nvgpu-test-ws-memory-planner="read-decision-file=%t.decisions" --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s
// XFAIL: *

// Regression test for B-18-F1 / T273497319.
// A replayed decision file must preserve the absence of buffer.offset on the
// owner of a TMEM reuse group. Reusers keep buffer.offset = 0, but the owner
// remains distinguishable by not carrying the attribute.

// CHECK-LABEL: tt.func public @_attn_bwd_persist
// CHECK: %ppT = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32, buffer.offset = 0 : i32}
// CHECK: %qkT, %qkT_{{[0-9]+}} = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32}
