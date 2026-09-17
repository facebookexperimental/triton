// RUN: triton-opt \
// RUN:   %S/Inputs/d120-rmsnorm-gemm-post-buffer-allocation.mlir \
// RUN:   -allow-unregistered-dialect \
// RUN:   --nvgpu-test-ws-memory-planner="num-buffers=3 smem-budget=232448" \
// RUN:   -mlir-print-debuginfo -mlir-use-nameloc-as-prefix | FileCheck %s

// Production-shaped D120426461 memory-planner oracle. A feeds both the RMS
// reduction and the MMA, while B feeds only the MMA. The existing heuristic
// therefore assigns A3/B2. Eight output subtiles share one three-copy staging
// ring. Search-mode expectations belong here once fixed-group import recognizes
// this real pre-planner representation.

// CHECK-LABEL: tt.func public @d120_rmsnorm_gemm
// CHECK: %a = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 0 : i32}
// CHECK: %acc = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32}
// CHECK: %a_5 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32}
// CHECK-COUNT-8: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 3 : i32, buffer.tmaStaging = 1 : i32}
