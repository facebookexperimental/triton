// RUN: env TRITON_USE_META_WS=1 triton-opt %s --nvgpu-partition-scheduling-meta '--nvgpu-warp-specialization=capability=90 num-stages=3 smem-budget=200000' | FileCheck %s

// A while with no partition anchors has no task union. It must pass through
// task propagation without materializing an unknown task attribute.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @while_without_tasks
  // CHECK-NOT:   scf.while
  // CHECK:       tt.return
  tt.func public @while_without_tasks(%bound: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result = scf.while (%i = %c0) : (i32) -> i32 {
      %valid = arith.cmpi slt, %i, %bound : i32
      scf.condition(%valid) %i : i32
    } do {
    ^bb0(%i: i32):
      %next = arith.addi %i, %c1 : i32
      scf.yield %next : i32
    } attributes {tt.warp_specialize}
    tt.return
  }
}
