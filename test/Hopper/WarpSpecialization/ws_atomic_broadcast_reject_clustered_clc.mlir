// RUN: triton-opt %s | FileCheck %s
// RUN: not triton-opt %s --nvgpu-test-ws-atomic-broadcast 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448" | FileCheck %s --check-prefix=FALLBACK --implicit-check-not=async_task_id --implicit-check-not=ttg.partition --implicit-check-not=ttg.warp_specialize

// A clustered CLC fetch must reject AutoWS even when task propagation assigns
// the read to only one partition.
// CHECK-LABEL: @reject_clustered_clc
// ERROR: error: clustered CLC AutoWS is unsupported for explicit cluster shape 2x2x1
// FALLBACK-LABEL: @reject_clustered_clc
// FALLBACK: scf.while
// FALLBACK: ttng.clc_try_cancel_async
// FALLBACK: ttng.clc_read
// FALLBACK: scf.yield
module attributes {
  "ttg.cluster-dim-x" = 2 : i32,
  "ttg.cluster-dim-y" = 2 : i32,
  "ttg.cluster-dim-z" = 1 : i32,
  "ttg.ctas-per-cga" = true,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  ttg.target = "cuda:100"
} {
  tt.func public @reject_clustered_clc() {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %r0, %r1, %r2, %r3 = scf.while (%valid = %true, %x = %c0_i32, %y = %c0_i32, %z = %c0_i32) : (i1, i32, i32, i32) -> (i1, i32, i32, i32) {
      scf.condition(%valid) %valid, %x, %y, %z : i1, i32, i32, i32
    } do {
    ^bb0(%valid: i1, %x: i32, %y: i32, %z: i32):
      %tok = ttng.clc_try_cancel_async {async_task_id = array<i32: 0>} : !ttg.async.token
      %next_valid, %next_x, %next_y, %next_z = ttng.clc_read %tok {async_task_id = array<i32: 0>} : !ttg.async.token -> i1, i32, i32, i32
      scf.yield %next_valid, %next_x, %next_y, %next_z : i1, i32, i32, i32
    } attributes {async_task_id = array<i32: 0, 1>}
    tt.return
  }
}
