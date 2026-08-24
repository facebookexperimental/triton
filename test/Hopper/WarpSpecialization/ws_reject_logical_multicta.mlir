// RUN: TRITON_USE_META_WS=1 triton-opt %s -allow-unregistered-dialect --nvgpu-warp-specialization="capability=100 num-stages=2 smem-budget=232448" | FileCheck %s --implicit-check-not=async_task_id --implicit-check-not=ttg.partition --implicit-check-not=ttg.warp_specialize

// Logical multi-CTA kernels cannot use AutoWS. Bail out before descriptor
// conversion or code partitioning, while preserving the multicast plan for
// the ordinary non-WS lowering path.

// CHECK-LABEL: @logical_multicta
// CHECK: "planned_multicast"()
// CHECK-SAME: tt.multicast_axes = array<i32: 0>

module attributes {
  "ttg.num-ctas" = 2 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  ttg.target = "cuda:100"
} {
  tt.func public @logical_multicta() {
    "planned_multicast"() {tt.multicast_axes = array<i32: 0>, ttg.partition = array<i32: 0>} : () -> ()
    tt.return
  }
}
