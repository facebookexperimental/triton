// RUN: triton-opt %s -split-input-file | FileCheck %s

// Test constraints attribute on barrier ops.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @barrier_with_subtile_constraints
  // CHECK: ttng.wait_barrier
  // CHECK-SAME: constraints = {loweringMask = array<i32: 1, 0>, numBuffers = 2 : i32}
  // CHECK: ttng.arrive_barrier
  // CHECK-SAME: constraints = {loweringMask = array<i32: 0, 1>}
  tt.func @barrier_with_subtile_constraints(
      %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %phase: i32) {
    ttng.wait_barrier %bar, %phase {constraints = {loweringMask = array<i32: 1, 0>, numBuffers = 2 : i32}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 1 {constraints = {loweringMask = array<i32: 0, 1>}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @barrier_with_ws_constraints
  // CHECK: ttng.wait_barrier
  // CHECK-SAME: constraints = {WSBarrier = {dstTask = 1 : i32}}
  // CHECK: ttng.arrive_barrier
  // CHECK-SAME: constraints = {WSBarrier = {channelGraph = array<i32: 0, 3>, dstTask = 0 : i32}}
  tt.func @barrier_with_ws_constraints(
      %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %phase: i32) {
    ttng.wait_barrier %bar, %phase {constraints = {WSBarrier = {dstTask = 1 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 1 {constraints = {WSBarrier = {channelGraph = array<i32: 0, 3>, dstTask = 0 : i32}}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
