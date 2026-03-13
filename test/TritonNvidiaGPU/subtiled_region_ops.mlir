// RUN: triton-opt %s | FileCheck %s

#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @subtiled_region
  // CHECK-SAME: %[[BAR:arg[0-9]+]]: !ttg.memdesc<1xi64, #shared, #smem, mutable>
  // CHECK-SAME: %[[PHASE:arg[0-9]+]]: i32
  tt.func @subtiled_region(
      %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %phase: i32) {
    // CHECK: ttng.subtiled_region
    // CHECK-SAME: barriers(%[[BAR]] : !ttg.memdesc<1xi64, #shared, #smem, mutable>)
    // CHECK-SAME: phases(%[[PHASE]] : i32)
    // CHECK-SAME: barrier_annotations = [#ttng.barrier_annotation<barrierIdx = 0, placement = after, targetOpIdx = 0, barrierOpKind = "arrive_barrier">]
    // CHECK: setup
    // CHECK: ttng.subtiled_region_yield
    // CHECK: tile
    // CHECK: ttng.subtiled_region_yield
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        phases(%phase : i32)
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 0, placement = after,
              targetOpIdx = 0, barrierOpKind = "arrive_barrier">
        ]
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // CHECK-LABEL: @subtiled_region_with_teardown
  tt.func @subtiled_region_with_teardown() {
    // CHECK: %[[R:.*]] = ttng.subtiled_region
    // CHECK-SAME: barrier_annotations = []
    // CHECK: setup
    // CHECK: ttng.subtiled_region_yield
    // CHECK: tile
    // CHECK: ttng.subtiled_region_yield
    // CHECK: teardown
    // CHECK: ttng.subtiled_region_yield
    // CHECK: -> (i32)
    %result = ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %v = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield %v : i32
      } teardown(%a: i32, %b: i32) {
        %j = arith.addi %a, %b : i32
        ttng.subtiled_region_yield %j : i32
      } -> (i32)
    tt.return
  }
}
