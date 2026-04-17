// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-lower-subtiled-region | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // Test basic lowering: two tiles, no barriers.
  // CHECK-LABEL: @basic_two_tiles
  tt.func @basic_two_tiles() {
    // Setup ops should be inlined:
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[C1:.*]] = arith.constant 1 : i32
    // Tile 0 (arg0 = c0):
    // CHECK: arith.index_cast %[[C0]]
    // Tile 1 (arg0 = c1):
    // CHECK: arith.index_cast %[[C1]]
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test lowering with arrive_barrier AFTER last tile.
  // CHECK-LABEL: @arrive_after_last
  tt.func @arrive_after_last(
      %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %phase: i32,
      %desc: !tt.tensordesc<tensor<128x128xf32, #blocked>>,
      %row: i32) {
    // Tile 0:
    // CHECK: arith.addi
    // CHECK-NOT: ttng.arrive_barrier
    // Tile 1 (last):
    // CHECK: arith.addi
    // arrive_barrier emitted AFTER last tile's op at index 0:
    // CHECK-NEXT: ttng.arrive_barrier %{{.*}}, 1
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>)
        phases(%phase : i32)
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 0, placement = after,
              targetOpIdx = 0, barrierOpKind = "arrive_barrier">
        ]
      setup {
        %c0 = arith.constant 0 : i32
        %c128 = arith.constant 128 : i32
        ttng.subtiled_region_yield %c0, %c128 : i32, i32
      } tile(%arg0: i32) {
        %off = arith.addi %arg0, %row : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test lowering with wait_barrier BEFORE first tile.
  // CHECK-LABEL: @wait_before_first
  tt.func @wait_before_first(
      %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %phase: i32) {
    // wait_barrier emitted BEFORE first tile's op at index 0:
    // CHECK: ttng.wait_barrier %{{.*}}, %{{.*}}
    // CHECK-NEXT: arith.addi
    // Tile 1: no wait_barrier
    // CHECK: arith.addi
    // CHECK-NOT: ttng.wait_barrier
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>)
        phases(%phase : i32)
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 0, placement = before,
              targetOpIdx = 0, barrierOpKind = "wait_barrier">
        ]
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test with multiple block args per tile.
  // CHECK-LABEL: @multi_arg_tiles
  tt.func @multi_arg_tiles() {
    // Setup outputs: c0, c1, c10, c20
    // Tile 0 maps [0, 2] => (c0, c10)
    // Tile 1 maps [1, 3] => (c1, c20)
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[C10:.*]] = arith.constant 10 : i32
    // CHECK-DAG: %[[C20:.*]] = arith.constant 20 : i32
    // Tile 0: addi c0, c10
    // CHECK: arith.addi %[[C0]], %[[C10]]
    // Tile 1: addi c1, c20
    // CHECK: arith.addi %[[C1]], %[[C20]]
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0, 2>, array<i32: 1, 3>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %c10 = arith.constant 10 : i32
        %c20 = arith.constant 20 : i32
        ttng.subtiled_region_yield %c0, %c1, %c10, %c20 : i32, i32, i32, i32
      } tile(%a: i32, %b: i32) {
        %sum = arith.addi %a, %b : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test with both wait_barrier BEFORE and arrive_barrier AFTER.
  // CHECK-LABEL: @wait_and_arrive
  tt.func @wait_and_arrive(
      %bar_wait: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %bar_arrive: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %phase: i32) {
    // wait_barrier BEFORE first tile's op at index 0:
    // CHECK: ttng.wait_barrier %{{.*}}, %{{.*}}
    // CHECK-NEXT: arith.muli
    // Tile 1:
    // CHECK: arith.muli
    // arrive_barrier AFTER last tile's op at index 0:
    // CHECK-NEXT: ttng.arrive_barrier %{{.*}}, 2
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        barriers(%bar_wait, %bar_arrive : !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>)
        phases(%phase, %phase : i32, i32)
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 0, placement = before,
              targetOpIdx = 0, barrierOpKind = "wait_barrier">,
          #ttng.barrier_annotation<barrierIdx = 1, placement = after,
              targetOpIdx = 0, barrierOpKind = "arrive_barrier",
              count = 2>
        ]
      setup {
        %c3 = arith.constant 3 : i32
        %c5 = arith.constant 5 : i32
        ttng.subtiled_region_yield %c3, %c5 : i32, i32
      } tile(%arg0: i32) {
        %res = arith.muli %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test with a single tile (degenerate case).
  // CHECK-LABEL: @single_tile
  tt.func @single_tile(
      %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %phase: i32) {
    // Both BEFORE and AFTER fire on the same (only) tile:
    // CHECK: ttng.wait_barrier
    // CHECK-NEXT: arith.addi
    // CHECK-NEXT: ttng.arrive_barrier
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>)
        phases(%phase : i32)
        tile_mappings = [array<i32: 0>]
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 0, placement = before,
              targetOpIdx = 0, barrierOpKind = "wait_barrier">,
          #ttng.barrier_annotation<barrierIdx = 0, placement = after,
              targetOpIdx = 0, barrierOpKind = "arrive_barrier">
        ]
      setup {
        %c42 = arith.constant 42 : i32
        ttng.subtiled_region_yield %c42 : i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test capturing values from the outer scope.
  // CHECK-LABEL: @capture_outer_value
  // CHECK-SAME: %[[OUTER:arg0]]: i32
  tt.func @capture_outer_value(%outer: i32) {
    // CHECK: arith.constant 0 : i32
    // Tile 0: addi c0, %outer
    // CHECK: arith.addi %{{.*}}, %[[OUTER]]
    // Tile 1: addi c1, %outer
    // CHECK: arith.addi %{{.*}}, %[[OUTER]]
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %outer : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test no barriers, no phases.
  // CHECK-LABEL: @no_barriers
  tt.func @no_barriers() {
    // CHECK: arith.constant 0 : i32
    // CHECK: arith.constant 1 : i32
    // CHECK: arith.index_cast
    // CHECK: arith.index_cast
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test teardown region with results.
  // CHECK-LABEL: @teardown_with_results
  tt.func @teardown_with_results() -> i32 {
    // CHECK: arith.constant 0 : i32
    // CHECK: arith.constant 1 : i32
    // Tiles:
    // CHECK: arith.addi
    // CHECK: arith.addi
    // Teardown:
    // CHECK: %[[RESULT:.*]] = arith.constant 42 : i32
    // CHECK: tt.return %[[RESULT]]
    // CHECK-NOT: ttng.subtiled_region
    %result = ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      } teardown {
        %c42 = arith.constant 42 : i32
        ttng.subtiled_region_yield %c42 : i32
      } -> (i32)
    tt.return %result : i32
  }

  // Test wait_barrier BEFORE a setup op (region = setup).
  // The barrier should be emitted in the setup region, before the target op.
  // CHECK-LABEL: @wait_before_setup_op
  tt.func @wait_before_setup_op(
      %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
      %phase: i32) {
    // wait_barrier should appear before the first setup op (arith.constant):
    // CHECK: ttng.wait_barrier
    // CHECK-NEXT: arith.constant 0
    // CHECK: arith.constant 1
    // Tiles:
    // CHECK: arith.index_cast
    // CHECK: arith.index_cast
    // CHECK-NOT: ttng.subtiled_region
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>)
        phases(%phase : i32)
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 0, placement = before,
            targetOpIdx = 0, barrierOpKind = "wait_barrier",
            region = setup>
        ]
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }

  // Test arrive_barrier AFTER a teardown op (region = teardown).
  // CHECK-LABEL: @arrive_after_teardown_op
  tt.func @arrive_after_teardown_op(
      %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) -> i32 {
    // Setup + tiles:
    // CHECK: arith.constant 0
    // CHECK: arith.constant 1
    // CHECK: arith.index_cast
    // CHECK: arith.index_cast
    // Teardown: arrive_barrier after the constant in teardown:
    // CHECK: arith.constant 42
    // CHECK-NEXT: ttng.arrive_barrier
    // CHECK-NOT: ttng.subtiled_region
    %result = ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>)
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 0, placement = after,
            targetOpIdx = 0, barrierOpKind = "arrive_barrier",
            region = teardown>
        ]
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      } teardown {
        %c42 = arith.constant 42 : i32
        ttng.subtiled_region_yield %c42 : i32
      } -> (i32)
    tt.return %result : i32
  }
}
