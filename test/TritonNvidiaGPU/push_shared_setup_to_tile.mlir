// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-push-shared-setup-to-tile | FileCheck %s

// Test: shared arg (same yield index for all tiles) is pushed into tile body.
// Arg position 1 maps to yield[2] for both tiles → shared.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @push_shared_constant
  // The shared value (yield[2] = %c42) should be pushed into the tile body
  // and removed from setup yield and tile args.
  // CHECK: ttng.subtiled_region
  // CHECK:   tile_mappings = [array<i32: 0>, array<i32: 1>]
  // CHECK:   setup {
  // CHECK:     ttng.subtiled_region_yield %{{.*}}, %{{.*}} : i32, i32
  // CHECK:   } tile{
  // CHECK:     %[[C42:.*]] = arith.constant 42 : i32
  // CHECK:     arith.addi %{{.*}}, %[[C42]]
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  tt.func @push_shared_constant() {
    ttng.subtiled_region
        tile_mappings = [array<i32: 0, 2>, array<i32: 1, 2>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c128 = arith.constant 128 : i32
        %c42 = arith.constant 42 : i32
        ttng.subtiled_region_yield %c0, %c128, %c42 : i32, i32, i32
      } tile(%arg0: i32, %arg1: i32) {
        %sum = arith.addi %arg0, %arg1 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Test: external value shared across tiles. No op to clone — just replace
// the tile arg with the external value directly.

#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @push_shared_external
  // The shared external value should be used directly in the tile body.
  // CHECK: ttng.subtiled_region
  // CHECK:   tile_mappings = [array<i32: 0>, array<i32: 1>]
  // CHECK:   setup {
  // CHECK:     ttng.subtiled_region_yield %{{.*}}, %{{.*}} : i32, i32
  // CHECK:   } tile{
  // CHECK:     arith.addi %{{.*}}, %{{.*}}
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  tt.func @push_shared_external(%ext: i32) {
    ttng.subtiled_region
        tile_mappings = [array<i32: 0, 2>, array<i32: 1, 2>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c128 = arith.constant 128 : i32
        ttng.subtiled_region_yield %c0, %c128, %ext : i32, i32, i32
      } tile(%arg0: i32, %arg1: i32) {
        %sum = arith.addi %arg0, %arg1 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Test: no shared args — nothing should change.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @no_shared_args
  // CHECK: tile_mappings = [array<i32: 0>, array<i32: 1>]
  // CHECK:   setup {
  // CHECK:     ttng.subtiled_region_yield %{{.*}}, %{{.*}} : i32, i32
  // CHECK:   } tile{
  // CHECK:     arith.index_cast
  tt.func @no_shared_args() {
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
}

// -----

// Test: shared arg with a chain of setup ops that need to move together.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @push_shared_chain
  // Both ops in the chain (constant + addi) should be pushed into tile body.
  // CHECK: ttng.subtiled_region
  // CHECK:   tile_mappings = [array<i32: 0>, array<i32: 1>]
  // CHECK:   setup {
  // CHECK:     ttng.subtiled_region_yield %{{.*}}, %{{.*}} : i32, i32
  // CHECK:   } tile{
  // CHECK:     arith.constant 10
  // CHECK:     arith.addi
  // CHECK:     arith.muli
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  tt.func @push_shared_chain(%ext: i32) {
    ttng.subtiled_region
        tile_mappings = [array<i32: 0, 2>, array<i32: 1, 2>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c128 = arith.constant 128 : i32
        %c10 = arith.constant 10 : i32
        %shared = arith.addi %c10, %ext : i32
        ttng.subtiled_region_yield %c0, %c128, %shared : i32, i32, i32
      } tile(%arg0: i32, %arg1: i32) {
        %prod = arith.muli %arg0, %arg1 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Test: barrier annotations have their targetOpIdx updated when ops are
// inserted at the start of the tile body.

#shared5 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem5 = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @barrier_annotation_reindex
  // The barrier annotation targetOpIdx should be incremented by the number
  // of pushed ops (1 constant op pushed).
  // CHECK: ttng.subtiled_region
  // CHECK-SAME: barrier_annotations =
  // CHECK-SAME: targetOpIdx = 1
  tt.func @barrier_annotation_reindex(
      %bar: !ttg.memdesc<1xi64, #shared5, #smem5, mutable>,
      %accum_cnt: i64) {
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared5, #smem5, mutable>)
        accum_cnts(%accum_cnt : i64)
        tile_mappings = [array<i32: 0, 2>, array<i32: 1, 2>]
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 0, placement = before,
              targetOpIdx = 0, barrierOpKind = "wait_barrier">
        ]
      setup {
        %c0 = arith.constant 0 : i32
        %c128 = arith.constant 128 : i32
        %c42 = arith.constant 42 : i32
        ttng.subtiled_region_yield %c0, %c128, %c42 : i32, i32, i32
      } tile(%arg0: i32, %arg1: i32) {
        %sum = arith.addi %arg0, %arg1 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}
