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
  // Barrier annotations use stable op IDs (subtile_op_id attribute), so
  // targetOpIdx is unchanged even when ops are inserted before the target.
  // CHECK: ttng.subtiled_region
  // CHECK-SAME: barrier_annotations =
  // CHECK-SAME: targetOpIdx = 0
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
        %sum = arith.addi %arg0, %arg1 {subtile_op_id = 0 : i32} : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Test: per-tile tmem_load is pushed from setup into tile body.
// The setup yields memdesc (tmem_subslice result) instead of tensor
// (tmem_load result), and the tile body receives a memdesc arg with
// tmem_load + convert_layout cloned inside.

#tmem6 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem6s = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear6 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @push_tmem_load_to_tile
  // The tile body should receive a memdesc arg and contain tmem_load + convert_layout.
  // CHECK: ttng.subtiled_region
  // CHECK:   setup {
  // CHECK:     ttng.tmem_subslice
  // CHECK:     ttng.tmem_subslice
  // CHECK:     ttng.subtiled_region_yield {{.*}} !ttg.memdesc{{.*}}, !ttg.memdesc
  // CHECK:   } tile{
  // CHECK:     ttng.tmem_load %{{.*}} :
  // CHECK:     ttg.convert_layout
  // CHECK:     arith.truncf
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  tt.func @push_tmem_load_to_tile(
      %tmem_buf: !ttg.memdesc<128x128xf32, #tmem6, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token) {
    ttng.subtiled_region
        tile_mappings = [array<i32: 0, 2>, array<i32: 1, 3>]
        barrier_annotations = []
      setup {
        %s0 = ttng.tmem_subslice %tmem_buf {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem6, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem6s, #ttng.tensor_memory, mutable, 128x128>
        %l0 = ttng.tmem_load %s0 : !ttg.memdesc<128x64xf32, #tmem6s, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #linear6>
        %cvt0 = ttg.convert_layout %l0 : tensor<128x64xf32, #linear6> -> tensor<128x64xf32, #blocked6>
        %s1 = ttng.tmem_subslice %tmem_buf {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem6, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem6s, #ttng.tensor_memory, mutable, 128x128>
        %l1 = ttng.tmem_load %s1 : !ttg.memdesc<128x64xf32, #tmem6s, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #linear6>
        %cvt1 = ttg.convert_layout %l1 : tensor<128x64xf32, #linear6> -> tensor<128x64xf32, #blocked6>
        %c0 = arith.constant 0 : i32
        %c64 = arith.constant 64 : i32
        ttng.subtiled_region_yield %cvt0, %cvt1, %cvt0, %cvt1, %c0, %c64 : tensor<128x64xf32, #blocked6>, tensor<128x64xf32, #blocked6>, tensor<128x64xf32, #blocked6>, tensor<128x64xf32, #blocked6>, i32, i32
      } tile(%arg0: tensor<128x64xf32, #blocked6>, %arg1: tensor<128x64xf32, #blocked6>, %nOff: i32) {
        %trunc = arith.truncf %arg1 : tensor<128x64xf32, #blocked6> to tensor<128x64xf16, #blocked6>
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Test: shared ops are sunk to their first consumer, not placed at tile
// body start. The constant should appear right before the addi, not
// before the muli.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @sink_shared_to_consumer
  // CHECK: ttng.subtiled_region
  // CHECK:   } tile{
  // CHECK:     arith.muli
  // CHECK:     arith.constant 42
  // CHECK:     arith.addi
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  tt.func @sink_shared_to_consumer() {
    ttng.subtiled_region
        tile_mappings = [array<i32: 0, 2>, array<i32: 1, 2>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c128 = arith.constant 128 : i32
        %c42 = arith.constant 42 : i32
        ttng.subtiled_region_yield %c0, %c128, %c42 : i32, i32, i32
      } tile(%arg0: i32, %arg1: i32) {
        %prod = arith.muli %arg0, %arg0 : i32
        %sum = arith.addi %prod, %arg1 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Test: lowering a tile body with a barrier annotation and pushed shared
// ops produces the barrier at the correct position (after the pushed ops,
// before the annotated op).

#shared8 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem8 = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @barrier_after_pushed_ops
  // After pushing the shared constant, the barrier annotation on the muli
  // (subtile_op_id=0) should still target the muli, not the pushed constant.
  // CHECK: ttng.subtiled_region
  // CHECK-SAME: targetOpIdx = 0
  // CHECK:   } tile{
  // CHECK:     arith.constant 42
  // CHECK:     arith.muli {{.*}} {subtile_op_id = 0 : i32}
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  tt.func @barrier_after_pushed_ops(
      %bar: !ttg.memdesc<1xi64, #shared8, #smem8, mutable>,
      %accum_cnt: i64) {
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared8, #smem8, mutable>)
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
        %prod = arith.muli %arg0, %arg1 {subtile_op_id = 0 : i32} : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}
