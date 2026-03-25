// RUN: triton-opt %s -split-input-file --nvgpu-test-fuse-subtile-regions | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // Test 1: Basic fusion of two adjacent 2-tile ops, each with 1 tile arg.
  // The fused op should have 2 tile args and interleaved yields.
  // CHECK-LABEL: @basic_fusion
  // CHECK: ttng.subtiled_region
  // CHECK:   setup
  // CHECK:     %[[A0:.*]] = arith.constant 10
  // CHECK:     %[[A1:.*]] = arith.constant 11
  // CHECK:     %[[B0:.*]] = arith.constant 20
  // CHECK:     %[[B1:.*]] = arith.constant 21
  // CHECK:     ttng.subtiled_region_yield %[[A0]], %[[B0]], %[[A1]], %[[B1]]
  // CHECK:   tile
  // CHECK:     arith.index_cast
  // CHECK:     arith.muli
  // CHECK:     ttng.subtiled_region_yield
  // Originals should be erased.
  // CHECK-NOT: ttng.subtiled_region barrier_annotations
  tt.func @basic_fusion() {
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c10 = arith.constant 10 : i32
        %c11 = arith.constant 11 : i32
        ttng.subtiled_region_yield %c10, %c11 : i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      }
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c20 = arith.constant 20 : i32
        %c21 = arith.constant 21 : i32
        ttng.subtiled_region_yield %c20, %c21 : i32, i32
      } tile(%arg0: i32) {
        %c2 = arith.constant 2 : i32
        %val = arith.muli %arg0, %c2 : i32
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // Test 2: Different tile arg counts. A has 1 arg, B has 2 args → fused has 3.
  // CHECK-LABEL: @different_tile_arg_counts
  // CHECK: ttng.subtiled_region
  // CHECK:   setup
  // A yields 2 values (1 arg x 2 tiles), B yields 4 values (2 args x 2 tiles)
  // Fused interleaved: a0, b0_0, b0_1, a1, b1_0, b1_1
  // CHECK:     ttng.subtiled_region_yield
  // CHECK-SAME: i32, i32, i32, i32, i32, i32
  // CHECK:   tile
  // CHECK:   ^bb0(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32):
  // CHECK-NOT: ttng.subtiled_region barrier_annotations
  tt.func @different_tile_arg_counts() {
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      }
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c10 = arith.constant 10 : i32
        %c11 = arith.constant 11 : i32
        %c12 = arith.constant 12 : i32
        %c13 = arith.constant 13 : i32
        ttng.subtiled_region_yield %c10, %c11, %c12, %c13 : i32, i32, i32, i32
      } tile(%arg0: i32, %arg1: i32) {
        %sum = arith.addi %arg0, %arg1 : i32
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // Test 3: Captured outer-scope values. Tile bodies reference function args.
  // CHECK-LABEL: @captured_outer_scope
  // CHECK: ttng.subtiled_region
  // CHECK:   tile
  // CHECK:     arith.addi
  // CHECK:     arith.addi
  // CHECK-NOT: ttng.subtiled_region barrier_annotations
  tt.func @captured_outer_scope(%base: i32) {
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %v = arith.addi %arg0, %base : i32
        ttng.subtiled_region_yield
      }
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c10 = arith.constant 10 : i32
        %c11 = arith.constant 11 : i32
        ttng.subtiled_region_yield %c10, %c11 : i32, i32
      } tile(%arg0: i32) {
        %v = arith.addi %arg0, %base : i32
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // Test 4: Non-fusible because of different numTiles (2 vs 4).
  // CHECK-LABEL: @different_num_tiles
  // CHECK: ttng.subtiled_region
  // CHECK: ttng.subtiled_region
  tt.func @different_num_tiles() {
    // 2 tiles (2 yields / 1 arg = 2)
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      }
    // 4 tiles (4 yields / 1 arg = 4)
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %c2 = arith.constant 2 : i32
        %c3 = arith.constant 3 : i32
        ttng.subtiled_region_yield %c0, %c1, %c2, %c3 : i32, i32, i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // Test 5: Non-fusible because of side-effectful intervening op.
  // CHECK-LABEL: @non_adjacent
  // CHECK: ttng.subtiled_region
  // CHECK: ttg.local_alloc
  // CHECK: ttng.subtiled_region
  tt.func @non_adjacent(%src: tensor<128xi32>) {
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      }
    // Side-effectful op between the two subtiled_region ops.
    %alloc = ttg.local_alloc %src : (tensor<128xi32>) -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c10 = arith.constant 10 : i32
        %c11 = arith.constant 11 : i32
        ttng.subtiled_region_yield %c10, %c11 : i32, i32
      } tile(%arg0: i32) {
        %c2 = arith.constant 2 : i32
        %val = arith.muli %arg0, %c2 : i32
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // Test 6: Chain fusion of three adjacent fusible ops into one.
  // CHECK-LABEL: @chain_fusion
  // CHECK: ttng.subtiled_region
  // Fused should have 3 tile args (1 from each).
  // CHECK:   tile
  // CHECK:   ^bb0(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32):
  tt.func @chain_fusion() {
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      }
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c10 = arith.constant 10 : i32
        %c11 = arith.constant 11 : i32
        ttng.subtiled_region_yield %c10, %c11 : i32, i32
      } tile(%arg0: i32) {
        %c2 = arith.constant 2 : i32
        %val = arith.muli %arg0, %c2 : i32
        ttng.subtiled_region_yield
      }
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c20 = arith.constant 20 : i32
        %c21 = arith.constant 21 : i32
        ttng.subtiled_region_yield %c20, %c21 : i32, i32
      } tile(%arg0: i32) {
        %c3 = arith.constant 3 : i32
        %val = arith.addi %arg0, %c3 : i32
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// RUN: triton-opt %s -split-input-file --nvgpu-test-fuse-subtile-regions --triton-nvidia-gpu-lower-subtiled-region | FileCheck %s --check-prefix=ROUNDTRIP

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // Test 7: Round-trip test — fuse then lower, verify interleaved flat IR.
  // ROUNDTRIP-LABEL: @roundtrip_fuse_lower
  // Setup ops inlined:
  // ROUNDTRIP-DAG: %[[A0:.*]] = arith.constant 10
  // ROUNDTRIP-DAG: %[[A1:.*]] = arith.constant 11
  // ROUNDTRIP-DAG: %[[B0:.*]] = arith.constant 20
  // ROUNDTRIP-DAG: %[[B1:.*]] = arith.constant 21
  // Tile 0: A's body then B's body
  // ROUNDTRIP: arith.index_cast %[[A0]]
  // ROUNDTRIP: arith.muli %[[B0]]
  // Tile 1: A's body then B's body
  // ROUNDTRIP: arith.index_cast %[[A1]]
  // ROUNDTRIP: arith.muli %[[B1]]
  // No subtiled_region ops remain.
  // ROUNDTRIP-NOT: ttng.subtiled_region barrier_annotations
  tt.func @roundtrip_fuse_lower() {
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c10 = arith.constant 10 : i32
        %c11 = arith.constant 11 : i32
        ttng.subtiled_region_yield %c10, %c11 : i32, i32
      } tile(%arg0: i32) {
        %idx = arith.index_cast %arg0 : i32 to index
        ttng.subtiled_region_yield
      }
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c20 = arith.constant 20 : i32
        %c21 = arith.constant 21 : i32
        ttng.subtiled_region_yield %c20, %c21 : i32, i32
      } tile(%arg0: i32) {
        %c2 = arith.constant 2 : i32
        %val = arith.muli %arg0, %c2 : i32
        ttng.subtiled_region_yield
      }
    tt.return
  }
}
