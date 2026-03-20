// RUN: triton-opt --split-input-file %s --verify-diagnostics

// Verify: setup outputs not divisible by tile block args
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_setup_not_divisible() {
    // expected-error @+1 {{setup region yields 3 values, which is not divisible by the number of tile block arguments (2)}}
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %c2 = arith.constant 2 : i32
        ttng.subtiled_region_yield %c0, %c1, %c2 : i32, i32, i32
      } tile(%arg0: i32, %arg1: i32) {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: type mismatch between setup output and tile block arg
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_type_mismatch() {
    // expected-error @+1 {{type mismatch: setup output 0 has type 'i32' but tile block arg 0 has type 'f32'}}
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: f32) {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: barrierIdx out of range
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_barrier_idx_out_of_range(
      %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %phase: i32) {
    // expected-error @+1 {{barrierAnnotations[0] has barrierIdx=3 but there are only 1 barriers}}
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        phases(%phase : i32)
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 3, placement = after,
              targetOpIdx = 0, barrierOpKind = "arrive_barrier">
        ]
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: i32) {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: wait_barrier without corresponding phase
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_wait_no_phase(
      %bar0: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %bar1: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %phase: i32) {
    // expected-error @+1 {{barrierAnnotations[0] is a wait_barrier with barrierIdx=1 but there are only 1 phases}}
    ttng.subtiled_region
        barriers(%bar0, %bar1 : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        phases(%phase : i32)
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 1, placement = before,
              targetOpIdx = 0, barrierOpKind = "wait_barrier">
        ]
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: i32) {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: unknown barrierOpKind
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_unknown_barrier_kind(
      %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %phase: i32) {
    // expected-error @+1 {{barrierAnnotations[0] has unknown barrierOpKind 'bogus'}}
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        phases(%phase : i32)
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 0, placement = after,
              targetOpIdx = 0, barrierOpKind = "bogus">
        ]
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: targetOpIdx out of range
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_target_op_idx_out_of_range(
      %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %phase: i32) {
    // expected-error @+1 {{barrierAnnotations[0] has targetOpIdx=5 but tile block has only 1 ops}}
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        phases(%phase : i32)
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 0, placement = after,
              targetOpIdx = 5, barrierOpKind = "arrive_barrier">
        ]
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: teardown present but tile yields no values
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_teardown_but_no_tile_yield() {
    // expected-error @+1 {{teardown region is present but tile region yields no values}}
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: i32) {
        ttng.subtiled_region_yield
      } teardown() {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: tile yields values but no teardown region
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_tile_yield_no_teardown() {
    // expected-error @+1 {{tile region yields values but no teardown region is present}}
    ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: i32) {
        %v = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield %v : i32
      }
    tt.return
  }
}

// -----

// Verify: teardown block arg count mismatch (expected numTiles * numTileYields)
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_teardown_arg_count_mismatch() {
    // expected-error @+1 {{teardown region has 1 block arguments but expected 2 (numTiles=2 * numTileYields=1)}}
    %r = ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %v = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield %v : i32
      } teardown(%a: i32) {
        ttng.subtiled_region_yield %a : i32
      } -> (i32)
    tt.return
  }
}

// -----

// Verify: teardown block arg type mismatch
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_teardown_type_mismatch() {
    // expected-error @+1 {{teardown block arg 0 has type 'f32' but tile yield operand 0 has type 'i32'}}
    %r = ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %v = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield %v : i32
      } teardown(%a: f32, %b: f32) {
        %j = arith.addf %a, %b : f32
        ttng.subtiled_region_yield %j : f32
      } -> (f32)
    tt.return
  }
}

// -----

// Verify: teardown yield count != op result count
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_teardown_yield_count_mismatch() {
    // expected-error @+1 {{teardown region yields 2 values but op has 1 results}}
    %r = ttng.subtiled_region
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %v = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield %v : i32
      } teardown(%a: i32, %b: i32) {
        ttng.subtiled_region_yield %a, %b : i32, i32
      } -> (i32)
    tt.return
  }
}

// -----

// Verify: teardown yield type != op result type
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_teardown_yield_type_mismatch() {
    // expected-error @+1 {{teardown yield type 'i32' at position 0 doesn't match op result type 'f32'}}
    %r = ttng.subtiled_region
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
      } -> (f32)
    tt.return
  }
}
