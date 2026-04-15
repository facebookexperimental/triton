// RUN: triton-opt %s -split-input-file -test-print-backward-slice-with-ws 2>&1 | FileCheck %s
// CHECK-NOT/CHECK-DAG cannot reliably exclude patterns that appear between
// out-of-order DAG matches. Use grep to assert NOT-IN-SLICE tags never appear.
// RUN: triton-opt %s -split-input-file -test-print-backward-slice-with-ws 2>&1 | not grep "NOT-IN-SLICE"

// Test 1: CF block args are traced through predecessors.
// %a feeds ^bb1 via cf.br. The slice of %add should include both %a and %b.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_no_ws_cf_block_args() {
    %a = arith.constant 1 : i32          // a-def
    %b = arith.constant 2 : i32          // b-def
    cf.br ^bb1(%a : i32)

  ^bb1(%x: i32):
    %add = arith.addi %x, %b {slice_target} : i32   // add-def
    tt.return
  }
}

// CHECK-DAG: a-def
// CHECK-DAG: b-def
// CHECK-DAG: add-def

// -----

// Test 2: WS partition entry block args are traced through to the
// corresponding WarpSpecializeOp operands.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_ws_cf_block_arg() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32         // NOT-IN-SLICE-trunk-c0
    %c1_i32 = arith.constant 1 : i32         // NOT-IN-SLICE-trunk-c1

    %cta_bars = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>  // cta-bars-def
    %other_bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>  // NOT-IN-SLICE-other-bar-def

    ttg.warp_specialize(%other_bar, %cta_bars) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2xi64, #shared, #smem, mutable>, %arg1: !ttg.memdesc<2xi64, #shared, #smem, mutable>) num_warps(1) {
      %c0 = arith.constant 0 : i32             // ws2-c0
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32             // ws2-c2
      %c10 = arith.constant 10 : i32
      cf.br ^bb1(%c0 : i32)

    ^bb1(%bar_idx: i32):
      %cmp = arith.cmpi slt, %bar_idx, %c10 : i32
      cf.cond_br %cmp, ^bb2, ^bb3

    ^bb2:
      %idx = arith.remsi %bar_idx, %c2 : i32   // ws2-idx
      %bar = ttg.memdesc_index %arg1[%idx] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>  // bar-def
      %remote = ttng.map_to_remote_buffer %bar, %c0 {slice_target} : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>  // remote-def
      %next = arith.addi %bar_idx, %c1 : i32   // ws2-next
      cf.br ^bb1(%next : i32)

    ^bb3:
      ttg.warp_return
    } : (!ttg.memdesc<2xi64, #shared, #smem, mutable>, !ttg.memdesc<2xi64, #shared, #smem, mutable>) -> ()

    tt.return
  }
}

// CHECK-DAG: cta-bars-def
// CHECK-DAG: bar-def
// CHECK-DAG: remote-def
// CHECK-DAG: ws2-c0
// CHECK-DAG: ws2-c2
// CHECK-DAG: ws2-idx
// CHECK-DAG: ws2-next

// -----

// Test 3: cf.cond_br — slice includes values from both branches.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_cf_cond_br() {
    %cond = arith.constant true     // condbr-cond
    %a = arith.constant 1 : i32    // condbr-a
    %b = arith.constant 2 : i32    // condbr-b
    %c = arith.constant 3 : i32    // condbr-c
    cf.cond_br %cond, ^bb1(%a : i32), ^bb1(%b : i32)

  ^bb1(%x: i32):
    %r = arith.addi %x, %c {slice_target} : i32  // condbr-r
    tt.return
  }
}

// CHECK-DAG: condbr-a
// CHECK-DAG: condbr-b
// CHECK-DAG: condbr-c
// CHECK-DAG: condbr-r

// -----

// Test 4: scf.if — slice includes yields from both branches.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_scf_if() {
    %cond = arith.constant true    // if-cond
    %a = arith.constant 1 : i32   // if-a
    %b = arith.constant 2 : i32   // if-b
    %r = scf.if %cond -> i32 {
      scf.yield %a : i32
    } else {
      scf.yield %b : i32
    }
    %out = arith.addi %r, %r {slice_target} : i32  // if-out
    tt.return
  }
}

// CHECK-DAG: if-a
// CHECK-DAG: if-b
// CHECK-DAG: if-out

// -----

// Test 5: scf.while — slice includes inits and condition forwarded args.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_scf_while() {
    %init = arith.constant 0 : i32          // while-init
    %step = arith.constant 1 : i32          // while-step
    %limit = arith.constant 10 : i32
    %r = scf.while (%arg = %init) : (i32) -> i32 {
      %cmp = arith.cmpi slt, %arg, %limit : i32
      scf.condition(%cmp) %arg : i32
    } do {
    ^bb0(%val: i32):
      %next = arith.addi %val, %step : i32  // while-next
      scf.yield %next : i32
    }
    %out = arith.addi %r, %r {slice_target} : i32  // while-out
    tt.return
  }
}

// CHECK-DAG: while-init
// CHECK-DAG: while-step
// CHECK-DAG: while-next
// CHECK-DAG: while-out

// -----

// Test 6: scf.while with slice target inside the "after" region — the
// after region is only reachable from the before region (via scf.condition),
// not from the parent op.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_scf_while_after_body() {
    %init = arith.constant 0 : i32          // whilebody-init
    %step = arith.constant 1 : i32          // whilebody-step
    %limit = arith.constant 10 : i32
    %r = scf.while (%arg = %init) : (i32) -> i32 {
      %cmp = arith.cmpi slt, %arg, %limit : i32
      scf.condition(%cmp) %arg : i32
    } do {
    ^bb0(%val: i32):
      %next = arith.addi %val, %step {slice_target} : i32  // whilebody-next
      scf.yield %next : i32
    }
    tt.return
  }
}

// CHECK-DAG: whilebody-init
// CHECK-DAG: whilebody-step
// CHECK-DAG: whilebody-next

// -----

// Test 7: scf.for with slice target inside the loop body — slice traces
// the iter_arg back to both the init value and the previous yield.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_scf_for_body() {
    %c0 = arith.constant 0 : index           // NOT-IN-SLICE-forbody-lb
    %c1 = arith.constant 1 : index           // NOT-IN-SLICE-forbody-step-idx
    %c4 = arith.constant 4 : index           // NOT-IN-SLICE-forbody-ub
    %init = arith.constant 0 : i32           // forbody-init
    %step = arith.constant 1 : i32           // forbody-step
    %r = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %init) -> i32 {
      %next = arith.addi %acc, %step {slice_target} : i32  // forbody-next
      scf.yield %next : i32
    }
    tt.return
  }
}

// CHECK-DAG: forbody-init
// CHECK-DAG: forbody-step
// CHECK-DAG: forbody-next

// -----

// Test 8: scf.for with slice target after the loop — slice includes
// the init value and the loop body yield.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_scf_for() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %init = arith.constant 0 : i32           // for-init
    %step = arith.constant 1 : i32           // for-step
    %r = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %init) -> i32 {
      %next = arith.addi %acc, %step : i32   // for-next
      scf.yield %next : i32
    }
    %out = arith.addi %r, %r {slice_target} : i32  // for-out
    tt.return
  }
}

// CHECK-DAG: for-init
// CHECK-DAG: for-step
// CHECK-DAG: for-next
// CHECK-DAG: for-out

// -----

// Test 9: WS partition containing scf.for — both WS entry arg and
// scf.for iter_arg are traced through.

#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem3 = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_ws_scf_for() {
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared3, #smem3, mutable>  // wsfor-bars

    ttg.warp_specialize(%bars) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2xi64, #shared3, #smem3, mutable>) num_warps(1) {
      %c0 = arith.constant 0 : index         // NOT-IN-SLICE-wsfor-lb
      %c1 = arith.constant 1 : index         // NOT-IN-SLICE-wsfor-step-idx
      %c4 = arith.constant 4 : index         // NOT-IN-SLICE-wsfor-ub
      %c0_i32 = arith.constant 0 : i32       // wsfor-init
      %r = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %c0_i32) -> i32 {
        %bar = ttg.memdesc_index %arg0[%acc] : !ttg.memdesc<2xi64, #shared3, #smem3, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem3, mutable>  // wsfor-bar
        %remote = ttng.map_to_remote_buffer %bar, %acc {slice_target} : !ttg.memdesc<1xi64, #shared3, #smem3, mutable> -> !ttg.memdesc<1xi64, #shared3, #ttng.shared_cluster_memory, mutable>  // wsfor-remote
        %next = arith.addi %acc, %acc : i32
        scf.yield %next : i32
      }
      ttg.warp_return
    } : (!ttg.memdesc<2xi64, #shared3, #smem3, mutable>) -> ()

    tt.return
  }
}

// CHECK-DAG: wsfor-bars
// CHECK-DAG: wsfor-init
// CHECK-DAG: wsfor-bar
// CHECK-DAG: wsfor-remote

// -----

// Test 10: scf.for with multiple iter_args — the second iter_arg's
// index must be adjusted correctly.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_scf_for_multi_iter() {
    %c0 = arith.constant 0 : index           // NOT-IN-SLICE-multi-lb
    %c1 = arith.constant 1 : index           // NOT-IN-SLICE-multi-step-idx
    %c4 = arith.constant 4 : index           // NOT-IN-SLICE-multi-ub
    %initA = arith.constant 0 : i32          // NOT-IN-SLICE-multi-initA
    %initB = arith.constant 10 : i32         // multi-initB
    %step = arith.constant 1 : i32           // multi-step
    %rA, %rB = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%a = %initA, %b = %initB) -> (i32, i32) {
      %nextA = arith.addi %a, %step : i32
      %nextB = arith.addi %b, %step {slice_target} : i32  // multi-nextB
      scf.yield %nextA, %nextB : i32, i32
    }
    tt.return
  }
}

// Slice of %nextB should include %initB (second init) and %step,
// but NOT %initA (first init).
// CHECK-DAG: multi-initB
// CHECK-DAG: multi-step
// CHECK-DAG: multi-nextB

// -----

// Test 11: Uses-from-above inside scf.for — a value defined outside
// the loop is used directly inside without being a block arg.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_uses_from_above() {
    %c0 = arith.constant 0 : index           // NOT-IN-SLICE-above-lb
    %c1 = arith.constant 1 : index           // NOT-IN-SLICE-above-step-idx
    %c4 = arith.constant 4 : index           // NOT-IN-SLICE-above-ub
    %init = arith.constant 0 : i32
    %bias = arith.constant 42 : i32          // above-bias
    %r = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %init) -> i32 {
      %next = arith.addi %acc, %bias {slice_target} : i32  // above-next
      scf.yield %next : i32
    }
    tt.return
  }
}

// CHECK-DAG: above-bias
// CHECK-DAG: above-next

// -----

// Test 12: scf.for with multiple results — slice target uses only
// result #1, so result #0's init and yield should NOT be in the slice.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_scf_for_multi_result() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %initA = arith.constant 0 : i32          // NOT-IN-SLICE-forres-initA
    %initB = arith.constant 10 : i32         // forres-initB
    %step = arith.constant 1 : i32           // forres-step
    %rA, %rB = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%a = %initA, %b = %initB) -> (i32, i32) {
      %nextA = arith.addi %a, %step : i32    // NOT-IN-SLICE-forres-nextA
      %nextB = arith.addi %b, %step : i32   // forres-nextB
      scf.yield %nextA, %nextB : i32, i32
    }
    %out = arith.addi %rB, %rB {slice_target} : i32  // forres-out
    tt.return
  }
}

// CHECK-DAG: forres-initB
// CHECK-DAG: forres-step
// CHECK-DAG: forres-nextB
// CHECK-DAG: forres-out

// -----

// Test 13: scf.for's upper bound is an scf.if result (a RegionBranchOp
// result used as a control operand). The scf.if result enters the worklist
// from the control-operand path; getBackwardSlice must not filter it out.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_nested_region_branch() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cond = arith.constant true
    %lo = arith.constant 4 : index           // nested-lo
    %hi = arith.constant 8 : index           // nested-hi
    %bound = scf.if %cond -> index {
      scf.yield %lo : index
    } else {
      scf.yield %hi : index
    }
    %init = arith.constant 0 : i32           // nested-init
    %step = arith.constant 1 : i32           // nested-step
    %r = scf.for %iv = %c0 to %bound step %c1 iter_args(%acc = %init) -> i32 {
      %next = arith.addi %acc, %step : i32   // nested-next
      scf.yield %next : i32
    }
    %out = arith.addi %r, %r {slice_target} : i32  // nested-out
    tt.return
  }
}

// CHECK-DAG: nested-lo
// CHECK-DAG: nested-hi
// CHECK-DAG: nested-init
// CHECK-DAG: nested-step
// CHECK-DAG: nested-next
// CHECK-DAG: nested-out

// -----

// Test 14: scf.while with multiple iter_args — slice target uses only
// result #1. Result #0's init should not be in the slice.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_scf_while_multi() {
    %initA = arith.constant 0 : i32          // NOT-IN-SLICE-while-initA
    %initB = arith.constant 10 : i32         // while-multi-initB
    %step = arith.constant 1 : i32           // while-multi-step
    %limit = arith.constant 100 : i32
    %rA, %rB = scf.while (%a = %initA, %b = %initB) : (i32, i32) -> (i32, i32) {
      %cmp = arith.cmpi slt, %a, %limit : i32
      scf.condition(%cmp) %a, %b : i32, i32
    } do {
    ^bb0(%valA: i32, %valB: i32):
      %nextA = arith.addi %valA, %step : i32
      %nextB = arith.addi %valB, %step : i32  // while-multi-nextB
      scf.yield %nextA, %nextB : i32, i32
    }
    %out = arith.addi %rB, %rB {slice_target} : i32  // while-multi-out
    tt.return
  }
}

// CHECK-DAG: while-multi-initB
// CHECK-DAG: while-multi-step
// CHECK-DAG: while-multi-nextB
// CHECK-DAG: while-multi-out
