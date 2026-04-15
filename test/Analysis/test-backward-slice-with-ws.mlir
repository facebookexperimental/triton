// RUN: triton-opt %s -split-input-file -test-print-backward-slice-with-ws 2>&1 | FileCheck %s

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
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    %cta_bars = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>  // cta-bars-def
    %other_bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>  // other-bar-def

    ttg.warp_specialize(%other_bar, %cta_bars) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2xi64, #shared, #smem, mutable>, %arg1: !ttg.memdesc<2xi64, #shared, #smem, mutable>) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %c10 = arith.constant 10 : i32
      cf.br ^bb1(%c0 : i32)

    ^bb1(%bar_idx: i32):
      %cmp = arith.cmpi slt, %bar_idx, %c10 : i32
      cf.cond_br %cmp, ^bb2, ^bb3

    ^bb2:
      %idx = arith.remsi %bar_idx, %c2 : i32
      %bar = ttg.memdesc_index %arg1[%idx] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>  // bar-def
      %remote = ttng.map_to_remote_buffer %bar, %c0 {slice_target} : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>  // remote-def
      %next = arith.addi %bar_idx, %c1 : i32
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
// CHECK-NOT: other-bar-def

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
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
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
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
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
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %initA = arith.constant 0 : i32          // multi-initA
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
// CHECK-NOT: multi-initA

// -----

// Test 11: Uses-from-above inside scf.for — a value defined outside
// the loop is used directly inside without being a block arg.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_uses_from_above() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
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
