// RUN: triton-opt %s -test-print-backward-slice-with-ws 2>&1 | FileCheck %s


// Test that getBackwardSliceWithWS correctly traces through
// WarpSpecializePartitionsOp entry block args but does NOT confuse
// non-entry CF block args (like loop induction vars) with partition args.
//
// %other_bar is operand#0 and %cta_bars is operand#1 of warp_specialize.
// Inside partition0, %arg1 maps to %cta_bars, and is used to index the
// remote barrier. The backward slice of map_to_remote_buffer should
// include %cta_bars's local_alloc but NOT %other_bar's.
//
// FIXME: Currently getBackwardSliceWithWS incorrectly maps non-entry CF
// block args (^bb1's %bar_idx, which is arg#0 of a cf.br target block)
// to wsOp.getOperand(0) = %other_bar, causing a spurious inclusion.
// Remove XFAIL once the bug is fixed.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_ws_cf_block_arg() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    %cta_bars = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    %other_bar = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>

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
      %bar = ttg.memdesc_index %arg1[%idx] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
      %remote = ttng.map_to_remote_buffer %bar, %c0 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #ttng.shared_cluster_memory, mutable>
      %next = arith.addi %bar_idx, %c1 : i32
      cf.br ^bb1(%next : i32)

    ^bb3:
      ttg.warp_return
    } : (!ttg.memdesc<2xi64, #shared, #smem, mutable>, !ttg.memdesc<2xi64, #shared, #smem, mutable>) -> ()

    tt.return
  }
}

// The slice should include %cta_bars's alloc and the ops leading to %remote.
// CHECK-DAG: in_slice: ttg.local_alloc %0
// CHECK-DAG: in_slice: ttg.memdesc_index
// CHECK-DAG: in_slice: ttng.map_to_remote_buffer

// The slice must NOT include %other_bar's alloc (%1).
// CHECK-NOT: in_slice: ttg.local_alloc %1
