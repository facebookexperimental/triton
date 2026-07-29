// RUN: triton-opt --tlx-print-ttgir-to-tlx %s | FileCheck %s

// Regression test for emitting ttg.memdesc_index as tlx.local_view.
//
// A ttg.memdesc_index indexes into a buffer/barrier array to produce the view
// referenced later in the kernel. It must be emitted as tlx.local_view when a
// real consumer needs that view, and skipped when every user is a barrier
// lifecycle op (init_barrier / inval_barrier, folded into alloc_barriers) or
// when it has no users at all.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // A view consumed by wait_barrier (a real consumer) must be emitted, even
  // though the same view is also init'd here.
  // CHECK-LABEL: def shared_barrier_view_emitted(
  // CHECK: tlx.alloc_barriers(
  // CHECK: tlx.local_view(
  // CHECK: tlx.barrier_wait(
  tt.func public @shared_barrier_view_emitted() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %view = ttg.memdesc_index %bar[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %view, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.wait_barrier %view, %c0_i32, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // A view used only by init_barrier folds into alloc_barriers and is NOT
  // emitted as a local_view.
  // CHECK-LABEL: def barrier_lifecycle_view_skipped(
  // CHECK: tlx.alloc_barriers(
  // CHECK-NOT: tlx.local_view(
  tt.func public @barrier_lifecycle_view_skipped() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %view = ttg.memdesc_index %bar[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %view, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // A view with no users is skipped (not emitted as a dead local_view).
  // CHECK-LABEL: def unused_view_skipped(
  // CHECK-NOT: tlx.local_view(
  tt.func public @unused_view_skipped() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %view = ttg.memdesc_index %bar[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
