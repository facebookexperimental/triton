// RUN: triton-opt %s -split-input-file -tritongpu-barrier-deadlock-detection 2>&1 | FileCheck %s

// Test: Correct producer-consumer pipeline → UNSAT
// Producer: wait(empty), expect(full), tma_load(full)
// Consumer: wait(full), arrive(empty)
// Pre-arrives on empty barriers so producer can start.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: === Barrier Deadlock Detection: correct_pipeline ===
// CHECK: Initial arrives (pre-task):
// CHECK:   bars_empty_0: 1
// CHECK:   bars_empty_1: 1
// CHECK: --- Z3 Script ---
// CHECK: from z3 import *
// CHECK: initial_arrives = {
// CHECK: def arrive_count(slot_key):
// CHECK: def arrive_count_before(slot_key, t):
// CHECK: def blocked_parity(slot_key, parity, ac):
// CHECK: # Phi_ord: intra-task operation ordering
// CHECK: # Phi_stall: stall only at WAIT positions
// CHECK: # Phi_B: stalled at WAIT -> barrier is blocked
// CHECK: # Phi_R: passed-through WAIT -> barrier was ready when reached
// CHECK: # Deadlock query: at least one task stuck
// CHECK: --- End Z3 Script ---

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @correct_pipeline(%K: i32, %desc: !tt.tensordesc<tensor<64x64xf16>>) {
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32

    %data = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %bars_empty = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable> loc("bars_empty")
    %be0 = ttg.memdesc_index %bars_empty[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %be0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %be1 = ttg.memdesc_index %bars_empty[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %be1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    %bars_full = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable> loc("bars_full")
    %bf0 = ttg.memdesc_index %bars_full[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bf0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %bf1 = ttg.memdesc_index %bars_full[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bf1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    // Pre-arrive empty barriers (buffers initially available)
    ttng.arrive_barrier %be0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %be1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    ttg.warp_specialize(%K, %data, %bars_empty, %bars_full, %desc)
    default {
      %p = scf.for %k = %c0_i32 to %K step %c1_i32 iter_args(%phase = %c0_i32) -> (i32) : i32 {
        %buf = arith.remsi %k, %c2_i32 : i32
        %empty_bar = ttg.memdesc_index %bars_empty[%buf] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %full_bar = ttg.memdesc_index %bars_full[%buf] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %data_buf = ttg.memdesc_index %data[%buf] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

        ttng.wait_barrier %empty_bar, %phase, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.barrier_expect %full_bar, 8192, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %data_buf, %full_bar, %true : !tt.tensordesc<tensor<64x64xf16>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

        // phase ^= (buf == 1)
        %is_last = arith.cmpi eq, %buf, %c1_i32 : i32
        %is_last_ext = arith.extui %is_last : i1 to i32
        %next_phase = arith.xori %phase, %is_last_ext : i32
        scf.yield %next_phase : i32
      }
      ttg.warp_yield
    }
    partition0(%arg0: i32, %arg1: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg3: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg4: !tt.tensordesc<tensor<64x64xf16>>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %t = arith.constant true
      %p2 = scf.for %k = %c0 to %arg0 step %c1 iter_args(%phase = %c0) -> (i32) : i32 {
        %buf = arith.remsi %k, %c2 : i32
        %full_bar = ttg.memdesc_index %arg3[%buf] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %empty_bar = ttg.memdesc_index %arg2[%buf] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>

        ttng.wait_barrier %full_bar, %phase, %t : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.arrive_barrier %empty_bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

        %is_last = arith.cmpi eq, %buf, %c1 : i32
        %is_last_ext = arith.extui %is_last : i1 to i32
        %next_phase = arith.xori %phase, %is_last_ext : i32
        scf.yield %next_phase : i32
      }
      ttg.warp_return
    } : (i32, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !tt.tensordesc<tensor<64x64xf16>>) -> ()
    tt.return
  }
}

// -----

// Test: Missing arrive on empty barrier → SAT (deadlock)
// Consumer does NOT arrive on empty barrier, so producer blocks forever
// on wait(empty) after the first iteration.

#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem2 = #ttg.shared_memory

// CHECK-LABEL: === Barrier Deadlock Detection: missing_arrive ===
// CHECK: --- Z3 Script ---
// CHECK: # Phi_B: stalled at WAIT -> barrier is blocked
// The producer waits on bars_empty but consumer never arrives on it.
// Z3 should find a deadlock (SAT).
// CHECK: # Deadlock query: at least one task stuck
// CHECK: --- End Z3 Script ---

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @missing_arrive(%K: i32, %desc: !tt.tensordesc<tensor<64x64xf16>>) {
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32

    %data = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xf16, #shared2, #smem2, mutable>
    %bars_empty = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared3, #smem2, mutable> loc("bars_empty")
    %be0 = ttg.memdesc_index %bars_empty[%c0_i32] : !ttg.memdesc<2xi64, #shared3, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
    ttng.init_barrier %be0, 1 : !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
    %be1 = ttg.memdesc_index %bars_empty[%c1_i32] : !ttg.memdesc<2xi64, #shared3, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
    ttng.init_barrier %be1, 1 : !ttg.memdesc<1xi64, #shared3, #smem2, mutable>

    %bars_full = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared3, #smem2, mutable> loc("bars_full")
    %bf0 = ttg.memdesc_index %bars_full[%c0_i32] : !ttg.memdesc<2xi64, #shared3, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
    ttng.init_barrier %bf0, 1 : !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
    %bf1 = ttg.memdesc_index %bars_full[%c1_i32] : !ttg.memdesc<2xi64, #shared3, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
    ttng.init_barrier %bf1, 1 : !ttg.memdesc<1xi64, #shared3, #smem2, mutable>

    // Pre-arrive empty barriers
    ttng.arrive_barrier %be0, 1 : !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
    ttng.arrive_barrier %be1, 1 : !ttg.memdesc<1xi64, #shared3, #smem2, mutable>

    ttg.warp_specialize(%K, %data, %bars_empty, %bars_full, %desc)
    default {
      // Producer: wait(empty), expect(full), tma_load(full)
      %p = scf.for %k = %c0_i32 to %K step %c1_i32 iter_args(%phase = %c0_i32) -> (i32) : i32 {
        %buf = arith.remsi %k, %c2_i32 : i32
        %empty_bar = ttg.memdesc_index %bars_empty[%buf] : !ttg.memdesc<2xi64, #shared3, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
        %full_bar = ttg.memdesc_index %bars_full[%buf] : !ttg.memdesc<2xi64, #shared3, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
        %data_buf = ttg.memdesc_index %data[%buf] : !ttg.memdesc<2x64x64xf16, #shared2, #smem2, mutable> -> !ttg.memdesc<64x64xf16, #shared2, #smem2, mutable>

        ttng.wait_barrier %empty_bar, %phase, %true : !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
        ttng.barrier_expect %full_bar, 8192, %true : !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %data_buf, %full_bar, %true : !tt.tensordesc<tensor<64x64xf16>>, !ttg.memdesc<1xi64, #shared3, #smem2, mutable> -> !ttg.memdesc<64x64xf16, #shared2, #smem2, mutable>

        %is_last = arith.cmpi eq, %buf, %c1_i32 : i32
        %is_last_ext = arith.extui %is_last : i1 to i32
        %next_phase = arith.xori %phase, %is_last_ext : i32
        scf.yield %next_phase : i32
      }
      ttg.warp_yield
    }
    partition0(%arg0: i32, %arg1: !ttg.memdesc<2x64x64xf16, #shared2, #smem2, mutable>, %arg2: !ttg.memdesc<2xi64, #shared3, #smem2, mutable>, %arg3: !ttg.memdesc<2xi64, #shared3, #smem2, mutable>, %arg4: !tt.tensordesc<tensor<64x64xf16>>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %t = arith.constant true
      // Consumer: wait(full) but NO arrive(empty) — BUG!
      %p2 = scf.for %k = %c0 to %arg0 step %c1 iter_args(%phase = %c0) -> (i32) : i32 {
        %buf = arith.remsi %k, %c2 : i32
        %full_bar = ttg.memdesc_index %arg3[%buf] : !ttg.memdesc<2xi64, #shared3, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem2, mutable>

        ttng.wait_barrier %full_bar, %phase, %t : !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
        // Missing: ttng.arrive_barrier %empty_bar, 1

        %is_last = arith.cmpi eq, %buf, %c1 : i32
        %is_last_ext = arith.extui %is_last : i1 to i32
        %next_phase = arith.xori %phase, %is_last_ext : i32
        scf.yield %next_phase : i32
      }
      ttg.warp_return
    } : (i32, !ttg.memdesc<2x64x64xf16, #shared2, #smem2, mutable>, !ttg.memdesc<2xi64, #shared3, #smem2, mutable>, !ttg.memdesc<2xi64, #shared3, #smem2, mutable>, !tt.tensordesc<tensor<64x64xf16>>) -> ()
    tt.return
  }
}

// -----

// Test: Missing pre-arrive on empty barriers → SAT (deadlock)
// Without pre-arrives, producer immediately blocks on wait(empty[0]) at phase=0
// because A(empty_0) = 0, parity 0 == phase 0 → blocked.

#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared5 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem4 = #ttg.shared_memory

// CHECK-LABEL: === Barrier Deadlock Detection: missing_pre_arrive ===
// No initial arrives should appear.
// CHECK-NOT: Initial arrives (pre-task):
// CHECK: --- Z3 Script ---
// CHECK: initial_arrives = {}
// CHECK: # Phi_B: stalled at WAIT -> barrier is blocked
// CHECK: # Deadlock query: at least one task stuck
// CHECK: --- End Z3 Script ---

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @missing_pre_arrive(%K: i32, %desc: !tt.tensordesc<tensor<64x64xf16>>) {
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32

    %data = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xf16, #shared4, #smem4, mutable>
    %bars_empty = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared5, #smem4, mutable> loc("bars_empty")
    %be0 = ttg.memdesc_index %bars_empty[%c0_i32] : !ttg.memdesc<2xi64, #shared5, #smem4, mutable> -> !ttg.memdesc<1xi64, #shared5, #smem4, mutable>
    ttng.init_barrier %be0, 1 : !ttg.memdesc<1xi64, #shared5, #smem4, mutable>
    %be1 = ttg.memdesc_index %bars_empty[%c1_i32] : !ttg.memdesc<2xi64, #shared5, #smem4, mutable> -> !ttg.memdesc<1xi64, #shared5, #smem4, mutable>
    ttng.init_barrier %be1, 1 : !ttg.memdesc<1xi64, #shared5, #smem4, mutable>

    %bars_full = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared5, #smem4, mutable> loc("bars_full")
    %bf0 = ttg.memdesc_index %bars_full[%c0_i32] : !ttg.memdesc<2xi64, #shared5, #smem4, mutable> -> !ttg.memdesc<1xi64, #shared5, #smem4, mutable>
    ttng.init_barrier %bf0, 1 : !ttg.memdesc<1xi64, #shared5, #smem4, mutable>
    %bf1 = ttg.memdesc_index %bars_full[%c1_i32] : !ttg.memdesc<2xi64, #shared5, #smem4, mutable> -> !ttg.memdesc<1xi64, #shared5, #smem4, mutable>
    ttng.init_barrier %bf1, 1 : !ttg.memdesc<1xi64, #shared5, #smem4, mutable>

    // BUG: No pre-arrives on empty barriers! Producer will block immediately.

    ttg.warp_specialize(%K, %data, %bars_empty, %bars_full, %desc)
    default {
      %p = scf.for %k = %c0_i32 to %K step %c1_i32 iter_args(%phase = %c0_i32) -> (i32) : i32 {
        %buf = arith.remsi %k, %c2_i32 : i32
        %empty_bar = ttg.memdesc_index %bars_empty[%buf] : !ttg.memdesc<2xi64, #shared5, #smem4, mutable> -> !ttg.memdesc<1xi64, #shared5, #smem4, mutable>
        %full_bar = ttg.memdesc_index %bars_full[%buf] : !ttg.memdesc<2xi64, #shared5, #smem4, mutable> -> !ttg.memdesc<1xi64, #shared5, #smem4, mutable>
        %data_buf = ttg.memdesc_index %data[%buf] : !ttg.memdesc<2x64x64xf16, #shared4, #smem4, mutable> -> !ttg.memdesc<64x64xf16, #shared4, #smem4, mutable>

        ttng.wait_barrier %empty_bar, %phase, %true : !ttg.memdesc<1xi64, #shared5, #smem4, mutable>
        ttng.barrier_expect %full_bar, 8192, %true : !ttg.memdesc<1xi64, #shared5, #smem4, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %data_buf, %full_bar, %true : !tt.tensordesc<tensor<64x64xf16>>, !ttg.memdesc<1xi64, #shared5, #smem4, mutable> -> !ttg.memdesc<64x64xf16, #shared4, #smem4, mutable>

        %is_last = arith.cmpi eq, %buf, %c1_i32 : i32
        %is_last_ext = arith.extui %is_last : i1 to i32
        %next_phase = arith.xori %phase, %is_last_ext : i32
        scf.yield %next_phase : i32
      }
      ttg.warp_yield
    }
    partition0(%arg0: i32, %arg1: !ttg.memdesc<2x64x64xf16, #shared4, #smem4, mutable>, %arg2: !ttg.memdesc<2xi64, #shared5, #smem4, mutable>, %arg3: !ttg.memdesc<2xi64, #shared5, #smem4, mutable>, %arg4: !tt.tensordesc<tensor<64x64xf16>>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %t = arith.constant true
      %p2 = scf.for %k = %c0 to %arg0 step %c1 iter_args(%phase = %c0) -> (i32) : i32 {
        %buf = arith.remsi %k, %c2 : i32
        %full_bar = ttg.memdesc_index %arg3[%buf] : !ttg.memdesc<2xi64, #shared5, #smem4, mutable> -> !ttg.memdesc<1xi64, #shared5, #smem4, mutable>
        %empty_bar = ttg.memdesc_index %arg2[%buf] : !ttg.memdesc<2xi64, #shared5, #smem4, mutable> -> !ttg.memdesc<1xi64, #shared5, #smem4, mutable>

        ttng.wait_barrier %full_bar, %phase, %t : !ttg.memdesc<1xi64, #shared5, #smem4, mutable>
        ttng.arrive_barrier %empty_bar, 1 : !ttg.memdesc<1xi64, #shared5, #smem4, mutable>

        %is_last = arith.cmpi eq, %buf, %c1 : i32
        %is_last_ext = arith.extui %is_last : i1 to i32
        %next_phase = arith.xori %phase, %is_last_ext : i32
        scf.yield %next_phase : i32
      }
      ttg.warp_return
    } : (i32, !ttg.memdesc<2x64x64xf16, #shared4, #smem4, mutable>, !ttg.memdesc<2xi64, #shared5, #smem4, mutable>, !ttg.memdesc<2xi64, #shared5, #smem4, mutable>, !tt.tensordesc<tensor<64x64xf16>>) -> ()
    tt.return
  }
}

// -----

// Test: Circular wait → SAT (deadlock)
// Two tasks cross-wait on each other's barriers with no pre-arrives.
// Task 0 (default): wait(bar_a), arrive(bar_b)
// Task 1 (partition0): wait(bar_b), arrive(bar_a)
// Neither can make progress since both are blocked at their first wait.

#shared6 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem6 = #ttg.shared_memory

// CHECK-LABEL: === Barrier Deadlock Detection: circular_wait ===
// CHECK: --- Z3 Script ---
// CHECK: initial_arrives = {}
// CHECK: # Phi_B: stalled at WAIT -> barrier is blocked
// CHECK: # Deadlock query: at least one task stuck
// CHECK: --- End Z3 Script ---

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @circular_wait(%K: i32) {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32

    %bar_a = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared6, #smem6, mutable> loc("bar_a")
    %ba0 = ttg.memdesc_index %bar_a[%c0_i32] : !ttg.memdesc<1xi64, #shared6, #smem6, mutable> -> !ttg.memdesc<1xi64, #shared6, #smem6, mutable>
    ttng.init_barrier %ba0, 1 : !ttg.memdesc<1xi64, #shared6, #smem6, mutable>

    %bar_b = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared6, #smem6, mutable> loc("bar_b")
    %bb0 = ttg.memdesc_index %bar_b[%c0_i32] : !ttg.memdesc<1xi64, #shared6, #smem6, mutable> -> !ttg.memdesc<1xi64, #shared6, #smem6, mutable>
    ttng.init_barrier %bb0, 1 : !ttg.memdesc<1xi64, #shared6, #smem6, mutable>

    // No pre-arrives: classic circular dependency.

    ttg.warp_specialize(%K, %bar_a, %bar_b)
    default {
      // Task 0: wait(bar_a) then arrive(bar_b)
      %p = scf.for %k = %c0_i32 to %K step %c1_i32 iter_args(%phase = %c0_i32) -> (i32) : i32 {
        %a = ttg.memdesc_index %bar_a[%c0_i32] : !ttg.memdesc<1xi64, #shared6, #smem6, mutable> -> !ttg.memdesc<1xi64, #shared6, #smem6, mutable>
        %b = ttg.memdesc_index %bar_b[%c0_i32] : !ttg.memdesc<1xi64, #shared6, #smem6, mutable> -> !ttg.memdesc<1xi64, #shared6, #smem6, mutable>

        ttng.wait_barrier %a, %phase, %true : !ttg.memdesc<1xi64, #shared6, #smem6, mutable>
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #shared6, #smem6, mutable>

        scf.yield %phase : i32
      }
      ttg.warp_yield
    }
    partition0(%arg0: i32, %arg1: !ttg.memdesc<1xi64, #shared6, #smem6, mutable>, %arg2: !ttg.memdesc<1xi64, #shared6, #smem6, mutable>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %t = arith.constant true
      // Task 1: wait(bar_b) then arrive(bar_a)
      %p2 = scf.for %k = %c0 to %arg0 step %c1 iter_args(%phase = %c0) -> (i32) : i32 {
        %b = ttg.memdesc_index %arg2[%c0] : !ttg.memdesc<1xi64, #shared6, #smem6, mutable> -> !ttg.memdesc<1xi64, #shared6, #smem6, mutable>
        %a = ttg.memdesc_index %arg1[%c0] : !ttg.memdesc<1xi64, #shared6, #smem6, mutable> -> !ttg.memdesc<1xi64, #shared6, #smem6, mutable>

        ttng.wait_barrier %b, %phase, %t : !ttg.memdesc<1xi64, #shared6, #smem6, mutable>
        ttng.arrive_barrier %a, 1 : !ttg.memdesc<1xi64, #shared6, #smem6, mutable>

        scf.yield %phase : i32
      }
      ttg.warp_return
    } : (i32, !ttg.memdesc<1xi64, #shared6, #smem6, mutable>, !ttg.memdesc<1xi64, #shared6, #smem6, mutable>) -> ()
    tt.return
  }
}

// -----

// Test: Arrive count mismatch → SAT (deadlock)
// Barrier initialized with count=2 (needs 2 arrives per phase), but only
// 1 pre-arrive and 1 arrive per cycle from the consumer.
// With ac=2: blocked_parity = (A/2) % 2 == phase.
// Pre-arrive gives A=1, (1/2)=0, 0%2==0 == phase(0) → blocked immediately.

#shared7 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared8 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem7 = #ttg.shared_memory

// CHECK-LABEL: === Barrier Deadlock Detection: arrive_count_mismatch ===
// CHECK: Initial arrives (pre-task):
// CHECK:   bars_empty_0: 1
// CHECK: --- Z3 Script ---
// CHECK: # Phi_B: stalled at WAIT -> barrier is blocked
// CHECK: # Deadlock query: at least one task stuck
// CHECK: --- End Z3 Script ---

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @arrive_count_mismatch(%K: i32, %desc: !tt.tensordesc<tensor<64x64xf16>>) {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32

    %data = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared7, #smem7, mutable>

    // BUG: init_barrier with count=2, but we only have 1 arrive source.
    %bars_empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared8, #smem7, mutable> loc("bars_empty")
    %be0 = ttg.memdesc_index %bars_empty[%c0_i32] : !ttg.memdesc<1xi64, #shared8, #smem7, mutable> -> !ttg.memdesc<1xi64, #shared8, #smem7, mutable>
    ttng.init_barrier %be0, 2 : !ttg.memdesc<1xi64, #shared8, #smem7, mutable>

    %bars_full = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared8, #smem7, mutable> loc("bars_full")
    %bf0 = ttg.memdesc_index %bars_full[%c0_i32] : !ttg.memdesc<1xi64, #shared8, #smem7, mutable> -> !ttg.memdesc<1xi64, #shared8, #smem7, mutable>
    ttng.init_barrier %bf0, 1 : !ttg.memdesc<1xi64, #shared8, #smem7, mutable>

    // Only 1 pre-arrive, but count=2 needed.
    ttng.arrive_barrier %be0, 1 : !ttg.memdesc<1xi64, #shared8, #smem7, mutable>

    ttg.warp_specialize(%K, %data, %bars_empty, %bars_full, %desc)
    default {
      // Producer: wait(empty[0]), expect(full[0]), tma_load(full[0])
      %p = scf.for %k = %c0_i32 to %K step %c1_i32 iter_args(%phase = %c0_i32) -> (i32) : i32 {
        %empty_bar = ttg.memdesc_index %bars_empty[%c0_i32] : !ttg.memdesc<1xi64, #shared8, #smem7, mutable> -> !ttg.memdesc<1xi64, #shared8, #smem7, mutable>
        %full_bar = ttg.memdesc_index %bars_full[%c0_i32] : !ttg.memdesc<1xi64, #shared8, #smem7, mutable> -> !ttg.memdesc<1xi64, #shared8, #smem7, mutable>
        %data_buf = ttg.memdesc_index %data[%c0_i32] : !ttg.memdesc<1x64x64xf16, #shared7, #smem7, mutable> -> !ttg.memdesc<64x64xf16, #shared7, #smem7, mutable>

        ttng.wait_barrier %empty_bar, %phase, %true : !ttg.memdesc<1xi64, #shared8, #smem7, mutable>
        ttng.barrier_expect %full_bar, 8192, %true : !ttg.memdesc<1xi64, #shared8, #smem7, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %data_buf, %full_bar, %true : !tt.tensordesc<tensor<64x64xf16>>, !ttg.memdesc<1xi64, #shared8, #smem7, mutable> -> !ttg.memdesc<64x64xf16, #shared7, #smem7, mutable>

        // Single buffer: phase flips every iteration.
        %one = arith.constant 1 : i32
        %next_phase = arith.xori %phase, %one : i32
        scf.yield %next_phase : i32
      }
      ttg.warp_yield
    }
    partition0(%arg0: i32, %arg1: !ttg.memdesc<1x64x64xf16, #shared7, #smem7, mutable>, %arg2: !ttg.memdesc<1xi64, #shared8, #smem7, mutable>, %arg3: !ttg.memdesc<1xi64, #shared8, #smem7, mutable>, %arg4: !tt.tensordesc<tensor<64x64xf16>>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %t = arith.constant true
      // Consumer: wait(full[0]), arrive(empty[0], count=1) — only 1 of 2 needed!
      %p2 = scf.for %k = %c0 to %arg0 step %c1 iter_args(%phase = %c0) -> (i32) : i32 {
        %full_bar = ttg.memdesc_index %arg3[%c0] : !ttg.memdesc<1xi64, #shared8, #smem7, mutable> -> !ttg.memdesc<1xi64, #shared8, #smem7, mutable>
        %empty_bar = ttg.memdesc_index %arg2[%c0] : !ttg.memdesc<1xi64, #shared8, #smem7, mutable> -> !ttg.memdesc<1xi64, #shared8, #smem7, mutable>

        ttng.wait_barrier %full_bar, %phase, %t : !ttg.memdesc<1xi64, #shared8, #smem7, mutable>
        ttng.arrive_barrier %empty_bar, 1 : !ttg.memdesc<1xi64, #shared8, #smem7, mutable>

        %one = arith.constant 1 : i32
        %next_phase = arith.xori %phase, %one : i32
        scf.yield %next_phase : i32
      }
      ttg.warp_return
    } : (i32, !ttg.memdesc<1x64x64xf16, #shared7, #smem7, mutable>, !ttg.memdesc<1xi64, #shared8, #smem7, mutable>, !ttg.memdesc<1xi64, #shared8, #smem7, mutable>, !tt.tensordesc<tensor<64x64xf16>>) -> ()
    tt.return
  }
}

// -----

// Test: Phase mismatch → SAT (deadlock)
// Consumer uses constant phase=0 (never flips), while producer correctly tracks
// phase. After 2 iterations (buf wraps around), consumer waits on full[0] at
// phase=0 again, but by then the barrier has flipped to phase=1.

#shared9 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared10 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem9 = #ttg.shared_memory

// CHECK-LABEL: === Barrier Deadlock Detection: phase_mismatch ===
// CHECK: Initial arrives (pre-task):
// CHECK:   bars_empty_0: 1
// CHECK:   bars_empty_1: 1
// CHECK: --- Z3 Script ---
// CHECK: # Phi_B: stalled at WAIT -> barrier is blocked
// CHECK: # Phi_R: passed-through WAIT -> barrier was ready when reached
// CHECK: # Deadlock query: at least one task stuck
// CHECK: --- End Z3 Script ---

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @phase_mismatch(%K: i32, %desc: !tt.tensordesc<tensor<64x64xf16>>) {
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32

    %data = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xf16, #shared9, #smem9, mutable>
    %bars_empty = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared10, #smem9, mutable> loc("bars_empty")
    %be0 = ttg.memdesc_index %bars_empty[%c0_i32] : !ttg.memdesc<2xi64, #shared10, #smem9, mutable> -> !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
    ttng.init_barrier %be0, 1 : !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
    %be1 = ttg.memdesc_index %bars_empty[%c1_i32] : !ttg.memdesc<2xi64, #shared10, #smem9, mutable> -> !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
    ttng.init_barrier %be1, 1 : !ttg.memdesc<1xi64, #shared10, #smem9, mutable>

    %bars_full = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared10, #smem9, mutable> loc("bars_full")
    %bf0 = ttg.memdesc_index %bars_full[%c0_i32] : !ttg.memdesc<2xi64, #shared10, #smem9, mutable> -> !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
    ttng.init_barrier %bf0, 1 : !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
    %bf1 = ttg.memdesc_index %bars_full[%c1_i32] : !ttg.memdesc<2xi64, #shared10, #smem9, mutable> -> !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
    ttng.init_barrier %bf1, 1 : !ttg.memdesc<1xi64, #shared10, #smem9, mutable>

    // Pre-arrive empty barriers
    ttng.arrive_barrier %be0, 1 : !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
    ttng.arrive_barrier %be1, 1 : !ttg.memdesc<1xi64, #shared10, #smem9, mutable>

    ttg.warp_specialize(%K, %data, %bars_empty, %bars_full, %desc)
    default {
      // Producer correctly tracks phase
      %p = scf.for %k = %c0_i32 to %K step %c1_i32 iter_args(%phase = %c0_i32) -> (i32) : i32 {
        %buf = arith.remsi %k, %c2_i32 : i32
        %empty_bar = ttg.memdesc_index %bars_empty[%buf] : !ttg.memdesc<2xi64, #shared10, #smem9, mutable> -> !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
        %full_bar = ttg.memdesc_index %bars_full[%buf] : !ttg.memdesc<2xi64, #shared10, #smem9, mutable> -> !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
        %data_buf = ttg.memdesc_index %data[%buf] : !ttg.memdesc<2x64x64xf16, #shared9, #smem9, mutable> -> !ttg.memdesc<64x64xf16, #shared9, #smem9, mutable>

        ttng.wait_barrier %empty_bar, %phase, %true : !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
        ttng.barrier_expect %full_bar, 8192, %true : !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %data_buf, %full_bar, %true : !tt.tensordesc<tensor<64x64xf16>>, !ttg.memdesc<1xi64, #shared10, #smem9, mutable> -> !ttg.memdesc<64x64xf16, #shared9, #smem9, mutable>

        %is_last = arith.cmpi eq, %buf, %c1_i32 : i32
        %is_last_ext = arith.extui %is_last : i1 to i32
        %next_phase = arith.xori %phase, %is_last_ext : i32
        scf.yield %next_phase : i32
      }
      ttg.warp_yield
    }
    partition0(%arg0: i32, %arg1: !ttg.memdesc<2x64x64xf16, #shared9, #smem9, mutable>, %arg2: !ttg.memdesc<2xi64, #shared10, #smem9, mutable>, %arg3: !ttg.memdesc<2xi64, #shared10, #smem9, mutable>, %arg4: !tt.tensordesc<tensor<64x64xf16>>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %t = arith.constant true
      // BUG: Consumer always uses phase=0 (never flips)!
      // After 2 iterations, full[0] parity flips but consumer still expects phase=0.
      %p2 = scf.for %k = %c0 to %arg0 step %c1 iter_args(%unused = %c0) -> (i32) : i32 {
        %buf = arith.remsi %k, %c2 : i32
        %full_bar = ttg.memdesc_index %arg3[%buf] : !ttg.memdesc<2xi64, #shared10, #smem9, mutable> -> !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
        %empty_bar = ttg.memdesc_index %arg2[%buf] : !ttg.memdesc<2xi64, #shared10, #smem9, mutable> -> !ttg.memdesc<1xi64, #shared10, #smem9, mutable>

        // Always wait with phase=0 — BUG!
        ttng.wait_barrier %full_bar, %c0, %t : !ttg.memdesc<1xi64, #shared10, #smem9, mutable>
        ttng.arrive_barrier %empty_bar, 1 : !ttg.memdesc<1xi64, #shared10, #smem9, mutable>

        scf.yield %unused : i32
      }
      ttg.warp_return
    } : (i32, !ttg.memdesc<2x64x64xf16, #shared9, #smem9, mutable>, !ttg.memdesc<2xi64, #shared10, #smem9, mutable>, !ttg.memdesc<2xi64, #shared10, #smem9, mutable>, !tt.tensordesc<tensor<64x64xf16>>) -> ()
    tt.return
  }
}
