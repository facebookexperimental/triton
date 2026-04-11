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
