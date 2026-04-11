// RUN: triton-opt %s -split-input-file -test-print-barrier-analysis="unroll-bound=5" 2>&1 | FileCheck %s

// Test: Simple producer-consumer pipeline with 4 stages.
// Producer: wait(empty), expect(full), tma_load(full)
// Consumer: wait(full), arrive(empty)

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: === Barrier Analysis: simple_pipeline ===
// CHECK: Barrier allocations:
// CHECK:   bars_empty: 4 slots, arrive_count=1
// CHECK:   bars_full: 4 slots, arrive_count=1
// CHECK: Task traces:
// CHECK:   default (task 0): 15 barrier ops
// CHECK:     [0] wait_barrier bars_empty[0] phase=0 (iter 0)
// CHECK:     [1] barrier_expect bars_full[0] bytes=8192 (iter 0)
// CHECK:     [2] tma_load bars_full[0] (iter 0)
// CHECK:     [3] wait_barrier bars_empty[1] phase=0 (iter 1)
// CHECK:     [4] barrier_expect bars_full[1] bytes=8192 (iter 1)
// CHECK:     [5] tma_load bars_full[1] (iter 1)
// CHECK:     [6] wait_barrier bars_empty[2] phase=0 (iter 2)
// CHECK:     [7] barrier_expect bars_full[2] bytes=8192 (iter 2)
// CHECK:     [8] tma_load bars_full[2] (iter 2)
// CHECK:     [9] wait_barrier bars_empty[3] phase=0 (iter 3)
// CHECK:     [10] barrier_expect bars_full[3] bytes=8192 (iter 3)
// CHECK:     [11] tma_load bars_full[3] (iter 3)
// CHECK:     [12] wait_barrier bars_empty[0] phase=1 (iter 4)
// CHECK:     [13] barrier_expect bars_full[0] bytes=8192 (iter 4)
// CHECK:     [14] tma_load bars_full[0] (iter 4)
// CHECK:   partition0 (task 1): 10 barrier ops
// CHECK:     [0] wait_barrier bars_full[0] phase=0 (iter 0)
// CHECK:     [1] arrive_barrier bars_empty[0] (iter 0)
// CHECK:     [2] wait_barrier bars_full[1] phase=0 (iter 1)
// CHECK:     [3] arrive_barrier bars_empty[1] (iter 1)
// CHECK:     [4] wait_barrier bars_full[2] phase=0 (iter 2)
// CHECK:     [5] arrive_barrier bars_empty[2] (iter 2)
// CHECK:     [6] wait_barrier bars_full[3] phase=0 (iter 3)
// CHECK:     [7] arrive_barrier bars_empty[3] (iter 3)
// CHECK:     [8] wait_barrier bars_full[0] phase=1 (iter 4)
// CHECK:     [9] arrive_barrier bars_empty[0] (iter 4)

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @simple_pipeline(%K: i32, %desc: !tt.tensordesc<tensor<64x64xf16>>) {
    %true = arith.constant true
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32

    %data = ttg.local_alloc : () -> !ttg.memdesc<4x64x64xf16, #shared, #smem, mutable>
    %bars_empty = ttg.local_alloc : () -> !ttg.memdesc<4xi64, #shared1, #smem, mutable> loc("bars_empty")
    %be0 = ttg.memdesc_index %bars_empty[%c0_i32] : !ttg.memdesc<4xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %be0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %be1 = ttg.memdesc_index %bars_empty[%c1_i32] : !ttg.memdesc<4xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %be1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %be2 = ttg.memdesc_index %bars_empty[%c2_i32] : !ttg.memdesc<4xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %be2, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %be3 = ttg.memdesc_index %bars_empty[%c3_i32] : !ttg.memdesc<4xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %be3, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    %bars_full = ttg.local_alloc : () -> !ttg.memdesc<4xi64, #shared1, #smem, mutable> loc("bars_full")
    %bf0 = ttg.memdesc_index %bars_full[%c0_i32] : !ttg.memdesc<4xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bf0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %bf1 = ttg.memdesc_index %bars_full[%c1_i32] : !ttg.memdesc<4xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bf1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %bf2 = ttg.memdesc_index %bars_full[%c2_i32] : !ttg.memdesc<4xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bf2, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %bf3 = ttg.memdesc_index %bars_full[%c3_i32] : !ttg.memdesc<4xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bf3, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    // Pre-arrive empty barriers so producer can start (buffers are initially empty/available)
    ttng.arrive_barrier %be0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %be1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %be2, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %be3, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    ttg.warp_specialize(%K, %data, %bars_empty, %bars_full, %desc)
    default {
      // Producer: wait(empty[buf]), expect(full[buf]), tma_load(full[buf])
      %p = scf.for %k = %c0_i32 to %K step %c1_i32 iter_args(%phase = %c0_i32) -> (i32) : i32 {
        %buf = arith.remsi %k, %c4_i32 : i32
        %empty_bar = ttg.memdesc_index %bars_empty[%buf] : !ttg.memdesc<4xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %full_bar = ttg.memdesc_index %bars_full[%buf] : !ttg.memdesc<4xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %data_buf = ttg.memdesc_index %data[%buf] : !ttg.memdesc<4x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

        ttng.wait_barrier %empty_bar, %phase, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.barrier_expect %full_bar, 8192, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %data_buf, %full_bar, %true : !tt.tensordesc<tensor<64x64xf16>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

        // phase ^= (buf == 3)
        %is_last = arith.cmpi eq, %buf, %c3_i32 : i32
        %is_last_ext = arith.extui %is_last : i1 to i32
        %next_phase = arith.xori %phase, %is_last_ext : i32
        scf.yield %next_phase : i32
      }
      ttg.warp_yield
    }
    partition0(%arg0: i32, %arg1: !ttg.memdesc<4x64x64xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<4xi64, #shared1, #smem, mutable>, %arg3: !ttg.memdesc<4xi64, #shared1, #smem, mutable>, %arg4: !tt.tensordesc<tensor<64x64xf16>>) num_warps(4) {
      // Consumer: wait(full[buf]), arrive(empty[buf])
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c3 = arith.constant 3 : i32
      %c4 = arith.constant 4 : i32
      %t = arith.constant true
      %p2 = scf.for %k = %c0 to %arg0 step %c1 iter_args(%phase = %c0) -> (i32) : i32 {
        %buf = arith.remsi %k, %c4 : i32
        %full_bar = ttg.memdesc_index %arg3[%buf] : !ttg.memdesc<4xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %empty_bar = ttg.memdesc_index %arg2[%buf] : !ttg.memdesc<4xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>

        ttng.wait_barrier %full_bar, %phase, %t : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.arrive_barrier %empty_bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

        %is_last = arith.cmpi eq, %buf, %c3 : i32
        %is_last_ext = arith.extui %is_last : i1 to i32
        %next_phase = arith.xori %phase, %is_last_ext : i32
        scf.yield %next_phase : i32
      }
      ttg.warp_return
    } : (i32, !ttg.memdesc<4x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<4xi64, #shared1, #smem, mutable>, !ttg.memdesc<4xi64, #shared1, #smem, mutable>, !tt.tensordesc<tensor<64x64xf16>>) -> ()
    tt.return
  }
}

// -----

// Test: Phase tracking correctness with 2 stages (phases flip every 2 iterations).

#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem2 = #ttg.shared_memory

// CHECK-LABEL: === Barrier Analysis: two_stage_pipeline ===
// CHECK: Barrier allocations:
// CHECK:   bars: 2 slots, arrive_count=1
// CHECK: Task traces:
// CHECK:   default (task 0): 5 barrier ops
// Phase flips at buf==1, so: iter0=phase0, iter1=phase0->flip, iter2=phase1, iter3=phase1->flip, iter4=phase0
// CHECK:     [0] wait_barrier bars[0] phase=0 (iter 0)
// CHECK:     [1] wait_barrier bars[1] phase=0 (iter 1)
// CHECK:     [2] wait_barrier bars[0] phase=1 (iter 2)
// CHECK:     [3] wait_barrier bars[1] phase=1 (iter 3)
// CHECK:     [4] wait_barrier bars[0] phase=0 (iter 4)

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @two_stage_pipeline(%K: i32) {
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32

    %bars = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared3, #smem2, mutable> loc("bars")
    %b0 = ttg.memdesc_index %bars[%c0_i32] : !ttg.memdesc<2xi64, #shared3, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
    ttng.init_barrier %b0, 1 : !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
    %b1 = ttg.memdesc_index %bars[%c1_i32] : !ttg.memdesc<2xi64, #shared3, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
    ttng.init_barrier %b1, 1 : !ttg.memdesc<1xi64, #shared3, #smem2, mutable>

    ttg.warp_specialize(%K, %bars)
    default {
      %p = scf.for %k = %c0_i32 to %K step %c1_i32 iter_args(%phase = %c0_i32) -> (i32) : i32 {
        %buf = arith.remsi %k, %c2_i32 : i32
        %bar = ttg.memdesc_index %bars[%buf] : !ttg.memdesc<2xi64, #shared3, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
        ttng.wait_barrier %bar, %phase, %true : !ttg.memdesc<1xi64, #shared3, #smem2, mutable>
        // phase ^= (buf == 1)
        %is_last = arith.cmpi eq, %buf, %c1_i32 : i32
        %is_last_ext = arith.extui %is_last : i1 to i32
        %next_phase = arith.xori %phase, %is_last_ext : i32
        scf.yield %next_phase : i32
      }
      ttg.warp_yield
    } : (i32, !ttg.memdesc<2xi64, #shared3, #smem2, mutable>) -> ()
    tt.return
  }
}
