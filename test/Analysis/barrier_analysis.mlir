// RUN: triton-opt %s -triton-print-barrier-analysis 2>&1 | FileCheck %s

// Test basic barrier analysis pass

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:89"} {

// CHECK: ========================================
// CHECK: Barrier Execution Order Analysis
// CHECK: ========================================

tt.func @test_barrier_analysis(%arg0: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true

  // Allocate barrier
  %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>

  // Initialize barrier
  ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

  // Set expected bytes
  ttng.barrier_expect %barrier, 256, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>

  // Wait on barrier
  ttng.wait_barrier %barrier, %c0_i32 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

  // Arrive
  ttng.arrive_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

  // Invalidate barrier
  ttng.inval_barrier %barrier : !ttg.memdesc<1xi64, #shared, #smem, mutable>

  tt.return
}

// CHECK: Barrier Operations by Warp Group:
// CHECK: init_barrier
// CHECK: barrier_expect
// CHECK: wait_barrier
// CHECK: arrive_barrier

// Test named barriers
// CHECK: Execution Trace Visualization

tt.func @test_named_barriers() {
  %c9 = arith.constant 9 : i32
  %c10 = arith.constant 10 : i32
  %c256 = arith.constant 256 : i32

  // Named barrier arrive
  ttng.arrive_barrier_named %c9, %c256 : i32, i32

  // Named barrier wait
  ttng.wait_barrier_named %c10, %c256 : i32, i32

  tt.return
}

}

// -----

// Test barrier analysis with WarpSpecializeOp - tests cross-region barrier tracing
// This exercises the getBarrierAllocAndIndex and traceValueThroughBlockArgs functions

#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem2 = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:100"} {

// CHECK: === Barrier Analysis for function: test_warp_specialize_barriers ===
// CHECK: Barrier Operations by Warp Group:
// CHECK: [main] (Producer)
// CHECK-DAG: init_barrier (bar=0)
// CHECK-DAG: init_barrier (bar=1)
// CHECK: [WG0] (Producer) (Consumer)
// The default region operations are in WG0, partition0 operations are also
// currently grouped in WG0 (the async_task_id logic assigns based on region index).
// Key test: barrier IDs are correctly traced across regions
// CHECK-DAG: wait_barrier (bar=0
// CHECK-DAG: arrive_barrier (bar=1)
// CHECK-DAG: wait_barrier (bar=1
// CHECK: Dependencies:
// CHECK-DAG: init_barrier{{.*}} --[init->use]--> wait_barrier{{.*}}(WG0)
// CHECK-DAG: init_barrier{{.*}} --[init->use]--> arrive_barrier{{.*}}(WG0)
// CHECK-DAG: arrive_barrier{{.*}}(WG0) --[arrive->wait]--> wait_barrier{{.*}}(WG0)

tt.func @test_warp_specialize_barriers() {
  %c0_i32 = arith.constant 0 : i32
  %true = arith.constant true

  // Allocate two barriers in the main function
  %bar0_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared2, #smem2, mutable>
  %bar0 = ttg.memdesc_index %bar0_alloc[%c0_i32] : !ttg.memdesc<1x1xi64, #shared2, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem2, mutable>
  ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #shared2, #smem2, mutable>

  %bar1_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared2, #smem2, mutable>
  %bar1 = ttg.memdesc_index %bar1_alloc[%c0_i32] : !ttg.memdesc<1x1xi64, #shared2, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem2, mutable>
  ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #shared2, #smem2, mutable>

  gpu.barrier

  // Pass barriers to warp_specialize - this tests cross-region barrier tracing
  // bar0_alloc and bar1_alloc are passed as explicit captures
  ttg.warp_specialize(%bar0_alloc, %bar1_alloc)
  default {
    // In the default region, we create new memdesc_index ops referencing the captured barriers
    // This tests that the analysis can match these to the original init_barrier ops
    %bar0_ref = ttg.memdesc_index %bar0_alloc[%c0_i32] : !ttg.memdesc<1x1xi64, #shared2, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem2, mutable>
    ttng.wait_barrier %bar0_ref, %c0_i32, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared2, #smem2, mutable>

    %bar1_ref = ttg.memdesc_index %bar1_alloc[%c0_i32] : !ttg.memdesc<1x1xi64, #shared2, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem2, mutable>
    ttng.arrive_barrier %bar1_ref, 1, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared2, #smem2, mutable>

    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<1x1xi64, #shared2, #smem2, mutable>, %arg1: !ttg.memdesc<1x1xi64, #shared2, #smem2, mutable>) num_warps(2) {
    // In partition regions, barriers are block arguments
    // This tests that traceValueThroughBlockArgs can trace back to original allocations
    // Note: partition regions are isolated, so we must define constants locally
    %c0 = arith.constant 0 : i32
    %bar1_part = ttg.memdesc_index %arg1[%c0] : !ttg.memdesc<1x1xi64, #shared2, #smem2, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem2, mutable>
    ttng.wait_barrier %bar1_part, %c0 : !ttg.memdesc<1xi64, #shared2, #smem2, mutable>
    ttg.warp_return
  }
  partition1(%arg0: !ttg.memdesc<1x1xi64, #shared2, #smem2, mutable>, %arg1: !ttg.memdesc<1x1xi64, #shared2, #smem2, mutable>) num_warps(2) {
    ttg.warp_return
  } : (!ttg.memdesc<1x1xi64, #shared2, #smem2, mutable>, !ttg.memdesc<1x1xi64, #shared2, #smem2, mutable>) -> ()

  tt.return
}

}
