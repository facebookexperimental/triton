// RUN: triton-opt %s -split-input-file --allocate-shared-memory -test-print-membar | FileCheck %s

// An elected notification (tcgen05 commit) that follows an mbarrier wait in the
// same block needs a rendezvous first: otherwise the elected warp can let a
// peer advance the barrier's phase before the other warps observe the wait.

#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @wait_then_commit
tt.func @wait_then_commit(%phase: i32) {
  %full = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  // CHECK: ttng.wait_barrier
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_commit
  ttng.wait_barrier %full, %phase : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.tc_gen5_commit %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}

// A barrier already between the wait and the commit: no extra one.
// CHECK-LABEL: @wait_barrier_then_commit
tt.func @wait_barrier_then_commit(%phase: i32) {
  %full = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  // CHECK: ttng.wait_barrier
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_commit
  ttng.wait_barrier %full, %phase : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttg.barrier local
  ttng.tc_gen5_commit %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}

// No wait before the commit: no barrier.
// CHECK-LABEL: @commit_without_wait
tt.func @commit_without_wait() {
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  // CHECK-NOT: ttg.barrier
  // CHECK: ttng.tc_gen5_commit
  ttng.tc_gen5_commit %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}
}

// -----

#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Single-warp functions have nothing to rendezvous with.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_warp_wait_then_commit
tt.func @one_warp_wait_then_commit(%phase: i32) {
  %full = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  %empty = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  // CHECK: ttng.wait_barrier
  // CHECK-NEXT: ttng.tc_gen5_commit
  ttng.wait_barrier %full, %phase : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  ttng.tc_gen5_commit %empty : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
  tt.return
}
}

// -----

// The rule also covers an MMA that signals completion barriers, and not one
// without them.

#sharedA = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#sharedB = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
// CHECK-LABEL: @wait_then_mma_with_barrier
tt.func @wait_then_mma_with_barrier(%phase: i32, %a: !ttg.memdesc<128x16xf16, #sharedA, #smem, mutable>,
    %b: !ttg.memdesc<16x128xf16, #sharedB, #smem, mutable>,
    %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    %b0: !ttg.memdesc<1xi64, #bar, #smem, mutable>, %b1: !ttg.memdesc<1xi64, #bar, #smem, mutable>) {
  %true = arith.constant true
  // CHECK: ttng.wait_barrier
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_mma
  ttng.wait_barrier %b0, %phase : !ttg.memdesc<1xi64, #bar, #smem, mutable>
  ttng.tc_gen5_mma %a, %b, %acc, %true, %true, %b1[%true] {is_async} :
     !ttg.memdesc<128x16xf16, #sharedA, #smem, mutable>,
     !ttg.memdesc<16x128xf16, #sharedB, #smem, mutable>,
     !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
     !ttg.memdesc<1xi64, #bar, #smem, mutable>
  tt.return
}

// CHECK-LABEL: @wait_then_mma_without_barrier
tt.func @wait_then_mma_without_barrier(%phase: i32, %a: !ttg.memdesc<128x16xf16, #sharedA, #smem, mutable>,
    %b: !ttg.memdesc<16x128xf16, #sharedB, #smem, mutable>,
    %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    %b0: !ttg.memdesc<1xi64, #bar, #smem, mutable>) {
  %true = arith.constant true
  // CHECK: ttng.wait_barrier
  // CHECK-NEXT: ttng.tc_gen5_mma
  ttng.wait_barrier %b0, %phase : !ttg.memdesc<1xi64, #bar, #smem, mutable>
  ttng.tc_gen5_mma %a, %b, %acc, %true, %true :
     !ttg.memdesc<128x16xf16, #sharedA, #smem, mutable>,
     !ttg.memdesc<16x128xf16, #sharedB, #smem, mutable>,
     !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}
}
