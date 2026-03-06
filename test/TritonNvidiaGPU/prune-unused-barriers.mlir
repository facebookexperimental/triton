// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-prune-unused-barriers | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// Test 1: Barrier with only init (no waits) should be fully pruned.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @prune_init_only
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttng.init_barrier
  // CHECK: tt.return
  tt.func @prune_init_only() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// Test 2: Barrier with init + arrive (no waits) should be fully pruned.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @prune_init_arrive
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttng.init_barrier
  // CHECK-NOT: ttng.arrive_barrier
  // CHECK-NOT: ttng.inval_barrier
  // CHECK: tt.return
  tt.func @prune_init_arrive(%pred: i1) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.arrive_barrier %bar, 1, %pred : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// Test 3: Barrier with init + wait should NOT be pruned.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @keep_barrier_with_wait
  // CHECK: ttg.local_alloc
  // CHECK: ttng.init_barrier
  // CHECK: ttng.wait_barrier
  // CHECK: ttng.inval_barrier
  // CHECK: tt.return
  tt.func @keep_barrier_with_wait() {
    %c0 = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.wait_barrier %bar, %c0 : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// Test 4: Barrier with init + expect + commit (no waits) should be fully pruned.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @prune_init_expect_commit
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttng.init_barrier
  // CHECK-NOT: ttng.barrier_expect
  // CHECK-NOT: ttng.tc_gen5_commit
  // CHECK-NOT: ttng.inval_barrier
  // CHECK: tt.return
  tt.func @prune_init_expect_commit(%pred: i1) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.barrier_expect %bar, 16384, %pred : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.tc_gen5_commit %bar, %pred : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// Test 5: Barrier used by MMA (no waits) should be disconnected from MMA.
// The MMA should become synchronous (is_async removed).
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @prune_mma_barrier
  // CHECK-NOT: ttg.local_alloc {{.*}}1xi64
  // CHECK-NOT: ttng.init_barrier
  // CHECK: ttng.tc_gen5_mma
  // CHECK-NOT: is_async
  // CHECK: tt.return
  tt.func @prune_mma_barrier(
      %a: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared, #smem>,
      %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %useD: i1, %pred: i1) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.tc_gen5_mma %a, %b, %c, %useD, %pred, %bar[%pred] {is_async} :
       !ttg.memdesc<128x64xf16, #shared, #smem>,
       !ttg.memdesc<64x128xf16, #shared, #smem>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// Test 6: Barrier used by TMEMCopy (no waits) should be disconnected.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @prune_tmem_copy_barrier
  // CHECK-NOT: ttg.local_alloc {{.*}}1xi64
  // CHECK-NOT: ttng.init_barrier
  // CHECK: ttng.tmem_copy
  // CHECK-NOT: !ttg.memdesc<1xi64
  // CHECK: tt.return
  tt.func @prune_tmem_copy_barrier(
      %src: !ttg.memdesc<128x128xf32, #shared, #smem>,
      %dst: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    ttng.tmem_copy %src, %dst, %bar : !ttg.memdesc<128x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    tt.return
  }
}
