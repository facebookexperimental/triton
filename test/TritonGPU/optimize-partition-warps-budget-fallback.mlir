// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-optimize-partition-warps 2>&1 | FileCheck %s

// Test that when the warp budget is exceeded and shrinking cannot bring the
// schedule under budget (partitions are at their minimums), the WarpSpecializeOp
// is removed and the default region is inlined — falling back to non-warp-
// specialized code.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory

// With num_warps=8 and maxnreg=128:
//   budget = 65536 / 128 / 32 = 16 warps
//   schedule = 8 (default) + comp0(4) + comp1(4) + scalar(1) + scalar(1)
//   padded = 8 + ceil(10/4)*4 = 8 + 12 = 20 > 16
// Shrinking can't help: comp0 and comp1 have TMEM ops (minimum 4 warps).
// The WarpSpecializeOp should be removed.

// CHECK-LABEL: @budget_exceeded_fallback
// The warp_specialize op should be gone — check that its ops are NOT present.
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: ttg.warp_yield
// CHECK-NOT: ttg.warp_return
// The default region's ops should be inlined.
// CHECK: arith.constant 42
// CHECK: tt.return
module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 8 : i32, "ttg.maxnreg" = 128 : i32} {

tt.func @budget_exceeded_fallback(%arg0: i32) {
  ttg.warp_specialize(%arg0)
    attributes {"ttg.partition.types" = ["default", "gemm", "load", "computation", "computation"]}
  default {
    %c = arith.constant 42 : i32
    ttg.warp_yield
  }
  // comp0: has TMEM op, minimum 4 warps, can't shrink below 4
  partition0(%a0: i32) num_warps(4) {
    %alloc, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    ttg.warp_return
  }
  // comp1: has TMEM op, minimum 4 warps, can't shrink below 4
  partition1(%a1: i32) num_warps(4) {
    %alloc, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    ttg.warp_return
  }
  // scalar partitions, shrink to 1 each
  partition2(%a2: i32) num_warps(4) {
    %0 = arith.addi %a2, %a2 : i32
    ttg.warp_return
  }
  partition3(%a3: i32) num_warps(4) {
    %0 = arith.muli %a3, %a3 : i32
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

// Verify that a schedule within budget is NOT removed.
// With num_warps=8 and maxnreg=128: budget=16
// schedule = 8 (default) + comp(4) + scalar(1) = 13
// padded = 8 + ceil(5/4)*4 = 8 + 8 = 16, fits.

// CHECK-LABEL: @budget_within_limits
// CHECK: ttg.warp_specialize
// CHECK: default {
// CHECK: partition0
// CHECK: partition1
tt.func @budget_within_limits(%arg0: i32) {
  ttg.warp_specialize(%arg0)
    attributes {"ttg.partition.types" = ["default", "gemm", "computation"]}
  default {
    ttg.warp_yield
  }
  partition0(%a0: i32) num_warps(4) {
    %0 = arith.addi %a0, %a0 : i32
    ttg.warp_return
  }
  partition1(%a1: i32) num_warps(4) {
    %alloc, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

}
