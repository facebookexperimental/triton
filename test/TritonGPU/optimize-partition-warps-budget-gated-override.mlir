// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-optimize-partition-warps | FileCheck %s

// Test that the BWD FA type-aware override (computation=8 warps) is
// budget-gated when maxnreg is set.
//
// With num_warps=8 and maxnreg=128:
//   budget = 65536 / 128 / 32 = 16 warps
//   BWD override would give: 8 (default) + 1 + 1 + 8 = 18, padded to 20
//   That exceeds the budget, so the override should be skipped.
//   Without override: 8 + 1 + 1 + 4 = 14, padded to 16 — fits.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory

// CHECK-LABEL: @bwd_override_budget_gated
// computation partition stays at 4 (not overridden to 8) because budget is tight
// CHECK: partition0({{.*}}) num_warps(1)
// CHECK: partition1({{.*}}) num_warps(1)
// CHECK: partition2({{.*}}) num_warps(4)
module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 8 : i32, "ttg.maxnreg" = 128 : i32} {

tt.func @bwd_override_budget_gated(%arg0: i32) {
  ttg.warp_specialize(%arg0)
    attributes {"ttg.partition.types" = ["reduction", "gemm", "load", "computation"]}
  default {
    ttg.warp_yield
  }
  // gemm: scalar, shrinks to 1
  partition0(%arg1: i32) num_warps(8) {
    %0 = arith.addi %arg1, %arg1 : i32
    ttg.warp_return
  }
  // load: scalar, shrinks to 1
  partition1(%arg1: i32) num_warps(8) {
    %0 = arith.muli %arg1, %arg1 : i32
    ttg.warp_return
  }
  // computation: has TMEM op (minimum 4 warps). Would be overridden to 8
  // by BWD pattern, but budget enforcement keeps it at 4.
  partition2(%arg1: i32) num_warps(4) {
    %alloc, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

}
