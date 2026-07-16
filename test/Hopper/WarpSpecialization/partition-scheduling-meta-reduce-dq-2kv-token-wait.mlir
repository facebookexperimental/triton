// RUN: triton-opt %s --nvgpu-partition-scheduling-meta="merge-correction=false merge-epilogue=false merge-epilogue-to-computation=false merge-reduction=false separate-epilogue-store=false" | FileCheck %s
//
// Minimal regression for the duplicated async_tma_reduce / double-reduce dq
// corruption in the 2-KV HSTU cross-attention backward under AutoWS
// (T279388065). Reduced from the full-attention reproducer to the load/mma/
// reduce/store shape that actually exercises the fix.
//
// A dq TMA-atomic reduce (ttng.async_tma_reduce) lives in the loop; its
// store-completion wait (ttng.async_tma_store_token_wait) consumes the reduce's
// async token, so PSM must co-locate the wait with the reduce in the reduction
// partition. Previously PSM categorized the wait as a generic epilogue store and
// scheduled it into the epilogue partition (where the dk/dv store waits live);
// doCodePartition then cloned the reduce into the epilogue partition to satisfy
// the cross-partition token use -> two async_tma_reduce ops -> dq reduced twice
// -> garbage dq.
//
// Fix (PartitionSchedulingMeta.cpp): isReduceTokenWait() routes a token_wait
// whose token comes from a TMA reduce to its reduce's (reduction) partition via
// categorizeEpilogueStores, and the epilogue scheduling walk skips it.
//
// CHECK-LABEL: @reduce_dq_token_wait
// The dq reduce and its token-wait share the reduction partition...
// CHECK: ttng.async_tma_reduce add{{.*}}ttg.partition = array<i32: [[RED:[0-9]+]]>
// CHECK-NEXT: ttng.async_tma_store_token_wait{{.*}}ttg.partition = array<i32: [[RED]]>
// ...and it is the ONLY reduce (a duplicate would later be cloned into the
// epilogue partition):
// CHECK-NOT: ttng.async_tma_reduce
// The reduction partition is index 0 (the reduce/wait partition above), the
// epilogue is a distinct partition (index 2):
// CHECK: ttg.partition.types = ["reduction", "gemm", "epilogue", "load"]
// The dk epilogue store token-wait lives in that distinct epilogue partition,
// confirming the reduce wait was not simply swept in with the epilogue stores:
// CHECK: ttng.async_tma_store_token_wait{{.*}}ttg.partition = array<i32: 2>

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_dq_token_wait(
      %dq_desc: !tt.tensordesc<tensor<64x128xf32, #shared1>>,
      %dk_desc: !tt.tensordesc<tensor<64x128xbf16, #shared>>,
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xbf16, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32, %off: i32) attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
    %cstk = arith.constant dense<0.000000e+00> : tensor<64x128xbf16, #blocked>
    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %last = scf.for %iv = %lb to %ub step %step iter_args(%tok = %acc_tok) -> (!ttg.async.token) : i32 {
      // dq accumulator MMA (gemm partition) writing the acc tile.
      %mma = ttng.tc_gen5_mma %a, %b, %acc[%tok], %true, %true : !ttg.memdesc<64x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
      // dq TMA-atomic reduce + its store-completion wait: the wait consumes the
      // reduce's token, so PSM must co-locate it with the reduce (reduction).
      %alloc = ttg.local_alloc %cst : (tensor<64x128xf32, #blocked1>) -> !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
      %rtok = ttng.async_tma_reduce add, %dq_desc[%off, %off] %alloc : !tt.tensordesc<tensor<64x128xf32, #shared1>>, !ttg.memdesc<64x128xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %rtok : !ttg.async.token
      scf.yield %mma : !ttg.async.token
    } {tt.warp_specialize}
    // dk epilogue store + its wait: token comes from a plain store, so it stays
    // in the distinct epilogue partition.
    %dk_smem = ttg.local_alloc %cstk : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
    %stok = ttng.async_tma_copy_local_to_global %dk_desc[%off, %off] %dk_smem : !tt.tensordesc<tensor<64x128xbf16, #shared>>, !ttg.memdesc<64x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %stok : !ttg.async.token
    tt.return
  }
}
