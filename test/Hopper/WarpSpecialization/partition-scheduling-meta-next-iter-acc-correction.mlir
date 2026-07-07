// RUN: triton-opt %s --nvgpu-partition-scheduling-meta | FileCheck %s --check-prefix=SPLIT
// RUN: triton-opt %s --nvgpu-partition-scheduling-meta="merge-correction" | FileCheck %s --check-prefix=MERGE

// Correction-trigger characterization for the HSTU reduce_dq "compute fold".
//
// When an accumulating MMA folds a second gradient into a loop-carried TMEM
// accumulator (dk = tl.dot(dqk, q, dk) with use_acc=true, dk carried across the
// KV loop), the MMA's result token is yielded, and the NEXT iteration reads that
// accumulator with a tmem_load at the loop top. categorizeCorrectionOps() flags
// that next-iteration read of a yielded MMA result as a Correction op:
//   for each MMA whose result is yielded, every uncategorized user of the
//   matching region iter-arg becomes Correction (dpId inherited from the MMA).
//
// SPLIT (default): the read lands in a dedicated "correction" partition.
// MERGE (merge-correction, as set on the outer AUTOWS KV loop): the same read is
// folded into the computation partition and NO correction partition is created.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// SPLIT-LABEL: @next_iter_acc_load_is_correction
// SPLIT: tt.warp_specialize
// SPLIT-SAME: ttg.partition.types = [{{.*}}"correction"{{.*}}]

// MERGE-LABEL: @next_iter_acc_load_is_correction
// MERGE: tt.warp_specialize
// MERGE-NOT: "correction"
tt.func public @next_iter_acc_load_is_correction(
  %A_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %B_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %out: !tt.ptr<f32>,
  %k_tiles: i32
) {
  %true = arith.constant true
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
  %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked>

  // Loop-carried dk accumulator (allocated once, carried by token).
  %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
  %acc_init = ttng.tmem_store %cst, %acc[%acc_tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

  %loop:1 = scf.for %i = %c0_i32 to %k_tiles step %c1_i32
      iter_args(%tok = %acc_init) -> (!ttg.async.token) : i32 {
    %offs_k = arith.muli %i, %c64_i32 : i32

    // NEXT-ITERATION read of the loop-carried accumulator (correction target):
    // %tok is the region iter-arg carrying the previous iteration's MMA result.
    %ld, %ld_tok = ttng.tmem_load %acc[%tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.store %ptrs, %ld : tensor<128x128x!tt.ptr<f32>, #blocked>

    // Operands (load partition).
    %a = tt.descriptor_load %A_desc[%c0_i32, %offs_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
    %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b = tt.descriptor_load %B_desc[%c0_i32, %offs_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
    %b_smem = ttg.local_alloc %b : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_trans = ttg.memdesc_trans %b_smem {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>

    // Accumulating MMA into the loop-carried accumulator; token is yielded.
    %mma_tok = ttng.tc_gen5_mma %a_smem, %b_trans, %acc[%ld_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    scf.yield %mma_tok : !ttg.async.token
  } {tt.warp_specialize}

  tt.return
}

}
