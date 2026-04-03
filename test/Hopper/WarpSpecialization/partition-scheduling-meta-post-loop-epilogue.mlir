// RUN: triton-opt %s --nvgpu-partition-scheduling-meta | FileCheck %s

// Tests that post-loop tmem_load and arithmetic ops are scheduled to the
// default partition (not the epilogue), while only epilogue store ops go to
// the epilogue partition. This prevents TMEM ops from landing in the epilogue,
// which would force it to use 4 warps (TMEM lane coverage hardware constraint).
//
// Before the fix, schedulePostLoopOps put ALL post-loop consumers of loop
// results into the epilogue, including tmem_load (accumulator reads). This
// forced the epilogue to 4 warps, causing non-persistent FA forward to exceed
// the 512-thread hardware limit (20 warps × 32 = 640 > 512).

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @post_loop_tmem_load_not_in_epilogue
//
// --- Pre-loop: acc inits → epilogue partition (no default partition) ---
// CHECK: ttng.tmem_store {{.*}}ttg.partition = array<i32: [[EPIL:[0-9]+]]>
// CHECK: ttng.tmem_store {{.*}}ttg.partition = array<i32: [[EPIL]]>
//
// --- In-loop: loads → load partition ---
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD:[0-9]+]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[LOAD]]>
// --- In-loop: memdesc_trans and MMAs → gemm partition ---
// CHECK: ttg.memdesc_trans {{.*}}ttg.partition = array<i32: [[GEMM:[0-9]+]]>
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[GEMM]]>
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[GEMM]]>
// --- In-loop: correction ops → computation partition ---
// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[COMP:[0-9]+]]>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttng.tmem_store {{.*}}ttg.partition = array<i32: [[COMP]]>
//
// --- Partition types ---
// CHECK: tt.warp_specialize
// CHECK-SAME: ttg.partition.types = ["gemm", "load", "epilogue", "computation"]
//
// --- Post-loop: tmem_load → epilogue ---
// CHECK: ttng.tmem_load
// CHECK-SAME: ttg.partition = array<i32: [[EPIL]]>
// --- Post-loop: truncf → epilogue ---
// CHECK: arith.truncf
// CHECK-SAME: ttg.partition = array<i32: [[EPIL]]>
// --- Post-loop: local_alloc → epilogue ---
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[EPIL]]>
// --- Post-loop: TMA store → epilogue partition ---
// CHECK: ttng.async_tma_copy_local_to_global
// CHECK-SAME: ttg.partition = array<i32: [[EPIL]]>
// CHECK: ttng.async_tma_store_token_wait
// CHECK-SAME: ttg.partition = array<i32: [[EPIL]]>
tt.func public @post_loop_tmem_load_not_in_epilogue(
  %A_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %B_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %C_desc: !tt.tensordesc<tensor<128x128xf16, #shared>>,
  %k_tiles: i32
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>

  // Accumulators for two data-partitioned MMAs
  %acc0_mem, %acc0_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
  %acc0_tok2 = ttng.tmem_store %cst, %acc0_mem[%acc0_tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %acc1_mem, %acc1_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
  %acc1_tok2 = ttng.tmem_store %cst, %acc1_mem[%acc1_tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

  // Inner KV loop (non-persistent FA forward pattern) with correction ops.
  // Two MMAs + their results are yielded AND have non-yield users that feed
  // the yield (accumulator rescaling), which triggers hasCorrection → UnifiedFA.
  %loop_out:4 = scf.for %i = %c0_i32 to %k_tiles step %c1_i32
      iter_args(%use_acc = %false, %loop_tok0 = %acc0_tok2, %loop_tok1 = %acc1_tok2,
                %prev_scale = %cst) -> (i1, !ttg.async.token, !ttg.async.token,
                tensor<128x128xf32, #blocked>) : i32 {
    %offs_k = arith.muli %i, %c64_i32 : i32

    // Load A
    %a0 = tt.descriptor_load %A_desc[%c0_i32, %offs_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
    %a0_smem = ttg.local_alloc %a0 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

    // Load B
    %b = tt.descriptor_load %B_desc[%c0_i32, %offs_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
    %b_smem = ttg.local_alloc %b : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_trans = ttg.memdesc_trans %b_smem {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>

    // MMA 0
    %mma_tok0 = ttng.tc_gen5_mma %a0_smem, %b_trans, %acc0_mem[%loop_tok0], %use_acc, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // MMA 1 (second data partition)
    %mma_tok1 = ttng.tc_gen5_mma %a0_smem, %b_trans, %acc1_mem[%loop_tok1], %use_acc, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // Correction: read MMA result, compute rescaling, yield back
    // (This is the online softmax pattern that triggers hasCorrection)
    %mma_result, %mma_result_tok = ttng.tmem_load %acc0_mem[%mma_tok0] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %scale = arith.mulf %mma_result, %prev_scale : tensor<128x128xf32, #blocked>
    %store_tok = ttng.tmem_store %scale, %acc0_mem[%mma_result_tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    scf.yield %true, %store_tok, %mma_tok1, %scale : i1, !ttg.async.token, !ttg.async.token, tensor<128x128xf32, #blocked>
  } {tt.warp_specialize}

  // Post-loop epilogue: tmem_load → truncf → TMA store
  // The tmem_load should go to default partition (not epilogue)
  // Only the TMA store should go to epilogue partition
  %result, %result_tok = ttng.tmem_load %acc0_mem[%loop_out#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
  %result_f16 = arith.truncf %result : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
  %result_smem = ttg.local_alloc %result_f16 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %store_tok = ttng.async_tma_copy_local_to_global %C_desc[%c0_i32, %c0_i32] %result_smem : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
  ttng.async_tma_store_token_wait %store_tok : !ttg.async.token

  tt.return
}

}
