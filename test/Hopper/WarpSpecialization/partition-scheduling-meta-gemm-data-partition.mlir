// RUN: triton-opt %s --nvgpu-partition-scheduling-meta="separate-epilogue-store" | FileCheck %s

// Tests that when #MMAs == data_partition_factor, the GEMM template is selected
// (not UnifiedFA). With dpFactor=2 and BLOCK_SIZE_M=256, the accumulator is
// split into two 128x128 halves, each with its own MMA — a pure data-partitioned
// GEMM, not flash attention.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @data_partitioned_gemm_uses_gemm_template
//
// --- Pre-loop: acc inits → epilogue partition (no default partition) ---
// CHECK: ttng.tmem_store {{.*}}ttg.partition = array<i32: [[EPIL:[0-9]+]]>
// CHECK: ttng.tmem_store {{.*}}ttg.partition = array<i32: [[EPIL]]>
//
// --- Inner k-loop: all descriptor_loads and local_allocs → load partition ---
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD:[0-9]+]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[LOAD]]>
// --- Inner k-loop: memdesc_trans and both MMAs → gemm partition ---
// CHECK: ttg.memdesc_trans {{.*}}ttg.partition = array<i32: [[GEMM:[0-9]+]]>
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[GEMM]]>
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[GEMM]]>
//
// --- Epilogue: tmem_load, truncf, local_alloc → computation partition ---
// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[COMP:[0-9]+]]>
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[COMP]]>
// --- Epilogue: TMA store → epilogue partition ---
// CHECK: ttng.async_tma_copy_local_to_global {{.*}}ttg.partition = array<i32: [[EPIL_STORE:[0-9]+]]>
// CHECK: ttng.async_tma_store_token_wait {{.*}}ttg.partition = array<i32: [[EPIL_STORE]]>
// --- Second half: tmem_load, truncf, local_alloc → computation; TMA store → epilogue ---
// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttng.async_tma_copy_local_to_global {{.*}}ttg.partition = array<i32: [[EPIL_STORE]]>
// CHECK: ttng.async_tma_store_token_wait {{.*}}ttg.partition = array<i32: [[EPIL_STORE]]>
//
// --- Partition types ---
// CHECK: tt.warp_specialize
// CHECK-SAME: ttg.partition.types = ["gemm", "load", "epilogue", "epilogue_store", "computation"]
tt.func public @data_partitioned_gemm_uses_gemm_template(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %c_desc: !tt.tensordesc<tensor<128x128xf16, #shared>>,
  %M: i32 {tt.divisibility = 16 : i32},
  %N: i32 {tt.divisibility = 16 : i32},
  %K: i32 {tt.divisibility = 16 : i32}
) {
  %false = arith.constant false
  %true = arith.constant true
  %c148_i32 = arith.constant 148 : i32
  %c8_i32 = arith.constant 8 : i32
  %c128_i32 = arith.constant 128 : i32
  %c256_i32 = arith.constant 256 : i32
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>

  %start_pid = tt.get_program_id x : i32
  %num_pid_m = arith.addi %M, %c256_i32 : i32
  %num_pid_m_div = arith.divsi %num_pid_m, %c256_i32 : i32
  %num_pid_n = arith.addi %N, %c128_i32 : i32
  %num_pid_n_div = arith.divsi %num_pid_n, %c128_i32 : i32
  %k_tiles = arith.addi %K, %c64_i32 : i32
  %k_tiles_div = arith.divsi %k_tiles, %c64_i32 : i32
  %num_tiles = arith.muli %num_pid_m_div, %num_pid_n_div : i32
  %tile_id_c_init = arith.subi %start_pid, %c148_i32 : i32
  %num_pid_in_group = arith.muli %num_pid_n_div, %c8_i32 : i32

  %tile_id_c_out = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32
      iter_args(%tile_id_c = %tile_id_c_init) -> (i32) : i32 {
    // Tile index computation
    %group_id = arith.divsi %tile_id, %num_pid_in_group : i32
    %first_pid_m = arith.muli %group_id, %c8_i32 : i32
    %group_size_m = arith.subi %num_pid_m_div, %first_pid_m : i32
    %group_size_m_clamped = arith.minsi %group_size_m, %c8_i32 : i32
    %pid_m = arith.remsi %tile_id, %group_size_m_clamped : i32
    %pid_m_final = arith.addi %first_pid_m, %pid_m : i32
    %pid_n_tmp = arith.remsi %tile_id, %num_pid_in_group : i32
    %pid_n = arith.divsi %pid_n_tmp, %group_size_m_clamped : i32
    %offs_am = arith.muli %pid_m_final, %c256_i32 : i32
    %offs_am_1 = arith.addi %offs_am, %c128_i32 : i32
    %offs_bn = arith.muli %pid_n, %c128_i32 : i32

    // Accumulator init for both halves
    %acc0_mem, %acc0_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc0_tok2 = ttng.tmem_store %cst, %acc0_mem[%acc0_tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc1_mem, %acc1_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc1_tok2 = ttng.tmem_store %cst, %acc1_mem[%acc1_tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // Inner k-loop with two MMAs (one per data partition half)
    %loop_out:3 = scf.for %ki = %c0_i32 to %k_tiles_div step %c1_i32
        iter_args(%use_acc = %false, %loop_tok0 = %acc0_tok2, %loop_tok1 = %acc1_tok2) -> (i1, !ttg.async.token, !ttg.async.token) : i32 {
      %offs_k = arith.muli %ki, %c64_i32 {loop.cluster = 5 : i32, loop.stage = 0 : i32} : i32

      // Load A half 0
      %a0 = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 5 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %a0_smem = ttg.local_alloc %a0 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

      // Load A half 1
      %a1 = tt.descriptor_load %a_desc[%offs_am_1, %offs_k] {loop.cluster = 5 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %a1_smem = ttg.local_alloc %a1 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

      // Load B (shared between both MMAs)
      %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {loop.cluster = 5 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %b_smem = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_trans = ttg.memdesc_trans %b_smem {loop.cluster = 0 : i32, loop.stage = 3 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>

      // MMA 0: A_half0 x B -> acc0
      %mma_tok0 = ttng.tc_gen5_mma %a0_smem, %b_trans, %acc0_mem[%loop_tok0], %use_acc, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // MMA 1: A_half1 x B -> acc1
      %mma_tok1 = ttng.tc_gen5_mma %a1_smem, %b_trans, %acc1_mem[%loop_tok1], %use_acc, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      scf.yield %true, %mma_tok0, %mma_tok1 : i1, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32}

    // Epilogue: next-tile index computation
    %tile_id_c_next = arith.addi %tile_id_c, %c148_i32 : i32
    %group_id_c = arith.divsi %tile_id_c_next, %num_pid_in_group : i32
    %first_pid_m_c = arith.muli %group_id_c, %c8_i32 : i32
    %group_size_m_c = arith.subi %num_pid_m_div, %first_pid_m_c : i32
    %group_size_m_c_clamped = arith.minsi %group_size_m_c, %c8_i32 : i32
    %pid_m_c = arith.remsi %tile_id_c_next, %group_size_m_c_clamped : i32
    %pid_m_c_final = arith.addi %first_pid_m_c, %pid_m_c : i32
    %pid_n_c_tmp = arith.remsi %tile_id_c_next, %num_pid_in_group : i32
    %pid_n_c = arith.divsi %pid_n_c_tmp, %group_size_m_c_clamped : i32
    %offs_am_c = arith.muli %pid_m_c_final, %c256_i32 : i32
    %offs_am_c_1 = arith.addi %offs_am_c, %c128_i32 : i32
    %offs_bn_c = arith.muli %pid_n_c, %c128_i32 : i32

    // Epilogue: tmem_load + truncf + TMA store for half 0
    %result0, %result0_tok = ttng.tmem_load %acc0_mem[%loop_out#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %c0_f16 = arith.truncf %result0 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %c0_smem = ttg.local_alloc %c0_f16 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %store_tok0 = ttng.async_tma_copy_local_to_global %c_desc[%offs_am_c, %offs_bn_c] %c0_smem : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %store_tok0 : !ttg.async.token

    // Epilogue: tmem_load + truncf + TMA store for half 1
    %result1, %result1_tok = ttng.tmem_load %acc1_mem[%loop_out#2] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %c1_f16 = arith.truncf %result1 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %c1_smem = ttg.local_alloc %c1_f16 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %store_tok1 = ttng.async_tma_copy_local_to_global %c_desc[%offs_am_c_1, %offs_bn_c] %c1_smem : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %store_tok1 : !ttg.async.token

    scf.yield %tile_id_c_next : i32
  } {tt.data_partition_factor = 2 : i32, tt.smem_alloc_algo = 0 : i32, tt.warp_specialize}

  tt.return
}

}
