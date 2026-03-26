// RUN: triton-opt %s --nvgpu-partition-scheduling-meta | FileCheck %s

// Tests that GEMM partition scheduling does not create a separate "computation"
// partition. Multi-def/sink clusters should merge into the default partition.

#blocked = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @persistent_gemm_no_computation_partition
// CHECK: tt.warp_specialize
// CHECK-SAME: ttg.partition.types = ["default", "gemm", "load", "epilogue"]
tt.func public @persistent_gemm_no_computation_partition(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<256x64xf16, #shared>>,
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
  %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>

  %start_pid = tt.get_program_id x : i32
  %num_pid_m = arith.addi %M, %c128_i32 : i32
  %num_pid_m_div = arith.divsi %num_pid_m, %c128_i32 : i32
  %num_pid_n = arith.addi %N, %c256_i32 : i32
  %num_pid_n_div = arith.divsi %num_pid_n, %c256_i32 : i32
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
    %offs_am = arith.muli %pid_m_final, %c128_i32 : i32
    %offs_bn = arith.muli %pid_n, %c256_i32 : i32

    // Accumulator init
    %acc_mem, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_tok2 = ttng.tmem_store %cst, %acc_mem[%acc_tok], %true : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>

    // Inner k-loop (warp specialized)
    %loop_out:2 = scf.for %ki = %c0_i32 to %k_tiles_div step %c1_i32
        iter_args(%use_acc = %false, %loop_tok = %acc_tok2) -> (i1, !ttg.async.token) : i32 {
      %offs_k = arith.muli %ki, %c64_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
      %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %a_smem = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<256x64xf16, #shared>> -> tensor<256x64xf16, #blocked1>
      %b_smem = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<256x64xf16, #blocked1>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
      %b_trans = ttg.memdesc_trans %b_smem {loop.cluster = 0 : i32, loop.stage = 3 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem> -> !ttg.memdesc<64x256xf16, #shared1, #smem>
      %mma_tok = ttng.tc_gen5_mma %a_smem, %b_trans, %acc_mem[%loop_tok], %use_acc, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x256xf16, #shared1, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %true, %mma_tok : i1, !ttg.async.token
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
    %offs_am_c = arith.muli %pid_m_c_final, %c128_i32 : i32
    %offs_bn_c = arith.muli %pid_n_c, %c256_i32 : i32

    // Epilogue: tmem_load + reshape + split + two TMA stores
    %result, %result_tok = ttng.tmem_load %acc_mem[%loop_out#1] : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
    %reshaped = tt.reshape %result : tensor<128x256xf32, #blocked> -> tensor<128x2x128xf32, #blocked2>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked2> -> tensor<128x128x2xf32, #blocked3>
    %lhs, %rhs = tt.split %transposed : tensor<128x128x2xf32, #blocked3> -> tensor<128x128xf32, #blocked4>

    %c0_f16 = arith.truncf %lhs : tensor<128x128xf32, #blocked4> to tensor<128x128xf16, #blocked4>
    %c0_cvt = ttg.convert_layout %c0_f16 : tensor<128x128xf16, #blocked4> -> tensor<128x128xf16, #blocked5>
    %c0_smem = ttg.local_alloc %c0_cvt : (tensor<128x128xf16, #blocked5>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %store_tok0 = ttng.async_tma_copy_local_to_global %c_desc[%offs_am_c, %offs_bn_c] %c0_smem : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %store_tok0 : !ttg.async.token

    %c1_f16 = arith.truncf %rhs : tensor<128x128xf32, #blocked4> to tensor<128x128xf16, #blocked4>
    %c1_cvt = ttg.convert_layout %c1_f16 : tensor<128x128xf16, #blocked4> -> tensor<128x128xf16, #blocked5>
    %offs_bn_c2 = arith.addi %offs_bn_c, %c128_i32 : i32
    %c1_smem = ttg.local_alloc %c1_cvt : (tensor<128x128xf16, #blocked5>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %store_tok1 = ttng.async_tma_copy_local_to_global %c_desc[%offs_am_c, %offs_bn_c2] %c1_smem : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
    ttng.async_tma_store_token_wait %store_tok1 : !ttg.async.token

    scf.yield %tile_id_c_next : i32
  } {tt.data_partition_factor = 1 : i32, tt.smem_alloc_algo = 1 : i32, tt.warp_specialize}

  tt.return
}

}
