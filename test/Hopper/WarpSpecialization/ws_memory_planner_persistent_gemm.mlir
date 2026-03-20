// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner=num-buffers=4 | FileCheck %s

// Test case: Persistent GEMM with warp specialization and TMEM accumulator.
// The TMEM accumulator (tmem_alloc) is used across the inner k-loop with a
// loop-carried acc_dep token, meaning the accumulator is reused across
// k-iterations. The memory planner should assign buffer.copy = 4 for the
// TMEM accumulator (multi-buffered across tile iterations), and annotate
// tmem_store / tc_gen5_mma / tmem_load with tmem.start / tmem.end.
//
// This test verifies the fix for a bug where the TMEM accumulator's buffer
// index would incorrectly rotate every inner k-loop iteration instead of
// only across outer tile-loop iterations.

// CHECK-LABEL: @matmul_kernel_tma_persistent
// TMEM accumulator gets buffer.copy = 4 (multi-buffered across tile iterations)
// CHECK: ttng.tmem_alloc {{{.*}}buffer.copy = 4 : i32, buffer.id = 4 : i32}
// CHECK-SAME: !ttg.memdesc<128x128xf32
// tmem_store gets tmem.start annotation
// CHECK: ttng.tmem_store {{.*}} tmem.start
// tc_gen5_mma gets tmem.end and tmem.start annotations
// CHECK: ttng.tc_gen5_mma {{.*}} tmem.end = {{.*}} tmem.start =
// tmem_load gets tmem.end annotation
// CHECK: ttng.tmem_load {{.*}} tmem.end

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent(
      %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %a_desc_0: i32, %a_desc_1: i32, %a_desc_2: i64, %a_desc_3: i64,
      %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %b_desc_4: i32, %b_desc_5: i32, %b_desc_6: i64, %b_desc_7: i64,
      %c_desc_or_ptr: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %c_desc_or_ptr_8: i32, %c_desc_or_ptr_9: i32,
      %c_desc_or_ptr_10: i64, %c_desc_or_ptr_11: i64,
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %K: i32 {tt.divisibility = 16 : i32},
      %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 148 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 128 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 64 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 1 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 127 : i32
    %k_tiles = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 63 : i32
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %start_pid = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_m = arith.addi %M, %c127_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_m_12 = arith.divsi %num_pid_m, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_n = arith.addi %N, %c127_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_n_13 = arith.divsi %num_pid_n, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %k_tiles_14 = arith.addi %K, %k_tiles {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %k_tiles_15 = arith.divsi %k_tiles_14, %c64_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_tiles = arith.muli %num_pid_m_12, %num_pid_n_13 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %tile_id_c = arith.subi %start_pid, %c148_i32 {async_task_id = array<i32: 3>} : i32
    %num_pid_in_group = arith.muli %num_pid_n_13, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
    %tile_id_c_16 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%tile_id_c_17 = %tile_id_c) -> (i32)  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group {async_task_id = array<i32: 2>} : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %group_size_m = arith.subi %num_pid_m_12, %first_pid_m {async_task_id = array<i32: 2>} : i32
      %group_size_m_18 = arith.minsi %group_size_m, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %pid_m = arith.remsi %tile_id, %group_size_m_18 {async_task_id = array<i32: 2>} : i32
      %pid_m_19 = arith.addi %first_pid_m, %pid_m {async_task_id = array<i32: 2>} : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group {async_task_id = array<i32: 2>} : i32
      %pid_n_20 = arith.divsi %pid_n, %group_size_m_18 {async_task_id = array<i32: 2>} : i32
      %offs_am = arith.muli %pid_m_19, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %offs_bn = arith.muli %pid_n_20, %c128_i32 {async_task_id = array<i32: 2>} : i32
      // TMEM accumulator alloc — used across inner k-loop with loop-carried token
      %accumulator, %accumulator_21 = ttng.tmem_alloc {async_task_id = array<i32: 0, 1, 4>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %accumulator_22 = ttng.tmem_store %cst, %accumulator[%accumulator_21], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // Inner k-loop: accumulator token is loop-carried (iter_arg -> yield)
      %accumulator_23:2 = scf.for %accumulator_38 = %c0_i32 to %k_tiles_15 step %c1_i32 iter_args(%arg22 = %false, %accumulator_39 = %accumulator_22) -> (i1, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %accumulator_38, %c64_i32 {async_task_id = array<i32: 2>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
        %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] {async_task_id = array<i32: 2>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        %a_40 = ttg.local_alloc %a {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {async_task_id = array<i32: 2>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        %arg2 = ttg.local_alloc %b {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %arg2_41 = ttg.memdesc_trans %arg2 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 3 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %accumulator_42 = ttng.tc_gen5_mma %a_40, %arg2_41, %accumulator[%accumulator_39], %arg22, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1, 4>} %true, %accumulator_42 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3, 4>, tt.scheduled_max_stage = 3 : i32}
      // Epilogue: load accumulator from TMEM, convert, store via TMA
      %tile_id_c_24 = arith.addi %tile_id_c_17, %c148_i32 {async_task_id = array<i32: 3>} : i32
      %group_id_25 = arith.divsi %tile_id_c_24, %num_pid_in_group {async_task_id = array<i32: 3>} : i32
      %first_pid_m_26 = arith.muli %group_id_25, %c8_i32 {async_task_id = array<i32: 3>} : i32
      %group_size_m_27 = arith.subi %num_pid_m_12, %first_pid_m_26 {async_task_id = array<i32: 3>} : i32
      %group_size_m_28 = arith.minsi %group_size_m_27, %c8_i32 {async_task_id = array<i32: 3>} : i32
      %pid_m_29 = arith.remsi %tile_id_c_24, %group_size_m_28 {async_task_id = array<i32: 3>} : i32
      %pid_m_30 = arith.addi %first_pid_m_26, %pid_m_29 {async_task_id = array<i32: 3>} : i32
      %pid_n_31 = arith.remsi %tile_id_c_24, %num_pid_in_group {async_task_id = array<i32: 3>} : i32
      %pid_n_32 = arith.divsi %pid_n_31, %group_size_m_28 {async_task_id = array<i32: 3>} : i32
      %offs_am_c = arith.muli %pid_m_30, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %offs_bn_c = arith.muli %pid_n_32, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %accumulator_33, %accumulator_34 = ttng.tmem_load %accumulator[%accumulator_23#1] {async_task_id = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc = tt.reshape %accumulator_33 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked2>
      %acc_35 = tt.trans %acc {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked2> -> tensor<128x64x2xf32, #blocked3>
      %outLHS, %outRHS = tt.split %acc_35 {async_task_id = array<i32: 4>} : tensor<128x64x2xf32, #blocked3> -> tensor<128x64xf32, #blocked4>
      %c0 = arith.truncf %outLHS {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked4> to tensor<128x64xf16, #blocked4>
      %c0_36 = ttg.convert_layout %c0 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked4> -> tensor<128x64xf16, #blocked1>
      %0 = ttg.local_alloc %c0_36 {async_task_id = array<i32: 4>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %1 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%offs_am_c, %offs_bn_c] %0 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %1   {async_task_id = array<i32: 3>} : !ttg.async.token
      %c1 = arith.truncf %outRHS {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked4> to tensor<128x64xf16, #blocked4>
      %c1_37 = ttg.convert_layout %c1 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked4> -> tensor<128x64xf16, #blocked1>
      %2 = arith.addi %offs_bn_c, %c64_i32 {async_task_id = array<i32: 3>} : i32
      %3 = ttg.local_alloc %c1_37 {async_task_id = array<i32: 4>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %4 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%offs_am_c, %2] %3 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %4   {async_task_id = array<i32: 3>} : !ttg.async.token
      scf.yield {async_task_id = array<i32: 3>} %tile_id_c_24 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3, 4>, tt.data_partition_factor = 1 : i32, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["default", "gemm", "load", "epilogue", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
