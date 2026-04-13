// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner=num-buffers=3 | FileCheck %s

// Test: Persistent GEMM with data_partition_factor=2 produces two separate
// tmem_loads, each with a 4-way split epilogue. The 4 epilogue SMEM buffers
// from each tmem_load should be fused into the same buffer.id (since they
// share the same original load and have disjoint liveness).
// This results in 2 distinct epilogue buffer IDs instead of 8.

// CHECK-LABEL: @matmul_kernel_tma_persistent
// 8 epilogue buffers should be fused into 2 buffer IDs (one per tmem_load).
// Buffers alternate: EP0, EP1, EP0, EP1, EP0, EP1, EP0, EP1.
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[EP0:[0-9]+]] : i32}
// CHECK-SAME: 128x64xf16
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[EP1:[0-9]+]] : i32}
// CHECK-SAME: 128x64xf16
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[EP0]] : i32}
// CHECK-SAME: 128x64xf16
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[EP1]] : i32}
// CHECK-SAME: 128x64xf16
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[EP0]] : i32}
// CHECK-SAME: 128x64xf16
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[EP1]] : i32}
// CHECK-SAME: 128x64xf16
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[EP0]] : i32}
// CHECK-SAME: 128x64xf16
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = [[EP1]] : i32}
// CHECK-SAME: 128x64xf16
// Innermost-loop buffers (multi-buffered):
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32
// CHECK-SAME: 256x64xf16
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32
// CHECK-SAME: 128x64xf16
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32
// CHECK-SAME: 128x64xf16

#blocked = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent(
      %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %b_desc: !tt.tensordesc<tensor<256x64xf16, #shared>>,
      %c_desc_or_ptr: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %K: i32 {tt.divisibility = 16 : i32},
      %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // 8 epilogue SMEM buffers (4 per data partition).
    %_0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %_1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %_1_12 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %_0_13 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %_1_14 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %_0_15 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %_1_16 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %_0_17 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // Innermost-loop SMEM buffers.
    %arg2 = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %a_1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %a_0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // Two accumulators (data partition factor = 2).
    %accumulator_1, %accumulator_1_18 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %accumulator_0, %accumulator_0_19 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 148 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 8 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 256 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 64 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i32
    %c192_i32 = arith.constant {async_task_id = array<i32: 3>} 192 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 1 : i32
    %c255_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 255 : i32
    %k_tiles = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 63 : i32
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %start_pid = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_m = arith.addi %M, %c255_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_m_20 = arith.divsi %num_pid_m, %c256_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_n = arith.addi %N, %c255_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_n_21 = arith.divsi %num_pid_n, %c256_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %k_tiles_22 = arith.addi %K, %k_tiles {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %k_tiles_23 = arith.divsi %k_tiles_22, %c64_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_tiles = arith.muli %num_pid_m_20, %num_pid_n_21 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %tile_id_c = arith.subi %start_pid, %c148_i32 {async_task_id = array<i32: 3>} : i32
    %num_pid_in_group = arith.muli %num_pid_n_21, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
    // Outer persistent loop.
    %tile_id_c_24 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%tile_id_c_25 = %tile_id_c) -> (i32)  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group {async_task_id = array<i32: 2>} : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %group_size_m = arith.subi %num_pid_m_20, %first_pid_m {async_task_id = array<i32: 2>} : i32
      %group_size_m_26 = arith.minsi %group_size_m, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %pid_m = arith.remsi %tile_id, %group_size_m_26 {async_task_id = array<i32: 2>} : i32
      %pid_m_27 = arith.addi %first_pid_m, %pid_m {async_task_id = array<i32: 2>} : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group {async_task_id = array<i32: 2>} : i32
      %pid_n_28 = arith.divsi %pid_n, %group_size_m_26 {async_task_id = array<i32: 2>} : i32
      %offs_am = arith.muli %pid_m_27, %c256_i32 {async_task_id = array<i32: 2>} : i32
      %a = arith.addi %offs_am, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %offs_bn = arith.muli %pid_n_28, %c256_i32 {async_task_id = array<i32: 2>} : i32
      // Init both accumulators.
      %accumulator = ttng.tmem_store %cst, %accumulator_0[%accumulator_0_19], %true {async_task_id = array<i32: 0>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %accumulator_29 = ttng.tmem_store %cst, %accumulator_1[%accumulator_1_18], %true {async_task_id = array<i32: 0>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      // Inner k-loop (innermost loop).
      %accumulator_30:3 = scf.for %accumulator_75 = %c0_i32 to %k_tiles_23 step %c1_i32 iter_args(%arg22 = %false, %accumulator_76 = %accumulator, %accumulator_77 = %accumulator_29) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %accumulator_75, %c64_i32 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %a_78 = tt.descriptor_load %a_desc[%offs_am, %offs_k] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        %a_79 = tt.descriptor_load %a_desc[%a, %offs_k] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        ttg.local_store %a_78, %a_0 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttg.local_store %a_79, %a_1 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<256x64xf16, #shared>> -> tensor<256x64xf16, #blocked1>
        ttg.local_store %b, %arg2 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        %arg2_80 = ttg.memdesc_trans %arg2 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
        %accumulator_81 = ttng.tc_gen5_mma %a_0, %arg2_80, %accumulator_0[%accumulator_76], %arg22, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        %accumulator_82 = ttng.tc_gen5_mma %a_1, %arg2_80, %accumulator_1[%accumulator_77], %arg22, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1, 4>} %true, %accumulator_81, %accumulator_82 : i1, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3, 4>, tt.scheduled_max_stage = 2 : i32}
      // Epilogue: compute next tile IDs.
      %tile_id_c_31 = arith.addi %tile_id_c_25, %c148_i32 {async_task_id = array<i32: 3>} : i32
      %group_id_32 = arith.divsi %tile_id_c_31, %num_pid_in_group {async_task_id = array<i32: 3>} : i32
      %first_pid_m_33 = arith.muli %group_id_32, %c8_i32 {async_task_id = array<i32: 3>} : i32
      %group_size_m_34 = arith.subi %num_pid_m_20, %first_pid_m_33 {async_task_id = array<i32: 3>} : i32
      %group_size_m_35 = arith.minsi %group_size_m_34, %c8_i32 {async_task_id = array<i32: 3>} : i32
      %pid_m_36 = arith.remsi %tile_id_c_31, %group_size_m_35 {async_task_id = array<i32: 3>} : i32
      %pid_m_37 = arith.addi %first_pid_m_33, %pid_m_36 {async_task_id = array<i32: 3>} : i32
      %pid_n_38 = arith.remsi %tile_id_c_31, %num_pid_in_group {async_task_id = array<i32: 3>} : i32
      %pid_n_39 = arith.divsi %pid_n_38, %group_size_m_35 {async_task_id = array<i32: 3>} : i32
      %offs_am_c = arith.muli %pid_m_37, %c256_i32 {async_task_id = array<i32: 3>} : i32
      %0 = arith.addi %offs_am_c, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %1 = arith.addi %offs_am_c, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %2 = arith.addi %offs_am_c, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %3 = arith.addi %offs_am_c, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %offs_bn_c = arith.muli %pid_n_39, %c256_i32 {async_task_id = array<i32: 3>} : i32
      // tmem_load for both data partitions.
      %accumulator_40, %accumulator_41 = ttng.tmem_load %accumulator_0[%accumulator_30#1] {async_task_id = array<i32: 4>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
      %accumulator_42, %accumulator_43 = ttng.tmem_load %accumulator_1[%accumulator_30#2] {async_task_id = array<i32: 4>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
      // Split chain for accumulator_0: reshape → trans → split → reshape → trans → split (4-way).
      %acc = tt.reshape %accumulator_40 {async_task_id = array<i32: 4>} : tensor<128x256xf32, #blocked> -> tensor<128x2x128xf32, #blocked2>
      %acc_44 = tt.reshape %accumulator_42 {async_task_id = array<i32: 4>} : tensor<128x256xf32, #blocked> -> tensor<128x2x128xf32, #blocked2>
      %acc_45 = tt.trans %acc {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked2> -> tensor<128x128x2xf32, #blocked3>
      %acc_46 = tt.trans %acc_44 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked2> -> tensor<128x128x2xf32, #blocked3>
      %outLHS, %outRHS = tt.split %acc_45 {async_task_id = array<i32: 4>} : tensor<128x128x2xf32, #blocked3> -> tensor<128x128xf32, #blocked4>
      %outLHS_47, %outRHS_48 = tt.split %acc_46 {async_task_id = array<i32: 4>} : tensor<128x128x2xf32, #blocked3> -> tensor<128x128xf32, #blocked4>
      %acc_lo = tt.reshape %outLHS {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked4> -> tensor<128x2x64xf32, #blocked5>
      %acc_lo_49 = tt.reshape %outLHS_47 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked4> -> tensor<128x2x64xf32, #blocked5>
      %acc_lo_50 = tt.trans %acc_lo {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked6>
      %acc_lo_51 = tt.trans %acc_lo_49 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked6>
      %outLHS_52, %outRHS_53 = tt.split %acc_lo_50 {async_task_id = array<i32: 4>} : tensor<128x64x2xf32, #blocked6> -> tensor<128x64xf32, #blocked7>
      %outLHS_54, %outRHS_55 = tt.split %acc_lo_51 {async_task_id = array<i32: 4>} : tensor<128x64x2xf32, #blocked6> -> tensor<128x64xf32, #blocked7>
      %acc_hi = tt.reshape %outRHS {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked4> -> tensor<128x2x64xf32, #blocked5>
      %acc_hi_56 = tt.reshape %outRHS_48 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked4> -> tensor<128x2x64xf32, #blocked5>
      %acc_hi_57 = tt.trans %acc_hi {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked6>
      %acc_hi_58 = tt.trans %acc_hi_56 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked6>
      %outLHS_59, %outRHS_60 = tt.split %acc_hi_57 {async_task_id = array<i32: 4>} : tensor<128x64x2xf32, #blocked6> -> tensor<128x64xf32, #blocked7>
      %outLHS_61, %outRHS_62 = tt.split %acc_hi_58 {async_task_id = array<i32: 4>} : tensor<128x64x2xf32, #blocked6> -> tensor<128x64xf32, #blocked7>
      // Epilogue stores: truncf → convert_layout → local_store → TMA store, sequentially.
      // Sub-tile c0 (from accumulator_0 and accumulator_1).
      %c0 = arith.truncf %outLHS_52 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked7> to tensor<128x64xf16, #blocked7>
      %c0_63 = arith.truncf %outLHS_54 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked7> to tensor<128x64xf16, #blocked7>
      %c0_64 = ttg.convert_layout %c0 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked7> -> tensor<128x64xf16, #blocked1>
      %c0_65 = ttg.convert_layout %c0_63 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked7> -> tensor<128x64xf16, #blocked1>
      ttg.local_store %c0_64, %_0_17 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %4 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%offs_am_c, %offs_bn_c] %_0_17 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %4 {async_task_id = array<i32: 3>} : !ttg.async.token
      ttg.local_store %c0_65, %_1_16 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %5 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%3, %offs_bn_c] %_1_16 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %5 {async_task_id = array<i32: 3>} : !ttg.async.token
      // Sub-tile c1.
      %c1 = arith.truncf %outRHS_53 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked7> to tensor<128x64xf16, #blocked7>
      %c1_66 = arith.truncf %outRHS_55 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked7> to tensor<128x64xf16, #blocked7>
      %c1_67 = ttg.convert_layout %c1 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked7> -> tensor<128x64xf16, #blocked1>
      %c1_68 = ttg.convert_layout %c1_66 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked7> -> tensor<128x64xf16, #blocked1>
      %6 = arith.addi %offs_bn_c, %c64_i32 {async_task_id = array<i32: 3>} : i32
      ttg.local_store %c1_67, %_0_15 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %7 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%offs_am_c, %6] %_0_15 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %7 {async_task_id = array<i32: 3>} : !ttg.async.token
      ttg.local_store %c1_68, %_1_14 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %8 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%2, %6] %_1_14 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %8 {async_task_id = array<i32: 3>} : !ttg.async.token
      // Sub-tile c2.
      %c2 = arith.truncf %outLHS_59 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked7> to tensor<128x64xf16, #blocked7>
      %c2_69 = arith.truncf %outLHS_61 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked7> to tensor<128x64xf16, #blocked7>
      %c2_70 = ttg.convert_layout %c2 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked7> -> tensor<128x64xf16, #blocked1>
      %c2_71 = ttg.convert_layout %c2_69 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked7> -> tensor<128x64xf16, #blocked1>
      %9 = arith.addi %offs_bn_c, %c128_i32 {async_task_id = array<i32: 3>} : i32
      ttg.local_store %c2_70, %_0_13 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %10 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%offs_am_c, %9] %_0_13 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %10 {async_task_id = array<i32: 3>} : !ttg.async.token
      ttg.local_store %c2_71, %_1_12 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %11 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%1, %9] %_1_12 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %11 {async_task_id = array<i32: 3>} : !ttg.async.token
      // Sub-tile c3.
      %c3 = arith.truncf %outRHS_60 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked7> to tensor<128x64xf16, #blocked7>
      %c3_72 = arith.truncf %outRHS_62 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked7> to tensor<128x64xf16, #blocked7>
      %c3_73 = ttg.convert_layout %c3 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked7> -> tensor<128x64xf16, #blocked1>
      %c3_74 = ttg.convert_layout %c3_72 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked7> -> tensor<128x64xf16, #blocked1>
      %12 = arith.addi %offs_bn_c, %c192_i32 {async_task_id = array<i32: 3>} : i32
      ttg.local_store %c3_73, %_1 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %13 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%offs_am_c, %12] %_1 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %13 {async_task_id = array<i32: 3>} : !ttg.async.token
      ttg.local_store %c3_74, %_0 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %14 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%0, %12] %_0 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %14 {async_task_id = array<i32: 3>} : !ttg.async.token
      scf.yield {async_task_id = array<i32: 3>} %tile_id_c_31 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3, 4>, tt.data_partition_factor = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["default", "gemm", "load", "epilogue", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
