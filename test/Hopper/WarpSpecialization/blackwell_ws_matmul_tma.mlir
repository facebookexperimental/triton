// RUN: triton-opt %s -split-input-file --nvgpu-warp-specialization="num-stages=3 capability=100" | FileCheck %s

// Test case: Basic Blackwell matrix multiplication with TMA and warp specialization.
// This IR represents a GEMM kernel that uses tensor memory for accumulator
// and has partition annotations on key operations.

// CHECK-LABEL: @matmul_kernel_tma_ws
// CHECK: ttg.warp_specialize
// Default group: MMA operations
// CHECK: default
// CHECK: ttng.tc_gen5_mma
// Group 0: Descriptor load operations (producer)
// CHECK: partition0
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.async_tma_copy_global_to_local
// Group 1: Epilogue operations
// CHECK: partition1
// CHECK: ttng.tmem_load
// CHECK: tt.descriptor_store

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_ws(%a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %a_desc_0: i32, %a_desc_1: i32, %a_desc_2: i64, %a_desc_3: i64, %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %b_desc_4: i32, %b_desc_5: i32, %b_desc_6: i64, %b_desc_7: i64, %c_desc: !tt.tensordesc<tensor<128x128xf16, #shared>>, %c_desc_8: i32, %c_desc_9: i32, %c_desc_10: i64, %c_desc_11: i64, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %accumulator = arith.constant false
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %k_tiles = arith.constant 63 : i32
    %accumulator_12 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %pid = tt.get_program_id x : i32
    %num_pid_m = arith.addi %M, %c127_i32 : i32
    %num_pid_m_13 = arith.divsi %num_pid_m, %c128_i32 : i32
    %num_pid_n = arith.addi %N, %c127_i32 : i32
    %num_pid_n_14 = arith.divsi %num_pid_n, %c128_i32 : i32
    %num_pid_in_group = arith.muli %num_pid_n_14, %c8_i32 : i32
    %group_id = arith.divsi %pid, %num_pid_in_group : i32
    %first_pid_m = arith.muli %group_id, %c8_i32 : i32
    %group_size_m = arith.subi %num_pid_m_13, %first_pid_m : i32
    %group_size_m_15 = arith.minsi %group_size_m, %c8_i32 : i32
    %pid_m = arith.remsi %pid, %group_size_m_15 : i32
    %pid_m_16 = arith.addi %first_pid_m, %pid_m : i32
    %pid_n = arith.remsi %pid, %num_pid_in_group : i32
    %pid_n_17 = arith.divsi %pid_n, %group_size_m_15 : i32
    %k_tiles_18 = arith.addi %K, %k_tiles : i32
    %k_tiles_19 = arith.divsi %k_tiles_18, %c64_i32 : i32
    %offs_am = arith.muli %pid_m_16, %c128_i32 : i32
    %offs_bn = arith.muli %pid_n_17, %c128_i32 : i32
    %accumulator_20, %accumulator_21 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %accumulator_23:2 = scf.for %accumulator_27 = %c0_i32 to %k_tiles_19 step %c1_i32 iter_args(%accumulator_28 = %accumulator, %accumulator_29 = %accumulator_21) -> (i1, !ttg.async.token)  : i32 {
      %offs_k = arith.muli %accumulator_27, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %a_30 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %accumulator_31 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %accumulator_32 = ttg.memdesc_trans %accumulator_31 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %accumulator_33 = ttng.tc_gen5_mma %a_30, %accumulator_32, %accumulator_20[%accumulator_29], %accumulator_28, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %true, %accumulator_33 : i1, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %accumulator_24, %accumulator_25 = ttng.tmem_load %accumulator_20[%accumulator_23#1] {ttg.partition = 3 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %c = arith.truncf %accumulator_24 {ttg.partition = 3 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %c_26 = ttg.convert_layout %c {ttg.partition = 3 : i32} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
    tt.descriptor_store %c_desc[%offs_am, %offs_bn], %c_26 {ttg.partition = 3 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
    tt.return
  }
}

// -----

// Test case: Blackwell matrix multiplication with explicit tmem_store before loop.
// This IR includes ttng.tmem_store to initialize the accumulator before the loop.

// CHECK-LABEL: @matmul_kernel_tma_ws_with_tmem_store
// CHECK: ttg.warp_specialize
// Default group: MMA operations
// CHECK: default
// CHECK: ttng.tmem_store
// CHECK: ttng.tc_gen5_mma
// Group 0: Descriptor load operations (producer)
// CHECK: partition0
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.async_tma_copy_global_to_local
// Group 1: Epilogue operations
// CHECK: partition1
// CHECK: ttng.tmem_load
// CHECK: tt.descriptor_store

#blocked3 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem2 = #ttg.shared_memory
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_ws_with_tmem_store(%a_desc: !tt.tensordesc<tensor<128x64xf16, #shared2>>, %a_desc_0: i32, %a_desc_1: i32, %a_desc_2: i64, %a_desc_3: i64, %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared2>>, %b_desc_4: i32, %b_desc_5: i32, %b_desc_6: i64, %b_desc_7: i64, %c_desc: !tt.tensordesc<tensor<128x128xf16, #shared2>>, %c_desc_8: i32, %c_desc_9: i32, %c_desc_10: i64, %c_desc_11: i64, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %accumulator = arith.constant false
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %k_tiles = arith.constant 63 : i32
    %accumulator_12 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked3>
    %pid = tt.get_program_id x : i32
    %num_pid_m = arith.addi %M, %c127_i32 : i32
    %num_pid_m_13 = arith.divsi %num_pid_m, %c128_i32 : i32
    %num_pid_n = arith.addi %N, %c127_i32 : i32
    %num_pid_n_14 = arith.divsi %num_pid_n, %c128_i32 : i32
    %num_pid_in_group = arith.muli %num_pid_n_14, %c8_i32 : i32
    %group_id = arith.divsi %pid, %num_pid_in_group : i32
    %first_pid_m = arith.muli %group_id, %c8_i32 : i32
    %group_size_m = arith.subi %num_pid_m_13, %first_pid_m : i32
    %group_size_m_15 = arith.minsi %group_size_m, %c8_i32 : i32
    %pid_m = arith.remsi %pid, %group_size_m_15 : i32
    %pid_m_16 = arith.addi %first_pid_m, %pid_m : i32
    %pid_n = arith.remsi %pid, %num_pid_in_group : i32
    %pid_n_17 = arith.divsi %pid_n, %group_size_m_15 : i32
    %k_tiles_18 = arith.addi %K, %k_tiles : i32
    %k_tiles_19 = arith.divsi %k_tiles_18, %c64_i32 : i32
    %offs_am = arith.muli %pid_m_16, %c128_i32 : i32
    %offs_bn = arith.muli %pid_n_17, %c128_i32 : i32
    %accumulator_20, %accumulator_21 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %accumulator_22 = ttng.tmem_store %accumulator_12, %accumulator_20[%accumulator_21], %true : tensor<128x128xf32, #blocked3> -> !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable>
    %accumulator_23:2 = scf.for %accumulator_27 = %c0_i32 to %k_tiles_19 step %c1_i32 iter_args(%accumulator_28 = %accumulator, %accumulator_29 = %accumulator_22) -> (i1, !ttg.async.token)  : i32 {
      %offs_k = arith.muli %accumulator_27, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared2>> -> tensor<128x64xf16, #blocked4>
      %a_30 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked4>) -> !ttg.memdesc<128x64xf16, #shared2, #smem2>
      %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared2>> -> tensor<128x64xf16, #blocked4>
      %accumulator_31 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked4>) -> !ttg.memdesc<128x64xf16, #shared2, #smem2>
      %accumulator_32 = ttg.memdesc_trans %accumulator_31 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared2, #smem2> -> !ttg.memdesc<64x128xf16, #shared3, #smem2>
      %accumulator_33 = ttng.tc_gen5_mma %a_30, %accumulator_32, %accumulator_20[%accumulator_29], %accumulator_28, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared2, #smem2>, !ttg.memdesc<64x128xf16, #shared3, #smem2>, !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable>
      scf.yield %true, %accumulator_33 : i1, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %accumulator_24, %accumulator_25 = ttng.tmem_load %accumulator_20[%accumulator_23#1] {ttg.partition = 3 : i32} : !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked3>
    %c = arith.truncf %accumulator_24 {ttg.partition = 3 : i32} : tensor<128x128xf32, #blocked3> to tensor<128x128xf16, #blocked3>
    %c_26 = ttg.convert_layout %c {ttg.partition = 3 : i32} : tensor<128x128xf16, #blocked3> -> tensor<128x128xf16, #blocked5>
    tt.descriptor_store %c_desc[%offs_am, %offs_bn], %c_26 {ttg.partition = 3 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared2>>, tensor<128x128xf16, #blocked5>
    tt.return
  }
}
