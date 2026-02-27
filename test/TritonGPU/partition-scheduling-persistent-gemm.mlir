// RUN: triton-opt %s --tritongpu-hoist-tmem-alloc --tritongpu-partition-scheduling | FileCheck %s

// Test that partition scheduling for a persistent GEMM with epilogue subtiling
// assigns only 3 partitions to operations (loads=2, MMA=1, epilogue stores=3).
// The inner scf.for must NOT be assigned to any partition (it is structural).
// The epilogue compute chain (tmem_load, reshape, trans, split, truncf,
// convert_layout) must not be assigned to any partition either.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @persistent_gemm_epilogue_subtile
tt.func public @persistent_gemm_epilogue_subtile(
    %a_desc: !tt.tensordesc<tensor<128x128xf16, #shared>>,
    %b_desc: !tt.tensordesc<tensor<128x128xf16, #shared>>,
    %c_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
    %M: i32 {tt.divisibility = 16 : i32},
    %N: i32 {tt.divisibility = 16 : i32},
    %K: i32 {tt.divisibility = 16 : i32}
) {
    %false = arith.constant false
    %true = arith.constant true
    %c148_i32 = arith.constant 148 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %start_pid = tt.get_program_id x : i32
    %num_pid_m = arith.addi %M, %c127_i32 : i32
    %num_pid_m_12 = arith.divsi %num_pid_m, %c128_i32 : i32
    %num_pid_n = arith.addi %N, %c127_i32 : i32
    %num_pid_n_13 = arith.divsi %num_pid_n, %c128_i32 : i32
    %k_tiles = arith.addi %K, %c127_i32 : i32
    %k_tiles_14 = arith.divsi %k_tiles, %c128_i32 : i32
    %num_tiles = arith.muli %num_pid_m_12, %num_pid_n_13 : i32
    %tile_id_c = arith.subi %start_pid, %c148_i32 : i32
    %num_pid_in_group = arith.muli %num_pid_n_13, %c8_i32 : i32
    // Outer persistent loop
    %tile_id_c_15 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%tile_id_c_16 = %tile_id_c) -> (i32) : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 : i32
      %group_size_m = arith.subi %num_pid_m_12, %first_pid_m : i32
      %group_size_m_17 = arith.minsi %group_size_m, %c8_i32 : i32
      %pid_m = arith.remsi %tile_id, %group_size_m_17 : i32
      %pid_m_18 = arith.addi %first_pid_m, %pid_m : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group : i32
      %pid_n_19 = arith.divsi %pid_n, %group_size_m_17 : i32
      %offs_am = arith.muli %pid_m_18, %c128_i32 : i32
      %offs_bn = arith.muli %pid_n_19, %c128_i32 : i32
      %accumulator, %accumulator_20 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %accumulator_21 = ttng.tmem_store %cst, %accumulator[%accumulator_20], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // Inner K-loop
      %accumulator_22:2 = scf.for %accumulator_37 = %c0_i32 to %k_tiles_14 step %c1_i32 iter_args(%arg21 = %false, %accumulator_38 = %accumulator_21) -> (i1, !ttg.async.token) : i32 {
        // Partition 2: loads and local_allocs
        // CHECK: tt.descriptor_load {{.*}} ttg.partition = array<i32: 2>
        %offs_k = arith.muli %accumulator_37, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
        // CHECK: ttg.local_alloc {{.*}} ttg.partition = array<i32: 2>
        %a_39 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        // CHECK: tt.descriptor_load {{.*}} ttg.partition = array<i32: 2>
        %b = tt.descriptor_load %b_desc[%offs_k, %offs_bn] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
        // CHECK: ttg.local_alloc {{.*}} ttg.partition = array<i32: 2>
        %b_40 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        // Partition 1: MMA
        // CHECK: ttng.tc_gen5_mma {{.*}} ttg.partition = array<i32: 1>
        %accumulator_41 = ttng.tc_gen5_mma %a_39, %b_40, %accumulator[%accumulator_38], %arg21, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %accumulator_41 : i1, !ttg.async.token
      // The inner scf.for must NOT have a ttg.partition attribute.
      // CHECK: {tt.scheduled_max_stage
      // CHECK-NOT: ttg.partition
      } {tt.scheduled_max_stage = 2 : i32}
      // Epilogue address computation
      %tile_id_c_23 = arith.addi %tile_id_c_16, %c148_i32 : i32
      %group_id_24 = arith.divsi %tile_id_c_23, %num_pid_in_group : i32
      %first_pid_m_25 = arith.muli %group_id_24, %c8_i32 : i32
      %group_size_m_26 = arith.subi %num_pid_m_12, %first_pid_m_25 : i32
      %group_size_m_27 = arith.minsi %group_size_m_26, %c8_i32 : i32
      %pid_m_28 = arith.remsi %tile_id_c_23, %group_size_m_27 : i32
      %pid_m_29 = arith.addi %first_pid_m_25, %pid_m_28 : i32
      %pid_n_30 = arith.remsi %tile_id_c_23, %num_pid_in_group : i32
      %pid_n_31 = arith.divsi %pid_n_30, %group_size_m_27 : i32
      %offs_am_c = arith.muli %pid_m_29, %c128_i32 : i32
      %offs_bn_c = arith.muli %pid_n_31, %c128_i32 : i32
      // Epilogue compute chain — none of these should have a partition.
      // CHECK: ttng.tmem_load
      // CHECK-NOT: ttg.partition
      // CHECK: tt.reshape
      // CHECK-NOT: ttg.partition
      // CHECK: tt.trans
      // CHECK-NOT: ttg.partition
      // CHECK: tt.split
      // CHECK-NOT: ttg.partition
      // CHECK: arith.truncf
      // CHECK-NOT: ttg.partition
      // CHECK: ttg.convert_layout
      %accumulator_32, %accumulator_33 = ttng.tmem_load %accumulator[%accumulator_22#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %acc = tt.reshape %accumulator_32 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked2>
      %acc_34 = tt.trans %acc {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked2> -> tensor<128x64x2xf32, #blocked3>
      %outLHS, %outRHS = tt.split %acc_34 : tensor<128x64x2xf32, #blocked3> -> tensor<128x64xf32, #blocked4>
      %c0 = arith.truncf %outLHS : tensor<128x64xf32, #blocked4> to tensor<128x64xf16, #blocked4>
      %c0_35 = ttg.convert_layout %c0 : tensor<128x64xf16, #blocked4> -> tensor<128x64xf16, #blocked5>
      // Partition 3: first epilogue store
      // CHECK: tt.descriptor_store {{.*ttg.partition = array<i32: 3>}}
      tt.descriptor_store %c_desc[%offs_am_c, %offs_bn_c], %c0_35 : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked5>
      // Second subtile compute — no partition
      // CHECK: arith.truncf
      // CHECK-NOT: ttg.partition
      // CHECK: ttg.convert_layout
      %c1 = arith.truncf %outRHS : tensor<128x64xf32, #blocked4> to tensor<128x64xf16, #blocked4>
      %c1_36 = ttg.convert_layout %c1 : tensor<128x64xf16, #blocked4> -> tensor<128x64xf16, #blocked5>
      // Partition 3: second epilogue store
      // CHECK: tt.descriptor_store {{.*ttg.partition = array<i32: 3>}}
      %0 = arith.addi %offs_bn_c, %c64_i32 : i32
      tt.descriptor_store %c_desc[%offs_am_c, %0], %c1_36 : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked5>
      scf.yield %tile_id_c_23 : i32
    // 4 entries in partition.stages (default=0, MMA=1, loads=2, stores=3),
    // but only partitions 1, 2, 3 are assigned to operations.
    // CHECK: tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32]
    } {tt.warp_specialize}
    tt.return
  }
}
