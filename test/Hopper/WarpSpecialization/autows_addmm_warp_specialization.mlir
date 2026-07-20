// RUN: triton-opt %s --nvgpu-warp-specialization | FileCheck %s
//
// Generated from python/test/unit/language/test_autows_addmm.py with MLIR_ENABLE_DUMP=1.
// Configuration: FLATTEN=False, EPILOGUE_SUBTILE=4, M=N=K=128, BLOCK_SIZE_M=N=128, BLOCK_SIZE_K=64.
//
// CHECK-LABEL: tt.func public @addmm_kernel_tma_persistent_ws
// CHECK: !tt.tensordesc<tensor<128x32xf16
// CHECK: ttg.warp_specialize
// CHECK-SAME: ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation"]
// CHECK: default
// CHECK: constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}
// CHECK: constraints = {WSBarrier = {channelGraph = array<i32: 1, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}
// CHECK: constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}
// CHECK: partition0
// CHECK: constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}
// CHECK: ttng.tc_gen5_mma
// CHECK: partition1
// CHECK: constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}
// CHECK: ttng.async_tma_copy_local_to_global
// CHECK: partition2
// CHECK-COUNT-6: ttng.async_tma_copy_global_to_local
// CHECK: ttg.local_load {{.*}} -> tensor<128x32xf16
// CHECK-NOT: tt.descriptor_load
// CHECK-NOT: nvws.descriptor_load
// CHECK: constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 0 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @addmm_kernel_tma_persistent_ws(%a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %a_desc_0: i32, %a_desc_1: i32, %a_desc_2: i64, %a_desc_3: i64, %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %b_desc_4: i32, %b_desc_5: i32, %b_desc_6: i64, %b_desc_7: i64, %c_desc: !tt.tensordesc<tensor<128x32xf16, #shared1>>, %c_desc_8: i32, %c_desc_9: i32, %c_desc_10: i64, %c_desc_11: i64, %bias_desc: !tt.tensordesc<tensor<128x32xf16, #shared1>>, %bias_desc_12: i32, %bias_desc_13: i32, %bias_desc_14: i64, %bias_desc_15: i64, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c148_i32 = arith.constant 148 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c96_i32 = arith.constant 96 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %k_tiles = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %start_pid = tt.get_program_id x : i32
    %num_pid_m = arith.addi %M, %c127_i32 : i32
    %num_pid_m_16 = arith.divsi %num_pid_m, %c128_i32 : i32
    %num_pid_n = arith.addi %N, %c127_i32 : i32
    %num_pid_n_17 = arith.divsi %num_pid_n, %c128_i32 : i32
    %k_tiles_18 = arith.addi %K, %k_tiles : i32
    %k_tiles_19 = arith.divsi %k_tiles_18, %c64_i32 : i32
    %num_tiles = arith.muli %num_pid_m_16, %num_pid_n_17 : i32
    %tile_id_c = arith.subi %start_pid, %c148_i32 : i32
    %num_pid_in_group = arith.muli %num_pid_n_17, %c8_i32 : i32
    %tile_id_c_20 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%tile_id_c_21 = %tile_id_c) -> (i32)  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 : i32
      %group_size_m = arith.subi %num_pid_m_16, %first_pid_m : i32
      %group_size_m_22 = arith.minsi %group_size_m, %c8_i32 : i32
      %pid_m = arith.remsi %tile_id, %group_size_m_22 : i32
      %pid_m_23 = arith.addi %first_pid_m, %pid_m : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group : i32
      %pid_n_24 = arith.divsi %pid_n, %group_size_m_22 : i32
      %offs_am = arith.muli %pid_m_23, %c128_i32 : i32
      %offs_bn = arith.muli %pid_n_24, %c128_i32 : i32
      %accumulator, %accumulator_25 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %accumulator_26 = ttng.tmem_store %cst, %accumulator[%accumulator_25], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %accumulator_27:2 = scf.for %accumulator_51 = %c0_i32 to %k_tiles_19 step %c1_i32 iter_args(%arg26 = %false, %accumulator_52 = %accumulator_26) -> (i1, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %accumulator_51, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
        %a_53 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
        %accumulator_54 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %accumulator_55 = ttg.memdesc_trans %accumulator_54 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared2, #smem>
        %accumulator_56 = ttng.tc_gen5_mma %a_53, %accumulator_55, %accumulator[%accumulator_52], %arg26, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %accumulator_56 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 1 : i32}
      %tile_id_c_28 = arith.addi %tile_id_c_21, %c148_i32 : i32
      %group_id_29 = arith.divsi %tile_id_c_28, %num_pid_in_group : i32
      %first_pid_m_30 = arith.muli %group_id_29, %c8_i32 : i32
      %group_size_m_31 = arith.subi %num_pid_m_16, %first_pid_m_30 : i32
      %group_size_m_32 = arith.minsi %group_size_m_31, %c8_i32 : i32
      %pid_m_33 = arith.remsi %tile_id_c_28, %group_size_m_32 : i32
      %pid_m_34 = arith.addi %first_pid_m_30, %pid_m_33 : i32
      %pid_n_35 = arith.remsi %tile_id_c_28, %num_pid_in_group : i32
      %pid_n_36 = arith.divsi %pid_n_35, %group_size_m_32 : i32
      %offs_cm = arith.muli %pid_m_34, %c128_i32 : i32
      %offs_cn = arith.muli %pid_n_36, %c128_i32 : i32
      %accumulator_37, %accumulator_38 = ttng.tmem_load %accumulator[%accumulator_27#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %acc = tt.reshape %accumulator_37 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
      %acc_39 = tt.trans %acc {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
      %outLHS, %outRHS = tt.split %acc_39 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>
      %0 = tt.reshape %outLHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> -> tensor<128x2x32xf32, #linear4>
      %1 = tt.trans %0 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #linear4> -> tensor<128x32x2xf32, #linear5>
      %outLHS_40, %outRHS_41 = tt.split %1 {ttg.partition = array<i32: 0>} : tensor<128x32x2xf32, #linear5> -> tensor<128x32xf32, #linear6>
      %2 = ttg.convert_layout %outRHS_41 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> -> tensor<128x32xf32, #blocked1>
      %3 = ttg.convert_layout %outLHS_40 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> -> tensor<128x32xf32, #blocked1>
      %4 = tt.reshape %outRHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> -> tensor<128x2x32xf32, #linear4>
      %5 = tt.trans %4 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #linear4> -> tensor<128x32x2xf32, #linear5>
      %outLHS_42, %outRHS_43 = tt.split %5 {ttg.partition = array<i32: 0>} : tensor<128x32x2xf32, #linear5> -> tensor<128x32xf32, #linear6>
      %6 = ttg.convert_layout %outRHS_43 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> -> tensor<128x32xf32, #blocked1>
      %7 = ttg.convert_layout %outLHS_42 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> -> tensor<128x32xf32, #blocked1>
      %bias00 = tt.descriptor_load %bias_desc[%offs_cm, %offs_cn] {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared1>> -> tensor<128x32xf16, #blocked1>
      %bias00_44 = arith.extf %bias00 {ttg.partition = array<i32: 3>} : tensor<128x32xf16, #blocked1> to tensor<128x32xf32, #blocked1>
      %acc00 = arith.addf %3, %bias00_44 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #blocked1>
      %c00 = arith.truncf %acc00 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %8 = ttg.local_alloc %c00 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked1>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %9 = ttng.async_tma_copy_local_to_global %c_desc[%offs_cm, %offs_cn] %8 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %9   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %bias01 = arith.addi %offs_cn, %c32_i32 : i32
      %bias01_45 = tt.descriptor_load %bias_desc[%offs_cm, %bias01] {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared1>> -> tensor<128x32xf16, #blocked1>
      %bias01_46 = arith.extf %bias01_45 {ttg.partition = array<i32: 3>} : tensor<128x32xf16, #blocked1> to tensor<128x32xf32, #blocked1>
      %acc01 = arith.addf %2, %bias01_46 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #blocked1>
      %c01 = arith.truncf %acc01 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %10 = ttg.local_alloc %c01 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked1>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %11 = ttng.async_tma_copy_local_to_global %c_desc[%offs_cm, %bias01] %10 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %11   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %bias10 = arith.addi %offs_cn, %c64_i32 : i32
      %bias10_47 = tt.descriptor_load %bias_desc[%offs_cm, %bias10] {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared1>> -> tensor<128x32xf16, #blocked1>
      %bias10_48 = arith.extf %bias10_47 {ttg.partition = array<i32: 3>} : tensor<128x32xf16, #blocked1> to tensor<128x32xf32, #blocked1>
      %acc10 = arith.addf %7, %bias10_48 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #blocked1>
      %c10 = arith.truncf %acc10 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %12 = ttg.local_alloc %c10 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked1>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %13 = ttng.async_tma_copy_local_to_global %c_desc[%offs_cm, %bias10] %12 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %13   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %bias11 = arith.addi %offs_cn, %c96_i32 : i32
      %bias11_49 = tt.descriptor_load %bias_desc[%offs_cm, %bias11] {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x32xf16, #shared1>> -> tensor<128x32xf16, #blocked1>
      %bias11_50 = arith.extf %bias11_49 {ttg.partition = array<i32: 3>} : tensor<128x32xf16, #blocked1> to tensor<128x32xf32, #blocked1>
      %acc11 = arith.addf %6, %bias11_50 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #blocked1>
      %c11 = arith.truncf %acc11 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %14 = ttg.local_alloc %c11 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked1>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %15 = ttng.async_tma_copy_local_to_global %c_desc[%offs_cm, %bias11] %14 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x32xf16, #shared1>>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %15   {ttg.partition = array<i32: 2>} : !ttg.async.token
      scf.yield %tile_id_c_28 : i32
    } {tt.data_partition_factor = 1 : i32, tt.disallow_acc_multi_buffer, tt.separate_epilogue_store = true, tt.smem_alloc_algo = 0 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
