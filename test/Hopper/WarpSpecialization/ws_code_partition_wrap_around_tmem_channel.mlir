// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=4 post-channel-creation=1" | FileCheck %s

// Test: In a warp-specialized persistent GEMM, three ops in separate partitions
// share the same TMEM accumulator buffer:
//   tmem_store (T0) → tc_gen5_mma (T1) → tmem_load (T4)
//
// The consecutive channels (6: T0→T1, 7: T1→T4) are not sufficient: the
// wrap-around channel (8: T0→T4) is needed so that tmem_load signals
// tmem_store via the Empty barrier before the next outer-loop iteration
// overwrites the buffer.
//
// Verify that:
// - default partition (T0) has 2 wait + 2 arrive barriers around tmem_store
// - partition with tmem_load (T4) has 2 wait + 2 arrive barriers around tmem_load

// CHECK-LABEL: @matmul_kernel_tma_persistent
// CHECK: ttg.warp_specialize
//
// default partition (T0): tmem_store with barriers for channels 6 (T0→T1)
// and 8 (T0→T4 wrap-around). One channel uses mbarrier, the other uses
// nvws tokens.
// CHECK: default
// CHECK: ttng.wait_barrier
// CHECK: nvws.producer_acquire
// CHECK: ttng.tmem_store
// CHECK: nvws.producer_commit
// CHECK: nvws.producer_commit
//
// partition0 (T1): MMA consumer
// CHECK: partition0
// CHECK: ttng.tc_gen5_mma
//
// partition1 (T2): producer TMA copies
// CHECK: partition1
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.async_tma_copy_global_to_local
//
// partition2 (T3): epilogue descriptor stores
// CHECK: partition2
// CHECK: tt.descriptor_store
//
// partition3 (T4): tmem_load with barriers for channels 7 (T1→T4) and
// 8 (T0→T4 wrap-around). Without the wrap-around channel, there would be
// only 1 wait/release pair here.
// CHECK: partition3
// CHECK: ttng.wait_barrier
// CHECK: nvws.consumer_wait
// CHECK: ttng.tmem_load
// CHECK: nvws.consumer_release
// CHECK: nvws.consumer_release

#blocked = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent(%a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %a_desc_0: i32, %a_desc_1: i32, %a_desc_2: i64, %a_desc_3: i64, %b_desc: !tt.tensordesc<tensor<64x256xf16, #shared>>, %b_desc_4: i32, %b_desc_5: i32, %b_desc_6: i64, %b_desc_7: i64, %c_desc_or_ptr: !tt.tensordesc<tensor<128x64xf16, #shared>>, %c_desc_or_ptr_8: i32, %c_desc_or_ptr_9: i32, %c_desc_or_ptr_10: i64, %c_desc_or_ptr_11: i64, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2 = ttg.local_alloc {async_task_id = array<i32: 4>, buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c3 = ttg.local_alloc {async_task_id = array<i32: 4>, buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c0 = ttg.local_alloc {async_task_id = array<i32: 4>, buffer.copy = 1 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c1 = ttg.local_alloc {async_task_id = array<i32: 4>, buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 4 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
    %a = ttg.local_alloc {buffer.copy = 4 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %accumulator, %accumulator_12 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 148 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 128 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 256 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 64 : i32
    %c192_i32 = arith.constant {async_task_id = array<i32: 3>} 192 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 1 : i32
    %num_pid_m = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 127 : i32
    %num_pid_n = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 255 : i32
    %k_tiles = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 63 : i32
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %start_pid = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_m_13 = arith.addi %M, %num_pid_m {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_m_14 = arith.divsi %num_pid_m_13, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_n_15 = arith.addi %N, %num_pid_n {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_n_16 = arith.divsi %num_pid_n_15, %c256_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %k_tiles_17 = arith.addi %K, %k_tiles {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %k_tiles_18 = arith.divsi %k_tiles_17, %c64_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_tiles = arith.muli %num_pid_m_14, %num_pid_n_16 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32
    %tile_id_c = arith.subi %start_pid, %c148_i32 {async_task_id = array<i32: 3>} : i32
    %num_pid_in_group = arith.muli %num_pid_n_16, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
    %tile_id_c_19 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%tile_id_c_20 = %tile_id_c) -> (i32)  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group {async_task_id = array<i32: 2>} : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %group_size_m = arith.subi %num_pid_m_14, %first_pid_m {async_task_id = array<i32: 2>} : i32
      %group_size_m_21 = arith.minsi %group_size_m, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %pid_m = arith.remsi %tile_id, %group_size_m_21 {async_task_id = array<i32: 2>} : i32
      %pid_m_22 = arith.addi %first_pid_m, %pid_m {async_task_id = array<i32: 2>} : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group {async_task_id = array<i32: 2>} : i32
      %pid_n_23 = arith.divsi %pid_n, %group_size_m_21 {async_task_id = array<i32: 2>} : i32
      %offs_am = arith.muli %pid_m_22, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %offs_bn = arith.muli %pid_n_23, %c256_i32 {async_task_id = array<i32: 2>} : i32
      %accumulator_24 = ttng.tmem_store %cst, %accumulator[%accumulator_12], %true {async_task_id = array<i32: 0>, tmem.start = array<i32: 6, 8>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %accumulator_25:2 = scf.for %accumulator_56 = %c0_i32 to %k_tiles_18 step %c1_i32 iter_args(%arg22 = %false, %accumulator_57 = %accumulator_24) -> (i1, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %accumulator_56, %c64_i32 {async_task_id = array<i32: 2>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
        %a_58 = tt.descriptor_load %a_desc[%offs_am, %offs_k] {async_task_id = array<i32: 2>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        ttg.local_store %a_58, %a {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %b_59 = tt.descriptor_load %b_desc[%offs_k, %offs_bn] {async_task_id = array<i32: 2>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<64x256xf16, #shared>> -> tensor<64x256xf16, #blocked2>
        ttg.local_store %b_59, %b {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64x256xf16, #blocked2> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
        %accumulator_60 = ttng.tc_gen5_mma %a, %b, %accumulator[%accumulator_57], %arg22, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 3 : i32, tmem.end = array<i32: 6>, tmem.start = array<i32: 7>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1, 4>} %true, %accumulator_60 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3, 4>, tt.scheduled_max_stage = 3 : i32}
      %tile_id_c_26 = arith.addi %tile_id_c_20, %c148_i32 {async_task_id = array<i32: 3>} : i32
      %group_id_27 = arith.divsi %tile_id_c_26, %num_pid_in_group {async_task_id = array<i32: 3>} : i32
      %first_pid_m_28 = arith.muli %group_id_27, %c8_i32 {async_task_id = array<i32: 3>} : i32
      %group_size_m_29 = arith.subi %num_pid_m_14, %first_pid_m_28 {async_task_id = array<i32: 3>} : i32
      %group_size_m_30 = arith.minsi %group_size_m_29, %c8_i32 {async_task_id = array<i32: 3>} : i32
      %pid_m_31 = arith.remsi %tile_id_c_26, %group_size_m_30 {async_task_id = array<i32: 3>} : i32
      %pid_m_32 = arith.addi %first_pid_m_28, %pid_m_31 {async_task_id = array<i32: 3>} : i32
      %pid_n_33 = arith.remsi %tile_id_c_26, %num_pid_in_group {async_task_id = array<i32: 3>} : i32
      %pid_n_34 = arith.divsi %pid_n_33, %group_size_m_30 {async_task_id = array<i32: 3>} : i32
      %offs_am_c = arith.muli %pid_m_32, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %offs_bn_c = arith.muli %pid_n_34, %c256_i32 {async_task_id = array<i32: 3>} : i32
      %accumulator_35, %accumulator_36 = ttng.tmem_load %accumulator[%accumulator_25#1] {async_task_id = array<i32: 4>, tmem.end = array<i32: 7, 8>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
      %acc = tt.reshape %accumulator_35 {async_task_id = array<i32: 4>} : tensor<128x256xf32, #blocked> -> tensor<128x2x128xf32, #blocked3>
      %acc_37 = tt.trans %acc {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked3> -> tensor<128x128x2xf32, #blocked4>
      %outLHS, %outRHS = tt.split %acc_37 {async_task_id = array<i32: 4>} : tensor<128x128x2xf32, #blocked4> -> tensor<128x128xf32, #blocked5>
      %acc_hi = tt.reshape %outRHS {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked5> -> tensor<128x2x64xf32, #blocked6>
      %acc_lo = tt.reshape %outLHS {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked5> -> tensor<128x2x64xf32, #blocked6>
      %acc_lo_38 = tt.trans %acc_lo {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked6> -> tensor<128x64x2xf32, #blocked7>
      %outLHS_39, %outRHS_40 = tt.split %acc_lo_38 {async_task_id = array<i32: 4>} : tensor<128x64x2xf32, #blocked7> -> tensor<128x64xf32, #blocked8>
      %c1_41 = arith.truncf %outRHS_40 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked8> to tensor<128x64xf16, #blocked8>
      ttg.local_store %c1_41, %c1 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked8> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %c0_42 = arith.truncf %outLHS_39 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked8> to tensor<128x64xf16, #blocked8>
      ttg.local_store %c0_42, %c0 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked8> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %acc_hi_43 = tt.trans %acc_hi {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked6> -> tensor<128x64x2xf32, #blocked7>
      %outLHS_44, %outRHS_45 = tt.split %acc_hi_43 {async_task_id = array<i32: 4>} : tensor<128x64x2xf32, #blocked7> -> tensor<128x64xf32, #blocked8>
      %c3_46 = arith.truncf %outRHS_45 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked8> to tensor<128x64xf16, #blocked8>
      ttg.local_store %c3_46, %c3 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked8> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %c2_47 = arith.truncf %outLHS_44 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #blocked8> to tensor<128x64xf16, #blocked8>
      ttg.local_store %c2_47, %c2 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #blocked8> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %c0_48 = ttg.local_load %c0 {async_task_id = array<i32: 3>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked8>
      %c0_49 = ttg.convert_layout %c0_48 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked8> -> tensor<128x64xf16, #blocked1>
      tt.descriptor_store %c_desc_or_ptr[%offs_am_c, %offs_bn_c], %c0_49 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked1>
      %c1_50 = ttg.local_load %c1 {async_task_id = array<i32: 3>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked8>
      %c1_51 = ttg.convert_layout %c1_50 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked8> -> tensor<128x64xf16, #blocked1>
      %0 = arith.addi %offs_bn_c, %c64_i32 {async_task_id = array<i32: 3>} : i32
      tt.descriptor_store %c_desc_or_ptr[%offs_am_c, %0], %c1_51 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked1>
      %c2_52 = ttg.local_load %c2 {async_task_id = array<i32: 3>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked8>
      %c2_53 = ttg.convert_layout %c2_52 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked8> -> tensor<128x64xf16, #blocked1>
      %1 = arith.addi %offs_bn_c, %c128_i32 {async_task_id = array<i32: 3>} : i32
      tt.descriptor_store %c_desc_or_ptr[%offs_am_c, %1], %c2_53 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked1>
      %c3_54 = ttg.local_load %c3 {async_task_id = array<i32: 3>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked8>
      %c3_55 = ttg.convert_layout %c3_54 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked8> -> tensor<128x64xf16, #blocked1>
      %2 = arith.addi %offs_bn_c, %c192_i32 {async_task_id = array<i32: 3>} : i32
      tt.descriptor_store %c_desc_or_ptr[%offs_am_c, %2], %c3_55 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked1>
      scf.yield {async_task_id = array<i32: 3>} %tile_id_c_26 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3, 4>, tt.data_partition_factor = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
