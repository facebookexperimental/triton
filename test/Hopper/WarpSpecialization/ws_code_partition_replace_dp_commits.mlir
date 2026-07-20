// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=3" | FileCheck %s
// XFAIL: *

// Regression test for B-13-F1 / T273488617.
//
// Test: data-partitioned D-channel commits for a persistent GEMM with
// tt.data_partition_factor = 2, producing two tc_gen5_mma ops in the inner
// k-loop.
//
// With multiple MMAs in the loop, D-channel completion tracking should avoid
// global tc_gen5_commit ops. Each MMA should wait on its A/B inline barrier
// and arrive on the corresponding D barrier.

// CHECK-LABEL: @matmul_kernel_tma_persistent
// CHECK: ttg.warp_specialize
//
// GEMM partition (partition0, task 1):
// CHECK: partition0
// CHECK: scf.for
// Inner k-loop with two MMAs (data_partition_factor = 2):
// CHECK: scf.for
// CHECK: ttng.tc_gen5_mma
// CHECK: ttng.tc_gen5_mma
// The k-loop ends:
// CHECK: scf.yield
//
// After the inner k-loop: D-channel completion must use per-MMA wait/arrive
// pairs, with no raw global tc_gen5_commit fallback before the outer yield.
//
// CHECK-NOT: ttng.tc_gen5_commit
// CHECK: ttng.wait_barrier
// CHECK-NOT: ttng.tc_gen5_commit
// CHECK: ttng.arrive_barrier
// CHECK-NOT: ttng.tc_gen5_commit
// CHECK: ttng.wait_barrier
// CHECK-NOT: ttng.tc_gen5_commit
// CHECK: ttng.arrive_barrier
// CHECK-NOT: ttng.tc_gen5_commit
//
// Outer loop yield:
// CHECK: scf.yield

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent(%a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %a_desc_0: i32, %a_desc_1: i32, %a_desc_2: i64, %a_desc_3: i64, %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %b_desc_4: i32, %b_desc_5: i32, %b_desc_6: i64, %b_desc_7: i64, %c_desc_or_ptr: !tt.tensordesc<tensor<128x128xf16, #shared>>, %c_desc_or_ptr_8: i32, %c_desc_or_ptr_9: i32, %c_desc_or_ptr_10: i64, %c_desc_or_ptr_11: i64, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %_1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %_0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %arg2 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %a_1 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %a_0 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %accumulator_1, %accumulator_1_12 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %accumulator_0, %accumulator_0_13 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %false = arith.constant {ttg.partition = array<i32: 1>} false
    %true = arith.constant {ttg.partition = array<i32: 0, 1>} true
    %c148_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 148 : i32
    %c8_i32 = arith.constant {ttg.partition = array<i32: 2, 3>} 8 : i32
    %c256_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 256 : i32
    %c128_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 128 : i32
    %c64_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 64 : i32
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 0 : i32
    %c1_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 1 : i32
    %num_pid_m = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 255 : i32
    %num_pid_n = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 127 : i32
    %k_tiles = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 63 : i32
    %cst = arith.constant {ttg.partition = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %start_pid = tt.get_program_id x {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_m_14 = arith.addi %M, %num_pid_m {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_m_15 = arith.divsi %num_pid_m_14, %c256_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_n_16 = arith.addi %N, %num_pid_n {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_pid_n_17 = arith.divsi %num_pid_n_16, %c128_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i32
    %k_tiles_18 = arith.addi %K, %k_tiles {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i32
    %k_tiles_19 = arith.divsi %k_tiles_18, %c64_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i32
    %num_tiles = arith.muli %num_pid_m_15, %num_pid_n_17 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i32
    %tile_id_c = arith.subi %start_pid, %c148_i32 {ttg.partition = array<i32: 3>} : i32
    %num_pid_in_group = arith.muli %num_pid_n_17, %c8_i32 {ttg.partition = array<i32: 2, 3>} : i32
    %tile_id_c_20 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%tile_id_c_21 = %tile_id_c) -> (i32)  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group {ttg.partition = array<i32: 2>} : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 {ttg.partition = array<i32: 2>} : i32
      %group_size_m = arith.subi %num_pid_m_15, %first_pid_m {ttg.partition = array<i32: 2>} : i32
      %group_size_m_22 = arith.minsi %group_size_m, %c8_i32 {ttg.partition = array<i32: 2>} : i32
      %pid_m = arith.remsi %tile_id, %group_size_m_22 {ttg.partition = array<i32: 2>} : i32
      %pid_m_23 = arith.addi %first_pid_m, %pid_m {ttg.partition = array<i32: 2>} : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group {ttg.partition = array<i32: 2>} : i32
      %pid_n_24 = arith.divsi %pid_n, %group_size_m_22 {ttg.partition = array<i32: 2>} : i32
      %offs_am = arith.muli %pid_m_23, %c256_i32 {ttg.partition = array<i32: 2>} : i32
      %a = arith.addi %offs_am, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %offs_bn = arith.muli %pid_n_24, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %accumulator = ttng.tmem_store %cst, %accumulator_0[%accumulator_0_13], %true {ttg.partition = array<i32: 0>, tmem.start = array<i32: 8, 10>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %accumulator_25 = ttng.tmem_store %cst, %accumulator_1[%accumulator_1_12], %true {ttg.partition = array<i32: 0>, tmem.start = array<i32: 5, 7>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %accumulator_26:3 = scf.for %accumulator_42 = %c0_i32 to %k_tiles_19 step %c1_i32 iter_args(%arg22 = %false, %accumulator_43 = %accumulator, %accumulator_44 = %accumulator_25) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %accumulator_42, %c64_i32 {ttg.partition = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %a_45 = tt.descriptor_load %a_desc[%offs_am, %offs_k] {ttg.partition = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        %a_46 = tt.descriptor_load %a_desc[%a, %offs_k] {ttg.partition = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        ttg.local_store %a_45, %a_0 {ttg.partition = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttg.local_store %a_46, %a_1 {ttg.partition = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {ttg.partition = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        ttg.local_store %b, %arg2 {ttg.partition = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %arg2_47 = ttg.memdesc_trans %arg2 {ttg.partition = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
        %accumulator_48 = ttng.tc_gen5_mma %a_0, %arg2_47, %accumulator_0[%accumulator_43], %arg22, %true {ttg.partition = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, tmem.end = array<i32: 8>, tmem.start = array<i32: 9>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %accumulator_49 = ttng.tc_gen5_mma %a_1, %arg2_47, %accumulator_1[%accumulator_44], %arg22, %true {ttg.partition = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, tmem.end = array<i32: 5>, tmem.start = array<i32: 6>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 0, 1, 4>} %true, %accumulator_48, %accumulator_49 : i1, !ttg.async.token, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1, 2, 3, 4>, tt.scheduled_max_stage = 2 : i32}
      %tile_id_c_27 = arith.addi %tile_id_c_21, %c148_i32 {ttg.partition = array<i32: 3>} : i32
      %group_id_28 = arith.divsi %tile_id_c_27, %num_pid_in_group {ttg.partition = array<i32: 3>} : i32
      %first_pid_m_29 = arith.muli %group_id_28, %c8_i32 {ttg.partition = array<i32: 3>} : i32
      %group_size_m_30 = arith.subi %num_pid_m_15, %first_pid_m_29 {ttg.partition = array<i32: 3>} : i32
      %group_size_m_31 = arith.minsi %group_size_m_30, %c8_i32 {ttg.partition = array<i32: 3>} : i32
      %pid_m_32 = arith.remsi %tile_id_c_27, %group_size_m_31 {ttg.partition = array<i32: 3>} : i32
      %pid_m_33 = arith.addi %first_pid_m_29, %pid_m_32 {ttg.partition = array<i32: 3>} : i32
      %pid_n_34 = arith.remsi %tile_id_c_27, %num_pid_in_group {ttg.partition = array<i32: 3>} : i32
      %pid_n_35 = arith.divsi %pid_n_34, %group_size_m_31 {ttg.partition = array<i32: 3>} : i32
      %offs_am_c = arith.muli %pid_m_33, %c256_i32 {ttg.partition = array<i32: 3>} : i32
      %0 = arith.addi %offs_am_c, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %offs_bn_c = arith.muli %pid_n_35, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %accumulator_36, %accumulator_37 = ttng.tmem_load %accumulator_0[%accumulator_26#1] {ttg.partition = array<i32: 4>, tmem.end = array<i32: 9, 10>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %accumulator_38, %accumulator_39 = ttng.tmem_load %accumulator_1[%accumulator_26#2] {ttg.partition = array<i32: 4>, tmem.end = array<i32: 6, 7>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %accumulator_40 = arith.truncf %accumulator_36 {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      %accumulator_41 = arith.truncf %accumulator_38 {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      %1 = ttg.convert_layout %accumulator_40 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
      %2 = ttg.convert_layout %accumulator_41 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
      ttg.local_store %1, %_0 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttng.fence_async_shared {bCluster = false}
      %3 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%offs_am_c, %offs_bn_c] %_0 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %3   {ttg.partition = array<i32: 3>} : !ttg.async.token
      ttg.local_store %2, %_1 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttng.fence_async_shared {bCluster = false}
      %4 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%0, %offs_bn_c] %_1 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %4   {ttg.partition = array<i32: 3>} : !ttg.async.token
      scf.yield {ttg.partition = array<i32: 3>} %tile_id_c_27 : i32
    } {ttg.partition = array<i32: 0, 1, 2, 3, 4>, tt.data_partition_factor = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["default", "gemm", "load", "epilogue", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
