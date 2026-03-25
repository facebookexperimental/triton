// RUN: triton-opt %s --nvgpu-test-ws-memory-planner=num-buffers=2 --mlir-print-debuginfo --mlir-print-local-scope | FileCheck %s

// Test: When data partitioning splits the M dimension (factor=2), the inner
// k-loop has 3 SMEM operands per iteration: a_0 (half 0 of A), a_1 (half 1
// of A), and b (full B tile). All three share the same element type (f16) and
// are in the innermost loop, so algorithm 0 assigns them the same buffer.id.
//
// With num-buffers=2, algorithm 0 would naively set buffer.copy=2 for all
// three. But 3 entries sharing 2 buffer slots causes index collisions:
//   (accumCnt + 0) % 2 == (accumCnt + 2) % 2
// leading to a deadlock where the load partition blocks waiting for a slot
// that the MMA partition also needs.
//
// The fix enforces buffer.copy >= number of entries per buffer.id, so
// buffer.copy is bumped from 2 to 3 for all three allocs.

// CHECK-LABEL: @matmul_kernel_tma_persistent
//
// The two epilogue buffers each get their own buffer.id with buffer.copy=1:
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id =
// CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id =
//
// All three innermost-loop SMEM allocs get the same buffer.id and buffer.copy=3
// (bumped from 2 because there are 3 entries sharing the reuse group):
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = [[ID:[0-9]+]]
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = [[ID]]
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = [[ID]]

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#loc = loc("test.py":1:0)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#loc1 = loc(unknown)
#loc5 = loc(unknown)
#loc30 = loc(unknown)
#loc36 = loc(unknown)
#loc37 = loc(unknown)
#loc45 = loc("_1"(#loc))
#loc46 = loc("_0"(#loc))
#loc47 = loc("arg2"(#loc))
#loc48 = loc("a_1"(#loc))
#loc49 = loc("a_0"(#loc))
#loc50 = loc("accumulator_1"(#loc))
#loc51 = loc("accumulator_0"(#loc))
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent(%a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>> loc("a_desc"(#loc)), %a_desc_0: i32 loc("a_desc"(#loc)), %a_desc_1: i32 loc("a_desc"(#loc)), %a_desc_2: i64 loc("a_desc"(#loc)), %a_desc_3: i64 loc("a_desc"(#loc)), %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>> loc("b_desc"(#loc)), %b_desc_4: i32 loc("b_desc"(#loc)), %b_desc_5: i32 loc("b_desc"(#loc)), %b_desc_6: i64 loc("b_desc"(#loc)), %b_desc_7: i64 loc("b_desc"(#loc)), %c_desc_or_ptr: !tt.tensordesc<tensor<128x128xf16, #shared>> loc("c_desc_or_ptr"(#loc)), %c_desc_or_ptr_8: i32 loc("c_desc_or_ptr"(#loc)), %c_desc_or_ptr_9: i32 loc("c_desc_or_ptr"(#loc)), %c_desc_or_ptr_10: i64 loc("c_desc_or_ptr"(#loc)), %c_desc_or_ptr_11: i64 loc("c_desc_or_ptr"(#loc)), %M: i32 {tt.divisibility = 16 : i32} loc("M"(#loc)), %N: i32 {tt.divisibility = 16 : i32} loc("N"(#loc)), %K: i32 {tt.divisibility = 16 : i32} loc("K"(#loc)), %stride_cm: i32 {tt.divisibility = 16 : i32} loc("stride_cm"(#loc))) attributes {noinline = false} {
    %_1 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc45)
    %_0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc46)
    %arg2 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc47)
    %a_1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc48)
    %a_0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc49)
    %accumulator_1, %accumulator_1_12 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc50)
    %accumulator_0, %accumulator_0_13 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc51)
    %false = arith.constant {async_task_id = array<i32: 1>} false loc(#loc5)
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true loc(#loc5)
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 148 : i32 loc(#loc5)
    %c8_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 8 : i32 loc(#loc5)
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 256 : i32 loc(#loc5)
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 128 : i32 loc(#loc5)
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 64 : i32 loc(#loc5)
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 0 : i32 loc(#loc5)
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 1 : i32 loc(#loc5)
    %num_pid_m = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 255 : i32 loc(#loc5)
    %num_pid_n = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 127 : i32 loc(#loc5)
    %k_tiles = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 63 : i32 loc(#loc5)
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #blocked> loc(#loc5)
    %start_pid = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc5)
    %num_pid_m_14 = arith.addi %M, %num_pid_m {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc5)
    %num_pid_m_15 = arith.divsi %num_pid_m_14, %c256_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc5)
    %num_pid_n_16 = arith.addi %N, %num_pid_n {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc5)
    %num_pid_n_17 = arith.divsi %num_pid_n_16, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc5)
    %k_tiles_18 = arith.addi %K, %k_tiles {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc5)
    %k_tiles_19 = arith.divsi %k_tiles_18, %c64_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc5)
    %num_tiles = arith.muli %num_pid_m_15, %num_pid_n_17 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc5)
    %tile_id_c = arith.subi %start_pid, %c148_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc5)
    %num_pid_in_group = arith.muli %num_pid_n_17, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32 loc(#loc5)
    %tile_id_c_20 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%tile_id_c_21 = %tile_id_c) -> (i32)  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group {async_task_id = array<i32: 2>} : i32 loc(#loc5)
      %first_pid_m = arith.muli %group_id, %c8_i32 {async_task_id = array<i32: 2>} : i32 loc(#loc5)
      %group_size_m = arith.subi %num_pid_m_15, %first_pid_m {async_task_id = array<i32: 2>} : i32 loc(#loc5)
      %group_size_m_22 = arith.minsi %group_size_m, %c8_i32 {async_task_id = array<i32: 2>} : i32 loc(#loc5)
      %pid_m = arith.remsi %tile_id, %group_size_m_22 {async_task_id = array<i32: 2>} : i32 loc(#loc5)
      %pid_m_23 = arith.addi %first_pid_m, %pid_m {async_task_id = array<i32: 2>} : i32 loc(#loc5)
      %pid_n = arith.remsi %tile_id, %num_pid_in_group {async_task_id = array<i32: 2>} : i32 loc(#loc5)
      %pid_n_24 = arith.divsi %pid_n, %group_size_m_22 {async_task_id = array<i32: 2>} : i32 loc(#loc5)
      %offs_am = arith.muli %pid_m_23, %c256_i32 {async_task_id = array<i32: 2>} : i32 loc(#loc5)
      %a = arith.addi %offs_am, %c128_i32 {async_task_id = array<i32: 2>} : i32 loc(#loc5)
      %offs_bn = arith.muli %pid_n_24, %c128_i32 {async_task_id = array<i32: 2>} : i32 loc(#loc5)
      %accumulator = ttng.tmem_store %cst, %accumulator_0[%accumulator_0_13], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc1)
      %accumulator_25 = ttng.tmem_store %cst, %accumulator_1[%accumulator_1_12], %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc1)
      %accumulator_26:3 = scf.for %accumulator_42 = %c0_i32 to %k_tiles_19 step %c1_i32 iter_args(%arg22 = %false, %accumulator_43 = %accumulator, %accumulator_44 = %accumulator_25) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %accumulator_42, %c64_i32 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 loc(#loc5)
        %a_45 = tt.descriptor_load %a_desc[%offs_am, %offs_k] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1> loc(#loc5)
        %a_46 = tt.descriptor_load %a_desc[%a, %offs_k] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1> loc(#loc5)
        ttg.local_store %a_45, %a_0 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc49)
        ttg.local_store %a_46, %a_1 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc48)
        %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1> loc(#loc5)
        ttg.local_store %b, %arg2 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc47)
        %arg2_47 = ttg.memdesc_trans %arg2 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> loc(#loc47)
        %accumulator_48 = ttng.tc_gen5_mma %a_0, %arg2_47, %accumulator_0[%accumulator_43], %arg22, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc1)
        %accumulator_49 = ttng.tc_gen5_mma %a_1, %arg2_47, %accumulator_1[%accumulator_44], %arg22, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc1)
        scf.yield {async_task_id = array<i32: 0, 1, 4>} %true, %accumulator_48, %accumulator_49 : i1, !ttg.async.token, !ttg.async.token loc(#loc30)
      } {async_task_id = array<i32: 0, 1, 2, 3, 4>, tt.scheduled_max_stage = 2 : i32} loc(#loc5)
      %tile_id_c_27 = arith.addi %tile_id_c_21, %c148_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc5)
      %group_id_28 = arith.divsi %tile_id_c_27, %num_pid_in_group {async_task_id = array<i32: 3>} : i32 loc(#loc5)
      %first_pid_m_29 = arith.muli %group_id_28, %c8_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc5)
      %group_size_m_30 = arith.subi %num_pid_m_15, %first_pid_m_29 {async_task_id = array<i32: 3>} : i32 loc(#loc5)
      %group_size_m_31 = arith.minsi %group_size_m_30, %c8_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc5)
      %pid_m_32 = arith.remsi %tile_id_c_27, %group_size_m_31 {async_task_id = array<i32: 3>} : i32 loc(#loc5)
      %pid_m_33 = arith.addi %first_pid_m_29, %pid_m_32 {async_task_id = array<i32: 3>} : i32 loc(#loc5)
      %pid_n_34 = arith.remsi %tile_id_c_27, %num_pid_in_group {async_task_id = array<i32: 3>} : i32 loc(#loc5)
      %pid_n_35 = arith.divsi %pid_n_34, %group_size_m_31 {async_task_id = array<i32: 3>} : i32 loc(#loc5)
      %offs_am_c = arith.muli %pid_m_33, %c256_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc5)
      %0 = arith.addi %offs_am_c, %c128_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc1)
      %offs_bn_c = arith.muli %pid_n_35, %c128_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc5)
      %accumulator_36, %accumulator_37 = ttng.tmem_load %accumulator_0[%accumulator_26#1] {async_task_id = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked> loc(#loc1)
      %accumulator_38, %accumulator_39 = ttng.tmem_load %accumulator_1[%accumulator_26#2] {async_task_id = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked> loc(#loc1)
      %accumulator_40 = arith.truncf %accumulator_36 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked> loc(#loc5)
      %accumulator_41 = arith.truncf %accumulator_38 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked> loc(#loc5)
      %1 = ttg.convert_layout %accumulator_40 {async_task_id = array<i32: 4>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2> loc(#loc1)
      %2 = ttg.convert_layout %accumulator_41 {async_task_id = array<i32: 4>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2> loc(#loc1)
      ttg.local_store %1, %_0 {async_task_id = array<i32: 4>} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc1)
      ttng.fence_async_shared {bCluster = false} loc(#loc1)
      %3 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%offs_am_c, %offs_bn_c] %_0 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token loc(#loc1)
      ttng.async_tma_store_token_wait %3   {async_task_id = array<i32: 3>} : !ttg.async.token loc(#loc1)
      ttg.local_store %2, %_1 {async_task_id = array<i32: 4>} : tensor<128x128xf16, #blocked2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc1)
      ttng.fence_async_shared {bCluster = false} loc(#loc1)
      %4 = ttng.async_tma_copy_local_to_global %c_desc_or_ptr[%0, %offs_bn_c] %_1 {async_task_id = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token loc(#loc1)
      ttng.async_tma_store_token_wait %4   {async_task_id = array<i32: 3>} : !ttg.async.token loc(#loc1)
      scf.yield {async_task_id = array<i32: 3>} %tile_id_c_27 : i32 loc(#loc36)
    } {async_task_id = array<i32: 0, 1, 2, 3, 4>, tt.data_partition_factor = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["default", "gemm", "load", "epilogue", "computation"], ttg.warp_specialize.tag = 0 : i32} loc(#loc5)
    tt.return loc(#loc37)
  } loc(#loc)
} loc(#loc)
