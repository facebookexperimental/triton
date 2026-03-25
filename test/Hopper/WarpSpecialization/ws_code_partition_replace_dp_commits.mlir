// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=3 post-channel-creation=1" | FileCheck %s

// Test: data-partitioned D-channel commits use raw barrier allocs (1x1xi64)
// rather than indexed barriers or wait+arrive replacement. Using indexed
// barriers in tc_gen5_commit caused GPU deadlocks at runtime.
//
// The input has a persistent GEMM with tt.data_partition_factor = 2, producing
// two tc_gen5_mma ops (both with async_task_id {1}) in the inner k-loop.
// After the k-loop, the pass emits tc_gen5_commit ops with raw 1x1xi64 barrier
// allocs (no memdesc_index indexing, no wait_barrier+arrive_barrier replacement).

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
// After the inner k-loop, tc_gen5_commit ops are emitted with raw barrier
// allocs (1x1xi64). No wait_barrier+arrive_barrier replacement.
//
// CHECK: ttng.tc_gen5_commit {{%[a-z0-9_]+}} {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64
// CHECK: ttng.tc_gen5_commit {{%[a-z0-9_]+}} {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64
// CHECK: ttng.tc_gen5_commit {{%[a-z0-9_]+}} {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64
//
// No wait+arrive replacement pattern:
// CHECK-NOT: ttng.arrive_barrier
//
// Outer loop yield:
// CHECK: scf.yield

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
#loc38 = loc("a_desc"(#loc))
#loc39 = loc("b_desc"(#loc))
#loc40 = loc("c_desc_or_ptr"(#loc))
#loc41 = loc("M"(#loc))
#loc42 = loc("N"(#loc))
#loc43 = loc("K"(#loc))
#loc44 = loc("stride_cm"(#loc))
#loc45 = loc("_1"(#loc))
#loc46 = loc("_0"(#loc))
#loc47 = loc("arg2"(#loc))
#loc48 = loc("a_1"(#loc))
#loc49 = loc("a_0"(#loc))
#loc50 = loc("accumulator_1"(#loc))
#loc51 = loc("accumulator_0"(#loc))
#loc55 = loc(unknown)
#loc56 = loc(unknown)
#loc57 = loc(unknown)
#loc58 = loc(unknown)
#loc59 = loc(unknown)
#loc68 = loc(unknown)
#loc69 = loc(unknown)
#loc70 = loc(unknown)
#loc71 = loc(unknown)
#loc72 = loc(unknown)
#loc73 = loc(unknown)
#loc74 = loc(unknown)
#loc75 = loc(unknown)
#loc76 = loc(unknown)
#loc77 = loc(unknown)
#loc78 = loc(unknown)
#loc79 = loc(unknown)
#loc80 = loc(unknown)
#loc81 = loc(unknown)
#loc82 = loc(unknown)
#loc83 = loc(unknown)
#loc84 = loc(unknown)
#loc85 = loc(unknown)
#loc86 = loc(unknown)
#loc87 = loc(unknown)
#loc88 = loc(unknown)
#loc89 = loc(unknown)
#loc90 = loc(unknown)
#loc91 = loc(unknown)
#loc92 = loc(unknown)
#loc93 = loc(unknown)
#loc94 = loc(unknown)
#loc95 = loc(unknown)
#loc96 = loc(unknown)
#loc97 = loc(unknown)
#loc98 = loc(unknown)
#loc99 = loc(unknown)
#loc100 = loc(unknown)
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent(%a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>> loc("a_desc"(#loc)), %a_desc_0: i32 loc("a_desc"(#loc)), %a_desc_1: i32 loc("a_desc"(#loc)), %a_desc_2: i64 loc("a_desc"(#loc)), %a_desc_3: i64 loc("a_desc"(#loc)), %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>> loc("b_desc"(#loc)), %b_desc_4: i32 loc("b_desc"(#loc)), %b_desc_5: i32 loc("b_desc"(#loc)), %b_desc_6: i64 loc("b_desc"(#loc)), %b_desc_7: i64 loc("b_desc"(#loc)), %c_desc_or_ptr: !tt.tensordesc<tensor<128x128xf16, #shared>> loc("c_desc_or_ptr"(#loc)), %c_desc_or_ptr_8: i32 loc("c_desc_or_ptr"(#loc)), %c_desc_or_ptr_9: i32 loc("c_desc_or_ptr"(#loc)), %c_desc_or_ptr_10: i64 loc("c_desc_or_ptr"(#loc)), %c_desc_or_ptr_11: i64 loc("c_desc_or_ptr"(#loc)), %M: i32 {tt.divisibility = 16 : i32} loc("M"(#loc)), %N: i32 {tt.divisibility = 16 : i32} loc("N"(#loc)), %K: i32 {tt.divisibility = 16 : i32} loc("K"(#loc)), %stride_cm: i32 {tt.divisibility = 16 : i32} loc("stride_cm"(#loc))) attributes {noinline = false} {
    %_1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc45)
    %_0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc46)
    %arg2 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc47)
    %a_1 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc48)
    %a_0 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc49)
    %accumulator_1, %accumulator_1_12 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc50)
    %accumulator_0, %accumulator_0_13 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc51)
    %false = arith.constant {async_task_id = array<i32: 1>} false loc(#loc5)
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true loc(#loc5)
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 148 : i32 loc(#loc5)
    %c8_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 8 : i32 loc(#loc5)
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 256 : i32 loc(#loc5)
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 128 : i32 loc(#loc5)
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 64 : i32 loc(#loc5)
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 0 : i32 loc(#loc5)
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 1 : i32 loc(#loc5)
    %num_pid_m = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 255 : i32 loc(#loc79)
    %num_pid_n = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 127 : i32 loc(#loc80)
    %k_tiles = arith.constant {async_task_id = array<i32: 0, 1, 2, 3, 4>} 63 : i32 loc(#loc81)
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #blocked> loc(#loc5)
    %start_pid = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc55)
    %num_pid_m_14 = arith.addi %M, %num_pid_m {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc79)
    %num_pid_m_15 = arith.divsi %num_pid_m_14, %c256_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc82)
    %num_pid_n_16 = arith.addi %N, %num_pid_n {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc80)
    %num_pid_n_17 = arith.divsi %num_pid_n_16, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc83)
    %k_tiles_18 = arith.addi %K, %k_tiles {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc81)
    %k_tiles_19 = arith.divsi %k_tiles_18, %c64_i32 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc84)
    %num_tiles = arith.muli %num_pid_m_15, %num_pid_n_17 {async_task_id = array<i32: 0, 1, 2, 3, 4>} : i32 loc(#loc56)
    %tile_id_c = arith.subi %start_pid, %c148_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc57)
    %num_pid_in_group = arith.muli %num_pid_n_17, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32 loc(#loc58)
    %tile_id_c_20 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%tile_id_c_21 = %tile_id_c) -> (i32)  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group {async_task_id = array<i32: 2>} : i32 loc(#loc85)
      %first_pid_m = arith.muli %group_id, %c8_i32 {async_task_id = array<i32: 2>} : i32 loc(#loc86)
      %group_size_m = arith.subi %num_pid_m_15, %first_pid_m {async_task_id = array<i32: 2>} : i32 loc(#loc87)
      %group_size_m_22 = arith.minsi %group_size_m, %c8_i32 {async_task_id = array<i32: 2>} : i32 loc(#loc88)
      %pid_m = arith.remsi %tile_id, %group_size_m_22 {async_task_id = array<i32: 2>} : i32 loc(#loc89)
      %pid_m_23 = arith.addi %first_pid_m, %pid_m {async_task_id = array<i32: 2>} : i32 loc(#loc90)
      %pid_n = arith.remsi %tile_id, %num_pid_in_group {async_task_id = array<i32: 2>} : i32 loc(#loc91)
      %pid_n_24 = arith.divsi %pid_n, %group_size_m_22 {async_task_id = array<i32: 2>} : i32 loc(#loc92)
      %offs_am = arith.muli %pid_m_23, %c256_i32 {async_task_id = array<i32: 2>} : i32 loc(#loc68)
      %a = arith.addi %offs_am, %c128_i32 {async_task_id = array<i32: 2>} : i32 loc(#loc69)
      %offs_bn = arith.muli %pid_n_24, %c128_i32 {async_task_id = array<i32: 2>} : i32 loc(#loc70)
      %accumulator = ttng.tmem_store %cst, %accumulator_0[%accumulator_0_13], %true {async_task_id = array<i32: 0>, tmem.start = array<i32: 8, 10>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc71)
      %accumulator_25 = ttng.tmem_store %cst, %accumulator_1[%accumulator_1_12], %true {async_task_id = array<i32: 0>, tmem.start = array<i32: 5, 7>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc71)
      %accumulator_26:3 = scf.for %accumulator_42 = %c0_i32 to %k_tiles_19 step %c1_i32 iter_args(%arg22 = %false, %accumulator_43 = %accumulator, %accumulator_44 = %accumulator_25) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %accumulator_42, %c64_i32 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 loc(#loc73)
        %a_45 = tt.descriptor_load %a_desc[%offs_am, %offs_k] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1> loc(#loc69)
        %a_46 = tt.descriptor_load %a_desc[%a, %offs_k] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1> loc(#loc69)
        ttg.local_store %a_45, %a_0 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc49)
        ttg.local_store %a_46, %a_1 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc48)
        %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1> loc(#loc74)
        ttg.local_store %b, %arg2 {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc47)
        %arg2_47 = ttg.memdesc_trans %arg2 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> loc(#loc47)
        %accumulator_48 = ttng.tc_gen5_mma %a_0, %arg2_47, %accumulator_0[%accumulator_43], %arg22, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, tmem.end = array<i32: 8>, tmem.start = array<i32: 9>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc71)
        %accumulator_49 = ttng.tc_gen5_mma %a_1, %arg2_47, %accumulator_1[%accumulator_44], %arg22, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32, tmem.end = array<i32: 5>, tmem.start = array<i32: 6>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc71)
        scf.yield {async_task_id = array<i32: 0, 1, 4>} %true, %accumulator_48, %accumulator_49 : i1, !ttg.async.token, !ttg.async.token loc(#loc30)
      } {async_task_id = array<i32: 0, 1, 2, 3, 4>, tt.scheduled_max_stage = 2 : i32} loc(#loc72)
      %tile_id_c_27 = arith.addi %tile_id_c_21, %c148_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc75)
      %group_id_28 = arith.divsi %tile_id_c_27, %num_pid_in_group {async_task_id = array<i32: 3>} : i32 loc(#loc93)
      %first_pid_m_29 = arith.muli %group_id_28, %c8_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc94)
      %group_size_m_30 = arith.subi %num_pid_m_15, %first_pid_m_29 {async_task_id = array<i32: 3>} : i32 loc(#loc95)
      %group_size_m_31 = arith.minsi %group_size_m_30, %c8_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc96)
      %pid_m_32 = arith.remsi %tile_id_c_27, %group_size_m_31 {async_task_id = array<i32: 3>} : i32 loc(#loc97)
      %pid_m_33 = arith.addi %first_pid_m_29, %pid_m_32 {async_task_id = array<i32: 3>} : i32 loc(#loc98)
      %pid_n_34 = arith.remsi %tile_id_c_27, %num_pid_in_group {async_task_id = array<i32: 3>} : i32 loc(#loc99)
      %pid_n_35 = arith.divsi %pid_n_34, %group_size_m_31 {async_task_id = array<i32: 3>} : i32 loc(#loc100)
      %offs_am_c = arith.muli %pid_m_33, %c256_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc76)
      %0 = arith.addi %offs_am_c, %c128_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc1)
      %offs_bn_c = arith.muli %pid_n_35, %c128_i32 {async_task_id = array<i32: 3>} : i32 loc(#loc77)
      %accumulator_36, %accumulator_37 = ttng.tmem_load %accumulator_0[%accumulator_26#1] {async_task_id = array<i32: 4>, tmem.end = array<i32: 9, 10>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked> loc(#loc71)
      %accumulator_38, %accumulator_39 = ttng.tmem_load %accumulator_1[%accumulator_26#2] {async_task_id = array<i32: 4>, tmem.end = array<i32: 6, 7>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked> loc(#loc71)
      %accumulator_40 = arith.truncf %accumulator_36 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked> loc(#loc78)
      %accumulator_41 = arith.truncf %accumulator_38 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked> loc(#loc78)
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
    } {async_task_id = array<i32: 0, 1, 2, 3, 4>, tt.data_partition_factor = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["default", "gemm", "load", "epilogue", "computation"], ttg.warp_specialize.tag = 0 : i32} loc(#loc59)
    tt.return loc(#loc37)
  } loc(#loc)
} loc(#loc)
