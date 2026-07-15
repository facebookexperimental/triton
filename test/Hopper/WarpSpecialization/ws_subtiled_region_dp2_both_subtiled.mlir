// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 generate-subtiled-region=true num-stages=3 pingpong-auto-ws=false smem-budget=232448" | FileCheck %s

// Test: both-endpoints-subtiled SMEM channels with DATA_PARTITION_FACTOR=2 +
// early_tma_store_lowering + EPILOGUE_SUBTILE=2 (tutorial09
// matmul_kernel_tma_persistent_ws, separate_epilogue_store=True, BLOCK_M=256).
//
// The epilogue produces FOUR per-tile staging allocs and FOUR
// ttng.subtiled_region ops: a producer (truncf + local_store, task 0) and a
// consumer (async_tma_copy_local_to_global, task 2) per data partition.
//
// Two things must hold (each pre-fix failed):
//  1. The two data partitions' staging buffers must NOT be fused into ONE
//     physical buffer. The memory planner now keys TMA-staging fusion on
//     (descriptor, source accumulator), so partition A's tiles and partition
//     B's tiles get DISTINCT buffer.ids. Without this they shared one
//     buffer/barrier -> data aliasing + runtime deadlock.
//  2. The per-tile staging allocs of one (producer region, consumer region)
//     pair collapse into a single ChannelPost (its numTiles per-tile buffers
//     are in-body instances). Without the split+collapse, getReuseChannels
//     walked the merged 4-channel cross-partition group and called getDstOp()
//     on an already-lowered sibling alloc -> `consumers.size() != 0` assert /
//     SIGSEGV in insertAsyncComm.

// CHECK-LABEL: @matmul_kernel_tma_persistent_ws
//
// Two independent per-partition staging multibuffers (distinct buffer.id), each
// 128x64xf16 -- the cross-partition reuse-group split. These are hoisted to
// function entry, before the ttg.warp_specialize op.
// CHECK-DAG: ttg.local_alloc {{.*}}buffer.id = 3 : i32{{.*}} : () -> !ttg.memdesc<{{[0-9]+}}x128x64xf16, #shared
// CHECK-DAG: ttg.local_alloc {{.*}}buffer.id = 4 : i32{{.*}} : () -> !ttg.memdesc<{{[0-9]+}}x128x64xf16, #shared
// CHECK: ttg.warp_specialize
//
// All four epilogue subtile stores (2 data partitions x 2 subtiles) are
// emitted -- the both-subtiled channels collapsed and lowered with no assert.
// CHECK-COUNT-4: ttng.async_tma_copy_local_to_global

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#loc = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":100:0)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#loc37 = loc("a_desc"(#loc))
#loc38 = loc("a_desc.shape.0"(#loc))
#loc39 = loc("a_desc.shape.1"(#loc))
#loc40 = loc("a_desc.stride.0"(#loc))
#loc41 = loc("a_desc.stride.1"(#loc))
#loc42 = loc("b_desc"(#loc))
#loc43 = loc("b_desc.shape.0"(#loc))
#loc44 = loc("b_desc.shape.1"(#loc))
#loc45 = loc("b_desc.stride.0"(#loc))
#loc46 = loc("b_desc.stride.1"(#loc))
#loc47 = loc("c_desc"(#loc))
#loc48 = loc("c_desc.shape.0"(#loc))
#loc49 = loc("c_desc.shape.1"(#loc))
#loc50 = loc("c_desc.stride.0"(#loc))
#loc51 = loc("c_desc.stride.1"(#loc))
#loc52 = loc("M"(#loc))
#loc53 = loc("N"(#loc))
#loc54 = loc("K"(#loc))
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent_ws(%a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>> loc("a_desc"(#loc)), %a_desc.shape.0: i32 loc("a_desc.shape.0"(#loc)), %a_desc.shape.1: i32 loc("a_desc.shape.1"(#loc)), %a_desc.stride.0: i64 loc("a_desc.stride.0"(#loc)), %a_desc.stride.1: i64 loc("a_desc.stride.1"(#loc)), %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>> loc("b_desc"(#loc)), %b_desc.shape.0: i32 loc("b_desc.shape.0"(#loc)), %b_desc.shape.1: i32 loc("b_desc.shape.1"(#loc)), %b_desc.stride.0: i64 loc("b_desc.stride.0"(#loc)), %b_desc.stride.1: i64 loc("b_desc.stride.1"(#loc)), %c_desc: !tt.tensordesc<tensor<128x64xf16, #shared>> loc("c_desc"(#loc)), %c_desc.shape.0: i32 loc("c_desc.shape.0"(#loc)), %c_desc.shape.1: i32 loc("c_desc.shape.1"(#loc)), %c_desc.stride.0: i64 loc("c_desc.stride.0"(#loc)), %c_desc.stride.1: i64 loc("c_desc.stride.1"(#loc)), %M: i32 {tt.divisibility = 16 : i32} loc("M"(#loc)), %N: i32 {tt.divisibility = 16 : i32} loc("N"(#loc)), %K: i32 {tt.divisibility = 16 : i32} loc("K"(#loc))) attributes {noinline = false} {
    %false = arith.constant false loc(#loc1)
    %true = arith.constant true loc(#loc1)
    %c8_i32 = arith.constant 8 : i32 loc(#loc1)
    %c256_i32 = arith.constant 256 : i32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %c148_i32 = arith.constant 148 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %num_pid_m = arith.constant 255 : i32 loc(#loc82)
    %num_pid_n = arith.constant 127 : i32 loc(#loc83)
    %k_tiles = arith.constant 63 : i32 loc(#loc84)
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear> loc(#loc1)
    %start_pid = tt.get_program_id x : i32 loc(#loc58)
    %num_pid_m_0 = arith.addi %M, %num_pid_m : i32 loc(#loc85)
    %num_pid_m_1 = arith.divsi %num_pid_m_0, %c256_i32 : i32 loc(#loc86)
    %num_pid_n_2 = arith.addi %N, %num_pid_n : i32 loc(#loc87)
    %num_pid_n_3 = arith.divsi %num_pid_n_2, %c128_i32 : i32 loc(#loc88)
    %k_tiles_4 = arith.addi %K, %k_tiles : i32 loc(#loc89)
    %k_tiles_5 = arith.divsi %k_tiles_4, %c64_i32 : i32 loc(#loc90)
    %num_tiles = arith.muli %num_pid_m_1, %num_pid_n_3 : i32 loc(#loc59)
    %num_pid_in_group = arith.muli %num_pid_n_3, %c8_i32 : i32 loc(#loc60)
    scf.for %tile_id = %start_pid to %num_tiles step %c148_i32  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group : i32 loc(#loc91)
      %first_pid_m = arith.muli %group_id, %c8_i32 : i32 loc(#loc92)
      %group_size_m = arith.subi %num_pid_m_1, %first_pid_m : i32 loc(#loc93)
      %group_size_m_6 = arith.minsi %group_size_m, %c8_i32 : i32 loc(#loc94)
      %pid_m = arith.remsi %tile_id, %group_size_m_6 : i32 loc(#loc95)
      %pid_m_7 = arith.addi %first_pid_m, %pid_m : i32 loc(#loc96)
      %pid_n = arith.remsi %tile_id, %num_pid_in_group : i32 loc(#loc97)
      %pid_n_8 = arith.divsi %pid_n, %group_size_m_6 : i32 loc(#loc98)
      %offs_am = arith.muli %pid_m_7, %c256_i32 : i32 loc(#loc69)
      %a = arith.addi %offs_am, %c128_i32 : i32 loc(#loc70)
      %0 = arith.addi %offs_am, %c128_i32 : i32 loc(#loc22)
      %1 = arith.addi %offs_am, %c128_i32 : i32 loc(#loc22)
      %offs_bn = arith.muli %pid_n_8, %c128_i32 : i32 loc(#loc71)
      %accumulator_0, %accumulator_0_9 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc72)
      %accumulator_1, %accumulator_1_10 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc73)
      %accumulator = ttg.convert_layout %cst : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1> loc(#loc74)
      %accumulator_11 = ttng.tmem_store %accumulator, %accumulator_0[%accumulator_0_9], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc74)
      %accumulator_12 = ttg.convert_layout %cst : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1> loc(#loc74)
      %accumulator_13 = ttng.tmem_store %accumulator_12, %accumulator_1[%accumulator_1_10], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc74)
      %accumulator_14:3 = scf.for %ki = %c0_i32 to %k_tiles_5 step %c1_i32 iter_args(%accumulator_28 = %false, %accumulator_29 = %accumulator_11, %accumulator_30 = %accumulator_13) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %ki, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 loc(#loc76)
        %a_31 = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked> loc(#loc70)
        %a_32 = tt.descriptor_load %a_desc[%a, %offs_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked> loc(#loc70)
        %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked> loc(#loc77)
        %a_0 = ttg.local_alloc %a_31 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem> loc(#loc78)
        %a_1 = ttg.local_alloc %a_32 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem> loc(#loc79)
        %accumulator_33 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem> loc(#loc80)
        %accumulator_34 = ttg.memdesc_trans %accumulator_33 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem> loc(#loc80)
        %accumulator_35 = ttng.tc_gen5_mma %a_0, %accumulator_34, %accumulator_0[%accumulator_29], %accumulator_28, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc74)
        %accumulator_36 = ttng.tc_gen5_mma %a_1, %accumulator_34, %accumulator_1[%accumulator_30], %accumulator_28, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc74)
        scf.yield %true, %accumulator_35, %accumulator_36 : i1, !ttg.async.token, !ttg.async.token loc(#loc29)
      } {tt.scheduled_max_stage = 2 : i32} loc(#loc75)
      %accumulator_15, %accumulator_16 = ttng.tmem_load %accumulator_0[%accumulator_14#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1> loc(#loc74)
      %accumulator_17 = ttg.convert_layout %accumulator_15 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear> loc(#loc74)
      %accumulator_18, %accumulator_19 = ttng.tmem_load %accumulator_1[%accumulator_14#2] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1> loc(#loc74)
      %accumulator_20 = ttg.convert_layout %accumulator_18 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear> loc(#loc74)
      %acc_slices = tt.reshape %accumulator_17 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear2> loc(#loc99)
      %acc_slices_21 = tt.reshape %accumulator_20 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear2> loc(#loc99)
      %acc_slices_22 = tt.trans %acc_slices {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3> loc(#loc100)
      %acc_slices_23 = tt.trans %acc_slices_21 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3> loc(#loc100)
      %acc_slices_24, %acc_slices_25 = tt.split %acc_slices_22 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4> loc(#loc101)
      %acc_slices_26, %acc_slices_27 = tt.split %acc_slices_23 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4> loc(#loc101)
      %2 = arith.truncf %acc_slices_24 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> to tensor<128x64xf16, #linear4> loc(#loc34)
      %3 = arith.truncf %acc_slices_26 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> to tensor<128x64xf16, #linear4> loc(#loc34)
      %4 = ttg.convert_layout %2 {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear4> -> tensor<128x64xf16, #blocked> loc(#loc34)
      %5 = ttg.convert_layout %3 {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear4> -> tensor<128x64xf16, #blocked> loc(#loc34)
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc22)
      %7 = ttng.async_tma_copy_local_to_global %c_desc[%offs_am, %offs_bn] %6 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token loc(#loc22)
      ttng.async_tma_store_token_wait %7   {ttg.partition = array<i32: 2>} : !ttg.async.token loc(#loc22)
      %8 = ttg.local_alloc %5 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc22)
      %9 = ttng.async_tma_copy_local_to_global %c_desc[%1, %offs_bn] %8 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token loc(#loc22)
      ttng.async_tma_store_token_wait %9   {ttg.partition = array<i32: 2>} : !ttg.async.token loc(#loc22)
      %10 = arith.addi %offs_bn, %c64_i32 : i32 loc(#loc35)
      %11 = arith.truncf %acc_slices_25 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> to tensor<128x64xf16, #linear4> loc(#loc34)
      %12 = arith.truncf %acc_slices_27 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> to tensor<128x64xf16, #linear4> loc(#loc34)
      %13 = ttg.convert_layout %11 {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear4> -> tensor<128x64xf16, #blocked> loc(#loc34)
      %14 = ttg.convert_layout %12 {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear4> -> tensor<128x64xf16, #blocked> loc(#loc34)
      %15 = ttg.local_alloc %13 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc22)
      %16 = ttng.async_tma_copy_local_to_global %c_desc[%offs_am, %10] %15 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token loc(#loc22)
      ttng.async_tma_store_token_wait %16   {ttg.partition = array<i32: 2>} : !ttg.async.token loc(#loc22)
      %17 = ttg.local_alloc %14 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> loc(#loc22)
      %18 = ttng.async_tma_copy_local_to_global %c_desc[%0, %10] %17 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token loc(#loc22)
      ttng.async_tma_store_token_wait %18   {ttg.partition = array<i32: 2>} : !ttg.async.token loc(#loc22)
    } {tt.data_partition_factor = 2 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation"], ttg.warp_specialize.tag = 0 : i32} loc(#loc2)
    tt.return loc(#loc36)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":137:12)
#loc3 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":122:27)
#loc4 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":123:27)
#loc5 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":124:25)
#loc6 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":121:30)
#loc7 = loc("/home/njriasan/tlx/triton/python/triton/language/standard.py":43:17)
#loc8 = loc("/home/njriasan/tlx/triton/python/triton/language/standard.py":43:30)
#loc9 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":125:28)
#loc10 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":127:38)
#loc11 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":21:26)
#loc12 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":139:88)
#loc13 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":22:29)
#loc14 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":23:35)
#loc15 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":23:48)
#loc16 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":24:37)
#loc17 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":24:27)
#loc18 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":25:23)
#loc19 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":25:44)
#loc20 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":140:26)
#loc21 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":149:32)
#loc22 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":161:16)
#loc23 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":141:26)
#loc24 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":154:41)
#loc25 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":144:24)
#loc26 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":145:26)
#loc27 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":153:32)
#loc28 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":154:36)
#loc29 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":154:12)
#loc30 = loc("/home/njriasan/tlx/triton/python/triton/language/extra/subtile_ops.py":10:27)
#loc31 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":156:46)
#loc32 = loc("/home/njriasan/tlx/triton/python/triton/language/extra/subtile_ops.py":10:75)
#loc33 = loc("/home/njriasan/tlx/triton/python/triton/language/extra/subtile_ops.py":10:17)
#loc34 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":161:40)
#loc35 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":160:36)
#loc36 = loc("/home/njriasan/tlx/triton/python/test/unit/language/test_tutorial09_warp_specialization.py":130:4)
#loc55 = loc("num_pid_m"(#loc3))
#loc56 = loc("num_pid_n"(#loc4))
#loc57 = loc("k_tiles"(#loc5))
#loc58 = loc("start_pid"(#loc6))
#loc59 = loc("num_tiles"(#loc9))
#loc60 = loc("num_pid_in_group"(#loc10))
#loc61 = loc("group_id"(#loc11))
#loc62 = loc("first_pid_m"(#loc13))
#loc63 = loc("group_size_m"(#loc14))
#loc64 = loc("group_size_m"(#loc15))
#loc65 = loc("pid_m"(#loc16))
#loc66 = loc("pid_m"(#loc17))
#loc67 = loc("pid_n"(#loc18))
#loc68 = loc("pid_n"(#loc19))
#loc69 = loc("offs_am"(#loc20))
#loc70 = loc("a"(#loc21))
#loc71 = loc("offs_bn"(#loc23))
#loc72 = loc("accumulator_0"(#loc24))
#loc73 = loc("accumulator_1"(#loc24))
#loc74 = loc("accumulator"(#loc24))
#loc75 = loc("accumulator"(#loc25))
#loc76 = loc("offs_k"(#loc26))
#loc77 = loc("b"(#loc27))
#loc78 = loc("a_0"(#loc21))
#loc79 = loc("a_1"(#loc21))
#loc80 = loc("accumulator"(#loc28))
#loc81 = loc("acc_slices"(#loc31))
#loc82 = loc(callsite(#loc1 at #loc55))
#loc83 = loc(callsite(#loc1 at #loc56))
#loc84 = loc(callsite(#loc1 at #loc57))
#loc85 = loc(callsite(#loc7 at #loc55))
#loc86 = loc(callsite(#loc8 at #loc55))
#loc87 = loc(callsite(#loc7 at #loc56))
#loc88 = loc(callsite(#loc8 at #loc56))
#loc89 = loc(callsite(#loc7 at #loc57))
#loc90 = loc(callsite(#loc8 at #loc57))
#loc91 = loc(callsite(#loc61 at #loc12))
#loc92 = loc(callsite(#loc62 at #loc12))
#loc93 = loc(callsite(#loc63 at #loc12))
#loc94 = loc(callsite(#loc64 at #loc12))
#loc95 = loc(callsite(#loc65 at #loc12))
#loc96 = loc(callsite(#loc66 at #loc12))
#loc97 = loc(callsite(#loc67 at #loc12))
#loc98 = loc(callsite(#loc68 at #loc12))
#loc99 = loc(callsite(#loc30 at #loc81))
#loc100 = loc(callsite(#loc32 at #loc81))
#loc101 = loc(callsite(#loc33 at #loc81))
