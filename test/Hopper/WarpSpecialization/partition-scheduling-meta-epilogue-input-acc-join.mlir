// RUN: triton-opt %s -split-input-file --nvgpu-partition-scheduling-meta | FileCheck %s

// Persistent GEMM with no default partition (gemm, load, computation) whose
// epilogue combines a TMA-loaded input with the accumulator, then row- and
// column-reduces it (fused mm + reduction epilogue, reduced from Inductor).
// Nothing places the outer-loop accumulator read before propagatePartitions.
// The input's users must stay with the accumulator read rather than go to the
// load partition, which would hand the fp32 accumulator and its transpose to
// the load partition through shared memory.

// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD:[0-9]+]]>{{.*}} -> tensor<128x64xf32
// CHECK-NEXT: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[ACC:[0-9]+]]>
// CHECK-NEXT: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[ACC]]>
// CHECK-NEXT: arith.mulf {{.*}}ttg.partition = array<i32: [[ACC]]>
// CHECK: tt.store {{.*}}ttg.partition = array<i32: [[ACC]]>
// CHECK: tt.trans {{.*}}ttg.partition = array<i32: [[ACC]]>
// CHECK: tt.store {{.*}}ttg.partition = array<i32: [[ACC]]>
// CHECK: ttg.partition.types = ["computation", "load", "gemm"]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, ttg.use_meta_ws = true, "ttng.two-ctas" = false} {
  tt.func public @kern(%arg_A: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg_B: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %ws_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %c969147_i32 = arith.constant 969147 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c8_i32 = arith.constant 8 : i32
    %c148_i32 = arith.constant 148 : i32
    %c64_i32 = arith.constant 64 : i32
    %num_pid_in_group = arith.constant 32 : i32
    %num_tiles = arith.constant 30288 : i32
    %k_tiles = arith.constant 2 : i32
    %grid_m = arith.constant 7572 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<256> : tensor<64x1xi32, #blocked>
    %c15544240_i32 = arith.constant 15544240 : i32
    %c37888_i32 = arith.constant 37888 : i32
    %cst_0 = arith.constant dense<969147> : tensor<128x1xi32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #linear>
    %tdesc3 = tt.make_tensor_descriptor %in_ptr3, [%c969147_i32, %c256_i32], [%c256_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x64xf32, #shared>
    %start_pid = tt.get_program_id x : i32
    %a_desc = tt.make_tensor_descriptor %arg_A, [%c969147_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared1>
    %b_desc = tt.make_tensor_descriptor %arg_B, [%c128_i32, %c256_i32], [%c256_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<64x64xbf16, #shared1>
    %tile_id_c = arith.subi %start_pid, %c148_i32 : i32
    %tile_xindex = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %tile_xindex_2 = tt.expand_dims %tile_xindex {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %x4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %x4_3 = tt.expand_dims %x4 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %0 = tt.addptr %ws_ptr, %c37888_i32 : !tt.ptr<i8>, i32
    %1 = tt.bitcast %0 : !tt.ptr<i8> -> !tt.ptr<f32>
    %2 = tt.addptr %ws_ptr, %c15544240_i32 : !tt.ptr<i8>, i32
    %3 = tt.bitcast %2 : !tt.ptr<i8> -> !tt.ptr<f32>
    %tile_id_c_4 = scf.for %tile_id = %start_pid to %num_tiles step %c148_i32 iter_args(%tile_id_c_5 = %tile_id_c) -> (i32)  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 : i32
      %GROUP_M = arith.subi %grid_m, %first_pid_m : i32
      %GROUP_M_6 = arith.minsi %GROUP_M, %c8_i32 : i32
      %pid_m = arith.remsi %tile_id, %GROUP_M_6 : i32
      %pid_m_7 = arith.addi %first_pid_m, %pid_m : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group : i32
      %pid_n_8 = arith.divsi %pid_n, %GROUP_M_6 : i32
      %offs_am = arith.muli %pid_m_7, %c128_i32 : i32
      %offs_bn = arith.muli %pid_n_8, %c64_i32 : i32
      %accumulator, %accumulator_9 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %accumulator_10 = ttng.tmem_store %cst_1, %accumulator[%accumulator_9], %true : tensor<128x64xf32, #linear> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %accumulator_11:2 = scf.for %ki = %c0_i32 to %k_tiles step %c1_i32 iter_args(%accumulator_32 = %false, %accumulator_33 = %accumulator_10) -> (i1, !ttg.async.token)  : i32 {
        %offs_k = arith.muli %ki, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xbf16, #shared1> -> tensor<128x64xbf16, #blocked1>
        %a_34 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared1, #smem>
        %b = tt.descriptor_load %b_desc[%offs_k, %offs_bn] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x64xbf16, #shared1> -> tensor<64x64xbf16, #blocked1>
        %b_35 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<64x64xbf16, #blocked1>) -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
        %accumulator_36 = ttng.tc_gen5_mma %a_34, %b_35, %accumulator[%accumulator_33], %accumulator_32, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared1, #smem>, !ttg.memdesc<64x64xbf16, #shared1, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %accumulator_36 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %tile_id_c_12 = arith.addi %tile_id_c_5, %c148_i32 : i32
      %group_id_13 = arith.divsi %tile_id_c_12, %num_pid_in_group : i32
      %first_pid_m_14 = arith.muli %group_id_13, %c8_i32 : i32
      %GROUP_M_15 = arith.subi %grid_m, %first_pid_m_14 : i32
      %GROUP_M_16 = arith.minsi %GROUP_M_15, %c8_i32 : i32
      %pid_m_17 = arith.remsi %tile_id_c_12, %GROUP_M_16 : i32
      %pid_m_18 = arith.addi %first_pid_m_14, %pid_m_17 : i32
      %pid_n_19 = arith.remsi %tile_id_c_12, %num_pid_in_group : i32
      %pid_n_20 = arith.divsi %pid_n_19, %GROUP_M_16 : i32
      %offs_cm = arith.muli %pid_m_18, %c128_i32 : i32
      %offs_cn = arith.muli %pid_n_20, %c64_i32 : i32
      %tile_xindex_21 = tt.splat %offs_cm : i32 -> tensor<128x1xi32, #blocked>
      %tile_xindex_22 = arith.addi %tile_xindex_21, %tile_xindex_2 : tensor<128x1xi32, #blocked>
      %tile_xindex_mask = arith.cmpi slt, %tile_xindex_22, %cst_0 : tensor<128x1xi32, #blocked>
      %x4_23 = tt.splat %offs_cn : i32 -> tensor<64x1xi32, #blocked>
      %x4_24 = arith.addi %x4_23, %x4_3 : tensor<64x1xi32, #blocked>
      %tmp5 = tt.descriptor_load %tdesc3[%offs_cm, %offs_cn] : !tt.tensordesc<128x64xf32, #shared> -> tensor<128x64xf32, #blocked2>
      %tmp5_25 = ttg.convert_layout %tmp5 : tensor<128x64xf32, #blocked2> -> tensor<128x64xf32, #linear>
      %accumulator_26, %accumulator_27 = ttng.tmem_load %accumulator[%accumulator_11#1] : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear>
      %tmp13 = arith.mulf %accumulator_26, %tmp5_25 : tensor<128x64xf32, #linear>
      %tmp17 = "tt.reduce"(%tmp13) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%tmp17_32: f32, %tmp17_33: f32):
        %tmp17_34 = arith.addf %tmp17_32, %tmp17_33 : f32
        tt.reduce.return %tmp17_34 : f32
      }) : (tensor<128x64xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %tmp17_28 = ttg.convert_layout %tmp17 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %tmp17_29 = tt.expand_dims %tmp17_28 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
      %4 = arith.divsi %offs_cn, %c64_i32 : i32
      %5 = arith.muli %4, %c969147_i32 : i32
      %6 = tt.addptr %1, %5 : !tt.ptr<f32>, i32
      %7 = tt.splat %6 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
      %8 = tt.addptr %7, %tile_xindex_22 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
      %9 = ttg.convert_layout %8 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x1x!tt.ptr<f32>, #blocked3>
      %10 = ttg.convert_layout %tile_xindex_mask : tensor<128x1xi1, #blocked> -> tensor<128x1xi1, #blocked3>
      tt.store %9, %tmp17_29, %10 : tensor<128x1x!tt.ptr<f32>, #blocked3>
      %tmp23 = tt.trans %tmp13 {order = array<i32: 1, 0>} : tensor<128x64xf32, #linear> -> tensor<64x128xf32, #linear1>
      %tmp27 = "tt.reduce"(%tmp23) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%tmp27_32: f32, %tmp27_33: f32):
        %tmp27_34 = arith.addf %tmp27_32, %tmp27_33 : f32
        tt.reduce.return %tmp27_34 : f32
      }) : (tensor<64x128xf32, #linear1>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear1}>>
      %tmp27_30 = ttg.convert_layout %tmp27 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear1}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %tmp27_31 = tt.expand_dims %tmp27_30 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<64x1xf32, #blocked4>
      %11 = arith.divsi %offs_cm, %c128_i32 : i32
      %12 = arith.muli %11, %c256_i32 : i32
      %13 = tt.addptr %3, %12 : !tt.ptr<f32>, i32
      %14 = tt.splat %13 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
      %15 = tt.addptr %14, %x4_24 : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
      %16 = arith.cmpi slt, %x4_24, %cst : tensor<64x1xi32, #blocked>
      %17 = ttg.convert_layout %15 : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x1x!tt.ptr<f32>, #blocked4>
      %18 = ttg.convert_layout %16 : tensor<64x1xi1, #blocked> -> tensor<64x1xi1, #blocked4>
      tt.store %17, %tmp27_31, %18 : tensor<64x1x!tt.ptr<f32>, #blocked4>
      scf.yield %tile_id_c_12 : i32
    } {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize}
    tt.return
  }
}
