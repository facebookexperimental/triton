// RUN: triton-opt %s --nvgpu-warp-specialization="generate-subtiled-region=true num-stages=3 smem-budget=232448" | FileCheck %s

// Test: the N epilogue subtiles that flow through a ttng.subtiled_region form a
// reuse group and must collapse onto ONE physical SMEM staging allocation.
//
// All subtiles feed the SAME subtiled_region consumer, so the reuse-group
// consumer-merge in doCodePartitionPost fires and removes the non-representative
// channel from orderedChannels. replaceBufferReuse used to iterate
// orderedChannels, so it never visited the merged-out channel and left its
// duplicate physical buffer alive -- doubling epilogue SMEM and causing
// `OutOfResources: shared memory` at BLOCK_SIZE_M=256 (test_tutorial09
// matmul_kernel_tma_persistent_ws, generate_subtiled_region=True,
// EPILOGUE_SUBTILE=2). replaceBufferReuse now iterates the reuse groups
// directly, so the merged channel is collapsed regardless of merge bookkeeping.
//
// This is the same reduced input as ws_subtiled_region_per_tile_barrier.mlir
// (EPILOGUE_SUBTILE=2 -> numTiles=2), captured just before
// NVGPUWarpSpecialization; here we assert the buffer collapse rather than the
// barrier indexing.

// CHECK-LABEL: @matmul_kernel_tma_persistent_ws
//
// Exactly one shared epilogue staging buffer survives (the reuse-group
// representative), hoisted to function entry ahead of ttg.warp_specialize.
// Before the fix there were two, each tagged allocation.shareGroup (the bug).
// The reuse-group buffers share buffer.id = 0; the first CHECK matches the
// representative and CHECK-NOT forbids any second shared buffer through EOF (and
// also fails loudly if warp specialization / buffering did not run at all).
// CHECK: ttg.local_alloc {allocation.shareGroup = 0 : i32, buffer.copy = 3 : i32, buffer.id = 0 : i32{{.*}}} : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
// CHECK-NOT: allocation.shareGroup

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  tt.func public @matmul_kernel_tma_persistent_ws(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg15, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg16, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg17, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.muli %4, %c8_i32 : i32
    scf.for %arg18 = %0 to %7 step %c148_i32  : i32 {
      %9 = arith.divsi %arg18, %8 : i32
      %10 = arith.muli %9, %c8_i32 : i32
      %11 = arith.subi %2, %10 : i32
      %12 = arith.minsi %11, %c8_i32 : i32
      %13 = arith.remsi %arg18, %12 : i32
      %14 = arith.addi %10, %13 : i32
      %15 = arith.remsi %arg18, %8 : i32
      %16 = arith.divsi %15, %12 : i32
      %17 = arith.muli %14, %c128_i32 : i32
      %18 = arith.muli %16, %c128_i32 : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %19 = ttng.tmem_store %cst, %result[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %20:2 = scf.for %arg19 = %c0_i32 to %6 step %c1_i32 iter_args(%arg20 = %false, %arg21 = %19) -> (i1, !ttg.async.token)  : i32 {
        %32 = arith.muli %arg19, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %33 = tt.descriptor_load %arg0[%17, %32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
        %34 = ttg.local_alloc %33 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %35 = tt.descriptor_load %arg5[%18, %32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
        %36 = ttg.local_alloc %35 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %37 = ttg.memdesc_trans %36 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %38 = ttng.tc_gen5_mma %34, %37, %result[%arg21], %arg20, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %38 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %result_0, %token_1 = ttng.tmem_load %result[%20#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %21 = tt.reshape %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
      %22 = tt.trans %21 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
      %outLHS, %outRHS = tt.split %22 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>
      %23 = arith.truncf %outLHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> to tensor<128x64xf16, #linear3>
      %24 = ttg.convert_layout %23 {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear3> -> tensor<128x64xf16, #blocked>
      %25 = ttg.local_alloc %24 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %26 = ttng.async_tma_copy_local_to_global %arg10[%17, %18] %25 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %26   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %27 = arith.addi %18, %c64_i32 : i32
      %28 = arith.truncf %outRHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> to tensor<128x64xf16, #linear3>
      %29 = ttg.convert_layout %28 {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear3> -> tensor<128x64xf16, #blocked>
      %30 = ttg.local_alloc %29 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %31 = ttng.async_tma_copy_local_to_global %arg10[%17, %27] %30 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %31   {ttg.partition = array<i32: 2>} : !ttg.async.token
    } {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
