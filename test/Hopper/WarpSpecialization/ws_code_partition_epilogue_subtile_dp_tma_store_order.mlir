// RUN: triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448 tma-store-pipelining=true" 2>&1 | FileCheck %s

// Regression test for a column-sum epilogue at DATA_PARTITION_FACTOR=2 and
// EPILOGUE_SUBTILE=2 with early TMA store lowering (the Inductor Blackwell
// persistent TMA template at BLOCK_M=256, num_stages=2 per partition).
//
// tt.split yields (LHS, RHS), but the RHS convert_layout is emitted first, so
// the reorderEpilogOps streamline step hoists RHS's staging local_alloc ahead of
// LHS's while the TMA copies stay LHS then RHS. The two staging buffers of each
// data partition share one N-buffer reuse group, and restoring producer order to
// match TMA store order only recognized tt.descriptor_store, not the already
// lowered ttng.async_tma_copy_local_to_global. The group therefore reached
// insertAsyncComm with producer and consumer orders reversed, which aborted the
// process with "N-buffer reuse group: producer and consumer orderings are
// inconsistent".

// CHECK-NOT: error
// CHECK: remark: reuse barrier: channel 3 waits on channel 2 (intra-iteration)
// CHECK-NOT: error
// CHECK: remark: reuse barrier: channel 5 waits on channel 4 (intra-iteration)
// CHECK-NOT: error
// CHECK: tt.func public @kernel
// CHECK-COUNT-4: ttng.async_tma_copy_local_to_global

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<128x64xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x64xf32, #shared1>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg16: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg17, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.addi %arg18, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg19, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.muli %4, %c8_i32 : i32
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    scf.for %arg20 = %0 to %7 step %c148_i32  : i32 {
      %10 = arith.divsi %arg20, %8 {ttg.partition = array<i32: 2, 3>} : i32
      %11 = arith.muli %10, %c8_i32 {ttg.partition = array<i32: 2, 3>} : i32
      %12 = arith.subi %2, %11 {ttg.partition = array<i32: 2, 3>} : i32
      %13 = arith.minsi %12, %c8_i32 {ttg.partition = array<i32: 2, 3>} : i32
      %14 = arith.remsi %arg20, %13 {ttg.partition = array<i32: 2, 3>} : i32
      %15 = arith.addi %11, %14 {ttg.partition = array<i32: 2, 3>} : i32
      %16 = arith.remsi %arg20, %8 {ttg.partition = array<i32: 2, 3>} : i32
      %17 = arith.divsi %16, %13 {ttg.partition = array<i32: 2, 3>} : i32
      %18 = arith.muli %15, %c256_i32 {ttg.partition = array<i32: 2, 3>} : i32
      %19 = arith.addi %18, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %20 = arith.addi %18, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %21 = arith.addi %18, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %22 = arith.muli %17, %c128_i32 {ttg.partition = array<i32: 2, 3>} : i32
      %23 = ttg.convert_layout %cst : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %24 = ttg.convert_layout %cst : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %25 = ttng.tmem_store %23, %result[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_0, %token_1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %26 = ttng.tmem_store %24, %result_0[%token_1], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %27:3 = scf.for %arg21 = %c0_i32 to %6 step %c1_i32 iter_args(%arg22 = %false, %arg23 = %25, %arg24 = %26) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
        %64 = arith.muli %arg21, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : i32
        %65 = tt.descriptor_load %arg5[%22, %64] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
        %66 = tt.descriptor_load %arg0[%18, %64] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
        %67 = tt.descriptor_load %arg0[%19, %64] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
        %68 = ttg.local_alloc %65 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %69 = ttg.memdesc_trans %68 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared2, #smem>
        %70 = ttg.local_alloc %66 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %71 = ttng.tc_gen5_mma %70, %69, %result[%arg23], %arg22, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %72 = ttg.local_alloc %67 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %73 = ttng.tc_gen5_mma %72, %69, %result_0[%arg24], %arg22, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %71, %73 : i1, !ttg.async.token, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %28 = tt.splat %22 : i32 -> tensor<64xi32, #blocked>
      %29 = arith.addi %28, %9 : tensor<64xi32, #blocked>
      %30 = arith.muli %15, %arg18 : i32
      %31 = tt.addptr %arg15, %30 : !tt.ptr<f32>, i32
      %32 = tt.splat %31 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
      %33 = tt.addptr %32, %29 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
      %34 = arith.addi %22, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %35 = tt.splat %34 : i32 -> tensor<64xi32, #blocked>
      %36 = arith.addi %35, %9 : tensor<64xi32, #blocked>
      %37 = tt.addptr %32, %36 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
      %result_2, %token_3 = ttng.tmem_load %result[%27#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %38 = ttg.convert_layout %result_2 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %39 = tt.reshape %38 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear2>
      %40 = tt.trans %39 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
      %outLHS, %outRHS = tt.split %40 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
      %41 = ttg.convert_layout %outRHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #blocked2>
      %42 = ttg.convert_layout %outLHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #blocked2>
      %43 = ttg.local_alloc %42 {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %44 = ttng.async_tma_copy_local_to_global %arg10[%18, %22] %43 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %44   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %45 = "tt.reduce"(%outLHS) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg21: f32, %arg22: f32):
        %64 = arith.addf %arg21, %arg22 {ttg.partition = array<i32: 0>} : f32
        tt.reduce.return %64 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #linear4>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear4}>>
      %46 = ttg.local_alloc %41 {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %47 = ttng.async_tma_copy_local_to_global %arg10[%18, %34] %46 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %47   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %48 = "tt.reduce"(%outRHS) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg21: f32, %arg22: f32):
        %64 = arith.addf %arg21, %arg22 {ttg.partition = array<i32: 0>} : f32
        tt.reduce.return %64 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #linear4>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear4}>>
      %result_4, %token_5 = ttng.tmem_load %result_0[%27#2] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %49 = ttg.convert_layout %result_4 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %50 = tt.reshape %49 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear2>
      %51 = tt.trans %50 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
      %outLHS_6, %outRHS_7 = tt.split %51 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
      %52 = ttg.convert_layout %outRHS_7 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #blocked2>
      %53 = ttg.convert_layout %outLHS_6 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #blocked2>
      %54 = ttg.local_alloc %53 {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %55 = ttng.async_tma_copy_local_to_global %arg10[%21, %22] %54 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %55   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %56 = "tt.reduce"(%outLHS_6) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg21: f32, %arg22: f32):
        %64 = arith.addf %arg21, %arg22 {ttg.partition = array<i32: 0>} : f32
        tt.reduce.return %64 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #linear4>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear4}>>
      %57 = arith.addf %45, %56 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear4}>>
      %58 = ttg.convert_layout %57 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear4}>> -> tensor<64xf32, #blocked>
      tt.store %33, %58 {ttg.partition = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #blocked>
      %59 = ttg.local_alloc %52 {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %60 = ttng.async_tma_copy_local_to_global %arg10[%20, %34] %59 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %60   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %61 = "tt.reduce"(%outRHS_7) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg21: f32, %arg22: f32):
        %64 = arith.addf %arg21, %arg22 {ttg.partition = array<i32: 0>} : f32
        tt.reduce.return %64 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #linear4>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear4}>>
      %62 = arith.addf %48, %61 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear4}>>
      %63 = ttg.convert_layout %62 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear4}>> -> tensor<64xf32, #blocked>
      tt.store %37, %63 {ttg.partition = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #blocked>
    } {tt.data_partition_factor = 2 : i32, tt.disallow_acc_multi_buffer, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
