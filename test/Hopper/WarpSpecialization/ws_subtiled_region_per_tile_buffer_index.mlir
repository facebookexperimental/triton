// RUN: triton-opt %s --nvgpu-warp-specialization="generate-subtiled-region=true num-stages=3 smem-budget=232448" | FileCheck %s

// Test: with EPILOGUE_SUBTILE=4 (numTiles=4) the N epilogue subtiles form a
// reuse group that shares ONE physical SMEM staging buffer (buffer.copy=3) AND
// ONE barrier pair. The physical staging-buffer slot index MUST equal the
// shared barrier's slot, both derived from the per-tile *flattened* count
// flattened = accumCnt + tileIdx, taken % numBuffers, where accumCnt advances by
// numTiles per iteration (the numTiles stride lives on the counter).
//
// The bug: createBufferPost derived the data-buffer index from the generic
// reuse-group stagger (accumCnt + reuseGroupPosition). Because all subtiles feed
// the one ttng.subtiled_region consumer, that position term collapsed to
// {0,1,1,1}, aliasing three of the four 128x32 column blocks onto the same
// physical slot while their barriers lived on distinct slots -- a data race
// (49.5% wrong output in test_tutorial09 matmul_kernel_tma_persistent_ws,
// generate_subtiled_region=True, EPILOGUE_SUBTILE=4). numTiles=2 happened to be
// correct (two slots are always distinct); numTiles>2 needs the flattened count.
//
// Fix: every SMEM-rotation value (the staging-buffer slot AND the shared
// barrier's bufferIdx/phase) is computed INSIDE the tile body from the builtin
// tileIdx, and the numTiles stride lives on the loop-carried counter. lowering
// replaces tileIdx with `arith.constant t`, so the first subtile (tileIdx 0, the
// +0 folds away) flattens to just accumCnt (the counter, which steps by
// numTiles=4 per iteration); the resulting %IDX = (flattened % 3) indexes BOTH
// the 3x128x32 data staging buffer and the 3x1xi64 barrier. This fails on the
// buggy collapsed reuse-position index and on any scheme where the data slot and
// barrier slot diverge.

// CHECK-LABEL: @matmul_kernel_tma_persistent_ws
//
// The epilogue staging buffer is a single shared 3-deep 128x32 alloc.
// CHECK: ttg.local_alloc {allocation.shareGroup = 0 : i32, buffer.copy = 3 : i32, buffer.id = 2 : i32{{.*}}} : () -> !ttg.memdesc<3x128x32xf16
//
// In the epilogue partition (async_task_id = 0): the numTiles factor lives on the
// loop-carried reuse-group counter, which advances by numTiles(4) per iteration
// (`addi %CNT, %c4_i64`). The SAME counter feeds the per-tile slot: tile 0
// (tileIdx 0, the +0 folds) flattens to just %CNT, then % numBuffers(3); the
// resulting %IDX indexes BOTH the 3x128x32 data staging buffer and the 3x1xi64
// barrier (data slot == barrier slot). There is NO in-body `* numTiles`.
// CHECK:      arith.addi %[[CNT:arg[0-9]+]], %c4_i64 {async_task_id = array<i32: 0>}
// CHECK:      %[[DIV:[0-9]+]] = arith.divui %[[CNT]], %c3_i64 {async_task_id = array<i32: 0>}
// CHECK:      %[[MUL:[0-9]+]] = arith.muli %[[DIV]], %c3_i64 {async_task_id = array<i32: 0>}
// CHECK:      %[[MOD:[0-9]+]] = arith.subi %[[CNT]], %[[MUL]] {async_task_id = array<i32: 0>}
// CHECK:      %[[IDX:[0-9]+]] = arith.trunci %[[MOD]] {async_task_id = array<i32: 0>} : i64 to i32
// CHECK:      ttg.memdesc_index %{{[0-9]+}}[%[[IDX]]] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x128x32xf16
// CHECK:      ttg.memdesc_index %{{[0-9]+}}[%[[IDX]]] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64
//
// Tiles 1-3 flatten to %CNT + {1,2,3}: four DISTINCT generations of the shared
// buffer (the buggy generic reuse-position stagger collapsed these to {0,1,1,1},
// aliasing three subtiles onto one slot -> the 49.5%-wrong data race).
// CHECK:      arith.addi %[[CNT]], %c1_i64 {async_task_id = array<i32: 0>}
// CHECK:      arith.addi %[[CNT]], %c2_i64 {async_task_id = array<i32: 0>}
// CHECK:      arith.addi %[[CNT]], %c3_i64 {async_task_id = array<i32: 0>}

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  tt.func public @matmul_kernel_tma_persistent_ws(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<128x64xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x32xf16, #shared1>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c96_i32 = arith.constant 96 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
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
        %33 = tt.descriptor_load %arg0[%17, %32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %34 = ttg.local_alloc %33 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %35 = tt.descriptor_load %arg5[%18, %32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %36 = ttg.local_alloc %35 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %37 = ttg.memdesc_trans %36 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared3, #smem>
        %38 = ttng.tc_gen5_mma %34, %37, %result[%arg21], %arg20, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared3, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %38 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %result_0, %token_1 = ttng.tmem_load %result[%20#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %21 = tt.reshape %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
      %22 = tt.trans %21 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
      %outLHS, %outRHS = tt.split %22 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>
      %r3 = tt.reshape %outLHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> -> tensor<128x2x32xf32, #linear4>
      %r4 = tt.trans %r3 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #linear4> -> tensor<128x32x2xf32, #linear5>
      %t0, %t1 = tt.split %r4 {ttg.partition = array<i32: 0>} : tensor<128x32x2xf32, #linear5> -> tensor<128x32xf32, #linear6>
      %r5 = tt.reshape %outRHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> -> tensor<128x2x32xf32, #linear4>
      %r6 = tt.trans %r5 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #linear4> -> tensor<128x32x2xf32, #linear5>
      %t2, %t3 = tt.split %r6 {ttg.partition = array<i32: 0>} : tensor<128x32x2xf32, #linear5> -> tensor<128x32xf32, #linear6>
      %cn1 = arith.addi %18, %c32_i32 : i32
      %cn2 = arith.addi %18, %c64_i32 : i32
      %cn3 = arith.addi %18, %c96_i32 : i32
      %f0 = arith.truncf %t0 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xf16, #linear6>
      %c0 = ttg.convert_layout %f0 {ttg.partition = array<i32: 0>} : tensor<128x32xf16, #linear6> -> tensor<128x32xf16, #blocked>
      %a0 = ttg.local_alloc %c0 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %s0 = ttng.async_tma_copy_local_to_global %arg10[%17, %18] %a0 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x32xf16, #shared1>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %s0   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %f1 = arith.truncf %t1 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xf16, #linear6>
      %c1 = ttg.convert_layout %f1 {ttg.partition = array<i32: 0>} : tensor<128x32xf16, #linear6> -> tensor<128x32xf16, #blocked>
      %a1 = ttg.local_alloc %c1 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %s1 = ttng.async_tma_copy_local_to_global %arg10[%17, %cn1] %a1 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x32xf16, #shared1>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %s1   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %f2 = arith.truncf %t2 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xf16, #linear6>
      %c2 = ttg.convert_layout %f2 {ttg.partition = array<i32: 0>} : tensor<128x32xf16, #linear6> -> tensor<128x32xf16, #blocked>
      %a2 = ttg.local_alloc %c2 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %s2 = ttng.async_tma_copy_local_to_global %arg10[%17, %cn2] %a2 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x32xf16, #shared1>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %s2   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %f3 = arith.truncf %t3 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xf16, #linear6>
      %c3 = ttg.convert_layout %f3 {ttg.partition = array<i32: 0>} : tensor<128x32xf16, #linear6> -> tensor<128x32xf16, #blocked>
      %a3 = ttg.local_alloc %c3 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %s3 = ttng.async_tma_copy_local_to_global %arg10[%17, %cn3] %a3 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x32xf16, #shared1>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %s3   {ttg.partition = array<i32: 2>} : !ttg.async.token
    } {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
