// RUN: triton-opt %s --nvgpu-warp-specialization="generate-subtiled-region=true num-stages=3 smem-budget=232448" | FileCheck %s

// Test: per-tile barrier index for the SMEM channel that flows through an
// epilogue ttng.subtiled_region. The N subtiles of a tile share ONE barrier
// (reuse group), so the barrier slot/phase must be indexed by a *flattened*
// accumulation count (accumCnt + tileIdx, with accumCnt advancing by numTiles
// per iteration) -- one monotonic stream -- rather than a single shared index.
// Without this, every subtile collapses onto the representative's barrier
// slot/phase and the kernel deadlocks.
//
// This is a reduced version of test_tutorial09 matmul_kernel_tma_persistent_ws
// (EPILOGUE_SUBTILE=2 -> numTiles=2), captured just before
// NVGPUWarpSpecialization.

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

  // CHECK-LABEL: @matmul_kernel_tma_persistent_ws
  // CHECK: ttg.warp_specialize
  //
  // Epilogue producer partition (async_task_id 0): the staging-buffer slot AND
  // the shared barrier slot are both derived IN-BODY from the per-tile flattened
  // count flattened = accumCnt + tileIdx, taken % numBuffers(3). The numTiles
  // factor lives on the loop-carried reuse-group counter, which advances by
  // numTiles(2) per iteration -- so the SAME counter that steps by +2 feeds the
  // slot, and tile 0 (tileIdx 0, the +0 folds) flattens to just accumCnt (there
  // is NO in-body `* numTiles`). The resulting %IDX indexes BOTH the 3x128x64
  // data staging buffer and the 3x1xi64 barrier (data slot == barrier slot).
  // CHECK:      arith.addi %[[CNT:arg[0-9]+]], %c2_i64 {async_task_id = array<i32: 0>}
  // CHECK:      %[[DIV:[0-9]+]] = arith.divui %[[CNT]], %c3_i64 {async_task_id = array<i32: 0>}
  // CHECK:      %[[MUL:[0-9]+]] = arith.muli %[[DIV]], %c3_i64 {async_task_id = array<i32: 0>}
  // CHECK:      %[[MOD:[0-9]+]] = arith.subi %[[CNT]], %[[MUL]] {async_task_id = array<i32: 0>}
  // CHECK:      %[[IDX:[0-9]+]] = arith.trunci %[[MOD]] {async_task_id = array<i32: 0>} : i64 to i32
  // CHECK:      ttg.memdesc_index %{{[0-9]+}}[%[[IDX]]] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x128x64xf16
  // CHECK:      ttng.wait_barrier {{.*}}WSBarrier = {dstTask = 2 : i32}
  // CHECK:      ttg.local_store
  // CHECK:      ttng.arrive_barrier {{.*}}WSBarrier = {dstTask = 2 : i32}
  //
  // The second subtile (tileIdx 1) flattens to `addi accumCnt, %c1_i64`, giving a
  // DISTINCT barrier generation -- the property the buggy shared-index version
  // lacked (it deadlocked).
  // CHECK:      arith.addi %[[CNT]], %c1_i64 {async_task_id = array<i32: 0>}
  // CHECK:      ttng.wait_barrier {{.*}}WSBarrier = {dstTask = 2 : i32}
  // CHECK:      ttg.local_store
  // CHECK:      ttng.arrive_barrier {{.*}}WSBarrier = {dstTask = 2 : i32}
  tt.func public @matmul_kernel_tma_persistent_ws(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<128x64xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x64xf16, #shared>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
        %33 = tt.descriptor_load %arg0[%17, %32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %34 = ttg.local_alloc %33 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %35 = tt.descriptor_load %arg5[%18, %32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
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
      %26 = ttng.async_tma_copy_local_to_global %arg10[%17, %18] %25 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %26   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %27 = arith.addi %18, %c64_i32 : i32
      %28 = arith.truncf %outRHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> to tensor<128x64xf16, #linear3>
      %29 = ttg.convert_layout %28 {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #linear3> -> tensor<128x64xf16, #blocked>
      %30 = ttg.local_alloc %29 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %31 = ttng.async_tma_copy_local_to_global %arg10[%17, %27] %30 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %31   {ttg.partition = array<i32: 2>} : !ttg.async.token
    } {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
