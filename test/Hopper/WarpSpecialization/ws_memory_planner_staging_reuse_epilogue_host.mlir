// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=2 smem-budget=232448" 2>/dev/null | FileCheck %s

// TMA output staging must not reuse a buffer that is still read after a staging
// write in the same loop body. Real planner input: a persistent CLC GEMM with
// separate_epilogue_store=True and EPILOGUE_SUBTILE=4 whose epilogue TMA-loads
// a [64] fp32 per-column vector for each subtile (buffers 10, 14, 18, 22, read
// in subtile order) and TMA-stores a [64] fp32 column partial per subtile
// (staging group 0) plus a [64] row result per tile (staging 4). The column
// partial staging used to land on buffer 14, the second subtile's vector: the
// first subtile's partial overwrote it before it was read (a silent wrong
// answer in the row result). It must land on buffer 10, read before any
// partial is staged; the row result staging, written after every read, may
// take buffer 14.

// CHECK-LABEL: @e114_fused_kernel
// CHECK-COUNT-4: ttg.local_alloc {allocation.reuseTarget = 10 : i32, allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32
// CHECK: ttg.local_alloc {allocation.reuseTarget = 14 : i32, allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [2, 1, 16], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0], [0, 128]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 64]], warp = [[16, 0, 0], [32, 0, 0], [0, 1, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 64, 0]], warp = [[16, 0, 0], [32, 0, 0], [0, 0, 1]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0], [32, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [1, 0, 0]], warp = [[2, 0, 0], [4, 0, 0], [8, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [16, 0], [32, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [1, 0]], warp = [[2, 0], [4, 0], [8, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @e114_fused_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %c8_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i32
    %c192_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 192 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c31_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 31 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 32 : i32
    %c255_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 255 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 256 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} true
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 64 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 63 : i32
    %cst_0 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %9 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %13 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %14 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %15 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %17 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %18 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %19 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %20 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %23 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
    %24 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
    %25 = arith.addi %arg9, %c63_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %26 = arith.divsi %25, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %27 = arith.addi %arg10, %c255_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %28 = arith.divsi %27, %c256_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %29 = arith.addi %arg11, %c31_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %30 = arith.divsi %29, %c32_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %31 = arith.extsi %arg11 {async_task_id = array<i32: 3>} : i32 to i64
    %32 = tt.make_tensor_descriptor %arg0, [%arg9, %arg11], [%31, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x32xbf16, #shared1>
    %33 = arith.extsi %arg10 {async_task_id = array<i32: 3>} : i32 to i64
    %34 = tt.make_tensor_descriptor %arg1, [%arg11, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<32x256xbf16, #shared2>
    %35 = tt.make_tensor_descriptor %arg2, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %36 = tt.make_tensor_descriptor %arg3, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %37 = tt.make_tensor_descriptor %arg5, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %38 = tt.make_tensor_descriptor %arg4, [%arg10], [%c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %39 = tt.make_tensor_descriptor %arg6, [%arg9], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %40 = arith.muli %26, %arg10 {async_task_id = array<i32: 2>} : i32
    %41 = tt.make_tensor_descriptor %arg7, [%40], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %42 = tt.get_program_id x {async_task_id = array<i32: 0, 2, 3>} : i32
    %43 = arith.muli %28, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %44 = tt.make_range {async_task_id = array<i32: 0>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %45 = tt.splat %arg9 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %46 = scf.while (%arg12 = %true, %arg13 = %42) : (i1, i32) -> i32 {
      scf.condition(%arg12) {async_task_id = array<i32: 0, 1, 2, 3>} %arg13 : i32
    } do {
    ^bb0(%arg12: i32):
      %47 = ttng.clc_try_cancel_async {async_task_id = array<i32: 3>} : !ttg.async.token
      %48 = arith.divsi %arg12, %43 {async_task_id = array<i32: 0, 2, 3>} : i32
      %49 = arith.muli %48, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %50 = arith.subi %26, %49 {async_task_id = array<i32: 0, 2, 3>} : i32
      %51 = arith.minsi %50, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %52 = arith.remsi %arg12, %51 {async_task_id = array<i32: 0, 2, 3>} : i32
      %53 = arith.addi %49, %52 {async_task_id = array<i32: 0, 2, 3>} : i32
      %54 = arith.remsi %arg12, %43 {async_task_id = array<i32: 2, 3>} : i32
      %55 = arith.divsi %54, %51 {async_task_id = array<i32: 2, 3>} : i32
      %56 = arith.muli %53, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %57 = arith.muli %55, %c256_i32 {async_task_id = array<i32: 2, 3>} : i32
      %58:2 = scf.for %arg13 = %c0_i32 to %30 step %c1_i32 iter_args(%arg14 = %false, %arg15 = %token) -> (i1, !ttg.async.token)  : i32 {
        %169 = arith.muli %arg13, %c32_i32 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %32[%56, %169] 4096 %5 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<64x32xbf16, #shared1>, i32, i32, !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>
        nvws.descriptor_load %34[%169, %57] 16384 %6 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<32x256xbf16, #shared2>, i32, i32, !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>
        %170 = ttng.tc_gen5_mma %5, %6, %result[%arg15], %arg14, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>, !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %true, %170 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      %result_1, %token_2 = ttng.tmem_load %result[%58#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x256xf32, #linear>
      %59 = tt.reshape %result_1 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear> -> tensor<64x2x128xf32, #linear1>
      %60 = tt.trans %59 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x128xf32, #linear1> -> tensor<64x128x2xf32, #linear2>
      %61 = ttg.convert_layout %60 {async_task_id = array<i32: 0>} : tensor<64x128x2xf32, #linear2> -> tensor<64x128x2xf32, #linear3>
      %outLHS, %outRHS = tt.split %61 {async_task_id = array<i32: 0>} : tensor<64x128x2xf32, #linear3> -> tensor<64x128xf32, #linear4>
      %62 = tt.reshape %outLHS {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear4> -> tensor<64x2x64xf32, #blocked1>
      %63 = tt.trans %62 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked1> -> tensor<64x64x2xf32, #blocked2>
      %outLHS_3, %outRHS_4 = tt.split %63 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #blocked2> -> tensor<64x64xf32, #blocked>
      %64 = tt.reshape %outRHS {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear4> -> tensor<64x2x64xf32, #blocked1>
      %65 = tt.trans %64 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked1> -> tensor<64x64x2xf32, #blocked2>
      %outLHS_5, %outRHS_6 = tt.split %65 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #blocked2> -> tensor<64x64xf32, #blocked>
      %66 = tt.splat %56 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %67 = arith.addi %66, %44 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %68 = arith.cmpi slt, %67, %45 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %69 = arith.truncf %outLHS_3 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %70 = arith.extf %69 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %57] 16384 %7 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %71 = ttg.local_load %7 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %57] 16384 %8 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %72 = ttg.local_load %8 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %73 = arith.mulf %70, %71 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %74 = arith.mulf %73, %72 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %57] 16384 %9 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %75 = ttg.local_load %9 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%57] 256 %10 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %76 = ttg.local_load %10 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %77 = ttg.convert_layout %76 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %78 = tt.expand_dims %77 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %79 = tt.broadcast %78 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %80 = arith.mulf %74, %79 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %81 = arith.mulf %75, %80 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %82 = "tt.reduce"(%81) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %83 = arith.addf %82, %cst_0 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %84 = tt.expand_dims %68 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
      %85 = arith.mulf %74, %75 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %86 = tt.broadcast %84 {async_task_id = array<i32: 0>} : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
      %87 = arith.select %86, %85, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %88 = "tt.reduce"(%87) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %89 = arith.muli %53, %arg10 {async_task_id = array<i32: 2>} : i32
      %90 = arith.addi %89, %57 {async_task_id = array<i32: 2>} : i32
      %91 = ttg.convert_layout %88 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %91, %0 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %92 = ttng.async_tma_copy_local_to_global %41[%90] %0 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %92   {async_task_id = array<i32: 2>} : !ttg.async.token
      %93 = arith.addi %57, %c64_i32 {async_task_id = array<i32: 2, 3>} : i32
      %94 = arith.truncf %outRHS_4 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %95 = arith.extf %94 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %93] 16384 %11 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %96 = ttg.local_load %11 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %93] 16384 %12 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %97 = ttg.local_load %12 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %98 = arith.mulf %95, %96 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %99 = arith.mulf %98, %97 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %93] 16384 %13 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %100 = ttg.local_load %13 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%93] 256 %14 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %101 = ttg.local_load %14 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %102 = ttg.convert_layout %101 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %103 = tt.expand_dims %102 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %104 = tt.broadcast %103 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %105 = arith.mulf %99, %104 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %106 = arith.mulf %100, %105 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %107 = "tt.reduce"(%106) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %108 = arith.addf %83, %107 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %109 = arith.mulf %99, %100 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %110 = arith.select %86, %109, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %111 = "tt.reduce"(%110) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %112 = arith.addi %89, %93 {async_task_id = array<i32: 2>} : i32
      %113 = ttg.convert_layout %111 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %113, %1 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %114 = ttng.async_tma_copy_local_to_global %41[%112] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %114   {async_task_id = array<i32: 2>} : !ttg.async.token
      %115 = arith.addi %57, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %116 = arith.truncf %outLHS_5 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %117 = arith.extf %116 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %115] 16384 %15 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %118 = ttg.local_load %15 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %115] 16384 %16 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %119 = ttg.local_load %16 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %120 = arith.mulf %117, %118 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %121 = arith.mulf %120, %119 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %115] 16384 %17 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %122 = ttg.local_load %17 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%115] 256 %18 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %123 = ttg.local_load %18 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %124 = ttg.convert_layout %123 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %125 = tt.expand_dims %124 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %126 = tt.broadcast %125 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %127 = arith.mulf %121, %126 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %128 = arith.mulf %122, %127 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %129 = "tt.reduce"(%128) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %130 = arith.addf %108, %129 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %131 = arith.mulf %121, %122 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %132 = arith.select %86, %131, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %133 = "tt.reduce"(%132) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %134 = arith.addi %89, %115 {async_task_id = array<i32: 2>} : i32
      %135 = ttg.convert_layout %133 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %135, %2 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %136 = ttng.async_tma_copy_local_to_global %41[%134] %2 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %136   {async_task_id = array<i32: 2>} : !ttg.async.token
      %137 = arith.addi %57, %c192_i32 {async_task_id = array<i32: 2, 3>} : i32
      %138 = arith.truncf %outRHS_6 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %139 = arith.extf %138 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %137] 16384 %19 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %140 = ttg.local_load %19 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %137] 16384 %20 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %141 = ttg.local_load %20 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %142 = arith.mulf %139, %140 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %143 = arith.mulf %142, %141 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %137] 16384 %21 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %144 = ttg.local_load %21 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%137] 256 %22 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %145 = ttg.local_load %22 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %146 = ttg.convert_layout %145 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %147 = tt.expand_dims %146 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %148 = tt.broadcast %147 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %149 = arith.mulf %143, %148 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %150 = arith.mulf %144, %149 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %151 = "tt.reduce"(%150) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %152 = arith.addf %130, %151 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %153 = arith.mulf %143, %144 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %154 = arith.select %86, %153, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %155 = "tt.reduce"(%154) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %156 = arith.addi %89, %137 {async_task_id = array<i32: 2>} : i32
      %157 = ttg.convert_layout %155 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %157, %3 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %158 = ttng.async_tma_copy_local_to_global %41[%156] %3 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %158   {async_task_id = array<i32: 2>} : !ttg.async.token
      %159 = ttg.convert_layout %152 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %159, %4 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %160 = ttng.async_tma_copy_local_to_global %39[%56] %4 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %160   {async_task_id = array<i32: 2>} : !ttg.async.token
      %isValid, %x, %y, %z = ttng.clc_read %47 {async_task_id = array<i32: 3>} : !ttg.async.token -> i1, i32, i32, i32
      %161 = arith.extui %isValid {async_task_id = array<i32: 3>} : i1 to i32
      %162 = tt.splat %161 {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked3>
      ttg.local_store %162, %24 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked3> -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
      %163 = ttg.local_load %24 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared4, #smem, mutable> -> tensor<1xi32, #blocked3>
      %164 = tt.unsplat %163 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked3>
      %165 = arith.trunci %164 {async_task_id = array<i32: 0, 1, 2, 3>} : i32 to i1
      %166 = tt.splat %x {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked3>
      ttg.local_store %166, %23 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked3> -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
      %167 = ttg.local_load %23 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared4, #smem, mutable> -> tensor<1xi32, #blocked3>
      %168 = tt.unsplat %167 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked3>
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %165, %168 : i1, i32
    } attributes {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// Same kernel with the first and last subtiles' vector buffers swapped: buffer
// 10, the planner's first choice, is now read by the last subtile, after three
// partials are staged. The partial staging must take buffer 22 (read first);
// the row staging, written after every read, may take buffer 10.
// CHECK-LABEL: @late_read_host
// CHECK-COUNT-4: ttg.local_alloc {allocation.reuseTarget = 22 : i32, allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32
// CHECK: ttg.local_alloc {allocation.reuseTarget = 10 : i32, allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [2, 1, 16], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0], [0, 128]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 64]], warp = [[16, 0, 0], [32, 0, 0], [0, 1, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 64, 0]], warp = [[16, 0, 0], [32, 0, 0], [0, 0, 1]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0], [32, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [1, 0, 0]], warp = [[2, 0, 0], [4, 0, 0], [8, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [16, 0], [32, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [1, 0]], warp = [[2, 0], [4, 0], [8, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @late_read_host(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %c8_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i32
    %c192_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 192 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c31_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 31 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 32 : i32
    %c255_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 255 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 256 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} true
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 64 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 63 : i32
    %cst_0 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %9 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %13 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %14 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %15 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %17 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %18 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %19 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %20 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %23 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
    %24 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
    %25 = arith.addi %arg9, %c63_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %26 = arith.divsi %25, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %27 = arith.addi %arg10, %c255_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %28 = arith.divsi %27, %c256_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %29 = arith.addi %arg11, %c31_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %30 = arith.divsi %29, %c32_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %31 = arith.extsi %arg11 {async_task_id = array<i32: 3>} : i32 to i64
    %32 = tt.make_tensor_descriptor %arg0, [%arg9, %arg11], [%31, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x32xbf16, #shared1>
    %33 = arith.extsi %arg10 {async_task_id = array<i32: 3>} : i32 to i64
    %34 = tt.make_tensor_descriptor %arg1, [%arg11, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<32x256xbf16, #shared2>
    %35 = tt.make_tensor_descriptor %arg2, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %36 = tt.make_tensor_descriptor %arg3, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %37 = tt.make_tensor_descriptor %arg5, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %38 = tt.make_tensor_descriptor %arg4, [%arg10], [%c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %39 = tt.make_tensor_descriptor %arg6, [%arg9], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %40 = arith.muli %26, %arg10 {async_task_id = array<i32: 2>} : i32
    %41 = tt.make_tensor_descriptor %arg7, [%40], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %42 = tt.get_program_id x {async_task_id = array<i32: 0, 2, 3>} : i32
    %43 = arith.muli %28, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %44 = tt.make_range {async_task_id = array<i32: 0>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %45 = tt.splat %arg9 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %46 = scf.while (%arg12 = %true, %arg13 = %42) : (i1, i32) -> i32 {
      scf.condition(%arg12) {async_task_id = array<i32: 0, 1, 2, 3>} %arg13 : i32
    } do {
    ^bb0(%arg12: i32):
      %47 = ttng.clc_try_cancel_async {async_task_id = array<i32: 3>} : !ttg.async.token
      %48 = arith.divsi %arg12, %43 {async_task_id = array<i32: 0, 2, 3>} : i32
      %49 = arith.muli %48, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %50 = arith.subi %26, %49 {async_task_id = array<i32: 0, 2, 3>} : i32
      %51 = arith.minsi %50, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %52 = arith.remsi %arg12, %51 {async_task_id = array<i32: 0, 2, 3>} : i32
      %53 = arith.addi %49, %52 {async_task_id = array<i32: 0, 2, 3>} : i32
      %54 = arith.remsi %arg12, %43 {async_task_id = array<i32: 2, 3>} : i32
      %55 = arith.divsi %54, %51 {async_task_id = array<i32: 2, 3>} : i32
      %56 = arith.muli %53, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %57 = arith.muli %55, %c256_i32 {async_task_id = array<i32: 2, 3>} : i32
      %58:2 = scf.for %arg13 = %c0_i32 to %30 step %c1_i32 iter_args(%arg14 = %false, %arg15 = %token) -> (i1, !ttg.async.token)  : i32 {
        %169 = arith.muli %arg13, %c32_i32 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %32[%56, %169] 4096 %5 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<64x32xbf16, #shared1>, i32, i32, !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>
        nvws.descriptor_load %34[%169, %57] 16384 %6 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<32x256xbf16, #shared2>, i32, i32, !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>
        %170 = ttng.tc_gen5_mma %5, %6, %result[%arg15], %arg14, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>, !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %true, %170 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      %result_1, %token_2 = ttng.tmem_load %result[%58#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x256xf32, #linear>
      %59 = tt.reshape %result_1 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear> -> tensor<64x2x128xf32, #linear1>
      %60 = tt.trans %59 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x128xf32, #linear1> -> tensor<64x128x2xf32, #linear2>
      %61 = ttg.convert_layout %60 {async_task_id = array<i32: 0>} : tensor<64x128x2xf32, #linear2> -> tensor<64x128x2xf32, #linear3>
      %outLHS, %outRHS = tt.split %61 {async_task_id = array<i32: 0>} : tensor<64x128x2xf32, #linear3> -> tensor<64x128xf32, #linear4>
      %62 = tt.reshape %outLHS {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear4> -> tensor<64x2x64xf32, #blocked1>
      %63 = tt.trans %62 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked1> -> tensor<64x64x2xf32, #blocked2>
      %outLHS_3, %outRHS_4 = tt.split %63 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #blocked2> -> tensor<64x64xf32, #blocked>
      %64 = tt.reshape %outRHS {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear4> -> tensor<64x2x64xf32, #blocked1>
      %65 = tt.trans %64 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked1> -> tensor<64x64x2xf32, #blocked2>
      %outLHS_5, %outRHS_6 = tt.split %65 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #blocked2> -> tensor<64x64xf32, #blocked>
      %66 = tt.splat %56 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %67 = arith.addi %66, %44 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %68 = arith.cmpi slt, %67, %45 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %69 = arith.truncf %outLHS_3 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %70 = arith.extf %69 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %57] 16384 %7 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %71 = ttg.local_load %7 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %57] 16384 %8 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %72 = ttg.local_load %8 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %73 = arith.mulf %70, %71 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %74 = arith.mulf %73, %72 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %57] 16384 %9 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %75 = ttg.local_load %9 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%57] 256 %22 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %76 = ttg.local_load %22 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %77 = ttg.convert_layout %76 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %78 = tt.expand_dims %77 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %79 = tt.broadcast %78 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %80 = arith.mulf %74, %79 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %81 = arith.mulf %75, %80 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %82 = "tt.reduce"(%81) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %83 = arith.addf %82, %cst_0 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %84 = tt.expand_dims %68 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
      %85 = arith.mulf %74, %75 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %86 = tt.broadcast %84 {async_task_id = array<i32: 0>} : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
      %87 = arith.select %86, %85, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %88 = "tt.reduce"(%87) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %89 = arith.muli %53, %arg10 {async_task_id = array<i32: 2>} : i32
      %90 = arith.addi %89, %57 {async_task_id = array<i32: 2>} : i32
      %91 = ttg.convert_layout %88 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %91, %0 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %92 = ttng.async_tma_copy_local_to_global %41[%90] %0 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %92   {async_task_id = array<i32: 2>} : !ttg.async.token
      %93 = arith.addi %57, %c64_i32 {async_task_id = array<i32: 2, 3>} : i32
      %94 = arith.truncf %outRHS_4 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %95 = arith.extf %94 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %93] 16384 %11 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %96 = ttg.local_load %11 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %93] 16384 %12 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %97 = ttg.local_load %12 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %98 = arith.mulf %95, %96 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %99 = arith.mulf %98, %97 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %93] 16384 %13 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %100 = ttg.local_load %13 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%93] 256 %14 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %101 = ttg.local_load %14 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %102 = ttg.convert_layout %101 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %103 = tt.expand_dims %102 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %104 = tt.broadcast %103 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %105 = arith.mulf %99, %104 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %106 = arith.mulf %100, %105 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %107 = "tt.reduce"(%106) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %108 = arith.addf %83, %107 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %109 = arith.mulf %99, %100 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %110 = arith.select %86, %109, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %111 = "tt.reduce"(%110) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %112 = arith.addi %89, %93 {async_task_id = array<i32: 2>} : i32
      %113 = ttg.convert_layout %111 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %113, %1 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %114 = ttng.async_tma_copy_local_to_global %41[%112] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %114   {async_task_id = array<i32: 2>} : !ttg.async.token
      %115 = arith.addi %57, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %116 = arith.truncf %outLHS_5 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %117 = arith.extf %116 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %115] 16384 %15 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %118 = ttg.local_load %15 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %115] 16384 %16 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %119 = ttg.local_load %16 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %120 = arith.mulf %117, %118 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %121 = arith.mulf %120, %119 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %115] 16384 %17 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %122 = ttg.local_load %17 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%115] 256 %18 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %123 = ttg.local_load %18 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %124 = ttg.convert_layout %123 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %125 = tt.expand_dims %124 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %126 = tt.broadcast %125 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %127 = arith.mulf %121, %126 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %128 = arith.mulf %122, %127 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %129 = "tt.reduce"(%128) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %130 = arith.addf %108, %129 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %131 = arith.mulf %121, %122 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %132 = arith.select %86, %131, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %133 = "tt.reduce"(%132) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %134 = arith.addi %89, %115 {async_task_id = array<i32: 2>} : i32
      %135 = ttg.convert_layout %133 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %135, %2 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %136 = ttng.async_tma_copy_local_to_global %41[%134] %2 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %136   {async_task_id = array<i32: 2>} : !ttg.async.token
      %137 = arith.addi %57, %c192_i32 {async_task_id = array<i32: 2, 3>} : i32
      %138 = arith.truncf %outRHS_6 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %139 = arith.extf %138 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %137] 16384 %19 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %140 = ttg.local_load %19 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %137] 16384 %20 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %141 = ttg.local_load %20 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %142 = arith.mulf %139, %140 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %143 = arith.mulf %142, %141 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %137] 16384 %21 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %144 = ttg.local_load %21 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%137] 256 %10 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %145 = ttg.local_load %10 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %146 = ttg.convert_layout %145 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %147 = tt.expand_dims %146 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %148 = tt.broadcast %147 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %149 = arith.mulf %143, %148 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %150 = arith.mulf %144, %149 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %151 = "tt.reduce"(%150) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %152 = arith.addf %130, %151 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %153 = arith.mulf %143, %144 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %154 = arith.select %86, %153, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %155 = "tt.reduce"(%154) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %156 = arith.addi %89, %137 {async_task_id = array<i32: 2>} : i32
      %157 = ttg.convert_layout %155 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %157, %3 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %158 = ttng.async_tma_copy_local_to_global %41[%156] %3 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %158   {async_task_id = array<i32: 2>} : !ttg.async.token
      %159 = ttg.convert_layout %152 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %159, %4 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %160 = ttng.async_tma_copy_local_to_global %39[%56] %4 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %160   {async_task_id = array<i32: 2>} : !ttg.async.token
      %isValid, %x, %y, %z = ttng.clc_read %47 {async_task_id = array<i32: 3>} : !ttg.async.token -> i1, i32, i32, i32
      %161 = arith.extui %isValid {async_task_id = array<i32: 3>} : i1 to i32
      %162 = tt.splat %161 {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked3>
      ttg.local_store %162, %24 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked3> -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
      %163 = ttg.local_load %24 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared4, #smem, mutable> -> tensor<1xi32, #blocked3>
      %164 = tt.unsplat %163 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked3>
      %165 = arith.trunci %164 {async_task_id = array<i32: 0, 1, 2, 3>} : i32 to i1
      %166 = tt.splat %x {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked3>
      ttg.local_store %166, %23 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked3> -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
      %167 = ttg.local_load %23 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared4, #smem, mutable> -> tensor<1xi32, #blocked3>
      %168 = tt.unsplat %167 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked3>
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %165, %168 : i1, i32
    } attributes {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}



// -----

// Same kernel with the epilogue inside a loop nested in the warp-specialized
// loop (an epilogue subtile loop). The program-order argument holds for one
// pass only; a later pass's read follows an earlier pass's staging write with
// nothing ordering them, so the staging must not take any vector buffer.
// CHECK-LABEL: @epilogue_inner_loop
// CHECK-NOT: allocation.reuseTarget
// CHECK: tt.return
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [2, 1, 16], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0], [0, 128]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 64]], warp = [[16, 0, 0], [32, 0, 0], [0, 1, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 64, 0]], warp = [[16, 0, 0], [32, 0, 0], [0, 0, 1]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0], [32, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [1, 0, 0]], warp = [[2, 0, 0], [4, 0, 0], [8, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [16, 0], [32, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [1, 0]], warp = [[2, 0], [4, 0], [8, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @epilogue_inner_loop(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %c8_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i32
    %c192_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 192 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c31_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 31 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 32 : i32
    %c255_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 255 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 256 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} true
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 64 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 63 : i32
    %cst_0 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %9 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %13 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %14 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %15 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %17 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %18 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %19 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %20 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %23 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
    %24 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
    %25 = arith.addi %arg9, %c63_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %26 = arith.divsi %25, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %27 = arith.addi %arg10, %c255_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %28 = arith.divsi %27, %c256_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %29 = arith.addi %arg11, %c31_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %30 = arith.divsi %29, %c32_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %31 = arith.extsi %arg11 {async_task_id = array<i32: 3>} : i32 to i64
    %32 = tt.make_tensor_descriptor %arg0, [%arg9, %arg11], [%31, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x32xbf16, #shared1>
    %33 = arith.extsi %arg10 {async_task_id = array<i32: 3>} : i32 to i64
    %34 = tt.make_tensor_descriptor %arg1, [%arg11, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<32x256xbf16, #shared2>
    %35 = tt.make_tensor_descriptor %arg2, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %36 = tt.make_tensor_descriptor %arg3, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %37 = tt.make_tensor_descriptor %arg5, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %38 = tt.make_tensor_descriptor %arg4, [%arg10], [%c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %39 = tt.make_tensor_descriptor %arg6, [%arg9], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %40 = arith.muli %26, %arg10 {async_task_id = array<i32: 2>} : i32
    %41 = tt.make_tensor_descriptor %arg7, [%40], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %42 = tt.get_program_id x {async_task_id = array<i32: 0, 2, 3>} : i32
    %43 = arith.muli %28, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %44 = tt.make_range {async_task_id = array<i32: 0>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %45 = tt.splat %arg9 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %46 = scf.while (%arg12 = %true, %arg13 = %42) : (i1, i32) -> i32 {
      scf.condition(%arg12) {async_task_id = array<i32: 0, 1, 2, 3>} %arg13 : i32
    } do {
    ^bb0(%arg12: i32):
      %47 = ttng.clc_try_cancel_async {async_task_id = array<i32: 3>} : !ttg.async.token
      %48 = arith.divsi %arg12, %43 {async_task_id = array<i32: 0, 2, 3>} : i32
      %49 = arith.muli %48, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %50 = arith.subi %26, %49 {async_task_id = array<i32: 0, 2, 3>} : i32
      %51 = arith.minsi %50, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %52 = arith.remsi %arg12, %51 {async_task_id = array<i32: 0, 2, 3>} : i32
      %53 = arith.addi %49, %52 {async_task_id = array<i32: 0, 2, 3>} : i32
      %54 = arith.remsi %arg12, %43 {async_task_id = array<i32: 2, 3>} : i32
      %55 = arith.divsi %54, %51 {async_task_id = array<i32: 2, 3>} : i32
      %56 = arith.muli %53, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %57 = arith.muli %55, %c256_i32 {async_task_id = array<i32: 2, 3>} : i32
      %58:2 = scf.for %arg13 = %c0_i32 to %30 step %c1_i32 iter_args(%arg14 = %false, %arg15 = %token) -> (i1, !ttg.async.token)  : i32 {
        %169 = arith.muli %arg13, %c32_i32 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %32[%56, %169] 4096 %5 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<64x32xbf16, #shared1>, i32, i32, !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>
        nvws.descriptor_load %34[%169, %57] 16384 %6 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<32x256xbf16, #shared2>, i32, i32, !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>
        %170 = ttng.tc_gen5_mma %5, %6, %result[%arg15], %arg14, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>, !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %true, %170 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      %result_1, %token_2 = ttng.tmem_load %result[%58#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x256xf32, #linear>
      %59 = tt.reshape %result_1 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear> -> tensor<64x2x128xf32, #linear1>
      %60 = tt.trans %59 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x128xf32, #linear1> -> tensor<64x128x2xf32, #linear2>
      %61 = ttg.convert_layout %60 {async_task_id = array<i32: 0>} : tensor<64x128x2xf32, #linear2> -> tensor<64x128x2xf32, #linear3>
      %outLHS, %outRHS = tt.split %61 {async_task_id = array<i32: 0>} : tensor<64x128x2xf32, #linear3> -> tensor<64x128xf32, #linear4>
      %62 = tt.reshape %outLHS {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear4> -> tensor<64x2x64xf32, #blocked1>
      %63 = tt.trans %62 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked1> -> tensor<64x64x2xf32, #blocked2>
      %outLHS_3, %outRHS_4 = tt.split %63 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #blocked2> -> tensor<64x64xf32, #blocked>
      %64 = tt.reshape %outRHS {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear4> -> tensor<64x2x64xf32, #blocked1>
      %65 = tt.trans %64 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked1> -> tensor<64x64x2xf32, #blocked2>
      %outLHS_5, %outRHS_6 = tt.split %65 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #blocked2> -> tensor<64x64xf32, #blocked>
      %66 = tt.splat %56 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %67 = arith.addi %66, %44 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %68 = arith.cmpi slt, %67, %45 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %69 = arith.truncf %outLHS_3 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %70 = arith.extf %69 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %57] 16384 %7 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %71 = ttg.local_load %7 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %57] 16384 %8 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %72 = ttg.local_load %8 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %73 = arith.mulf %70, %71 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %74 = arith.mulf %73, %72 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %57] 16384 %9 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %75 = ttg.local_load %9 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      scf.for %jj = %c0_i32 to %c1_i32 step %c1_i32  : i32 {
        scf.for %kk = %c0_i32 to %c1_i32 step %c1_i32  : i32 {
        } {async_task_id = array<i32: 0, 1, 2, 3>}
        nvws.descriptor_load %38[%57] 256 %10 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
        %76 = ttg.local_load %10 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
        %77 = ttg.convert_layout %76 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %78 = tt.expand_dims %77 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
        %79 = tt.broadcast %78 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
        %80 = arith.mulf %74, %79 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %81 = arith.mulf %75, %80 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %82 = "tt.reduce"(%81) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
          tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
        }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %83 = arith.addf %82, %cst_0 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %84 = tt.expand_dims %68 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
        %85 = arith.mulf %74, %75 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %86 = tt.broadcast %84 {async_task_id = array<i32: 0>} : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %87 = arith.select %86, %85, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
        %88 = "tt.reduce"(%87) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
          tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
        }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %89 = arith.muli %53, %arg10 {async_task_id = array<i32: 2>} : i32
        %90 = arith.addi %89, %57 {async_task_id = array<i32: 2>} : i32
        %91 = ttg.convert_layout %88 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
        ttg.local_store %91, %0 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
        %92 = ttng.async_tma_copy_local_to_global %41[%90] %0 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %92   {async_task_id = array<i32: 2>} : !ttg.async.token
        %93 = arith.addi %57, %c64_i32 {async_task_id = array<i32: 2, 3>} : i32
        %94 = arith.truncf %outRHS_4 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
        %95 = arith.extf %94 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
        nvws.descriptor_load %35[%56, %93] 16384 %11 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
        %96 = ttg.local_load %11 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
        nvws.descriptor_load %36[%56, %93] 16384 %12 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
        %97 = ttg.local_load %12 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
        %98 = arith.mulf %95, %96 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %99 = arith.mulf %98, %97 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        nvws.descriptor_load %37[%56, %93] 16384 %13 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
        %100 = ttg.local_load %13 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
        nvws.descriptor_load %38[%93] 256 %14 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
        %101 = ttg.local_load %14 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
        %102 = ttg.convert_layout %101 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %103 = tt.expand_dims %102 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
        %104 = tt.broadcast %103 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
        %105 = arith.mulf %99, %104 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %106 = arith.mulf %100, %105 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %107 = "tt.reduce"(%106) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
          tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
        }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %108 = arith.addf %83, %107 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %109 = arith.mulf %99, %100 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %110 = arith.select %86, %109, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
        %111 = "tt.reduce"(%110) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
          tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
        }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %112 = arith.addi %89, %93 {async_task_id = array<i32: 2>} : i32
        %113 = ttg.convert_layout %111 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
        ttg.local_store %113, %1 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
        %114 = ttng.async_tma_copy_local_to_global %41[%112] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %114   {async_task_id = array<i32: 2>} : !ttg.async.token
        %115 = arith.addi %57, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
        %116 = arith.truncf %outLHS_5 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
        %117 = arith.extf %116 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
        nvws.descriptor_load %35[%56, %115] 16384 %15 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
        %118 = ttg.local_load %15 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
        nvws.descriptor_load %36[%56, %115] 16384 %16 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
        %119 = ttg.local_load %16 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
        %120 = arith.mulf %117, %118 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %121 = arith.mulf %120, %119 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        nvws.descriptor_load %37[%56, %115] 16384 %17 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
        %122 = ttg.local_load %17 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
        nvws.descriptor_load %38[%115] 256 %18 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
        %123 = ttg.local_load %18 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
        %124 = ttg.convert_layout %123 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %125 = tt.expand_dims %124 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
        %126 = tt.broadcast %125 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
        %127 = arith.mulf %121, %126 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %128 = arith.mulf %122, %127 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %129 = "tt.reduce"(%128) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
          tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
        }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %130 = arith.addf %108, %129 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %131 = arith.mulf %121, %122 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %132 = arith.select %86, %131, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
        %133 = "tt.reduce"(%132) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
          tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
        }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %134 = arith.addi %89, %115 {async_task_id = array<i32: 2>} : i32
        %135 = ttg.convert_layout %133 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
        ttg.local_store %135, %2 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
        %136 = ttng.async_tma_copy_local_to_global %41[%134] %2 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %136   {async_task_id = array<i32: 2>} : !ttg.async.token
        %137 = arith.addi %57, %c192_i32 {async_task_id = array<i32: 2, 3>} : i32
        %138 = arith.truncf %outRHS_6 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
        %139 = arith.extf %138 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
        nvws.descriptor_load %35[%56, %137] 16384 %19 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
        %140 = ttg.local_load %19 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
        nvws.descriptor_load %36[%56, %137] 16384 %20 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
        %141 = ttg.local_load %20 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
        %142 = arith.mulf %139, %140 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %143 = arith.mulf %142, %141 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        nvws.descriptor_load %37[%56, %137] 16384 %21 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
        %144 = ttg.local_load %21 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
        nvws.descriptor_load %38[%137] 256 %22 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
        %145 = ttg.local_load %22 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
        %146 = ttg.convert_layout %145 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %147 = tt.expand_dims %146 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
        %148 = tt.broadcast %147 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
        %149 = arith.mulf %143, %148 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %150 = arith.mulf %144, %149 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %151 = "tt.reduce"(%150) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
          tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
        }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %152 = arith.addf %130, %151 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %153 = arith.mulf %143, %144 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
        %154 = arith.select %86, %153, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
        %155 = "tt.reduce"(%154) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
          tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
        }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %156 = arith.addi %89, %137 {async_task_id = array<i32: 2>} : i32
        %157 = ttg.convert_layout %155 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
        ttg.local_store %157, %3 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
        %158 = ttng.async_tma_copy_local_to_global %41[%156] %3 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %158   {async_task_id = array<i32: 2>} : !ttg.async.token
        %159 = ttg.convert_layout %152 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked3>
        ttg.local_store %159, %4 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
        %160 = ttng.async_tma_copy_local_to_global %39[%56] %4 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %160   {async_task_id = array<i32: 2>} : !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>}
      %isValid, %x, %y, %z = ttng.clc_read %47 {async_task_id = array<i32: 3>} : !ttg.async.token -> i1, i32, i32, i32
      %161 = arith.extui %isValid {async_task_id = array<i32: 3>} : i1 to i32
      %162 = tt.splat %161 {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked3>
      ttg.local_store %162, %24 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked3> -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
      %163 = ttg.local_load %24 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared4, #smem, mutable> -> tensor<1xi32, #blocked3>
      %164 = tt.unsplat %163 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked3>
      %165 = arith.trunci %164 {async_task_id = array<i32: 0, 1, 2, 3>} : i32 to i1
      %166 = tt.splat %x {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked3>
      ttg.local_store %166, %23 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked3> -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
      %167 = ttg.local_load %23 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared4, #smem, mutable> -> tensor<1xi32, #blocked3>
      %168 = tt.unsplat %167 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked3>
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %165, %168 : i1, i32
    } attributes {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// The late-read variant with that read (of buffer 10) wrapped in a one-trip
// loop placed after the staged partials: an inner loop counts at its position,
// so it is still read after a staging write. The partial staging must take
// buffer 22, not 10.
// CHECK-LABEL: @late_read_in_inner_loop
// CHECK-COUNT-4: ttg.local_alloc {allocation.reuseTarget = 22 : i32, allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [2, 1, 16], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0], [0, 128]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 64]], warp = [[16, 0, 0], [32, 0, 0], [0, 1, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 64, 0]], warp = [[16, 0, 0], [32, 0, 0], [0, 0, 1]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0], [32, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [1, 0, 0]], warp = [[2, 0, 0], [4, 0, 0], [8, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [16, 0], [32, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [1, 0]], warp = [[2, 0], [4, 0], [8, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @late_read_in_inner_loop(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %c8_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i32
    %c192_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 192 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %cst_adv = arith.constant {async_task_id = array<i32: 0>} dense<0.0> : tensor<64xf32, #blocked3>
    %c31_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 31 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 32 : i32
    %c255_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 255 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 256 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} true
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 64 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 63 : i32
    %cst_0 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %9 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %13 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %14 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %15 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %17 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %18 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %19 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %20 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %23 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
    %24 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
    %25 = arith.addi %arg9, %c63_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %26 = arith.divsi %25, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %27 = arith.addi %arg10, %c255_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %28 = arith.divsi %27, %c256_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %29 = arith.addi %arg11, %c31_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %30 = arith.divsi %29, %c32_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %31 = arith.extsi %arg11 {async_task_id = array<i32: 3>} : i32 to i64
    %32 = tt.make_tensor_descriptor %arg0, [%arg9, %arg11], [%31, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x32xbf16, #shared1>
    %33 = arith.extsi %arg10 {async_task_id = array<i32: 3>} : i32 to i64
    %34 = tt.make_tensor_descriptor %arg1, [%arg11, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<32x256xbf16, #shared2>
    %35 = tt.make_tensor_descriptor %arg2, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %36 = tt.make_tensor_descriptor %arg3, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %37 = tt.make_tensor_descriptor %arg5, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %38 = tt.make_tensor_descriptor %arg4, [%arg10], [%c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %39 = tt.make_tensor_descriptor %arg6, [%arg9], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %40 = arith.muli %26, %arg10 {async_task_id = array<i32: 2>} : i32
    %41 = tt.make_tensor_descriptor %arg7, [%40], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %42 = tt.get_program_id x {async_task_id = array<i32: 0, 2, 3>} : i32
    %43 = arith.muli %28, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %44 = tt.make_range {async_task_id = array<i32: 0>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %45 = tt.splat %arg9 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %46 = scf.while (%arg12 = %true, %arg13 = %42) : (i1, i32) -> i32 {
      scf.condition(%arg12) {async_task_id = array<i32: 0, 1, 2, 3>} %arg13 : i32
    } do {
    ^bb0(%arg12: i32):
      %47 = ttng.clc_try_cancel_async {async_task_id = array<i32: 3>} : !ttg.async.token
      %48 = arith.divsi %arg12, %43 {async_task_id = array<i32: 0, 2, 3>} : i32
      %49 = arith.muli %48, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %50 = arith.subi %26, %49 {async_task_id = array<i32: 0, 2, 3>} : i32
      %51 = arith.minsi %50, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %52 = arith.remsi %arg12, %51 {async_task_id = array<i32: 0, 2, 3>} : i32
      %53 = arith.addi %49, %52 {async_task_id = array<i32: 0, 2, 3>} : i32
      %54 = arith.remsi %arg12, %43 {async_task_id = array<i32: 2, 3>} : i32
      %55 = arith.divsi %54, %51 {async_task_id = array<i32: 2, 3>} : i32
      %56 = arith.muli %53, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %57 = arith.muli %55, %c256_i32 {async_task_id = array<i32: 2, 3>} : i32
      %58:2 = scf.for %arg13 = %c0_i32 to %30 step %c1_i32 iter_args(%arg14 = %false, %arg15 = %token) -> (i1, !ttg.async.token)  : i32 {
        %169 = arith.muli %arg13, %c32_i32 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %32[%56, %169] 4096 %5 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<64x32xbf16, #shared1>, i32, i32, !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>
        nvws.descriptor_load %34[%169, %57] 16384 %6 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<32x256xbf16, #shared2>, i32, i32, !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>
        %170 = ttng.tc_gen5_mma %5, %6, %result[%arg15], %arg14, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>, !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %true, %170 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      %result_1, %token_2 = ttng.tmem_load %result[%58#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x256xf32, #linear>
      %59 = tt.reshape %result_1 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear> -> tensor<64x2x128xf32, #linear1>
      %60 = tt.trans %59 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x128xf32, #linear1> -> tensor<64x128x2xf32, #linear2>
      %61 = ttg.convert_layout %60 {async_task_id = array<i32: 0>} : tensor<64x128x2xf32, #linear2> -> tensor<64x128x2xf32, #linear3>
      %outLHS, %outRHS = tt.split %61 {async_task_id = array<i32: 0>} : tensor<64x128x2xf32, #linear3> -> tensor<64x128xf32, #linear4>
      %62 = tt.reshape %outLHS {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear4> -> tensor<64x2x64xf32, #blocked1>
      %63 = tt.trans %62 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked1> -> tensor<64x64x2xf32, #blocked2>
      %outLHS_3, %outRHS_4 = tt.split %63 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #blocked2> -> tensor<64x64xf32, #blocked>
      %64 = tt.reshape %outRHS {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear4> -> tensor<64x2x64xf32, #blocked1>
      %65 = tt.trans %64 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked1> -> tensor<64x64x2xf32, #blocked2>
      %outLHS_5, %outRHS_6 = tt.split %65 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #blocked2> -> tensor<64x64xf32, #blocked>
      %66 = tt.splat %56 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %67 = arith.addi %66, %44 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %68 = arith.cmpi slt, %67, %45 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %69 = arith.truncf %outLHS_3 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %70 = arith.extf %69 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %57] 16384 %7 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %71 = ttg.local_load %7 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %57] 16384 %8 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %72 = ttg.local_load %8 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %73 = arith.mulf %70, %71 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %74 = arith.mulf %73, %72 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %57] 16384 %9 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %75 = ttg.local_load %9 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%57] 256 %22 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %76 = ttg.local_load %22 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %77 = ttg.convert_layout %76 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %78 = tt.expand_dims %77 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %79 = tt.broadcast %78 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %80 = arith.mulf %74, %79 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %81 = arith.mulf %75, %80 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %82 = "tt.reduce"(%81) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %83 = arith.addf %82, %cst_0 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %84 = tt.expand_dims %68 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
      %85 = arith.mulf %74, %75 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %86 = tt.broadcast %84 {async_task_id = array<i32: 0>} : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
      %87 = arith.select %86, %85, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %88 = "tt.reduce"(%87) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %89 = arith.muli %53, %arg10 {async_task_id = array<i32: 2>} : i32
      %90 = arith.addi %89, %57 {async_task_id = array<i32: 2>} : i32
      %91 = ttg.convert_layout %88 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %91, %0 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %92 = ttng.async_tma_copy_local_to_global %41[%90] %0 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %92   {async_task_id = array<i32: 2>} : !ttg.async.token
      %93 = arith.addi %57, %c64_i32 {async_task_id = array<i32: 2, 3>} : i32
      %94 = arith.truncf %outRHS_4 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %95 = arith.extf %94 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %93] 16384 %11 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %96 = ttg.local_load %11 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %93] 16384 %12 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %97 = ttg.local_load %12 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %98 = arith.mulf %95, %96 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %99 = arith.mulf %98, %97 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %93] 16384 %13 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %100 = ttg.local_load %13 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%93] 256 %14 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %101 = ttg.local_load %14 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %102 = ttg.convert_layout %101 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %103 = tt.expand_dims %102 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %104 = tt.broadcast %103 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %105 = arith.mulf %99, %104 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %106 = arith.mulf %100, %105 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %107 = "tt.reduce"(%106) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %108 = arith.addf %83, %107 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %109 = arith.mulf %99, %100 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %110 = arith.select %86, %109, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %111 = "tt.reduce"(%110) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %112 = arith.addi %89, %93 {async_task_id = array<i32: 2>} : i32
      %113 = ttg.convert_layout %111 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %113, %1 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %114 = ttng.async_tma_copy_local_to_global %41[%112] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %114   {async_task_id = array<i32: 2>} : !ttg.async.token
      %115 = arith.addi %57, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %116 = arith.truncf %outLHS_5 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %117 = arith.extf %116 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %115] 16384 %15 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %118 = ttg.local_load %15 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %115] 16384 %16 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %119 = ttg.local_load %16 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %120 = arith.mulf %117, %118 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %121 = arith.mulf %120, %119 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %115] 16384 %17 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %122 = ttg.local_load %17 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%115] 256 %18 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %123 = ttg.local_load %18 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %124 = ttg.convert_layout %123 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %125 = tt.expand_dims %124 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %126 = tt.broadcast %125 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %127 = arith.mulf %121, %126 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %128 = arith.mulf %122, %127 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %129 = "tt.reduce"(%128) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %130 = arith.addf %108, %129 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %131 = arith.mulf %121, %122 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %132 = arith.select %86, %131, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %133 = "tt.reduce"(%132) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %134 = arith.addi %89, %115 {async_task_id = array<i32: 2>} : i32
      %135 = ttg.convert_layout %133 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %135, %2 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %136 = ttng.async_tma_copy_local_to_global %41[%134] %2 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %136   {async_task_id = array<i32: 2>} : !ttg.async.token
      %137 = arith.addi %57, %c192_i32 {async_task_id = array<i32: 2, 3>} : i32
      %138 = arith.truncf %outRHS_6 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %139 = arith.extf %138 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %137] 16384 %19 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %140 = ttg.local_load %19 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %137] 16384 %20 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %141 = ttg.local_load %20 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %142 = arith.mulf %139, %140 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %143 = arith.mulf %142, %141 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %137] 16384 %21 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %144 = ttg.local_load %21 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%137] 256 %10 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %145 = scf.for %adv_i = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%adv_a = %cst_adv) -> (tensor<64xf32, #blocked3>)  : i32 {
      %adv_v = ttg.local_load %10 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
        scf.yield {async_task_id = array<i32: 0>} %adv_v : tensor<64xf32, #blocked3>
      } {async_task_id = array<i32: 0>}
      %146 = ttg.convert_layout %145 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %147 = tt.expand_dims %146 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %148 = tt.broadcast %147 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %149 = arith.mulf %143, %148 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %150 = arith.mulf %144, %149 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %151 = "tt.reduce"(%150) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %152 = arith.addf %130, %151 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %153 = arith.mulf %143, %144 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %154 = arith.select %86, %153, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %155 = "tt.reduce"(%154) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %156 = arith.addi %89, %137 {async_task_id = array<i32: 2>} : i32
      %157 = ttg.convert_layout %155 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %157, %3 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %158 = ttng.async_tma_copy_local_to_global %41[%156] %3 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %158   {async_task_id = array<i32: 2>} : !ttg.async.token
      %159 = ttg.convert_layout %152 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %159, %4 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %160 = ttng.async_tma_copy_local_to_global %39[%56] %4 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %160   {async_task_id = array<i32: 2>} : !ttg.async.token
      %isValid, %x, %y, %z = ttng.clc_read %47 {async_task_id = array<i32: 3>} : !ttg.async.token -> i1, i32, i32, i32
      %161 = arith.extui %isValid {async_task_id = array<i32: 3>} : i1 to i32
      %162 = tt.splat %161 {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked3>
      ttg.local_store %162, %24 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked3> -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
      %163 = ttg.local_load %24 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared4, #smem, mutable> -> tensor<1xi32, #blocked3>
      %164 = tt.unsplat %163 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked3>
      %165 = arith.trunci %164 {async_task_id = array<i32: 0, 1, 2, 3>} : i32 to i1
      %166 = tt.splat %x {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked3>
      ttg.local_store %166, %23 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked3> -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
      %167 = ttg.local_load %23 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared4, #smem, mutable> -> tensor<1xi32, #blocked3>
      %168 = tt.unsplat %167 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked3>
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %165, %168 : i1, i32
    } attributes {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// The late-read variant with the first-read vector (buffer 22) read under an
// scf.if by a different partition (task 1) than the staging writer (task 0).
// Program order does not order partitions, so buffer 22 is rejected; no other
// vector buffer is read before the first staged partial, so the partial
// staging stays standalone. The row staging may still take buffer 10.
// CHECK-LABEL: @cross_partition_read_in_if
// CHECK-NOT: allocation.reuseTarget = 22
// CHECK-COUNT-4: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32
// CHECK: ttg.local_alloc {allocation.reuseTarget = 10 : i32, allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<64xf32
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [2, 1, 16], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0], [0, 128]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 64]], warp = [[16, 0, 0], [32, 0, 0], [0, 1, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 64, 0]], warp = [[16, 0, 0], [32, 0, 0], [0, 0, 1]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0], [32, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [1, 0, 0]], warp = [[2, 0, 0], [4, 0, 0], [8, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [16, 0], [32, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [1, 0]], warp = [[2, 0], [4, 0], [8, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @cross_partition_read_in_if(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %c8_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i32
    %c192_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 192 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c31_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 31 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 32 : i32
    %c255_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 255 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 256 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} true
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 64 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 2, 3>} 63 : i32
    %cst_0 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %9 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %13 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %14 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %15 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %17 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %18 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %19 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %20 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %23 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
    %24 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
    %25 = arith.addi %arg9, %c63_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %26 = arith.divsi %25, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %27 = arith.addi %arg10, %c255_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %28 = arith.divsi %27, %c256_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %29 = arith.addi %arg11, %c31_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %30 = arith.divsi %29, %c32_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %31 = arith.extsi %arg11 {async_task_id = array<i32: 3>} : i32 to i64
    %32 = tt.make_tensor_descriptor %arg0, [%arg9, %arg11], [%31, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x32xbf16, #shared1>
    %33 = arith.extsi %arg10 {async_task_id = array<i32: 3>} : i32 to i64
    %34 = tt.make_tensor_descriptor %arg1, [%arg11, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<32x256xbf16, #shared2>
    %35 = tt.make_tensor_descriptor %arg2, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %36 = tt.make_tensor_descriptor %arg3, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %37 = tt.make_tensor_descriptor %arg5, [%arg9, %arg10], [%33, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x64xf32, #shared3>
    %38 = tt.make_tensor_descriptor %arg4, [%arg10], [%c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %39 = tt.make_tensor_descriptor %arg6, [%arg9], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %40 = arith.muli %26, %arg10 {async_task_id = array<i32: 2>} : i32
    %41 = tt.make_tensor_descriptor %arg7, [%40], [%c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<64xf32, #shared>
    %42 = tt.get_program_id x {async_task_id = array<i32: 0, 2, 3>} : i32
    %43 = arith.muli %28, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
    %44 = tt.make_range {async_task_id = array<i32: 0>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %45 = tt.splat %arg9 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %46 = scf.while (%arg12 = %true, %arg13 = %42) : (i1, i32) -> i32 {
      scf.condition(%arg12) {async_task_id = array<i32: 0, 1, 2, 3>} %arg13 : i32
    } do {
    ^bb0(%arg12: i32):
      %47 = ttng.clc_try_cancel_async {async_task_id = array<i32: 3>} : !ttg.async.token
      %48 = arith.divsi %arg12, %43 {async_task_id = array<i32: 0, 2, 3>} : i32
      %49 = arith.muli %48, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %50 = arith.subi %26, %49 {async_task_id = array<i32: 0, 2, 3>} : i32
      %51 = arith.minsi %50, %c8_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %52 = arith.remsi %arg12, %51 {async_task_id = array<i32: 0, 2, 3>} : i32
      %53 = arith.addi %49, %52 {async_task_id = array<i32: 0, 2, 3>} : i32
      %54 = arith.remsi %arg12, %43 {async_task_id = array<i32: 2, 3>} : i32
      %55 = arith.divsi %54, %51 {async_task_id = array<i32: 2, 3>} : i32
      %56 = arith.muli %53, %c64_i32 {async_task_id = array<i32: 0, 2, 3>} : i32
      %57 = arith.muli %55, %c256_i32 {async_task_id = array<i32: 2, 3>} : i32
      %58:2 = scf.for %arg13 = %c0_i32 to %30 step %c1_i32 iter_args(%arg14 = %false, %arg15 = %token) -> (i1, !ttg.async.token)  : i32 {
        %169 = arith.muli %arg13, %c32_i32 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %32[%56, %169] 4096 %5 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<64x32xbf16, #shared1>, i32, i32, !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>
        nvws.descriptor_load %34[%169, %57] 16384 %6 {async_task_id = array<i32: 3>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<32x256xbf16, #shared2>, i32, i32, !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>
        %170 = ttng.tc_gen5_mma %5, %6, %result[%arg15], %arg14, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<64x32xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared2, #smem, mutable>, !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %true, %170 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      %result_1, %token_2 = ttng.tmem_load %result[%58#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x256xf32, #linear>
      %59 = tt.reshape %result_1 {async_task_id = array<i32: 0>} : tensor<64x256xf32, #linear> -> tensor<64x2x128xf32, #linear1>
      %60 = tt.trans %59 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x128xf32, #linear1> -> tensor<64x128x2xf32, #linear2>
      %61 = ttg.convert_layout %60 {async_task_id = array<i32: 0>} : tensor<64x128x2xf32, #linear2> -> tensor<64x128x2xf32, #linear3>
      %outLHS, %outRHS = tt.split %61 {async_task_id = array<i32: 0>} : tensor<64x128x2xf32, #linear3> -> tensor<64x128xf32, #linear4>
      %62 = tt.reshape %outLHS {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear4> -> tensor<64x2x64xf32, #blocked1>
      %63 = tt.trans %62 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked1> -> tensor<64x64x2xf32, #blocked2>
      %outLHS_3, %outRHS_4 = tt.split %63 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #blocked2> -> tensor<64x64xf32, #blocked>
      %64 = tt.reshape %outRHS {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear4> -> tensor<64x2x64xf32, #blocked1>
      %65 = tt.trans %64 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #blocked1> -> tensor<64x64x2xf32, #blocked2>
      %outLHS_5, %outRHS_6 = tt.split %65 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #blocked2> -> tensor<64x64xf32, #blocked>
      %66 = tt.splat %56 {async_task_id = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %67 = arith.addi %66, %44 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %68 = arith.cmpi slt, %67, %45 {async_task_id = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %69 = arith.truncf %outLHS_3 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %70 = arith.extf %69 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %57] 16384 %7 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %71 = ttg.local_load %7 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %57] 16384 %8 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %72 = ttg.local_load %8 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %73 = arith.mulf %70, %71 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %74 = arith.mulf %73, %72 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %57] 16384 %9 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %75 = ttg.local_load %9 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%57] 256 %22 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %76 = scf.if %true -> (tensor<64xf32, #blocked3>) {
        %if_r = ttg.local_load %22 {async_task_id = array<i32: 1>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
        scf.yield {async_task_id = array<i32: 0, 1>} %if_r : tensor<64xf32, #blocked3>
      } else {
        %if_z = arith.constant {async_task_id = array<i32: 0, 1>} dense<0.000000e+00> : tensor<64xf32, #blocked3>
        scf.yield {async_task_id = array<i32: 0, 1>} %if_z : tensor<64xf32, #blocked3>
      } {async_task_id = array<i32: 0, 1>}
      %77 = ttg.convert_layout %76 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %78 = tt.expand_dims %77 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %79 = tt.broadcast %78 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %80 = arith.mulf %74, %79 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %81 = arith.mulf %75, %80 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %82 = "tt.reduce"(%81) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %83 = arith.addf %82, %cst_0 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %84 = tt.expand_dims %68 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<64xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked>
      %85 = arith.mulf %74, %75 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %86 = tt.broadcast %84 {async_task_id = array<i32: 0>} : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
      %87 = arith.select %86, %85, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %88 = "tt.reduce"(%87) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %89 = arith.muli %53, %arg10 {async_task_id = array<i32: 2>} : i32
      %90 = arith.addi %89, %57 {async_task_id = array<i32: 2>} : i32
      %91 = ttg.convert_layout %88 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %91, %0 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %92 = ttng.async_tma_copy_local_to_global %41[%90] %0 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %92   {async_task_id = array<i32: 2>} : !ttg.async.token
      %93 = arith.addi %57, %c64_i32 {async_task_id = array<i32: 2, 3>} : i32
      %94 = arith.truncf %outRHS_4 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %95 = arith.extf %94 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %93] 16384 %11 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %96 = ttg.local_load %11 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %93] 16384 %12 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %97 = ttg.local_load %12 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %98 = arith.mulf %95, %96 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %99 = arith.mulf %98, %97 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %93] 16384 %13 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %100 = ttg.local_load %13 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%93] 256 %14 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %101 = ttg.local_load %14 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %102 = ttg.convert_layout %101 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %103 = tt.expand_dims %102 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %104 = tt.broadcast %103 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %105 = arith.mulf %99, %104 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %106 = arith.mulf %100, %105 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %107 = "tt.reduce"(%106) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %108 = arith.addf %83, %107 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %109 = arith.mulf %99, %100 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %110 = arith.select %86, %109, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %111 = "tt.reduce"(%110) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %112 = arith.addi %89, %93 {async_task_id = array<i32: 2>} : i32
      %113 = ttg.convert_layout %111 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %113, %1 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %114 = ttng.async_tma_copy_local_to_global %41[%112] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %114   {async_task_id = array<i32: 2>} : !ttg.async.token
      %115 = arith.addi %57, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %116 = arith.truncf %outLHS_5 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %117 = arith.extf %116 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %115] 16384 %15 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %118 = ttg.local_load %15 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %115] 16384 %16 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %119 = ttg.local_load %16 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %120 = arith.mulf %117, %118 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %121 = arith.mulf %120, %119 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %115] 16384 %17 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %122 = ttg.local_load %17 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%115] 256 %18 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %123 = ttg.local_load %18 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %124 = ttg.convert_layout %123 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %125 = tt.expand_dims %124 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %126 = tt.broadcast %125 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %127 = arith.mulf %121, %126 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %128 = arith.mulf %122, %127 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %129 = "tt.reduce"(%128) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %130 = arith.addf %108, %129 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %131 = arith.mulf %121, %122 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %132 = arith.select %86, %131, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %133 = "tt.reduce"(%132) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %134 = arith.addi %89, %115 {async_task_id = array<i32: 2>} : i32
      %135 = ttg.convert_layout %133 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %135, %2 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %136 = ttng.async_tma_copy_local_to_global %41[%134] %2 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %136   {async_task_id = array<i32: 2>} : !ttg.async.token
      %137 = arith.addi %57, %c192_i32 {async_task_id = array<i32: 2, 3>} : i32
      %138 = arith.truncf %outRHS_6 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
      %139 = arith.extf %138 {async_task_id = array<i32: 0>} : tensor<64x64xbf16, #blocked> to tensor<64x64xf32, #blocked>
      nvws.descriptor_load %35[%56, %137] 16384 %19 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %140 = ttg.local_load %19 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %36[%56, %137] 16384 %20 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %141 = ttg.local_load %20 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %142 = arith.mulf %139, %140 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %143 = arith.mulf %142, %141 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      nvws.descriptor_load %37[%56, %137] 16384 %21 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x64xf32, #shared3>, i32, i32, !ttg.memdesc<64x64xf32, #shared3, #smem, mutable>
      %144 = ttg.local_load %21 {async_task_id = array<i32: 0>} : !ttg.memdesc<64x64xf32, #shared3, #smem, mutable> -> tensor<64x64xf32, #blocked>
      nvws.descriptor_load %38[%137] 256 %10 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64xf32, #shared>, i32, !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %145 = ttg.local_load %10 {async_task_id = array<i32: 0>} : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked3>
      %146 = ttg.convert_layout %145 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %147 = tt.expand_dims %146 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %148 = tt.broadcast %147 {async_task_id = array<i32: 0>} : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %149 = arith.mulf %143, %148 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %150 = arith.mulf %144, %149 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %151 = "tt.reduce"(%150) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %152 = arith.addf %130, %151 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %153 = arith.mulf %143, %144 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #blocked>
      %154 = arith.select %86, %153, %cst {async_task_id = array<i32: 0>} : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %155 = "tt.reduce"(%154) <{axis = 0 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %169 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %169 {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %156 = arith.addi %89, %137 {async_task_id = array<i32: 2>} : i32
      %157 = ttg.convert_layout %155 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %157, %3 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %158 = ttng.async_tma_copy_local_to_global %41[%156] %3 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %158   {async_task_id = array<i32: 2>} : !ttg.async.token
      %159 = ttg.convert_layout %152 {async_task_id = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked3>
      ttg.local_store %159, %4 {async_task_id = array<i32: 0>} : tensor<64xf32, #blocked3> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
      %160 = ttng.async_tma_copy_local_to_global %39[%56] %4 {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared>, !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %160   {async_task_id = array<i32: 2>} : !ttg.async.token
      %isValid, %x, %y, %z = ttng.clc_read %47 {async_task_id = array<i32: 3>} : !ttg.async.token -> i1, i32, i32, i32
      %161 = arith.extui %isValid {async_task_id = array<i32: 3>} : i1 to i32
      %162 = tt.splat %161 {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked3>
      ttg.local_store %162, %24 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked3> -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
      %163 = ttg.local_load %24 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared4, #smem, mutable> -> tensor<1xi32, #blocked3>
      %164 = tt.unsplat %163 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked3>
      %165 = arith.trunci %164 {async_task_id = array<i32: 0, 1, 2, 3>} : i32 to i1
      %166 = tt.splat %x {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked3>
      ttg.local_store %166, %23 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked3> -> !ttg.memdesc<1xi32, #shared4, #smem, mutable>
      %167 = ttg.local_load %23 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared4, #smem, mutable> -> tensor<1xi32, #blocked3>
      %168 = tt.unsplat %167 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked3>
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %165, %168 : i1, i32
    } attributes {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
