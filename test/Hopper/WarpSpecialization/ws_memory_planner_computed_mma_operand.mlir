// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=4 smem-alloc-algo=1 smem-budget=232448 reserve-auxiliary-smem=1" | FileCheck %s

// A computed BF16 operand fed by two FP32 descriptor scratch tiles is cheaper
// to pipeline than retaining the raw source tiles. Reserve the requested four
// copies before growing the output TMA staging ring.
// CHECK-LABEL: @triton_tem_fused__to_copy_mm_mul_sigmoid_0
// CHECK: ttg.local_alloc {buffer.copy = 4 : i32, buffer.id = [[BID:[0-9]+]] : i32} : () -> !ttg.memdesc<64x128xbf16
// CHECK-COUNT-2: ttg.local_alloc {{.*}}buffer.copy = 2 : i32{{.*}}buffer.tmaStaging = 1 : i32{{.*}}!ttg.memdesc<128x64xf32
// CHECK: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = {{[0-9]+}} : i32} : () -> !ttg.memdesc<64x128xbf16
// CHECK-COUNT-2: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = {{[0-9]+}} : i32} : () -> !ttg.memdesc<64x128xf32

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 1, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 0, 1]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_tem_fused__to_copy_mm_mul_sigmoid_0(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c256_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 256 : i64
    %c256_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 256 : i32
    %c18816_i32 = arith.constant {async_task_id = array<i32: 2>} 18816 : i32
    %c939505_i32 = arith.constant {async_task_id = array<i32: 3>} 939505 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 128 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 8 : i32
    %c2_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 2 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 64 : i32
    %c128_i64 = arith.constant {async_task_id = array<i32: 3>} 128 : i64
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c16_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 16 : i32
    %c100_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 100 : i32
    %c6400_i32 = arith.constant {async_task_id = array<i32: 3>} 6400 : i32
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<1.000000e+00> : tensor<64x128xf32, #blocked>
    %cst_0 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %6 = tt.make_tensor_descriptor %arg3, [%c18816_i32, %c256_i32], [%c256_i64, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<f32>, !tt.tensordesc<128x64xf32, #shared1>
    %7 = tt.make_tensor_descriptor %arg1, [%c939505_i32, %c256_i32], [%c256_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x128xf32, #shared1>
    %8 = tt.make_tensor_descriptor %arg2, [%c939505_i32, %c256_i32], [%c256_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<f32>, !tt.tensordesc<64x128xf32, #shared1>
    %9 = tt.get_program_id y {async_task_id = array<i32: 2, 3>} : i32
    %10 = arith.muli %9, %c6400_i32 {async_task_id = array<i32: 3>} : i32
    %11 = tt.make_tensor_descriptor %arg0, [%c939505_i32, %c128_i32], [%c128_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x128xbf16, #shared>
    %12 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %13 = arith.subi %12, %c2_i32 {async_task_id = array<i32: 2>} : i32
    %14 = arith.muli %9, %c128_i32 {async_task_id = array<i32: 2>} : i32
    %15 = scf.for %arg4 = %12 to %c2_i32 step %c2_i32 iter_args(%arg5 = %13) -> (i32)  : i32 {
      %16 = arith.divsi %arg4, %c16_i32 {async_task_id = array<i32: 3>} : i32
      %17 = arith.muli %16, %c8_i32 {async_task_id = array<i32: 3>} : i32
      %18 = arith.subi %c1_i32, %17 {async_task_id = array<i32: 3>} : i32
      %19 = arith.minsi %18, %c8_i32 {async_task_id = array<i32: 3>} : i32
      %20 = arith.remsi %arg4, %19 {async_task_id = array<i32: 3>} : i32
      %21 = arith.addi %17, %20 {async_task_id = array<i32: 3>} : i32
      %22 = arith.remsi %arg4, %c16_i32 {async_task_id = array<i32: 3>} : i32
      %23 = arith.divsi %22, %19 {async_task_id = array<i32: 3>} : i32
      %24 = arith.muli %21, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %25 = arith.muli %23, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %26:2 = scf.for %arg6 = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg7 = %false, %arg8 = %token) -> (i1, !ttg.async.token)  : i32 {
        %45 = arith.muli %arg6, %c64_i32 {async_task_id = array<i32: 3>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
        %46 = arith.addi %10, %45 {async_task_id = array<i32: 3>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %11[%46, %24] 16384 %3 {async_task_id = array<i32: 3>, loop.cluster = 3 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<64x128xbf16, #shared>, i32, i32, !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
        nvws.descriptor_load %7[%46, %25] 32768 %4 {async_task_id = array<i32: 3>, loop.cluster = 3 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<64x128xf32, #shared1>, i32, i32, !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
        %47 = ttg.local_load %4 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<64x128xf32, #shared1, #smem, mutable> -> tensor<64x128xf32, #blocked>
        nvws.descriptor_load %8[%46, %25] 32768 %5 {async_task_id = array<i32: 3>, loop.cluster = 3 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<64x128xf32, #shared1>, i32, i32, !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
        %48 = ttg.local_load %5 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<64x128xf32, #shared1, #smem, mutable> -> tensor<64x128xf32, #blocked>
        %49 = arith.negf %48 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64x128xf32, #blocked>
        %50 = math.exp %49 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64x128xf32, #blocked>
        %51 = arith.addf %50, %cst {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64x128xf32, #blocked>
        %52 = arith.divf %cst, %51 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64x128xf32, #blocked>
        %53 = arith.mulf %47, %52 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64x128xf32, #blocked>
        %54 = arith.truncf %53 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64x128xf32, #blocked> to tensor<64x128xbf16, #blocked>
        ttg.local_store %54, %0 {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64x128xbf16, #blocked> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
        %55 = ttg.memdesc_trans %3 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 3 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared2, #smem, mutable>
        %56 = ttng.tc_gen5_mma %55, %0, %result[%arg8], %arg7, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared2, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1>} %true, %56 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 3 : i32}
      %27 = arith.addi %arg5, %c2_i32 {async_task_id = array<i32: 2>} : i32
      %28 = arith.divsi %27, %c16_i32 {async_task_id = array<i32: 2>} : i32
      %29 = arith.muli %28, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %30 = arith.subi %c1_i32, %29 {async_task_id = array<i32: 2>} : i32
      %31 = arith.minsi %30, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %32 = arith.remsi %27, %31 {async_task_id = array<i32: 2>} : i32
      %33 = arith.addi %29, %32 {async_task_id = array<i32: 2>} : i32
      %34 = arith.remsi %27, %c16_i32 {async_task_id = array<i32: 2>} : i32
      %35 = arith.divsi %34, %31 {async_task_id = array<i32: 2>} : i32
      %36 = arith.muli %33, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %37 = arith.muli %35, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %result_1, %token_2 = ttng.tmem_load %result[%26#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %38 = tt.reshape %result_1 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
      %39 = tt.trans %38 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
      %40 = ttg.convert_layout %39 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #linear2> -> tensor<128x64x2xf32, #blocked1>
      %outLHS, %outRHS = tt.split %40 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #blocked1> -> tensor<128x64xf32, #blocked2>
      %41 = arith.addi %14, %36 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %outLHS, %1 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %42 = ttng.async_tma_copy_local_to_global %6[%41, %37] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %42   {async_task_id = array<i32: 2>} : !ttg.async.token
      %43 = arith.addi %37, %c64_i32 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %outRHS, %2 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #shared1, #smem, mutable>
      %44 = ttng.async_tma_copy_local_to_global %6[%41, %43] %2 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x64xf32, #shared1>, !ttg.memdesc<128x64xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %44   {async_task_id = array<i32: 2>} : !ttg.async.token
      scf.yield {async_task_id = array<i32: 2>} %27 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
