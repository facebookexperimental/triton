// RUN: env TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448 tma-store-pipelining=true" 2>/dev/null | FileCheck %s
// RUN: env TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448 tma-store-pipelining=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN

// DP=2 addmm whose one bias load feeds both halves' accumulator init: no WS.

// WARN: warning: meta autoWS does not support a register value used by consumers in different partitions
// CHECK-LABEL: @gemm
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: async_task_id
// CHECK: ttng.tc_gen5_mma
// CHECK: ttng.tc_gen5_mma

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @gemm(%arg0: !tt.tensordesc<128x64xbf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<128x64xbf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf32, #shared1>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked>
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg15, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.addi %arg16, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.muli %2, %4 : i32
    %6 = arith.muli %4, %c8_i32 : i32
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %8 = tt.splat %arg16 : i32 -> tensor<128xi32, #blocked>
    %9 = tt.splat %arg17 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_0, %token_1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %10:2 = scf.for %arg18 = %0 to %5 step %c148_i32 iter_args(%arg19 = %token, %arg20 = %token_1) -> (!ttg.async.token, !ttg.async.token)  : i32 {
      %11 = arith.divsi %arg18, %6 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %12 = arith.divsi %arg18, %6 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %13 = arith.divsi %arg18, %6 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %14 = arith.divsi %arg18, %6 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      %15 = arith.divsi %arg18, %6 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %16 = arith.divsi %arg18, %6 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      %17 = arith.divsi %arg18, %6 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %18 = arith.divsi %arg18, %6 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %19 = arith.muli %12, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %20 = arith.muli %11, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %21 = arith.muli %15, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %22 = arith.muli %16, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      %23 = arith.muli %13, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %24 = arith.muli %14, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      %25 = arith.muli %17, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %26 = arith.muli %18, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %27 = arith.subi %2, %20 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %28 = arith.subi %2, %23 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %29 = arith.subi %2, %24 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      %30 = arith.subi %2, %25 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %31 = arith.subi %2, %26 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %32 = arith.minsi %27, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %33 = arith.minsi %28, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %34 = arith.minsi %29, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      %35 = arith.minsi %30, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %36 = arith.minsi %31, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %37 = arith.remsi %arg18, %32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %38 = arith.remsi %arg18, %33 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %39 = arith.remsi %arg18, %34 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      %40 = arith.remsi %arg18, %36 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %41 = arith.addi %19, %37 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %42 = arith.addi %21, %38 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %43 = arith.addi %22, %39 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      %44 = arith.addi %26, %40 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %45 = arith.remsi %arg18, %6 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %46 = arith.remsi %arg18, %6 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %47 = arith.divsi %45, %35 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %48 = arith.divsi %46, %36 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %49 = arith.muli %41, %c256_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %50 = arith.muli %42, %c256_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %51 = arith.muli %43, %c256_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      %52 = arith.muli %44, %c256_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %53 = arith.addi %52, %c128_i32 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : i32
      %54 = arith.addi %50, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %55 = arith.muli %47, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %56 = arith.muli %48, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %57 = tt.descriptor_load %arg5[%55, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked1>
      %58 = ttg.local_alloc %57 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
      %59 = ttg.memdesc_trans %58 {loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 3>} : !ttg.memdesc<128x64xbf16, #shared, #smem> -> !ttg.memdesc<64x128xbf16, #shared2, #smem>
      %60 = tt.splat %56 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32 -> tensor<128xi32, #blocked>
      %61 = arith.addi %60, %7 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128xi32, #blocked>
      %62 = arith.cmpi slt, %61, %8 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128xi32, #blocked>
      %63 = tt.addptr %9, %61 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %64 = tt.load %63, %62, %cst {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x!tt.ptr<f32>, #blocked>
      %65 = tt.descriptor_load %arg0[%49, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked1>
      %66 = tt.descriptor_load %arg0[%54, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked1>
      %67 = ttg.convert_layout %64 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
      %68 = ttg.convert_layout %64 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
      %69 = ttg.convert_layout %64 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
      %70 = tt.expand_dims %67 {axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
      %71 = tt.expand_dims %68 {axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
      %72 = ttg.local_alloc %65 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
      %73 = tt.broadcast %70 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 4>} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
      %74 = ttg.convert_layout %73 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %75 = ttng.tmem_store %74, %result[%arg19], %true {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %76 = ttng.tc_gen5_mma %72, %59, %result[%75], %true, %true {loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 3>} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_2, %token_3 = ttng.tmem_load %result[%76] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %77 = ttg.convert_layout %result_2 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %78 = ttg.convert_layout %77 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked2>
      %79 = ttg.local_alloc %78 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf32, #blocked2>) -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
      %80 = ttng.async_tma_copy_local_to_global %arg10[%52, %56] %79 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<128x128xf32, #shared1>, !ttg.memdesc<128x128xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %80   {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.async.token
      %81 = ttg.local_alloc %66 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
      %82 = tt.broadcast %71 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
      %83 = ttg.convert_layout %82 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %84 = ttng.tmem_store %83, %result_0[%arg20], %true {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %85 = ttng.tc_gen5_mma %81, %59, %result_0[%84], %true, %true {loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 3>} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_4, %token_5 = ttng.tmem_load %result_0[%85] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %86 = ttg.convert_layout %result_4 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %87 = ttg.convert_layout %86 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked2>
      %88 = ttg.local_alloc %87 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked2>) -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
      %89 = ttng.async_tma_copy_local_to_global %arg10[%53, %56] %88 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<128x128xf32, #shared1>, !ttg.memdesc<128x128xf32, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %89   {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.async.token
      scf.yield %token_3, %token_5 : !ttg.async.token, !ttg.async.token
    } {tt.data_partition_factor = 2 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["computation", "epilogue", "load", "gemm", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

