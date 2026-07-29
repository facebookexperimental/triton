// RUN: triton-opt %s -split-input-file --nvgpu-ws-data-partition=num-warp-groups=2 | FileCheck %s

// HSTU self-attention forward: the PV (output) MMA is a loop-carried in-place
// TMEM accumulator that is read only AFTER the loop. Data partitioning (DP=2)
// must split its full [256,128] accumulator into two per-partition [128,128]
// tiles, each with its OWN loop-carried token — mirroring the QK accumulator.
//
// Regression: DP used to slice the MMAs but leave the original full [256,128]
// accumulator's zero-init store (and hence its loop-carried token) un-sliced.
// The split MMAs then shared the full accumulator's token, keeping the full
// (dead-data) [256,128] TMEM tile live via the token cycle -> 2x TMEM -> OOM.
// After the fix no full-tile TMEM accumulator may survive.

// CHECK-LABEL: @_hstu_attn_fwd
// A per-partition [128,128] accumulator must exist (DP actually fired) ...
// CHECK: ttng.tmem_alloc {{.*}}128x128xf32{{.*}}#ttng.tensor_memory
// ... and NO full-tile [256,x] TMEM accumulator may survive the split.
// CHECK-NOT: 256x128xf32, {{.+}}#ttng.tensor_memory
// CHECK-NOT: 256x128xbf16, {{.+}}#ttng.tensor_memory

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_hstu_attn_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #linear>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<256x128xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c383_i32 = arith.constant 383 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant dense<0> : tensor<256x128xi32, #linear>
    %0 = tt.get_program_id y : i32
    %1 = arith.divsi %0, %arg17 : i32
    %2 = arith.remsi %0, %arg17 : i32
    %3 = tt.get_program_id x : i32
    %4 = arith.extsi %2 : i32 to i64
    %5 = arith.extsi %1 : i32 to i64
    %6 = tt.addptr %arg3, %5 : !tt.ptr<i64>, i64
    %7 = tt.load %6 : !tt.ptr<i64>
    %8 = tt.addptr %6, %c1_i32 : !tt.ptr<i64>, i32
    %9 = tt.load %8 : !tt.ptr<i64>
    %10 = tt.addptr %arg4, %5 : !tt.ptr<i64>, i64
    %11 = tt.load %10 : !tt.ptr<i64>
    %12 = tt.addptr %10, %c1_i32 : !tt.ptr<i64>, i32
    %13 = tt.load %12 : !tt.ptr<i64>
    %14 = arith.subi %13, %11 : i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = arith.trunci %9 : i64 to i32
    %17 = arith.muli %arg17, %arg20 : i32
    %18 = arith.extsi %17 : i32 to i64
    %19 = tt.make_tensor_descriptor %arg1, [%16, %17], [%18, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %20 = arith.muli %arg17, %arg21 : i32
    %21 = arith.extsi %20 : i32 to i64
    %22 = tt.make_tensor_descriptor %arg2, [%16, %20], [%21, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %23 = arith.muli %3, %c256_i32 : i32
    %24 = arith.cmpi slt, %23, %15 : i32
    scf.if %24 {
      %25 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %26 = tt.splat %23 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %27 = arith.addi %26, %25 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %28 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
      %29 = tt.load %arg18 : !tt.ptr<f32>
      %30 = arith.trunci %13 : i64 to i32
      %31 = tt.make_tensor_descriptor %arg0, [%30, %17], [%18, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<256x128xbf16, #shared>
      %32 = arith.extsi %23 : i32 to i64
      %33 = arith.addi %11, %32 : i64
      %34 = arith.trunci %33 : i64 to i32
      %35 = arith.extsi %arg8 : i32 to i64
      %36 = arith.muli %4, %35 : i64
      %37 = arith.trunci %36 : i64 to i32
      %38 = tt.descriptor_load %31[%34, %37] : !tt.tensordesc<256x128xbf16, #shared> -> tensor<256x128xbf16, #blocked>
      %39 = ttg.local_alloc %38 : (tensor<256x128xbf16, #blocked>) -> !ttg.memdesc<256x128xbf16, #shared, #smem>
      %40 = arith.addi %23, %c256_i32 : i32
      %41 = arith.addi %23, %c383_i32 : i32
      %42 = arith.divsi %41, %c128_i32 : i32
      %43 = arith.extsi %arg10 : i32 to i64
      %44 = arith.muli %4, %43 : i64
      %45 = arith.extsi %arg12 : i32 to i64
      %46 = arith.muli %4, %45 : i64
      %47 = arith.trunci %44 : i64 to i32
      %48 = tt.expand_dims %27 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<256x1xi32, #linear>
      %49 = tt.broadcast %48 : tensor<256x1xi32, #linear> -> tensor<256x128xi32, #linear>
      %50 = tt.splat %arg6 : f32 -> tensor<256x128xf32, #linear>
      %51 = tt.splat %29 : f32 -> tensor<256x128xf32, #linear>
      %52 = arith.trunci %46 : i64 to i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_2, %token_3 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %53 = ttng.tmem_store %cst, %result_2[%token_3], %true : tensor<256x128xf32, #linear> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %54:3 = scf.for %arg24 = %c0_i32 to %42 step %c1_i32 iter_args(%arg25 = %false, %arg26 = %token, %arg27 = %53) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
        %61 = arith.cmpi sge, %arg24, %42 : i32
        %62 = arith.subi %arg24, %42 : i32
        %63 = arith.select %61, %62, %arg24 : i32
        %64 = arith.muli %63, %c128_i32 : i32
        %65 = arith.addi %40, %64 : i32
        %66 = arith.select %61, %65, %64 : i32
        %67 = tt.splat %66 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %68 = arith.addi %28, %67 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %69 = arith.extsi %66 : i32 to i64
        %70 = arith.addi %7, %69 : i64
        %71 = arith.trunci %70 : i64 to i32
        %72 = tt.descriptor_load %19[%71, %47] : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
        %73 = ttg.local_alloc %72 : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %74 = ttg.memdesc_trans %73 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
        %75 = ttng.tc_gen5_mma %39, %74, %result[%arg26], %false, %true : !ttg.memdesc<256x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %76 = tt.expand_dims %68 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xi32, #linear>
        %77 = tt.broadcast %76 : tensor<1x128xi32, #linear> -> tensor<256x128xi32, #linear>
        %78 = arith.cmpi eq, %49, %77 : tensor<256x128xi32, #linear>
        %79 = arith.subi %49, %77 : tensor<256x128xi32, #linear>
        %80 = arith.cmpi sgt, %79, %cst_1 : tensor<256x128xi32, #linear>
        %81 = arith.ori %78, %80 : tensor<256x128xi1, #linear>
        %result_6, %token_7 = ttng.tmem_load %result[%75] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear>
        %82 = arith.mulf %result_6, %50 : tensor<256x128xf32, #linear>
        %83 = arith.negf %82 : tensor<256x128xf32, #linear>
        %84 = math.exp %83 : tensor<256x128xf32, #linear>
        %85 = arith.addf %84, %cst_0 : tensor<256x128xf32, #linear>
        %86 = tt.extern_elementwise %82, %85 {libname = "", libpath = "", pure = true, symbol = "__nv_fast_fdividef"} : (tensor<256x128xf32, #linear>, tensor<256x128xf32, #linear>) -> tensor<256x128xf32, #linear>
        %87 = arith.mulf %86, %51 : tensor<256x128xf32, #linear>
        %88 = arith.select %81, %87, %cst : tensor<256x128xi1, #linear>, tensor<256x128xf32, #linear>
        %89 = tt.descriptor_load %22[%71, %52] : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
        %90 = ttg.local_alloc %89 : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %91 = arith.truncf %88 : tensor<256x128xf32, #linear> to tensor<256x128xbf16, #linear>
        %result_8 = ttng.tmem_alloc %91 : (tensor<256x128xbf16, #linear>) -> !ttg.memdesc<256x128xbf16, #tmem, #ttng.tensor_memory>
        %92 = ttng.tc_gen5_mma %result_8, %90, %result_2[%arg27], %arg25, %true : !ttg.memdesc<256x128xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %token_7, %92 : i1, !ttg.async.token, !ttg.async.token
      } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}
      %result_4, %token_5 = ttng.tmem_load %result_2[%54#2] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear>
      %55 = arith.truncf %result_4 : tensor<256x128xf32, #linear> to tensor<256x128xbf16, #linear>
      %56 = tt.make_tensor_descriptor %arg5, [%30, %20], [%21, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<256x128xbf16, #shared>
      %57 = arith.extsi %arg14 : i32 to i64
      %58 = arith.muli %4, %57 : i64
      %59 = arith.trunci %58 : i64 to i32
      %60 = ttg.convert_layout %55 : tensor<256x128xbf16, #linear> -> tensor<256x128xbf16, #blocked>
      tt.descriptor_store %56[%34, %59], %60 : !tt.tensordesc<256x128xbf16, #shared>, tensor<256x128xbf16, #blocked>
    }
    tt.return
  }
}
