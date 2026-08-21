// RUN: triton-opt %s --nvgpu-analyze-2cta-dependencies | FileCheck %s --check-prefix=DEP
// RUN: triton-opt %s --nvgpu-analyze-2cta-dependencies | FileCheck %s --check-prefix=MMA

// Full TTGIR captured immediately before NVGPUAnalyze2CTADependencies from the
// persistent _BWD_DOT_ATTRS_BM64_TMEM 2-CTA configuration.
// DEP-COUNT-2: ttng.two_cta_dependency = "collective_contraction"
// DEP-COUNT-1: ttng.two_cta_dependency = "requires_peer_gather"
// MMA-COUNT-5: ttng.tc_gen5_mma

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [4, 1, 8], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 64], [0, 32], [0, 1], [0, 2], [16, 0]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [[1, 0, 0], [0, 0, 64]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[0, 0, 1], [0, 64, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [0, 32, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [1, 0, 0], [2, 0, 0]], warp = [[4, 0, 0], [8, 0, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 1, 0], [0, 0, 32], [0, 0, 1], [0, 0, 2], [16, 0, 0]], lane = [[0, 0, 4], [0, 0, 8], [0, 0, 16], [1, 0, 0], [2, 0, 0]], warp = [[4, 0, 0], [8, 0, 0]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 0, 1], [0, 32, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [1, 0, 0], [2, 0, 0]], warp = [[4, 0, 0], [8, 0, 0]], block = []}>
#linear9 = #ttg.linear<{register = [[0, 32], [0, 1], [0, 2], [16, 0]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#linear10 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear11 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1, twoCTAs = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true, ctaMode = twocta_rhs>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @_attn_bwd_persist(%arg0: !tt.tensordesc<64x128xf16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<64x128xf16>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf16>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<128x128xf16>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.tensordesc<128x128xf16>, %arg21: i32, %arg22: i32, %arg23: i64, %arg24: i64, %arg25: f32, %arg26: !tt.tensordesc<64x128xf16>, %arg27: i32, %arg28: i32, %arg29: i64, %arg30: i64, %arg31: !tt.tensordesc<64x128xf16>, %arg32: i32, %arg33: i32, %arg34: i64, %arg35: i64, %arg36: !tt.tensordesc<32x32xf32>, %arg37: i32, %arg38: i32, %arg39: i64, %arg40: i64, %arg41: !tt.tensordesc<128x64xf16>, %arg42: i32, %arg43: i32, %arg44: i64, %arg45: i64, %arg46: !tt.tensordesc<128x64xf16>, %arg47: i32, %arg48: i32, %arg49: i64, %arg50: i64, %arg51: !tt.tensordesc<64xf32>, %arg52: i32, %arg53: i64, %arg54: !tt.tensordesc<64xf32>, %arg55: i32, %arg56: i64, %arg57: i32 {tt.divisibility = 16 : i32}, %arg58: i32 {tt.divisibility = 16 : i32}, %arg59: i32 {tt.divisibility = 16 : i32}, %arg60: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #linear>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #linear1>
    %cst_1 = arith.constant dense<0.693147182> : tensor<32x32xf32, #blocked>
    %c96_i32 = arith.constant 96 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear2>
    %0 = arith.addi %arg60, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %2 = tt.get_program_id x : i32
    %3 = arith.divsi %2, %c2_i32 : i32
    %4 = tt.get_num_programs x : i32
    %5 = arith.divsi %4, %c2_i32 : i32
    %6 = arith.divsi %1, %5 : i32
    %7 = arith.remsi %1, %5 : i32
    %8 = arith.cmpi slt, %3, %7 : i32
    %9 = scf.if %8 -> (i32) {
      %20 = arith.addi %6, %c1_i32 : i32
      scf.yield %20 : i32
    } else {
      scf.yield %6 : i32
    }
    %10 = arith.extsi %arg59 : i32 to i64
    %11 = arith.divsi %arg60, %c64_i32 : i32
    %12 = arith.remsi %2, %c2_i32 : i32
    %13 = arith.cmpi eq, %12, %c0_i32 : i32
    %14 = tt.splat %13 : i1 -> tensor<32x128xi1, #linear3>
    %15 = arith.muli %12, %c32_i32 : i32
    %16 = arith.extsi %15 : i32 to i64
    %17 = tt.splat %arg25 : f32 -> tensor<128x64xf32, #linear1>
    %18 = tt.splat %arg25 : f32 -> tensor<128x64xf32, #linear1>
    %19 = scf.for %arg61 = %c0_i32 to %9 step %c1_i32 iter_args(%arg62 = %3) -> (i32)  : i32 {
      %20 = arith.remsi %arg62, %1 : i32
      %21 = arith.divsi %arg62, %1 : i32
      %22 = arith.muli %21, %arg60 : i32
      %23 = arith.extsi %22 : i32 to i64
      %24 = arith.muli %arg57, %21 : i32
      %25 = arith.extsi %24 : i32 to i64
      %26 = arith.divsi %25, %10 : i64
      %27 = arith.muli %20, %c128_i32 : i32
      %28 = arith.extsi %27 : i32 to i64
      %29 = arith.addi %26, %28 : i64
      %30 = arith.trunci %29 : i64 to i32
      %31 = tt.descriptor_load %arg10[%30, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked1>
      %32 = ttg.local_alloc %31 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %33 = tt.descriptor_load %arg15[%30, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked1>
      %34 = ttg.local_alloc %33 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %35 = tt.descriptor_load %arg20[%30, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked1>
      %36 = ttg.local_alloc %35 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %37:3 = scf.for %arg63 = %c0_i32 to %11 step %c1_i32 iter_args(%arg64 = %cst_2, %arg65 = %cst_2, %arg66 = %c0_i32) -> (tensor<128x128xf32, #linear2>, tensor<128x128xf32, #linear2>, i32)  : i32 {
        %53 = arith.extsi %arg66 : i32 to i64
        %54 = arith.addi %26, %53 : i64
        %55 = arith.trunci %54 : i64 to i32
        %56 = tt.descriptor_load %arg0[%55, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked1>
        %57 = ttg.local_alloc %56 : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %58 = tt.descriptor_load %arg5[%55, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked1>
        %59 = ttg.local_alloc %58 : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %60 = ttg.memdesc_trans %59 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared1, #smem>
        %61 = arith.addi %23, %53 : i64
        %62 = arith.trunci %61 : i64 to i32
        %63 = tt.descriptor_load %arg51[%62] : !tt.tensordesc<64xf32> -> tensor<64xf32, #blocked2>
        %result, %token = ttng.tmem_alloc %cst_0 : (tensor<128x64xf32, #linear1>) -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %64 = ttng.tc_gen5_mma %32, %60, %result[%token], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared1, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %result_5, %token_6 = ttng.tmem_load %result[%64] : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear1>
        %65 = ttg.convert_layout %63 : tensor<64xf32, #blocked2> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %66 = tt.expand_dims %65 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1>
        %67 = tt.broadcast %66 : tensor<1x64xf32, #linear1> -> tensor<128x64xf32, #linear1>
        %68 = arith.subf %result_5, %67 : tensor<128x64xf32, #linear1>
        %69 = math.exp2 %68 : tensor<128x64xf32, #linear1>
        %70 = tt.descriptor_load %arg26[%55, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked1>
        %71 = ttg.local_alloc %70 : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %72 = tt.descriptor_load %arg31[%55, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked1>
        %73 = arith.truncf %69 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
        %74 = ttg.local_alloc %73 : (tensor<128x64xf16, #linear1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %75 = ttg.local_alloc %72 : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %76 = ttg.memdesc_trans %75 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared1, #smem>
        %result_7, %token_8 = ttng.tmem_alloc %cst_0 : (tensor<128x64xf32, #linear1>) -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %77 = ttng.tc_gen5_mma %36, %76, %result_7[%token_8], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared1, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %result_9, %token_10 = ttng.tmem_load %result_7[%77] : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear1>
        %78 = tt.descriptor_load %arg54[%62] : !tt.tensordesc<64xf32> -> tensor<64xf32, #blocked2>
        %result_11, %token_12 = ttng.tmem_alloc %arg65 : (tensor<128x128xf32, #linear2>) -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %79 = ttng.tc_gen5_mma %74, %71, %result_11[%token_12], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", two_ctas} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_13, %token_14 = ttng.tmem_load %result_11[%79] : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear2>
        %80 = ttg.convert_layout %78 : tensor<64xf32, #blocked2> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %81 = tt.expand_dims %80 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1>
        %82 = tt.broadcast %81 : tensor<1x64xf32, #linear1> -> tensor<128x64xf32, #linear1>
        %83 = arith.subf %result_9, %82 : tensor<128x64xf32, #linear1>
        %84 = arith.mulf %69, %83 : tensor<128x64xf32, #linear1>
        %85 = arith.truncf %84 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
        %86 = ttg.local_alloc %85 : (tensor<128x64xf16, #linear1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %result_15, %token_16 = ttng.tmem_alloc %arg64 : (tensor<128x128xf32, #linear2>) -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %87 = ttng.tc_gen5_mma %86, %57, %result_15[%token_16], %true, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,tmem,1,5\22, \22opndD,tmem,1,10\22]}", two_ctas} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_17, %token_18 = ttng.tmem_load %result_15[%87] : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear2>
        %88 = ttg.local_alloc %85 : (tensor<128x64xf16, #linear1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %89 = ttg.memdesc_trans %88 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %result_19, %token_20 = ttng.tmem_alloc %cst : (tensor<64x128xf32, #linear>) -> (!ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %90 = ttng.tc_gen5_mma %89, %34, %result_19[%token_20], %true, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,11\22]}", two_ctas} : !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
        %result_21, %token_22 = ttng.tmem_load %result_19[%90] : !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
        %91 = tt.reshape %result_21 : tensor<64x128xf32, #linear> -> tensor<2x32x128xf32, #linear4>
        %92 = tt.trans %91 {order = array<i32: 1, 2, 0>} : tensor<2x32x128xf32, #linear4> -> tensor<32x128x2xf32, #linear5>
        %93 = ttg.convert_layout %92 : tensor<32x128x2xf32, #linear5> -> tensor<32x128x2xf32, #linear6>
        %outLHS_23, %outRHS_24 = tt.split %93 : tensor<32x128x2xf32, #linear6> -> tensor<32x128xf32, #linear3>
        %94 = arith.select %14, %outLHS_23, %outRHS_24 : tensor<32x128xi1, #linear3>, tensor<32x128xf32, #linear3>
        %95 = tt.reshape %94 : tensor<32x128xf32, #linear3> -> tensor<32x2x64xf32, #linear7>
        %96 = tt.trans %95 {order = array<i32: 0, 2, 1>} : tensor<32x2x64xf32, #linear7> -> tensor<32x64x2xf32, #linear8>
        %outLHS_25, %outRHS_26 = tt.split %96 : tensor<32x64x2xf32, #linear8> -> tensor<32x64xf32, #linear9>
        %97 = tt.reshape %outLHS_25 : tensor<32x64xf32, #linear9> -> tensor<32x2x32xf32, #blocked3>
        %98 = tt.trans %97 {order = array<i32: 0, 2, 1>} : tensor<32x2x32xf32, #blocked3> -> tensor<32x32x2xf32, #blocked4>
        %outLHS_27, %outRHS_28 = tt.split %98 : tensor<32x32x2xf32, #blocked4> -> tensor<32x32xf32, #blocked>
        %99 = tt.reshape %outRHS_26 : tensor<32x64xf32, #linear9> -> tensor<32x2x32xf32, #blocked3>
        %100 = tt.trans %99 {order = array<i32: 0, 2, 1>} : tensor<32x2x32xf32, #blocked3> -> tensor<32x32x2xf32, #blocked4>
        %outLHS_29, %outRHS_30 = tt.split %100 : tensor<32x32x2xf32, #blocked4> -> tensor<32x32xf32, #blocked>
        %101 = arith.addi %54, %16 : i64
        %102 = arith.mulf %outLHS_27, %cst_1 : tensor<32x32xf32, #blocked>
        %103 = arith.trunci %101 : i64 to i32
        tt.descriptor_reduce add, %arg36[%103, %c0_i32], %102 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked>
        %104 = arith.mulf %outRHS_28, %cst_1 : tensor<32x32xf32, #blocked>
        tt.descriptor_reduce add, %arg36[%103, %c32_i32], %104 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked>
        %105 = arith.mulf %outLHS_29, %cst_1 : tensor<32x32xf32, #blocked>
        tt.descriptor_reduce add, %arg36[%103, %c64_i32], %105 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked>
        %106 = arith.mulf %outRHS_30, %cst_1 : tensor<32x32xf32, #blocked>
        tt.descriptor_reduce add, %arg36[%103, %c96_i32], %106 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked>
        %107 = arith.addi %arg66, %c64_i32 : i32
        scf.yield %result_17, %result_13, %107 : tensor<128x128xf32, #linear2>, tensor<128x128xf32, #linear2>, i32
      }
      %38 = tt.reshape %37#1 : tensor<128x128xf32, #linear2> -> tensor<128x2x64xf32, #linear10>
      %39 = tt.trans %38 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear10> -> tensor<128x64x2xf32, #linear11>
      %outLHS, %outRHS = tt.split %39 : tensor<128x64x2xf32, #linear11> -> tensor<128x64xf32, #linear1>
      %40 = arith.truncf %outLHS : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %41 = ttg.convert_layout %40 : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked5>
      tt.descriptor_store %arg46[%30, %c0_i32], %41 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked5>
      %42 = arith.truncf %outRHS : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %43 = ttg.convert_layout %42 : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked5>
      tt.descriptor_store %arg46[%30, %c64_i32], %43 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked5>
      %44 = tt.reshape %37#0 : tensor<128x128xf32, #linear2> -> tensor<128x2x64xf32, #linear10>
      %45 = tt.trans %44 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear10> -> tensor<128x64x2xf32, #linear11>
      %outLHS_3, %outRHS_4 = tt.split %45 : tensor<128x64x2xf32, #linear11> -> tensor<128x64xf32, #linear1>
      %46 = arith.mulf %outLHS_3, %18 : tensor<128x64xf32, #linear1>
      %47 = arith.truncf %46 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %48 = ttg.convert_layout %47 : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked5>
      tt.descriptor_store %arg41[%30, %c0_i32], %48 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked5>
      %49 = arith.mulf %outRHS_4, %17 : tensor<128x64xf32, #linear1>
      %50 = arith.truncf %49 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %51 = ttg.convert_layout %50 : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked5>
      tt.descriptor_store %arg41[%30, %c64_i32], %51 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked5>
      %52 = arith.addi %arg62, %5 : i32
      scf.yield %52 : i32
    } {tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
