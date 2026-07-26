// RUN: triton-opt %s --tritongpu-accelerate-matmul | FileCheck %s --check-prefix=MMA
// RUN: triton-opt %s --tritongpu-accelerate-matmul | FileCheck %s --check-prefix=RHS

// Full TTGIR captured immediately before TritonGPUAccelerateMatmul from the
// persistent _BWD_DOT_ATTRS_BM64_TMEM 2-CTA configuration.
// RHS: #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true, ctaMode = twocta_rhs>
// MMA-COUNT-5: ttng.tc_gen5_mma {{.*}}two_ctas

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 4, 4], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked10 = #ttg.blocked<{sizePerThread = [4, 4, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [4, 1, 8], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked12 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked13 = #ttg.blocked<{sizePerThread = [4, 1, 4], threadsPerWarp = [1, 2, 16], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked14 = #ttg.blocked<{sizePerThread = [4, 4, 1], threadsPerWarp = [1, 16, 2], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked15 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [1, 0], [2, 0], [16, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], warp = [[4, 0], [8, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [1, 0, 0], [2, 0, 0], [16, 0, 0]], lane = [[0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], warp = [[4, 0, 0], [8, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [16, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], warp = [[4, 0, 0], [8, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 32, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [1, 0, 0], [2, 0, 0]], warp = [[4, 0, 0], [8, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 32], [0, 1], [0, 2], [16, 0]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @_attn_bwd_persist(%arg0: !tt.tensordesc<64x128xf16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<64x128xf16>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf16>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<128x128xf16>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.tensordesc<128x128xf16>, %arg21: i32, %arg22: i32, %arg23: i64, %arg24: i64, %arg25: f32, %arg26: !tt.tensordesc<64x128xf16>, %arg27: i32, %arg28: i32, %arg29: i64, %arg30: i64, %arg31: !tt.tensordesc<64x128xf16>, %arg32: i32, %arg33: i32, %arg34: i64, %arg35: i64, %arg36: !tt.tensordesc<32x32xf32>, %arg37: i32, %arg38: i32, %arg39: i64, %arg40: i64, %arg41: !tt.tensordesc<128x64xf16>, %arg42: i32, %arg43: i32, %arg44: i64, %arg45: i64, %arg46: !tt.tensordesc<128x64xf16>, %arg47: i32, %arg48: i32, %arg49: i64, %arg50: i64, %arg51: !tt.tensordesc<64xf32>, %arg52: i32, %arg53: i64, %arg54: !tt.tensordesc<64xf32>, %arg55: i32, %arg56: i64, %arg57: i32 {tt.divisibility = 16 : i32}, %arg58: i32 {tt.divisibility = 16 : i32}, %arg59: i32 {tt.divisibility = 16 : i32}, %arg60: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked1>
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c96_i32 = arith.constant 96 : i32
    %cst_2 = arith.constant dense<0.693147182> : tensor<32x32xf32, #blocked2>
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
      %19 = arith.addi %6, %c1_i32 : i32
      scf.yield %19 : i32
    } else {
      scf.yield %6 : i32
    }
    %10 = arith.extsi %arg59 : i32 to i64
    %11 = arith.divsi %arg60, %c64_i32 : i32
    %12 = arith.remsi %2, %c2_i32 : i32
    %13 = arith.cmpi eq, %12, %c0_i32 : i32
    %14 = tt.splat %13 : i1 -> tensor<32x128xi1, #linear>
    %15 = arith.muli %12, %c32_i32 : i32
    %16 = arith.extsi %15 : i32 to i64
    %17 = tt.splat %arg25 : f32 -> tensor<128x64xf32, #blocked3>
    %18 = scf.for %arg61 = %c0_i32 to %9 step %c1_i32 iter_args(%arg62 = %3) -> (i32)  : i32 {
      %19 = arith.remsi %arg62, %1 : i32
      %20 = arith.divsi %arg62, %1 : i32
      %21 = arith.muli %20, %arg60 : i32
      %22 = arith.extsi %21 : i32 to i64
      %23 = arith.muli %arg57, %20 : i32
      %24 = arith.extsi %23 : i32 to i64
      %25 = arith.divsi %24, %10 : i64
      %26 = arith.muli %19, %c128_i32 : i32
      %27 = arith.extsi %26 : i32 to i64
      %28 = arith.addi %25, %27 : i64
      %29 = arith.trunci %28 : i64 to i32
      %30 = tt.descriptor_load %arg10[%29, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked4>
      %31 = tt.descriptor_load %arg15[%29, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked4>
      %32 = tt.descriptor_load %arg20[%29, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked4>
      %33:3 = scf.for %arg63 = %c0_i32 to %11 step %c1_i32 iter_args(%arg64 = %cst, %arg65 = %cst, %arg66 = %c0_i32) -> (tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>, i32)  : i32 {
        %47 = arith.extsi %arg66 : i32 to i64
        %48 = arith.addi %25, %47 : i64
        %49 = arith.trunci %48 : i64 to i32
        %50 = tt.descriptor_load %arg0[%49, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked4>
        %51 = tt.descriptor_load %arg5[%49, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked4>
        %52 = tt.trans %51 {order = array<i32: 1, 0>} : tensor<64x128xf16, #blocked4> -> tensor<128x64xf16, #blocked5>
        %53 = arith.addi %22, %47 : i64
        %54 = arith.trunci %53 : i64 to i32
        %55 = tt.descriptor_load %arg51[%54] : !tt.tensordesc<64xf32> -> tensor<64xf32, #blocked6>
        %56 = ttg.convert_layout %30 : tensor<128x128xf16, #blocked4> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
        %57 = ttg.convert_layout %52 : tensor<128x64xf16, #blocked5> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
        %58 = tt.dot %56, %57, %cst_1, inputPrecision = tf32 {tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf32, #blocked1>
        %59 = ttg.convert_layout %55 : tensor<64xf32, #blocked6> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked7}>>
        %60 = tt.expand_dims %59 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked7}>> -> tensor<1x64xf32, #blocked7>
        %61 = ttg.convert_layout %60 : tensor<1x64xf32, #blocked7> -> tensor<1x64xf32, #blocked1>
        %62 = tt.broadcast %61 : tensor<1x64xf32, #blocked1> -> tensor<128x64xf32, #blocked1>
        %63 = arith.subf %58, %62 : tensor<128x64xf32, #blocked1>
        %64 = math.exp2 %63 : tensor<128x64xf32, #blocked1>
        %65 = tt.descriptor_load %arg26[%49, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked4>
        %66 = tt.descriptor_load %arg31[%49, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked4>
        %67 = arith.truncf %64 : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
        %68 = tt.trans %66 {order = array<i32: 1, 0>} : tensor<64x128xf16, #blocked4> -> tensor<128x64xf16, #blocked5>
        %69 = ttg.convert_layout %32 : tensor<128x128xf16, #blocked4> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
        %70 = ttg.convert_layout %68 : tensor<128x64xf16, #blocked5> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
        %71 = tt.dot %69, %70, %cst_1, inputPrecision = tf32 {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf32, #blocked1>
        %72 = tt.descriptor_load %arg54[%54] : !tt.tensordesc<64xf32> -> tensor<64xf32, #blocked6>
        %73 = ttg.convert_layout %67 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
        %74 = ttg.convert_layout %65 : tensor<64x128xf16, #blocked4> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
        %75 = tt.dot %73, %74, %arg65, inputPrecision = tf32 {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", two_ctas} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
        %76 = ttg.convert_layout %72 : tensor<64xf32, #blocked6> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked7}>>
        %77 = tt.expand_dims %76 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked7}>> -> tensor<1x64xf32, #blocked7>
        %78 = ttg.convert_layout %77 : tensor<1x64xf32, #blocked7> -> tensor<1x64xf32, #blocked1>
        %79 = tt.broadcast %78 : tensor<1x64xf32, #blocked1> -> tensor<128x64xf32, #blocked1>
        %80 = arith.subf %71, %79 : tensor<128x64xf32, #blocked1>
        %81 = arith.mulf %64, %80 : tensor<128x64xf32, #blocked1>
        %82 = arith.truncf %81 : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
        %83 = ttg.convert_layout %82 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
        %84 = ttg.convert_layout %50 : tensor<64x128xf16, #blocked4> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
        %85 = tt.dot %83, %84, %arg64, inputPrecision = tf32 {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,tmem,1,5\22, \22opndD,tmem,1,10\22]}", two_ctas} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
        %86 = tt.trans %82 {order = array<i32: 1, 0>} : tensor<128x64xf16, #blocked1> -> tensor<64x128xf16, #blocked8>
        %87 = ttg.convert_layout %86 : tensor<64x128xf16, #blocked8> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
        %88 = ttg.convert_layout %31 : tensor<128x128xf16, #blocked4> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
        %89 = tt.dot %87, %88, %cst_0, inputPrecision = tf32 {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,11\22]}", two_ctas} : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x128xf32, #blocked>
        %90 = tt.reshape %89 : tensor<64x128xf32, #blocked> -> tensor<2x32x128xf32, #blocked9>
        %91 = tt.trans %90 {order = array<i32: 1, 2, 0>} : tensor<2x32x128xf32, #blocked9> -> tensor<32x128x2xf32, #blocked10>
        %outLHS_5, %outRHS_6 = tt.split %91 : tensor<32x128x2xf32, #blocked10> -> tensor<32x128xf32, #linear>
        %92 = arith.select %14, %outLHS_5, %outRHS_6 : tensor<32x128xi1, #linear>, tensor<32x128xf32, #linear>
        %93 = tt.reshape %92 : tensor<32x128xf32, #linear> -> tensor<32x2x64xf32, #linear1>
        %94 = tt.trans %93 {order = array<i32: 0, 2, 1>} : tensor<32x2x64xf32, #linear1> -> tensor<32x64x2xf32, #linear2>
        %95 = ttg.convert_layout %94 : tensor<32x64x2xf32, #linear2> -> tensor<32x64x2xf32, #linear3>
        %outLHS_7, %outRHS_8 = tt.split %95 : tensor<32x64x2xf32, #linear3> -> tensor<32x64xf32, #linear4>
        %96 = tt.reshape %outLHS_7 : tensor<32x64xf32, #linear4> -> tensor<32x2x32xf32, #blocked11>
        %97 = tt.trans %96 {order = array<i32: 0, 2, 1>} : tensor<32x2x32xf32, #blocked11> -> tensor<32x32x2xf32, #blocked12>
        %outLHS_9, %outRHS_10 = tt.split %97 : tensor<32x32x2xf32, #blocked12> -> tensor<32x32xf32, #blocked2>
        %98 = tt.reshape %outRHS_8 : tensor<32x64xf32, #linear4> -> tensor<32x2x32xf32, #blocked11>
        %99 = tt.trans %98 {order = array<i32: 0, 2, 1>} : tensor<32x2x32xf32, #blocked11> -> tensor<32x32x2xf32, #blocked12>
        %outLHS_11, %outRHS_12 = tt.split %99 : tensor<32x32x2xf32, #blocked12> -> tensor<32x32xf32, #blocked2>
        %100 = arith.addi %48, %16 : i64
        %101 = arith.mulf %outLHS_9, %cst_2 : tensor<32x32xf32, #blocked2>
        %102 = arith.trunci %100 : i64 to i32
        tt.descriptor_reduce add, %arg36[%102, %c0_i32], %101 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked2>
        %103 = arith.mulf %outRHS_10, %cst_2 : tensor<32x32xf32, #blocked2>
        tt.descriptor_reduce add, %arg36[%102, %c32_i32], %103 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked2>
        %104 = arith.mulf %outLHS_11, %cst_2 : tensor<32x32xf32, #blocked2>
        tt.descriptor_reduce add, %arg36[%102, %c64_i32], %104 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked2>
        %105 = arith.mulf %outRHS_12, %cst_2 : tensor<32x32xf32, #blocked2>
        tt.descriptor_reduce add, %arg36[%102, %c96_i32], %105 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked2>
        %106 = arith.addi %arg66, %c64_i32 : i32
        scf.yield %85, %75, %106 : tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>, i32
      }
      %34 = tt.reshape %33#1 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked13>
      %35 = tt.trans %34 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked13> -> tensor<128x64x2xf32, #blocked14>
      %36 = ttg.convert_layout %35 : tensor<128x64x2xf32, #blocked14> -> tensor<128x64x2xf32, #blocked15>
      %outLHS, %outRHS = tt.split %36 : tensor<128x64x2xf32, #blocked15> -> tensor<128x64xf32, #blocked3>
      %37 = arith.truncf %outLHS : tensor<128x64xf32, #blocked3> to tensor<128x64xf16, #blocked3>
      tt.descriptor_store %arg46[%29, %c0_i32], %37 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked3>
      %38 = arith.truncf %outRHS : tensor<128x64xf32, #blocked3> to tensor<128x64xf16, #blocked3>
      tt.descriptor_store %arg46[%29, %c64_i32], %38 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked3>
      %39 = tt.reshape %33#0 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked13>
      %40 = tt.trans %39 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked13> -> tensor<128x64x2xf32, #blocked14>
      %41 = ttg.convert_layout %40 : tensor<128x64x2xf32, #blocked14> -> tensor<128x64x2xf32, #blocked15>
      %outLHS_3, %outRHS_4 = tt.split %41 : tensor<128x64x2xf32, #blocked15> -> tensor<128x64xf32, #blocked3>
      %42 = arith.mulf %outLHS_3, %17 : tensor<128x64xf32, #blocked3>
      %43 = arith.truncf %42 : tensor<128x64xf32, #blocked3> to tensor<128x64xf16, #blocked3>
      tt.descriptor_store %arg41[%29, %c0_i32], %43 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked3>
      %44 = arith.mulf %outRHS_4, %17 : tensor<128x64xf32, #blocked3>
      %45 = arith.truncf %44 : tensor<128x64xf32, #blocked3> to tensor<128x64xf16, #blocked3>
      tt.descriptor_store %arg41[%29, %c64_i32], %45 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked3>
      %46 = arith.addi %arg62, %5 : i32
      scf.yield %46 : i32
    } {tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
