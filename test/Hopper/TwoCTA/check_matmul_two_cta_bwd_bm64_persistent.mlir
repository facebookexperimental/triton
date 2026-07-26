// RUN: triton-opt %s --triton-nvidia-check-matmul-two-cta="allow-dependent-chains=true" | FileCheck %s

// Full TTGIR captured immediately before TritonNvidiaGPUCheckMatmulTwoCTAPass
// from persistent _BWD_DOT_ATTRS_BM64_TMEM with ctas_per_cga=(2,1,1).
// Keep the complete backward graph and pass-adjacent metadata in this fixture.
// CHECK: "ttng.two-ctas" = true
// CHECK-COUNT-5: tt.dot

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked7 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 4, 1], order = [1, 0, 2]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 16, 2], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked12 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked13 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
#blocked14 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 2, 2], order = [1, 2, 0]}>
#blocked15 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
#blocked16 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
#blocked17 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 1, 2], order = [1, 2, 0]}>
#blocked18 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 16, 2], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
#blocked19 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd_persist(%arg0: !tt.tensordesc<64x128xf16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<64x128xf16>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf16>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<128x128xf16>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.tensordesc<128x128xf16>, %arg21: i32, %arg22: i32, %arg23: i64, %arg24: i64, %arg25: f32, %arg26: !tt.tensordesc<64x128xf16>, %arg27: i32, %arg28: i32, %arg29: i64, %arg30: i64, %arg31: !tt.tensordesc<64x128xf16>, %arg32: i32, %arg33: i32, %arg34: i64, %arg35: i64, %arg36: !tt.tensordesc<32x32xf32>, %arg37: i32, %arg38: i32, %arg39: i64, %arg40: i64, %arg41: !tt.tensordesc<128x64xf16>, %arg42: i32, %arg43: i32, %arg44: i64, %arg45: i64, %arg46: !tt.tensordesc<128x64xf16>, %arg47: i32, %arg48: i32, %arg49: i64, %arg50: i64, %arg51: !tt.tensordesc<64xf32>, %arg52: i32, %arg53: i64, %arg54: !tt.tensordesc<64xf32>, %arg55: i32, %arg56: i64, %arg57: i32 {tt.divisibility = 16 : i32}, %arg58: i32 {tt.divisibility = 16 : i32}, %arg59: i32 {tt.divisibility = 16 : i32}, %arg60: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %cst_2 = arith.constant dense<0.693147182> : tensor<32x32xf32, #blocked2>
    %c96_i32 = arith.constant 96 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
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
    %14 = tt.splat %13 : i1 -> tensor<32x128xi1, #blocked>
    %15 = arith.muli %12, %c32_i32 : i32
    %16 = arith.extsi %15 : i32 to i64
    %17 = tt.splat %arg25 : f32 -> tensor<128x64xf32, #blocked1>
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
      %30 = tt.descriptor_load %arg10[%29, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked>
      %31 = tt.descriptor_load %arg15[%29, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked>
      %32 = tt.descriptor_load %arg20[%29, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked>
      %33:3 = scf.for %arg63 = %c0_i32 to %11 step %c1_i32 iter_args(%arg64 = %cst, %arg65 = %cst, %arg66 = %c0_i32) -> (tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>, i32)  : i32 {
        %49 = arith.extsi %arg66 : i32 to i64
        %50 = arith.addi %25, %49 : i64
        %51 = arith.trunci %50 : i64 to i32
        %52 = tt.descriptor_load %arg0[%51, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked>
        %53 = tt.descriptor_load %arg5[%51, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked>
        %54 = tt.trans %53 {order = array<i32: 1, 0>} : tensor<64x128xf16, #blocked> -> tensor<128x64xf16, #blocked3>
        %55 = ttg.convert_layout %54 : tensor<128x64xf16, #blocked3> -> tensor<128x64xf16, #blocked1>
        %56 = arith.addi %22, %49 : i64
        %57 = arith.trunci %56 : i64 to i32
        %58 = tt.descriptor_load %arg51[%57] : !tt.tensordesc<64xf32> -> tensor<64xf32, #blocked4>
        %59 = ttg.convert_layout %30 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked5}>>
        %60 = ttg.convert_layout %55 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked5}>>
        %61 = ttg.convert_layout %cst_0 : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #blocked5>
        %62 = tt.dot %59, %60, %61, inputPrecision = tf32 {tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked5}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked5}>> -> tensor<128x64xf32, #blocked5>
        %63 = ttg.convert_layout %62 : tensor<128x64xf32, #blocked5> -> tensor<128x64xf32, #blocked1>
        %64 = ttg.convert_layout %58 : tensor<64xf32, #blocked4> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked6}>>
        %65 = tt.expand_dims %64 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked6}>> -> tensor<1x64xf32, #blocked6>
        %66 = ttg.convert_layout %65 : tensor<1x64xf32, #blocked6> -> tensor<1x64xf32, #blocked1>
        %67 = tt.broadcast %66 : tensor<1x64xf32, #blocked1> -> tensor<128x64xf32, #blocked1>
        %68 = arith.subf %63, %67 : tensor<128x64xf32, #blocked1>
        %69 = math.exp2 %68 : tensor<128x64xf32, #blocked1>
        %70 = tt.descriptor_load %arg26[%51, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked>
        %71 = tt.descriptor_load %arg31[%51, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked>
        %72 = arith.truncf %69 : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
        %73 = tt.trans %71 {order = array<i32: 1, 0>} : tensor<64x128xf16, #blocked> -> tensor<128x64xf16, #blocked3>
        %74 = ttg.convert_layout %73 : tensor<128x64xf16, #blocked3> -> tensor<128x64xf16, #blocked1>
        %75 = ttg.convert_layout %32 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked5}>>
        %76 = ttg.convert_layout %74 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked5}>>
        %77 = ttg.convert_layout %cst_0 : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #blocked5>
        %78 = tt.dot %75, %76, %77, inputPrecision = tf32 {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked5}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked5}>> -> tensor<128x64xf32, #blocked5>
        %79 = ttg.convert_layout %78 : tensor<128x64xf32, #blocked5> -> tensor<128x64xf32, #blocked1>
        %80 = tt.descriptor_load %arg54[%57] : !tt.tensordesc<64xf32> -> tensor<64xf32, #blocked4>
        %81 = ttg.convert_layout %72 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>>
        %82 = ttg.convert_layout %70 : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked7}>>
        %83 = ttg.convert_layout %arg65 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked7>
        %84 = tt.dot %81, %82, %83, inputPrecision = tf32 {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", two_ctas} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked7}>> -> tensor<128x128xf32, #blocked7>
        %85 = ttg.convert_layout %84 : tensor<128x128xf32, #blocked7> -> tensor<128x128xf32, #blocked>
        %86 = ttg.convert_layout %80 : tensor<64xf32, #blocked4> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked6}>>
        %87 = tt.expand_dims %86 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked6}>> -> tensor<1x64xf32, #blocked6>
        %88 = ttg.convert_layout %87 : tensor<1x64xf32, #blocked6> -> tensor<1x64xf32, #blocked1>
        %89 = tt.broadcast %88 : tensor<1x64xf32, #blocked1> -> tensor<128x64xf32, #blocked1>
        %90 = arith.subf %79, %89 : tensor<128x64xf32, #blocked1>
        %91 = arith.mulf %69, %90 : tensor<128x64xf32, #blocked1>
        %92 = arith.truncf %91 : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
        %93 = ttg.convert_layout %92 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>>
        %94 = ttg.convert_layout %52 : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked7}>>
        %95 = ttg.convert_layout %arg64 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked7>
        %96 = tt.dot %93, %94, %95, inputPrecision = tf32 {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,tmem,1,5\22, \22opndD,tmem,1,10\22]}", two_ctas} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked7}>> -> tensor<128x128xf32, #blocked7>
        %97 = ttg.convert_layout %96 : tensor<128x128xf32, #blocked7> -> tensor<128x128xf32, #blocked>
        %98 = tt.trans %92 {order = array<i32: 1, 0>} : tensor<128x64xf16, #blocked1> -> tensor<64x128xf16, #blocked8>
        %99 = ttg.convert_layout %98 : tensor<64x128xf16, #blocked8> -> tensor<64x128xf16, #blocked>
        %100 = ttg.convert_layout %99 : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>>
        %101 = ttg.convert_layout %31 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked7}>>
        %102 = ttg.convert_layout %cst_1 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked7>
        %103 = tt.dot %100, %101, %102, inputPrecision = tf32 {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,11\22]}", two_ctas} : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked7}>> -> tensor<64x128xf32, #blocked7>
        %104 = ttg.convert_layout %103 : tensor<64x128xf32, #blocked7> -> tensor<64x128xf32, #blocked>
        %105 = tt.reshape %104 : tensor<64x128xf32, #blocked> -> tensor<2x32x128xf32, #blocked9>
        %106 = tt.trans %105 {order = array<i32: 1, 2, 0>} : tensor<2x32x128xf32, #blocked9> -> tensor<32x128x2xf32, #blocked10>
        %107 = ttg.convert_layout %106 : tensor<32x128x2xf32, #blocked10> -> tensor<32x128x2xf32, #blocked11>
        %108 = ttg.convert_layout %107 : tensor<32x128x2xf32, #blocked11> -> tensor<32x128x2xf32, #blocked12>
        %outLHS_5, %outRHS_6 = tt.split %108 : tensor<32x128x2xf32, #blocked12> -> tensor<32x128xf32, #blocked>
        %109 = arith.select %14, %outLHS_5, %outRHS_6 : tensor<32x128xi1, #blocked>, tensor<32x128xf32, #blocked>
        %110 = tt.reshape %109 : tensor<32x128xf32, #blocked> -> tensor<32x2x64xf32, #blocked13>
        %111 = tt.trans %110 {order = array<i32: 0, 2, 1>} : tensor<32x2x64xf32, #blocked13> -> tensor<32x64x2xf32, #blocked14>
        %112 = ttg.convert_layout %111 : tensor<32x64x2xf32, #blocked14> -> tensor<32x64x2xf32, #blocked11>
        %113 = ttg.convert_layout %112 : tensor<32x64x2xf32, #blocked11> -> tensor<32x64x2xf32, #blocked15>
        %outLHS_7, %outRHS_8 = tt.split %113 : tensor<32x64x2xf32, #blocked15> -> tensor<32x64xf32, #blocked1>
        %114 = tt.reshape %outLHS_7 : tensor<32x64xf32, #blocked1> -> tensor<32x2x32xf32, #blocked16>
        %115 = tt.trans %114 {order = array<i32: 0, 2, 1>} : tensor<32x2x32xf32, #blocked16> -> tensor<32x32x2xf32, #blocked17>
        %116 = ttg.convert_layout %115 : tensor<32x32x2xf32, #blocked17> -> tensor<32x32x2xf32, #blocked18>
        %117 = ttg.convert_layout %116 : tensor<32x32x2xf32, #blocked18> -> tensor<32x32x2xf32, #blocked19>
        %outLHS_9, %outRHS_10 = tt.split %117 : tensor<32x32x2xf32, #blocked19> -> tensor<32x32xf32, #blocked2>
        %118 = tt.reshape %outRHS_8 : tensor<32x64xf32, #blocked1> -> tensor<32x2x32xf32, #blocked16>
        %119 = tt.trans %118 {order = array<i32: 0, 2, 1>} : tensor<32x2x32xf32, #blocked16> -> tensor<32x32x2xf32, #blocked17>
        %120 = ttg.convert_layout %119 : tensor<32x32x2xf32, #blocked17> -> tensor<32x32x2xf32, #blocked18>
        %121 = ttg.convert_layout %120 : tensor<32x32x2xf32, #blocked18> -> tensor<32x32x2xf32, #blocked19>
        %outLHS_11, %outRHS_12 = tt.split %121 : tensor<32x32x2xf32, #blocked19> -> tensor<32x32xf32, #blocked2>
        %122 = arith.addi %50, %16 : i64
        %123 = arith.mulf %outLHS_9, %cst_2 : tensor<32x32xf32, #blocked2>
        %124 = arith.trunci %122 : i64 to i32
        tt.descriptor_reduce add, %arg36[%124, %c0_i32], %123 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked2>
        %125 = arith.mulf %outRHS_10, %cst_2 : tensor<32x32xf32, #blocked2>
        tt.descriptor_reduce add, %arg36[%124, %c32_i32], %125 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked2>
        %126 = arith.mulf %outLHS_11, %cst_2 : tensor<32x32xf32, #blocked2>
        tt.descriptor_reduce add, %arg36[%124, %c64_i32], %126 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked2>
        %127 = arith.mulf %outRHS_12, %cst_2 : tensor<32x32xf32, #blocked2>
        tt.descriptor_reduce add, %arg36[%124, %c96_i32], %127 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked2>
        %128 = arith.addi %arg66, %c64_i32 : i32
        scf.yield %97, %85, %128 : tensor<128x128xf32, #blocked>, tensor<128x128xf32, #blocked>, i32
      }
      %34 = tt.reshape %33#1 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked13>
      %35 = tt.trans %34 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked13> -> tensor<128x64x2xf32, #blocked14>
      %36 = ttg.convert_layout %35 : tensor<128x64x2xf32, #blocked14> -> tensor<128x64x2xf32, #blocked11>
      %37 = ttg.convert_layout %36 : tensor<128x64x2xf32, #blocked11> -> tensor<128x64x2xf32, #blocked15>
      %outLHS, %outRHS = tt.split %37 : tensor<128x64x2xf32, #blocked15> -> tensor<128x64xf32, #blocked1>
      %38 = arith.truncf %outLHS : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
      tt.descriptor_store %arg46[%29, %c0_i32], %38 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked1>
      %39 = arith.truncf %outRHS : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
      tt.descriptor_store %arg46[%29, %c64_i32], %39 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked1>
      %40 = tt.reshape %33#0 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked13>
      %41 = tt.trans %40 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked13> -> tensor<128x64x2xf32, #blocked14>
      %42 = ttg.convert_layout %41 : tensor<128x64x2xf32, #blocked14> -> tensor<128x64x2xf32, #blocked11>
      %43 = ttg.convert_layout %42 : tensor<128x64x2xf32, #blocked11> -> tensor<128x64x2xf32, #blocked15>
      %outLHS_3, %outRHS_4 = tt.split %43 : tensor<128x64x2xf32, #blocked15> -> tensor<128x64xf32, #blocked1>
      %44 = arith.mulf %outLHS_3, %17 : tensor<128x64xf32, #blocked1>
      %45 = arith.truncf %44 : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
      tt.descriptor_store %arg41[%29, %c0_i32], %45 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked1>
      %46 = arith.mulf %outRHS_4, %17 : tensor<128x64xf32, #blocked1>
      %47 = arith.truncf %46 : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
      tt.descriptor_store %arg41[%29, %c64_i32], %47 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked1>
      %48 = arith.addi %arg62, %5 : i32
      scf.yield %48 : i32
    } {tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
