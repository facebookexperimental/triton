// RUN: triton-opt %s --nvgpu-plan-2cta-exchange | FileCheck %s --check-prefix=COUNT
// RUN: triton-opt %s --nvgpu-plan-2cta-exchange | FileCheck %s --check-prefix=PLAN

// Full TTGIR captured immediately after NVGPU2CTATransformLoads and before
// NVGPUPlan2CTAExchange from the persistent _BWD_DOT_ATTRS_BM64_TMEM 2-CTA
// configuration.
// COUNT-COUNT-1: ttng.two_cta_peer_gather
// PLAN: %[[GATHER:.*]] = ttng.two_cta_peer_gather %{{.*}} split_dim = 0 num_ctas = 2
// PLAN: ttg.local_alloc %[[GATHER]]

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [4, 1, 8], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
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
  tt.func public @_attn_bwd_persist(%arg0: !tt.tensordesc<64x64xf16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<32x128xf16>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf16>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<128x64xf16>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.tensordesc<128x128xf16>, %arg21: i32, %arg22: i32, %arg23: i64, %arg24: i64, %arg25: f32, %arg26: !tt.tensordesc<64x64xf16>, %arg27: i32, %arg28: i32, %arg29: i64, %arg30: i64, %arg31: !tt.tensordesc<32x128xf16>, %arg32: i32, %arg33: i32, %arg34: i64, %arg35: i64, %arg36: !tt.tensordesc<32x32xf32>, %arg37: i32, %arg38: i32, %arg39: i64, %arg40: i64, %arg41: !tt.tensordesc<128x64xf16>, %arg42: i32, %arg43: i32, %arg44: i64, %arg45: i64, %arg46: !tt.tensordesc<128x64xf16>, %arg47: i32, %arg48: i32, %arg49: i64, %arg50: i64, %arg51: !tt.tensordesc<64xf32>, %arg52: i32, %arg53: i64, %arg54: !tt.tensordesc<64xf32>, %arg55: i32, %arg56: i64, %arg57: i32 {tt.divisibility = 16 : i32}, %arg58: i32 {tt.divisibility = 16 : i32}, %arg59: i32 {tt.divisibility = 16 : i32}, %arg60: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
      %33 = nvg.cluster_id
      %c2_i32_3 = arith.constant 2 : i32
      %34 = arith.remsi %33, %c2_i32_3 : i32
      %c64_i32_4 = arith.constant 64 : i32
      %35 = arith.muli %34, %c64_i32_4 : i32
      %36 = arith.addi %c0_i32, %35 : i32
      %37 = tt.descriptor_load %arg15[%30, %36] {two_cta_b} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked2>
      %38 = ttg.local_alloc %37 : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %39 = tt.descriptor_load %arg20[%30, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked1>
      %40 = ttg.local_alloc %39 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %41:3 = scf.for %arg63 = %c0_i32 to %11 step %c1_i32 iter_args(%arg64 = %cst_2, %arg65 = %cst_2, %arg66 = %c0_i32) -> (tensor<128x128xf32, #linear2>, tensor<128x128xf32, #linear2>, i32)  : i32 {
        %57 = arith.extsi %arg66 : i32 to i64
        %58 = arith.addi %26, %57 : i64
        %59 = arith.trunci %58 : i64 to i32
        %60 = nvg.cluster_id
        %c2_i32_7 = arith.constant 2 : i32
        %61 = arith.remsi %60, %c2_i32_7 : i32
        %c64_i32_8 = arith.constant 64 : i32
        %62 = arith.muli %61, %c64_i32_8 : i32
        %63 = arith.addi %c0_i32, %62 : i32
        %64 = tt.descriptor_load %arg0[%59, %63] {two_cta_b} : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked2>
        %65 = ttg.local_alloc %64 : (tensor<64x64xf16, #blocked2>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
        %66 = nvg.cluster_id
        %c2_i32_9 = arith.constant 2 : i32
        %67 = arith.remsi %66, %c2_i32_9 : i32
        %c32_i32_10 = arith.constant 32 : i32
        %68 = arith.muli %67, %c32_i32_10 : i32
        %69 = arith.addi %59, %68 : i32
        %70 = tt.descriptor_load %arg5[%69, %c0_i32] {two_cta_b} : !tt.tensordesc<32x128xf16> -> tensor<32x128xf16, #blocked1>
        %71 = ttg.local_alloc %70 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
        %72 = ttg.memdesc_trans %71 {order = array<i32: 1, 0>} : !ttg.memdesc<32x128xf16, #shared, #smem> -> !ttg.memdesc<128x32xf16, #shared1, #smem>
        %73 = arith.addi %23, %57 : i64
        %74 = arith.trunci %73 : i64 to i32
        %75 = tt.descriptor_load %arg51[%74] : !tt.tensordesc<64xf32> -> tensor<64xf32, #blocked3>
        %result, %token = ttng.tmem_alloc %cst_0 : (tensor<128x64xf32, #linear1>) -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %76 = ttng.tc_gen5_mma %32, %72, %result[%token], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x32xf16, #shared1, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %result_11, %token_12 = ttng.tmem_load %result[%76] : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear1>
        %77 = ttg.convert_layout %75 : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %78 = tt.expand_dims %77 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1>
        %79 = tt.broadcast %78 : tensor<1x64xf32, #linear1> -> tensor<128x64xf32, #linear1>
        %80 = arith.subf %result_11, %79 : tensor<128x64xf32, #linear1>
        %81 = math.exp2 %80 : tensor<128x64xf32, #linear1>
        %82 = nvg.cluster_id
        %c2_i32_13 = arith.constant 2 : i32
        %83 = arith.remsi %82, %c2_i32_13 : i32
        %c64_i32_14 = arith.constant 64 : i32
        %84 = arith.muli %83, %c64_i32_14 : i32
        %85 = arith.addi %c0_i32, %84 : i32
        %86 = tt.descriptor_load %arg26[%59, %85] {two_cta_b} : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked2>
        %87 = ttg.local_alloc %86 : (tensor<64x64xf16, #blocked2>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
        %88 = nvg.cluster_id
        %c2_i32_15 = arith.constant 2 : i32
        %89 = arith.remsi %88, %c2_i32_15 : i32
        %c32_i32_16 = arith.constant 32 : i32
        %90 = arith.muli %89, %c32_i32_16 : i32
        %91 = arith.addi %59, %90 : i32
        %92 = tt.descriptor_load %arg31[%91, %c0_i32] {two_cta_b} : !tt.tensordesc<32x128xf16> -> tensor<32x128xf16, #blocked1>
        %93 = arith.truncf %81 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
        %94 = ttg.local_alloc %93 : (tensor<128x64xf16, #linear1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %95 = ttg.local_alloc %92 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
        %96 = ttg.memdesc_trans %95 {order = array<i32: 1, 0>} : !ttg.memdesc<32x128xf16, #shared, #smem> -> !ttg.memdesc<128x32xf16, #shared1, #smem>
        %result_17, %token_18 = ttng.tmem_alloc %cst_0 : (tensor<128x64xf32, #linear1>) -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %97 = ttng.tc_gen5_mma %40, %96, %result_17[%token_18], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x32xf16, #shared1, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %result_19, %token_20 = ttng.tmem_load %result_17[%97] : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear1>
        %98 = tt.descriptor_load %arg54[%74] : !tt.tensordesc<64xf32> -> tensor<64xf32, #blocked3>
        %result_21, %token_22 = ttng.tmem_alloc %arg65 : (tensor<128x128xf32, #linear2>) -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %99 = ttng.tc_gen5_mma %94, %87, %result_21[%token_22], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_23, %token_24 = ttng.tmem_load %result_21[%99] : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear2>
        %100 = ttg.convert_layout %98 : tensor<64xf32, #blocked3> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %101 = tt.expand_dims %100 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1>
        %102 = tt.broadcast %101 : tensor<1x64xf32, #linear1> -> tensor<128x64xf32, #linear1>
        %103 = arith.subf %result_19, %102 : tensor<128x64xf32, #linear1>
        %104 = arith.mulf %81, %103 : tensor<128x64xf32, #linear1>
        %105 = arith.truncf %104 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
        %106 = ttg.local_alloc %105 : (tensor<128x64xf16, #linear1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %result_25, %token_26 = ttng.tmem_alloc %arg64 : (tensor<128x128xf32, #linear2>) -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %107 = ttng.tc_gen5_mma %106, %65, %result_25[%token_26], %true, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,tmem,1,5\22, \22opndD,tmem,1,10\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_27, %token_28 = ttng.tmem_load %result_25[%107] : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear2>
        %108 = ttg.local_alloc %105 : (tensor<128x64xf16, #linear1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %109 = ttg.memdesc_trans %108 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %result_29, %token_30 = ttng.tmem_alloc %cst : (tensor<64x128xf32, #linear>) -> (!ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %110 = ttng.tc_gen5_mma %109, %38, %result_29[%token_30], %true, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,11\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
        %result_31, %token_32 = ttng.tmem_load %result_29[%110] : !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
        %111 = tt.reshape %result_31 : tensor<64x128xf32, #linear> -> tensor<2x32x128xf32, #linear4>
        %112 = tt.trans %111 {order = array<i32: 1, 2, 0>} : tensor<2x32x128xf32, #linear4> -> tensor<32x128x2xf32, #linear5>
        %113 = ttg.convert_layout %112 : tensor<32x128x2xf32, #linear5> -> tensor<32x128x2xf32, #linear6>
        %outLHS_33, %outRHS_34 = tt.split %113 : tensor<32x128x2xf32, #linear6> -> tensor<32x128xf32, #linear3>
        %114 = arith.select %14, %outLHS_33, %outRHS_34 : tensor<32x128xi1, #linear3>, tensor<32x128xf32, #linear3>
        %115 = tt.reshape %114 : tensor<32x128xf32, #linear3> -> tensor<32x2x64xf32, #linear7>
        %116 = tt.trans %115 {order = array<i32: 0, 2, 1>} : tensor<32x2x64xf32, #linear7> -> tensor<32x64x2xf32, #linear8>
        %outLHS_35, %outRHS_36 = tt.split %116 : tensor<32x64x2xf32, #linear8> -> tensor<32x64xf32, #linear9>
        %117 = tt.reshape %outLHS_35 : tensor<32x64xf32, #linear9> -> tensor<32x2x32xf32, #blocked4>
        %118 = tt.trans %117 {order = array<i32: 0, 2, 1>} : tensor<32x2x32xf32, #blocked4> -> tensor<32x32x2xf32, #blocked5>
        %outLHS_37, %outRHS_38 = tt.split %118 : tensor<32x32x2xf32, #blocked5> -> tensor<32x32xf32, #blocked>
        %119 = tt.reshape %outRHS_36 : tensor<32x64xf32, #linear9> -> tensor<32x2x32xf32, #blocked4>
        %120 = tt.trans %119 {order = array<i32: 0, 2, 1>} : tensor<32x2x32xf32, #blocked4> -> tensor<32x32x2xf32, #blocked5>
        %outLHS_39, %outRHS_40 = tt.split %120 : tensor<32x32x2xf32, #blocked5> -> tensor<32x32xf32, #blocked>
        %121 = arith.addi %58, %16 : i64
        %122 = arith.mulf %outLHS_37, %cst_1 : tensor<32x32xf32, #blocked>
        %123 = arith.trunci %121 : i64 to i32
        tt.descriptor_reduce add, %arg36[%123, %c0_i32], %122 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked>
        %124 = arith.mulf %outRHS_38, %cst_1 : tensor<32x32xf32, #blocked>
        tt.descriptor_reduce add, %arg36[%123, %c32_i32], %124 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked>
        %125 = arith.mulf %outLHS_39, %cst_1 : tensor<32x32xf32, #blocked>
        tt.descriptor_reduce add, %arg36[%123, %c64_i32], %125 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked>
        %126 = arith.mulf %outRHS_40, %cst_1 : tensor<32x32xf32, #blocked>
        tt.descriptor_reduce add, %arg36[%123, %c96_i32], %126 : !tt.tensordesc<32x32xf32>, tensor<32x32xf32, #blocked>
        %127 = arith.addi %arg66, %c64_i32 : i32
        scf.yield %result_27, %result_23, %127 : tensor<128x128xf32, #linear2>, tensor<128x128xf32, #linear2>, i32
      }
      %42 = tt.reshape %41#1 : tensor<128x128xf32, #linear2> -> tensor<128x2x64xf32, #linear10>
      %43 = tt.trans %42 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear10> -> tensor<128x64x2xf32, #linear11>
      %outLHS, %outRHS = tt.split %43 : tensor<128x64x2xf32, #linear11> -> tensor<128x64xf32, #linear1>
      %44 = arith.truncf %outLHS : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %45 = ttg.convert_layout %44 : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked2>
      tt.descriptor_store %arg46[%30, %c0_i32], %45 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked2>
      %46 = arith.truncf %outRHS : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %47 = ttg.convert_layout %46 : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked2>
      tt.descriptor_store %arg46[%30, %c64_i32], %47 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked2>
      %48 = tt.reshape %41#0 : tensor<128x128xf32, #linear2> -> tensor<128x2x64xf32, #linear10>
      %49 = tt.trans %48 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear10> -> tensor<128x64x2xf32, #linear11>
      %outLHS_5, %outRHS_6 = tt.split %49 : tensor<128x64x2xf32, #linear11> -> tensor<128x64xf32, #linear1>
      %50 = arith.mulf %outLHS_5, %18 : tensor<128x64xf32, #linear1>
      %51 = arith.truncf %50 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %52 = ttg.convert_layout %51 : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked2>
      tt.descriptor_store %arg41[%30, %c0_i32], %52 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked2>
      %53 = arith.mulf %outRHS_6, %17 : tensor<128x64xf32, #linear1>
      %54 = arith.truncf %53 : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %55 = ttg.convert_layout %54 : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked2>
      tt.descriptor_store %arg41[%30, %c64_i32], %55 : !tt.tensordesc<128x64xf16>, tensor<128x64xf16, #blocked2>
      %56 = arith.addi %arg62, %5 : i32
      scf.yield %56 : i32
    } {tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
