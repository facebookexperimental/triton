// RUN: triton-opt %s --nvgpu-partition-scheduling-meta | FileCheck %s
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: 2>
// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: 0>
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: 3>
// CHECK: ttng.two_cta_peer_relay {{.*}}ttg.partition = array<i32: 4>
// CHECK: ttng.async_tma_reduce {{.*}}ttg.partition = array<i32: 1>
// CHECK: ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"]

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 2, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [8, 1, 4], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 2, 8], threadsPerWarp = [16, 1, 2], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 128], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64], [32, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[128, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 32]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 1, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 0, 1]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 64], [0, 32]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0], [0, 1, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0], [0, 0, 1]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 0, 1], [0, 16, 0], [0, 1, 0], [0, 2, 0], [64, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0], [32, 0, 0]], block = []}>
#linear9 = #ttg.linear<{register = [[0, 16], [0, 1], [0, 2], [64, 0]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [32, 0]], block = []}>
#linear10 = #ttg.linear<{register = [[0, 0, 1], [0, 32, 0], [0, 16, 0], [0, 1, 0], [0, 2, 0], [0, 4, 0]], lane = [[0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]], warp = [[16, 0, 0], [32, 0, 0], [64, 0, 0]], block = []}>
#linear11 = #ttg.linear<{register = [[0, 32], [0, 16], [0, 1], [0, 2], [0, 4]], lane = [[0, 8], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[16, 0], [32, 0], [64, 0]], block = []}>
#linear12 = #ttg.linear<{register = [[0, 1, 0], [0, 0, 16], [0, 0, 1], [0, 0, 2], [0, 0, 4]], lane = [[0, 0, 8], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]], warp = [[16, 0, 0], [32, 0, 0], [64, 0, 0]], block = []}>
#linear13 = #ttg.linear<{register = [[0, 0, 1], [0, 16, 0], [0, 1, 0], [0, 2, 0], [0, 4, 0]], lane = [[0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]], warp = [[16, 0, 0], [32, 0, 0], [64, 0, 0]], block = []}>
#linear14 = #ttg.linear<{register = [[0, 16], [0, 1], [0, 2], [0, 4]], lane = [[0, 8], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[16, 0], [32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true, ctaMode = twocta_rhs>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @_attn_bwd_persist(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<64x128xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf16, #shared>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<256x64xf16, #shared>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.tensordesc<128x128xf16, #shared>, %arg21: i32, %arg22: i32, %arg23: i64, %arg24: i64, %arg25: f32, %arg26: !tt.tensordesc<128x64xf16, #shared>, %arg27: i32, %arg28: i32, %arg29: i64, %arg30: i64, %arg31: !tt.tensordesc<64x128xf16, #shared>, %arg32: i32, %arg33: i32, %arg34: i64, %arg35: i64, %arg36: !tt.tensordesc<128x16xf32, #shared1>, %arg37: i32, %arg38: i32, %arg39: i64, %arg40: i64, %arg41: !tt.tensordesc<128x16xf16, #shared2>, %arg42: i32, %arg43: i32, %arg44: i64, %arg45: i64, %arg46: !tt.tensordesc<128x16xf16, #shared2>, %arg47: i32, %arg48: i32, %arg49: i64, %arg50: i64, %arg51: !tt.tensordesc<128xf32, #shared3>, %arg52: i32, %arg53: i64, %arg54: !tt.tensordesc<128xf32, #shared3>, %arg55: i32, %arg56: i64, %arg57: i32 {tt.divisibility = 16 : i32}, %arg58: i32 {tt.divisibility = 16 : i32}, %arg59: i32 {tt.divisibility = 16 : i32}, %arg60: i32, %arg61: i32, %arg62: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<0.693147182> : tensor<128x16xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %c2_i64 = arith.constant 2 : i64
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c48_i32 = arith.constant 48 : i32
    %c64_i32 = arith.constant 64 : i32
    %c80_i32 = arith.constant 80 : i32
    %c96_i32 = arith.constant 96 : i32
    %c112_i32 = arith.constant 112 : i32
    %true = arith.constant true
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = arith.addi %arg62, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %2 = tt.get_program_id x : i32
    %3 = arith.remsi %2, %c2_i32 : i32
    %4 = arith.divsi %2, %c2_i32 : i32
    %5 = tt.get_num_programs x : i32
    %6 = arith.divsi %5, %c2_i32 : i32
    %7 = arith.divsi %1, %c2_i32 : i32
    %8 = arith.muli %7, %arg60 : i32
    %9 = arith.muli %8, %arg61 : i32
    %10 = arith.divsi %9, %6 : i32
    %11 = arith.remsi %9, %6 : i32
    %12 = arith.cmpi slt, %4, %11 : i32
    %13 = scf.if %12 -> (i32) {
      %24 = arith.addi %10, %c1_i32 : i32
      scf.yield %24 : i32
    } else {
      scf.yield %10 : i32
    }
    %14 = arith.extsi %arg59 : i32 to i64
    %15 = arith.muli %3, %c128_i32 : i32
    %16 = arith.divsi %arg62, %c128_i32 : i32
    %17 = arith.muli %3, %c64_i32 : i32
    %18 = arith.extsi %17 : i32 to i64
    %19 = tt.splat %arg25 : f32 -> tensor<128x16xf32, #blocked1>
    %20 = nvg.cluster_id
    %21 = arith.remsi %20, %c2_i32 : i32
    %22 = arith.muli %21, %c64_i32 : i32
    %23 = scf.for %arg63 = %c0_i32 to %13 step %c1_i32 iter_args(%arg64 = %4) -> (i32)  : i32 {
      %24 = arith.remsi %arg64, %7 : i32
      %25 = arith.divsi %arg64, %7 : i32
      %26 = arith.muli %24, %c2_i32 : i32
      %27 = arith.addi %26, %3 : i32
      %28 = arith.muli %25, %arg62 : i32
      %29 = arith.extsi %28 : i32 to i64
      %30 = arith.remsi %25, %arg61 : i32
      %31 = arith.muli %arg58, %30 : i32
      %32 = arith.divsi %25, %arg61 : i32
      %33 = arith.muli %arg57, %32 : i32
      %34 = arith.addi %31, %33 : i32
      %35 = arith.extsi %34 : i32 to i64
      %36 = arith.divsi %35, %14 : i64
      %37 = arith.muli %27, %c128_i32 : i32
      %38 = arith.extsi %37 : i32 to i64
      %39 = arith.addi %36, %38 : i64
      %40 = arith.trunci %39 : i64 to i32
      %41 = tt.descriptor_load %arg10[%40, %c0_i32] : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked2>
      %42 = ttg.local_alloc %41 : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %43 = arith.subi %37, %15 : i32
      %44 = arith.extsi %43 : i32 to i64
      %45 = arith.addi %36, %44 : i64
      %46 = arith.trunci %45 : i64 to i32
      %47 = tt.descriptor_load %arg15[%46, %22] {two_cta_b} : !tt.tensordesc<256x64xf16, #shared> -> tensor<256x64xf16, #blocked3>
      %48 = ttg.local_alloc %47 : (tensor<256x64xf16, #blocked3>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
      %49 = tt.descriptor_load %arg20[%40, %c0_i32] : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked2>
      %50 = ttg.local_alloc %49 : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_1, %token_2 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_3, %token_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_5, %token_6 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_7, %token_8 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %51 = ttng.tmem_store %cst_0, %result_5[%token_6], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %52 = ttng.tmem_store %cst_0, %result_3[%token_4], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %53:7 = scf.for %arg65 = %c0_i32 to %16 step %c1_i32 iter_args(%arg66 = %c0_i32, %arg67 = %false, %arg68 = %token, %arg69 = %token_2, %arg70 = %52, %arg71 = %51, %arg72 = %token_8) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %141 = arith.extsi %arg66 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 to i64
        %142 = arith.addi %36, %141 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %143 = arith.trunci %142 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
        %144 = arith.addi %143, %22 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        %145 = tt.descriptor_load %arg5[%144, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, two_cta_b} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked2>
        %146 = ttg.local_alloc %145 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<64x128xf16, #blocked2>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %147 = ttg.memdesc_trans %146 {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared4, #smem>
        %148 = tt.descriptor_load %arg0[%143, %22] {loop.cluster = 1 : i32, loop.stage = 0 : i32, two_cta_b} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked3>
        %149 = ttg.local_alloc %148 {loop.cluster = 5 : i32, loop.stage = 0 : i32} : (tensor<128x64xf16, #blocked3>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %150 = arith.addi %29, %141 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %151 = arith.trunci %150 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
        %152 = tt.descriptor_load %arg51[%151] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128xf32, #shared3> -> tensor<128xf32, #blocked4>
        %153 = ttng.tc_gen5_mma %42, %147, %result[%arg68], %false, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,1,1\22, \22opndD,tmem,1,2\22]}", tt.self_latency = 0 : i32, two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared4, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %154 = ttg.convert_layout %152 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked4> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
        %155 = tt.expand_dims %154 {axis = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
        %156 = tt.broadcast %155 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %result_39, %token_40 = ttng.tmem_load %result[%153] {loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %157 = arith.subf %result_39, %156 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #linear>
        %158 = math.exp2 %157 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #linear>
        %159 = tt.descriptor_load %arg26[%143, %22] {loop.cluster = 1 : i32, loop.stage = 0 : i32, two_cta_b} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked3>
        %160 = ttg.local_alloc %159 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : (tensor<128x64xf16, #blocked3>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %161 = tt.descriptor_load %arg31[%144, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, two_cta_b} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked2>
        %162 = arith.truncf %158 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %result_41 = ttng.tmem_alloc %162 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
        %163 = ttg.local_alloc %161 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : (tensor<64x128xf16, #blocked2>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %164 = ttg.memdesc_trans %163 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared4, #smem>
        %165 = ttng.tc_gen5_mma %50, %164, %result_1[%arg69], %false, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", tt.self_latency = 0 : i32, two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared4, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %166 = tt.descriptor_load %arg54[%151] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128xf32, #shared3> -> tensor<128xf32, #blocked4>
        %167 = ttng.tc_gen5_mma %result_41, %160, %result_3[%arg70], %arg67, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", tt.self_latency = 0 : i32, ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %168 = ttg.convert_layout %166 {loop.cluster = 5 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked4> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
        %169 = tt.expand_dims %168 {axis = 0 : i32, loop.cluster = 5 : i32, loop.stage = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
        %170 = tt.broadcast %169 {loop.cluster = 5 : i32, loop.stage = 0 : i32} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %result_42, %token_43 = ttng.tmem_load %result_1[%165] {loop.cluster = 5 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %171 = arith.subf %result_42, %170 {loop.cluster = 5 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #linear>
        %172 = arith.mulf %158, %171 {loop.cluster = 5 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #linear>
        %173 = arith.truncf %172 {loop.cluster = 5 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %result_44 = ttng.tmem_alloc %173 {loop.cluster = 5 : i32, loop.stage = 0 : i32} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
        %174 = ttng.tc_gen5_mma %result_44, %149, %result_5[%arg71], %arg67, %true {loop.cluster = 5 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \223\22, \22channels\22: [\22opndD,tmem,1,10\22]}", tt.self_latency = 0 : i32, ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %175 = ttng.two_cta_peer_gather %173 split_dim = 1 num_ctas = 2 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #linear> -> tensor<64x256xf16, #linear1>
        %176 = tt.trans %175 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : tensor<64x256xf16, #linear1> -> tensor<256x64xf16, #linear2>
        %177 = tt.reshape %173 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #linear> -> tensor<128x2x64xf16, #linear3>
        %178 = tt.trans %177 {loop.cluster = 1 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf16, #linear3> -> tensor<128x64x2xf16, #linear4>
        %179 = ttg.convert_layout %178 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x64x2xf16, #linear4> -> tensor<128x64x2xf16, #blocked5>
        %outLHS_45, %outRHS_46 = tt.split %179 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x64x2xf16, #blocked5> -> tensor<128x64xf16, #blocked6>
        %180 = ttg.local_alloc %outLHS_45 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : (tensor<128x64xf16, #blocked6>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %181 = ttg.local_alloc %176 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : (tensor<256x64xf16, #linear2>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
        ttng.two_cta_peer_relay %180 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : <128x64xf16, #shared, #smem>
        %182 = ttg.memdesc_trans %181 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem> -> !ttg.memdesc<64x256xf16, #shared4, #smem>
        %183 = ttng.tc_gen5_mma %182, %48, %result_7[%arg72], %false, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x256xf16, #shared4, #smem>, !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_47, %token_48 = ttng.tmem_load %result_7[%183] {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear5>
        %184 = tt.reshape %result_47 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #linear5> -> tensor<128x2x32xf32, #linear6>
        %185 = tt.trans %184 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear6> -> tensor<128x32x2xf32, #linear7>
        %186 = ttg.convert_layout %185 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #linear7> -> tensor<128x32x2xf32, #linear8>
        %outLHS_49, %outRHS_50 = tt.split %186 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #linear8> -> tensor<128x32xf32, #linear9>
        %187 = tt.reshape %outLHS_49 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #linear9> -> tensor<128x2x16xf32, #blocked7>
        %188 = tt.trans %187 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked7> -> tensor<128x16x2xf32, #blocked8>
        %outLHS_51, %outRHS_52 = tt.split %188 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x16x2xf32, #blocked8> -> tensor<128x16xf32, #blocked>
        %189 = tt.reshape %outRHS_50 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #linear9> -> tensor<128x2x16xf32, #blocked7>
        %190 = tt.trans %189 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked7> -> tensor<128x16x2xf32, #blocked8>
        %outLHS_53, %outRHS_54 = tt.split %190 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x16x2xf32, #blocked8> -> tensor<128x16xf32, #blocked>
        %191 = arith.addi %142, %18 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : i64
        %192 = arith.muli %191, %c2_i64 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : i64
        %193 = arith.mulf %outLHS_51, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x16xf32, #blocked>
        %194 = arith.trunci %192 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : i64 to i32
        %195 = ttg.local_alloc %193 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : (tensor<128x16xf32, #blocked>) -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        %196 = ttng.async_tma_reduce add, %arg36[%194, %c0_i32] %195 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %196   {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token
        %197 = arith.mulf %outRHS_52, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x16xf32, #blocked>
        %198 = ttg.local_alloc %197 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : (tensor<128x16xf32, #blocked>) -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        %199 = ttng.async_tma_reduce add, %arg36[%194, %c16_i32] %198 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %199   {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token
        %200 = arith.mulf %outLHS_53, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x16xf32, #blocked>
        %201 = ttg.local_alloc %200 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : (tensor<128x16xf32, #blocked>) -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        %202 = ttng.async_tma_reduce add, %arg36[%194, %c32_i32] %201 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %202   {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token
        %203 = arith.mulf %outRHS_54, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x16xf32, #blocked>
        %204 = ttg.local_alloc %203 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : (tensor<128x16xf32, #blocked>) -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        %205 = ttng.async_tma_reduce add, %arg36[%194, %c48_i32] %204 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %205   {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token
        %206 = arith.addi %arg66, %c128_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
        scf.yield %206, %true, %token_40, %token_43, %167, %174, %token_48 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {tt.scheduled_max_stage = 1 : i32}
      %result_9, %token_10 = ttng.tmem_load %result_3[%53#4] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %54 = tt.reshape %result_9 : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear3>
      %55 = tt.trans %54 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear3> -> tensor<128x64x2xf32, #linear4>
      %56 = ttg.convert_layout %55 : tensor<128x64x2xf32, #linear4> -> tensor<128x64x2xf32, #linear10>
      %outLHS, %outRHS = tt.split %56 : tensor<128x64x2xf32, #linear10> -> tensor<128x64xf32, #linear11>
      %57 = tt.reshape %outLHS : tensor<128x64xf32, #linear11> -> tensor<128x2x32xf32, #linear12>
      %58 = tt.trans %57 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear12> -> tensor<128x32x2xf32, #linear13>
      %outLHS_11, %outRHS_12 = tt.split %58 : tensor<128x32x2xf32, #linear13> -> tensor<128x32xf32, #linear14>
      %59 = tt.reshape %outLHS_11 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked9>
      %60 = tt.trans %59 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked9> -> tensor<128x16x2xf32, #blocked10>
      %outLHS_13, %outRHS_14 = tt.split %60 : tensor<128x16x2xf32, #blocked10> -> tensor<128x16xf32, #blocked1>
      %61 = tt.reshape %outRHS_12 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked9>
      %62 = tt.trans %61 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked9> -> tensor<128x16x2xf32, #blocked10>
      %outLHS_15, %outRHS_16 = tt.split %62 : tensor<128x16x2xf32, #blocked10> -> tensor<128x16xf32, #blocked1>
      %63 = tt.reshape %outRHS : tensor<128x64xf32, #linear11> -> tensor<128x2x32xf32, #linear12>
      %64 = tt.trans %63 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear12> -> tensor<128x32x2xf32, #linear13>
      %outLHS_17, %outRHS_18 = tt.split %64 : tensor<128x32x2xf32, #linear13> -> tensor<128x32xf32, #linear14>
      %65 = tt.reshape %outLHS_17 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked9>
      %66 = tt.trans %65 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked9> -> tensor<128x16x2xf32, #blocked10>
      %outLHS_19, %outRHS_20 = tt.split %66 : tensor<128x16x2xf32, #blocked10> -> tensor<128x16xf32, #blocked1>
      %67 = tt.reshape %outRHS_18 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked9>
      %68 = tt.trans %67 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked9> -> tensor<128x16x2xf32, #blocked10>
      %outLHS_21, %outRHS_22 = tt.split %68 : tensor<128x16x2xf32, #blocked10> -> tensor<128x16xf32, #blocked1>
      %69 = arith.truncf %outLHS_13 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %70 = ttg.local_alloc %69 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %71 = ttng.async_tma_copy_local_to_global %arg46[%40, %c0_i32] %70 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %71   : !ttg.async.token
      %72 = arith.truncf %outRHS_14 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %73 = ttg.local_alloc %72 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %74 = ttng.async_tma_copy_local_to_global %arg46[%40, %c16_i32] %73 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %74   : !ttg.async.token
      %75 = arith.truncf %outLHS_15 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %76 = ttg.local_alloc %75 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %77 = ttng.async_tma_copy_local_to_global %arg46[%40, %c32_i32] %76 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %77   : !ttg.async.token
      %78 = arith.truncf %outRHS_16 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %79 = ttg.local_alloc %78 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %80 = ttng.async_tma_copy_local_to_global %arg46[%40, %c48_i32] %79 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %80   : !ttg.async.token
      %81 = arith.truncf %outLHS_19 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %82 = ttg.local_alloc %81 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %83 = ttng.async_tma_copy_local_to_global %arg46[%40, %c64_i32] %82 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %83   : !ttg.async.token
      %84 = arith.truncf %outRHS_20 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %85 = ttg.local_alloc %84 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %86 = ttng.async_tma_copy_local_to_global %arg46[%40, %c80_i32] %85 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %86   : !ttg.async.token
      %87 = arith.truncf %outLHS_21 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %88 = ttg.local_alloc %87 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %89 = ttng.async_tma_copy_local_to_global %arg46[%40, %c96_i32] %88 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %89   : !ttg.async.token
      %90 = arith.truncf %outRHS_22 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %91 = ttg.local_alloc %90 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %92 = ttng.async_tma_copy_local_to_global %arg46[%40, %c112_i32] %91 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %92   : !ttg.async.token
      %result_23, %token_24 = ttng.tmem_load %result_5[%53#5] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %93 = tt.reshape %result_23 : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear3>
      %94 = tt.trans %93 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear3> -> tensor<128x64x2xf32, #linear4>
      %95 = ttg.convert_layout %94 : tensor<128x64x2xf32, #linear4> -> tensor<128x64x2xf32, #linear10>
      %outLHS_25, %outRHS_26 = tt.split %95 : tensor<128x64x2xf32, #linear10> -> tensor<128x64xf32, #linear11>
      %96 = tt.reshape %outLHS_25 : tensor<128x64xf32, #linear11> -> tensor<128x2x32xf32, #linear12>
      %97 = tt.trans %96 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear12> -> tensor<128x32x2xf32, #linear13>
      %outLHS_27, %outRHS_28 = tt.split %97 : tensor<128x32x2xf32, #linear13> -> tensor<128x32xf32, #linear14>
      %98 = tt.reshape %outLHS_27 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked9>
      %99 = tt.trans %98 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked9> -> tensor<128x16x2xf32, #blocked10>
      %outLHS_29, %outRHS_30 = tt.split %99 : tensor<128x16x2xf32, #blocked10> -> tensor<128x16xf32, #blocked1>
      %100 = tt.reshape %outRHS_28 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked9>
      %101 = tt.trans %100 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked9> -> tensor<128x16x2xf32, #blocked10>
      %outLHS_31, %outRHS_32 = tt.split %101 : tensor<128x16x2xf32, #blocked10> -> tensor<128x16xf32, #blocked1>
      %102 = tt.reshape %outRHS_26 : tensor<128x64xf32, #linear11> -> tensor<128x2x32xf32, #linear12>
      %103 = tt.trans %102 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear12> -> tensor<128x32x2xf32, #linear13>
      %outLHS_33, %outRHS_34 = tt.split %103 : tensor<128x32x2xf32, #linear13> -> tensor<128x32xf32, #linear14>
      %104 = tt.reshape %outLHS_33 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked9>
      %105 = tt.trans %104 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked9> -> tensor<128x16x2xf32, #blocked10>
      %outLHS_35, %outRHS_36 = tt.split %105 : tensor<128x16x2xf32, #blocked10> -> tensor<128x16xf32, #blocked1>
      %106 = tt.reshape %outRHS_34 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked9>
      %107 = tt.trans %106 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked9> -> tensor<128x16x2xf32, #blocked10>
      %outLHS_37, %outRHS_38 = tt.split %107 : tensor<128x16x2xf32, #blocked10> -> tensor<128x16xf32, #blocked1>
      %108 = arith.mulf %outLHS_29, %19 : tensor<128x16xf32, #blocked1>
      %109 = arith.truncf %108 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %110 = ttg.local_alloc %109 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %111 = ttng.async_tma_copy_local_to_global %arg41[%40, %c0_i32] %110 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %111   : !ttg.async.token
      %112 = arith.mulf %outRHS_30, %19 : tensor<128x16xf32, #blocked1>
      %113 = arith.truncf %112 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %114 = ttg.local_alloc %113 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %115 = ttng.async_tma_copy_local_to_global %arg41[%40, %c16_i32] %114 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %115   : !ttg.async.token
      %116 = arith.mulf %outLHS_31, %19 : tensor<128x16xf32, #blocked1>
      %117 = arith.truncf %116 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %118 = ttg.local_alloc %117 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %119 = ttng.async_tma_copy_local_to_global %arg41[%40, %c32_i32] %118 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %119   : !ttg.async.token
      %120 = arith.mulf %outRHS_32, %19 : tensor<128x16xf32, #blocked1>
      %121 = arith.truncf %120 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %122 = ttg.local_alloc %121 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %123 = ttng.async_tma_copy_local_to_global %arg41[%40, %c48_i32] %122 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %123   : !ttg.async.token
      %124 = arith.mulf %outLHS_35, %19 : tensor<128x16xf32, #blocked1>
      %125 = arith.truncf %124 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %126 = ttg.local_alloc %125 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %127 = ttng.async_tma_copy_local_to_global %arg41[%40, %c64_i32] %126 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %127   : !ttg.async.token
      %128 = arith.mulf %outRHS_36, %19 : tensor<128x16xf32, #blocked1>
      %129 = arith.truncf %128 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %130 = ttg.local_alloc %129 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %131 = ttng.async_tma_copy_local_to_global %arg41[%40, %c80_i32] %130 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %131   : !ttg.async.token
      %132 = arith.mulf %outLHS_37, %19 : tensor<128x16xf32, #blocked1>
      %133 = arith.truncf %132 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %134 = ttg.local_alloc %133 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %135 = ttng.async_tma_copy_local_to_global %arg41[%40, %c96_i32] %134 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %135   : !ttg.async.token
      %136 = arith.mulf %outRHS_38, %19 : tensor<128x16xf32, #blocked1>
      %137 = arith.truncf %136 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      %138 = ttg.local_alloc %137 : (tensor<128x16xf16, #blocked1>) -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
      %139 = ttng.async_tma_copy_local_to_global %arg41[%40, %c112_i32] %138 : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %139   : !ttg.async.token
      %140 = arith.addi %arg64, %6 : i32
      scf.yield %140 : i32
    } {tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
