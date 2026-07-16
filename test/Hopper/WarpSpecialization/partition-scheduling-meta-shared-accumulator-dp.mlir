// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-partition-scheduling-meta | FileCheck %s

// Test for the accumulator-chain data-partition grouping.
//
// This HSTU cross-attention backward reduce_dq kernel processes two KV blocks
// per program. Both blocks' dq dots accumulate into ONE shared tensor-memory
// tile (the dq accumulator, encoding #tmem1): b0 fresh-writes it, b1 adds into
// it (use_acc). Data-partition detection used to treat the two blocks as
// independent groups (dpFactor=2) purely from their disjoint compute slices,
// and split their compute into two computation partitions -- inflating the
// partition count and warp budget (18/16, over budget) even though the two
// streams converge on a single serialized reduction and gain no real
// parallelism.
//
// The accumulator-chain grouping unions MMAs that write the same accumulator
// buffer, so the two streams are recognized as one data-partition group
// (dpFactor collapses to 1). The compute merges into a single computation
// partition, the budget fits (14/16), and every MMA -- including both dq dots
// -- shares the gemm partition. (This is a partition-count/budget lever; it is
// NOT a correctness fix for the separate gemm->reduction dq store-reduce
// handoff bug tracked in T279388065.)

// CHECK-LABEL: @_hstu_attn_bwd_redq_2kv
// Both dq MMAs (accumulator memdesc #tmem1) share the same partition:
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[DQP:[0-9]+]]>{{.*}}memdesc<128x64xf32, #tmem1
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[DQP]]>{{.*}}memdesc<128x64xf32, #tmem1

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_hstu_attn_bwd_redq_2kv(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: f32, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg28: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg29: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #linear>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c63_i32 = arith.constant 63 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant 1.44269502 : f32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #linear1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #linear2>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %arg33 : i32
    %2 = arith.remsi %0, %arg33 : i32
    %3 = tt.addptr %arg6, %1 : !tt.ptr<i64>, i32
    %4 = tt.load %3 : !tt.ptr<i64>
    %5 = arith.trunci %4 : i64 to i32
    %6 = tt.addptr %3, %c1_i32 : !tt.ptr<i64>, i32
    %7 = tt.load %6 : !tt.ptr<i64>
    %8 = arith.trunci %7 : i64 to i32
    %9 = arith.subi %8, %5 : i32
    %10 = tt.addptr %arg7, %1 : !tt.ptr<i64>, i32
    %11 = tt.load %10 : !tt.ptr<i64>
    %12 = arith.trunci %11 : i64 to i32
    %13 = tt.addptr %10, %c1_i32 : !tt.ptr<i64>, i32
    %14 = tt.load %13 : !tt.ptr<i64>
    %15 = arith.trunci %14 : i64 to i32
    %16 = arith.subi %15, %12 : i32
    %17 = arith.cmpi eq, %9, %c0_i32 : i32
    cf.cond_br %17, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %18 = arith.muli %arg33, %c128_i32 : i32
    %19 = arith.extsi %18 : i32 to i64
    %20 = tt.make_tensor_descriptor %arg0, [%arg4, %18], [%19, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<64x128xbf16, #shared>>
    %21 = tt.make_tensor_descriptor %arg1, [%arg5, %18], [%19, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<64x128xbf16, #shared>>
    %22 = tt.make_tensor_descriptor %arg3, [%arg4, %18], [%19, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<64x128xbf16, #shared>>
    %23 = tt.make_tensor_descriptor %arg9, [%8, %18], [%19, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<64x128xbf16, #shared>>
    %24 = tt.make_tensor_descriptor %arg8, [%15, %18], [%19, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<tensor<64x128xf32, #shared1>>
    %25 = arith.addi %9, %c63_i32 : i32
    %26 = arith.divsi %25, %c64_i32 : i32
    %27 = arith.addi %26, %c1_i32 : i32
    %28 = arith.divsi %27, %c2_i32 : i32
    %29 = arith.muli %28, %c128_i32 : i32
    %30 = arith.muli %2, %arg14 : i32
    %31 = arith.cmpi slt, %2, %c2_i32 : i32
    %32:2 = scf.if %31 -> (!tt.ptr<f32>, !tt.ptr<f32>) {
      %48 = tt.addptr %arg28, %2 : !tt.ptr<f32>, i32
      %49 = arith.muli %12, %arg30 : i32
      %50 = tt.addptr %48, %49 : !tt.ptr<f32>, i32
      %51 = tt.addptr %arg29, %2 : !tt.ptr<f32>, i32
      %52 = tt.addptr %51, %49 : !tt.ptr<f32>, i32
      scf.yield %52, %50 : !tt.ptr<f32>, !tt.ptr<f32>
    } else {
      scf.yield %arg29, %arg28 : !tt.ptr<f32>, !tt.ptr<f32>
    }
    %33 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
    %34 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>>
    %35 = arith.mulf %arg25, %cst_0 : f32
    %36 = tt.splat %16 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
    %37 = arith.muli %2, %arg12 : i32
    %38 = arith.muli %2, %arg18 : i32
    %39 = tt.splat %16 : i32 -> tensor<1x64xi32, #linear1>
    %40 = tt.splat %9 : i32 -> tensor<64x1xi32, #linear1>
    %41 = tt.splat %35 : f32 -> tensor<64x64xf32, #linear1>
    %42 = tt.splat %arg30 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
    %43 = tt.splat %32#1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear1}>>
    %44 = tt.splat %32#0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear1}>>
    %45 = tt.splat %arg25 : f32 -> tensor<64x64xf32, #linear1>
    %46 = arith.muli %2, %c128_i32 : i32
    %47 = arith.muli %2, %c128_i32 : i32
    scf.for %arg35 = %c0_i32 to %29 step %c128_i32  : i32 {
      %48 = arith.addi %5, %arg35 : i32
      %49 = arith.addi %48, %c64_i32 : i32
      %50 = tt.descriptor_load %21[%48, %30] : !tt.tensordesc<tensor<64x128xbf16, #shared>> -> tensor<64x128xbf16, #blocked>
      %51 = ttg.local_alloc %50 : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
      %52 = tt.descriptor_load %21[%49, %30] : !tt.tensordesc<tensor<64x128xbf16, #shared>> -> tensor<64x128xbf16, #blocked>
      %53 = ttg.local_alloc %52 : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
      %54 = tt.splat %arg35 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>>
      %55 = arith.addi %54, %34 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>>
      %56 = arith.addi %arg35, %c64_i32 : i32
      %57 = tt.splat %56 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>>
      %58 = arith.addi %57, %34 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>>
      %59 = tt.expand_dims %55 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>> -> tensor<64x1xi32, #linear1>
      %60 = arith.cmpi slt, %59, %40 : tensor<64x1xi32, #linear1>
      %61 = tt.broadcast %60 : tensor<64x1xi1, #linear1> -> tensor<64x64xi1, #linear1>
      %62 = tt.expand_dims %58 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>> -> tensor<64x1xi32, #linear1>
      %63 = arith.cmpi slt, %62, %40 : tensor<64x1xi32, #linear1>
      %64 = tt.broadcast %63 : tensor<64x1xi1, #linear1> -> tensor<64x64xi1, #linear1>
      %65 = ttg.memdesc_trans %51 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
      %66 = ttg.memdesc_trans %53 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_3, %token_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_5, %token_6 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_7, %token_8 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_9, %token_10 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_11, %token_12 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %result_13, %token_14 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %67 = ttng.tmem_store %cst_2, %result_13[%token_14], %true : tensor<64x128xf32, #linear2> -> !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
      %68 = ttng.tmem_store %cst_2, %result_11[%token_12], %true : tensor<64x128xf32, #linear2> -> !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
      %69:8 = scf.for %arg36 = %c0_i32 to %16 step %c64_i32 iter_args(%arg37 = %false, %arg38 = %token, %arg39 = %token_4, %arg40 = %token_6, %arg41 = %token_8, %arg42 = %token_10, %arg43 = %68, %arg44 = %67) -> (i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %78 = tt.splat %arg36 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %79 = arith.addi %78, %33 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %80 = arith.cmpi slt, %79, %36 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %81 = arith.addi %12, %arg36 : i32
        %82 = tt.descriptor_load %20[%81, %37] : !tt.tensordesc<tensor<64x128xbf16, #shared>> -> tensor<64x128xbf16, #blocked>
        %83 = ttg.local_alloc %82 : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %84 = tt.descriptor_load %22[%81, %38] : !tt.tensordesc<tensor<64x128xbf16, #shared>> -> tensor<64x128xbf16, #blocked>
        %85 = ttg.local_alloc %84 : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %86 = ttg.memdesc_trans %83 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
        %87 = ttng.tc_gen5_mma %51, %86, %result[%arg38], %false, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,2\22]}"} : !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x64xbf16, #shared2, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %88 = tt.expand_dims %79 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xi32, #linear1>
        %89 = arith.cmpi slt, %88, %39 : tensor<1x64xi32, #linear1>
        %90 = tt.broadcast %89 : tensor<1x64xi1, #linear1> -> tensor<64x64xi1, #linear1>
        %91 = arith.andi %90, %61 : tensor<64x64xi1, #linear1>
        %result_19, %token_20 = ttng.tmem_load %result[%87] : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear1>
        %92 = arith.mulf %result_19, %41 : tensor<64x64xf32, #linear1>
        %93 = arith.muli %79, %42 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %94 = tt.addptr %43, %93 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear1}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %95 = tt.load %94, %80 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear1}>>
        %96 = tt.expand_dims %95 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1>
        %97 = tt.broadcast %96 : tensor<1x64xf32, #linear1> -> tensor<64x64xf32, #linear1>
        %98 = tt.load %94, %80 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear1}>>
        %99 = tt.expand_dims %98 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1>
        %100 = tt.broadcast %99 : tensor<1x64xf32, #linear1> -> tensor<64x64xf32, #linear1>
        %101 = arith.subf %92, %100 : tensor<64x64xf32, #linear1>
        %102 = math.exp2 %101 : tensor<64x64xf32, #linear1>
        %103 = arith.select %91, %102, %cst_1 : tensor<64x64xi1, #linear1>, tensor<64x64xf32, #linear1>
        %104 = arith.truncf %103 : tensor<64x64xf32, #linear1> to tensor<64x64xbf16, #linear1>
        %result_21 = ttng.tmem_alloc %104 : (tensor<64x64xbf16, #linear1>) -> !ttg.memdesc<64x64xbf16, #tmem, #ttng.tensor_memory>
        %105 = ttg.memdesc_trans %85 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
        %106 = ttng.tc_gen5_mma %51, %105, %result_3[%arg39], %false, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndD,tmem,1,5\22]}"} : !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x64xbf16, #shared2, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %107 = tt.addptr %44, %93 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear1}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %108 = tt.load %107, %80 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear1}>>
        %109 = tt.expand_dims %108 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1>
        %110 = tt.broadcast %109 : tensor<1x64xf32, #linear1> -> tensor<64x64xf32, #linear1>
        %result_22, %token_23 = ttng.tmem_load %result_3[%106] : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear1>
        %111 = tt.load %107, %80 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear1}>>
        %112 = tt.expand_dims %111 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x64xf32, #linear1>
        %113 = tt.broadcast %112 : tensor<1x64xf32, #linear1> -> tensor<64x64xf32, #linear1>
        %114 = arith.subf %result_22, %113 : tensor<64x64xf32, #linear1>
        %115 = arith.mulf %103, %114 : tensor<64x64xf32, #linear1>
        %116 = ttng.tc_gen5_mma %53, %86, %result_5[%arg40], %false, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,12\22]}"} : !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x64xbf16, #shared2, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %117 = arith.andi %90, %64 : tensor<64x64xi1, #linear1>
        %result_24, %token_25 = ttng.tmem_load %result_5[%116] : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear1>
        %118 = arith.mulf %result_24, %41 : tensor<64x64xf32, #linear1>
        %119 = arith.subf %118, %97 : tensor<64x64xf32, #linear1>
        %120 = math.exp2 %119 : tensor<64x64xf32, #linear1>
        %121 = arith.select %117, %120, %cst_1 : tensor<64x64xi1, #linear1>, tensor<64x64xf32, #linear1>
        %122 = arith.truncf %121 : tensor<64x64xf32, #linear1> to tensor<64x64xbf16, #linear1>
        %result_26 = ttng.tmem_alloc %122 : (tensor<64x64xbf16, #linear1>) -> !ttg.memdesc<64x64xbf16, #tmem, #ttng.tensor_memory>
        %123 = ttng.tc_gen5_mma %53, %105, %result_7[%arg41], %false, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndD,tmem,1,15\22]}"} : !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x64xbf16, #shared2, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %result_27, %token_28 = ttng.tmem_load %result_7[%123] : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear1>
        %124 = arith.subf %result_27, %110 : tensor<64x64xf32, #linear1>
        %125 = arith.mulf %121, %124 : tensor<64x64xf32, #linear1>
        %126 = arith.mulf %115, %45 : tensor<64x64xf32, #linear1>
        %127 = arith.truncf %126 : tensor<64x64xf32, #linear1> to tensor<64x64xbf16, #linear1>
        %128 = ttg.local_alloc %127 : (tensor<64x64xbf16, #linear1>) -> !ttg.memdesc<64x64xbf16, #shared, #smem>
        %129 = ttng.tmem_store %cst, %result_9[%arg42], %true : tensor<128x64xf32, #linear> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %130 = ttng.tc_gen5_mma %65, %128, %result_9[%129], %false, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndB,smem,1,8\22, \22opndD,tmem,1,11\22]}"} : !ttg.memdesc<128x64xbf16, #shared2, #smem>, !ttg.memdesc<64x64xbf16, #shared, #smem>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %131 = ttng.tc_gen5_mma %result_21, %85, %result_11[%arg43], %arg37, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}"} : !ttg.memdesc<64x64xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
        %132 = ttng.tc_gen5_mma %128, %83, %result_11[%131], %true, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,7\22]}"} : !ttg.memdesc<64x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
        %133 = arith.mulf %125, %45 : tensor<64x64xf32, #linear1>
        %134 = arith.truncf %133 : tensor<64x64xf32, #linear1> to tensor<64x64xbf16, #linear1>
        %135 = ttg.local_alloc %134 : (tensor<64x64xbf16, #linear1>) -> !ttg.memdesc<64x64xbf16, #shared, #smem>
        %136 = ttng.tc_gen5_mma %66, %135, %result_9[%130], %true, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndB,smem,1,9\22, \22opndD,tmem,1,11\22]}"} : !ttg.memdesc<128x64xbf16, #shared2, #smem>, !ttg.memdesc<64x64xbf16, #shared, #smem>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %137 = ttng.tc_gen5_mma %result_26, %85, %result_13[%arg44], %arg37, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,12\22, \22opndD,tmem,1,17\22]}"} : !ttg.memdesc<64x64xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
        %138 = ttng.tc_gen5_mma %135, %83, %result_13[%137], %true, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,9\22, \22opndD,tmem,1,17\22]}"} : !ttg.memdesc<64x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
        %result_29, %token_30 = ttng.tmem_load %result_9[%136] : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear>
        %139 = tt.trans %result_29 {order = array<i32: 1, 0>} : tensor<128x64xf32, #linear> -> tensor<64x128xf32, #linear3>
        %140 = ttg.convert_layout %139 : tensor<64x128xf32, #linear3> -> tensor<64x128xf32, #blocked1>
        %141 = ttg.local_alloc %140 : (tensor<64x128xf32, #blocked1>) -> !ttg.memdesc<64x128xf32, #shared1, #smem, mutable>
        %142 = ttng.async_tma_reduce add, %24[%81, %46] %141 : !tt.tensordesc<tensor<64x128xf32, #shared1>>, !ttg.memdesc<64x128xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %142   : !ttg.async.token
        scf.yield %true, %token_20, %token_23, %token_25, %token_28, %token_30, %132, %138 : i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {tt.list_schedule_pick = 0 : i32, tt.num_stages = 1 : i32, tt.warp_specialize}
      %result_15, %token_16 = ttng.tmem_load %result_11[%69#6] : !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear2>
      %70 = arith.truncf %result_15 : tensor<64x128xf32, #linear2> to tensor<64x128xbf16, #linear2>
      %71 = ttg.convert_layout %70 : tensor<64x128xbf16, #linear2> -> tensor<64x128xbf16, #blocked>
      %72 = ttg.local_alloc %71 : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
      %73 = ttng.async_tma_copy_local_to_global %23[%48, %47] %72 : !tt.tensordesc<tensor<64x128xbf16, #shared>>, !ttg.memdesc<64x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %73   : !ttg.async.token
      %result_17, %token_18 = ttng.tmem_load %result_13[%69#7] : !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear2>
      %74 = arith.truncf %result_17 : tensor<64x128xf32, #linear2> to tensor<64x128xbf16, #linear2>
      %75 = ttg.convert_layout %74 : tensor<64x128xbf16, #linear2> -> tensor<64x128xbf16, #blocked>
      %76 = ttg.local_alloc %75 : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
      %77 = ttng.async_tma_copy_local_to_global %23[%49, %47] %76 : !tt.tensordesc<tensor<64x128xbf16, #shared>>, !ttg.memdesc<64x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %77   : !ttg.async.token
    }
    tt.return
  }
}
