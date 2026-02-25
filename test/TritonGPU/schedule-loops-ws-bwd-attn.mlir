// RUN: triton-opt %s "-tritongpu-schedule-loops=num-stages=2 use-meta-ws=true" | FileCheck %s

// Backward attention kernel with 5 MMA ops in a WS loop with
// tt.disallow_acc_multi_buffer. Verify that schedule-loops preserves the
// expected stage/cluster assignments for descriptor_load, tc_gen5_mma, and
// descriptor_reduce ops.

// CHECK-LABEL: @_attn_bwd

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd(%arg0: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: f32, %arg16: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %arg17: i32, %arg18: i32, %arg19: i64, %arg20: i64, %arg21: !tt.tensordesc<tensor<128x128xf32, #shared1>>, %arg22: i32, %arg23: i32, %arg24: i64, %arg25: i64, %arg26: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %arg27: i32, %arg28: i32, %arg29: i64, %arg30: i64, %arg31: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %arg32: i32, %arg33: i32, %arg34: i64, %arg35: i64, %arg36: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg37: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg38: i32 {tt.divisibility = 16 : i32}, %arg39: i32 {tt.divisibility = 16 : i32}, %arg40: i32 {tt.divisibility = 16 : i32}, %arg41: i32 {tt.divisibility = 16 : i32}, %arg42: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.693147182> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id z : i32
    %1 = arith.muli %0, %arg42 : i32
    %2 = arith.extsi %1 : i32 to i64
    %3 = arith.remsi %0, %arg41 : i32
    %4 = arith.muli %arg39, %3 : i32
    %5 = arith.divsi %0, %arg41 : i32
    %6 = arith.muli %arg38, %5 : i32
    %7 = arith.addi %4, %6 : i32
    %8 = arith.extsi %7 : i32 to i64
    %9 = arith.extsi %arg40 : i32 to i64
    %10 = arith.divsi %8, %9 : i64
    %11 = tt.get_program_id x : i32
    %12 = tt.addptr %arg36, %2 : !tt.ptr<f32>, i64
    %13 = tt.addptr %arg37, %2 : !tt.ptr<f32>, i64
    %14 = arith.muli %11, %c128_i32 : i32
    %15 = arith.extsi %14 : i32 to i64
    %16 = arith.addi %10, %15 : i64
    %17 = arith.trunci %16 : i64 to i32
    %18 = tt.descriptor_load %arg5[%17, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
    %19 = ttg.local_alloc %18 : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    %20 = tt.descriptor_load %arg10[%17, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
    %21 = ttg.local_alloc %20 : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    %22 = arith.divsi %arg42, %c128_i32 : i32
    %23 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %24 = tt.splat %12 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
    %25 = tt.splat %13 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_1, %token_2 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_3, %token_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_5, %token_6 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_7, %token_8 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %26 = ttng.tmem_store %cst_0, %result_5[%token_6], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %27 = ttng.tmem_store %cst_0, %result_1[%token_2], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %28:7 = scf.for %arg43 = %c0_i32 to %22 step %c1_i32 iter_args(%arg44 = %c0_i32, %arg45 = %false, %arg46 = %token, %arg47 = %27, %arg48 = %token_4, %arg49 = %26, %arg50 = %token_8) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %35 = arith.extsi %arg44 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 to i64
      %36 = arith.addi %10, %35 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64
      %37 = arith.trunci %36 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64 to i32
      // q descriptor_load: stage 0, cluster 2
      // CHECK: tt.descriptor_load %arg0{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
      %38 = tt.descriptor_load %arg0[%37, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
      %39 = ttg.local_alloc %38 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %40 = ttg.memdesc_trans %39 {loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
      %41 = tt.splat %arg44 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, #blocked2>
      %42 = arith.addi %41, %23 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xi32, #blocked2>
      %43 = tt.addptr %24, %42 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
      %44 = tt.load %43 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
      // qkT MMA: stage 0, cluster 2
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32}
      %45 = ttng.tc_gen5_mma %19, %40, %result[%arg46], %false, %true {loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %46 = ttg.convert_layout %44 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %47 = tt.expand_dims %46 {axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xf32, #blocked>
      %48 = tt.broadcast %47 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %result_13, %token_14 = ttng.tmem_load %result[%45] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %49 = arith.subf %result_13, %48 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked>
      %50 = math.exp2 %49 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked>
      // do descriptor_load: stage 0, cluster 2
      // CHECK: tt.descriptor_load %arg16{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
      %51 = tt.descriptor_load %arg16[%37, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
      %52 = ttg.local_alloc %51 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %53 = arith.truncf %50 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
      %result_15 = ttng.tmem_alloc %53 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>
      // dv MMA: stage 1, cluster 0
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32}
      %54 = ttng.tc_gen5_mma %result_15, %52, %result_1[%arg47], %arg45, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %55 = tt.addptr %25, %42 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
      %56 = tt.load %55 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
      %57 = ttg.memdesc_trans %52 {loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
      // dpT MMA: stage 0, cluster 2
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32}
      %58 = ttng.tc_gen5_mma %21, %57, %result_3[%arg48], %false, %true {loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %59 = ttg.convert_layout %56 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %60 = tt.expand_dims %59 {axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xf32, #blocked>
      %61 = tt.broadcast %60 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %result_16, %token_17 = ttng.tmem_load %result_3[%58] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %62 = arith.subf %result_16, %61 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked>
      %63 = arith.mulf %50, %62 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked>
      %64 = arith.truncf %63 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
      %result_18 = ttng.tmem_alloc %64 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>
      // dk MMA: stage 1, cluster 0
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32}
      %65 = ttng.tc_gen5_mma %result_18, %39, %result_5[%arg49], %arg45, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %66 = ttg.local_alloc %64 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
      %67 = ttg.memdesc_trans %66 {loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared2, #smem> -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      // dq MMA: stage 0, cluster 2
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32}
      %68 = ttng.tc_gen5_mma %67, %19, %result_7[%arg50], %false, %true {loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_19, %token_20 = ttng.tmem_load %result_7[%68] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %69 = arith.mulf %result_19, %cst {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
      %70 = ttg.convert_layout %69 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked1>
      // descriptor_reduce: stage 1, cluster 0
      // CHECK: tt.descriptor_reduce add, %arg21{{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32}
      tt.descriptor_reduce add, %arg21[%37, %c0_i32], %70 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x128xf32, #shared1>>, tensor<128x128xf32, #blocked1>
      %71 = arith.addi %arg44, %c128_i32 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : i32
      scf.yield %71, %true, %token_14, %54, %token_17, %65, %token_20 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
    %result_9, %token_10 = ttng.tmem_load %result_1[%28#3] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %result_11, %token_12 = ttng.tmem_load %result_5[%28#5] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %29 = arith.truncf %result_9 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
    %30 = ttg.convert_layout %29 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked1>
    tt.descriptor_store %arg31[%17, %c0_i32], %30 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, tensor<128x128xbf16, #blocked1>
    %31 = tt.splat %arg15 : f32 -> tensor<128x128xf32, #blocked>
    %32 = arith.mulf %result_11, %31 : tensor<128x128xf32, #blocked>
    %33 = arith.truncf %32 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
    %34 = ttg.convert_layout %33 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked1>
    tt.descriptor_store %arg26[%17, %c0_i32], %34 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, tensor<128x128xbf16, #blocked1>
    tt.return
  }
}
