// RUN: triton-opt %s -split-input-file --nvgpu-partition-scheduling-meta="separate-epilogue-store" | FileCheck %s

// An epilogue store whose value does not depend on the accumulator (x * 2
// loaded through a pointer) is not reachable from any partition, so partition
// propagation used to leave it unscheduled. It must be scheduled with the
// accumulator store, both when it has its own mask and when it shares the
// accumulator store's mask.

// CHECK-LABEL: @side_store_own_mask
// CHECK: tt.store {{.*}} {ttg.partition = array<i32: [[P:[0-9]+]]>}
// CHECK: tt.load
// CHECK: tt.store {{.*}} {ttg.partition = array<i32: [[P]]>}
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @side_store_own_mask(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<131072> : tensor<128x128xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<128x1xi32, #blocked>
    %cst_1 = arith.constant dense<128> : tensor<1x128xi32, #blocked>
    %cst_2 = arith.constant dense<1024> : tensor<128x1xi32, #blocked>
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_3 = arith.constant dense<2.000000e+00> : tensor<128x128xbf16, #blocked>
    %true = arith.constant true
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = tt.make_tensor_descriptor %arg0, [%c1024_i32, %c2048_i32], [%c2048_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %1 = tt.make_tensor_descriptor %arg1, [%c2048_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %2 = tt.get_program_id x : i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %6 = arith.cmpi slt, %5, %cst_1 : tensor<1x128xi32, #blocked>
    %7 = tt.broadcast %6 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %8 = tt.broadcast %5 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %9 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    %10 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    %11 = tt.splat %arg4 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    scf.for %arg5 = %2 to %c8_i32 step %c148_i32  : i32 {
      %12 = arith.muli %arg5, %c128_i32 : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %13 = ttng.tmem_store %cst_4, %result[%token], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %14:2 = scf.for %arg6 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg7 = %false, %arg8 = %13) -> (i1, !ttg.async.token)  : i32 {
        %32 = arith.muli %arg6, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %33 = tt.descriptor_load %0[%12, %32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
        %34 = ttg.local_alloc %33 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %35 = tt.descriptor_load %1[%32, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
        %36 = ttg.local_alloc %35 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %37 = ttng.tc_gen5_mma %34, %36, %result[%arg8], %arg7, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %37 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %15 = tt.splat %12 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %16 = arith.addi %15, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = tt.expand_dims %16 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %18 = arith.cmpi slt, %17, %cst_2 : tensor<128x1xi32, #blocked>
      %19 = tt.broadcast %18 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %20 = arith.andi %19, %7 : tensor<128x128xi1, #blocked>
      %21 = arith.muli %17, %cst_0 : tensor<128x1xi32, #blocked>
      %22 = tt.broadcast %21 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %23 = arith.addi %22, %8 : tensor<128x128xi32, #blocked>
      %24 = tt.addptr %9, %23 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %result_5, %token_6 = ttng.tmem_load %result[%14#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %25 = arith.truncf %result_5 : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %26 = ttg.convert_layout %25 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked>
      tt.store %24, %26, %20 : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %27 = tt.addptr %10, %23 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %28 = tt.load %27, %20 : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %29 = tt.addptr %11, %23 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %30 = arith.mulf %28, %cst_3 : tensor<128x128xbf16, #blocked>
      %31 = arith.cmpi slt, %23, %cst : tensor<128x128xi32, #blocked>
      tt.store %29, %30, %31 : tensor<128x128x!tt.ptr<bf16>, #blocked>
    } {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize}
    tt.return
  }
}


// -----

// CHECK-LABEL: @side_store_shared_mask
// CHECK: tt.store {{.*}} {ttg.partition = array<i32: [[P:[0-9]+]]>}
// CHECK: tt.load
// CHECK: tt.store {{.*}} {ttg.partition = array<i32: [[P]]>}
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @side_store_shared_mask(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<131072> : tensor<128x128xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<128x1xi32, #blocked>
    %cst_1 = arith.constant dense<128> : tensor<1x128xi32, #blocked>
    %cst_2 = arith.constant dense<1024> : tensor<128x1xi32, #blocked>
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c2048_i64 = arith.constant 2048 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_3 = arith.constant dense<2.000000e+00> : tensor<128x128xbf16, #blocked>
    %true = arith.constant true
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = tt.make_tensor_descriptor %arg0, [%c1024_i32, %c2048_i32], [%c2048_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %1 = tt.make_tensor_descriptor %arg1, [%c2048_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %2 = tt.get_program_id x : i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %6 = arith.cmpi slt, %5, %cst_1 : tensor<1x128xi32, #blocked>
    %7 = tt.broadcast %6 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %8 = tt.broadcast %5 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %9 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    %10 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    %11 = tt.splat %arg4 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    scf.for %arg5 = %2 to %c8_i32 step %c148_i32  : i32 {
      %12 = arith.muli %arg5, %c128_i32 : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %13 = ttng.tmem_store %cst_4, %result[%token], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %14:2 = scf.for %arg6 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg7 = %false, %arg8 = %13) -> (i1, !ttg.async.token)  : i32 {
        %32 = arith.muli %arg6, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %33 = tt.descriptor_load %0[%12, %32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
        %34 = ttg.local_alloc %33 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %35 = tt.descriptor_load %1[%32, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked>
        %36 = ttg.local_alloc %35 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %37 = ttng.tc_gen5_mma %34, %36, %result[%arg8], %arg7, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %37 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %15 = tt.splat %12 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %16 = arith.addi %15, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = tt.expand_dims %16 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %18 = arith.cmpi slt, %17, %cst_2 : tensor<128x1xi32, #blocked>
      %19 = tt.broadcast %18 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %20 = arith.andi %19, %7 : tensor<128x128xi1, #blocked>
      %21 = arith.muli %17, %cst_0 : tensor<128x1xi32, #blocked>
      %22 = tt.broadcast %21 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %23 = arith.addi %22, %8 : tensor<128x128xi32, #blocked>
      %24 = tt.addptr %9, %23 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %result_5, %token_6 = ttng.tmem_load %result[%14#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %25 = arith.truncf %result_5 : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %26 = ttg.convert_layout %25 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked>
      tt.store %24, %26, %20 : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %27 = tt.addptr %10, %23 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %28 = tt.load %27, %20 : tensor<128x128x!tt.ptr<bf16>, #blocked>
      %29 = tt.addptr %11, %23 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
      %30 = arith.mulf %28, %cst_3 : tensor<128x128xbf16, #blocked>
      tt.store %29, %30, %20 : tensor<128x128x!tt.ptr<bf16>, #blocked>
    } {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize}
    tt.return
  }
}
