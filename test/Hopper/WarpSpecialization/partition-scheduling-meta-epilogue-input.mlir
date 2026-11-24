// RUN: triton-opt %s -split-input-file --nvgpu-partition-scheduling-meta | FileCheck %s

// A per-tile epilogue input loaded by TMA (it never reaches the MMA) is
// combined with the accumulator. Its users stay in the partition that reads
// the accumulator instead of going to the default partition (fused bmm +
// batch-sum epilogue, reduced from Inductor). The loop's own yield gets no
// partition.

// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[ACC:[0-9]+]]>
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: {{[0-9]+}}>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[ACC]]>
// CHECK: arith.addf {{.*}}ttg.partition = array<i32: [[ACC]]>
// CHECK: tt.store {{.*}}ttg.partition = array<i32: [[ACC]]>
// CHECK: scf.yield %

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 64]], warp = [[0, 16, 0], [0, 32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, rank = 3}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, ttg.use_meta_ws = true, "ttng.two-ctas" = false} {
  tt.func public @triton_(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c12288_i32 = arith.constant 12288 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c128_i32 = arith.constant 128 : i32
    %c96_i32 = arith.constant 96 : i32
    %c0_i32 = arith.constant 0 : i32
    %c96_i64 = arith.constant 96 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c8_i32 = arith.constant 8 : i32
    %c148_i32 = arith.constant 148 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<96> : tensor<1x64x1xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<1x1x128xi32, #blocked>
    %cst_1 = arith.constant dense<128> : tensor<1x64x1xi32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<1x64x128xf32, #blocked>
    %0 = tt.make_tensor_descriptor %arg3, [%arg4, %c96_i32, %c128_i32], [%c12288_i64, %c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x64x128xbf16, #shared>
    %1 = tt.make_tensor_descriptor %arg2, [%arg4, %c96_i32, %c128_i32], [%c12288_i64, %c128_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<1x64x128xf32, #shared1>
    %2 = tt.make_tensor_descriptor %arg0, [%arg4, %c128_i32, %c96_i32], [%c12288_i64, %c96_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x128x64xbf16, #shared>
    %3 = tt.make_tensor_descriptor %arg1, [%arg4, %c128_i32, %c128_i32], [%c16384_i64, %c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x128x128xbf16, #shared>
    %4 = tt.get_program_id x : i32
    %5 = arith.muli %arg4, %c2_i32 : i32
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %8 = tt.bitcast %arg5 : !tt.ptr<i8> -> !tt.ptr<f32>
    %9 = arith.divsi %4, %c2_i32 : i32
    %10 = arith.muli %9, %c12288_i32 : i32
    %11 = tt.addptr %8, %10 : !tt.ptr<f32>, i32
    %12 = tt.splat %11 : !tt.ptr<f32> -> tensor<1x1x128x!tt.ptr<f32>, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %13:2 = scf.for %arg6 = %4 to %5 step %c148_i32 iter_args(%arg7 = %cst_2, %arg8 = %token) -> (tensor<1x64x128xf32, #blocked>, !ttg.async.token)  : i32 {
      %14 = arith.divsi %arg6, %c2_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %15 = arith.remsi %arg6, %c2_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %16 = arith.divsi %15, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %17 = arith.muli %16, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %18 = arith.subi %c2_i32, %17 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %19 = arith.minsi %18, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %20 = arith.remsi %15, %19 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %21 = arith.addi %17, %20 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %22 = arith.remsi %15, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %23 = arith.divsi %22, %19 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %24 = arith.muli %21, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %25 = arith.muli %23, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %26 = tt.descriptor_load %2[%14, %c0_i32, %24] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<1x128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked1>
      %27 = tt.descriptor_load %3[%14, %c0_i32, %25] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<1x128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked2>
      %28 = ttg.local_alloc %27 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
      %29 = ttg.local_alloc %26 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
      %30 = ttg.memdesc_trans %29 {loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared2, #smem> -> !ttg.memdesc<64x128xbf16, #shared3, #smem>
      %31 = ttng.tc_gen5_mma %30, %28, %result[%arg8], %false, %true {loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared3, #smem>, !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_3, %token_4 = ttng.tmem_load %result[%31] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
      %32 = tt.reshape %result_3 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x128xf32, #linear> -> tensor<1x64x128xf32, #linear1>
      %33 = tt.splat %24 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
      %34 = arith.addi %33, %6 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
      %35 = tt.expand_dims %34 {axis = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
      %36 = tt.expand_dims %35 {axis = 2 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x64x1xi32, #blocked>
      %37 = arith.cmpi slt, %36, %cst {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x1xi32, #blocked>
      %38 = tt.splat %25 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
      %39 = arith.addi %38, %7 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
      %40 = tt.expand_dims %39 {axis = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %41 = tt.expand_dims %40 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x128xi32, #blocked>
      %42 = arith.cmpi slt, %41, %cst_0 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x1x128xi32, #blocked>
      %43 = arith.truncf %32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #linear1> to tensor<1x64x128xbf16, #linear1>
      %44 = arith.extf %43 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xbf16, #linear1> to tensor<1x64x128xf32, #linear1>
      %45 = ttg.convert_layout %44 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #linear1> -> tensor<1x64x128xf32, #blocked>
      %46 = arith.truncf %44 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #linear1> to tensor<1x64x128xbf16, #linear1>
      %47 = ttg.convert_layout %46 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xbf16, #linear1> -> tensor<1x64x128xbf16, #blocked3>
      %48 = ttg.local_alloc %47 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<1x64x128xbf16, #blocked3>) -> !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>
      %49 = ttng.async_tma_copy_local_to_global %0[%14, %24, %25] %48 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.tensordesc<1x64x128xbf16, #shared>, !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %49   {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.async.token
      %50 = tt.descriptor_load %1[%14, %24, %25] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.tensordesc<1x64x128xf32, #shared1> -> tensor<1x64x128xf32, #blocked>
      %51 = arith.mulf %45, %50 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #blocked>
      %52 = arith.addf %arg7, %51 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #blocked>
      %53 = tt.addptr %12, %41 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x1x128x!tt.ptr<f32>, #blocked>, tensor<1x1x128xi32, #blocked>
      %54 = arith.muli %36, %cst_1 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x1xi32, #blocked>
      %55 = tt.broadcast %53 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x1x128x!tt.ptr<f32>, #blocked> -> tensor<1x64x128x!tt.ptr<f32>, #blocked>
      %56 = tt.broadcast %54 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x1xi32, #blocked> -> tensor<1x64x128xi32, #blocked>
      %57 = tt.addptr %55, %56 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128x!tt.ptr<f32>, #blocked>, tensor<1x64x128xi32, #blocked>
      %58 = tt.broadcast %37 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x1xi1, #blocked> -> tensor<1x64x128xi1, #blocked>
      %59 = tt.broadcast %42 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x1x128xi1, #blocked> -> tensor<1x64x128xi1, #blocked>
      %60 = arith.andi %58, %59 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xi1, #blocked>
      %61 = arith.addi %arg6, %c148_i32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %62 = arith.cmpi sge, %61, %5 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %63 = tt.splat %62 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i1 -> tensor<1x64x128xi1, #blocked>
      %64 = arith.andi %60, %63 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xi1, #blocked>
      tt.store %57, %52, %64 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128x!tt.ptr<f32>, #blocked>
      scf.yield %52, %token_4 : tensor<1x64x128xf32, #blocked>, !ttg.async.token
    } {tt.data_partition_factor = 1 : i32, tt.scheduled_max_stage = 2 : i32, tt.separate_epilogue_store = true, tt.warp_specialize}
    tt.return
  }
}

// -----

// Same with a bf16 input: the extf between the load and the combine also
// stays with the accumulator rather than in the default partition.

// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[ACC2:[0-9]+]]>
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: {{[0-9]+}}>
// CHECK-NEXT: arith.extf {{.*}}ttg.partition = array<i32: [[ACC2]]>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[ACC2]]>

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 64]], warp = [[0, 16, 0], [0, 32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, rank = 3}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, ttg.use_meta_ws = true, "ttng.two-ctas" = false} {
  tt.func public @triton_(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c12288_i32 = arith.constant 12288 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c128_i32 = arith.constant 128 : i32
    %c96_i32 = arith.constant 96 : i32
    %c0_i32 = arith.constant 0 : i32
    %c96_i64 = arith.constant 96 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c8_i32 = arith.constant 8 : i32
    %c148_i32 = arith.constant 148 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<96> : tensor<1x64x1xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<1x1x128xi32, #blocked>
    %cst_1 = arith.constant dense<128> : tensor<1x64x1xi32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<1x64x128xf32, #blocked>
    %0 = tt.make_tensor_descriptor %arg3, [%arg4, %c96_i32, %c128_i32], [%c12288_i64, %c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x64x128xbf16, #shared>
    %1 = tt.make_tensor_descriptor %arg2, [%arg4, %c96_i32, %c128_i32], [%c12288_i64, %c128_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<1x64x128xf32, #shared1>
    %2 = tt.make_tensor_descriptor %arg0, [%arg4, %c128_i32, %c96_i32], [%c12288_i64, %c96_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x128x64xbf16, #shared>
    %3 = tt.make_tensor_descriptor %arg1, [%arg4, %c128_i32, %c128_i32], [%c16384_i64, %c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x128x128xbf16, #shared>
    %4 = tt.get_program_id x : i32
    %5 = arith.muli %arg4, %c2_i32 : i32
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %8 = tt.bitcast %arg5 : !tt.ptr<i8> -> !tt.ptr<f32>
    %9 = arith.divsi %4, %c2_i32 : i32
    %10 = arith.muli %9, %c12288_i32 : i32
    %11 = tt.addptr %8, %10 : !tt.ptr<f32>, i32
    %12 = tt.splat %11 : !tt.ptr<f32> -> tensor<1x1x128x!tt.ptr<f32>, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %13:2 = scf.for %arg6 = %4 to %5 step %c148_i32 iter_args(%arg7 = %cst_2, %arg8 = %token) -> (tensor<1x64x128xf32, #blocked>, !ttg.async.token)  : i32 {
      %14 = arith.divsi %arg6, %c2_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %15 = arith.remsi %arg6, %c2_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %16 = arith.divsi %15, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %17 = arith.muli %16, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %18 = arith.subi %c2_i32, %17 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %19 = arith.minsi %18, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %20 = arith.remsi %15, %19 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %21 = arith.addi %17, %20 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %22 = arith.remsi %15, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %23 = arith.divsi %22, %19 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %24 = arith.muli %21, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %25 = arith.muli %23, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %26 = tt.descriptor_load %2[%14, %c0_i32, %24] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<1x128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked1>
      %27 = tt.descriptor_load %3[%14, %c0_i32, %25] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<1x128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked2>
      %28 = ttg.local_alloc %27 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
      %29 = ttg.local_alloc %26 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
      %30 = ttg.memdesc_trans %29 {loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared2, #smem> -> !ttg.memdesc<64x128xbf16, #shared3, #smem>
      %31 = ttng.tc_gen5_mma %30, %28, %result[%arg8], %false, %true {loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared3, #smem>, !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_3, %token_4 = ttng.tmem_load %result[%31] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
      %32 = tt.reshape %result_3 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x128xf32, #linear> -> tensor<1x64x128xf32, #linear1>
      %33 = tt.splat %24 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
      %34 = arith.addi %33, %6 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
      %35 = tt.expand_dims %34 {axis = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
      %36 = tt.expand_dims %35 {axis = 2 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x64x1xi32, #blocked>
      %37 = arith.cmpi slt, %36, %cst {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x1xi32, #blocked>
      %38 = tt.splat %25 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
      %39 = arith.addi %38, %7 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
      %40 = tt.expand_dims %39 {axis = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %41 = tt.expand_dims %40 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x128xi32, #blocked>
      %42 = arith.cmpi slt, %41, %cst_0 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x1x128xi32, #blocked>
      %43 = arith.truncf %32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #linear1> to tensor<1x64x128xbf16, #linear1>
      %44 = arith.extf %43 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xbf16, #linear1> to tensor<1x64x128xf32, #linear1>
      %45 = ttg.convert_layout %44 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #linear1> -> tensor<1x64x128xf32, #blocked>
      %46 = arith.truncf %44 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #linear1> to tensor<1x64x128xbf16, #linear1>
      %47 = ttg.convert_layout %46 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xbf16, #linear1> -> tensor<1x64x128xbf16, #blocked3>
      %48 = ttg.local_alloc %47 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<1x64x128xbf16, #blocked3>) -> !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>
      %49 = ttng.async_tma_copy_local_to_global %0[%14, %24, %25] %48 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.tensordesc<1x64x128xbf16, #shared>, !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %49   {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.async.token
      %50 = tt.descriptor_load %0[%14, %24, %25] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.tensordesc<1x64x128xbf16, #shared> -> tensor<1x64x128xbf16, #blocked>
      %51 = arith.extf %50 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xbf16, #blocked> to tensor<1x64x128xf32, #blocked>
      %52 = arith.mulf %45, %51 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #blocked>
      %53 = arith.addf %arg7, %52 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #blocked>
      %54 = tt.addptr %12, %41 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x1x128x!tt.ptr<f32>, #blocked>, tensor<1x1x128xi32, #blocked>
      %55 = arith.muli %36, %cst_1 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x1xi32, #blocked>
      %56 = tt.broadcast %54 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x1x128x!tt.ptr<f32>, #blocked> -> tensor<1x64x128x!tt.ptr<f32>, #blocked>
      %57 = tt.broadcast %55 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x1xi32, #blocked> -> tensor<1x64x128xi32, #blocked>
      %58 = tt.addptr %56, %57 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128x!tt.ptr<f32>, #blocked>, tensor<1x64x128xi32, #blocked>
      %59 = tt.broadcast %37 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x1xi1, #blocked> -> tensor<1x64x128xi1, #blocked>
      %60 = tt.broadcast %42 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x1x128xi1, #blocked> -> tensor<1x64x128xi1, #blocked>
      %61 = arith.andi %59, %60 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xi1, #blocked>
      %62 = arith.addi %arg6, %c148_i32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %63 = arith.cmpi sge, %62, %5 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %64 = tt.splat %63 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i1 -> tensor<1x64x128xi1, #blocked>
      %65 = arith.andi %61, %64 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xi1, #blocked>
      tt.store %58, %53, %65 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128x!tt.ptr<f32>, #blocked>
      scf.yield %53, %token_4 : tensor<1x64x128xf32, #blocked>, !ttg.async.token
    } {tt.data_partition_factor = 1 : i32, tt.scheduled_max_stage = 2 : i32, tt.separate_epilogue_store = true, tt.warp_specialize}
    tt.return
  }
}

// -----

// A TMA epilogue input that is never combined with an accumulator keeps the
// default (epilogue, 1) partition.

// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[ACC3:[0-9]+]]>
// CHECK: arith.addf {{.*}}ttg.partition = array<i32: [[ACC3]]>
// CHECK: %[[X:.*]] = tt.descriptor_load
// CHECK-NEXT: arith.mulf %[[X]], %[[X]] {{.*}}ttg.partition = array<i32: 1>
// CHECK-NEXT: tt.store {{.*}}ttg.partition = array<i32: 1>

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 64]], warp = [[0, 16, 0], [0, 32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, rank = 3}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, ttg.use_meta_ws = true, "ttng.two-ctas" = false} {
  tt.func public @triton_(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c12288_i32 = arith.constant 12288 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c128_i32 = arith.constant 128 : i32
    %c96_i32 = arith.constant 96 : i32
    %c0_i32 = arith.constant 0 : i32
    %c96_i64 = arith.constant 96 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c8_i32 = arith.constant 8 : i32
    %c148_i32 = arith.constant 148 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<96> : tensor<1x64x1xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<1x1x128xi32, #blocked>
    %cst_1 = arith.constant dense<128> : tensor<1x64x1xi32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<1x64x128xf32, #blocked>
    %0 = tt.make_tensor_descriptor %arg3, [%arg4, %c96_i32, %c128_i32], [%c12288_i64, %c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x64x128xbf16, #shared>
    %1 = tt.make_tensor_descriptor %arg2, [%arg4, %c96_i32, %c128_i32], [%c12288_i64, %c128_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<1x64x128xf32, #shared1>
    %2 = tt.make_tensor_descriptor %arg0, [%arg4, %c128_i32, %c96_i32], [%c12288_i64, %c96_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x128x64xbf16, #shared>
    %3 = tt.make_tensor_descriptor %arg1, [%arg4, %c128_i32, %c128_i32], [%c16384_i64, %c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x128x128xbf16, #shared>
    %4 = tt.get_program_id x : i32
    %5 = arith.muli %arg4, %c2_i32 : i32
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %8 = tt.bitcast %arg5 : !tt.ptr<i8> -> !tt.ptr<f32>
    %9 = arith.divsi %4, %c2_i32 : i32
    %10 = arith.muli %9, %c12288_i32 : i32
    %11 = tt.addptr %8, %10 : !tt.ptr<f32>, i32
    %12 = tt.splat %11 : !tt.ptr<f32> -> tensor<1x1x128x!tt.ptr<f32>, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %13:2 = scf.for %arg6 = %4 to %5 step %c148_i32 iter_args(%arg7 = %cst_2, %arg8 = %token) -> (tensor<1x64x128xf32, #blocked>, !ttg.async.token)  : i32 {
      %14 = arith.divsi %arg6, %c2_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %15 = arith.remsi %arg6, %c2_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %16 = arith.divsi %15, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %17 = arith.muli %16, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %18 = arith.subi %c2_i32, %17 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %19 = arith.minsi %18, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %20 = arith.remsi %15, %19 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %21 = arith.addi %17, %20 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %22 = arith.remsi %15, %c8_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %23 = arith.divsi %22, %19 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %24 = arith.muli %21, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %25 = arith.muli %23, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %26 = tt.descriptor_load %2[%14, %c0_i32, %24] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<1x128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked1>
      %27 = tt.descriptor_load %3[%14, %c0_i32, %25] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<1x128x128xbf16, #shared> -> tensor<128x128xbf16, #blocked2>
      %28 = ttg.local_alloc %27 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared2, #smem>
      %29 = ttg.local_alloc %26 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
      %30 = ttg.memdesc_trans %29 {loop.cluster = 2 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared2, #smem> -> !ttg.memdesc<64x128xbf16, #shared3, #smem>
      %31 = ttng.tc_gen5_mma %30, %28, %result[%arg8], %false, %true {loop.cluster = 2 : i32, loop.stage = 0 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared3, #smem>, !ttg.memdesc<128x128xbf16, #shared2, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_3, %token_4 = ttng.tmem_load %result[%31] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
      %32 = tt.reshape %result_3 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x128xf32, #linear> -> tensor<1x64x128xf32, #linear1>
      %33 = tt.splat %24 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
      %34 = arith.addi %33, %6 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
      %35 = tt.expand_dims %34 {axis = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
      %36 = tt.expand_dims %35 {axis = 2 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x64x1xi32, #blocked>
      %37 = arith.cmpi slt, %36, %cst {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x1xi32, #blocked>
      %38 = tt.splat %25 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
      %39 = arith.addi %38, %7 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
      %40 = tt.expand_dims %39 {axis = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %41 = tt.expand_dims %40 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x128xi32, #blocked>
      %42 = arith.cmpi slt, %41, %cst_0 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x1x128xi32, #blocked>
      %43 = arith.truncf %32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #linear1> to tensor<1x64x128xbf16, #linear1>
      %44 = arith.extf %43 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xbf16, #linear1> to tensor<1x64x128xf32, #linear1>
      %45 = ttg.convert_layout %44 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #linear1> -> tensor<1x64x128xf32, #blocked>
      %46 = arith.truncf %44 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #linear1> to tensor<1x64x128xbf16, #linear1>
      %47 = ttg.convert_layout %46 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xbf16, #linear1> -> tensor<1x64x128xbf16, #blocked3>
      %48 = ttg.local_alloc %47 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<1x64x128xbf16, #blocked3>) -> !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable>
      %49 = ttng.async_tma_copy_local_to_global %0[%14, %24, %25] %48 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.tensordesc<1x64x128xbf16, #shared>, !ttg.memdesc<1x64x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %49   {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.async.token
      %50 = tt.descriptor_load %1[%14, %24, %25] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.tensordesc<1x64x128xf32, #shared1> -> tensor<1x64x128xf32, #blocked>
      %51 = arith.mulf %45, %50 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #blocked>
      %52 = arith.addf %arg7, %51 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #blocked>
      %53 = tt.addptr %12, %41 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x1x128x!tt.ptr<f32>, #blocked>, tensor<1x1x128xi32, #blocked>
      %54 = arith.muli %36, %cst_1 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x1xi32, #blocked>
      %55 = tt.broadcast %53 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x1x128x!tt.ptr<f32>, #blocked> -> tensor<1x64x128x!tt.ptr<f32>, #blocked>
      %56 = tt.broadcast %54 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x1xi32, #blocked> -> tensor<1x64x128xi32, #blocked>
      %57 = tt.addptr %55, %56 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128x!tt.ptr<f32>, #blocked>, tensor<1x64x128xi32, #blocked>
      %58 = tt.broadcast %37 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x1xi1, #blocked> -> tensor<1x64x128xi1, #blocked>
      %59 = tt.broadcast %42 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x1x128xi1, #blocked> -> tensor<1x64x128xi1, #blocked>
      %60 = arith.andi %58, %59 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xi1, #blocked>
      %61 = arith.addi %arg6, %c148_i32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %62 = arith.cmpi sge, %61, %5 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %63 = tt.splat %62 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i1 -> tensor<1x64x128xi1, #blocked>
      %64 = arith.andi %60, %63 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xi1, #blocked>
      tt.store %57, %52, %64 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128x!tt.ptr<f32>, #blocked>
      %65 = tt.descriptor_load %1[%14, %24, %25] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !tt.tensordesc<1x64x128xf32, #shared1> -> tensor<1x64x128xf32, #blocked>
      %66 = arith.mulf %65, %65 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128xf32, #blocked>
      tt.store %57, %66, %64 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<1x64x128x!tt.ptr<f32>, #blocked>
      scf.yield %52, %token_4 : tensor<1x64x128xf32, #blocked>, !ttg.async.token
    } {tt.data_partition_factor = 1 : i32, tt.scheduled_max_stage = 2 : i32, tt.separate_epilogue_store = true, tt.warp_specialize}
    tt.return
  }
}
