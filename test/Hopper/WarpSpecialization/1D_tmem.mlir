// RUN: triton-opt %s -split-input-file --nvgpu-test-1D-tmem-alloc | FileCheck %s

// CHECK-LABEL: @_attn_fwd_persist

module attributes {ttg.maxnreg = 168 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_persist(%arg0: f32, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: !tt.tensordesc<tensor<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg5: i32, %arg6: i32, %arg7: i64, %arg8: i64, %arg9: !tt.tensordesc<tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg10: i32, %arg11: i32, %arg12: i64, %arg13: i64, %arg14: !tt.tensordesc<tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg15: i32, %arg16: i32, %arg17: i64, %arg18: i64, %arg19: !tt.tensordesc<tensor<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg20: i32, %arg21: i32, %arg22: i64, %arg23: i64, %arg24: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // Verify two new tmem_allocs are allocated on the top
    // CHECK: arith.constant false
    // CHECK: ttng.tmem_alloc
    // CHECK: ttng.tmem_alloc
    %false = arith.constant false
    %true = arith.constant true
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c64_i32 = arith.constant 64 : i32
    %cst_1 = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
    %0 = arith.addi %arg24, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %2 = tt.get_program_id x : i32
    %3 = tt.get_num_programs x : i32
    %4 = arith.muli %1, %arg2 : i32
    %5 = arith.muli %4, %arg3 : i32
    %6 = arith.divsi %5, %3 : i32
    %7 = arith.remsi %5, %3 : i32
    %8 = arith.cmpi slt, %2, %7 : i32
    %9 = scf.if %8 -> (i32) {
      %27 = arith.addi %6, %c1_i32 : i32
      scf.yield %27 : i32
    } else {
      scf.yield %6 : i32
    }
    %10 = tt.get_program_id y : i32
    %11 = arith.remsi %10, %arg3 : i32
    %12 = arith.muli %11, %arg24 : i32
    %13 = arith.muli %2, %c128_i32 : i32
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %15 = tt.splat %13 : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %16 = arith.addi %15, %14 : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %17 = tt.make_range {end = 128 : i32, start = 64 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %18 = arith.addi %15, %17 : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %19 = arith.mulf %arg0, %cst : f32
    %20 = tt.splat %19 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
    %21 = tt.splat %19 : f32 -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
    %22 = arith.muli %10, %arg24 : i32
    %23 = tt.addptr %arg1, %22 : !tt.ptr<f32>, i32
    %24 = tt.splat %23 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %25 = tt.addptr %24, %16 : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %26 = tt.addptr %24, %18 : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    scf.for %arg25 = %c0_i32 to %9 step %c1_i32  : i32 {
      // Probably need to mark partition for scalar ops
      %27 = arith.divsi %10, %arg3 : i32
      %28 = arith.addi %27, %12 : i32
      %29 = arith.addi %28, %13 : i32
      // correction in partition 0, softmax in partition 1, 2, gemm in partition 3, load in partition 4, epilogue in partition 5
      %30 = tt.descriptor_load %arg4[%29, %c0_i32] {ttg.partition = 4 : i32} : !tt.tensordesc<tensor<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
      %31 = ttg.local_alloc %30 {ttg.partition = 4 : i32} : (tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>) -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> // q0
      %32 = arith.addi %29, %c64_i32 : i32
      %33 = tt.descriptor_load %arg4[%32, %c0_i32] {ttg.partition = 4 : i32} : !tt.tensordesc<tensor<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
      %34 = ttg.local_alloc %33 {ttg.partition = 4 : i32} : (tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>) -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> // q1
      // Should we lift out the tmem_alloc?
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>, !ttg.async.token) // qk0
      %result_3, %token_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>, !ttg.async.token) // acc0
      %result_5, %token_6 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>, !ttg.async.token) // qk1
      %result_7, %token_8 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>, !ttg.async.token) // acc1
      // TODO: fix this later
      %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %35 = ttng.tmem_store %cst_0, %result_7[%token_8], %true : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
      %36 = ttng.tmem_store %cst_0, %result_3[%token_4], %true : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
      %37:9 = scf.for %arg26 = %c0_i32 to %arg24 step %c128_i32 iter_args(%arg27 = %cst_2, %arg28 = %cst_2, %arg29 = %cst_1, %arg30 = %cst_1, %arg31 = %28, %arg32 = %token, %arg33 = %36, %arg34 = %token_6, %arg35 = %35) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %54 = tt.descriptor_load %arg9[%arg31, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = 4 : i32} : !tt.tensordesc<tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
        %55 = ttg.local_alloc %54 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : (tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>) -> !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> // k
        // Used by gemm partition 3
        %56 = ttg.memdesc_trans %55 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = 3 : i32} : !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>
        %57 = tt.descriptor_load %arg14[%arg31, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = 4 : i32} : !tt.tensordesc<tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
        %58 = ttg.local_alloc %57 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : (tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>) -> !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> // v
        // consumer of 2nd channel: %31/q0
        %59 = ttng.tc_gen5_mma %31, %56, %result[%arg32], %false, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
        // First softmax in partition 1
        // consumer of 1st channel: qk0
        %result_13, %token_14 = ttng.tmem_load %result[%59] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %60 = "tt.reduce"(%result_13) <{axis = 1 : i32}> ({
        ^bb0(%arg36: f32, %arg37: f32):
          %116 = arith.maxnumf %arg36, %arg37 : f32
          tt.reduce.return %116 : f32
        }) {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %61 = arith.mulf %60, %20 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %62 = arith.maxnumf %arg29, %61 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %63 = arith.mulf %result_13, %21 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %64 = tt.expand_dims %62 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %65 = tt.broadcast %64 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %66 = arith.subf %63, %65 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %67 = math.exp2 %66 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %68 = arith.subf %arg29, %62 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        // CHECK-NOT: tmem.start
        %69 = math.exp2 %68 {tmem.start = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        // CHECK: tt.expand_dims
        // CHECK: ttg.convert_layout
        // CHECK: ttng.tmem_store
        // CHECK: tt.reduce
        %70 = "tt.reduce"(%67) <{axis = 1 : i32}> ({
        ^bb0(%arg36: f32, %arg37: f32):
          %116 = arith.addf %arg36, %arg37 : f32
          tt.reduce.return %116 : f32
        }) {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        // Correction in partition 0
        %result_15, %token_16 = ttng.tmem_load %result_3[%arg33] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %71 = tt.reshape %result_15 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>>
        %72 = tt.trans %71 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>, ttg.partition = 0 : i32} : tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>> -> tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>>
        %73 = ttg.convert_layout %72 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>> -> tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>>
        %outLHS, %outRHS = tt.split %73 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>> -> tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        // consumer of %69 (alpha) in correction
        // CHECK: ttng.tmem_load
        // CHECK: tt.reshape
        // CHECK: ttg.convert_layout
        // Note: The existing tt.expand_dims should be unchanged.
        // If we want to optimize the IR to optimize out the tt.expand_dims
        // that should be done in a separate pass.
        // CHECK: tt.expand_dims
        %74 = tt.expand_dims %69 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %75 = tt.broadcast %74 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %76 = arith.mulf %outLHS, %75 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %77 = arith.mulf %outRHS, %75 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %78 = tt.join %76, %77 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>>
        %79 = tt.trans %78 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>, ttg.partition = 0 : i32} : tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>> -> tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>>
        %80 = tt.reshape %79 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>> -> tensor<64x128xf32, #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>>
        // Generate p from softmax0
        %81 = arith.truncf %67 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> to tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %result_17 = ttng.tmem_alloc %81 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : (tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>) -> !ttg.memdesc<64x128xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = false>, #ttng.tensor_memory> // p0
        // Save acc from correction
        %82 = ttg.convert_layout %80 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %83 = ttng.tmem_store %82, %result_3[%token_16], %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
        // consumer of p0
        %84 = ttng.tc_gen5_mma %result_17, %58, %result_3[%83], %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<64x128xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = false>, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
        // Calculate l_i in softmax0
        %85 = arith.mulf %arg27, %69 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %86 = arith.addf %85, %70 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        // consumer of q1
        %87 = ttng.tc_gen5_mma %34, %56, %result_5[%arg34], %false, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
        // Second softmax in partition 2
        // consumer of qk1
        %result_18, %token_19 = ttng.tmem_load %result_5[%87] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %88 = "tt.reduce"(%result_18) <{axis = 1 : i32}> ({
        ^bb0(%arg36: f32, %arg37: f32):
          %116 = arith.maxnumf %arg36, %arg37 : f32
          tt.reduce.return %116 : f32
        }) {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %89 = arith.mulf %88, %20 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %90 = arith.maxnumf %arg30, %89 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %91 = arith.mulf %result_18, %21 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %92 = tt.expand_dims %90 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %93 = tt.broadcast %92 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %94 = arith.subf %91, %93 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %95 = math.exp2 %94 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %96 = arith.subf %arg30, %90 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        // CHECK-NOT: tmem.start
        %97 = math.exp2 %96 {tmem.start = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        // CHECK: tt.expand_dims
        // CHECK: ttg.convert_layout
        // CHECK: ttng.tmem_store
        // CHECK: tt.reduce
        %98 = "tt.reduce"(%95) <{axis = 1 : i32}> ({
        ^bb0(%arg36: f32, %arg37: f32):
          %116 = arith.addf %arg36, %arg37 : f32
          tt.reduce.return %116 : f32
        }) {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        // Correction
        %result_20, %token_21 = ttng.tmem_load %result_7[%arg35] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %99 = tt.reshape %result_20 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>>
        %100 = tt.trans %99 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>, ttg.partition = 0 : i32} : tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>> -> tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>>
        %101 = ttg.convert_layout %100 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>> -> tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>>
        %outLHS_22, %outRHS_23 = tt.split %101 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>> -> tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        // consumer of alpha in correction
        // CHECK: ttng.tmem_load
        // CHECK: tt.reshape
        // CHECK: ttg.convert_layout
        // Note: The existing tt.expand_dims should be unchanged.
        // If we want to optimize the IR to optimize out the tt.expand_dims
        // that should be done in a separate pass.
        // CHECK: tt.expand_dims
        %102 = tt.expand_dims %97 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %103 = tt.broadcast %102 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %104 = arith.mulf %outLHS_22, %103 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %105 = arith.mulf %outRHS_23, %103 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %106 = tt.join %104, %105 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>>
        %107 = tt.trans %106 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>, ttg.partition = 0 : i32} : tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>> -> tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>>
        %108 = tt.reshape %107 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>> -> tensor<64x128xf32, #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>>
        // In softmax1 to emit p
        %109 = arith.truncf %95 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> to tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %result_24 = ttng.tmem_alloc %109 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : (tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>) -> !ttg.memdesc<64x128xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = false>, #ttng.tensor_memory> // p1
        // Save acc after correction
        %110 = ttg.convert_layout %108 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %111 = ttng.tmem_store %110, %result_7[%token_21], %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
        // consumer of p1
        %112 = ttng.tc_gen5_mma %result_24, %58, %result_7[%111], %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<64x128xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = false>, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
        // In Softmax1 to emit l_i
        %113 = arith.mulf %arg28, %97 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %114 = arith.addf %113, %98 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %115 = arith.addi %arg31, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : i32
        scf.yield %86, %114, %62, %90, %115, %token_14, %84, %token_19, %112 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {tt.disallow_acc_multi_buffer, tt.scheduled_max_stage = 2 : i32}
      // Part of the epilogue is in correction
      // consumer of l_i in correction
      %38 = math.log2 %37#0 {ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
      // consumer of a channel: %37#2 m_i0
      %39 = arith.addf %37#2, %38 {ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
      // consumer of l_i
      %40 = tt.expand_dims %37#0 {axis = 1 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %41 = tt.broadcast %40 {ttg.partition = 0 : i32} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      // consumer of acc in correction_epilogue
      %result_9, %token_10 = ttng.tmem_load %result_3[%37#6] {ttg.partition = 0 : i32} : !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %42 = arith.divf %result_9, %41 {ttg.partition = 0 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %43 = ttg.convert_layout %39 {ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      tt.store %25, %43 {ttg.partition = 0 : i32} : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      %44 = arith.truncf %42 {ttg.partition = 0 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> to tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %45 = ttg.convert_layout %44 {ttg.partition = 0 : i32} : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
      // Code partitioning will need to create a channel to save %45 in smem
      // consumer of output from TMA store
      tt.descriptor_store %arg19[%29, %c0_i32], %45 {ttg.partition = 5 : i32} : !tt.tensordesc<tensor<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
      // consumer of l_i
      %46 = math.log2 %37#1 {ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
      // consumer of a channel %37#3 m_i1
      %47 = arith.addf %37#3, %46 {ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
      // consumer of l_i
      %48 = tt.expand_dims %37#1 {axis = 1 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %49 = tt.broadcast %48 {ttg.partition = 0 : i32} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      // consumer of acc in correction epilogue
      %result_11, %token_12 = ttng.tmem_load %result_7[%37#8] {ttg.partition = 0 : i32} : !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %50 = arith.divf %result_11, %49 {ttg.partition = 0 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %51 = ttg.convert_layout %47 {ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      tt.store %26, %51 {ttg.partition = 0 : i32} : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      %52 = arith.truncf %50 {ttg.partition = 0 : i32} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> to tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %53 = ttg.convert_layout %52 {ttg.partition = 0 : i32} : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
      // consumer of output in tma store
      tt.descriptor_store %arg19[%32, %c0_i32], %53 {ttg.partition = 5 : i32} : !tt.tensordesc<tensor<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// CHECK the ability to reuse result, as specified tmem.start_buffer to
// reuse the same buffer via a reinterpret.
// CHECK-LABEL: @_dummy_repro

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = true, elementBitWidth = 32, CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 520 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @_dummy_repro(%in_desc: !tt.tensordesc<tensor<128xf32, #shared>>, %in_desc_0: i32, %in_desc_1: i64, %out_desc: !tt.tensordesc<tensor<128x1xf32, #shared1>>, %out_desc_2: i32, %out_desc_3: i32, %out_desc_4: i64, %out_desc_5: i64) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %result, %token = ttng.tmem_alloc {tmem.start_buffer = 0 : i32}  : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: ttng.tmem_subslice
    // CHECK: ttg.memdesc_reinterpret
    %cst = arith.constant dense<3.000000e+00> : tensor<128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %pid = tt.get_program_id x : i32
    %alpha = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    %alpha_6 = ttg.local_alloc {allocation.offset = 512 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %alpha_6, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %alpha_6, 512, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %in_desc[%pid] %alpha, %alpha_6, %true : !tt.tensordesc<tensor<128xf32, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttng.wait_barrier %alpha_6, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %alpha_6 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %alpha_7 = ttg.local_load %alpha : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
    %alpha_i = arith.mulf %alpha_7, %cst : tensor<128xf32, #blocked>
    // CHECK-NOT: tmem.start
    %0 = ttg.convert_layout %alpha_i {tmem.start = 0 : i32, ttg.partition = 0 : i32} : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    // CHECK: tt.expand_dims
    // CHECK: ttng.tmem_store
    // CHECK: ttng.tmem_load
    // CHECK: tt.reshape
    // CHECK: ttg.convert_layout
    // CHECK: tt.expand_dims
    %1 = tt.expand_dims %0 {axis = 1 : i32, ttg.partition = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
    %2 = ttg.local_alloc %1 {allocation.offset = 0 : i32} : (tensor<128x1xf32, #blocked1>) -> !ttg.memdesc<128x1xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %out_desc[%pid, %c0_i32] %2 : !tt.tensordesc<tensor<128x1xf32, #shared1>>, !ttg.memdesc<128x1xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}


// -----

// CHECK the ability to handle generating
// CHECK-LABEL: @_dummy_repro_expand_dims

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = true, elementBitWidth = 32, CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 520 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @_dummy_repro_expand_dims(%in_desc: !tt.tensordesc<tensor<128xf32, #shared>>, %in_desc_0: i32, %in_desc_1: i64, %out_desc: !tt.tensordesc<tensor<128x1xf32, #shared1>>, %out_desc_2: i32, %out_desc_3: i32, %out_desc_4: i64, %out_desc_5: i64) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %result, %token = ttng.tmem_alloc {tmem.start_buffer = 0 : i32}  : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: ttng.tmem_subslice
    // CHECK: ttg.memdesc_reinterpret
    %cst = arith.constant dense<3.000000e+00> : tensor<128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %pid = tt.get_program_id x : i32
    %alpha = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    %alpha_6 = ttg.local_alloc {allocation.offset = 512 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %alpha_6, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %alpha_6, 512, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %in_desc[%pid] %alpha, %alpha_6, %true : !tt.tensordesc<tensor<128xf32, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
    ttng.wait_barrier %alpha_6, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %alpha_6 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %alpha_7 = ttg.local_load %alpha : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
    %alpha_i = arith.mulf %alpha_7, %cst : tensor<128xf32, #blocked>
    // CHECK-NOT: tmem.start
    %0 = ttg.convert_layout %alpha_i {tmem.start = 0 : i32, ttg.partition = 0 : i32} : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    // CHECK: tt.expand_dims
    // CHECK: ttng.tmem_store
    // CHECK: ttng.tmem_load
    // CHECK: tt.reshape
    // CHECK: ttg.convert_layout
    // CHECK: tt.expand_dims
    %2 = ttg.local_alloc %alpha_i {allocation.offset = 0 : i32, ttg.partition = 1 : i32} : (tensor<128xf32, #blocked1>) -> !ttg.memdesc<128xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %out_desc[%pid] %2 : !tt.tensordesc<tensor<128xf32, #shared1>>, !ttg.memdesc<128xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}
