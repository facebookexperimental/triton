// RUN: triton-opt %s -split-input-file --nvgpu-test-taskid-propagate="num-warp-groups=3" --nvgpu-test-ws-code-partition="num-buffers=1 post-channel-creation=1" | FileCheck %s
// CHECK-LABEL: _attn_fwd_persist
// CHECK: ttg.warp_specialize
// CHECK: default
// CHECK: partition0{{.*}}num_warps(4)
// CHECK: partition1{{.*}}num_warps(4)
// CHECK: partition2{{.*}}num_warps(4)
// CHECK: partition3{{.*}}num_warps(4)
// CHECK: partition4{{.*}}num_warps(4)

module attributes {ttg.maxnreg = 168 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_persist(%arg0: f32, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: !tt.tensordesc<tensor<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg5: i32, %arg6: i32, %arg7: i64, %arg8: i64, %arg9: !tt.tensordesc<tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg10: i32, %arg11: i32, %arg12: i64, %arg13: i64, %arg14: !tt.tensordesc<tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg15: i32, %arg16: i32, %arg17: i64, %arg18: i64, %arg19: !tt.tensordesc<tensor<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg20: i32, %arg21: i32, %arg22: i64, %arg23: i64, %arg24: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %31 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
    %34 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
    %55 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> // k
    %58 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> // v

    %out0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
    %out1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>

    %tmem_qk0, %token = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> (!ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token) // qk0
    %tmem_acc0, %token_4 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 6 : i32} : () -> (!ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token) // acc0
    %tmem_qk1, %token_6 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> (!ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token) // qk1
    %tmem_acc1, %token_8 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> (!ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token) // acc1

    %tmem_p0, %token_p0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<64x128xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token) // p0
    %tmem_p1, %token_p1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<64x128xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token) // p1

    // alpha/l_i/m_i/output
    %alpha0, %token_alpha0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 64 : i32} : () -> (!ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %alpha1, %token_alpha1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 64 : i32} : () -> (!ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %l_i0, %token_li0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 65 : i32} : () -> (!ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %l_i1, %token_li1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 65 : i32} : () -> (!ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m_i0, %token_mi0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 66 : i32} : () -> (!ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m_i1, %token_mi1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32, buffer.offset = 66 : i32} : () -> (!ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token)


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

    %19 = arith.mulf %arg0, %cst : f32

    %22 = arith.muli %10, %arg24 : i32
    %23 = tt.addptr %arg1, %22 : !tt.ptr<f32>, i32

    scf.for %arg25 = %c0_i32 to %9 step %c1_i32  : i32 {
      // Probably need to mark partition for scalar ops
      %27 = arith.divsi %10, %arg3 {ttg.partition = array<i32: 4>} : i32
      %28 = arith.addi %27, %12 {ttg.partition = array<i32: 4>} : i32
      %29 = arith.addi %28, %13 {ttg.partition = array<i32: 4>} : i32
      %527 = arith.divsi %10, %arg3 {ttg.partition = array<i32: 3>} : i32
      %528 = arith.addi %527, %12 {ttg.partition = array<i32: 3>} : i32
      %529 = arith.addi %528, %13 {ttg.partition = array<i32: 3>} : i32
      // correction in partition 0, softmax in partition 1, 2, gemm in partition 3, load in partition 4, epilogue in partition 5
      %30 = tt.descriptor_load %arg4[%29, %c0_i32] {ttg.partition = array<i32: 4>} : !tt.tensordesc<tensor<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
      ttg.local_store %30, %31 {ttg.partition = array<i32: 4>} : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> // q0
      %32 = arith.addi %29, %c64_i32 {ttg.partition = array<i32: 4>} : i32
      %532 = arith.addi %529, %c64_i32 {ttg.partition = array<i32: 3>} : i32
      %33 = tt.descriptor_load %arg4[%32, %c0_i32] {ttg.partition = array<i32: 4>} : !tt.tensordesc<tensor<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
      ttg.local_store %33, %34 {ttg.partition = array<i32: 4>} : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> // q1
      // Should we lift out the tmem_alloc?
      // TODO: fix this later
      %cst_0 = arith.constant {ttg.partition = array<i32: 0>} dense<0.000000e+00> : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %35 = ttng.tmem_store %cst_0, %tmem_acc1[%token_8], %true {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>
      %36 = ttng.tmem_store %cst_0, %tmem_acc0[%token_4], %true {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>
      %37:9 = scf.for %arg26 = %c0_i32 to %arg24 step %c128_i32 iter_args(%arg27 = %cst_2, %arg28 = %cst_2, %arg29 = %cst_1, %arg30 = %cst_1, %arg31 = %28, %arg32 = %token, %arg33 = %36, %arg34 = %token_6, %arg35 = %35) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %54 = tt.descriptor_load %arg9[%arg31, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 4>} : !tt.tensordesc<tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
        ttg.local_store %54, %55 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> // k
        // Used by gemm partition 3
        %56 = ttg.memdesc_trans %55 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 5>} : !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %57 = tt.descriptor_load %arg14[%arg31, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 4>} : !tt.tensordesc<tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
        ttg.local_store %57, %58 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> // v
        // consumer of 2nd channel: %31/q0
        %59 = ttng.tc_gen5_mma %31, %56, %tmem_qk0[%arg32], %false, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 5>} : !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>

        // First softmax in partition 1
        // consumer of 1st channel: qk0
        %reg_qk0, %token_14 = ttng.tmem_load %tmem_qk0[%59] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %60 = "tt.reduce"(%reg_qk0) <{axis = 1 : i32}> ({
        ^bb0(%arg36: f32, %arg37: f32):
          %116 = arith.maxnumf %arg36, %arg37 : f32
          tt.reduce.return %116 : f32
        }) {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        // qk_scale
        %20 = tt.splat %19 {ttg.partition = array<i32: 1>} : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        %61 = arith.mulf %60, %20 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %62 = arith.maxnumf %arg29, %61 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        // qk_scale
        %21 = tt.splat %19 {ttg.partition = array<i32: 1>} : f32 -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>

        %63 = arith.mulf %reg_qk0, %21 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %64 = tt.expand_dims %62 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %65 = tt.broadcast %64 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %66 = arith.subf %63, %65 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %67 = math.exp2 %66 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %68 = arith.subf %arg29, %62 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %69 = math.exp2 %68 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        // store alpha0
        %1004 = tt.expand_dims %69 {axis = 1 : i32, ttg.partition = array<i32: 1>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        // source layout is not TMEM compatible
        %1005 = ttg.convert_layout %1004 {ttg.partition = array<i32: 1>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        ttng.tmem_store %1005, %alpha0, %true {ttg.partition = array<i32: 1>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable>
        %70 = "tt.reduce"(%67) <{axis = 1 : i32}> ({
        ^bb0(%arg36: f32, %arg37: f32):
          %116 = arith.addf %arg36, %arg37 : f32
          tt.reduce.return %116 : f32
        }) {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        // Correction in partition 0
        %reg_acc0, %token_16 = ttng.tmem_load %tmem_acc0[%arg33] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %71 = tt.reshape %reg_acc0 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>>
        %72 = tt.trans %71 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>> -> tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>>
        %73 = ttg.convert_layout %72 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>> -> tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>>
        %outLHS, %outRHS = tt.split %73 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>> -> tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        // consumer of %69 (alpha) in correction
        %1169 = ttng.tmem_load %alpha0 {ttg.partition = array<i32: 0>} : !ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %1170 = tt.reshape %1169 {ttg.partition = array<i32: 0>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64xf32, #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [0]], warp = [[16], [32]], block = []}>>
        %1171 = ttg.convert_layout %1170 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [0]], warp = [[16], [32]], block = []}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        %74 = tt.expand_dims %1171 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %75 = tt.broadcast %74 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %76 = arith.mulf %outLHS, %75 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %77 = arith.mulf %outRHS, %75 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %78 = tt.join %76, %77 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>>
        %79 = tt.trans %78 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>> -> tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>>
        %80 = tt.reshape %79 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>> -> tensor<64x128xf32, #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>>

        // Generate p from softmax0
        %81 = arith.truncf %67 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> to tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        ttng.tmem_store %81, %tmem_p0, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x128xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable> // p0

        // Save acc from correction
        %82 = ttg.convert_layout %80 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x128xf32, #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %83 = ttng.tmem_store %82, %tmem_acc0[%token_16], %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>

        // consumer of p0
        %84 = ttng.tc_gen5_mma %tmem_p0, %58, %tmem_acc0[%83], %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 5>} : !ttg.memdesc<64x128xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>
        // Calculate l_i in softmax0
        %85 = arith.mulf %arg27, %69 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %86 = arith.addf %85, %70 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        // consumer of q1
        %87 = ttng.tc_gen5_mma %34, %56, %tmem_qk1[%arg34], %false, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 5>} : !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>

        // Second softmax in partition 2
        // consumer of qk1
        %reg_qk1, %token_19 = ttng.tmem_load %tmem_qk1[%87] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %88 = "tt.reduce"(%reg_qk1) <{axis = 1 : i32}> ({
        ^bb0(%arg36: f32, %arg37: f32):
          %116 = arith.maxnumf %arg36, %arg37 : f32
          tt.reduce.return %116 : f32
        }) {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        // qk_scale
        %220 = tt.splat %19 {ttg.partition = array<i32: 2>} : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        %89 = arith.mulf %88, %220 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %90 = arith.maxnumf %arg30, %89 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        // qk_scale
        %221 = tt.splat %19 {ttg.partition = array<i32: 2>} : f32 -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>

        %91 = arith.mulf %reg_qk1, %221 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %92 = tt.expand_dims %90 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %93 = tt.broadcast %92 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %94 = arith.subf %91, %93 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %95 = math.exp2 %94 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %96 = arith.subf %arg30, %90 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %97 = math.exp2 %96 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        // store alpha1
        %1014 = tt.expand_dims %97 {axis = 1 : i32, ttg.partition = array<i32: 2>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        // source layout is not TMEM compatible
        %1015 = ttg.convert_layout %1014 {ttg.partition = array<i32: 2>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        ttng.tmem_store %1015, %alpha1, %true {ttg.partition = array<i32: 2>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable>
        %98 = "tt.reduce"(%95) <{axis = 1 : i32}> ({
        ^bb0(%arg36: f32, %arg37: f32):
          %116 = arith.addf %arg36, %arg37 : f32
          tt.reduce.return %116 : f32
        }) {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        // Correction
        %reg_acc1, %token_21 = ttng.tmem_load %tmem_acc1[%arg35] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %99 = tt.reshape %reg_acc1 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>>
        %100 = tt.trans %99 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>> -> tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>>
        %101 = ttg.convert_layout %100 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>> -> tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>>
        %outLHS_22, %outRHS_23 = tt.split %101 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>> -> tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        // consumer of alpha in correction
        %1197 = ttng.tmem_load %alpha1 {ttg.partition = array<i32: 0>} : !ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %1198 = tt.reshape %1197 {ttg.partition = array<i32: 0>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64xf32, #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [0]], warp = [[16], [32]], block = []}>>
        %1199 = ttg.convert_layout %1198 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [0]], warp = [[16], [32]], block = []}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

        %102 = tt.expand_dims %1199 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %103 = tt.broadcast %102 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %104 = arith.mulf %outLHS_22, %103 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %105 = arith.mulf %outRHS_23, %103 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %106 = tt.join %104, %105 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>>
        %107 = tt.trans %106 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<64x64x2xf32, #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 0, 1]}>> -> tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>>
        %108 = tt.reshape %107 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x2x64xf32, #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>> -> tensor<64x128xf32, #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>>

        // In softmax1 to emit p
        %109 = arith.truncf %95 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> to tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        ttng.tmem_store %109, %tmem_p1, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x128xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable> // p1

        // Save acc after correction
        %110 = ttg.convert_layout %108 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x128xf32, #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
        %111 = ttng.tmem_store %110, %tmem_acc1[%token_21], %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>

        // consumer of p1
        %112 = ttng.tc_gen5_mma %tmem_p1, %58, %tmem_acc1[%111], %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 5>} : !ttg.memdesc<64x128xbf16, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable>

        // In Softmax1 to emit l_i
        %113 = arith.mulf %arg28, %97 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %114 = arith.addf %113, %98 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
        %115 = arith.addi %arg31, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : i32
        scf.yield %86, %114, %62, %90, %115, %token_14, %84, %token_19, %112 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {tt.disallow_acc_multi_buffer, tt.scheduled_max_stage = 2 : i32}
      // Save l_i in softmax0
      %1204 = tt.expand_dims %37#0 {axis = 1 : i32, ttg.partition = array<i32: 1>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      // source layout is not TMEM compatible
      %1205 = ttg.convert_layout %1204 {ttg.partition = array<i32: 1>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      ttng.tmem_store %1205, %l_i0, %true {ttg.partition = array<i32: 1>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable>

      // Part of the epilogue is in correction
      // consumer of l_i in correction
      %1269 = ttng.tmem_load %l_i0 {ttg.partition = array<i32: 0>} : !ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %1270 = tt.reshape %1269 {ttg.partition = array<i32: 0>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64xf32, #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [0]], warp = [[16], [32]], block = []}>>
      %1271 = ttg.convert_layout %1270 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [0]], warp = [[16], [32]], block = []}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

      %38 = math.log2 %1271 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

      // Save m_i in softmax0
      %2204 = tt.expand_dims %37#2 {axis = 1 : i32, ttg.partition = array<i32: 1>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      // source layout is not TMEM compatible
      %2205 = ttg.convert_layout %2204 {ttg.partition = array<i32: 1>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      ttng.tmem_store %2205, %m_i0, %true {ttg.partition = array<i32: 1>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable>
      // consumer of a channel: %37#2 m_i0
      %2269 = ttng.tmem_load %m_i0 {ttg.partition = array<i32: 0>} : !ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %2270 = tt.reshape %2269 {ttg.partition = array<i32: 0>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64xf32, #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [0]], warp = [[16], [32]], block = []}>>
      %2271 = ttg.convert_layout %2270 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [0]], warp = [[16], [32]], block = []}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
      %39 = arith.addf %2271, %38 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

      // consumer of l_i0
      %40 = tt.expand_dims %1271 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %41 = tt.broadcast %40 {ttg.partition = array<i32: 0>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      // consumer of acc in correction_epilogue
      %reg_acc0_ce, %token_10 = ttng.tmem_load %tmem_acc0[%37#6] {ttg.partition = array<i32: 0>} : !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %42 = arith.divf %reg_acc0_ce, %41 {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %43 = ttg.convert_layout %39 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>

      /////////////
      // %16, %18: used below to calculate %25, %26
      %14 = tt.make_range {ttg.partition = array<i32: 0>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      %15 = tt.splat %13 {ttg.partition = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      %16 = arith.addi %15, %14 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      %17 = tt.make_range {ttg.partition = array<i32: 0>, end = 128 : i32, start = 64 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      %18 = arith.addi %15, %17 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      // calculate store_address for m_i0 m_i1
      %24 = tt.splat %23 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      // users of %25: in partition 0
      %25 = tt.addptr %24, %16 {ttg.partition = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      // users of %26: in partition 0
      %26 = tt.addptr %24, %18 {ttg.partition = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>

      tt.store %25, %43 {ttg.partition = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      %44 = arith.truncf %42 {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> to tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %45 = ttg.convert_layout %44 {ttg.partition = array<i32: 0>} : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
      // Code partitioning will need to create a channel to save %45 in smem
      // consumer of output from TMA store
      ttg.local_store %45, %out0 {ttg.partition = array<i32: 0>} : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
      %1145 = ttg.local_load %out0 {ttg.partition = array<i32: 3>} : !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
      tt.descriptor_store %arg19[%529, %c0_i32], %1145 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>

      %1304 = tt.expand_dims %37#1 {axis = 1 : i32, ttg.partition = array<i32: 2>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      // source layout is not TMEM compatible
      %1305 = ttg.convert_layout %1304 {ttg.partition = array<i32: 2>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      ttng.tmem_store %1305, %l_i1, %true {ttg.partition = array<i32: 2>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable>
      // consumer of l_i1
      %1369 = ttng.tmem_load %l_i1 {ttg.partition = array<i32: 0>} : !ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %1370 = tt.reshape %1369 {ttg.partition = array<i32: 0>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64xf32, #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [0]], warp = [[16], [32]], block = []}>>
      %1371 = ttg.convert_layout %1370 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [0]], warp = [[16], [32]], block = []}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

      %46 = math.log2 %1371 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

      %2304 = tt.expand_dims %37#3 {axis = 1 : i32, ttg.partition = array<i32: 2>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      // source layout is not TMEM compatible
      %2305 = ttg.convert_layout %2304 {ttg.partition = array<i32: 2>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      ttng.tmem_store %2305, %m_i1, %true {ttg.partition = array<i32: 2>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> !ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable>
      // consumer of a channel %37#3 m_i1
      %2369 = ttng.tmem_load %m_i1 {ttg.partition = array<i32: 0>} : !ttg.memdesc<64x1xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %2370 = tt.reshape %2369 {ttg.partition = array<i32: 0>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64xf32, #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [0]], warp = [[16], [32]], block = []}>>
      %2371 = ttg.convert_layout %2370 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [0]], warp = [[16], [32]], block = []}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>

      %47 = arith.addf %2371, %46 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
      // consumer of l_i1
      %48 = tt.expand_dims %1371 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %49 = tt.broadcast %48 {ttg.partition = array<i32: 0>} : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      // consumer of acc in correction epilogue
      %reg_acc1_ce, %token_12 = ttng.tmem_load %tmem_acc1[%37#8] {ttg.partition = array<i32: 0>} : !ttg.memdesc<64x128xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %50 = arith.divf %reg_acc1_ce, %49 {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %51 = ttg.convert_layout %47 {ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      tt.store %26, %51 {ttg.partition = array<i32: 0>} : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
      %52 = arith.truncf %50 {ttg.partition = array<i32: 0>} : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> to tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>>
      %53 = ttg.convert_layout %52 {ttg.partition = array<i32: 0>} : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
      // consumer of output in tma store
      ttg.local_store %53, %out1 {ttg.partition = array<i32: 0>} : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
      %1153 = ttg.local_load %out1 {ttg.partition = array<i32: 3>} : !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
      tt.descriptor_store %arg19[%532, %c0_i32], %1153 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
