// RUN: triton-opt %s --nvgpu-test-ws-memory-planner=num-buffers=3 | FileCheck %s

// Test: When two SMEM buffers are in the same innermost loop, the memory
// planner assigns both the same buffer.id (reuse group). The code partition
// pass later merges consumer groups for channels sharing a reuse group, so a
// single barrier_expect + wait is emitted.
//
// A (128x64xf16): inner dim = 64 * 2B = 128B = swizzle -> no split
// B (64x256xf16): inner dim = 256 * 2B = 512B > 128B swizzle -> split copies
//
// Both buffers share buffer.id = 0 (same reuse group).

// CHECK-LABEL: @matmul_kernel_tma_persistent
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 0 : i32}
// CHECK-SAME: 64x256xf16
// CHECK: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 0 : i32}
// CHECK-SAME: 128x64xf16

#blocked = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<64x256xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %false = arith.constant {async_task_id = array<i32: 0>} false
    %true = arith.constant {async_task_id = array<i32: 0>} true
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 148 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 1, 2>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 128 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 256 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 127 : i32
    %c255_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 255 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 63 : i32
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %2 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %3 = arith.addi %arg15, %c127_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %4 = arith.divsi %3, %c128_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %5 = arith.addi %arg16, %c255_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %6 = arith.divsi %5, %c256_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %7 = arith.addi %arg17, %c63_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %8 = arith.divsi %7, %c64_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %9 = arith.muli %4, %6 {async_task_id = array<i32: 0, 1, 2>} : i32
    %10 = arith.subi %2, %c148_i32 {async_task_id = array<i32: 2>} : i32
    %11 = arith.muli %6, %c8_i32 {async_task_id = array<i32: 1, 2>} : i32
    %12 = scf.for %arg19 = %2 to %9 step %c148_i32 iter_args(%arg20 = %10) -> (i32)  : i32 {
      %13 = arith.divsi %arg19, %11 {async_task_id = array<i32: 1>} : i32
      %14 = arith.muli %13, %c8_i32 {async_task_id = array<i32: 1>} : i32
      %15 = arith.subi %4, %14 {async_task_id = array<i32: 1>} : i32
      %16 = arith.minsi %15, %c8_i32 {async_task_id = array<i32: 1>} : i32
      %17 = arith.remsi %arg19, %16 {async_task_id = array<i32: 1>} : i32
      %18 = arith.addi %14, %17 {async_task_id = array<i32: 1>} : i32
      %19 = arith.remsi %arg19, %11 {async_task_id = array<i32: 1>} : i32
      %20 = arith.divsi %19, %16 {async_task_id = array<i32: 1>} : i32
      %21 = arith.muli %18, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %22 = arith.muli %20, %c256_i32 {async_task_id = array<i32: 1>} : i32
      %23 = ttng.tmem_store %cst, %result[%token], %true {async_task_id = array<i32: 0>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %24:2 = scf.for %arg21 = %c0_i32 to %8 step %c1_i32 iter_args(%arg22 = %false, %arg23 = %23) -> (i1, !ttg.async.token)  : i32 {
        %43 = arith.muli %arg21, %c64_i32 {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %44 = tt.descriptor_load %arg0[%21, %43] {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        ttg.local_store %44, %1 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %45 = tt.descriptor_load %arg5[%43, %22] {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<64x256xf16, #shared>> -> tensor<64x256xf16, #blocked2>
        ttg.local_store %45, %0 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x256xf16, #blocked2> -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
        %46 = ttng.tc_gen5_mma %1, %0, %result[%arg23], %arg22, %true {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 2>} %true, %46 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2>, tt.scheduled_max_stage = 2 : i32}
      %25 = arith.addi %arg20, %c148_i32 {async_task_id = array<i32: 2>} : i32
      %26 = arith.divsi %25, %11 {async_task_id = array<i32: 2>} : i32
      %27 = arith.muli %26, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %28 = arith.subi %4, %27 {async_task_id = array<i32: 2>} : i32
      %29 = arith.minsi %28, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %30 = arith.remsi %25, %29 {async_task_id = array<i32: 2>} : i32
      %31 = arith.addi %27, %30 {async_task_id = array<i32: 2>} : i32
      %32 = arith.remsi %25, %11 {async_task_id = array<i32: 2>} : i32
      %33 = arith.divsi %32, %29 {async_task_id = array<i32: 2>} : i32
      %34 = arith.muli %31, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %35 = arith.muli %33, %c256_i32 {async_task_id = array<i32: 2>} : i32
      %result_0, %token_1 = ttng.tmem_load %result[%24#1] {async_task_id = array<i32: 2>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
      %36 = tt.reshape %result_0 {async_task_id = array<i32: 2>} : tensor<128x256xf32, #blocked> -> tensor<128x2x128xf32, #blocked3>
      %37 = tt.trans %36 {async_task_id = array<i32: 2>, order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked3> -> tensor<128x128x2xf32, #blocked4>
      %outLHS, %outRHS = tt.split %37 {async_task_id = array<i32: 2>} : tensor<128x128x2xf32, #blocked4> -> tensor<128x128xf32, #blocked5>
      %38 = arith.truncf %outRHS {async_task_id = array<i32: 2>} : tensor<128x128xf32, #blocked5> to tensor<128x128xf16, #blocked5>
      %39 = arith.truncf %outLHS {async_task_id = array<i32: 2>} : tensor<128x128xf32, #blocked5> to tensor<128x128xf16, #blocked5>
      %40 = ttg.convert_layout %39 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked5> -> tensor<128x128xf16, #blocked6>
      tt.descriptor_store %arg10[%34, %35], %40 {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked6>
      %41 = ttg.convert_layout %38 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked5> -> tensor<128x128xf16, #blocked6>
      %42 = arith.addi %35, %c128_i32 {async_task_id = array<i32: 2>} : i32
      tt.descriptor_store %arg10[%34, %42], %41 {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked6>
      scf.yield {async_task_id = array<i32: 2>} %25 : i32
    } {async_task_id = array<i32: 0, 1, 2>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
