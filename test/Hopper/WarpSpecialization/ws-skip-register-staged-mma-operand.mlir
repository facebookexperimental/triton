// RUN: triton-opt %s -split-input-file --nvgpu-warp-specialization="num-stages=1 capability=100 smem-budget=232448" | FileCheck %s

// FP32 A from a transposed TMA load is staged through registers (TMA cannot
// write the MMA's layout); warp specialization must bail out.

// CHECK-LABEL: @fp32_transposed_a
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: nvws.descriptor_load
// CHECK: tt.descriptor_load
// CHECK: ttng.tc_gen5_mma

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @fp32_transposed_a(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg5, %arg3], [%0, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<32x128xf32, #shared>
    %2 = arith.extsi %arg5 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%2, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x32xf32, #shared>
    %4 = arith.addi %arg4, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.addi %arg3, %c127_i32 : i32
    %7 = arith.divsi %6, %c128_i32 : i32
    %8 = arith.muli %7, %5 : i32
    %9 = tt.get_program_id x : i32
    %10 = tt.get_num_programs x : i32
    %11 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %13 = tt.splat %arg4 : i32 -> tensor<128x1xi32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    scf.for %arg6 = %9 to %8 step %10  : i32 {
      %15 = arith.divsi %arg6, %5 {ttg.partition = array<i32: 1>} : i32
      %16 = arith.remsi %arg6, %5 {ttg.partition = array<i32: 1>} : i32
      %17 = arith.muli %15, %c128_i32 {ttg.partition = array<i32: 1>} : i32
      %18 = arith.muli %16, %c128_i32 {ttg.partition = array<i32: 1>} : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %19 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %20:2 = scf.for %arg7 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg8 = %false, %arg9 = %19) -> (i1, !ttg.async.token)  : i32 {
        %33 = tt.descriptor_load %1[%arg7, %17] {ttg.partition = array<i32: 1>} : !tt.tensordesc<32x128xf32, #shared> -> tensor<32x128xf32, #blocked>
        %34 = ttg.local_alloc %33 : (tensor<32x128xf32, #blocked>) -> !ttg.memdesc<32x128xf32, #shared1, #smem>
        %35 = ttg.memdesc_trans %34 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<32x128xf32, #shared1, #smem> -> !ttg.memdesc<128x32xf32, #shared, #smem>
        %36 = tt.descriptor_load %3[%18, %arg7] {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x32xf32, #shared> -> tensor<128x32xf32, #blocked1>
        %37 = ttg.local_alloc %36 {ttg.partition = array<i32: 1>} : (tensor<128x32xf32, #blocked1>) -> !ttg.memdesc<128x32xf32, #shared, #smem>
        %38 = ttg.memdesc_trans %37 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x32xf32, #shared, #smem> -> !ttg.memdesc<32x128xf32, #shared1, #smem>
        %39 = ttng.tc_gen5_mma %35, %38, %result[%arg9], %arg8, %true {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x32xf32, #shared, #smem>, !ttg.memdesc<32x128xf32, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %39 : i1, !ttg.async.token
      }
      %21 = tt.splat %17 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %22 = arith.addi %21, %11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %23 = tt.splat %18 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %24 = arith.addi %23, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %25 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %26 = arith.muli %25, %13 : tensor<128x1xi32, #blocked>
      %27 = tt.addptr %14, %26 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
      %28 = tt.expand_dims %24 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %29 = tt.broadcast %27 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x128x!tt.ptr<f32>, #blocked>
      %30 = tt.broadcast %28 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %31 = tt.addptr %29, %30 : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %result_0, %token_1 = ttng.tmem_load %result[%20#1] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %32 = ttg.convert_layout %result_0 {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked>
      tt.store %31, %32 {ttg.partition = array<i32: 2>} : tensor<128x128x!tt.ptr<f32>, #blocked>
    } {tt.warp_specialize, ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
