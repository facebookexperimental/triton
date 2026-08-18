// RUN: triton-opt %s --nvgpu-warp-specialization="num-stages=2 capability=100 smem-budget=232448" | FileCheck %s

// The MMA producer is in the inner K loop while the TMEM consumer is in a
// sibling epilogue condition. The completion wait must be lifted before the
// condition because padded CTAs skip its body but still execute the MMA.

// CHECK-LABEL: @matmul_kernel_tma_persistent_ws
// CHECK: ttg.warp_specialize
// CHECK: default
// CHECK: scf.for {{.*}} iter_args(%[[DEFAULT_D_COUNT:arg[0-9]+]] = %{{.*}}) -> (i64)
// CHECK: ttng.wait_barrier
// CHECK: scf.if
// CHECK-NOT: ttng.wait_barrier
// CHECK: ttng.tmem_load
// CHECK: }
// CHECK: ttng.arrive_barrier
// CHECK: arith.addi %[[DEFAULT_D_COUNT]], %c1_i64
// CHECK: partition0
// CHECK: scf.for {{.*}} iter_args(%[[D_COUNT:arg[0-9]+]] = %{{.*}}, %[[K_COUNT:arg[0-9]+]] = %{{.*}}) -> (i64, i64)
// CHECK: ttng.wait_barrier
// CHECK: scf.for
// CHECK: ttng.tc_gen5_mma
// CHECK: scf.yield
// CHECK: }
// CHECK: ttng.tc_gen5_commit
// CHECK: arith.addi %[[D_COUNT]], %c1_i64
// CHECK: scf.yield {{.*}}, %{{.*}}#1 : i64, i64

// A consumer in an earlier sibling region reads the previous TMEM value. The
// forward-only sibling protocol cannot represent that loop-carried edge, so
// AutoWS must leave the function unspecialized. The endpoints intentionally
// use distinct views of the same root allocation.
// CHECK-LABEL: @reversed_sibling_tmem_channel
// CHECK-NOT: warp_specialize
// CHECK-NOT: async_task_id
// CHECK-NOT: ttg.partition
// CHECK: ttng.tc_gen5_mma
// CHECK-NOT: warp_specialize
// CHECK-NOT: async_task_id
// CHECK-NOT: ttg.partition
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 2 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @matmul_kernel_tma_persistent_ws(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<128x64xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf16, #shared>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c127_i32 = arith.constant 127 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = arith.addi %arg15, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %2 = arith.addi %arg16, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.addi %arg17, %c63_i32 : i32
    %5 = arith.divsi %4, %c64_i32 : i32
    %6 = tt.get_program_id x : i32
    %7 = arith.remsi %6, %c2_i32 : i32
    %8 = tt.get_program_id y : i32
    %9 = arith.remsi %8, %c2_i32 : i32
    %10 = arith.divsi %6, %c2_i32 : i32
    %11 = arith.divsi %8, %c2_i32 : i32
    %12 = arith.addi %1, %c1_i32 : i32
    %13 = arith.divsi %12, %c2_i32 : i32
    %14 = arith.addi %3, %c1_i32 : i32
    %15 = arith.divsi %14, %c2_i32 : i32
    %16 = arith.muli %10, %15 : i32
    %17 = arith.addi %16, %11 : i32
    %18 = arith.muli %13, %15 : i32
    %19 = tt.get_num_programs x : i32
    %20 = arith.divsi %19, %c2_i32 : i32
    %21 = tt.get_num_programs y : i32
    %22 = arith.divsi %21, %c2_i32 : i32
    %23 = arith.muli %20, %22 : i32
    scf.for %arg18 = %17 to %18 step %23  : i32 {
      %24 = arith.divsi %arg18, %15 : i32
      %25 = arith.muli %24, %c2_i32 : i32
      %26 = arith.addi %25, %7 : i32
      %27 = arith.remsi %arg18, %15 : i32
      %28 = arith.muli %27, %c2_i32 : i32
      %29 = arith.addi %28, %9 : i32
      %30 = arith.muli %26, %c128_i32 : i32
      %31 = arith.muli %29, %c128_i32 : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %32 = ttng.tmem_store %cst, %result[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %33:2 = scf.for %arg19 = %c0_i32 to %5 step %c1_i32 iter_args(%arg20 = %false, %arg21 = %32) -> (i1, !ttg.async.token)  : i32 {
        %38 = arith.muli %arg19, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        %39 = tt.descriptor_load %arg0[%30, %38] {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.multicast_axes = array<i32: 1>, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %40 = ttg.local_alloc %39 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %41 = tt.descriptor_load %arg5[%31, %38] {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.multicast_axes = array<i32: 0>, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %42 = ttg.local_alloc %41 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %43 = ttg.memdesc_trans %42 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %44 = ttng.tc_gen5_mma %40, %43, %result[%arg21], %arg20, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %44 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 1 : i32}
      %34 = arith.cmpi slt, %26, %1 : i32
      %35 = arith.cmpi slt, %29, %3 : i32
      %36 = arith.andi %34, %35 : i1
      %37 = scf.if %36 -> (!ttg.async.token) {
        %result_0, %token_1 = ttng.tmem_load %result[%33#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %38 = arith.truncf %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %39 = ttg.convert_layout %38 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
        %40 = ttg.local_alloc %39 {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %41 = ttng.async_tma_copy_local_to_global %arg10[%30, %31] %40 {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %41   {ttg.partition = array<i32: 0>} : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0>} %token_1 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0>} %33#1 : !ttg.async.token
      }
    } {tt.data_partition_factor = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  tt.func public @reversed_sibling_tmem_channel(
      %a: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>) {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %outer = %c0_i32 to %c1_i32 step %c1_i32 : i32 {
      %acc, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      scf.if %true {
        %load_view = ttg.memdesc_index %acc[%c0_i32] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %value, %load_token = ttng.tmem_load %load_view[%token] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      }
      scf.for %k = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%mma_token = %token) -> (!ttg.async.token) : i32 {
        %mma_view = ttg.memdesc_index %acc[%c0_i32] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %next = ttng.tc_gen5_mma %a, %b, %mma_view[%mma_token], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %next : !ttg.async.token
      }
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm"], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}
