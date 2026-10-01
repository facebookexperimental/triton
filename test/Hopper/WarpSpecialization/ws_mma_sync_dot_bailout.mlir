// RUN: triton-opt %s --nvgpu-partition-scheduling-meta="separate-epilogue-store" --nvgpu-warp-specialization="capability=100 num-stages=2 smem-budget=232448" | FileCheck %s

// A BLOCK_M=32 persistent bmm on Blackwell: the tt.dot cannot use tcgen05 and
// stays an mma.sync (#mma v2) dot, so there is no MMA to build partitions
// around. PartitionSchedulingMeta used to put everything in a lone "load"
// partition, and code partitioning then crashed in getAccumCount (num_warps=4)
// or emitted a warp_specialize region with no terminator (num_warps=8). The
// loop must instead be left unspecialized, with its WS metadata stripped.

// CHECK-LABEL: @triton_blackwell_bmm
// CHECK-NOT: ttg.partition
// CHECK-NOT: async_task_id
// CHECK-NOT: nvws.
// CHECK: tt.descriptor_load
// CHECK: tt.dot
// CHECK-NOT: tt.warp_specialize
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 0], [0, 16]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, rank = 3}>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_blackwell_bmm(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<128> : tensor<32x1xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<1x32xi32, #blocked>
    %c16384_i32 = arith.constant 16384 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %c83952_i32 = arith.constant 83952 : i32
    %c6_i32 = arith.constant 6 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64
    %c5247_i32 = arith.constant 5247 : i32
    %c128_i32 = arith.constant 128 : i32
    %c176_i32 = arith.constant 176 : i32
    %c22528_i64 = arith.constant 22528 : i64
    %c128_i64 = arith.constant 128 : i64
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %0 = tt.make_tensor_descriptor %arg0, [%c5247_i32, %c176_i32, %c128_i32], [%c22528_i64, %c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x32x32xbf16, #shared>
    %1 = tt.make_tensor_descriptor %arg1, [%c5247_i32, %c176_i32, %c128_i32], [%c22528_i64, %c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<1x32x32xbf16, #shared>
    %2 = tt.get_program_id x : i32
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %7 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #blocked>
    scf.for %arg3 = %2 to %c83952_i32 step %c148_i32  : i32 {
      %8 = arith.divsi %arg3, %c16_i32 : i32
      %9 = arith.remsi %arg3, %c16_i32 : i32
      %10 = arith.divsi %9, %c32_i32 : i32
      %11 = arith.muli %10, %c8_i32 : i32
      %12 = arith.subi %c4_i32, %11 : i32
      %13 = arith.minsi %12, %c8_i32 : i32
      %14 = arith.remsi %9, %13 : i32
      %15 = arith.addi %11, %14 : i32
      %16 = arith.remsi %9, %c32_i32 : i32
      %17 = arith.divsi %16, %13 : i32
      %18 = arith.muli %15, %c32_i32 : i32
      %19 = arith.muli %17, %c32_i32 : i32
      %20 = scf.for %arg4 = %c0_i32 to %c6_i32 step %c1_i32 iter_args(%arg5 = %cst_1) -> (tensor<32x32xf32, #mma>)  : i32 {
        %40 = arith.muli %arg4, %c32_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        %41 = tt.descriptor_load %0[%8, %40, %18] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<1x32x32xbf16, #shared> -> tensor<32x32xbf16, #blocked>
        %42 = ttg.convert_layout %41 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #linear>
        %43 = tt.descriptor_load %1[%8, %40, %19] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<1x32x32xbf16, #shared> -> tensor<32x32xbf16, #blocked>
        %44 = tt.trans %42 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : tensor<32x32xbf16, #linear> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
        %45 = ttg.convert_layout %43 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
        %46 = tt.dot %44, %45, %arg5 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
        scf.yield %46 : tensor<32x32xf32, #mma>
      } {tt.scheduled_max_stage = 1 : i32}
      %21 = tt.splat %18 : i32 -> tensor<32x1xi32, #blocked>
      %22 = arith.addi %21, %4 : tensor<32x1xi32, #blocked>
      %23 = tt.splat %19 : i32 -> tensor<1x32xi32, #blocked>
      %24 = arith.addi %23, %6 : tensor<1x32xi32, #blocked>
      %25 = arith.cmpi slt, %22, %cst : tensor<32x1xi32, #blocked>
      %26 = arith.cmpi slt, %24, %cst_0 : tensor<1x32xi32, #blocked>
      %27 = tt.broadcast %25 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %28 = tt.broadcast %26 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %29 = arith.andi %27, %28 : tensor<32x32xi1, #blocked>
      %30 = arith.muli %22, %cst : tensor<32x1xi32, #blocked>
      %31 = tt.broadcast %24 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
      %32 = tt.broadcast %30 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
      %33 = arith.addi %31, %32 : tensor<32x32xi32, #blocked>
      %34 = arith.muli %8, %c16384_i32 : i32
      %35 = tt.splat %34 : i32 -> tensor<32x32xi32, #blocked>
      %36 = arith.addi %33, %35 : tensor<32x32xi32, #blocked>
      %37 = tt.addptr %7, %36 : tensor<32x32x!tt.ptr<bf16>, #blocked>, tensor<32x32xi32, #blocked>
      %38 = arith.truncf %20 : tensor<32x32xf32, #mma> to tensor<32x32xbf16, #mma>
      %39 = ttg.convert_layout %38 : tensor<32x32xbf16, #mma> -> tensor<32x32xbf16, #blocked>
      tt.store %37, %39, %29 : tensor<32x32x!tt.ptr<bf16>, #blocked>
    } {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize}
    tt.return
  }
}
