// RUN: env TRITON_USE_META_WS=1 triton-opt %s -split-input-file --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448 tma-store-pipelining=true" -verify-diagnostics | FileCheck %s

// A persistent GEMM whose tile loop was flattened (tl.range(flatten=True),
// tritongpu-fuse-nested-loops) with the K loop: the epilogue tmem_load of the
// MMA accumulator lands in an scf.if nested in the flat loop body.
// handleOperandD only handles accumulator users that are direct children of
// the MMA's loop body, so this used to assert (TmemAllocChannel "single
// consumer" with a TMA store epilogue, "Unexpected Producer Found" with a
// tl.store epilogue). Meta autoWS must instead warn and fall back to a plain
// non-WS kernel with all WS metadata stripped.

// TMA store epilogue.
// CHECK-LABEL: @flat_tma_store
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-NOT: ttg.partition
// CHECK: ttng.tc_gen5_mma
// CHECK: ttng.tmem_load
// CHECK: ttng.async_tma_copy_local_to_global
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 128]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @flat_tma_store(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %B: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %acc = arith.constant false
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %a_desc = arith.constant 512 : i64
    %c512_i32 = arith.constant 512 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c148_i32 = arith.constant 148 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #linear>
    %a_desc_0 = tt.make_tensor_descriptor %A, [%c8192_i32, %c512_i32], [%a_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared>
    %b_desc = tt.make_tensor_descriptor %B, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<64x256xbf16, #shared>
    %c_desc = tt.make_tensor_descriptor %C, [%c8192_i32, %c256_i32], [%c256_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x256xbf16, #shared>
    %0 = tt.get_program_id x : i32
    %1 = arith.subi %c64_i32, %0 : i32
    %2 = arith.ceildivsi %1, %c148_i32 : i32
    %3 = arith.muli %2, %c8_i32 : i32
    %4 = arith.subi %0, %c148_i32 : i32
    %acc_1, %acc_2 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_3 = ttng.tmem_store %cst, %acc_1[%acc_2], %true {ttg.partition = array<i32: 1>} : tensor<128x256xf32, #linear> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %5:6 = scf.for %arg3 = %c0_i32 to %3 step %c1_i32 iter_args(%arg4 = %c0_i32, %arg5 = %4, %arg6 = %c0_i32, %arg7 = %c0_i32, %acc_4 = %acc, %acc_5 = %acc_3) -> (i32, i32, i32, i32, i1, !ttg.async.token)  : i32 {
      %6 = arith.cmpi eq, %arg4, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %7 = arith.select %6, %c0_i32, %arg6 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %8:2 = scf.if %6 -> (i32, i32) {
        %15 = arith.addi %arg5, %c148_i32 : i32
        %off_m = arith.muli %15, %c128_i32 : i32
        scf.yield %off_m, %15 : i32, i32
      } else {
        scf.yield %arg7, %arg5 : i32, i32
      } {loop.cluster = 1 : i32, loop.stage = 0 : i32}
      %off_k = arith.muli %7, %c64_i32 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %acc_6 = tt.descriptor_load %a_desc_0[%8#0, %off_k] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
      %acc_7 = ttg.local_alloc %acc_6 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
      %acc_8 = tt.descriptor_load %b_desc[%off_k, %c0_i32] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<64x256xbf16, #shared> -> tensor<64x256xbf16, #blocked1>
      %acc_9 = ttg.local_alloc %acc_8 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (tensor<64x256xbf16, #blocked1>) -> !ttg.memdesc<64x256xbf16, #shared, #smem>
      %acc_10 = ttng.tc_gen5_mma %acc_7, %acc_9, %acc_1[%acc_5], %acc_4, %true {loop.cluster = 5 : i32, loop.stage = 0 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x256xbf16, #shared, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %9 = arith.addi %7, %c1_i32 {loop.cluster = 4 : i32, loop.stage = 1 : i32} : i32
      %10 = arith.cmpi eq, %arg4, %c7_i32 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : i32
      %acc_11 = arith.select %10, %acc, %true {loop.cluster = 4 : i32, loop.stage = 1 : i32} : i1
      %11 = scf.if %10 -> (!ttg.async.token) {
        // expected-warning @below {{meta autoWS does not support an MMA accumulator used inside a nested region of its loop}}
        %acc_12, %acc_13 = ttng.tmem_load %acc_1[%acc_10] : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #linear>
        %15 = arith.truncf %acc_12 : tensor<128x256xf32, #linear> to tensor<128x256xbf16, #linear>
        %16 = ttg.convert_layout %15 : tensor<128x256xbf16, #linear> -> tensor<128x256xbf16, #blocked1>
        %c_desc_staging = ttg.local_alloc %16 : (tensor<128x256xbf16, #blocked1>) -> !ttg.memdesc<128x256xbf16, #shared, #smem, mutable>
        %17 = ttng.async_tma_copy_local_to_global %c_desc[%8#0, %c0_i32] %c_desc_staging {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x256xbf16, #shared>, !ttg.memdesc<128x256xbf16, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %17   {ttg.partition = array<i32: 1>} : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 3>} %acc_13 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 3>} %acc_10 : !ttg.async.token
      } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>}
      %12 = arith.addi %arg4, %c1_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %13 = arith.cmpi eq, %arg4, %c7_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %14 = arith.select %13, %c0_i32, %12 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      scf.yield %14, %8#1, %9, %8#0, %acc_11, %11 : i32, i32, i32, i32, i1, !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["gemm", "epilogue", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}



// -----

// tl.store epilogue.
// CHECK-LABEL: @flat_ptr_store
// CHECK-NOT: ttg.warp_specialize
// CHECK-NOT: async_task_id
// CHECK-NOT: ttg.partition
// CHECK: ttng.tc_gen5_mma
// CHECK: ttng.tmem_load
// CHECK: tt.store
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 128]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @flat_ptr_store(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %B: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %acc = arith.constant false
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<256> : tensor<128x1xi32, #blocked>
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c512_i32 = arith.constant 512 : i32
    %a_desc = arith.constant 512 : i64
    %c1_i64 = arith.constant 1 : i64
    %true = arith.constant true
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #linear>
    %a_desc_1 = tt.make_tensor_descriptor %A, [%c8192_i32, %c512_i32], [%a_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared>
    %b_desc = tt.make_tensor_descriptor %B, [%c512_i32, %c256_i32], [%c256_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<64x256xbf16, #shared>
    %0 = tt.get_program_id x : i32
    %rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cols = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %1 = tt.splat %C : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>, #blocked>
    %2 = tt.broadcast %cols_2 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %3 = arith.subi %c64_i32, %0 : i32
    %4 = arith.ceildivsi %3, %c148_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.subi %0, %c148_i32 : i32
    %acc_3, %acc_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_5 = ttng.tmem_store %cst_0, %acc_3[%acc_4], %true : tensor<128x256xf32, #linear> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %7:6 = scf.for %arg3 = %c0_i32 to %5 step %c1_i32 iter_args(%arg4 = %c0_i32, %arg5 = %6, %arg6 = %c0_i32, %arg7 = %c0_i32, %acc_6 = %acc, %acc_7 = %acc_5) -> (i32, i32, i32, i32, i1, !ttg.async.token)  : i32 {
      %8 = arith.cmpi eq, %arg4, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %9 = arith.select %8, %c0_i32, %arg6 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %10:2 = scf.if %8 -> (i32, i32) {
        %17 = arith.addi %arg5, %c148_i32 : i32
        %off_m = arith.muli %17, %c128_i32 : i32
        scf.yield %off_m, %17 : i32, i32
      } else {
        scf.yield %arg7, %arg5 : i32, i32
      } {loop.cluster = 1 : i32, loop.stage = 0 : i32}
      %off_k = arith.muli %9, %c64_i32 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : i32
      %acc_8 = tt.descriptor_load %a_desc_1[%10#0, %off_k] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked1>
      %acc_9 = ttg.local_alloc %acc_8 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
      %acc_10 = tt.descriptor_load %b_desc[%off_k, %c0_i32] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<64x256xbf16, #shared> -> tensor<64x256xbf16, #blocked>
      %acc_11 = ttg.local_alloc %acc_10 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<64x256xbf16, #blocked>) -> !ttg.memdesc<64x256xbf16, #shared, #smem>
      %acc_12 = ttng.tc_gen5_mma %acc_9, %acc_11, %acc_3[%acc_7], %acc_6, %true {loop.cluster = 5 : i32, loop.stage = 0 : i32, tt.self_latency = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x256xbf16, #shared, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %11 = arith.addi %9, %c1_i32 {loop.cluster = 4 : i32, loop.stage = 1 : i32} : i32
      %12 = arith.cmpi eq, %arg4, %c7_i32 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : i32
      %acc_13 = arith.select %12, %acc, %true {loop.cluster = 4 : i32, loop.stage = 1 : i32} : i1
      %13 = scf.if %12 -> (!ttg.async.token) {
        %rows_14 = tt.splat %10#0 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %rows_15 = arith.addi %rows_14, %rows : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %rows_16 = tt.expand_dims %rows_15 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
        %17 = arith.muli %rows_16, %cst : tensor<128x1xi32, #blocked>
        %18 = tt.addptr %1, %17 : tensor<128x1x!tt.ptr<bf16>, #blocked>, tensor<128x1xi32, #blocked>
        %19 = tt.broadcast %18 : tensor<128x1x!tt.ptr<bf16>, #blocked> -> tensor<128x256x!tt.ptr<bf16>, #blocked>
        %20 = tt.addptr %19, %2 : tensor<128x256x!tt.ptr<bf16>, #blocked>, tensor<128x256xi32, #blocked>
        // expected-warning @below {{meta autoWS does not support an MMA accumulator used inside a nested region of its loop}}
        %acc_17, %acc_18 = ttng.tmem_load %acc_3[%acc_12] : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #linear>
        %21 = arith.truncf %acc_17 : tensor<128x256xf32, #linear> to tensor<128x256xbf16, #linear>
        %22 = ttg.convert_layout %21 : tensor<128x256xbf16, #linear> -> tensor<128x256xbf16, #blocked>
        tt.store %20, %22 : tensor<128x256x!tt.ptr<bf16>, #blocked>
        scf.yield {ttg.partition = array<i32: 2>} %acc_18 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 2>} %acc_12 : !ttg.async.token
      } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>}
      %14 = arith.addi %arg4, %c1_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %15 = arith.cmpi eq, %arg4, %c7_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %16 = arith.select %15, %c0_i32, %14 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      scf.yield %16, %10#1, %11, %10#0, %acc_13, %13 : i32, i32, i32, i32, i1, !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
