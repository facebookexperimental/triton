// RUN: env TRITON_USE_META_WS=1 triton-opt %s --nvgpu-warp-specialization="capability=100 num-stages=3 smem-budget=232448 tma-store-pipelining=true" -verify-diagnostics | FileCheck %s

// Flattened persistent GEMM: the epilogue accumulator tmem_load sits in an scf.if inside
// the MMA loop. Meta autoWS must warn and fall back to non-WS instead of asserting.

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
