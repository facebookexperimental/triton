// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=2" | FileCheck %s

// Regression test for the persistent FA-bwd dv/dk staging SMEM cross-tile race
// (bug #9 in .llms/rules/partition-scheduler-bugs.md / D109859261).
//
// The dv/dk TMA-staging buffers alias the v/do operand SMEM
// (allocation.reuseTarget, realized by mergeStagingReuseIntoHost): the operand
// buffers are buffer.copy=1 and the staging is a reinterpret view of them, so
// there is no second slot to pipeline into. Across the persistent outer tile
// loop the *next* tile's operand load (load task) must not overwrite that SMEM
// until the *previous* tile's staging TMA store (staging task) has drained.
//
// doCodePartition "Step 7.5" inserts a dedicated single-buffered CROSS-partition
// reuse token for this write-after-read edge:
//   * the LOAD task (async_task_id = 2) producer_acquires it at the top of the
//     persistent outer loop, with a loop-carried (induction-variable derived)
//     phase, targeting the staging task (dstTask = 3);
//   * the STAGING task (async_task_id = 3) consumer_releases it at the bottom of
//     the outer-loop body (region 4), with a constant buffer index, targeting
//     the load task (dstTask = 2).
// The acquire/release reference the SAME token (a single dedicated 1x!nvws.token
// allocated at function entry), so the edge is a coarse cross-tile barrier, not
// per-iteration serialization of legitimate pipelining.
//
// Before the fix Step 7.5 emitted a degenerate same-partition producer_acquire
// on the host token with constant bufferIdx=0/phase=0, which WSLowerToken elided
// to a no-op -> cross-tile SMEM race (non-deterministic wrong dv/dk gradients on
// the persistent path). E2E regression: test_bwd_tmem_dsT_reuse_3group_persistent.

// CHECK-LABEL: @_attn_bwd_persist
// Load task (2) acquires the dedicated single-buffered reuse token at the top of
// the persistent outer loop (loop-carried phase), targeting the staging task.
// CHECK: nvws.producer_acquire %[[WAR_TOK:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}} {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 3 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : tensor<1x!nvws.token>, i32, i1
// Staging task (3) releases the SAME token at the bottom of the outer-loop body
// (region 4) with a constant buffer index, targeting the load task (2).
// CHECK: nvws.consumer_release %[[WAR_TOK]], %{{[a-zA-Z0-9_]+}} {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 2 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : tensor<1x!nvws.token>, i32

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd_persist(%arg0: !tt.tensordesc<128x128xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<128x128xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf16, #shared>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: f32, %arg16: !tt.tensordesc<128x128xf16, #shared>, %arg17: i32, %arg18: i32, %arg19: i64, %arg20: i64, %arg21: !tt.tensordesc<128x32xf32, #shared1>, %arg22: i32, %arg23: i32, %arg24: i64, %arg25: i64, %arg26: !tt.tensordesc<128x64xf16, #shared>, %arg27: i32, %arg28: i32, %arg29: i64, %arg30: i64, %arg31: !tt.tensordesc<128x64xf16, #shared>, %arg32: i32, %arg33: i32, %arg34: i64, %arg35: i64, %arg36: !tt.tensordesc<128xf32, #shared2>, %arg37: i32, %arg38: i64, %arg39: !tt.tensordesc<128xf32, #shared2>, %arg40: i32, %arg41: i64, %arg42: i32 {tt.divisibility = 16 : i32}, %arg43: i32 {tt.divisibility = 16 : i32}, %arg44: i32 {tt.divisibility = 16 : i32}, %arg45: i32, %arg46: i32 {tt.divisibility = 16 : i32}, %arg47: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.693147182> : tensor<128x32xf32, #linear>
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 3>} 64 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 127 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 128 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 0 : i32
    %c96_i32 = arith.constant {async_task_id = array<i32: 0>} 96 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0>} 32 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %result, %token = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_0, %token_1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 10 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %2 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 14 : i32} : () -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
    %result_2, %token_3 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %4 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %result_4 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %result_5, %token_6 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %5 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 16 : i32} : () -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
    %6 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %result_7 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %result_8, %token_9 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %7 = arith.addi %arg47, %c127_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %8 = arith.divsi %7, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %9 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %10 = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %11 = arith.muli %8, %arg45 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %12 = arith.muli %11, %arg46 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %13 = arith.divsi %12, %10 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %14 = arith.remsi %12, %10 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %15 = arith.cmpi slt, %9, %14 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %16 = scf.if %15 -> (i32) {
      %29 = arith.addi %13, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %29 : i32
    } else {
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %13 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>}
    %17 = arith.extsi %arg44 {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
    %18 = arith.divsi %arg47, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %19 = tt.splat %arg15 {async_task_id = array<i32: 3>} : f32 -> tensor<128x64xf32, #linear1>
    %20 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 18 : i32, buffer.tmaStaging = 2 : i32} : () -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
    %21 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 18 : i32, buffer.tmaStaging = 2 : i32} : () -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
    %22 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 18 : i32, buffer.tmaStaging = 2 : i32} : () -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
    %23 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 18 : i32, buffer.tmaStaging = 2 : i32} : () -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
    %24 = ttg.local_alloc {allocation.reuseTarget = 3 : i32, allocation.shareGroup = 22 : i32, buffer.copy = 2 : i32, buffer.id = 22 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %25 = ttg.local_alloc {allocation.reuseTarget = 3 : i32, allocation.shareGroup = 22 : i32, buffer.copy = 2 : i32, buffer.id = 22 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %26 = ttg.local_alloc {allocation.reuseTarget = 4 : i32, allocation.shareGroup = 24 : i32, buffer.copy = 2 : i32, buffer.id = 24 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %27 = ttg.local_alloc {allocation.reuseTarget = 4 : i32, allocation.shareGroup = 24 : i32, buffer.copy = 2 : i32, buffer.id = 24 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %28 = scf.for %arg48 = %c0_i32 to %16 step %c1_i32 iter_args(%arg49 = %9) -> (i32)  : i32 {
      %29 = arith.remsi %arg49, %8 {async_task_id = array<i32: 2, 3>} : i32
      %30 = arith.divsi %arg49, %8 {async_task_id = array<i32: 0, 2, 3>} : i32
      %31 = arith.muli %30, %arg47 {async_task_id = array<i32: 2>} : i32
      %32 = arith.extsi %31 {async_task_id = array<i32: 2>} : i32 to i64
      %33 = arith.remsi %30, %arg46 {async_task_id = array<i32: 0, 2, 3>} : i32
      %34 = arith.muli %arg43, %33 {async_task_id = array<i32: 0, 2, 3>} : i32
      %35 = arith.divsi %30, %arg46 {async_task_id = array<i32: 0, 2, 3>} : i32
      %36 = arith.muli %arg42, %35 {async_task_id = array<i32: 0, 2, 3>} : i32
      %37 = arith.addi %34, %36 {async_task_id = array<i32: 0, 2, 3>} : i32
      %38 = arith.extsi %37 {async_task_id = array<i32: 0, 2, 3>} : i32 to i64
      %39 = arith.divsi %38, %17 {async_task_id = array<i32: 0, 2, 3>} : i64
      %40 = arith.muli %29, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %41 = arith.extsi %40 {async_task_id = array<i32: 2, 3>} : i32 to i64
      %42 = arith.addi %39, %41 {async_task_id = array<i32: 2, 3>} : i64
      %43 = arith.trunci %42 {async_task_id = array<i32: 2, 3>} : i64 to i32
      %44 = tt.descriptor_load %arg5[%43, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked>
      ttg.local_store %44, %0 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %45 = tt.descriptor_load %arg10[%43, %c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked>
      ttg.local_store %45, %1 {async_task_id = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %46:7 = scf.for %arg50 = %c0_i32 to %18 step %c1_i32 iter_args(%arg51 = %c0_i32, %arg52 = %false, %arg53 = %token_3, %arg54 = %token_6, %arg55 = %token, %arg56 = %token_1, %arg57 = %token_9) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %66 = arith.extsi %arg51 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 to i64
        %67 = arith.addi %39, %66 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %68 = arith.trunci %67 {async_task_id = array<i32: 0, 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
        %69 = tt.descriptor_load %arg0[%68, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked>
        ttg.local_store %69, %2 {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %70 = ttg.memdesc_trans %2 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
        %71 = arith.addi %32, %66 {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %72 = arith.trunci %71 {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
        %73 = tt.descriptor_load %arg36[%72] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128xf32, #shared2> -> tensor<128xf32, #blocked1>
        ttg.local_store %73, %3 {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked1> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
        %74 = ttng.tc_gen5_mma %0, %70, %result_2[%arg53], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %75 = ttg.local_load %3 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128xf32, #shared2, #smem, mutable> -> tensor<128xf32, #blocked1>
        %76 = ttg.convert_layout %75 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked1> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear2}>>
        %77 = tt.expand_dims %76 {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear2}>> -> tensor<1x128xf32, #linear2>
        %78 = tt.broadcast %77 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<1x128xf32, #linear2> -> tensor<128x128xf32, #linear2>
        %result_16, %token_17 = ttng.tmem_load %result_2[%74] {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear2>
        %79 = arith.subf %result_16, %78 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #linear2>
        %80 = math.exp2 %79 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #linear2>
        %81 = tt.descriptor_load %arg16[%68, %c0_i32] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked>
        ttg.local_store %81, %4 {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %82 = arith.truncf %80 {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #linear2> to tensor<128x128xf16, #linear2>
        ttng.tmem_store %82, %result_4, %true {async_task_id = array<i32: 3>, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf16, #linear2> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %83 = ttg.memdesc_trans %4 {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
        %84 = ttng.tc_gen5_mma %1, %83, %result_5[%arg54], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %85 = tt.descriptor_load %arg39[%72] {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128xf32, #shared2> -> tensor<128xf32, #blocked1>
        ttg.local_store %85, %5 {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked1> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
        %86 = ttng.tc_gen5_mma %result_4, %4, %result[%arg55], %arg52, %true {async_task_id = array<i32: 1>, loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.start = array<i32: 2>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %87 = ttg.local_load %5 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128xf32, #shared2, #smem, mutable> -> tensor<128xf32, #blocked1>
        %88 = ttg.convert_layout %87 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128xf32, #blocked1> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear2}>>
        %89 = tt.expand_dims %88 {async_task_id = array<i32: 3>, axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear2}>> -> tensor<1x128xf32, #linear2>
        %90 = tt.broadcast %89 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<1x128xf32, #linear2> -> tensor<128x128xf32, #linear2>
        %result_18, %token_19 = ttng.tmem_load %result_5[%84] {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear2>
        %91 = arith.subf %result_18, %90 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear2>
        %92 = arith.mulf %80, %91 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear2>
        %93 = arith.truncf %92 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear2> to tensor<128x128xf16, #linear2>
        ttg.local_store %93, %6 {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #linear2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        ttng.tmem_store %93, %result_7, %true {async_task_id = array<i32: 3>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #linear2> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %94 = ttng.tc_gen5_mma %result_7, %2, %result_0[%arg56], %arg52, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.start = array<i32: 3>, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,tmem,1,5\22, \22opndD,tmem,1,10\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %95 = ttg.memdesc_trans %6 {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
        %96 = ttng.tc_gen5_mma %95, %0, %result_8[%arg57], %false, %true {async_task_id = array<i32: 1>, loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}"} : !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %result_20, %token_21 = ttng.tmem_load %result_8[%96] {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear2>
        %97 = tt.reshape %result_20 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #linear2> -> tensor<128x2x64xf32, #linear3>
        %98 = tt.trans %97 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear3> -> tensor<128x64x2xf32, #linear4>
        %outLHS_22, %outRHS_23 = tt.split %98 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64x2xf32, #linear4> -> tensor<128x64xf32, #linear1>
        %99 = tt.reshape %outLHS_22 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear1> -> tensor<128x2x32xf32, #linear5>
        %100 = tt.trans %99 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
        %outLHS_24, %outRHS_25 = tt.split %100 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear>
        %101 = tt.reshape %outRHS_23 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #linear1> -> tensor<128x2x32xf32, #linear5>
        %102 = tt.trans %101 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
        %outLHS_26, %outRHS_27 = tt.split %102 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear>
        %103 = arith.mulf %outLHS_24, %cst {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #linear>
        %104 = ttg.convert_layout %103 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #linear> -> tensor<128x32xf32, #blocked2>
        ttg.local_store %104, %20 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked2> -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
        %105 = ttng.async_tma_reduce add, %arg21[%68, %c0_i32] %20 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x32xf32, #shared1>, !ttg.memdesc<128x32xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %105   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token
        %106 = arith.mulf %outRHS_25, %cst {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #linear>
        %107 = ttg.convert_layout %106 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #linear> -> tensor<128x32xf32, #blocked2>
        ttg.local_store %107, %21 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked2> -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
        %108 = ttng.async_tma_reduce add, %arg21[%68, %c32_i32] %21 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x32xf32, #shared1>, !ttg.memdesc<128x32xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %108   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token
        %109 = arith.mulf %outLHS_26, %cst {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #linear>
        %110 = ttg.convert_layout %109 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #linear> -> tensor<128x32xf32, #blocked2>
        ttg.local_store %110, %22 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked2> -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
        %111 = ttng.async_tma_reduce add, %arg21[%68, %c64_i32] %22 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x32xf32, #shared1>, !ttg.memdesc<128x32xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %111   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token
        %112 = arith.mulf %outRHS_27, %cst {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #linear>
        %113 = ttg.convert_layout %112 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #linear> -> tensor<128x32xf32, #blocked2>
        ttg.local_store %113, %23 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked2> -> !ttg.memdesc<128x32xf32, #shared1, #smem, mutable>
        %114 = ttng.async_tma_reduce add, %arg21[%68, %c96_i32] %23 {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<128x32xf32, #shared1>, !ttg.memdesc<128x32xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %114   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.async.token
        %115 = arith.addi %arg51, %c128_i32 {async_task_id = array<i32: 0, 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %115, %true, %token_17, %token_19, %86, %94, %token_21 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 1 : i32}
      %result_10, %token_11 = ttng.tmem_load %result[%46#4] {async_task_id = array<i32: 3>, tmem.end = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear2>
      %47 = tt.reshape %result_10 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear2> -> tensor<128x2x64xf32, #linear3>
      %48 = tt.trans %47 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear3> -> tensor<128x64x2xf32, #linear4>
      %outLHS, %outRHS = tt.split %48 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #linear4> -> tensor<128x64xf32, #linear1>
      %49 = arith.truncf %outLHS {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %50 = ttg.convert_layout %49 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked3>
      ttg.local_store %50, %24 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %51 = ttng.async_tma_copy_local_to_global %arg31[%43, %c0_i32] %24 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %51   {async_task_id = array<i32: 3>, can_rotate_by_buffer_count = 2 : i32} : !ttg.async.token
      %52 = arith.truncf %outRHS {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %53 = ttg.convert_layout %52 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked3>
      ttg.local_store %53, %25 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %54 = ttng.async_tma_copy_local_to_global %arg31[%43, %c64_i32] %25 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %54   {async_task_id = array<i32: 3>, can_rotate_by_buffer_count = 2 : i32} : !ttg.async.token
      %result_12, %token_13 = ttng.tmem_load %result_0[%46#5] {async_task_id = array<i32: 3>, tmem.end = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear2>
      %55 = tt.reshape %result_12 {async_task_id = array<i32: 3>} : tensor<128x128xf32, #linear2> -> tensor<128x2x64xf32, #linear3>
      %56 = tt.trans %55 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear3> -> tensor<128x64x2xf32, #linear4>
      %outLHS_14, %outRHS_15 = tt.split %56 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #linear4> -> tensor<128x64xf32, #linear1>
      %57 = arith.mulf %outLHS_14, %19 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1>
      %58 = arith.truncf %57 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %59 = ttg.convert_layout %58 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked3>
      ttg.local_store %59, %26 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %60 = ttng.async_tma_copy_local_to_global %arg26[%43, %c0_i32] %26 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %60   {async_task_id = array<i32: 3>, can_rotate_by_buffer_count = 2 : i32} : !ttg.async.token
      %61 = arith.mulf %outRHS_15, %19 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1>
      %62 = arith.truncf %61 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear1> to tensor<128x64xf16, #linear1>
      %63 = ttg.convert_layout %62 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear1> -> tensor<128x64xf16, #blocked3>
      ttg.local_store %63, %27 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %64 = ttng.async_tma_copy_local_to_global %arg26[%43, %c64_i32] %27 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %64   {async_task_id = array<i32: 3>, can_rotate_by_buffer_count = 2 : i32} : !ttg.async.token
      %65 = arith.addi %arg49, %10 {async_task_id = array<i32: 0, 2, 3>} : i32
      scf.yield {async_task_id = array<i32: 0, 2, 3>} %65 : i32
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 220000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
