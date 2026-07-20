// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=1" | FileCheck %s
//
// Regression for T278685041: a two-MMA chained accumulator sharing one operand-D
// TMEM tile. The HSTU reduce_dq compute fold coalesces the dv and dk_attn dots
// into a single in-place dk accumulator, so two MMAv5 ops write the same
// tmem_alloc as operand D (dv with use_acc=flag, then dk_attn with use_acc=true),
// followed by a cross-partition tmem_load. Both writers execute in the gemm
// partition (task 1); the load runs in a computation partition (task 4 for the
// dk_0 DP half, task 3 for dk_1).
//
// Before the fix, handleOperandD classified operand-D writers by identity
// (== the single representative mmaOp) and rejected the second writer with
//   "handleOperandD: unexpected MMA using same TMEM as operand D".
// After the fix it classifies by role (accumulator == this alloc): both MMAs are
// producer+consumer links in the operand-D chain. Because they share a task, the
// needsChannel check skips a (redundant) MMA->MMA channel -- the same-task W->W
// handoff is left to program order + hardware MMA ordering -- and only the
// cross-partition MMA->tmem_load channels are emitted.
//
// CHECK-LABEL: @_hstu_attn_bwd_redq
//
// dk_0 tile: folded-dv writer (reads opndA tmem channel 2, produces opndD
// channel 3) and dk_attn writer (produces opndD channel 4), both in task 1.
// Neither carries a tmem.end for the other's operand-D channel, i.e. no
// MMA->MMA channel is created for the same-task chain.
// CHECK: ttng.tc_gen5_mma {{.*}}tmem.end = array<i32: 2>, tmem.start = array<i32: 3>{{.*}}ttg.partition = array<i32: 1>
// CHECK: ttng.tc_gen5_mma {{.*}}tmem.start = array<i32: 4>{{.*}}ttg.partition = array<i32: 1>
// The task-4 consumer waits on both writers via the operand-D channels.
// CHECK: ttng.tmem_load {{.*}}tmem.end = array<i32: 3, 4, 5>{{.*}}ttg.partition = array<i32: 4>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_hstu_attn_bwd_redq(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: f32, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg28: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg29: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg30: i32, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {ttg.partition = array<i32: 0>} dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %cst_0 = arith.constant {ttg.partition = array<i32: 0>} 1.44269502 : f32
    %c2_i32 = arith.constant {ttg.partition = array<i32: 0>} 2 : i32
    %c256_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 256 : i32
    %c1_i64 = arith.constant {ttg.partition = array<i32: 0, 2, 3, 4>} 1 : i64
    %c128_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 128 : i32
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 0 : i32
    %c1_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2, 3, 4>} 1 : i32
    %true = arith.constant {ttg.partition = array<i32: 0, 1>} true
    %false = arith.constant {ttg.partition = array<i32: 1>} false
    %0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 24 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 25 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %result, %token = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 14 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_1, %token_2 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 15 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %2 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 26 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %result_3, %token_4 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_5, %token_6 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %3 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 27 : i32} : () -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
    %4 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 28 : i32} : () -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
    %result_7 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>
    %result_8 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>
    %5 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 29 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %result_9, %token_10 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 22 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_11, %token_12 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 23 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %6 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 30 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
    %7 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 31 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
    %8 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %9 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %result_13, %token_14 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 22 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_15, %token_16 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 23 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %10 = tt.get_program_id x {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i32
    %11 = arith.divsi %10, %arg33 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i32
    %12 = arith.remsi %10, %arg33 {ttg.partition = array<i32: 0, 2, 3, 4>} : i32
    %13 = tt.addptr %arg6, %11 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : !tt.ptr<i64>, i32
    %14 = tt.load %13 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : !tt.ptr<i64>
    %15 = arith.trunci %14 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i64 to i32
    %16 = tt.addptr %13, %c1_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : !tt.ptr<i64>, i32
    %17 = tt.load %16 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : !tt.ptr<i64>
    %18 = arith.trunci %17 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i64 to i32
    %19 = arith.subi %18, %15 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i32
    %20 = tt.addptr %arg7, %11 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : !tt.ptr<i64>, i32
    %21 = tt.load %20 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : !tt.ptr<i64>
    %22 = arith.trunci %21 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i64 to i32
    %23 = tt.addptr %20, %c1_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : !tt.ptr<i64>, i32
    %24 = tt.load %23 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : !tt.ptr<i64>
    %25 = arith.trunci %24 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i64 to i32
    %26 = arith.subi %25, %22 {ttg.partition = array<i32: 0, 1, 2, 3, 4>} : i32
    %27 = arith.cmpi eq, %19, %c0_i32 : i32
    cf.cond_br %27, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %28 = arith.muli %arg33, %c128_i32 {ttg.partition = array<i32: 0, 2, 3, 4>} : i32
    %29 = arith.extsi %28 {ttg.partition = array<i32: 0, 2, 3, 4>} : i32 to i64
    %30 = tt.make_tensor_descriptor %arg0, [%arg4, %28], [%29, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<bf16>, !tt.tensordesc<tensor<128x128xbf16, #shared>>
    %31 = tt.make_tensor_descriptor %arg1, [%arg5, %28], [%29, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<bf16>, !tt.tensordesc<tensor<128x128xbf16, #shared>>
    %32 = tt.make_tensor_descriptor %arg1, [%arg5, %28], [%29, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<bf16>, !tt.tensordesc<tensor<128x128xbf16, #shared>>
    %33 = tt.make_tensor_descriptor %arg3, [%arg4, %28], [%29, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<bf16>, !tt.tensordesc<tensor<128x128xbf16, #shared>>
    %34 = tt.make_tensor_descriptor %arg9, [%18, %28], [%29, %c1_i64] {ttg.partition = array<i32: 4>} : !tt.ptr<bf16>, !tt.tensordesc<tensor<128x128xbf16, #shared>>
    %35 = tt.make_tensor_descriptor %arg9, [%18, %28], [%29, %c1_i64] {ttg.partition = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<tensor<128x128xbf16, #shared>>
    %36 = tt.make_tensor_descriptor %arg8, [%25, %28], [%29, %c1_i64] {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, !tt.tensordesc<tensor<128x128xf32, #shared3>>
    %37 = arith.muli %12, %arg14 {ttg.partition = array<i32: 2>} : i32
    %38 = arith.cmpi slt, %12, %c2_i32 {ttg.partition = array<i32: 0>} : i32
    %39:2 = scf.if %38 -> (!tt.ptr<f32>, !tt.ptr<f32>) {
      %64 = tt.addptr %arg28, %12 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      %65 = arith.muli %22, %arg30 {ttg.partition = array<i32: 0>} : i32
      %66 = tt.addptr %64, %65 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      %67 = tt.addptr %arg29, %12 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      %68 = tt.addptr %67, %65 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
      scf.yield {ttg.partition = array<i32: 0>} %68, %66 : !tt.ptr<f32>, !tt.ptr<f32>
    } else {
      scf.yield {ttg.partition = array<i32: 0>} %arg29, %arg28 : !tt.ptr<f32>, !tt.ptr<f32>
    } {ttg.partition = array<i32: 0>}
    %40 = tt.make_range {ttg.partition = array<i32: 0>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %41 = tt.make_range {ttg.partition = array<i32: 0>, end = 256 : i32, start = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %42 = arith.mulf %arg25, %cst_0 {ttg.partition = array<i32: 0>} : f32
    %43 = tt.make_range {ttg.partition = array<i32: 0>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %44 = tt.make_range {ttg.partition = array<i32: 0>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %45 = tt.splat %26 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #blocked>
    %46 = arith.muli %12, %arg12 {ttg.partition = array<i32: 2>} : i32
    %47 = tt.splat %26 {ttg.partition = array<i32: 0>} : i32 -> tensor<1x128xi32, #linear>
    %48 = tt.splat %19 {ttg.partition = array<i32: 0>} : i32 -> tensor<128x1xi32, #linear>
    %49 = tt.splat %19 {ttg.partition = array<i32: 0>} : i32 -> tensor<128x1xi32, #linear>
    %50 = tt.splat %42 {ttg.partition = array<i32: 0>} : f32 -> tensor<128x128xf32, #linear>
    %51 = tt.splat %42 {ttg.partition = array<i32: 0>} : f32 -> tensor<128x128xf32, #linear>
    %52 = tt.splat %arg30 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #blocked>
    %53 = tt.splat %39#1 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %54 = arith.muli %12, %arg18 {ttg.partition = array<i32: 2>} : i32
    %55 = tt.splat %39#0 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %56 = tt.splat %arg25 {ttg.partition = array<i32: 0>} : f32 -> tensor<128x128xf32, #linear>
    %57 = tt.splat %arg25 {ttg.partition = array<i32: 0>} : f32 -> tensor<128x128xf32, #linear>
    %58 = arith.muli %12, %c128_i32 {ttg.partition = array<i32: 0>} : i32
    %59 = arith.muli %12, %c128_i32 {ttg.partition = array<i32: 3, 4>} : i32
    %60 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 34 : i32, buffer.tmaStaging = 2 : i32} : () -> !ttg.memdesc<128x128xf32, #shared3, #smem, mutable>
    %61 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 35 : i32, buffer.tmaStaging = 2 : i32} : () -> !ttg.memdesc<128x128xf32, #shared3, #smem, mutable>
    %62 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 36 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %63 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 37 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    scf.for %arg35 = %c0_i32 to %19 step %c256_i32  : i32 {
      %64 = arith.addi %15, %arg35 {ttg.partition = array<i32: 2, 3, 4>} : i32
      %65 = arith.addi %64, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %66 = arith.addi %64, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %67 = tt.descriptor_load %31[%64, %37] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
      %68 = tt.descriptor_load %32[%66, %37] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
      ttg.local_store %67, %0 {ttg.partition = array<i32: 2>} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      ttg.local_store %68, %1 {ttg.partition = array<i32: 2>} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %69 = tt.splat %arg35 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %70 = tt.splat %arg35 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %71 = arith.addi %69, %40 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %72 = arith.addi %70, %41 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %73 = tt.expand_dims %71 {ttg.partition = array<i32: 0>, axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xi32, #linear>
      %74 = tt.expand_dims %72 {ttg.partition = array<i32: 0>, axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xi32, #linear>
      %75 = arith.cmpi slt, %73, %48 {ttg.partition = array<i32: 0>} : tensor<128x1xi32, #linear>
      %76 = arith.cmpi slt, %74, %49 {ttg.partition = array<i32: 0>} : tensor<128x1xi32, #linear>
      %77 = tt.broadcast %75 {ttg.partition = array<i32: 0>} : tensor<128x1xi1, #linear> -> tensor<128x128xi1, #linear>
      %78 = tt.broadcast %76 {ttg.partition = array<i32: 0>} : tensor<128x1xi1, #linear> -> tensor<128x128xi1, #linear>
      %79 = ttg.memdesc_trans %0 {ttg.partition = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
      %80 = ttg.memdesc_trans %1 {ttg.partition = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
      %81 = ttg.convert_layout %cst {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %82 = ttng.tmem_store %81, %result[%token], %true {ttg.partition = array<i32: 0>, tmem.end = array<i32: 6>, tmem.start = array<i32: 2, 5>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %83 = ttg.convert_layout %cst {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %84 = ttng.tmem_store %83, %result_1[%token_2], %true {ttg.partition = array<i32: 0>, tmem.end = array<i32: 11>, tmem.start = array<i32: 7, 10>} : tensor<128x128xf32, #linear1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %85:9 = scf.for %arg36 = %c0_i32 to %26 step %c128_i32 iter_args(%arg37 = %false, %arg38 = %token_4, %arg39 = %token_10, %arg40 = %82, %arg41 = %token_14, %arg42 = %token_6, %arg43 = %token_12, %arg44 = %84, %arg45 = %token_16) -> (i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %94 = tt.splat %arg36 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %95 = tt.splat %arg36 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #blocked>
        %96 = arith.addi %94, %43 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %97 = arith.addi %95, %44 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
        %98 = arith.cmpi slt, %97, %45 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
        %99 = arith.addi %22, %arg36 {ttg.partition = array<i32: 0, 2>} : i32
        %100 = tt.descriptor_load %30[%99, %46] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
        ttg.local_store %100, %2 {ttg.partition = array<i32: 2>} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
        %101 = ttg.memdesc_trans %2 {ttg.partition = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
        %102 = ttng.tc_gen5_mma %0, %101, %result_3[%arg38], %false, %true {ttg.partition = array<i32: 1>, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,4\22]}"} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %103 = ttng.tc_gen5_mma %1, %101, %result_5[%arg42], %false, %true {ttg.partition = array<i32: 1>, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,5\22]}"} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %104 = tt.expand_dims %96 {ttg.partition = array<i32: 0>, axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xi32, #linear>
        %105 = arith.cmpi slt, %104, %47 {ttg.partition = array<i32: 0>} : tensor<1x128xi32, #linear>
        %106 = tt.broadcast %105 {ttg.partition = array<i32: 0>} : tensor<1x128xi1, #linear> -> tensor<128x128xi1, #linear>
        %107 = tt.broadcast %105 {ttg.partition = array<i32: 0>} : tensor<1x128xi1, #linear> -> tensor<128x128xi1, #linear>
        %108 = arith.andi %106, %77 {ttg.partition = array<i32: 0>} : tensor<128x128xi1, #linear>
        %109 = arith.andi %107, %78 {ttg.partition = array<i32: 0>} : tensor<128x128xi1, #linear>
        %result_21, %token_22 = ttng.tmem_load %result_3[%102] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %110 = ttg.convert_layout %result_21 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %result_23, %token_24 = ttng.tmem_load %result_5[%103] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %111 = ttg.convert_layout %result_23 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %112 = arith.mulf %110, %50 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %113 = arith.mulf %111, %51 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %114 = arith.muli %97, %52 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
        %115 = tt.addptr %53, %114 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
        %116 = tt.descriptor_load %33[%99, %54] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked1>
        %117 = tt.load %115, %98 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
        %118 = ttg.convert_layout %117 {ttg.partition = array<i32: 0>} : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
        %119 = tt.expand_dims %118 {ttg.partition = array<i32: 0>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
        %120 = tt.broadcast %119 {ttg.partition = array<i32: 0>} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %121 = arith.subf %112, %120 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %122 = tt.load %115, %98 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
        %123 = ttg.convert_layout %122 {ttg.partition = array<i32: 0>} : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
        %124 = tt.expand_dims %123 {ttg.partition = array<i32: 0>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
        %125 = tt.broadcast %124 {ttg.partition = array<i32: 0>} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %126 = arith.subf %113, %125 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %127 = math.exp2 %121 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %128 = math.exp2 %126 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %129 = ttg.convert_layout %81 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %130 = arith.select %108, %127, %129 {ttg.partition = array<i32: 0>} : tensor<128x128xi1, #linear>, tensor<128x128xf32, #linear>
        ttg.local_store %130, %3 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
        %131 = ttg.convert_layout %83 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %132 = arith.select %109, %128, %131 {ttg.partition = array<i32: 0>} : tensor<128x128xi1, #linear>, tensor<128x128xf32, #linear>
        ttg.local_store %132, %4 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
        %133 = ttg.local_load %3 {ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf32, #shared1, #smem, mutable> -> tensor<128x128xf32, #linear>
        %134 = arith.truncf %133 {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
        %135 = ttg.local_load %4 {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf32, #shared1, #smem, mutable> -> tensor<128x128xf32, #linear>
        %136 = arith.truncf %135 {ttg.partition = array<i32: 3>} : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
        %137 = ttg.convert_layout %134 {ttg.partition = array<i32: 4>} : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #linear1>
        ttng.tmem_store %137, %result_7, %true {ttg.partition = array<i32: 4>} : tensor<128x128xbf16, #linear1> -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>
        %138 = ttg.convert_layout %136 {ttg.partition = array<i32: 3>} : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #linear1>
        ttng.tmem_store %138, %result_8, %true {ttg.partition = array<i32: 3>} : tensor<128x128xbf16, #linear1> -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>
        ttg.local_store %116, %5 {ttg.partition = array<i32: 2>} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
        %139 = ttg.memdesc_trans %5 {ttg.partition = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
        %140 = ttng.tc_gen5_mma %0, %139, %result_9[%arg39], %false, %true {ttg.partition = array<i32: 1>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndD,tmem,1,22\22]}"} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %141 = ttng.tc_gen5_mma %1, %139, %result_11[%arg43], %false, %true {ttg.partition = array<i32: 1>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndD,tmem,1,23\22]}"} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %142 = ttng.tc_gen5_mma %result_7, %5, %result[%arg40], %arg37, %true {ttg.partition = array<i32: 1>, tmem.end = array<i32: 2>, tmem.start = array<i32: 3>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,4\22, \22opndD,tmem,1,14\22]}"} : !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %143 = ttng.tc_gen5_mma %result_8, %5, %result_1[%arg44], %arg37, %true {ttg.partition = array<i32: 1>, tmem.end = array<i32: 7>, tmem.start = array<i32: 8>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,5\22, \22opndD,tmem,1,15\22]}"} : !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %144 = tt.addptr %55, %114 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
        %result_25, %token_26 = ttng.tmem_load %result_9[%140] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %145 = ttg.convert_layout %result_25 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %result_27, %token_28 = ttng.tmem_load %result_11[%141] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %146 = ttg.convert_layout %result_27 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %147 = tt.load %144, %98 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
        %148 = ttg.convert_layout %147 {ttg.partition = array<i32: 0>} : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
        %149 = tt.expand_dims %148 {ttg.partition = array<i32: 0>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
        %150 = tt.broadcast %149 {ttg.partition = array<i32: 0>} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %151 = arith.subf %145, %150 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %152 = tt.load %144, %98 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
        %153 = ttg.convert_layout %152 {ttg.partition = array<i32: 0>} : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
        %154 = tt.expand_dims %153 {ttg.partition = array<i32: 0>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
        %155 = tt.broadcast %154 {ttg.partition = array<i32: 0>} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %156 = arith.subf %146, %155 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %157 = arith.mulf %130, %151 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %158 = arith.mulf %132, %156 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %159 = arith.mulf %157, %56 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %160 = arith.mulf %158, %57 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
        %161 = arith.truncf %159 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
        ttg.local_store %161, %6 {ttg.partition = array<i32: 0>} : tensor<128x128xbf16, #linear> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
        %162 = arith.truncf %160 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
        ttg.local_store %162, %7 {ttg.partition = array<i32: 0>} : tensor<128x128xbf16, #linear> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
        %163 = ttg.local_load %6 {ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xbf16, #linear>
        ttg.local_store %163, %8 {ttg.partition = array<i32: 4>} : tensor<128x128xbf16, #linear> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
        %164 = ttg.local_load %7 {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xbf16, #linear>
        ttg.local_store %164, %9 {ttg.partition = array<i32: 3>} : tensor<128x128xbf16, #linear> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
        %165 = ttng.tc_gen5_mma %79, %8, %result_13[%arg41], %false, %true {ttg.partition = array<i32: 1>, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndB,smem,1,8\22, \22opndD,tmem,1,22\22]}"} : !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %166 = ttng.tc_gen5_mma %80, %9, %result_15[%arg45], %false, %true {ttg.partition = array<i32: 1>, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndB,smem,1,8\22, \22opndD,tmem,1,23\22]}"} : !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %167 = ttng.tc_gen5_mma %8, %2, %result[%142], %true, %true {ttg.partition = array<i32: 1>, tmem.start = array<i32: 4>, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,14\22]}"} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %168 = ttng.tc_gen5_mma %9, %2, %result_1[%143], %true, %true {ttg.partition = array<i32: 1>, tmem.start = array<i32: 9>, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,15\22]}"} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %result_29, %token_30 = ttng.tmem_load %result_13[%165] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %result_31, %token_32 = ttng.tmem_load %result_15[%166] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %169 = tt.trans %result_29 {ttg.partition = array<i32: 0>, order = array<i32: 1, 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear2>
        %170 = tt.trans %result_31 {ttg.partition = array<i32: 0>, order = array<i32: 1, 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear2>
        %171 = ttg.convert_layout %169 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear2> -> tensor<128x128xf32, #blocked2>
        %172 = ttg.convert_layout %170 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear2> -> tensor<128x128xf32, #blocked2>
        ttg.local_store %171, %60 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked2> -> !ttg.memdesc<128x128xf32, #shared3, #smem, mutable>
        %173 = ttng.async_tma_reduce add, %36[%99, %58] %60 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf32, #shared3>>, !ttg.memdesc<128x128xf32, #shared3, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %173   {ttg.partition = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        ttg.local_store %172, %61 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked2> -> !ttg.memdesc<128x128xf32, #shared3, #smem, mutable>
        %174 = ttng.async_tma_reduce add, %36[%99, %58] %61 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf32, #shared3>>, !ttg.memdesc<128x128xf32, #shared3, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %174   {ttg.partition = array<i32: 3>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1, 3, 4>} %true, %token_22, %token_26, %167, %token_30, %token_24, %token_28, %168, %token_32 : i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1, 2, 3, 4>, tt.merge_epilogue = true}
      %result_17, %token_18 = ttng.tmem_load %result[%85#3] {ttg.partition = array<i32: 4>, tmem.end = array<i32: 3, 4, 5>, tmem.start = array<i32: 6>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %86 = ttg.convert_layout %result_17 {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %result_19, %token_20 = ttng.tmem_load %result_1[%85#7] {ttg.partition = array<i32: 3>, tmem.end = array<i32: 8, 9, 10>, tmem.start = array<i32: 11>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %87 = ttg.convert_layout %result_19 {ttg.partition = array<i32: 3>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %88 = arith.truncf %86 {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %89 = arith.truncf %87 {ttg.partition = array<i32: 3>} : tensor<128x128xf32, #linear> to tensor<128x128xbf16, #linear>
      %90 = ttg.convert_layout %88 {ttg.partition = array<i32: 4>} : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked1>
      %91 = ttg.convert_layout %89 {ttg.partition = array<i32: 3>} : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked1>
      ttg.local_store %90, %62 {ttg.partition = array<i32: 4>} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %92 = ttng.async_tma_copy_local_to_global %34[%64, %59] %62 {ttg.partition = array<i32: 4>} : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %92   {ttg.partition = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
      ttg.local_store %91, %63 {ttg.partition = array<i32: 3>} : tensor<128x128xbf16, #blocked1> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
      %93 = ttng.async_tma_copy_local_to_global %35[%65, %59] %63 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %93   {ttg.partition = array<i32: 3>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1, 2, 3, 4>, tt.data_partition_factor = 2 : i32, tt.merge_correction = true, tt.merge_epilogue_to_computation = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
