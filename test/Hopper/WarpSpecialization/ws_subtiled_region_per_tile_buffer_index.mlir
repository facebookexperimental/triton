// RUN: triton-opt %s --split-input-file --nvgpu-warp-specialization="generate-subtiled-region=true num-stages=3 smem-budget=232448" | FileCheck %s --check-prefixes=CHECK,WHILE

// Test: with EPILOGUE_SUBTILE=4 (numTiles=4) the N epilogue subtiles form a
// reuse group that shares ONE physical SMEM staging buffer (buffer.copy=3) AND
// ONE barrier pair. The physical staging-buffer slot index MUST equal the
// shared barrier's slot, both derived from the per-tile *flattened* count
// flattened = accumCnt + tileIdx, taken % numBuffers, where accumCnt advances by
// numTiles per iteration (the numTiles stride lives on the counter).
//
// The bug: createBufferPost derived the data-buffer index from the generic
// reuse-group stagger (accumCnt + reuseGroupPosition). Because all subtiles feed
// the one ttng.subtiled_region consumer, that position term collapsed to
// {0,1,1,1}, aliasing three of the four 128x32 column blocks onto the same
// physical slot while their barriers lived on distinct slots -- a data race
// (49.5% wrong output in test_tutorial09 matmul_kernel_tma_persistent_ws,
// generate_subtiled_region=True, EPILOGUE_SUBTILE=4). numTiles=2 happened to be
// correct (two slots are always distinct); numTiles>2 needs the flattened count.
//
// Fix: every SMEM-rotation value (the staging-buffer slot AND the shared
// barrier's bufferIdx/phase) is computed INSIDE the tile body from the builtin
// tileIdx, and the numTiles stride lives on the loop-carried counter. lowering
// replaces tileIdx with `arith.constant t`, so the first subtile (tileIdx 0, the
// +0 folds away) flattens to just accumCnt (the counter, which steps by
// numTiles=4 per iteration); the resulting %IDX = (flattened % 3) indexes BOTH
// the 3x128x32 data staging buffer and the 3x1xi64 barrier. This fails on the
// buggy collapsed reuse-position index and on any scheme where the data slot and
// barrier slot diverge.

// CHECK-LABEL: @matmul_kernel_tma_persistent_ws
//
// The epilogue staging buffer is a single shared 3-deep 128x32 alloc.
// CHECK: ttg.local_alloc {allocation.shareGroup = 0 : i32, buffer.copy = 3 : i32, buffer.id = 2 : i32{{.*}}} : () -> !ttg.memdesc<3x128x32xf16
//
// In the epilogue partition (async_task_id = 0): the numTiles factor lives on the
// loop-carried reuse-group counter, which advances by numTiles(4) per iteration
// (`addi %CNT, %c4_i64`). The SAME counter feeds the per-tile slot: tile 0
// (tileIdx 0, the +0 folds) flattens to just %CNT, then % numBuffers(3); the
// resulting %IDX indexes BOTH the 3x128x32 data staging buffer and the 3x1xi64
// barrier (data slot == barrier slot). There is NO in-body `* numTiles`.
// CHECK:      arith.addi %[[CNT:arg[0-9]+]], %c4_i64 {async_task_id = array<i32: 0>}
// CHECK:      %[[DIV:[0-9]+]] = arith.divui %[[CNT]], %c3_i64 {async_task_id = array<i32: 0>}
// CHECK:      %[[MUL:[0-9]+]] = arith.muli %[[DIV]], %c3_i64 {async_task_id = array<i32: 0>}
// CHECK:      %[[MOD:[0-9]+]] = arith.subi %[[CNT]], %[[MUL]] {async_task_id = array<i32: 0>}
// CHECK:      %[[IDX:[0-9]+]] = arith.trunci %[[MOD]] {async_task_id = array<i32: 0>} : i64 to i32
// CHECK:      ttg.memdesc_index %{{[0-9]+}}[%[[IDX]]] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x128x32xf16
// CHECK:      ttg.memdesc_index %{{[0-9]+}}[%[[IDX]]] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64
//
// Tiles 1-3 flatten to %CNT + {1,2,3}: four DISTINCT generations of the shared
// buffer (the buggy generic reuse-position stagger collapsed these to {0,1,1,1},
// aliasing three subtiles onto one slot -> the 49.5%-wrong data race).
// CHECK:      arith.addi %[[CNT]], %c1_i64 {async_task_id = array<i32: 0>}
// CHECK:      arith.addi %[[CNT]], %c2_i64 {async_task_id = array<i32: 0>}
// CHECK:      arith.addi %[[CNT]], %c3_i64 {async_task_id = array<i32: 0>}

// Companion coverage for the SAME reuse-group invariant when the persistent
// outer loop is a dynamic scf.while instead of an scf.for. It lives in its own
// split-input-file module (separated by the dashed marker below) because
// doGenerateSubtiledRegion runs its nested pass over the whole enclosing module;
// keeping the two subtiled-region kernels in one module would make that pass
// process both at once. This function is the
// real pre-warp-specialization IR captured immediately before
// nvgpu-warp-specialization for the passing Python configuration
// test_tutorial09_matmul_tma_unified_persistent_while_loop_warp_specialize with
// generate_subtiled_region=True and separate_epilogue_store=True (the
// separate-epilogue-store variant is the one that forms the shared reuse group,
// matching the scf.for coverage above). The generated same-task
// ttng.subtiled_region nested directly in the persistent scf.while must resolve
// its channel counter to the while's loop-carried accumulation counter, NOT
// assign an index to the ttng.subtiled_region itself. The while carries trailing
// i64 accumulation counters; the shared reuse-group counter advances by
// numTiles(4) per persistent iteration and each of the four subtiles flattens to
// that counter + {0,1,2,3}, taken % numBuffers(3), indexing BOTH the shared
// 3-deep staging buffer and its barrier at the same slot.
// WHILE-LABEL: @matmul_kernel_tma_persistent_ws_while
//
// The epilogue staging buffer is a single shared 3-deep 128x32 alloc.
// WHILE: ttg.local_alloc {allocation.shareGroup = 0 : i32, buffer.copy = 3 : i32, buffer.id = 2 : i32{{.*}}} : () -> !ttg.memdesc<3x128x32xf16
//
// The kernel is physically warp specialized.
// WHILE: ttg.warp_specialize
//
// The dynamic persistent outer loop stays an scf.while and receives the trailing
// i64 accumulation-counter carries. The reuse-group counter is the last i64.
// WHILE: scf.while
// WHILE: ^bb0({{.*}}%[[CNT:arg[0-9]+]]: i64):
//
// The reuse-group counter advances by numTiles(4) per persistent iteration.
// WHILE:      arith.addi %[[CNT]], %c4_i64 {async_task_id = array<i32: 0>}
//
// Subtile 0 (flattened = %CNT + 0) mod numBuffers(3) indexes BOTH the 3x128x32
// data staging buffer and the 3x1xi64 barrier at the SAME slot (data slot ==
// barrier slot), all derived from the while's loop-carried counter.
// WHILE:      %[[DIV:[0-9]+]] = arith.divui %[[CNT]], %c3_i64 {async_task_id = array<i32: 0>}
// WHILE:      %[[MUL:[0-9]+]] = arith.muli %[[DIV]], %c3_i64 {async_task_id = array<i32: 0>}
// WHILE:      %[[MOD:[0-9]+]] = arith.subi %[[CNT]], %[[MUL]] {async_task_id = array<i32: 0>}
// WHILE:      %[[IDX:[0-9]+]] = arith.trunci %[[MOD]] {async_task_id = array<i32: 0>} : i64 to i32
// WHILE:      ttg.memdesc_index %{{[0-9]+}}[%[[IDX]]] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x128x32xf16
// WHILE:      ttg.memdesc_index %{{[0-9]+}}[%[[IDX]]] {async_task_id = array<i32: 0>} : !ttg.memdesc<3x1xi64
//
// Subtiles 1-3 flatten to %CNT + {1,2,3}: four DISTINCT generations of the shared
// buffer per persistent iteration.
// WHILE:      arith.addi %[[CNT]], %c1_i64 {async_task_id = array<i32: 0>}
// WHILE:      arith.addi %[[CNT]], %c2_i64 {async_task_id = array<i32: 0>}
// WHILE:      arith.addi %[[CNT]], %c3_i64 {async_task_id = array<i32: 0>}

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  tt.func public @matmul_kernel_tma_persistent_ws(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<128x64xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x32xf16, #shared1>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c96_i32 = arith.constant 96 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg15, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg16, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg17, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.muli %4, %c8_i32 : i32
    scf.for %arg18 = %0 to %7 step %c148_i32  : i32 {
      %9 = arith.divsi %arg18, %8 : i32
      %10 = arith.muli %9, %c8_i32 : i32
      %11 = arith.subi %2, %10 : i32
      %12 = arith.minsi %11, %c8_i32 : i32
      %13 = arith.remsi %arg18, %12 : i32
      %14 = arith.addi %10, %13 : i32
      %15 = arith.remsi %arg18, %8 : i32
      %16 = arith.divsi %15, %12 : i32
      %17 = arith.muli %14, %c128_i32 : i32
      %18 = arith.muli %16, %c128_i32 : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %19 = ttng.tmem_store %cst, %result[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %20:2 = scf.for %arg19 = %c0_i32 to %6 step %c1_i32 iter_args(%arg20 = %false, %arg21 = %19) -> (i1, !ttg.async.token)  : i32 {
        %32 = arith.muli %arg19, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %33 = tt.descriptor_load %arg0[%17, %32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %34 = ttg.local_alloc %33 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %35 = tt.descriptor_load %arg5[%18, %32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %36 = ttg.local_alloc %35 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %37 = ttg.memdesc_trans %36 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared3, #smem>
        %38 = ttng.tc_gen5_mma %34, %37, %result[%arg21], %arg20, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared3, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %38 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %result_0, %token_1 = ttng.tmem_load %result[%20#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %21 = tt.reshape %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
      %22 = tt.trans %21 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
      %outLHS, %outRHS = tt.split %22 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>
      %r3 = tt.reshape %outLHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> -> tensor<128x2x32xf32, #linear4>
      %r4 = tt.trans %r3 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #linear4> -> tensor<128x32x2xf32, #linear5>
      %t0, %t1 = tt.split %r4 {ttg.partition = array<i32: 0>} : tensor<128x32x2xf32, #linear5> -> tensor<128x32xf32, #linear6>
      %r5 = tt.reshape %outRHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> -> tensor<128x2x32xf32, #linear4>
      %r6 = tt.trans %r5 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #linear4> -> tensor<128x32x2xf32, #linear5>
      %t2, %t3 = tt.split %r6 {ttg.partition = array<i32: 0>} : tensor<128x32x2xf32, #linear5> -> tensor<128x32xf32, #linear6>
      %cn1 = arith.addi %18, %c32_i32 : i32
      %cn2 = arith.addi %18, %c64_i32 : i32
      %cn3 = arith.addi %18, %c96_i32 : i32
      %f0 = arith.truncf %t0 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xf16, #linear6>
      %c0 = ttg.convert_layout %f0 {ttg.partition = array<i32: 0>} : tensor<128x32xf16, #linear6> -> tensor<128x32xf16, #blocked>
      %a0 = ttg.local_alloc %c0 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %s0 = ttng.async_tma_copy_local_to_global %arg10[%17, %18] %a0 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x32xf16, #shared1>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %s0   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %f1 = arith.truncf %t1 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xf16, #linear6>
      %c1 = ttg.convert_layout %f1 {ttg.partition = array<i32: 0>} : tensor<128x32xf16, #linear6> -> tensor<128x32xf16, #blocked>
      %a1 = ttg.local_alloc %c1 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %s1 = ttng.async_tma_copy_local_to_global %arg10[%17, %cn1] %a1 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x32xf16, #shared1>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %s1   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %f2 = arith.truncf %t2 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xf16, #linear6>
      %c2 = ttg.convert_layout %f2 {ttg.partition = array<i32: 0>} : tensor<128x32xf16, #linear6> -> tensor<128x32xf16, #blocked>
      %a2 = ttg.local_alloc %c2 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %s2 = ttng.async_tma_copy_local_to_global %arg10[%17, %cn2] %a2 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x32xf16, #shared1>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %s2   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %f3 = arith.truncf %t3 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xf16, #linear6>
      %c3 = ttg.convert_layout %f3 {ttg.partition = array<i32: 0>} : tensor<128x32xf16, #linear6> -> tensor<128x32xf16, #blocked>
      %a3 = ttg.local_alloc %c3 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %s3 = ttng.async_tma_copy_local_to_global %arg10[%17, %cn3] %a3 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x32xf16, #shared1>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %s3   {ttg.partition = array<i32: 2>} : !ttg.async.token
    } {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent_ws_while(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<128x64xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x32xf16, #shared1>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %c127_i32 = arith.constant 127 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c96_i32 = arith.constant 96 : i32
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = arith.addi %arg16, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %2 = arith.addi %arg17, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.addi %arg18, %c63_i32 : i32
    %5 = arith.divsi %4, %c64_i32 : i32
    %6 = arith.muli %3, %c8_i32 : i32
    %7 = tt.get_program_id x : i32
    %8 = arith.muli %1, %3 : i32
    %9 = scf.while (%arg19 = %7) : (i32) -> i32 {
      %10 = arith.cmpi slt, %arg19, %8 : i32
      scf.condition(%10) %arg19 : i32
    } do {
    ^bb0(%arg19: i32):
      %10 = arith.divsi %arg19, %6 : i32
      %11 = arith.muli %10, %c8_i32 : i32
      %12 = arith.subi %1, %11 : i32
      %13 = arith.minsi %12, %c8_i32 : i32
      %14 = arith.remsi %arg19, %13 : i32
      %15 = arith.addi %11, %14 : i32
      %16 = arith.remsi %arg19, %6 : i32
      %17 = arith.divsi %16, %13 : i32
      %18 = arith.muli %15, %c128_i32 : i32
      %19 = arith.muli %17, %c128_i32 : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %20 = ttng.tmem_store %cst, %result[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %21:2 = scf.for %arg20 = %c0_i32 to %5 step %c1_i32 iter_args(%arg21 = %false, %arg22 = %20) -> (i1, !ttg.async.token)  : i32 {
        %48 = arith.muli %arg20, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %49 = tt.descriptor_load %arg0[%18, %48] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %50 = ttg.local_alloc %49 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %51 = tt.descriptor_load %arg5[%19, %48] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %52 = ttg.local_alloc %51 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %53 = ttg.memdesc_trans %52 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared2, #smem>
        %54 = ttng.tc_gen5_mma %50, %53, %result[%arg22], %arg21, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %54 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %result_0, %token_1 = ttng.tmem_load %result[%21#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %22 = tt.reshape %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
      %23 = tt.trans %22 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
      %outLHS, %outRHS = tt.split %23 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>
      %24 = tt.reshape %outLHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> -> tensor<128x2x32xf32, #linear4>
      %25 = tt.trans %24 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #linear4> -> tensor<128x32x2xf32, #linear5>
      %outLHS_2, %outRHS_3 = tt.split %25 {ttg.partition = array<i32: 0>} : tensor<128x32x2xf32, #linear5> -> tensor<128x32xf32, #linear6>
      %26 = tt.reshape %outRHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> -> tensor<128x2x32xf32, #linear4>
      %27 = tt.trans %26 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #linear4> -> tensor<128x32x2xf32, #linear5>
      %outLHS_4, %outRHS_5 = tt.split %27 {ttg.partition = array<i32: 0>} : tensor<128x32x2xf32, #linear5> -> tensor<128x32xf32, #linear6>
      %28 = arith.truncf %outLHS_2 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xf16, #linear6>
      %29 = ttg.convert_layout %28 {ttg.partition = array<i32: 0>} : tensor<128x32xf16, #linear6> -> tensor<128x32xf16, #blocked1>
      %30 = ttg.local_alloc %29 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked1>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %31 = ttng.async_tma_copy_local_to_global %arg10[%18, %19] %30 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x32xf16, #shared1>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %31   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %32 = arith.addi %19, %c32_i32 : i32
      %33 = arith.truncf %outRHS_3 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xf16, #linear6>
      %34 = ttg.convert_layout %33 {ttg.partition = array<i32: 0>} : tensor<128x32xf16, #linear6> -> tensor<128x32xf16, #blocked1>
      %35 = ttg.local_alloc %34 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked1>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %36 = ttng.async_tma_copy_local_to_global %arg10[%18, %32] %35 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x32xf16, #shared1>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %36   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %37 = arith.addi %19, %c64_i32 : i32
      %38 = arith.truncf %outLHS_4 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xf16, #linear6>
      %39 = ttg.convert_layout %38 {ttg.partition = array<i32: 0>} : tensor<128x32xf16, #linear6> -> tensor<128x32xf16, #blocked1>
      %40 = ttg.local_alloc %39 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked1>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %41 = ttng.async_tma_copy_local_to_global %arg10[%18, %37] %40 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x32xf16, #shared1>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %41   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %42 = arith.addi %19, %c96_i32 : i32
      %43 = arith.truncf %outRHS_5 {ttg.partition = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xf16, #linear6>
      %44 = ttg.convert_layout %43 {ttg.partition = array<i32: 0>} : tensor<128x32xf16, #linear6> -> tensor<128x32xf16, #blocked1>
      %45 = ttg.local_alloc %44 {ttg.partition = array<i32: 0>} : (tensor<128x32xf16, #blocked1>) -> !ttg.memdesc<128x32xf16, #shared1, #smem, mutable>
      %46 = ttng.async_tma_copy_local_to_global %arg10[%18, %42] %45 {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x32xf16, #shared1>, !ttg.memdesc<128x32xf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %46   {ttg.partition = array<i32: 2>} : !ttg.async.token
      %47 = tt.atomic_rmw add, acq_rel, gpu, %arg15, %c1_i32, %true : (!tt.ptr<i32>, i32, i1) -> i32
      scf.yield %47 : i32
    } attributes {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
