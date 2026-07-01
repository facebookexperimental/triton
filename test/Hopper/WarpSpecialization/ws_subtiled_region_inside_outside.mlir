// RUN: triton-opt %s --nvgpu-warp-specialization="generate-subtiled-region=true num-stages=3 smem-budget=232448" | FileCheck %s

// Test: asymmetric (inside -> outside) SMEM channels through an epilogue
// ttng.subtiled_region. This is the EPILOGUE_SUBTILE=2, separate_epilogue_store,
// early_tma_store_lowering=FALSE config (tutorial09 matmul_kernel_tma_persistent_ws):
// the truncf + local_store producers are wrapped in a subtiled_region (task 0),
// but each subtile's consumer (local_load -> convert -> descriptor_store) is a
// FLAT op outside the region (task 2). Both subtiles share one in-body template
// store, so the two per-tile channels (one per outside consumer) collapse onto
// it.
//
// Three things must hold:
//  1. The pass must not crash. Before the fix, the second sibling channel's
//     getSrcOp() returned null after the first sibling's lowering removed the
//     shared per-tile buffer position -> SIGSEGV in insertAsyncComm.
//  2. The producer-side ProducerAcquire/Commit must be emitted exactly ONCE per
//     tile (the shared template is processed by one sibling only); after
//     lowering that yields one wait_barrier/arrive_barrier pair per subtile.
//  3. The producer's tile order must match the consumer's column order, so the
//     subtile halves are NOT swapped: tile 0 = outLHS (leftmost) at flattened
//     slot accumCnt+0 -> first column; tile 1 = outRHS at accumCnt+1 -> second
//     column.

// CHECK-LABEL: @matmul_kernel_tma_persistent_ws
// CHECK: ttg.warp_specialize
//
// Epilogue producer (async_task_id 0). tile 0 is outLHS at slot accumCnt+0
// (the +0 folds in), guarded by exactly one wait/arrive on the dstTask=2 barrier.
// CHECK:      %[[LHS:.*]], %[[RHS:.*]] = tt.split
// CHECK:      arith.truncf %[[LHS]] {async_task_id = array<i32: 0>}
// CHECK:      ttng.wait_barrier {{.*}}WSBarrier = {dstTask = 2 : i32}
// CHECK:      ttg.local_store
// CHECK:      ttng.arrive_barrier {{.*}}WSBarrier = {dstTask = 2 : i32}
//
// tile 1 is outRHS at the DISTINCT slot accumCnt+1.
// CHECK:      arith.addi %[[CNT:.*]], %c1_i64 {async_task_id = array<i32: 0>}
// CHECK:      arith.truncf %[[RHS]] {async_task_id = array<i32: 0>}
// CHECK:      ttng.wait_barrier {{.*}}WSBarrier = {dstTask = 2 : i32}
// CHECK:      ttg.local_store
// CHECK:      ttng.arrive_barrier {{.*}}WSBarrier = {dstTask = 2 : i32}
//
// Epilogue-store consumer (async_task_id 2). The first column's consumer reads
// slot accumCnt+0 (= tile 0 = outLHS); the second column's consumer reads the
// DISTINCT slot accumCnt+1 (= tile 1 = outRHS). Same +0/+1 pairing as the
// producer -> no swap.
// CHECK:      ttng.wait_barrier
// CHECK:      ttg.local_load
// CHECK:      tt.descriptor_store
// CHECK:      arith.addi %{{.*}}, %c1_i64{{.*}} {async_task_id = array<i32: 2>}
// CHECK:      ttng.wait_barrier
// CHECK:      ttg.local_load
// CHECK:      tt.descriptor_store

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent_ws(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
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
        %28 = arith.muli %arg19, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %29 = tt.descriptor_load %arg0[%17, %28] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
        %30 = ttg.local_alloc %29 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %31 = tt.descriptor_load %arg5[%18, %28] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
        %32 = ttg.local_alloc %31 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %33 = ttg.memdesc_trans %32 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %34 = ttng.tc_gen5_mma %30, %33, %result[%arg21], %arg20, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %34 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %result_0, %token_1 = ttng.tmem_load %result[%20#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %21 = tt.reshape %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
      %22 = tt.trans %21 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
      %outLHS, %outRHS = tt.split %22 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>
      %23 = arith.truncf %outLHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> to tensor<128x64xf16, #linear3>
      %24 = ttg.convert_layout %23 {ttg.partition = array<i32: 2>} : tensor<128x64xf16, #linear3> -> tensor<128x64xf16, #blocked>
      tt.descriptor_store %arg10[%17, %18], %24 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked>
      %25 = arith.addi %18, %c64_i32 : i32
      %26 = arith.truncf %outRHS {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear3> to tensor<128x64xf16, #linear3>
      %27 = ttg.convert_layout %26 {ttg.partition = array<i32: 2>} : tensor<128x64xf16, #linear3> -> tensor<128x64xf16, #blocked>
      tt.descriptor_store %arg10[%17, %25], %27 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked>
    } {tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
