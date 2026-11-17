// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=8 smem-budget=232448" | FileCheck %s --check-prefix=CLC
// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=4 smem-budget=232448" | FileCheck %s --check-prefix=FULL
// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=4 smem-budget=150000" | FileCheck %s --check-prefix=TIGHT

// Cross-partition TMA output staging (separate epilogue-store partition) gets
// at most two copies before the operand rings are buffered; it only grows
// further, to a depth aligned with its subtile count, with the budget the
// operand rings leave. Real planner input of persistent bf16 GEMMs.

// 128x128x128 CLC GEMM, num_stages=8, EPILOGUE_SUBTILE=2. Reserving staging
// up to num_stages first gave staging 8 / A 1 / B 1 (every operand ring
// single-buffered, 2.1x slower than separate_epilogue_store=False).
// CLC-LABEL: @clc_gemm_staging_reserve
// CLC-DAG: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<128x64xbf16
// CLC-DAG: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<128x128xbf16
// CLC-DAG: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<128x128xbf16
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @clc_gemm_staging_reserve(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8_i32 = arith.constant {async_task_id = array<i32: 2, 3>} 8 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 2>} 64 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} true
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 128 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 127 : i32
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
    %6 = arith.addi %arg5, %c127_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %7 = arith.divsi %6, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %8 = arith.addi %arg6, %c127_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %9 = arith.divsi %8, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %10 = arith.addi %arg7, %c127_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %11 = arith.divsi %10, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %12 = arith.extsi %arg8 {async_task_id = array<i32: 3>} : i32 to i64
    %13 = tt.make_tensor_descriptor %arg0, [%arg7, %arg5], [%12, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %14 = arith.extsi %arg9 {async_task_id = array<i32: 3>} : i32 to i64
    %15 = tt.make_tensor_descriptor %arg1, [%arg7, %arg6], [%14, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x128xbf16, #shared>
    %16 = arith.extsi %arg10 {async_task_id = array<i32: 2>} : i32 to i64
    %17 = tt.make_tensor_descriptor %arg2, [%arg5, %arg6], [%16, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared>
    %18 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %19 = arith.muli %7, %9 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %20 = arith.muli %9, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
    %21 = scf.while (%arg11 = %true, %arg12 = %18) : (i1, i32) -> i32 {
      scf.condition(%arg11) {async_task_id = array<i32: 0, 1, 2, 3>} %arg12 : i32
    } do {
    ^bb0(%arg11: i32):
      %22 = ttng.clc_try_cancel_async {async_task_id = array<i32: 3>} : !ttg.async.token
      %23 = arith.divsi %arg11, %19 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %24 = arith.remsi %arg11, %19 {async_task_id = array<i32: 2, 3>} : i32
      %25 = arith.divsi %24, %20 {async_task_id = array<i32: 2, 3>} : i32
      %26 = arith.muli %25, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
      %27 = arith.subi %7, %26 {async_task_id = array<i32: 2, 3>} : i32
      %28 = arith.minsi %27, %c8_i32 {async_task_id = array<i32: 2, 3>} : i32
      %29 = arith.remsi %24, %28 {async_task_id = array<i32: 2, 3>} : i32
      %30 = arith.addi %26, %29 {async_task_id = array<i32: 2, 3>} : i32
      %31 = arith.remsi %24, %20 {async_task_id = array<i32: 2, 3>} : i32
      %32 = arith.divsi %31, %28 {async_task_id = array<i32: 2, 3>} : i32
      %33 = arith.muli %30, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %34 = arith.muli %32, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %35 = arith.muli %23, %11 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %36 = arith.addi %23, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %37 = arith.muli %36, %11 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %38:2 = scf.for %arg12 = %35 to %37 step %c1_i32 iter_args(%arg13 = %false, %arg14 = %token) -> (i1, !ttg.async.token)  : i32 {
        %56 = arith.muli %arg12, %c128_i32 {async_task_id = array<i32: 3>, loop.cluster = 7 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %13[%56, %33] 32768 %2 {async_task_id = array<i32: 3>, loop.cluster = 7 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<128x128xbf16, #shared>, i32, i32, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
        %57 = ttg.memdesc_trans %2 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 7 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
        nvws.descriptor_load %15[%56, %34] 32768 %3 {async_task_id = array<i32: 3>, loop.cluster = 7 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<128x128xbf16, #shared>, i32, i32, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
        %58 = ttng.tc_gen5_mma %57, %3, %result[%arg14], %arg13, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 7 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %true, %58 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 7 : i32}
      %result_0, %token_1 = ttng.tmem_load %result[%38#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %39 = tt.reshape %result_0 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
      %40 = tt.trans %39 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
      %outLHS, %outRHS = tt.split %40 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>
      %41 = arith.truncf %outLHS {async_task_id = array<i32: 0>} : tensor<128x64xf32, #linear3> to tensor<128x64xbf16, #linear3>
      %42 = ttg.convert_layout %41 {async_task_id = array<i32: 0>} : tensor<128x64xbf16, #linear3> -> tensor<128x64xbf16, #blocked>
      ttg.local_store %42, %0 {async_task_id = array<i32: 0>} : tensor<128x64xbf16, #blocked> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
      %43 = ttng.async_tma_copy_local_to_global %17[%33, %34] %0 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %43   {async_task_id = array<i32: 2>} : !ttg.async.token
      %44 = arith.truncf %outRHS {async_task_id = array<i32: 0>} : tensor<128x64xf32, #linear3> to tensor<128x64xbf16, #linear3>
      %45 = ttg.convert_layout %44 {async_task_id = array<i32: 0>} : tensor<128x64xbf16, #linear3> -> tensor<128x64xbf16, #blocked>
      %46 = arith.addi %34, %c64_i32 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %45, %1 {async_task_id = array<i32: 0>} : tensor<128x64xbf16, #blocked> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
      %47 = ttng.async_tma_copy_local_to_global %17[%33, %46] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x64xbf16, #shared>, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %47   {async_task_id = array<i32: 2>} : !ttg.async.token
      %isValid, %x, %y, %z = ttng.clc_read %22 {async_task_id = array<i32: 3>} : !ttg.async.token -> i1, i32, i32, i32
      %48 = arith.extui %isValid {async_task_id = array<i32: 3>} : i1 to i32
      %49 = tt.splat %48 {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked1>
      ttg.local_store %49, %5 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked1> -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
      %50 = ttg.local_load %5 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #blocked1>
      %51 = tt.unsplat %50 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked1>
      %52 = arith.trunci %51 {async_task_id = array<i32: 0, 1, 2, 3>} : i32 to i1
      %53 = tt.splat %x {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked1>
      ttg.local_store %53, %4 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked1> -> !ttg.memdesc<1xi32, #shared1, #smem, mutable>
      %54 = ttg.local_load %4 {async_task_id = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #blocked1>
      %55 = tt.unsplat %54 {async_task_id = array<i32: 0, 1, 2, 3>} : tensor<1xi32, #blocked1>
      scf.yield {async_task_id = array<i32: 0, 1, 2, 3>} %52, %55 : i1, i32
    } attributes {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}


// -----

// 128x128x64 GEMM, num_stages=4, EPILOGUE_SUBTILE=4. With room for both, the
// operand rings reach num_stages and staging then grows to 4 (FULL). Under a
// tight budget the operand rings keep num_stages and staging stays at its
// reserve; 3 copies are skipped because they do not divide the 4 subtiles
// (TIGHT).
// FULL-LABEL: @gemm_staging_grows_after_operands
// FULL-COUNT-4: ttg.local_alloc {buffer.copy = 4 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32}
// FULL: ttg.local_alloc {buffer.copy = 4 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<128x64xbf16
// FULL: ttg.local_alloc {buffer.copy = 4 : i32, buffer.id = 5 : i32} : () -> !ttg.memdesc<128x64xbf16
// TIGHT-LABEL: @gemm_staging_grows_after_operands
// TIGHT-COUNT-4: ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32, buffer.tmaStaging = 1 : i32}
// TIGHT: ttg.local_alloc {buffer.copy = 4 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<128x64xbf16
// TIGHT: ttg.local_alloc {buffer.copy = 4 : i32, buffer.id = 5 : i32} : () -> !ttg.memdesc<128x64xbf16
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @gemm_staging_grows_after_operands(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c96_i32 = arith.constant {async_task_id = array<i32: 2>} 96 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 2>} 32 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 1 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 63 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 64 : i32
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 148 : i32
    %c1_i64 = arith.constant {async_task_id = array<i32: 2, 3>} 1 : i64
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 128 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2, 3>} 127 : i32
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
    %6 = arith.addi %arg5, %c127_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %7 = arith.divsi %6, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %8 = arith.addi %arg6, %c127_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %9 = arith.divsi %8, %c128_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %10 = arith.muli %7, %9 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %11 = arith.addi %arg7, %c63_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %12 = arith.divsi %11, %c64_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    %13 = arith.extsi %arg8 {async_task_id = array<i32: 3>} : i32 to i64
    %14 = tt.make_tensor_descriptor %arg0, [%arg5, %arg7], [%13, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared1>
    %15 = arith.extsi %arg9 {async_task_id = array<i32: 3>} : i32 to i64
    %16 = tt.make_tensor_descriptor %arg1, [%arg6, %arg7], [%15, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared1>
    %17 = arith.extsi %arg10 {async_task_id = array<i32: 2>} : i32 to i64
    %18 = tt.make_tensor_descriptor %arg2, [%arg5, %arg6], [%17, %c1_i64] {async_task_id = array<i32: 2>} : !tt.ptr<bf16>, !tt.tensordesc<128x32xbf16, #shared>
    %19 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2, 3>} : i32
    scf.for %arg11 = %19 to %10 step %c148_i32  : i32 {
      %20 = arith.divsi %arg11, %10 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %21 = arith.remsi %arg11, %10 {async_task_id = array<i32: 2, 3>} : i32
      %22 = arith.divsi %21, %9 {async_task_id = array<i32: 2, 3>} : i32
      %23 = arith.subi %7, %22 {async_task_id = array<i32: 2, 3>} : i32
      %24 = arith.minsi %23, %c1_i32 {async_task_id = array<i32: 2, 3>} : i32
      %25 = arith.remsi %21, %24 {async_task_id = array<i32: 2, 3>} : i32
      %26 = arith.addi %22, %25 {async_task_id = array<i32: 2, 3>} : i32
      %27 = arith.remsi %21, %9 {async_task_id = array<i32: 2, 3>} : i32
      %28 = arith.divsi %27, %24 {async_task_id = array<i32: 2, 3>} : i32
      %29 = arith.muli %26, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %30 = arith.muli %28, %c128_i32 {async_task_id = array<i32: 2, 3>} : i32
      %31 = arith.muli %20, %12 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %32 = arith.addi %20, %c1_i32 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %33 = arith.muli %32, %12 {async_task_id = array<i32: 0, 1, 2, 3>} : i32
      %34:2 = scf.for %arg12 = %31 to %33 step %c1_i32 iter_args(%arg13 = %false, %arg14 = %token) -> (i1, !ttg.async.token)  : i32 {
        %56 = arith.muli %arg12, %c64_i32 {async_task_id = array<i32: 3>, loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
        nvws.descriptor_load %14[%29, %56] 16384 %4 {async_task_id = array<i32: 3>, loop.cluster = 3 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<128x64xbf16, #shared1>, i32, i32, !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
        nvws.descriptor_load %16[%30, %56] 16384 %5 {async_task_id = array<i32: 3>, loop.cluster = 3 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<128x64xbf16, #shared1>, i32, i32, !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
        %57 = ttg.memdesc_trans %5 {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 3 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared2, #smem, mutable>
        %58 = ttng.tc_gen5_mma %4, %57, %result[%arg14], %arg13, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1>} %true, %58 : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2, 3>, tt.scheduled_max_stage = 3 : i32}
      %result_0, %token_1 = ttng.tmem_load %result[%34#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %35 = tt.reshape %result_0 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear1>
      %36 = tt.trans %35 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear1> -> tensor<128x64x2xf32, #linear2>
      %outLHS, %outRHS = tt.split %36 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #linear2> -> tensor<128x64xf32, #linear3>
      %37 = tt.reshape %outLHS {async_task_id = array<i32: 0>} : tensor<128x64xf32, #linear3> -> tensor<128x2x32xf32, #linear4>
      %38 = tt.trans %37 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear4> -> tensor<128x32x2xf32, #linear5>
      %outLHS_2, %outRHS_3 = tt.split %38 {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #linear5> -> tensor<128x32xf32, #linear6>
      %39 = tt.reshape %outRHS {async_task_id = array<i32: 0>} : tensor<128x64xf32, #linear3> -> tensor<128x2x32xf32, #linear4>
      %40 = tt.trans %39 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear4> -> tensor<128x32x2xf32, #linear5>
      %outLHS_4, %outRHS_5 = tt.split %40 {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #linear5> -> tensor<128x32xf32, #linear6>
      %41 = arith.truncf %outLHS_2 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xbf16, #linear6>
      %42 = ttg.convert_layout %41 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear6> -> tensor<128x32xbf16, #blocked>
      ttg.local_store %42, %0 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #blocked> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %43 = ttng.async_tma_copy_local_to_global %18[%29, %30] %0 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %43   {async_task_id = array<i32: 2>} : !ttg.async.token
      %44 = arith.truncf %outRHS_3 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xbf16, #linear6>
      %45 = ttg.convert_layout %44 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear6> -> tensor<128x32xbf16, #blocked>
      %46 = arith.addi %30, %c32_i32 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %45, %1 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #blocked> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %47 = ttng.async_tma_copy_local_to_global %18[%29, %46] %1 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %47   {async_task_id = array<i32: 2>} : !ttg.async.token
      %48 = arith.truncf %outLHS_4 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xbf16, #linear6>
      %49 = ttg.convert_layout %48 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear6> -> tensor<128x32xbf16, #blocked>
      %50 = arith.addi %30, %c64_i32 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %49, %2 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #blocked> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %51 = ttng.async_tma_copy_local_to_global %18[%29, %50] %2 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %51   {async_task_id = array<i32: 2>} : !ttg.async.token
      %52 = arith.truncf %outRHS_5 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear6> to tensor<128x32xbf16, #linear6>
      %53 = ttg.convert_layout %52 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear6> -> tensor<128x32xbf16, #blocked>
      %54 = arith.addi %30, %c96_i32 {async_task_id = array<i32: 2>} : i32
      ttg.local_store %53, %3 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #blocked> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %55 = ttng.async_tma_copy_local_to_global %18[%29, %54] %3 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %55   {async_task_id = array<i32: 2>} : !ttg.async.token
    } {async_task_id = array<i32: 0, 1, 2, 3>, tt.data_partition_factor = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["epilogue", "gemm", "epilogue_store", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
