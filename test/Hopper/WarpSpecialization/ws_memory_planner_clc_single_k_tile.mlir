// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-memory-planner="num-buffers=4 smem-alloc-algo=1 smem-budget=232448 reserve-auxiliary-smem=1" | FileCheck %s

// A CLC persistent GEMM whose K loop has a single tile: the TMA operand loads
// and the MMA sit directly in the persistent scf.while (no inner scf.for).
// That while is the innermost loop, so the operand buffers are multi-buffered
// across tiles instead of being left at a single copy.
// CHECK-LABEL: @clc_single_k_tile
// CHECK-DAG: ttg.local_alloc {buffer.copy = {{[2-9]}} : i32, buffer.id = {{[0-9]+}} : i32} : () -> !ttg.memdesc<64x128xbf16
// CHECK-DAG: ttg.local_alloc {buffer.copy = {{[2-9]}} : i32, buffer.id = {{[0-9]+}} : i32} : () -> !ttg.memdesc<128x64xbf16
// CHECK-DAG: ttg.local_alloc {buffer.copy = {{[2-9]}} : i32, buffer.id = {{[0-9]+}} : i32} : () -> !ttg.memdesc<128x64xbf16
// CHECK: scf.while

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @clc_single_k_tile(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i64, %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4096_i32 = arith.constant {async_task_id = array<i32: 1, 3>} 4096 : i32
    %c4096_i64 = arith.constant {async_task_id = array<i32: 1, 3>} 4096 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 1, 3>} 1 : i64
    %c64_i64 = arith.constant {async_task_id = array<i32: 1, 3>} 64 : i64
    %c8_i64 = arith.constant {async_task_id = array<i32: 0, 1, 3>} 8 : i64
    %true = arith.constant {async_task_id = array<i32: 0, 1, 3, 4, 5>} true
    %c256_i64 = arith.constant {async_task_id = array<i32: 0, 1, 3>} 256 : i64
    %c128_i64 = arith.constant {async_task_id = array<i32: 1, 3>} 128 : i64
    %c32_i64 = arith.constant {async_task_id = array<i32: 1>} 32 : i64
    %c96_i64 = arith.constant {async_task_id = array<i32: 1>} 96 : i64
    %c0_i64 = arith.constant 0 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 3>} 64 : i32
    %c255_i64 = arith.constant {async_task_id = array<i32: 0, 1, 3>} 255 : i64
    %c0_i32 = arith.constant {async_task_id = array<i32: 3>} 0 : i32
    %false = arith.constant {async_task_id = array<i32: 4>} false
    %c128_i32 = arith.constant {async_task_id = array<i32: 1, 3>} 128 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %result_0, %token_1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %4 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
    %9 = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared2, #smem, mutable>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared2, #smem, mutable>
    %13 = arith.trunci %arg3 {async_task_id = array<i32: 1, 3>} : i64 to i32
    %14 = arith.muli %arg3, %c4096_i64 : i64
    %15 = arith.cmpi eq, %14, %c0_i64 : i64
    %16 = tt.make_tensor_descriptor %arg2, [%13, %c4096_i32], [%c4096_i64, %c1_i64] {async_task_id = array<i32: 1>} : !tt.ptr<bf16>, !tt.tensordesc<128x32xbf16, #shared>
    %17 = tt.make_tensor_descriptor %arg2, [%13, %c4096_i32], [%c4096_i64, %c1_i64] {async_task_id = array<i32: 1>} : !tt.ptr<bf16>, !tt.tensordesc<128x32xbf16, #shared>
    cf.cond_br %15, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %18 = arith.addi %arg3, %c255_i64 {async_task_id = array<i32: 0, 1, 3>} : i64
    %19 = arith.divsi %18, %c256_i64 {async_task_id = array<i32: 0, 1, 3>} : i64
    %20 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c4096_i32], [%c4096_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x128xbf16, #shared1>
    %21 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 3>} : i32
    %22 = tt.make_tensor_descriptor %arg0, [%13, %c64_i32], [%c64_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared1>
    %23 = tt.make_tensor_descriptor %arg0, [%13, %c64_i32], [%c64_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared1>
    %24 = scf.while (%arg5 = %true, %arg6 = %21) : (i1, i32) -> i32 {
      scf.condition(%arg5) {async_task_id = array<i32: 0, 1, 3, 4, 5>} %arg6 : i32
    } do {
    ^bb0(%arg5: i32):
      %25 = ttng.clc_try_cancel_async {async_task_id = array<i32: 3>} : !ttg.async.token
      %26 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %27 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %28 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %29 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %30 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %31 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %32 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %33 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %34 = arith.extsi %arg5 {async_task_id = array<i32: 1>} : i32 to i64
      %35 = arith.divsi %27, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %36 = arith.divsi %28, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %37 = arith.divsi %30, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %38 = arith.divsi %31, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %39 = arith.divsi %32, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %40 = arith.divsi %34, %c256_i64 {async_task_id = array<i32: 1>} : i64
      %41 = arith.muli %36, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %42 = arith.muli %35, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %43 = arith.muli %38, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %44 = arith.muli %37, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %45 = arith.muli %39, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %46 = arith.muli %40, %c8_i64 {async_task_id = array<i32: 1>} : i64
      %47 = arith.subi %19, %42 {async_task_id = array<i32: 3>} : i64
      %48 = arith.subi %19, %44 {async_task_id = array<i32: 3>} : i64
      %49 = arith.subi %19, %45 {async_task_id = array<i32: 3>} : i64
      %50 = arith.subi %19, %46 {async_task_id = array<i32: 1>} : i64
      %51 = arith.minsi %47, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %52 = arith.minsi %48, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %53 = arith.minsi %49, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %54 = arith.minsi %50, %c8_i64 {async_task_id = array<i32: 1>} : i64
      %55 = arith.remsi %26, %51 {async_task_id = array<i32: 3>} : i64
      %56 = arith.remsi %29, %52 {async_task_id = array<i32: 3>} : i64
      %57 = arith.remsi %34, %54 {async_task_id = array<i32: 1>} : i64
      %58 = arith.addi %41, %55 {async_task_id = array<i32: 3>} : i64
      %59 = arith.addi %43, %56 {async_task_id = array<i32: 3>} : i64
      %60 = arith.addi %46, %57 {async_task_id = array<i32: 1>} : i64
      %61 = arith.remsi %33, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %62 = arith.remsi %34, %c256_i64 {async_task_id = array<i32: 1>} : i64
      %63 = arith.divsi %61, %53 {async_task_id = array<i32: 3>} : i64
      %64 = arith.divsi %62, %54 {async_task_id = array<i32: 1>} : i64
      %65 = arith.muli %58, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %66 = arith.muli %59, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %67 = arith.muli %60, %c256_i64 {async_task_id = array<i32: 1>} : i64
      %68 = arith.muli %63, %c128_i64 {async_task_id = array<i32: 3>} : i64
      %69 = arith.muli %64, %c128_i64 {async_task_id = array<i32: 1>} : i64
      %70 = arith.trunci %65 {async_task_id = array<i32: 3>} : i64 to i32
      %71 = arith.trunci %66 {async_task_id = array<i32: 3>} : i64 to i32
      %72 = arith.trunci %67 {async_task_id = array<i32: 1>} : i64 to i32
      %73 = arith.addi %72, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %74 = arith.addi %72, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %75 = arith.addi %72, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %76 = arith.addi %72, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %77 = arith.addi %70, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %78 = arith.trunci %68 {async_task_id = array<i32: 3>} : i64 to i32
      %79 = arith.trunci %69 {async_task_id = array<i32: 1>} : i64 to i32
      nvws.descriptor_load %20[%c0_i32, %78] 16384 %8 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x128xbf16, #shared1>, i32, i32, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
      nvws.descriptor_load %22[%71, %c0_i32] 16384 %9 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<128x64xbf16, #shared1>, i32, i32, !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
      nvws.descriptor_load %23[%77, %c0_i32] 16384 %10 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<128x64xbf16, #shared1>, i32, i32, !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
      %80 = arith.addi %69, %c32_i64 {async_task_id = array<i32: 1>} : i64
      %81 = arith.trunci %80 {async_task_id = array<i32: 1>} : i64 to i32
      %82 = arith.addi %69, %c64_i64 {async_task_id = array<i32: 1>} : i64
      %83 = arith.trunci %82 {async_task_id = array<i32: 1>} : i64 to i32
      %84 = arith.addi %69, %c96_i64 {async_task_id = array<i32: 1>} : i64
      %85 = arith.trunci %84 {async_task_id = array<i32: 1>} : i64 to i32
      %isValid, %x, %y, %z = ttng.clc_read %25 {async_task_id = array<i32: 3>} : !ttg.async.token -> i1, i32, i32, i32
      %86 = arith.extui %isValid {async_task_id = array<i32: 3>} : i1 to i32
      %87 = tt.splat %86 {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked>
      ttg.local_store %87, %12 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared2, #smem, mutable>
      %88 = ttg.local_load %12 {async_task_id = array<i32: 0, 1, 3, 4, 5>} : !ttg.memdesc<1xi32, #shared2, #smem, mutable> -> tensor<1xi32, #blocked>
      %89 = tt.unsplat %88 {async_task_id = array<i32: 0, 1, 3, 4, 5>} : tensor<1xi32, #blocked>
      %90 = arith.trunci %89 {async_task_id = array<i32: 0, 1, 3, 4, 5>} : i32 to i1
      %91 = tt.splat %x {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked>
      ttg.local_store %91, %11 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared2, #smem, mutable>
      %92 = ttg.local_load %11 {async_task_id = array<i32: 0, 1, 3, 4, 5>} : !ttg.memdesc<1xi32, #shared2, #smem, mutable> -> tensor<1xi32, #blocked>
      %93 = tt.unsplat %92 {async_task_id = array<i32: 0, 1, 3, 4, 5>} : tensor<1xi32, #blocked>
      %94 = ttng.tc_gen5_mma %9, %8, %result[%token], %false, %true {async_task_id = array<i32: 4>} : !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_2, %token_3 = ttng.tmem_load %result[%94] {async_task_id = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %95 = ttg.convert_layout %result_2 {async_task_id = array<i32: 5>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %96 = tt.reshape %95 {async_task_id = array<i32: 5>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
      %97 = tt.trans %96 {async_task_id = array<i32: 5>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
      %outLHS, %outRHS = tt.split %97 {async_task_id = array<i32: 5>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
      %98 = tt.reshape %outLHS {async_task_id = array<i32: 5>} : tensor<128x64xf32, #linear4> -> tensor<128x2x32xf32, #linear5>
      %99 = tt.trans %98 {async_task_id = array<i32: 5>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
      %outLHS_4, %outRHS_5 = tt.split %99 {async_task_id = array<i32: 5>} : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear7>
      %100 = tt.reshape %outRHS {async_task_id = array<i32: 5>} : tensor<128x64xf32, #linear4> -> tensor<128x2x32xf32, #linear5>
      %101 = tt.trans %100 {async_task_id = array<i32: 5>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
      %outLHS_6, %outRHS_7 = tt.split %101 {async_task_id = array<i32: 5>} : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear7>
      %102 = arith.truncf %outLHS_4 {async_task_id = array<i32: 5>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %103 = ttg.convert_layout %102 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %103, %0 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %104 = ttng.async_tma_copy_local_to_global %16[%72, %79] %0 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %104   {async_task_id = array<i32: 1>} : !ttg.async.token
      %105 = arith.truncf %outRHS_5 {async_task_id = array<i32: 5>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %106 = ttg.convert_layout %105 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %106, %1 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %107 = ttng.async_tma_copy_local_to_global %16[%72, %81] %1 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %107   {async_task_id = array<i32: 1>} : !ttg.async.token
      %108 = arith.truncf %outLHS_6 {async_task_id = array<i32: 5>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %109 = ttg.convert_layout %108 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %109, %2 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %110 = ttng.async_tma_copy_local_to_global %16[%72, %83] %2 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %110   {async_task_id = array<i32: 1>} : !ttg.async.token
      %111 = arith.truncf %outRHS_7 {async_task_id = array<i32: 5>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %112 = ttg.convert_layout %111 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %112, %3 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %113 = ttng.async_tma_copy_local_to_global %16[%72, %85] %3 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %113   {async_task_id = array<i32: 1>} : !ttg.async.token
      %114 = ttng.tc_gen5_mma %10, %8, %result_0[%token_1], %false, %true {async_task_id = array<i32: 4>} : !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_8, %token_9 = ttng.tmem_load %result_0[%114] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %115 = ttg.convert_layout %result_8 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %116 = tt.reshape %115 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
      %117 = tt.trans %116 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
      %outLHS_10, %outRHS_11 = tt.split %117 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
      %118 = tt.reshape %outLHS_10 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x2x32xf32, #linear5>
      %119 = tt.trans %118 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
      %outLHS_12, %outRHS_13 = tt.split %119 {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear7>
      %120 = tt.reshape %outRHS_11 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x2x32xf32, #linear5>
      %121 = tt.trans %120 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
      %outLHS_14, %outRHS_15 = tt.split %121 {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear7>
      %122 = arith.truncf %outLHS_12 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %123 = ttg.convert_layout %122 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %123, %4 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %124 = ttng.async_tma_copy_local_to_global %17[%76, %79] %4 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %124   {async_task_id = array<i32: 1>} : !ttg.async.token
      %125 = arith.truncf %outRHS_13 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %126 = ttg.convert_layout %125 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %126, %5 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %127 = ttng.async_tma_copy_local_to_global %17[%75, %81] %5 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %127   {async_task_id = array<i32: 1>} : !ttg.async.token
      %128 = arith.truncf %outLHS_14 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %129 = ttg.convert_layout %128 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %129, %6 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %130 = ttng.async_tma_copy_local_to_global %17[%74, %83] %6 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %130   {async_task_id = array<i32: 1>} : !ttg.async.token
      %131 = arith.truncf %outRHS_15 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %132 = ttg.convert_layout %131 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %132, %7 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %133 = ttng.async_tma_copy_local_to_global %17[%73, %85] %7 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %133   {async_task_id = array<i32: 1>} : !ttg.async.token
      scf.yield {async_task_id = array<i32: 0, 1, 3, 4, 5>} %90, %93 : i1, i32
    } attributes {async_task_id = array<i32: 0, 1, 3, 4, 5>, tt.data_partition_factor = 2 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["computation", "epilogue", "epilogue_store", "load", "gemm", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// Same kernel with a loop nested in the persistent while: the while is not the
// innermost loop, so the operand buffers stay at one copy as before.
// CHECK-LABEL: @clc_single_k_tile_with_inner_loop
// CHECK-DAG: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = {{[0-9]+}} : i32} : () -> !ttg.memdesc<64x128xbf16
// CHECK-DAG: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = {{[0-9]+}} : i32} : () -> !ttg.memdesc<128x64xbf16
// CHECK-DAG: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = {{[0-9]+}} : i32} : () -> !ttg.memdesc<128x64xbf16
// CHECK: scf.while

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @clc_single_k_tile_with_inner_loop(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i64, %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4096_i32 = arith.constant {async_task_id = array<i32: 1, 3>} 4096 : i32
    %c4096_i64 = arith.constant {async_task_id = array<i32: 1, 3>} 4096 : i64
    %c1_i64 = arith.constant {async_task_id = array<i32: 1, 3>} 1 : i64
    %c64_i64 = arith.constant {async_task_id = array<i32: 1, 3>} 64 : i64
    %c8_i64 = arith.constant {async_task_id = array<i32: 0, 1, 3>} 8 : i64
    %true = arith.constant {async_task_id = array<i32: 0, 1, 3, 4, 5>} true
    %c256_i64 = arith.constant {async_task_id = array<i32: 0, 1, 3>} 256 : i64
    %c128_i64 = arith.constant {async_task_id = array<i32: 1, 3>} 128 : i64
    %c32_i64 = arith.constant {async_task_id = array<i32: 1>} 32 : i64
    %c96_i64 = arith.constant {async_task_id = array<i32: 1>} 96 : i64
    %c0_i64 = arith.constant 0 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 3>} 64 : i32
    %c255_i64 = arith.constant {async_task_id = array<i32: 0, 1, 3>} 255 : i64
    %c0_i32 = arith.constant {async_task_id = array<i32: 3>} 0 : i32
    %false = arith.constant {async_task_id = array<i32: 4>} false
    %c128_i32 = arith.constant {async_task_id = array<i32: 1, 3>} 128 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %result_0, %token_1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %4 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
    %9 = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared2, #smem, mutable>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared2, #smem, mutable>
    %13 = arith.trunci %arg3 {async_task_id = array<i32: 1, 3>} : i64 to i32
    %14 = arith.muli %arg3, %c4096_i64 : i64
    %15 = arith.cmpi eq, %14, %c0_i64 : i64
    %16 = tt.make_tensor_descriptor %arg2, [%13, %c4096_i32], [%c4096_i64, %c1_i64] {async_task_id = array<i32: 1>} : !tt.ptr<bf16>, !tt.tensordesc<128x32xbf16, #shared>
    %17 = tt.make_tensor_descriptor %arg2, [%13, %c4096_i32], [%c4096_i64, %c1_i64] {async_task_id = array<i32: 1>} : !tt.ptr<bf16>, !tt.tensordesc<128x32xbf16, #shared>
    cf.cond_br %15, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %18 = arith.addi %arg3, %c255_i64 {async_task_id = array<i32: 0, 1, 3>} : i64
    %19 = arith.divsi %18, %c256_i64 {async_task_id = array<i32: 0, 1, 3>} : i64
    %20 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c4096_i32], [%c4096_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<64x128xbf16, #shared1>
    %21 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 3>} : i32
    %22 = tt.make_tensor_descriptor %arg0, [%13, %c64_i32], [%c64_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared1>
    %23 = tt.make_tensor_descriptor %arg0, [%13, %c64_i32], [%c64_i64, %c1_i64] {async_task_id = array<i32: 3>} : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16, #shared1>
    %24 = scf.while (%arg5 = %true, %arg6 = %21) : (i1, i32) -> i32 {
      scf.condition(%arg5) {async_task_id = array<i32: 0, 1, 3, 4, 5>} %arg6 : i32
    } do {
    ^bb0(%arg5: i32):
      scf.for %iv = %c0_i32 to %c64_i32 step %c64_i32  : i32 {
      } {async_task_id = array<i32: 3>}
      %25 = ttng.clc_try_cancel_async {async_task_id = array<i32: 3>} : !ttg.async.token
      %26 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %27 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %28 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %29 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %30 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %31 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %32 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %33 = arith.extsi %arg5 {async_task_id = array<i32: 3>} : i32 to i64
      %34 = arith.extsi %arg5 {async_task_id = array<i32: 1>} : i32 to i64
      %35 = arith.divsi %27, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %36 = arith.divsi %28, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %37 = arith.divsi %30, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %38 = arith.divsi %31, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %39 = arith.divsi %32, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %40 = arith.divsi %34, %c256_i64 {async_task_id = array<i32: 1>} : i64
      %41 = arith.muli %36, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %42 = arith.muli %35, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %43 = arith.muli %38, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %44 = arith.muli %37, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %45 = arith.muli %39, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %46 = arith.muli %40, %c8_i64 {async_task_id = array<i32: 1>} : i64
      %47 = arith.subi %19, %42 {async_task_id = array<i32: 3>} : i64
      %48 = arith.subi %19, %44 {async_task_id = array<i32: 3>} : i64
      %49 = arith.subi %19, %45 {async_task_id = array<i32: 3>} : i64
      %50 = arith.subi %19, %46 {async_task_id = array<i32: 1>} : i64
      %51 = arith.minsi %47, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %52 = arith.minsi %48, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %53 = arith.minsi %49, %c8_i64 {async_task_id = array<i32: 3>} : i64
      %54 = arith.minsi %50, %c8_i64 {async_task_id = array<i32: 1>} : i64
      %55 = arith.remsi %26, %51 {async_task_id = array<i32: 3>} : i64
      %56 = arith.remsi %29, %52 {async_task_id = array<i32: 3>} : i64
      %57 = arith.remsi %34, %54 {async_task_id = array<i32: 1>} : i64
      %58 = arith.addi %41, %55 {async_task_id = array<i32: 3>} : i64
      %59 = arith.addi %43, %56 {async_task_id = array<i32: 3>} : i64
      %60 = arith.addi %46, %57 {async_task_id = array<i32: 1>} : i64
      %61 = arith.remsi %33, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %62 = arith.remsi %34, %c256_i64 {async_task_id = array<i32: 1>} : i64
      %63 = arith.divsi %61, %53 {async_task_id = array<i32: 3>} : i64
      %64 = arith.divsi %62, %54 {async_task_id = array<i32: 1>} : i64
      %65 = arith.muli %58, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %66 = arith.muli %59, %c256_i64 {async_task_id = array<i32: 3>} : i64
      %67 = arith.muli %60, %c256_i64 {async_task_id = array<i32: 1>} : i64
      %68 = arith.muli %63, %c128_i64 {async_task_id = array<i32: 3>} : i64
      %69 = arith.muli %64, %c128_i64 {async_task_id = array<i32: 1>} : i64
      %70 = arith.trunci %65 {async_task_id = array<i32: 3>} : i64 to i32
      %71 = arith.trunci %66 {async_task_id = array<i32: 3>} : i64 to i32
      %72 = arith.trunci %67 {async_task_id = array<i32: 1>} : i64 to i32
      %73 = arith.addi %72, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %74 = arith.addi %72, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %75 = arith.addi %72, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %76 = arith.addi %72, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %77 = arith.addi %70, %c128_i32 {async_task_id = array<i32: 3>} : i32
      %78 = arith.trunci %68 {async_task_id = array<i32: 3>} : i64 to i32
      %79 = arith.trunci %69 {async_task_id = array<i32: 1>} : i64 to i32
      nvws.descriptor_load %20[%c0_i32, %78] 16384 %8 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<64x128xbf16, #shared1>, i32, i32, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
      nvws.descriptor_load %22[%71, %c0_i32] 16384 %9 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<128x64xbf16, #shared1>, i32, i32, !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
      nvws.descriptor_load %23[%77, %c0_i32] 16384 %10 {async_task_id = array<i32: 3>, multicast = false} : !tt.tensordesc<128x64xbf16, #shared1>, i32, i32, !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
      %80 = arith.addi %69, %c32_i64 {async_task_id = array<i32: 1>} : i64
      %81 = arith.trunci %80 {async_task_id = array<i32: 1>} : i64 to i32
      %82 = arith.addi %69, %c64_i64 {async_task_id = array<i32: 1>} : i64
      %83 = arith.trunci %82 {async_task_id = array<i32: 1>} : i64 to i32
      %84 = arith.addi %69, %c96_i64 {async_task_id = array<i32: 1>} : i64
      %85 = arith.trunci %84 {async_task_id = array<i32: 1>} : i64 to i32
      %isValid, %x, %y, %z = ttng.clc_read %25 {async_task_id = array<i32: 3>} : !ttg.async.token -> i1, i32, i32, i32
      %86 = arith.extui %isValid {async_task_id = array<i32: 3>} : i1 to i32
      %87 = tt.splat %86 {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked>
      ttg.local_store %87, %12 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared2, #smem, mutable>
      %88 = ttg.local_load %12 {async_task_id = array<i32: 0, 1, 3, 4, 5>} : !ttg.memdesc<1xi32, #shared2, #smem, mutable> -> tensor<1xi32, #blocked>
      %89 = tt.unsplat %88 {async_task_id = array<i32: 0, 1, 3, 4, 5>} : tensor<1xi32, #blocked>
      %90 = arith.trunci %89 {async_task_id = array<i32: 0, 1, 3, 4, 5>} : i32 to i1
      %91 = tt.splat %x {async_task_id = array<i32: 3>} : i32 -> tensor<1xi32, #blocked>
      ttg.local_store %91, %11 {async_task_id = array<i32: 3>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared2, #smem, mutable>
      %92 = ttg.local_load %11 {async_task_id = array<i32: 0, 1, 3, 4, 5>} : !ttg.memdesc<1xi32, #shared2, #smem, mutable> -> tensor<1xi32, #blocked>
      %93 = tt.unsplat %92 {async_task_id = array<i32: 0, 1, 3, 4, 5>} : tensor<1xi32, #blocked>
      %94 = ttng.tc_gen5_mma %9, %8, %result[%token], %false, %true {async_task_id = array<i32: 4>} : !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_2, %token_3 = ttng.tmem_load %result[%94] {async_task_id = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %95 = ttg.convert_layout %result_2 {async_task_id = array<i32: 5>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %96 = tt.reshape %95 {async_task_id = array<i32: 5>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
      %97 = tt.trans %96 {async_task_id = array<i32: 5>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
      %outLHS, %outRHS = tt.split %97 {async_task_id = array<i32: 5>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
      %98 = tt.reshape %outLHS {async_task_id = array<i32: 5>} : tensor<128x64xf32, #linear4> -> tensor<128x2x32xf32, #linear5>
      %99 = tt.trans %98 {async_task_id = array<i32: 5>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
      %outLHS_4, %outRHS_5 = tt.split %99 {async_task_id = array<i32: 5>} : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear7>
      %100 = tt.reshape %outRHS {async_task_id = array<i32: 5>} : tensor<128x64xf32, #linear4> -> tensor<128x2x32xf32, #linear5>
      %101 = tt.trans %100 {async_task_id = array<i32: 5>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
      %outLHS_6, %outRHS_7 = tt.split %101 {async_task_id = array<i32: 5>} : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear7>
      %102 = arith.truncf %outLHS_4 {async_task_id = array<i32: 5>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %103 = ttg.convert_layout %102 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %103, %0 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %104 = ttng.async_tma_copy_local_to_global %16[%72, %79] %0 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %104   {async_task_id = array<i32: 1>} : !ttg.async.token
      %105 = arith.truncf %outRHS_5 {async_task_id = array<i32: 5>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %106 = ttg.convert_layout %105 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %106, %1 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %107 = ttng.async_tma_copy_local_to_global %16[%72, %81] %1 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %107   {async_task_id = array<i32: 1>} : !ttg.async.token
      %108 = arith.truncf %outLHS_6 {async_task_id = array<i32: 5>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %109 = ttg.convert_layout %108 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %109, %2 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %110 = ttng.async_tma_copy_local_to_global %16[%72, %83] %2 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %110   {async_task_id = array<i32: 1>} : !ttg.async.token
      %111 = arith.truncf %outRHS_7 {async_task_id = array<i32: 5>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %112 = ttg.convert_layout %111 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %112, %3 {async_task_id = array<i32: 5>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %113 = ttng.async_tma_copy_local_to_global %16[%72, %85] %3 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %113   {async_task_id = array<i32: 1>} : !ttg.async.token
      %114 = ttng.tc_gen5_mma %10, %8, %result_0[%token_1], %false, %true {async_task_id = array<i32: 4>} : !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_8, %token_9 = ttng.tmem_load %result_0[%114] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %115 = ttg.convert_layout %result_8 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear1>
      %116 = tt.reshape %115 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
      %117 = tt.trans %116 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
      %outLHS_10, %outRHS_11 = tt.split %117 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
      %118 = tt.reshape %outLHS_10 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x2x32xf32, #linear5>
      %119 = tt.trans %118 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
      %outLHS_12, %outRHS_13 = tt.split %119 {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear7>
      %120 = tt.reshape %outRHS_11 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x2x32xf32, #linear5>
      %121 = tt.trans %120 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
      %outLHS_14, %outRHS_15 = tt.split %121 {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear7>
      %122 = arith.truncf %outLHS_12 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %123 = ttg.convert_layout %122 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %123, %4 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %124 = ttng.async_tma_copy_local_to_global %17[%76, %79] %4 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %124   {async_task_id = array<i32: 1>} : !ttg.async.token
      %125 = arith.truncf %outRHS_13 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %126 = ttg.convert_layout %125 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %126, %5 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %127 = ttng.async_tma_copy_local_to_global %17[%75, %81] %5 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %127   {async_task_id = array<i32: 1>} : !ttg.async.token
      %128 = arith.truncf %outLHS_14 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %129 = ttg.convert_layout %128 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %129, %6 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %130 = ttng.async_tma_copy_local_to_global %17[%74, %83] %6 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %130   {async_task_id = array<i32: 1>} : !ttg.async.token
      %131 = arith.truncf %outRHS_15 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear7> to tensor<128x32xbf16, #linear7>
      %132 = ttg.convert_layout %131 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear7> -> tensor<128x32xbf16, #blocked1>
      ttg.local_store %132, %7 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #blocked1> -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
      %133 = ttng.async_tma_copy_local_to_global %17[%73, %85] %7 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x32xbf16, #shared>, !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %133   {async_task_id = array<i32: 1>} : !ttg.async.token
      scf.yield {async_task_id = array<i32: 0, 1, 3, 4, 5>} %90, %93 : i1, i32
    } attributes {async_task_id = array<i32: 0, 1, 3, 4, 5>, tt.data_partition_factor = 2 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["computation", "epilogue", "epilogue_store", "load", "gemm", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
