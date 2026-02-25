// RUN: triton-opt --tlx-print-ttgir-to-tlx %s | FileCheck %s

// Test TTGIR to TLX simplified output on FlashAttention persistent kernel
// The pass outputs simplified TLX-style code:
// - No layouts or types
// - Parentheses for operands
// - Simplified operation names
// - local_alloc differentiation between barriers and buffers

// Check function signature
// CHECK: func _attn_fwd_persist(

// Verify barrier allocations are detected and converted
// CHECK-DAG: tlx.alloc_barriers(1)
// CHECK-DAG: tlx.alloc_barriers(3)

// Verify regular buffer allocations are converted with shape, dtype, count
// CHECK-DAG: tlx.local_alloc((128, 128), bf16, 1)
// CHECK-DAG: tlx.local_alloc((128, 128), bf16, 3)

// Verify barrier operations are replaced
// CHECK-DAG: tlx.barrier_wait(
// CHECK-DAG: tlx.barrier_arrive(
// CHECK-DAG: tlx.barrier_expect_bytes(

// Verify MMA operations are replaced
// CHECK-DAG: tlx.tc_gen5_mma(

// Verify TMA operations are replaced
// CHECK-DAG: tlx.async_descriptor_load(
// CHECK-DAG: tlx.async_descriptor_store(

// Verify memory operations are replaced
// CHECK-DAG: tlx.tmem_alloc(
// CHECK-DAG: tlx.tmem_load(
// CHECK-DAG: tlx.tmem_store(
// CHECK-DAG: tlx.local_trans(
// CHECK-DAG: tlx.subslice(

// Verify warp specialization uses Python-like async_tasks syntax
// CHECK-DAG: with tlx.async_tasks():
// CHECK-DAG: with tlx.async_task("default"):
// CHECK-DAG: with tlx.async_task():

// Verify control flow is simplified - for loops use Python range syntax
// CHECK-DAG: for arg{{[0-9]+}} in range(

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#linear = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
#tmem3 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_persist(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %desc_o: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8064_i32 = arith.constant 8064 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %3 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %5 = ttg.memdesc_index %4[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %7 = ttg.memdesc_index %6[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %9 = ttg.memdesc_index %8[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %11 = ttg.memdesc_index %10[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %11, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64, #shared, #smem, mutable>
    %13 = ttg.memdesc_index %12[%c0_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %13, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %14 = ttg.memdesc_index %12[%c1_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %14, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %15 = ttg.memdesc_index %12[%c2_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %15, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64, #shared, #smem, mutable>
    %17 = ttg.memdesc_index %16[%c0_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %17, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %18 = ttg.memdesc_index %16[%c1_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %18, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %19 = ttg.memdesc_index %16[%c2_i32] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %19, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %20 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %21 = ttg.memdesc_index %20[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %21, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %23 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %23, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %24 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %25 = ttg.memdesc_index %24[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %25, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %26 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %27 = ttg.memdesc_index %26[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %27, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %28 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %29 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %30 = ttg.memdesc_index %28[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %30, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %31 = ttg.memdesc_index %29[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %31, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %32 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %33 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %34 = ttg.memdesc_index %32[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %34, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %35 = ttg.memdesc_index %33[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %35, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %36 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %37 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %38 = ttg.memdesc_index %36[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %38, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %39 = ttg.memdesc_index %37[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %39, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %40 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %41 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %42 = ttg.memdesc_index %40[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %42, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %43 = ttg.memdesc_index %41[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %43, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %44 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %45 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %46 = ttg.memdesc_index %44[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %46, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %47 = ttg.memdesc_index %45[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %47, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %48 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %49 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %50 = ttg.memdesc_index %48[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %50, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %51 = ttg.memdesc_index %49[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %51, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %52 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %53 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %54 = ttg.memdesc_index %52[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %54, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %55 = ttg.memdesc_index %53[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %55, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %56 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %57 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %58 = ttg.memdesc_index %56[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %58, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %59 = ttg.memdesc_index %57[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %59, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %60 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %61 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %62 = ttg.memdesc_index %60[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %62, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %63 = ttg.memdesc_index %61[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %63, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %64 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %65 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %66 = ttg.memdesc_index %64[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %66, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %67 = ttg.memdesc_index %65[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %67, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %68 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %69 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %70 = ttg.memdesc_index %68[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %70, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %71 = ttg.memdesc_index %69[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %71, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %72 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %73 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %74 = ttg.memdesc_index %72[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %74, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %75 = ttg.memdesc_index %73[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %75, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %76 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %77 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %78 = ttg.memdesc_index %76[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %78, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %79 = ttg.memdesc_index %77[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %79, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %80 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %81 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %82 = ttg.memdesc_index %80[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %82, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %83 = ttg.memdesc_index %81[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %83, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %84 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %85 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %86 = ttg.memdesc_index %84[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %86, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %87 = ttg.memdesc_index %85[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %87, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %88 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %89 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %90 = ttg.memdesc_index %88[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %90, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %91 = ttg.memdesc_index %89[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %91, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    %92 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>
    %93 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>
    %v = ttg.local_alloc {allocation.shareGroup = 1 : i32, buffer.copy = 3 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable>
    %q0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>
    %q0_0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>
    %qk = ttng.tmem_alloc {allocation.shareGroup = 3 : i32, buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %qk_1 = ttng.tmem_alloc {allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc = ttng.tmem_alloc {allocation.shareGroup = 2 : i32, buffer.copy = 1 : i32, buffer.id = 6 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_2 = ttng.tmem_alloc {allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_specialize(%Z, %H, %4, %2, %v, %16, %qk_1, %q0_0, %20, %qk, %q0, %12, %22, %acc_2, %24, %8, %acc, %26, %10, %6, %0, %desc_q, %desc_k, %desc_v, %desc_o, %93, %92, %sm_scale, %28, %29, %32, %33, %36, %40, %44, %45, %48, %49, %53, %57, %60, %61, %64, %65, %68, %69, %72, %73, %76, %81, %84, %89) attributes {requestedRegisters = array<i32: 24, 24, 24, 152, 152>}
    default {
      %prog_id = tt.get_program_id x {async_task_id = array<i32: 0>} : i32
      %num_progs = tt.get_num_programs x {async_task_id = array<i32: 0>} : i32
      %total_tiles = arith.muli %Z, %c32_i32 {async_task_id = array<i32: 0>} : i32
      %total_tiles_3 = arith.muli %total_tiles, %H {async_task_id = array<i32: 0>} : i32
      %tiles_per_sm = arith.divsi %total_tiles_3, %num_progs {async_task_id = array<i32: 0>} : i32
      %94 = arith.remsi %total_tiles_3, %num_progs {async_task_id = array<i32: 0>} : i32
      %95 = arith.cmpi slt, %prog_id, %94 {async_task_id = array<i32: 0>} : i32
      %96 = scf.if %95 -> (i32) {
        %tiles_per_sm_5 = arith.addi %tiles_per_sm, %c1_i32 {async_task_id = array<i32: 0>} : i32
        scf.yield {async_task_id = array<i32: 0>} %tiles_per_sm_5 : i32
      } else {
        scf.yield {async_task_id = array<i32: 0>} %tiles_per_sm : i32
      } {async_task_id = array<i32: 0>}
      %offs_m0 = tt.make_range {async_task_id = array<i32: 0>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
      %offs_m0_4 = tt.make_range {async_task_id = array<i32: 0>, end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked1>
      %tile_idx:3 = scf.for %tile_idx_5 = %c0_i32 to %96 step %c1_i32 iter_args(%prog_id_6 = %prog_id, %arg10 = %c0_i64, %arg11 = %c0_i64) -> (i32, i64, i64)  : i32 {
        %pid = arith.remsi %prog_id_6, %c32_i32 {async_task_id = array<i32: 0>} : i32
        %off_hz = arith.divsi %prog_id_6, %c32_i32 {async_task_id = array<i32: 0>} : i32
        %qo_offset_y = arith.muli %pid, %c256_i32 {async_task_id = array<i32: 0>} : i32
        %offs_m0_7 = tt.splat %qo_offset_y {async_task_id = array<i32: 0>} : i32 -> tensor<128xi32, #blocked1>
        %offs_m0_8 = arith.addi %offs_m0_7, %offs_m0 {async_task_id = array<i32: 0>} : tensor<128xi32, #blocked1>
        %offs_m0_9 = arith.addi %offs_m0_7, %offs_m0_4 {async_task_id = array<i32: 0>} : tensor<128xi32, #blocked1>
        %acc_10 = ttg.memdesc_index %acc_2[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        ttng.tmem_store %cst, %acc_10, %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_11 = ttg.memdesc_index %acc[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        ttng.tmem_store %cst, %acc_11, %true {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %offsetkv_y = arith.andi %arg10, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %offsetkv_y_12 = arith.trunci %offsetkv_y {async_task_id = array<i32: 0>} : i64 to i1
        %alpha = arith.andi %arg11, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %alpha_13 = arith.trunci %alpha {async_task_id = array<i32: 0>} : i64 to i1
        %97 = ttg.memdesc_index %8[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_14 = arith.xori %alpha_13, %true {async_task_id = array<i32: 0>} : i1
        %acc_15 = arith.extui %acc_14 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %97, %acc_15, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_16 = ttng.tmem_subslice %acc_10 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
        %acc_17 = ttng.tmem_subslice %acc_10 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
        %qk_18 = ttng.tmem_subslice %qk_1 {N = 64 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
        %qk_19 = ttg.memdesc_reinterpret %qk_18 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %alpha_20 = ttg.memdesc_index %qk_19[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %98 = ttg.memdesc_index %48[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc0 = arith.extui %alpha_13 : i1 to i32
        ttng.wait_barrier %98, %acc0, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %99 = ttg.memdesc_index %49[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_21 = ttng.tmem_load %acc_16 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #blocked2>
        %acc_22 = ttng.tmem_load %acc_17 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #blocked2>
        %acc0_23 = ttng.tmem_load %alpha_20 {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
        ttng.arrive_barrier %99, 1, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc0_24 = tt.reshape %acc0_23 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
        %acc0_25 = ttg.convert_layout %acc0_24 : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc0_26 = tt.expand_dims %acc0_25 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc0_27 = ttg.convert_layout %acc0_26 : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked2>
        %acc0_28 = tt.broadcast %acc0_27 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
        %acc0_29 = arith.mulf %acc_21, %acc0_28 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2>
        %acc1 = arith.mulf %acc_22, %acc0_28 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2>
        %acc_30 = tt.join %acc0_29, %acc1 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> -> tensor<128x64x2xf32, #blocked4>
        %acc_31 = tt.trans %acc_30 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x2x64xf32, #blocked5>
        %acc_32 = tt.reshape %acc_31 : tensor<128x2x64xf32, #blocked5> -> tensor<128x128xf32, #blocked>
        ttng.tmem_store %acc_32, %acc_10, %true {async_task_id = array<i32: 0>, tmem.start = array<i32: 18, 18>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %100 = ttg.memdesc_index %84[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.arrive_barrier %100, 1, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %offsetkv_y_33 = scf.for %offsetkv_y_99 = %c0_i32 to %c8064_i32 step %c128_i32 iter_args(%arg13 = %arg11) -> (i64)  : i32 {
          %alpha_100 = arith.andi %arg13, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %alpha_101 = arith.trunci %alpha_100 {async_task_id = array<i32: 0>} : i64 to i1
          %128 = ttg.memdesc_index %10[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc_102 = arith.xori %alpha_101, %true {async_task_id = array<i32: 0>} : i1
          %acc_103 = arith.extui %acc_102 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %128, %acc_103 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc_104 = ttng.tmem_subslice %acc_11 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
          %acc_105 = ttng.tmem_subslice %acc_11 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
          %qk_106 = ttng.tmem_subslice %qk {N = 64 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
          %qk_107 = ttg.memdesc_reinterpret %qk_106 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
          %alpha_108 = ttg.memdesc_index %qk_107[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
          %129 = ttg.memdesc_index %44[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc0_109 = arith.extui %alpha_101 : i1 to i32
          ttng.wait_barrier %129, %acc0_109 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %130 = ttg.memdesc_index %45[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc_110 = ttng.tmem_load %acc_104 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #blocked2>
          %acc_111 = ttng.tmem_load %acc_105 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #blocked2>
          %acc0_112 = ttng.tmem_load %alpha_108 {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
          ttng.arrive_barrier %130, 1 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc0_113 = tt.reshape %acc0_112 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
          %acc0_114 = ttg.convert_layout %acc0_113 : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %acc0_115 = tt.expand_dims %acc0_114 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
          %acc0_116 = ttg.convert_layout %acc0_115 : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked2>
          %acc0_117 = tt.broadcast %acc0_116 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
          %acc0_118 = arith.mulf %acc_110, %acc0_117 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2>
          %acc1_119 = arith.mulf %acc_111, %acc0_117 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2>
          %acc_120 = tt.join %acc0_118, %acc1_119 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> -> tensor<128x64x2xf32, #blocked4>
          %acc_121 = tt.trans %acc_120 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x2x64xf32, #blocked5>
          %acc_122 = tt.reshape %acc_121 : tensor<128x2x64xf32, #blocked5> -> tensor<128x128xf32, #blocked>
          ttng.tmem_store %acc_122, %acc_11, %true {async_task_id = array<i32: 0>, tmem.start = array<i32: 15, 15>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %131 = ttg.memdesc_index %76[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.arrive_barrier %131, 1 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %offsetkv_y_123 = arith.addi %arg13, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %alpha_124 = arith.andi %offsetkv_y_123, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %alpha_125 = arith.trunci %alpha_124 {async_task_id = array<i32: 0>} : i64 to i1
          %acc_126 = arith.xori %alpha_125, %true {async_task_id = array<i32: 0>} : i1
          %acc_127 = arith.extui %acc_126 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %97, %acc_127, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc0_128 = arith.extui %alpha_125 : i1 to i32
          ttng.wait_barrier %98, %acc0_128, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc_129 = ttng.tmem_load %acc_16 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #blocked2>
          %acc_130 = ttng.tmem_load %acc_17 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #blocked2>
          %acc0_131 = ttng.tmem_load %alpha_20 {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
          ttng.arrive_barrier %99, 1, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc0_132 = tt.reshape %acc0_131 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
          %acc0_133 = ttg.convert_layout %acc0_132 : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %acc0_134 = tt.expand_dims %acc0_133 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
          %acc0_135 = ttg.convert_layout %acc0_134 : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked2>
          %acc0_136 = tt.broadcast %acc0_135 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
          %acc0_137 = arith.mulf %acc_129, %acc0_136 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2>
          %acc1_138 = arith.mulf %acc_130, %acc0_136 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2>
          %acc_139 = tt.join %acc0_137, %acc1_138 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> -> tensor<128x64x2xf32, #blocked4>
          %acc_140 = tt.trans %acc_139 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x2x64xf32, #blocked5>
          %acc_141 = tt.reshape %acc_140 : tensor<128x2x64xf32, #blocked5> -> tensor<128x128xf32, #blocked>
          ttng.tmem_store %acc_141, %acc_10, %true {async_task_id = array<i32: 0>, tmem.start = array<i32: 18, 18>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          ttng.arrive_barrier %100, 1, %true {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          scf.yield %offsetkv_y_123 : i64
        } {async_task_id = array<i32: 0>, tt.warp_specialize}
        %alpha_34 = arith.andi %offsetkv_y_33, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %alpha_35 = arith.trunci %alpha_34 {async_task_id = array<i32: 0>} : i64 to i1
        %101 = ttg.memdesc_index %10[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_36 = arith.xori %alpha_35, %true {async_task_id = array<i32: 0>} : i1
        %acc_37 = arith.extui %acc_36 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %101, %acc_37 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_38 = ttng.tmem_subslice %acc_11 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
        %acc_39 = ttng.tmem_subslice %acc_11 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
        %qk_40 = ttng.tmem_subslice %qk {N = 64 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
        %qk_41 = ttg.memdesc_reinterpret %qk_40 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %alpha_42 = ttg.memdesc_index %qk_41[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %102 = ttg.memdesc_index %44[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc0_43 = arith.extui %alpha_35 : i1 to i32
        ttng.wait_barrier %102, %acc0_43 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %103 = ttg.memdesc_index %45[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_44 = ttng.tmem_load %acc_38 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #blocked2>
        %acc_45 = ttng.tmem_load %acc_39 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #blocked2>
        %acc0_46 = ttng.tmem_load %alpha_42 {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
        ttng.arrive_barrier %103, 1 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc0_47 = tt.reshape %acc0_46 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
        %acc0_48 = ttg.convert_layout %acc0_47 : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc0_49 = tt.expand_dims %acc0_48 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc0_50 = ttg.convert_layout %acc0_49 : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked2>
        %acc0_51 = tt.broadcast %acc0_50 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
        %acc0_52 = arith.mulf %acc_44, %acc0_51 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2>
        %acc1_53 = arith.mulf %acc_45, %acc0_51 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2>
        %acc_54 = tt.join %acc0_52, %acc1_53 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> -> tensor<128x64x2xf32, #blocked4>
        %acc_55 = tt.trans %acc_54 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x2x64xf32, #blocked5>
        %acc_56 = tt.reshape %acc_55 : tensor<128x2x64xf32, #blocked5> -> tensor<128x128xf32, #blocked>
        ttng.tmem_store %acc_56, %acc_11, %true {async_task_id = array<i32: 0>, tmem.start = array<i32: 15, 15>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %104 = ttg.memdesc_index %76[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.arrive_barrier %104, 1 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %offsetkv_y_57 = arith.addi %offsetkv_y_33, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %qk_58 = ttng.tmem_subslice %qk_1 {N = 66 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
        %qk_59 = ttg.memdesc_reinterpret %qk_58 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %offsetkv_y_60 = ttg.memdesc_index %qk_59[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %105 = ttg.memdesc_index %72[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %m_i0 = arith.extui %offsetkv_y_12 : i1 to i32
        ttng.wait_barrier %105, %m_i0 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %106 = ttg.memdesc_index %73[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %m_i0_61 = ttng.tmem_load %offsetkv_y_60 {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
        ttng.arrive_barrier %106, 1 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %m_i0_62 = tt.reshape %m_i0_61 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
        %m_i0_63 = ttg.convert_layout %m_i0_62 : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_i0_64 = math.log2 %m_i0_62 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear>
        %qk_65 = ttng.tmem_subslice %qk_1 {N = 65 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
        %qk_66 = ttg.memdesc_reinterpret %qk_65 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %offsetkv_y_67 = ttg.memdesc_index %qk_66[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %107 = ttg.memdesc_index %68[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %107, %m_i0 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %108 = ttg.memdesc_index %69[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %m_i0_68 = ttng.tmem_load %offsetkv_y_67 {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
        ttng.arrive_barrier %108, 1 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %m_i0_69 = tt.reshape %m_i0_68 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
        %m_i0_70 = arith.addf %m_i0_69, %m_i0_64 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear>
        %109 = ttg.convert_layout %m_i0_70 : tensor<128xf32, #linear> -> tensor<128xf32, #blocked1>
        %m_ptrs0 = arith.muli %off_hz, %c8192_i32 {async_task_id = array<i32: 0>} : i32
        %m_ptrs0_71 = tt.addptr %M, %m_ptrs0 {async_task_id = array<i32: 0>} : !tt.ptr<f32>, i32
        %m_ptrs0_72 = tt.splat %m_ptrs0_71 {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
        %m_ptrs0_73 = tt.addptr %m_ptrs0_72, %offs_m0_8 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
        tt.store %m_ptrs0_73, %109 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>
        %acc0_74 = tt.expand_dims %m_i0_63 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc0_75 = tt.broadcast %acc0_74 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %110 = ttg.memdesc_index %6[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_76 = arith.extui %offsetkv_y_12 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %110, %acc_76 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %111 = ttg.memdesc_index %89[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_77 = ttng.tmem_load %acc_10 {async_task_id = array<i32: 0>, tmem.end = array<i32: 19, 19>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        ttng.arrive_barrier %111, 1 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc0_78 = arith.divf %acc_77, %acc0_75 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked>
        %112 = arith.truncf %acc0_78 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %113 = ttg.memdesc_index %93[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %114 = ttg.memdesc_index %33[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %115 = arith.xori %offsetkv_y_12, %true : i1
        %116 = arith.extui %115 : i1 to i32
        ttng.wait_barrier %114, %116 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttg.local_store %112, %113 {async_task_id = array<i32: 0>} : tensor<128x128xbf16, #blocked> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %117 = ttg.memdesc_index %32[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.arrive_barrier %117, 1 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %qk_79 = ttng.tmem_subslice %qk {N = 66 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
        %qk_80 = ttg.memdesc_reinterpret %qk_79 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %offsetkv_y_81 = ttg.memdesc_index %qk_80[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %118 = ttg.memdesc_index %64[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %118, %m_i0 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %119 = ttg.memdesc_index %65[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %m_i0_82 = ttng.tmem_load %offsetkv_y_81 {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
        ttng.arrive_barrier %119, 1 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %m_i0_83 = tt.reshape %m_i0_82 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
        %m_i0_84 = ttg.convert_layout %m_i0_83 : tensor<128xf32, #linear> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_i0_85 = math.log2 %m_i0_83 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear>
        %qk_86 = ttng.tmem_subslice %qk {N = 65 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
        %qk_87 = ttg.memdesc_reinterpret %qk_86 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %offsetkv_y_88 = ttg.memdesc_index %qk_87[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %120 = ttg.memdesc_index %60[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %120, %m_i0 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %121 = ttg.memdesc_index %61[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %m_i0_89 = ttng.tmem_load %offsetkv_y_88 {async_task_id = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked3>
        ttng.arrive_barrier %121, 1 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %m_i0_90 = tt.reshape %m_i0_89 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked3> -> tensor<128xf32, #linear>
        %m_i0_91 = arith.addf %m_i0_90, %m_i0_85 {async_task_id = array<i32: 0>} : tensor<128xf32, #linear>
        %122 = ttg.convert_layout %m_i0_91 : tensor<128xf32, #linear> -> tensor<128xf32, #blocked1>
        %m_ptrs0_92 = tt.addptr %m_ptrs0_72, %offs_m0_9 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
        tt.store %m_ptrs0_92, %122 {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked1>
        %acc0_93 = tt.expand_dims %m_i0_84 {async_task_id = array<i32: 0>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc0_94 = tt.broadcast %acc0_93 {async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %123 = ttg.memdesc_index %81[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_95 = ttng.tmem_load %acc_11 {async_task_id = array<i32: 0>, tmem.end = array<i32: 16, 16>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        ttng.arrive_barrier %123, 1 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc0_96 = arith.divf %acc_95, %acc0_94 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked>
        %124 = arith.truncf %acc0_96 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %125 = ttg.memdesc_index %92[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %126 = ttg.memdesc_index %29[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %126, %116 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttg.local_store %124, %125 {async_task_id = array<i32: 0>} : tensor<128x128xbf16, #blocked> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %127 = ttg.memdesc_index %28[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.arrive_barrier %127, 1 {async_task_id = array<i32: 0>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %tile_idx_97 = arith.addi %prog_id_6, %num_progs {async_task_id = array<i32: 0>} : i32
        %tile_idx_98 = arith.addi %arg10, %c1_i64 {async_task_id = array<i32: 0>} : i64
        scf.yield %tile_idx_97, %tile_idx_98, %offsetkv_y_57 : i32, i64, i64
      } {async_task_id = array<i32: 0>, tt.data_partition_factor = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_yield {async_task_id = array<i32: 0>}
    }
    partition0(%Z_3: i32, %H_4: i32, %arg10: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg11: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %v_5: !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable>, %arg13: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %qk_6: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q0_7: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg16: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %qk_8: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q0_9: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg19: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %arg20: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %acc_10: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg22: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg23: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %acc_11: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg25: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg26: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg27: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg28: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %desc_q_12: !tt.ptr<bf16>, %desc_k_13: !tt.ptr<bf16>, %desc_v_14: !tt.ptr<bf16>, %desc_o_15: !tt.ptr<bf16>, %arg33: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg34: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %sm_scale_16: f32, %arg36: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg37: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg38: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg39: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg40: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg41: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg42: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg43: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg44: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg45: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg46: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg47: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg48: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg49: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg50: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg51: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg52: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg53: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg54: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg55: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg56: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg57: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg58: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg59: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>) num_warps(1) {
      %c8064_i32_17 = arith.constant 8064 : i32
      %c2_i64 = arith.constant {async_task_id = array<i32: 1>} 2 : i64
      %c3_i64 = arith.constant {async_task_id = array<i32: 1>} 3 : i64
      %c1_i64_18 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
      %c0_i64_19 = arith.constant {async_task_id = array<i32: 1>} 0 : i64
      %false = arith.constant {async_task_id = array<i32: 1>} false
      %true_20 = arith.constant {async_task_id = array<i32: 1>} true
      %n_tile_num = arith.constant {async_task_id = array<i32: 1>} 32 : i32
      %c1_i32_21 = arith.constant {async_task_id = array<i32: 1>} 1 : i32
      %c128_i32_22 = arith.constant {async_task_id = array<i32: 1>} 128 : i32
      %c0_i32_23 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
      %prog_id = tt.get_program_id x {async_task_id = array<i32: 1>} : i32
      %num_progs = tt.get_num_programs x {async_task_id = array<i32: 1>} : i32
      %total_tiles = arith.muli %Z_3, %n_tile_num {async_task_id = array<i32: 1>} : i32
      %total_tiles_24 = arith.muli %total_tiles, %H_4 {async_task_id = array<i32: 1>} : i32
      %tiles_per_sm = arith.divsi %total_tiles_24, %num_progs {async_task_id = array<i32: 1>} : i32
      %94 = arith.remsi %total_tiles_24, %num_progs {async_task_id = array<i32: 1>} : i32
      %95 = arith.cmpi slt, %prog_id, %94 {async_task_id = array<i32: 1>} : i32
      %96 = scf.if %95 -> (i32) {
        %tiles_per_sm_25 = arith.addi %tiles_per_sm, %c1_i32_21 {async_task_id = array<i32: 1>} : i32
        scf.yield {async_task_id = array<i32: 1>} %tiles_per_sm_25 : i32
      } else {
        scf.yield {async_task_id = array<i32: 1>} %tiles_per_sm : i32
      } {async_task_id = array<i32: 1>}
      %tile_idx:3 = scf.for %tile_idx_25 = %c0_i32_23 to %96 step %c1_i32_21 iter_args(%tile_idx_26 = %c0_i64_19, %tile_idx_27 = %c0_i64_19, %tile_idx_28 = %c0_i64_19) -> (i64, i64, i64)  : i32 {
        %offsetkv_y = arith.andi %tile_idx_26, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
        %offsetkv_y_29 = arith.trunci %offsetkv_y {async_task_id = array<i32: 1>} : i64 to i1
        %97 = ttg.memdesc_index %arg10[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %98 = arith.extui %offsetkv_y_29 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %97, %98, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %99 = ttg.memdesc_index %arg11[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %99, %98, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %100 = ttg.memdesc_index %arg57[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_30 = arith.xori %offsetkv_y_29, %true_20 : i1
        %acc_31 = arith.extui %acc_30 : i1 to i32
        ttng.wait_barrier %100, %acc_31 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %101 = ttg.memdesc_index %arg59[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %101, %acc_31 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %k = arith.divui %tile_idx_28, %c3_i64 {async_task_id = array<i32: 1>} : i64
        %k_32 = arith.muli %k, %c3_i64 {async_task_id = array<i32: 1>} : i64
        %k_33 = arith.subi %tile_idx_28, %k_32 {async_task_id = array<i32: 1>} : i64
        %k_34 = arith.trunci %k_33 {async_task_id = array<i32: 1>} : i64 to i32
        %k_35 = arith.andi %k, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
        %k_36 = arith.trunci %k_35 {async_task_id = array<i32: 1>} : i64 to i1
        %k_37 = ttg.memdesc_index %v_5[%k_34] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %k_38 = ttg.memdesc_trans %k_37 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
        %102 = ttg.memdesc_index %arg13[%k_34] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %103 = arith.extui %k_36 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %102, %103, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %qk_39 = ttg.memdesc_index %qk_6[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %q0_40 = ttg.memdesc_index %q0_7[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %qk_41 = arith.andi %tile_idx_27, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
        %qk_42 = arith.trunci %qk_41 {async_task_id = array<i32: 1>} : i64 to i1
        %104 = ttg.memdesc_index %arg16[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %105 = ttg.memdesc_index %arg47[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %qk_43 = arith.xori %qk_42, %true_20 : i1
        %qk_44 = arith.extui %qk_43 : i1 to i32
        ttng.wait_barrier %105, %qk_44, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.tc_gen5_mma %q0_40, %k_38, %qk_39, %false, %true_20, %104[%true_20] {async_task_id = array<i32: 1>, is_async, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %qk_45 = ttg.memdesc_index %qk_8[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %q0_46 = ttg.memdesc_index %q0_9[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %106 = ttg.memdesc_index %arg19[%k_34] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %107 = ttg.memdesc_index %arg20[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %108 = ttg.memdesc_index %arg46[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %108, %qk_44, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.tc_gen5_mma %q0_46, %k_38, %qk_45, %false, %true_20, %106[%true_20], %107[%true_20] {async_task_id = array<i32: 1>, is_async, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %v_47 = arith.addi %tile_idx_28, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
        %v_48 = arith.divui %v_47, %c3_i64 {async_task_id = array<i32: 1>} : i64
        %v_49 = arith.muli %v_48, %c3_i64 {async_task_id = array<i32: 1>} : i64
        %v_50 = arith.subi %v_47, %v_49 {async_task_id = array<i32: 1>} : i64
        %v_51 = arith.trunci %v_50 {async_task_id = array<i32: 1>} : i64 to i32
        %v_52 = arith.andi %v_48, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
        %v_53 = arith.trunci %v_52 {async_task_id = array<i32: 1>} : i64 to i1
        %qk_54 = ttng.tmem_subslice %qk_6 {N = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
        %qk_55 = ttg.memdesc_reinterpret %qk_54 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>
        %acc_56 = ttg.memdesc_index %qk_55[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xbf16, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>
        %v_57 = ttg.memdesc_index %v_5[%v_51] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %acc_58 = ttg.memdesc_index %acc_10[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %109 = ttg.memdesc_index %arg13[%v_51] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %110 = arith.extui %v_53 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %109, %110, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %111 = ttg.memdesc_index %arg22[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %112 = ttg.memdesc_index %arg41[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_59 = arith.extui %qk_42 : i1 to i32
        ttng.wait_barrier %112, %acc_59, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %113 = ttg.memdesc_index %arg23[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %114 = ttg.memdesc_index %arg58[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %114, %acc_59, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.tc_gen5_mma %acc_56, %v_57, %acc_58, %false, %true_20, %111[%true_20], %113[%true_20] {async_task_id = array<i32: 1>, is_async, tmem.end = array<i32: 18, 18>, tmem.start = array<i32: 17, 17, 19, 19>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %offsetkv_y_60:4 = scf.for %offsetkv_y_77 = %c0_i32_23 to %c8064_i32_17 step %c128_i32_22 iter_args(%arg65 = %false, %tile_idx_78 = %tile_idx_27, %tile_idx_79 = %tile_idx_28, %v_80 = %v_51) -> (i1, i64, i64, i32)  : i32 {
          %offsetkv_y_81 = arith.addi %tile_idx_79, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %offsetkv_y_82 = arith.addi %tile_idx_78, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
          %k_83 = arith.divui %offsetkv_y_81, %c3_i64 {async_task_id = array<i32: 1>} : i64
          %k_84 = arith.muli %k_83, %c3_i64 {async_task_id = array<i32: 1>} : i64
          %k_85 = arith.subi %offsetkv_y_81, %k_84 {async_task_id = array<i32: 1>} : i64
          %k_86 = arith.trunci %k_85 {async_task_id = array<i32: 1>} : i64 to i32
          %k_87 = arith.andi %k_83, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
          %k_88 = arith.trunci %k_87 {async_task_id = array<i32: 1>} : i64 to i1
          %k_89 = ttg.memdesc_index %v_5[%k_86] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
          %k_90 = ttg.memdesc_trans %k_89 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
          %120 = ttg.memdesc_index %arg13[%k_86] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %121 = arith.extui %k_88 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %120, %121, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %qk_91 = arith.andi %offsetkv_y_82, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
          %qk_92 = arith.trunci %qk_91 {async_task_id = array<i32: 1>} : i64 to i1
          %qk_93 = arith.xori %qk_92, %true_20 : i1
          %qk_94 = arith.extui %qk_93 : i1 to i32
          ttng.wait_barrier %105, %qk_94, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.tc_gen5_mma %q0_40, %k_90, %qk_39, %false, %true_20, %104[%true_20] {async_task_id = array<i32: 1>, is_async, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc_95 = arith.andi %tile_idx_78, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
          %acc_96 = arith.trunci %acc_95 {async_task_id = array<i32: 1>} : i64 to i1
          %qk_97 = ttng.tmem_subslice %qk_8 {N = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
          %qk_98 = ttg.memdesc_reinterpret %qk_97 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>
          %acc_99 = ttg.memdesc_index %qk_98[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xbf16, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>
          %acc_100 = arith.addi %tile_idx_79, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
          %acc_101 = arith.divui %acc_100, %c3_i64 {async_task_id = array<i32: 1>} : i64
          %acc_102 = arith.muli %acc_101, %c3_i64 {async_task_id = array<i32: 1>} : i64
          %acc_103 = arith.subi %acc_100, %acc_102 {async_task_id = array<i32: 1>} : i64
          %acc_104 = arith.trunci %acc_103 {async_task_id = array<i32: 1>} : i64 to i32
          %v_105 = ttg.memdesc_index %v_5[%acc_104] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
          %acc_106 = ttg.memdesc_index %acc_11[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %122 = ttg.memdesc_index %arg19[%v_80] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %123 = ttg.memdesc_index %arg25[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %124 = ttg.memdesc_index %arg40[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc_107 = arith.extui %acc_96 : i1 to i32
          ttng.wait_barrier %124, %acc_107 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %125 = ttg.memdesc_index %arg26[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %126 = ttg.memdesc_index %arg56[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.wait_barrier %126, %acc_107 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.tc_gen5_mma %acc_99, %v_105, %acc_106, %arg65, %true_20, %122[%true_20], %123[%true_20], %125[%true_20] {async_task_id = array<i32: 1>, is_async, tmem.end = array<i32: 15, 15>, tmem.start = array<i32: 14, 14, 16, 16>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %127 = ttg.memdesc_index %arg19[%k_86] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.wait_barrier %108, %qk_94, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.tc_gen5_mma %q0_46, %k_90, %qk_45, %false, %true_20, %127[%true_20], %107[%true_20] {async_task_id = array<i32: 1>, is_async, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %v_108 = arith.addi %tile_idx_79, %c3_i64 : i64
          %v_109 = arith.divui %v_108, %c3_i64 {async_task_id = array<i32: 1>} : i64
          %v_110 = arith.muli %v_109, %c3_i64 {async_task_id = array<i32: 1>} : i64
          %v_111 = arith.subi %v_108, %v_110 {async_task_id = array<i32: 1>} : i64
          %v_112 = arith.trunci %v_111 {async_task_id = array<i32: 1>} : i64 to i32
          %v_113 = arith.andi %v_109, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
          %v_114 = arith.trunci %v_113 {async_task_id = array<i32: 1>} : i64 to i1
          %v_115 = ttg.memdesc_index %v_5[%v_112] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
          %128 = ttg.memdesc_index %arg13[%v_112] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %129 = arith.extui %v_114 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %128, %129, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc_116 = arith.extui %qk_92 : i1 to i32
          ttng.wait_barrier %112, %acc_116, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.wait_barrier %114, %acc_116, %true_20 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.tc_gen5_mma %acc_56, %v_115, %acc_58, %true_20, %true_20, %111[%true_20], %113[%true_20] {async_task_id = array<i32: 1>, is_async, tmem.end = array<i32: 18, 18>, tmem.start = array<i32: 17, 17, 19, 19>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
          scf.yield %true_20, %offsetkv_y_82, %offsetkv_y_81, %v_112 : i1, i64, i64, i32
        } {async_task_id = array<i32: 1>, tt.warp_specialize}
        %offsetkv_y_61 = arith.addi %offsetkv_y_60#2, %c2_i64 {async_task_id = array<i32: 1>} : i64
        %offsetkv_y_62 = arith.addi %offsetkv_y_60#1, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
        %acc_63 = arith.andi %offsetkv_y_60#1, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
        %acc_64 = arith.trunci %acc_63 {async_task_id = array<i32: 1>} : i64 to i1
        %qk_65 = ttng.tmem_subslice %qk_8 {N = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
        %qk_66 = ttg.memdesc_reinterpret %qk_65 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>
        %acc_67 = ttg.memdesc_index %qk_66[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xbf16, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>
        %acc_68 = arith.addi %offsetkv_y_60#2, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
        %acc_69 = arith.divui %acc_68, %c3_i64 {async_task_id = array<i32: 1>} : i64
        %acc_70 = arith.muli %acc_69, %c3_i64 {async_task_id = array<i32: 1>} : i64
        %acc_71 = arith.subi %acc_68, %acc_70 {async_task_id = array<i32: 1>} : i64
        %acc_72 = arith.trunci %acc_71 {async_task_id = array<i32: 1>} : i64 to i32
        %v_73 = ttg.memdesc_index %v_5[%acc_72] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %acc_74 = ttg.memdesc_index %acc_11[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %115 = ttg.memdesc_index %arg19[%offsetkv_y_60#3] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %116 = ttg.memdesc_index %arg25[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %117 = ttg.memdesc_index %arg40[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %acc_75 = arith.extui %acc_64 : i1 to i32
        ttng.wait_barrier %117, %acc_75 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %118 = ttg.memdesc_index %arg26[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %119 = ttg.memdesc_index %arg56[%c0_i32_23] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %119, %acc_75 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.tc_gen5_mma %acc_67, %v_73, %acc_74, %offsetkv_y_60#0, %true_20, %115[%true_20], %116[%true_20], %118[%true_20], %arg27[%true_20], %arg28[%true_20] {async_task_id = array<i32: 1>, is_async, tmem.end = array<i32: 15, 15>, tmem.start = array<i32: 14, 14, 16, 16>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
        %tile_idx_76 = arith.addi %tile_idx_26, %c1_i64_18 {async_task_id = array<i32: 1>} : i64
        scf.yield {async_task_id = array<i32: 1>} %tile_idx_76, %offsetkv_y_62, %offsetkv_y_61 : i64, i64, i64
      } {async_task_id = array<i32: 1>, tt.data_partition_factor = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return {async_task_id = array<i32: 1>}
    }
    partition1(%Z_3: i32, %H_4: i32, %arg10: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg11: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %v_5: !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable>, %arg13: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %qk_6: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q0_7: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg16: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %qk_8: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q0_9: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg19: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %arg20: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %acc_10: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg22: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg23: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %acc_11: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg25: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg26: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg27: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg28: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %desc_q_12: !tt.ptr<bf16>, %desc_k_13: !tt.ptr<bf16>, %desc_v_14: !tt.ptr<bf16>, %desc_o_15: !tt.ptr<bf16>, %arg33: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg34: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %sm_scale_16: f32, %arg36: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg37: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg38: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg39: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg40: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg41: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg42: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg43: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg44: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg45: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg46: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg47: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg48: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg49: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg50: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg51: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg52: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg53: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg54: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg55: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg56: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg57: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg58: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg59: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>) num_warps(1) {
      %c256_i64 = arith.constant 256 : i64
      %c64_i32 = arith.constant 64 : i32
      %c2_i64 = arith.constant {async_task_id = array<i32: 2>} 2 : i64
      %c3_i64 = arith.constant {async_task_id = array<i32: 2>} 3 : i64
      %true_17 = arith.constant {async_task_id = array<i32: 2>} true
      %c0_i64_18 = arith.constant {async_task_id = array<i32: 2>} 0 : i64
      %n_tile_num = arith.constant {async_task_id = array<i32: 2>} 32 : i32
      %c1_i32_19 = arith.constant {async_task_id = array<i32: 2>} 1 : i32
      %c8192_i32_20 = arith.constant {async_task_id = array<i32: 2>} 8192 : i32
      %c128_i32_21 = arith.constant {async_task_id = array<i32: 2>} 128 : i32
      %c1_i64_22 = arith.constant {async_task_id = array<i32: 2>} 1 : i64
      %c0_i32_23 = arith.constant {async_task_id = array<i32: 2>} 0 : i32
      %c256_i32_24 = arith.constant {async_task_id = array<i32: 2>} 256 : i32
      %prog_id = tt.get_program_id x {async_task_id = array<i32: 2>} : i32
      %num_progs = tt.get_num_programs x {async_task_id = array<i32: 2>} : i32
      %total_tiles = arith.muli %Z_3, %n_tile_num {async_task_id = array<i32: 2>} : i32
      %total_tiles_25 = arith.muli %total_tiles, %H_4 {async_task_id = array<i32: 2>} : i32
      %tiles_per_sm = arith.divsi %total_tiles_25, %num_progs {async_task_id = array<i32: 2>} : i32
      %94 = arith.remsi %total_tiles_25, %num_progs {async_task_id = array<i32: 2>} : i32
      %95 = arith.cmpi slt, %prog_id, %94 {async_task_id = array<i32: 2>} : i32
      %96 = scf.if %95 -> (i32) {
        %tiles_per_sm_34 = arith.addi %tiles_per_sm, %c1_i32_19 {async_task_id = array<i32: 2>} : i32
        scf.yield {async_task_id = array<i32: 2>} %tiles_per_sm_34 : i32
      } else {
        scf.yield {async_task_id = array<i32: 2>} %tiles_per_sm : i32
      } {async_task_id = array<i32: 2>}
      %desc_q_26 = arith.muli %Z_3, %H_4 {async_task_id = array<i32: 2>} : i32
      %desc_q_27 = arith.muli %desc_q_26, %c8192_i32_20 {async_task_id = array<i32: 2>} : i32
      %desc_q_28 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
      ttng.tensormap_create %desc_q_28, %desc_q_12, [%c64_i32, %c128_i32_21], [%c128_i32_21, %desc_q_27], [%c256_i64], [%c1_i32_19, %c1_i32_19] {elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
      ttng.tensormap_fenceproxy_acquire %desc_q_28 : !tt.ptr<i8>
      %desc_q_29 = ttng.reinterpret_tensor_descriptor %desc_q_28 : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16, #shared1>>
      %desc_k_30 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
      ttng.tensormap_create %desc_k_30, %desc_k_13, [%c64_i32, %c128_i32_21], [%c128_i32_21, %desc_q_27], [%c256_i64], [%c1_i32_19, %c1_i32_19] {elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
      ttng.tensormap_fenceproxy_acquire %desc_k_30 : !tt.ptr<i8>
      %desc_k_31 = ttng.reinterpret_tensor_descriptor %desc_k_30 : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16, #shared1>>
      %desc_v_32 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
      ttng.tensormap_create %desc_v_32, %desc_v_14, [%c64_i32, %c128_i32_21], [%c128_i32_21, %desc_q_27], [%c256_i64], [%c1_i32_19, %c1_i32_19] {elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
      ttng.tensormap_fenceproxy_acquire %desc_v_32 : !tt.ptr<i8>
      %desc_v_33 = ttng.reinterpret_tensor_descriptor %desc_v_32 : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16, #shared1>>
      %offset_y = arith.muli %H_4, %c8192_i32_20 {async_task_id = array<i32: 2>} : i32
      %tile_idx:3 = scf.for %tile_idx_34 = %c0_i32_23 to %96 step %c1_i32_19 iter_args(%prog_id_35 = %prog_id, %arg62 = %c0_i64_18, %arg63 = %c0_i64_18) -> (i32, i64, i64)  : i32 {
        %pid = arith.remsi %prog_id_35, %n_tile_num {async_task_id = array<i32: 2>} : i32
        %off_hz = arith.divsi %prog_id_35, %n_tile_num {async_task_id = array<i32: 2>} : i32
        %off_z = arith.divsi %off_hz, %H_4 {async_task_id = array<i32: 2>} : i32
        %off_h = arith.remsi %off_hz, %H_4 {async_task_id = array<i32: 2>} : i32
        %offset_y_36 = arith.muli %off_z, %offset_y {async_task_id = array<i32: 2>} : i32
        %offset_y_37 = arith.muli %off_h, %c8192_i32_20 {async_task_id = array<i32: 2>} : i32
        %offset_y_38 = arith.addi %offset_y_36, %offset_y_37 {async_task_id = array<i32: 2>} : i32
        %qo_offset_y = arith.muli %pid, %c256_i32_24 {async_task_id = array<i32: 2>} : i32
        %qo_offset_y_39 = arith.addi %offset_y_38, %qo_offset_y {async_task_id = array<i32: 2>} : i32
        %q0_40 = arith.addi %qo_offset_y_39, %c128_i32_21 {async_task_id = array<i32: 2>} : i32
        %offsetkv_y = arith.andi %arg62, %c1_i64_22 {async_task_id = array<i32: 2>} : i64
        %offsetkv_y_41 = arith.trunci %offsetkv_y {async_task_id = array<i32: 2>} : i64 to i1
        %97 = ttg.memdesc_index %arg28[%c0_i32_23] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %q0_42 = arith.xori %offsetkv_y_41, %true_17 {async_task_id = array<i32: 2>} : i1
        %q0_43 = arith.extui %q0_42 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %97, %q0_43 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %98 = ttg.memdesc_index %arg11[%c0_i32_23] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.barrier_expect %98, 32768 {async_task_id = array<i32: 2>}, %true_17 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %q0_44 = ttg.memdesc_index %q0_7[%c0_i32_23] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc_q_29[%qo_offset_y_39, %c0_i32_23] %q0_44, %98, %true_17 {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xbf16, #shared1>>, !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %99 = ttg.memdesc_index %arg10[%c0_i32_23] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.barrier_expect %99, 32768 {async_task_id = array<i32: 2>}, %true_17 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %q0_45 = ttg.memdesc_index %q0_9[%c0_i32_23] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc_q_29[%q0_40, %c0_i32_23] %q0_45, %99, %true_17 {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xbf16, #shared1>>, !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %offsetkv_y_46:2 = scf.for %offsetkv_y_49 = %c0_i32_23 to %c8192_i32_20 step %c128_i32_21 iter_args(%offset_y_50 = %offset_y_38, %arg66 = %arg63) -> (i32, i64)  : i32 {
          %k = arith.divui %arg66, %c3_i64 {async_task_id = array<i32: 2>} : i64
          %k_51 = arith.muli %k, %c3_i64 {async_task_id = array<i32: 2>} : i64
          %k_52 = arith.subi %arg66, %k_51 {async_task_id = array<i32: 2>} : i64
          %k_53 = arith.trunci %k_52 {async_task_id = array<i32: 2>} : i64 to i32
          %k_54 = arith.andi %k, %c1_i64_22 {async_task_id = array<i32: 2>} : i64
          %k_55 = arith.trunci %k_54 {async_task_id = array<i32: 2>} : i64 to i1
          %100 = ttg.memdesc_index %arg19[%k_53] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %k_56 = arith.xori %k_55, %true_17 {async_task_id = array<i32: 2>} : i1
          %k_57 = arith.extui %k_56 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %100, %k_57 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %101 = ttg.memdesc_index %arg13[%k_53] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.barrier_expect %101, 32768 {async_task_id = array<i32: 2>}, %true_17 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %k_58 = ttg.memdesc_index %v_5[%k_53] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
          ttng.async_tma_copy_global_to_local %desc_k_31[%offset_y_50, %c0_i32_23] %k_58, %101, %true_17 {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xbf16, #shared1>>, !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
          %v_59 = arith.addi %arg66, %c1_i64_22 {async_task_id = array<i32: 2>} : i64
          %v_60 = arith.divui %v_59, %c3_i64 {async_task_id = array<i32: 2>} : i64
          %v_61 = arith.muli %v_60, %c3_i64 {async_task_id = array<i32: 2>} : i64
          %v_62 = arith.subi %v_59, %v_61 {async_task_id = array<i32: 2>} : i64
          %v_63 = arith.trunci %v_62 {async_task_id = array<i32: 2>} : i64 to i32
          %v_64 = arith.andi %v_60, %c1_i64_22 {async_task_id = array<i32: 2>} : i64
          %v_65 = arith.trunci %v_64 {async_task_id = array<i32: 2>} : i64 to i1
          %102 = ttg.memdesc_index %arg19[%v_63] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %v_66 = arith.xori %v_65, %true_17 {async_task_id = array<i32: 2>} : i1
          %v_67 = arith.extui %v_66 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %102, %v_67 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %103 = ttg.memdesc_index %arg13[%v_63] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.barrier_expect %103, 32768 {async_task_id = array<i32: 2>}, %true_17 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %v_68 = ttg.memdesc_index %v_5[%v_63] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
          ttng.async_tma_copy_global_to_local %desc_v_33[%offset_y_50, %c0_i32_23] %v_68, %103, %true_17 {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xbf16, #shared1>>, !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
          %offsetkv_y_69 = arith.addi %arg66, %c2_i64 {async_task_id = array<i32: 2>} : i64
          %offsetkv_y_70 = arith.addi %offset_y_50, %c128_i32_21 {async_task_id = array<i32: 2>} : i32
          scf.yield %offsetkv_y_70, %offsetkv_y_69 : i32, i64
        } {async_task_id = array<i32: 2>, tt.warp_specialize}
        %tile_idx_47 = arith.addi %prog_id_35, %num_progs {async_task_id = array<i32: 2>} : i32
        %tile_idx_48 = arith.addi %arg62, %c1_i64_22 {async_task_id = array<i32: 2>} : i64
        scf.yield %tile_idx_47, %tile_idx_48, %offsetkv_y_46#1 : i32, i64, i64
      } {async_task_id = array<i32: 2>, tt.data_partition_factor = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return {async_task_id = array<i32: 2>}
    }
    partition2(%Z_3: i32, %H_4: i32, %arg10: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg11: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %v_5: !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable>, %arg13: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %qk_6: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q0_7: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg16: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %qk_8: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q0_9: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg19: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %arg20: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %acc_10: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg22: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg23: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %acc_11: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg25: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg26: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg27: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg28: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %desc_q_12: !tt.ptr<bf16>, %desc_k_13: !tt.ptr<bf16>, %desc_v_14: !tt.ptr<bf16>, %desc_o_15: !tt.ptr<bf16>, %arg33: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg34: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %sm_scale_16: f32, %arg36: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg37: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg38: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg39: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg40: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg41: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg42: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg43: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg44: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg45: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg46: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg47: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg48: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg49: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg50: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg51: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg52: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg53: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg54: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg55: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg56: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg57: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg58: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg59: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>) num_warps(1) {
      %desc_o_17 = arith.constant 256 : i64
      %desc_o_18 = arith.constant 64 : i32
      %c0_i64_19 = arith.constant {async_task_id = array<i32: 3>} 0 : i64
      %n_tile_num = arith.constant {async_task_id = array<i32: 3>} 32 : i32
      %c1_i32_20 = arith.constant {async_task_id = array<i32: 3>} 1 : i32
      %c8192_i32_21 = arith.constant {async_task_id = array<i32: 3>} 8192 : i32
      %c128_i32_22 = arith.constant {async_task_id = array<i32: 3>} 128 : i32
      %c1_i64_23 = arith.constant {async_task_id = array<i32: 3>} 1 : i64
      %c0_i32_24 = arith.constant {async_task_id = array<i32: 3>} 0 : i32
      %c256_i32_25 = arith.constant {async_task_id = array<i32: 3>} 256 : i32
      %prog_id = tt.get_program_id x {async_task_id = array<i32: 3>} : i32
      %num_progs = tt.get_num_programs x {async_task_id = array<i32: 3>} : i32
      %total_tiles = arith.muli %Z_3, %n_tile_num {async_task_id = array<i32: 3>} : i32
      %total_tiles_26 = arith.muli %total_tiles, %H_4 {async_task_id = array<i32: 3>} : i32
      %tiles_per_sm = arith.divsi %total_tiles_26, %num_progs {async_task_id = array<i32: 3>} : i32
      %94 = arith.remsi %total_tiles_26, %num_progs {async_task_id = array<i32: 3>} : i32
      %95 = arith.cmpi slt, %prog_id, %94 {async_task_id = array<i32: 3>} : i32
      %96 = scf.if %95 -> (i32) {
        %tiles_per_sm_31 = arith.addi %tiles_per_sm, %c1_i32_20 {async_task_id = array<i32: 3>} : i32
        scf.yield {async_task_id = array<i32: 3>} %tiles_per_sm_31 : i32
      } else {
        scf.yield {async_task_id = array<i32: 3>} %tiles_per_sm : i32
      } {async_task_id = array<i32: 3>}
      %desc_q_27 = arith.muli %Z_3, %H_4 {async_task_id = array<i32: 3>} : i32
      %desc_q_28 = arith.muli %desc_q_27, %c8192_i32_21 {async_task_id = array<i32: 3>} : i32
      %desc_o_29 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
      ttng.tensormap_create %desc_o_29, %desc_o_15, [%desc_o_18, %c128_i32_22], [%c128_i32_22, %desc_q_28], [%desc_o_17], [%c1_i32_20, %c1_i32_20] {elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
      ttng.tensormap_fenceproxy_acquire %desc_o_29 : !tt.ptr<i8>
      %desc_o_30 = ttng.reinterpret_tensor_descriptor %desc_o_29 : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16, #shared1>>
      %offset_y = arith.muli %H_4, %c8192_i32_21 {async_task_id = array<i32: 3>} : i32
      %tile_idx:2 = scf.for %tile_idx_31 = %c0_i32_24 to %96 step %c1_i32_20 iter_args(%prog_id_32 = %prog_id, %tile_idx_33 = %c0_i64_19) -> (i32, i64)  : i32 {
        %pid = arith.remsi %prog_id_32, %n_tile_num {async_task_id = array<i32: 3>} : i32
        %off_hz = arith.divsi %prog_id_32, %n_tile_num {async_task_id = array<i32: 3>} : i32
        %off_z = arith.divsi %off_hz, %H_4 {async_task_id = array<i32: 3>} : i32
        %off_h = arith.remsi %off_hz, %H_4 {async_task_id = array<i32: 3>} : i32
        %offset_y_34 = arith.muli %off_z, %offset_y {async_task_id = array<i32: 3>} : i32
        %offset_y_35 = arith.muli %off_h, %c8192_i32_21 {async_task_id = array<i32: 3>} : i32
        %offset_y_36 = arith.addi %offset_y_34, %offset_y_35 {async_task_id = array<i32: 3>} : i32
        %qo_offset_y = arith.muli %pid, %c256_i32_25 {async_task_id = array<i32: 3>} : i32
        %qo_offset_y_37 = arith.addi %offset_y_36, %qo_offset_y {async_task_id = array<i32: 3>} : i32
        %97 = arith.addi %qo_offset_y_37, %c128_i32_22 {async_task_id = array<i32: 3>} : i32
        %98 = arith.andi %tile_idx_33, %c1_i64_23 {async_task_id = array<i32: 3>} : i64
        %99 = arith.trunci %98 {async_task_id = array<i32: 3>} : i64 to i1
        %100 = ttg.memdesc_index %arg33[%c0_i32_24] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %101 = ttg.memdesc_index %arg38[%c0_i32_24] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %102 = arith.extui %99 : i1 to i32
        ttng.wait_barrier %101, %102 {async_task_id = array<i32: 3>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        ttng.async_tma_copy_local_to_global %desc_o_30[%qo_offset_y_37, %c0_i32_24] %100 : !tt.tensordesc<tensor<128x128xbf16, #shared1>>, !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        ttng.async_tma_store_wait {pendings = 0 : i32}
        %103 = ttg.memdesc_index %arg39[%c0_i32_24] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.arrive_barrier %103, 1 {async_task_id = array<i32: 3>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %104 = ttg.memdesc_index %arg34[%c0_i32_24] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        %105 = ttg.memdesc_index %arg36[%c0_i32_24] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %105, %102 {async_task_id = array<i32: 3>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        ttng.async_tma_copy_local_to_global %desc_o_30[%97, %c0_i32_24] %104 : !tt.tensordesc<tensor<128x128xbf16, #shared1>>, !ttg.memdesc<128x128xbf16, #shared1, #smem, mutable>
        ttng.async_tma_store_wait {pendings = 0 : i32}
        %106 = ttg.memdesc_index %arg37[%c0_i32_24] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.arrive_barrier %106, 1 {async_task_id = array<i32: 3>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %tile_idx_38 = arith.addi %prog_id_32, %num_progs {async_task_id = array<i32: 3>} : i32
        %tile_idx_39 = arith.addi %tile_idx_33, %c1_i64_23 {async_task_id = array<i32: 3>} : i64
        scf.yield {async_task_id = array<i32: 3>} %tile_idx_38, %tile_idx_39 : i32, i64
      } {async_task_id = array<i32: 3>, tt.data_partition_factor = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return {async_task_id = array<i32: 3>}
    }
    partition3(%Z_3: i32, %H_4: i32, %arg10: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg11: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %v_5: !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable>, %arg13: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %qk_6: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q0_7: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg16: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %qk_8: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q0_9: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg19: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %arg20: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %acc_10: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg22: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg23: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %acc_11: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg25: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg26: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg27: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg28: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %desc_q_12: !tt.ptr<bf16>, %desc_k_13: !tt.ptr<bf16>, %desc_v_14: !tt.ptr<bf16>, %desc_o_15: !tt.ptr<bf16>, %arg33: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg34: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %sm_scale_16: f32, %arg36: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg37: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg38: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg39: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg40: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg41: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg42: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg43: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg44: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg45: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg46: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg47: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg48: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg49: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg50: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg51: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg52: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg53: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg54: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg55: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg56: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg57: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg58: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg59: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>) num_warps(4) {
      %true_17 = arith.constant {async_task_id = array<i32: 4>} true
      %c1_i64_18 = arith.constant {async_task_id = array<i32: 4>} 1 : i64
      %c0_i64_19 = arith.constant {async_task_id = array<i32: 4>} 0 : i64
      %n_tile_num = arith.constant {async_task_id = array<i32: 4>} 32 : i32
      %c1_i32_20 = arith.constant {async_task_id = array<i32: 4>} 1 : i32
      %c8192_i32_21 = arith.constant {async_task_id = array<i32: 4>} 8192 : i32
      %c128_i32_22 = arith.constant {async_task_id = array<i32: 4>} 128 : i32
      %c0_i32_23 = arith.constant {async_task_id = array<i32: 4>} 0 : i32
      %cst_24 = arith.constant {async_task_id = array<i32: 4>} 1.44269502 : f32
      %cst_25 = arith.constant {async_task_id = array<i32: 4>} dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %cst_26 = arith.constant {async_task_id = array<i32: 4>} dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %prog_id = tt.get_program_id x {async_task_id = array<i32: 4>} : i32
      %num_progs = tt.get_num_programs x {async_task_id = array<i32: 4>} : i32
      %total_tiles = arith.muli %Z_3, %n_tile_num {async_task_id = array<i32: 4>} : i32
      %total_tiles_27 = arith.muli %total_tiles, %H_4 {async_task_id = array<i32: 4>} : i32
      %tiles_per_sm = arith.divsi %total_tiles_27, %num_progs {async_task_id = array<i32: 4>} : i32
      %94 = arith.remsi %total_tiles_27, %num_progs {async_task_id = array<i32: 4>} : i32
      %95 = arith.cmpi slt, %prog_id, %94 {async_task_id = array<i32: 4>} : i32
      %96 = scf.if %95 -> (i32) {
        %tiles_per_sm_29 = arith.addi %tiles_per_sm, %c1_i32_20 {async_task_id = array<i32: 4>} : i32
        scf.yield {async_task_id = array<i32: 4>} %tiles_per_sm_29 : i32
      } else {
        scf.yield {async_task_id = array<i32: 4>} %tiles_per_sm : i32
      } {async_task_id = array<i32: 4>}
      %qk_scale = arith.mulf %sm_scale_16, %cst_24 {async_task_id = array<i32: 4>} : f32
      %m_ij = tt.splat %qk_scale {async_task_id = array<i32: 4>} : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %qk_28 = tt.splat %qk_scale {async_task_id = array<i32: 4>} : f32 -> tensor<128x128xf32, #blocked>
      %tile_idx:2 = scf.for %tile_idx_29 = %c0_i32_23 to %96 step %c1_i32_20 iter_args(%arg61 = %c0_i64_19, %arg62 = %c0_i64_19) -> (i64, i64)  : i32 {
        %offsetkv_y:3 = scf.for %offsetkv_y_45 = %c0_i32_23 to %c8192_i32_21 step %c128_i32_22 iter_args(%arg64 = %cst_26, %arg65 = %cst_25, %arg66 = %arg62) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i64)  : i32 {
          %qk_46 = arith.andi %arg66, %c1_i64_18 {async_task_id = array<i32: 4>} : i64
          %qk_47 = arith.trunci %qk_46 {async_task_id = array<i32: 4>} : i64 to i1
          %qk_48 = ttg.memdesc_index %qk_8[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %101 = ttg.memdesc_index %arg20[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %qk_49 = arith.extui %qk_47 {async_task_id = array<i32: 4>} : i1 to i32
          ttng.wait_barrier %101, %qk_49 {async_task_id = array<i32: 4>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %102 = ttg.memdesc_index %arg46[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %qk_50 = ttng.tmem_load %qk_48 {async_task_id = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          ttng.arrive_barrier %102, 1 {async_task_id = array<i32: 4>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %m_ij_51 = "tt.reduce"(%qk_50) <{axis = 1 : i32}> ({
          ^bb0(%m_ij_74: f32, %m_ij_75: f32):
            %m_ij_76 = arith.maxnumf %m_ij_74, %m_ij_75 {async_task_id = array<i32: 4>} : f32
            tt.reduce.return %m_ij_76 {async_task_id = array<i32: 4>} : f32
          }) {async_task_id = array<i32: 4>} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %m_ij_52 = arith.mulf %m_ij_51, %m_ij {async_task_id = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %m_ij_53 = arith.maxnumf %arg65, %m_ij_52 {async_task_id = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %qk_54 = arith.mulf %qk_50, %qk_28 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked>
          %qk_55 = tt.expand_dims %m_ij_53 {async_task_id = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
          %qk_56 = tt.broadcast %qk_55 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
          %qk_57 = arith.subf %qk_54, %qk_56 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked>
          %p = math.exp2 %qk_57 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked>
          %alpha = arith.subf %arg65, %m_ij_53 {async_task_id = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %alpha_58 = math.exp2 %alpha {async_task_id = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %alpha_59 = ttg.convert_layout %alpha_58 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>>
          %alpha_60 = tt.expand_dims %alpha_59 {async_task_id = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
          %qk_61 = ttng.tmem_subslice %qk_8 {N = 64 : i32, async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
          %qk_62 = ttg.memdesc_reinterpret %qk_61 {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
          %alpha_63 = ttg.memdesc_index %qk_62[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
          %103 = ttg.memdesc_index %arg43[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %alpha_64 = arith.xori %qk_47, %true_17 : i1
          %alpha_65 = arith.extui %alpha_64 : i1 to i32
          ttng.wait_barrier %103, %alpha_65 {async_task_id = array<i32: 4>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.tmem_store %alpha_60, %alpha_63, %true_17 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
          %104 = ttg.memdesc_index %arg42[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.arrive_barrier %104, 1 {async_task_id = array<i32: 4>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
          ^bb0(%l_ij_74: f32, %l_ij_75: f32):
            %l_ij_76 = arith.addf %l_ij_74, %l_ij_75 {async_task_id = array<i32: 4>} : f32
            tt.reduce.return %l_ij_76 {async_task_id = array<i32: 4>} : f32
          }) {async_task_id = array<i32: 4>} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %p_66 = arith.truncf %p {async_task_id = array<i32: 4>} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
          %qk_67 = ttng.tmem_subslice %qk_8 {N = 0 : i32, async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
          %qk_68 = ttg.memdesc_reinterpret %qk_67 {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>
          %acc_69 = ttg.memdesc_index %qk_68[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xbf16, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>
          %105 = ttg.memdesc_index %arg25[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc_70 = arith.xori %qk_47, %true_17 {async_task_id = array<i32: 4>} : i1
          %acc_71 = arith.extui %acc_70 {async_task_id = array<i32: 4>} : i1 to i32
          ttng.wait_barrier %105, %acc_71 {async_task_id = array<i32: 4>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.tmem_store %p_66, %acc_69, %true_17 {async_task_id = array<i32: 4>} : tensor<128x128xbf16, #blocked> -> !ttg.memdesc<128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>
          %106 = ttg.memdesc_index %arg40[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.arrive_barrier %106, 1 {async_task_id = array<i32: 4>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %l_i0 = arith.mulf %arg64, %alpha_58 {async_task_id = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %l_i0_72 = arith.addf %l_i0, %l_ij {async_task_id = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %offsetkv_y_73 = arith.addi %arg66, %c1_i64_18 {async_task_id = array<i32: 4>} : i64
          scf.yield %l_i0_72, %m_ij_53, %offsetkv_y_73 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i64
        } {async_task_id = array<i32: 4>, tt.warp_specialize}
        %offsetkv_y_30 = ttg.convert_layout %offsetkv_y#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %offsetkv_y_31 = tt.expand_dims %offsetkv_y_30 {async_task_id = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
        %qk_32 = ttng.tmem_subslice %qk_8 {N = 65 : i32, async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
        %qk_33 = ttg.memdesc_reinterpret %qk_32 {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %offsetkv_y_34 = ttg.memdesc_index %qk_33[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %offsetkv_y_35 = arith.andi %arg61, %c1_i64_18 {async_task_id = array<i32: 4>} : i64
        %offsetkv_y_36 = arith.trunci %offsetkv_y_35 {async_task_id = array<i32: 4>} : i64 to i1
        %97 = ttg.memdesc_index %arg49[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %offsetkv_y_37 = arith.xori %offsetkv_y_36, %true_17 : i1
        %offsetkv_y_38 = arith.extui %offsetkv_y_37 : i1 to i32
        ttng.wait_barrier %97, %offsetkv_y_38 {async_task_id = array<i32: 4>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.tmem_store %offsetkv_y_31, %offsetkv_y_34, %true_17 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %98 = ttg.memdesc_index %arg48[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.arrive_barrier %98, 1 {async_task_id = array<i32: 4>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %offsetkv_y_39 = ttg.convert_layout %offsetkv_y#0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %offsetkv_y_40 = tt.expand_dims %offsetkv_y_39 {async_task_id = array<i32: 4>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
        %qk_41 = ttng.tmem_subslice %qk_8 {N = 66 : i32, async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
        %qk_42 = ttg.memdesc_reinterpret %qk_41 {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %offsetkv_y_43 = ttg.memdesc_index %qk_42[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %99 = ttg.memdesc_index %arg51[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %99, %offsetkv_y_38 {async_task_id = array<i32: 4>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.tmem_store %offsetkv_y_40, %offsetkv_y_43, %true_17 {async_task_id = array<i32: 4>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %100 = ttg.memdesc_index %arg50[%c0_i32_23] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.arrive_barrier %100, 1 {async_task_id = array<i32: 4>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %tile_idx_44 = arith.addi %arg61, %c1_i64_18 {async_task_id = array<i32: 4>} : i64
        scf.yield %tile_idx_44, %offsetkv_y#2 : i64, i64
      } {async_task_id = array<i32: 4>, tt.data_partition_factor = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return {async_task_id = array<i32: 4>}
    }
    partition4(%Z_3: i32, %H_4: i32, %arg10: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg11: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %v_5: !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable>, %arg13: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %qk_6: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q0_7: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg16: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %qk_8: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q0_9: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg19: !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, %arg20: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %acc_10: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg22: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg23: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %acc_11: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg25: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg26: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg27: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg28: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %desc_q_12: !tt.ptr<bf16>, %desc_k_13: !tt.ptr<bf16>, %desc_v_14: !tt.ptr<bf16>, %desc_o_15: !tt.ptr<bf16>, %arg33: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %arg34: !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, %sm_scale_16: f32, %arg36: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg37: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg38: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg39: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg40: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg41: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg42: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg43: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg44: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg45: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg46: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg47: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg48: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg49: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg50: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg51: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg52: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg53: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg54: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg55: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg56: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg57: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg58: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, %arg59: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>) num_warps(4) {
      %true_17 = arith.constant {async_task_id = array<i32: 5>} true
      %c1_i64_18 = arith.constant {async_task_id = array<i32: 5>} 1 : i64
      %c0_i64_19 = arith.constant {async_task_id = array<i32: 5>} 0 : i64
      %n_tile_num = arith.constant {async_task_id = array<i32: 5>} 32 : i32
      %c1_i32_20 = arith.constant {async_task_id = array<i32: 5>} 1 : i32
      %c8192_i32_21 = arith.constant {async_task_id = array<i32: 5>} 8192 : i32
      %c128_i32_22 = arith.constant {async_task_id = array<i32: 5>} 128 : i32
      %c0_i32_23 = arith.constant {async_task_id = array<i32: 5>} 0 : i32
      %cst_24 = arith.constant {async_task_id = array<i32: 5>} 1.44269502 : f32
      %cst_25 = arith.constant {async_task_id = array<i32: 5>} dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %cst_26 = arith.constant {async_task_id = array<i32: 5>} dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %prog_id = tt.get_program_id x {async_task_id = array<i32: 5>} : i32
      %num_progs = tt.get_num_programs x {async_task_id = array<i32: 5>} : i32
      %total_tiles = arith.muli %Z_3, %n_tile_num {async_task_id = array<i32: 5>} : i32
      %total_tiles_27 = arith.muli %total_tiles, %H_4 {async_task_id = array<i32: 5>} : i32
      %tiles_per_sm = arith.divsi %total_tiles_27, %num_progs {async_task_id = array<i32: 5>} : i32
      %94 = arith.remsi %total_tiles_27, %num_progs {async_task_id = array<i32: 5>} : i32
      %95 = arith.cmpi slt, %prog_id, %94 {async_task_id = array<i32: 5>} : i32
      %96 = scf.if %95 -> (i32) {
        %tiles_per_sm_29 = arith.addi %tiles_per_sm, %c1_i32_20 {async_task_id = array<i32: 5>} : i32
        scf.yield {async_task_id = array<i32: 5>} %tiles_per_sm_29 : i32
      } else {
        scf.yield {async_task_id = array<i32: 5>} %tiles_per_sm : i32
      } {async_task_id = array<i32: 5>}
      %qk_scale = arith.mulf %sm_scale_16, %cst_24 {async_task_id = array<i32: 5>} : f32
      %m_ij = tt.splat %qk_scale {async_task_id = array<i32: 5>} : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %qk_28 = tt.splat %qk_scale {async_task_id = array<i32: 5>} : f32 -> tensor<128x128xf32, #blocked>
      %tile_idx:2 = scf.for %tile_idx_29 = %c0_i32_23 to %96 step %c1_i32_20 iter_args(%arg61 = %c0_i64_19, %arg62 = %c0_i64_19) -> (i64, i64)  : i32 {
        %offsetkv_y:3 = scf.for %offsetkv_y_45 = %c0_i32_23 to %c8192_i32_21 step %c128_i32_22 iter_args(%arg64 = %cst_26, %arg65 = %cst_25, %arg66 = %arg62) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i64)  : i32 {
          %qk_46 = arith.andi %arg66, %c1_i64_18 {async_task_id = array<i32: 5>} : i64
          %qk_47 = arith.trunci %qk_46 {async_task_id = array<i32: 5>} : i64 to i1
          %qk_48 = ttg.memdesc_index %qk_6[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %101 = ttg.memdesc_index %arg16[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %qk_49 = arith.extui %qk_47 {async_task_id = array<i32: 5>} : i1 to i32
          ttng.wait_barrier %101, %qk_49 {async_task_id = array<i32: 5>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %102 = ttg.memdesc_index %arg47[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %qk_50 = ttng.tmem_load %qk_48 {async_task_id = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          ttng.arrive_barrier %102, 1 {async_task_id = array<i32: 5>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %m_ij_51 = "tt.reduce"(%qk_50) <{axis = 1 : i32}> ({
          ^bb0(%m_ij_74: f32, %m_ij_75: f32):
            %m_ij_76 = arith.maxnumf %m_ij_74, %m_ij_75 {async_task_id = array<i32: 5>} : f32
            tt.reduce.return %m_ij_76 {async_task_id = array<i32: 5>} : f32
          }) {async_task_id = array<i32: 5>} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %m_ij_52 = arith.mulf %m_ij_51, %m_ij {async_task_id = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %m_ij_53 = arith.maxnumf %arg65, %m_ij_52 {async_task_id = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %qk_54 = arith.mulf %qk_50, %qk_28 {async_task_id = array<i32: 5>} : tensor<128x128xf32, #blocked>
          %qk_55 = tt.expand_dims %m_ij_53 {async_task_id = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
          %qk_56 = tt.broadcast %qk_55 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
          %qk_57 = arith.subf %qk_54, %qk_56 {async_task_id = array<i32: 5>} : tensor<128x128xf32, #blocked>
          %p = math.exp2 %qk_57 {async_task_id = array<i32: 5>} : tensor<128x128xf32, #blocked>
          %alpha = arith.subf %arg65, %m_ij_53 {async_task_id = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %alpha_58 = math.exp2 %alpha {async_task_id = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %alpha_59 = ttg.convert_layout %alpha_58 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>>
          %alpha_60 = tt.expand_dims %alpha_59 {async_task_id = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
          %qk_61 = ttng.tmem_subslice %qk_6 {N = 64 : i32, async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
          %qk_62 = ttg.memdesc_reinterpret %qk_61 {async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
          %alpha_63 = ttg.memdesc_index %qk_62[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
          %103 = ttg.memdesc_index %arg45[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %alpha_64 = arith.xori %qk_47, %true_17 : i1
          %alpha_65 = arith.extui %alpha_64 : i1 to i32
          ttng.wait_barrier %103, %alpha_65 {async_task_id = array<i32: 5>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.tmem_store %alpha_60, %alpha_63, %true_17 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
          %104 = ttg.memdesc_index %arg44[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.arrive_barrier %104, 1 {async_task_id = array<i32: 5>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
          ^bb0(%l_ij_74: f32, %l_ij_75: f32):
            %l_ij_76 = arith.addf %l_ij_74, %l_ij_75 {async_task_id = array<i32: 5>} : f32
            tt.reduce.return %l_ij_76 {async_task_id = array<i32: 5>} : f32
          }) {async_task_id = array<i32: 5>} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %p_66 = arith.truncf %p {async_task_id = array<i32: 5>} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
          %qk_67 = ttng.tmem_subslice %qk_6 {N = 0 : i32, async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
          %qk_68 = ttg.memdesc_reinterpret %qk_67 {async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>
          %acc_69 = ttg.memdesc_index %qk_68[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x128xbf16, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>
          %105 = ttg.memdesc_index %arg22[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %acc_70 = arith.xori %qk_47, %true_17 {async_task_id = array<i32: 5>} : i1
          %acc_71 = arith.extui %acc_70 {async_task_id = array<i32: 5>} : i1 to i32
          ttng.wait_barrier %105, %acc_71 {async_task_id = array<i32: 5>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.tmem_store %p_66, %acc_69, %true_17 {async_task_id = array<i32: 5>} : tensor<128x128xbf16, #blocked> -> !ttg.memdesc<128x128xbf16, #tmem3, #ttng.tensor_memory, mutable>
          %106 = ttg.memdesc_index %arg41[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
          ttng.arrive_barrier %106, 1 {async_task_id = array<i32: 5>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
          %l_i0 = arith.mulf %arg64, %alpha_58 {async_task_id = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %l_i0_72 = arith.addf %l_i0, %l_ij {async_task_id = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %offsetkv_y_73 = arith.addi %arg66, %c1_i64_18 {async_task_id = array<i32: 5>} : i64
          scf.yield %l_i0_72, %m_ij_53, %offsetkv_y_73 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i64
        } {async_task_id = array<i32: 5>, tt.warp_specialize}
        %offsetkv_y_30 = ttg.convert_layout %offsetkv_y#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %offsetkv_y_31 = tt.expand_dims %offsetkv_y_30 {async_task_id = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
        %qk_32 = ttng.tmem_subslice %qk_6 {N = 65 : i32, async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
        %qk_33 = ttg.memdesc_reinterpret %qk_32 {async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %offsetkv_y_34 = ttg.memdesc_index %qk_33[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %offsetkv_y_35 = arith.andi %arg61, %c1_i64_18 {async_task_id = array<i32: 5>} : i64
        %offsetkv_y_36 = arith.trunci %offsetkv_y_35 {async_task_id = array<i32: 5>} : i64 to i1
        %97 = ttg.memdesc_index %arg53[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %offsetkv_y_37 = arith.xori %offsetkv_y_36, %true_17 : i1
        %offsetkv_y_38 = arith.extui %offsetkv_y_37 : i1 to i32
        ttng.wait_barrier %97, %offsetkv_y_38 {async_task_id = array<i32: 5>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.tmem_store %offsetkv_y_31, %offsetkv_y_34, %true_17 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %98 = ttg.memdesc_index %arg52[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.arrive_barrier %98, 1 {async_task_id = array<i32: 5>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %offsetkv_y_39 = ttg.convert_layout %offsetkv_y#0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %offsetkv_y_40 = tt.expand_dims %offsetkv_y_39 {async_task_id = array<i32: 5>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xf32, #blocked3>
        %qk_41 = ttng.tmem_subslice %qk_6 {N = 66 : i32, async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
        %qk_42 = ttg.memdesc_reinterpret %qk_41 {async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %offsetkv_y_43 = ttg.memdesc_index %qk_42[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %99 = ttg.memdesc_index %arg55[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.wait_barrier %99, %offsetkv_y_38 {async_task_id = array<i32: 5>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.tmem_store %offsetkv_y_40, %offsetkv_y_43, %true_17 {async_task_id = array<i32: 5>} : tensor<128x1xf32, #blocked3> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
        %100 = ttg.memdesc_index %arg54[%c0_i32_23] {async_task_id = array<i32: 5>} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
        ttng.arrive_barrier %100, 1 {async_task_id = array<i32: 5>} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
        %tile_idx_44 = arith.addi %arg61, %c1_i64_18 {async_task_id = array<i32: 5>} : i64
        scf.yield %tile_idx_44, %offsetkv_y#2 : i64, i64
      } {async_task_id = array<i32: 5>, tt.data_partition_factor = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return {async_task_id = array<i32: 5>}
    } : (i32, i32, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<3x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !tt.ptr<bf16>, !tt.ptr<bf16>, !tt.ptr<bf16>, !tt.ptr<bf16>, !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<1x128x128xbf16, #shared1, #smem, mutable>, f32, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}
