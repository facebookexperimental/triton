// RUN: triton-opt %s --tritongpu-optimize-partition-warps | FileCheck %s
// CHECK: ttg.warp_specialize
// CHECK-SAME: requestedRegisters = array<i32: 88, 88, 88, 40>
// CHECK-SAME: ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"]

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [8, 1, 4], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 128], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64], [32, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[128, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 32]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 1, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 0, 1]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 64], [0, 32]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0], [0, 1, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0], [0, 0, 1]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 0, 1], [0, 16, 0], [0, 1, 0], [0, 2, 0], [64, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0], [32, 0, 0]], block = []}>
#linear9 = #ttg.linear<{register = [[0, 16], [0, 1], [0, 2], [64, 0]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1, twoCTAs = true>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2, twoCTAs = true>
#tmem3 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true>
#tmem4 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true, ctaMode = twocta_rhs>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 88 : i32, ttg.min_reg_auto_ws = 88 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @_attn_bwd(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<64x128xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf16, #shared>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<256x64xf16, #shared>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.tensordesc<128x128xf16, #shared>, %arg21: i32, %arg22: i32, %arg23: i64, %arg24: i64, %arg25: f32, %arg26: !tt.tensordesc<128x64xf16, #shared>, %arg27: i32, %arg28: i32, %arg29: i64, %arg30: i64, %arg31: !tt.tensordesc<64x128xf16, #shared>, %arg32: i32, %arg33: i32, %arg34: i64, %arg35: i64, %arg36: !tt.tensordesc<128x16xf32, #shared1>, %arg37: i32, %arg38: i32, %arg39: i64, %arg40: i64, %arg41: !tt.tensordesc<128x64xf16, #shared>, %arg42: i32, %arg43: i32, %arg44: i64, %arg45: i64, %arg46: !tt.tensordesc<128x64xf16, #shared>, %arg47: i32, %arg48: i32, %arg49: i64, %arg50: i64, %arg51: !tt.tensordesc<128xf32, #shared2>, %arg52: i32, %arg53: i64, %arg54: !tt.tensordesc<128xf32, #shared2>, %arg55: i32, %arg56: i64, %arg57: i32 {tt.divisibility = 16 : i32}, %arg58: i32 {tt.divisibility = 16 : i32}, %arg59: i32 {tt.divisibility = 16 : i32}, %arg60: i32, %arg61: i32, %arg62: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
    %c0_i64 = arith.constant {async_task_id = array<i32: 0>} 0 : i64
    %c1_i32 = arith.constant {async_task_id = array<i32: 0>} 1 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0>} 128 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0>} 64 : i32
    %true = arith.constant {async_task_id = array<i32: 0>} true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %1 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %2 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %2, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %3 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %4 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %5 = ttg.memdesc_index %4[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %6 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %7 = ttg.memdesc_index %6[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %8 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %9 = ttg.memdesc_index %8[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %10 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %11 = ttg.memdesc_index %10[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %11, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %12 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %13 = ttg.memdesc_index %12[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %13, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %14 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %15 = ttg.memdesc_index %14[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %15, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %16 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %17 = ttg.memdesc_index %16[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %17, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %18 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %19 = ttg.memdesc_index %18[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %19, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %20 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %21 = ttg.memdesc_index %20[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %21, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %22 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %23 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %23, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %24 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %25 = ttg.memdesc_index %24[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %25, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %26 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %27 = ttg.memdesc_index %26[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %27, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %28 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %29 = ttg.memdesc_index %28[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %29, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %30 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %31 = ttg.memdesc_index %30[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %31, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %32 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %33 = ttg.memdesc_index %32[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %33, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %34 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %35 = ttg.memdesc_index %34[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %35, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %36 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %37 = ttg.memdesc_index %36[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %37, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %38 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %39 = ttg.memdesc_index %38[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %39, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %40 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %41 = ttg.memdesc_index %40[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %41, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %42 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %43 = ttg.memdesc_index %42[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %43, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %44 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %45 = ttg.memdesc_index %44[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %45, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %46 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %47 = ttg.memdesc_index %46[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %47, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %48 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %49 = ttg.memdesc_index %48[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %49, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %50 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %51 = ttg.memdesc_index %50[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %51, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %52 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %53 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %54 = ttg.memdesc_index %52[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %54, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %55 = ttg.memdesc_index %53[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %55, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %56 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %57 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %58 = ttg.memdesc_index %56[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %58, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %59 = ttg.memdesc_index %57[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %59, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %60 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %61 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %62 = ttg.memdesc_index %60[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %62, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %63 = ttg.memdesc_index %61[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %63, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %64 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %65 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %66 = ttg.memdesc_index %64[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %66, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %67 = ttg.memdesc_index %65[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %67, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %68 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %69 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %70 = ttg.memdesc_index %68[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %70, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %71 = ttg.memdesc_index %69[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %71, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %72 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %73 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %74 = ttg.memdesc_index %72[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %74, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %75 = ttg.memdesc_index %73[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %75, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %76 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %77 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %78 = ttg.memdesc_index %76[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %78, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %79 = ttg.memdesc_index %77[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %79, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %80 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %81 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %82 = ttg.memdesc_index %80[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %82, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %83 = ttg.memdesc_index %81[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %83, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %84 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %85 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %86 = ttg.memdesc_index %84[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %86, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %87 = ttg.memdesc_index %85[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %87, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %88 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %89 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %90 = ttg.memdesc_index %88[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %90, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %91 = ttg.memdesc_index %89[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %91, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %92 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %93 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %94 = ttg.memdesc_index %92[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %94, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %95 = ttg.memdesc_index %93[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %95, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %96 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %97 = ttg.memdesc_reinterpret %96 {allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 26 : i32} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %98 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 12 : i32} : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %99 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %result, %token = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_0, %token_1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 10 : i32} : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %100 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %101 = ttg.memdesc_reinterpret %100 {allocation.shareGroup = 2 : i32, buffer.copy = 1 : i32, buffer.id = 28 : i32} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %102 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 15 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %103 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 16 : i32} : () -> !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>
    %result_2, %token_3 = ttng.tmem_alloc {allocation.shareGroup = 1 : i32, buffer.copy = 1 : i32, buffer.id = 2 : i32} : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %104 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 17 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %105 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %result_4, %token_5 = ttng.tmem_alloc {allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %106 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 19 : i32} : () -> !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>
    %107 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 20 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %108 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %109 = ttg.local_alloc {allocation.shareGroup = 3 : i32, buffer.copy = 1 : i32, buffer.id = 22 : i32, buffer.tmaStaging = 2 : i32} : () -> !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>
    ttg.warp_specialize(%arg59, %arg62, %arg61, %arg58, %arg57, %result_4, %4, %109, %arg36, %50, %46, %42, %100, %34, %96, %result_2, %32, %24, %18, %105, %16, %99, %14, %12, %result, %104, %20, %22, %result_0, %102, %28, %30, %8, %108, %98, %6, %36, %38, %40, %44, %48, %arg10, %arg15, %arg20, %arg5, %arg0, %26, %103, %arg51, %arg26, %arg31, %10, %106, %arg54, %107, %0, %1, %52, %53, %56, %57, %60, %61, %64, %65, %68, %69, %72, %73, %76, %77, %80, %81, %84, %85, %88, %89, %92, %93) attributes {ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"]}
    default {
      %158 = tt.get_program_id z {async_task_id = array<i32: 0>} : i32
      %159 = tt.get_program_id x {async_task_id = array<i32: 0>} : i32
      %160 = tt.get_num_programs y {async_task_id = array<i32: 0>} : i32
      %161 = arith.extsi %arg59 {async_task_id = array<i32: 0>} : i32 to i64
      %162 = arith.muli %159, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %163 = arith.extsi %162 {async_task_id = array<i32: 0>} : i32 to i64
      %164 = arith.divsi %arg62, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %165 = tt.splat %arg25 {async_task_id = array<i32: 0>} : f32 -> tensor<128x64xf32, #blocked>
      %166:2 = scf.for %arg63 = %c0_i32 to %160 step %c1_i32 iter_args(%arg64 = %c0_i64, %arg65 = %c0_i64) -> (i64, i64)  : i32 {
        %167 = arith.remsi %158, %arg61 {async_task_id = array<i32: 0>} : i32
        %168 = arith.muli %arg58, %167 {async_task_id = array<i32: 0>} : i32
        %169 = arith.divsi %158, %arg61 {async_task_id = array<i32: 0>} : i32
        %170 = arith.muli %arg57, %169 {async_task_id = array<i32: 0>} : i32
        %171 = arith.addi %168, %170 {async_task_id = array<i32: 0>} : i32
        %172 = arith.extsi %171 {async_task_id = array<i32: 0>} : i32 to i64
        %173 = arith.divsi %172, %161 {async_task_id = array<i32: 0>} : i64
        %174 = arith.addi %173, %163 {async_task_id = array<i32: 0>} : i64
        %175 = arith.trunci %174 {async_task_id = array<i32: 0>} : i64 to i32
        %176 = arith.andi %arg64, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %177 = arith.trunci %176 {async_task_id = array<i32: 0>} : i64 to i1
        %178 = arith.andi %arg64, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %179 = arith.trunci %178 {async_task_id = array<i32: 0>} : i64 to i1
        %180 = scf.for %arg66 = %c0_i32 to %164 step %c1_i32 iter_args(%arg67 = %arg65) -> (i64)  : i32 {
          %215 = arith.andi %arg67, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %216 = arith.trunci %215 {async_task_id = array<i32: 0>} : i64 to i1
          %217 = arith.andi %arg67, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %218 = arith.trunci %217 {async_task_id = array<i32: 0>} : i64 to i1
          %219 = arith.andi %arg67, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %220 = arith.trunci %219 {async_task_id = array<i32: 0>} : i64 to i1
          %221 = ttg.memdesc_index %103[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
          %222 = ttg.memdesc_index %26[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %223 = arith.extui %216 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %222, %223, %true {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %224 = ttg.local_load %221 {async_task_id = array<i32: 0>} : !ttg.memdesc<128xf32, #shared2, #smem, mutable> -> tensor<128xf32, #blocked1>
          %225 = ttg.memdesc_index %61[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %225, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %226 = ttg.convert_layout %224 {async_task_id = array<i32: 0>} : tensor<128xf32, #blocked1> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
          %227 = tt.expand_dims %226 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
          %228 = tt.broadcast %227 {async_task_id = array<i32: 0>} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
          %229 = ttg.memdesc_index %result_2[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %230 = ttg.memdesc_index %24[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %231 = arith.extui %218 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %230, %231 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %result_12, %token_13 = ttng.tmem_load %229[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
          %232 = ttg.memdesc_index %65[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %232, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %233 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            sub.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {async_task_id = array<i32: 0>, constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true} %result_12, %228 : tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear>
          %234 = math.exp2 %233 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear>
          %235 = arith.truncf %234 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
          %236 = ttng.tmem_subslice %result_2 {N = 0 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
          %237 = ttg.memdesc_reinterpret %236 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
          %238 = ttg.memdesc_index %237[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %239 = ttg.memdesc_index %65[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %240 = arith.extui %220 : i1 to i32
          ttng.wait_barrier %239, %240 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {dstTask = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.tmem_store %235, %238, %true {async_task_id = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %241 = ttg.memdesc_index %68[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %241, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %242 = arith.andi %arg67, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %243 = arith.trunci %242 {async_task_id = array<i32: 0>} : i64 to i1
          %244 = arith.andi %arg67, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %245 = arith.trunci %244 {async_task_id = array<i32: 0>} : i64 to i1
          %246 = ttg.memdesc_index %106[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
          %247 = ttg.memdesc_index %10[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %248 = arith.extui %245 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %247, %248, %true {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %249 = ttg.local_load %246 {async_task_id = array<i32: 0>} : !ttg.memdesc<128xf32, #shared2, #smem, mutable> -> tensor<128xf32, #blocked1>
          %250 = ttg.memdesc_index %77[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %250, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %251 = ttg.convert_layout %249 {async_task_id = array<i32: 0>} : tensor<128xf32, #blocked1> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
          %252 = tt.expand_dims %251 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
          %253 = tt.broadcast %252 {async_task_id = array<i32: 0>} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
          %254 = ttg.memdesc_index %result_4[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %255 = ttg.memdesc_index %12[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %256 = arith.extui %243 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %255, %256 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %result_14, %token_15 = ttng.tmem_load %254[] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
          %257 = ttg.memdesc_index %73[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %257, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %258 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            sub.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {async_task_id = array<i32: 0>, constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true} %result_14, %253 : tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear>
          %259 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {async_task_id = array<i32: 0>, constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true} %234, %258 : tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear>
          %260 = arith.truncf %259 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
          %261 = ttng.tmem_subslice %result_4 {N = 0 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
          %262 = ttg.memdesc_reinterpret %261 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
          %263 = ttg.memdesc_index %262[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %264 = arith.andi %arg67, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %265 = arith.trunci %264 {async_task_id = array<i32: 0>} : i64 to i1
          %266 = ttg.memdesc_index %8[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %267 = arith.xori %265, %true {async_task_id = array<i32: 0>} : i1
          %268 = arith.extui %267 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %266, %268 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.tmem_store %260, %263, %true {async_task_id = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %269 = ttg.memdesc_index %80[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %269, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %270 = ttng.two_cta_peer_gather %260 split_dim = 1 num_ctas = 2 {async_task_id = array<i32: 0>} : tensor<128x128xf16, #linear> -> tensor<64x256xf16, #linear1>
          %271 = tt.trans %270 {async_task_id = array<i32: 0>, order = array<i32: 1, 0>} : tensor<64x256xf16, #linear1> -> tensor<256x64xf16, #linear2>
          %272 = tt.reshape %260 {async_task_id = array<i32: 0>} : tensor<128x128xf16, #linear> -> tensor<128x2x64xf16, #linear3>
          %273 = tt.trans %272 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf16, #linear3> -> tensor<128x64x2xf16, #linear4>
          %274 = ttg.convert_layout %273 {async_task_id = array<i32: 0>} : tensor<128x64x2xf16, #linear4> -> tensor<128x64x2xf16, #blocked2>
          %outLHS_16, %outRHS_17 = tt.split %274 {async_task_id = array<i32: 0>} : tensor<128x64x2xf16, #blocked2> -> tensor<128x64xf16, #blocked3>
          %275 = ttg.memdesc_index %107[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %276 = arith.andi %arg67, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %277 = arith.trunci %276 {async_task_id = array<i32: 0>} : i64 to i1
          %278 = ttg.memdesc_index %85[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %279 = arith.xori %277, %true : i1
          %280 = arith.extui %279 : i1 to i32
          ttng.wait_barrier %278, %280 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 4>, dstTask = 4 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttg.local_store %outLHS_16, %275 {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %281 = ttg.memdesc_index %84[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %281, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 4>, dstTask = 4 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %282 = ttg.memdesc_index %108[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %283 = arith.andi %arg67, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %284 = arith.trunci %283 {async_task_id = array<i32: 0>} : i64 to i1
          %285 = ttg.memdesc_index %6[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %286 = arith.xori %284, %true {async_task_id = array<i32: 0>} : i1
          %287 = arith.extui %286 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %285, %287 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttg.local_store %271, %282 {async_task_id = array<i32: 0>} : tensor<256x64xf16, #linear2> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %288 = ttg.memdesc_index %88[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %288, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %289 = arith.addi %arg67, %c1_i64 {async_task_id = array<i32: 0>} : i64
          scf.yield {async_task_id = array<i32: 0>} %289 : i64
        } {async_task_id = array<i32: 0>, tt.warp_specialize}
        %181 = ttg.memdesc_index %result[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %182 = ttg.memdesc_index %38[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %183 = arith.extui %177 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %182, %183 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %result_6, %token_7 = ttng.tmem_load %181[] {async_task_id = array<i32: 0>, tmem.end = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %184 = ttg.memdesc_index %53[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %184, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %185 = tt.reshape %result_6 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear3>
        %186 = tt.trans %185 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear3> -> tensor<128x64x2xf32, #linear4>
        %187 = ttg.convert_layout %186 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #linear4> -> tensor<128x64x2xf32, #blocked4>
        %outLHS, %outRHS = tt.split %187 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked>
        %188 = arith.truncf %outLHS {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
        %189 = ttg.memdesc_index %97[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttg.local_store %188, %189 {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %190 = ttg.memdesc_index %97[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %191 = ttng.async_tma_copy_local_to_global %arg46[%175, %c0_i32] %190 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %191   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %192 = arith.truncf %outRHS {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
        %193 = ttg.memdesc_index %97[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttg.local_store %192, %193 {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %194 = ttg.memdesc_index %97[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %195 = ttng.async_tma_copy_local_to_global %arg46[%175, %c64_i32] %194 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %195   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %196 = ttg.memdesc_index %result_0[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %197 = ttg.memdesc_index %36[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %198 = arith.extui %179 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %197, %198 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %result_8, %token_9 = ttng.tmem_load %196[] {async_task_id = array<i32: 0>, tmem.end = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %199 = ttg.memdesc_index %57[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %199, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %200 = tt.reshape %result_8 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear3>
        %201 = tt.trans %200 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear3> -> tensor<128x64x2xf32, #linear4>
        %202 = ttg.convert_layout %201 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #linear4> -> tensor<128x64x2xf32, #blocked4>
        %outLHS_10, %outRHS_11 = tt.split %202 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked>
        %203 = arith.mulf %outLHS_10, %165 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked>
        %204 = arith.truncf %203 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
        %205 = ttg.memdesc_index %101[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttg.local_store %204, %205 {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %206 = ttg.memdesc_index %101[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %207 = ttng.async_tma_copy_local_to_global %arg41[%175, %c0_i32] %206 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %207   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %208 = arith.mulf %outRHS_11, %165 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked>
        %209 = arith.truncf %208 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
        %210 = ttg.memdesc_index %101[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttg.local_store %209, %210 {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %211 = ttg.memdesc_index %101[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %212 = ttng.async_tma_copy_local_to_global %arg41[%175, %c64_i32] %211 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %212   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %213 = arith.addi %arg64, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %214 = ttg.memdesc_index %1[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %214, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 3 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        scf.yield {async_task_id = array<i32: 0>} %213, %180 : i64, i64
      } {async_task_id = array<i32: 0>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_yield
    }
    partition0(%arg63: i32, %arg64: i32, %arg65: i32, %arg66: i32, %arg67: i32, %arg68: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg69: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg70: !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, %arg71: !tt.tensordesc<128x16xf32, #shared1>, %arg72: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg73: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg74: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg75: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg76: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg77: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg78: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg79: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg80: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg81: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg82: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg83: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg84: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg85: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg86: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg87: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg88: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg89: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg90: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg91: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg92: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg93: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg94: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg95: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg96: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg97: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg98: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg99: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg100: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg101: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg102: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg103: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg104: !tt.tensordesc<128x128xf16, #shared>, %arg105: !tt.tensordesc<256x64xf16, #shared>, %arg106: !tt.tensordesc<128x128xf16, #shared>, %arg107: !tt.tensordesc<64x128xf16, #shared>, %arg108: !tt.tensordesc<128x64xf16, #shared>, %arg109: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg110: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %arg111: !tt.tensordesc<128xf32, #shared2>, %arg112: !tt.tensordesc<128x64xf16, #shared>, %arg113: !tt.tensordesc<64x128xf16, #shared>, %arg114: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg115: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %arg116: !tt.tensordesc<128xf32, #shared2>, %arg117: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg118: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg119: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg120: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg121: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg122: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg126: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg127: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg128: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg129: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg130: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg131: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg132: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg133: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg134: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg135: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg136: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg137: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg138: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg139: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg140: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg141: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) num_warps(8) {
      %c1_i64_6 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
      %c0_i64_7 = arith.constant {async_task_id = array<i32: 1>} 0 : i64
      %c64_i32_8 = arith.constant {async_task_id = array<i32: 1>} 64 : i32
      %c2_i32 = arith.constant {async_task_id = array<i32: 1>} 2 : i32
      %c128_i32_9 = arith.constant {async_task_id = array<i32: 1>} 128 : i32
      %c2_i64 = arith.constant {async_task_id = array<i32: 1>} 2 : i64
      %c0_i32_10 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
      %c1_i32_11 = arith.constant {async_task_id = array<i32: 1>} 1 : i32
      %c48_i32 = arith.constant {async_task_id = array<i32: 1>} 48 : i32
      %c32_i32 = arith.constant {async_task_id = array<i32: 1>} 32 : i32
      %c16_i32 = arith.constant {async_task_id = array<i32: 1>} 16 : i32
      %cst = arith.constant {async_task_id = array<i32: 1>} dense<0.693147182> : tensor<128x16xf32, #blocked5>
      %158 = tt.get_program_id z {async_task_id = array<i32: 1>} : i32
      %159 = tt.get_program_id x {async_task_id = array<i32: 1>} : i32
      %160 = tt.get_num_programs y {async_task_id = array<i32: 1>} : i32
      %161 = arith.extsi %arg63 {async_task_id = array<i32: 1>} : i32 to i64
      %162 = arith.remsi %159, %c2_i32 {async_task_id = array<i32: 1>} : i32
      %163 = arith.divsi %arg64, %c128_i32_9 {async_task_id = array<i32: 1>} : i32
      %164 = arith.muli %162, %c64_i32_8 {async_task_id = array<i32: 1>} : i32
      %165 = arith.extsi %164 {async_task_id = array<i32: 1>} : i32 to i64
      %166 = scf.for %arg142 = %c0_i32_10 to %160 step %c1_i32_11 iter_args(%arg143 = %c0_i64_7) -> (i64)  : i32 {
        %167 = arith.remsi %158, %arg65 {async_task_id = array<i32: 1>} : i32
        %168 = arith.muli %arg66, %167 {async_task_id = array<i32: 1>} : i32
        %169 = arith.divsi %158, %arg65 {async_task_id = array<i32: 1>} : i32
        %170 = arith.muli %arg67, %169 {async_task_id = array<i32: 1>} : i32
        %171 = arith.addi %168, %170 {async_task_id = array<i32: 1>} : i32
        %172 = arith.extsi %171 {async_task_id = array<i32: 1>} : i32 to i64
        %173 = arith.divsi %172, %161 {async_task_id = array<i32: 1>} : i64
        %174:2 = scf.for %arg144 = %c0_i32_10 to %163 step %c1_i32_11 iter_args(%arg145 = %c0_i32_10, %arg146 = %arg143) -> (i32, i64)  : i32 {
          %175 = arith.extsi %arg145 {async_task_id = array<i32: 1>} : i32 to i64
          %176 = arith.addi %173, %175 {async_task_id = array<i32: 1>} : i64
          %177 = arith.andi %arg146, %c1_i64_6 {async_task_id = array<i32: 1>} : i64
          %178 = arith.trunci %177 {async_task_id = array<i32: 1>} : i64 to i1
          %179 = ttng.tmem_subslice %arg68 {N = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %180 = ttg.memdesc_reinterpret %179 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x128xf32, #tmem3, #ttng.tensor_memory, mutable>
          %181 = ttg.memdesc_index %180[%c0_i32_10] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x64x128xf32, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem4, #ttng.tensor_memory, mutable>
          %182 = ttg.memdesc_index %arg69[%c0_i32_10] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %183 = arith.extui %178 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %182, %183 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3, 4>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %result_12, %token_13 = ttng.tmem_load %181[] {async_task_id = array<i32: 1>} : !ttg.memdesc<64x128xf32, #tmem4, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear5>
          %184 = ttg.memdesc_index %arg141[%c0_i32_10] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %184, 1 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3, 4>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %185 = tt.reshape %result_12 {async_task_id = array<i32: 1>} : tensor<64x128xf32, #linear5> -> tensor<128x2x32xf32, #linear6>
          %186 = tt.trans %185 {async_task_id = array<i32: 1>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear6> -> tensor<128x32x2xf32, #linear7>
          %187 = ttg.convert_layout %186 {async_task_id = array<i32: 1>} : tensor<128x32x2xf32, #linear7> -> tensor<128x32x2xf32, #linear8>
          %outLHS, %outRHS = tt.split %187 {async_task_id = array<i32: 1>} : tensor<128x32x2xf32, #linear8> -> tensor<128x32xf32, #linear9>
          %188 = tt.reshape %outLHS {async_task_id = array<i32: 1>} : tensor<128x32xf32, #linear9> -> tensor<128x2x16xf32, #blocked6>
          %189 = tt.trans %188 {async_task_id = array<i32: 1>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked6> -> tensor<128x16x2xf32, #blocked7>
          %outLHS_14, %outRHS_15 = tt.split %189 {async_task_id = array<i32: 1>} : tensor<128x16x2xf32, #blocked7> -> tensor<128x16xf32, #blocked5>
          %190 = tt.reshape %outRHS {async_task_id = array<i32: 1>} : tensor<128x32xf32, #linear9> -> tensor<128x2x16xf32, #blocked6>
          %191 = tt.trans %190 {async_task_id = array<i32: 1>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked6> -> tensor<128x16x2xf32, #blocked7>
          %outLHS_16, %outRHS_17 = tt.split %191 {async_task_id = array<i32: 1>} : tensor<128x16x2xf32, #blocked7> -> tensor<128x16xf32, #blocked5>
          %192 = arith.addi %176, %165 {async_task_id = array<i32: 1>} : i64
          %193 = arith.muli %192, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %194 = arith.mulf %outLHS_14, %cst {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked5>
          %195 = arith.trunci %193 {async_task_id = array<i32: 1>} : i64 to i32
          %196 = ttg.memdesc_index %arg70[%c0_i32_10] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          ttg.local_store %194, %196 {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked5> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          %197 = ttg.memdesc_index %arg70[%c0_i32_10] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          %198 = ttng.async_tma_reduce add, %arg71[%195, %c0_i32_10] %197 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          ttng.async_tma_store_token_wait %198   {async_task_id = array<i32: 1>} : !ttg.async.token
          %199 = arith.mulf %outRHS_15, %cst {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked5>
          %200 = ttg.memdesc_index %arg70[%c0_i32_10] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          ttg.local_store %199, %200 {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked5> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          %201 = ttg.memdesc_index %arg70[%c0_i32_10] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          %202 = ttng.async_tma_reduce add, %arg71[%195, %c16_i32] %201 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          ttng.async_tma_store_token_wait %202   {async_task_id = array<i32: 1>} : !ttg.async.token
          %203 = arith.mulf %outLHS_16, %cst {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked5>
          %204 = ttg.memdesc_index %arg70[%c0_i32_10] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          ttg.local_store %203, %204 {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked5> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          %205 = ttg.memdesc_index %arg70[%c0_i32_10] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          %206 = ttng.async_tma_reduce add, %arg71[%195, %c32_i32] %205 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          ttng.async_tma_store_token_wait %206   {async_task_id = array<i32: 1>} : !ttg.async.token
          %207 = arith.mulf %outRHS_17, %cst {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked5>
          %208 = ttg.memdesc_index %arg70[%c0_i32_10] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          ttg.local_store %207, %208 {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked5> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          %209 = ttg.memdesc_index %arg70[%c0_i32_10] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          %210 = ttng.async_tma_reduce add, %arg71[%195, %c48_i32] %209 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          ttng.async_tma_store_token_wait %210   {async_task_id = array<i32: 1>} : !ttg.async.token
          %211 = arith.addi %arg145, %c128_i32_9 {async_task_id = array<i32: 1>} : i32
          %212 = arith.addi %arg146, %c1_i64_6 {async_task_id = array<i32: 1>} : i64
          scf.yield {async_task_id = array<i32: 1>} %211, %212 : i32, i64
        } {async_task_id = array<i32: 1>, tt.warp_specialize}
        scf.yield {async_task_id = array<i32: 1>} %174#1 : i64
      } {async_task_id = array<i32: 1>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition1(%arg63: i32, %arg64: i32, %arg65: i32, %arg66: i32, %arg67: i32, %arg68: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg69: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg70: !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, %arg71: !tt.tensordesc<128x16xf32, #shared1>, %arg72: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg73: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg74: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg75: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg76: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg77: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg78: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg79: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg80: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg81: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg82: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg83: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg84: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg85: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg86: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg87: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg88: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg89: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg90: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg91: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg92: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg93: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg94: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg95: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg96: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg97: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg98: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg99: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg100: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg101: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg102: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg103: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg104: !tt.tensordesc<128x128xf16, #shared>, %arg105: !tt.tensordesc<256x64xf16, #shared>, %arg106: !tt.tensordesc<128x128xf16, #shared>, %arg107: !tt.tensordesc<64x128xf16, #shared>, %arg108: !tt.tensordesc<128x64xf16, #shared>, %arg109: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg110: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %arg111: !tt.tensordesc<128xf32, #shared2>, %arg112: !tt.tensordesc<128x64xf16, #shared>, %arg113: !tt.tensordesc<64x128xf16, #shared>, %arg114: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg115: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %arg116: !tt.tensordesc<128xf32, #shared2>, %arg117: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg118: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg119: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg120: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg121: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg122: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg126: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg127: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg128: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg129: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg130: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg131: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg132: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg133: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg134: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg135: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg136: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg137: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg138: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg139: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg140: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg141: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) num_warps(8) {
      %158 = ub.poison : i1
      %c1_i64_6 = arith.constant {async_task_id = array<i32: 2>} 1 : i64
      %c0_i64_7 = arith.constant {async_task_id = array<i32: 2>} 0 : i64
      %true_8 = arith.constant {async_task_id = array<i32: 2>} true
      %c128_i32_9 = arith.constant {async_task_id = array<i32: 2>} 128 : i32
      %c0_i32_10 = arith.constant {async_task_id = array<i32: 2>} 0 : i32
      %c1_i32_11 = arith.constant {async_task_id = array<i32: 2>} 1 : i32
      %false = arith.constant {async_task_id = array<i32: 2>} false
      %159 = tt.get_num_programs y {async_task_id = array<i32: 2>} : i32
      %160 = arith.divsi %arg64, %c128_i32_9 {async_task_id = array<i32: 2>} : i32
      %161:2 = scf.for %arg142 = %c0_i32_10 to %159 step %c1_i32_11 iter_args(%arg143 = %c0_i64_7, %arg144 = %c0_i64_7) -> (i64, i64)  : i32 {
        %162 = arith.andi %arg143, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        %163 = arith.trunci %162 {async_task_id = array<i32: 2>} : i64 to i1
        %164 = arith.andi %arg143, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        %165 = arith.trunci %164 {async_task_id = array<i32: 2>} : i64 to i1
        %166 = arith.andi %arg143, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        %167 = arith.trunci %166 {async_task_id = array<i32: 2>} : i64 to i1
        %168 = ttg.memdesc_index %arg72[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %169 = arith.extui %163 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %168, %169, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %170 = ttg.memdesc_index %arg73[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %171 = arith.extui %165 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %170, %171, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %172 = ttg.memdesc_index %arg74[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %173 = arith.extui %167 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %172, %173, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %174 = arith.andi %arg143, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        %175 = arith.trunci %174 {async_task_id = array<i32: 2>} : i64 to i1
        %176 = ttg.memdesc_index %arg121[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %177 = arith.xori %175, %true_8 : i1
        %178 = arith.extui %177 : i1 to i32
        ttng.wait_barrier %176, %178 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %179 = arith.andi %arg143, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        %180 = arith.trunci %179 {async_task_id = array<i32: 2>} : i64 to i1
        %181 = ttg.memdesc_index %arg123[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %182 = arith.xori %180, %true_8 : i1
        %183 = arith.extui %182 : i1 to i32
        ttng.wait_barrier %181, %183 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %184 = arith.cmpi sgt, %160, %c0_i32_10 : i32
        %185 = arith.andi %arg144, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        %186 = arith.trunci %185 {async_task_id = array<i32: 2>} : i64 to i1
        %187 = ttg.memdesc_index %arg75[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        %188 = ttg.memdesc_index %arg76[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %189 = arith.extui %186 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %188, %189, %184 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %190 = ttg.memdesc_trans %187 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>
        %191 = ttg.memdesc_index %arg77[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %192 = ttg.memdesc_index %arg78[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %193 = ttg.memdesc_index %arg79[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %194 = arith.andi %arg144, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        %195 = arith.trunci %194 {async_task_id = array<i32: 2>} : i64 to i1
        %196 = ttg.memdesc_index %arg80[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %197 = ttg.memdesc_index %arg127[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %198 = arith.xori %195, %true_8 : i1
        %199 = arith.extui %198 : i1 to i32
        ttng.wait_barrier %197, %199, %184 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %200 = arith.andi %arg144, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        %201 = arith.trunci %200 {async_task_id = array<i32: 2>} : i64 to i1
        %202 = ttg.memdesc_index %arg81[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %203 = arith.xori %201, %true_8 {async_task_id = array<i32: 2>} : i1
        %204 = arith.extui %203 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %202, %204, %184 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {direction = "backward", dstTask = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %205 = ttng.tc_gen5_mma %191, %190, %192[], %false, %184, %193[%true_8], %196[%true_8] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,1,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %206 = arith.andi %arg144, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        %207 = arith.trunci %206 {async_task_id = array<i32: 2>} : i64 to i1
        %208 = arith.andi %arg144, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        %209 = arith.trunci %208 {async_task_id = array<i32: 2>} : i64 to i1
        %210 = ttg.memdesc_index %arg82[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        %211 = ttg.memdesc_index %arg83[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %212 = arith.extui %209 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %211, %212, %184 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %213 = ttg.memdesc_trans %210 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>
        %214 = ttg.memdesc_index %arg84[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %215 = ttg.memdesc_index %arg68[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %216 = ttg.memdesc_index %arg85[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %217 = arith.andi %arg144, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        %218 = arith.trunci %217 {async_task_id = array<i32: 2>} : i64 to i1
        %219 = ttg.memdesc_index %arg86[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %220 = ttg.memdesc_index %arg131[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %221 = arith.xori %218, %true_8 : i1
        %222 = arith.extui %221 : i1 to i32
        ttng.wait_barrier %220, %222, %184 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %223 = arith.andi %arg144, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        %224 = arith.trunci %223 {async_task_id = array<i32: 2>} : i64 to i1
        %225 = ttg.memdesc_index %arg141[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %226 = arith.xori %224, %true_8 : i1
        %227 = arith.extui %226 : i1 to i32
        ttng.wait_barrier %225, %227, %184 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 1>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %228 = ttng.tc_gen5_mma %214, %213, %215[], %false, %184, %216[%true_8], %219[%true_8] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %229 = ttg.memdesc_index %arg87[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %230 = ttg.memdesc_index %arg88[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %231 = ttng.tmem_subslice %arg78 {N = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
        %232 = ttg.memdesc_reinterpret %231 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        %233 = ttg.memdesc_index %232[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %234 = ttg.memdesc_index %arg89[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %235 = ttg.memdesc_index %arg90[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %236 = arith.extui %207 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %235, %236, %184 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %237 = ttg.memdesc_index %arg81[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %238 = ttg.memdesc_index %arg128[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %239 = arith.extui %201 : i1 to i32
        ttng.wait_barrier %238, %239, %184 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %240 = ttng.tc_gen5_mma %233, %230, %229[], %false, %184, %234[%true_8], %237[%true_8] {async_task_id = array<i32: 2>, is_async, tmem.start = array<i32: 3>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %241 = arith.subi %160, %c1_i32_11 : i32
        %242:3 = scf.for %arg145 = %c0_i32_10 to %241 step %c1_i32_11 iter_args(%arg146 = %false, %arg147 = %arg144, %arg148 = %224) -> (i1, i64, i1)  : i32 {
          %251 = arith.addi %arg147, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %252 = arith.andi %251, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %253 = arith.trunci %252 {async_task_id = array<i32: 2>} : i64 to i1
          %254 = ttg.memdesc_index %arg75[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          %255 = ttg.memdesc_index %arg76[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %256 = arith.extui %253 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %255, %256, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %257 = ttg.memdesc_trans %254 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>
          %258 = arith.andi %arg147, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %259 = arith.trunci %258 {async_task_id = array<i32: 2>} : i64 to i1
          %260 = ttg.memdesc_index %arg77[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
          %261 = ttg.memdesc_index %arg78[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %262 = ttg.memdesc_index %arg79[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %263 = arith.andi %251, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %264 = arith.trunci %263 {async_task_id = array<i32: 2>} : i64 to i1
          %265 = ttg.memdesc_index %arg80[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %266 = ttg.memdesc_index %arg127[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %267 = arith.xori %264, %true_8 : i1
          %268 = arith.extui %267 : i1 to i32
          ttng.wait_barrier %266, %268, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %269 = arith.andi %251, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %270 = arith.trunci %269 {async_task_id = array<i32: 2>} : i64 to i1
          %271 = ttg.memdesc_index %arg81[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %272 = arith.xori %270, %true_8 {async_task_id = array<i32: 2>} : i1
          %273 = arith.extui %272 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %271, %273, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {direction = "backward", dstTask = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %274 = ttng.tc_gen5_mma %260, %257, %261[], %false, %true_8, %262[%true_8], %265[%true_8] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,1,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %275 = arith.andi %arg147, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %276 = arith.trunci %275 {async_task_id = array<i32: 2>} : i64 to i1
          %277 = ttg.memdesc_index %arg91[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %278 = ttg.memdesc_index %arg92[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %279 = ttng.tmem_subslice %arg68 {N = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
          %280 = ttg.memdesc_reinterpret %279 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
          %281 = ttg.memdesc_index %280[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %282 = ttg.memdesc_index %arg93[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %283 = ttg.memdesc_index %arg94[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %284 = arith.extui %259 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %283, %284, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %285 = ttg.memdesc_index %arg95[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %286 = ttg.memdesc_index %arg134[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %287 = arith.extui %276 : i1 to i32
          ttng.wait_barrier %286, %287 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %288 = ttng.tc_gen5_mma %281, %278, %277[], %arg146, %true_8, %282[%true_8], %285[%true_8] {async_task_id = array<i32: 2>, is_async, tmem.start = array<i32: 4>, tt.autows = "{\22stage\22: \221\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,10\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %289 = arith.andi %arg147, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %290 = arith.trunci %289 {async_task_id = array<i32: 2>} : i64 to i1
          %291 = ttg.memdesc_index %arg96[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %292 = ttg.memdesc_index %arg138[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %293 = arith.extui %290 : i1 to i32
          ttng.wait_barrier %292, %293 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %294 = ttg.memdesc_trans %291 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared4, #smem, mutable>
          %295 = ttg.memdesc_index %arg97[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %296 = ttng.tmem_subslice %arg68 {N = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %297 = ttg.memdesc_reinterpret %296 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x128xf32, #tmem3, #ttng.tensor_memory, mutable>
          %298 = ttg.memdesc_index %297[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf32, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem4, #ttng.tensor_memory, mutable>
          %299 = ttg.memdesc_index %arg98[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %300 = ttg.memdesc_index %arg69[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %301 = ttg.memdesc_index %arg131[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %302 = arith.extui %arg148 : i1 to i32
          ttng.wait_barrier %301, %302 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %303 = ttng.tc_gen5_mma %294, %295, %298[], %false, %true_8, %299[%true_8], %300[%true_8] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x256xf16, #shared4, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem4, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %304 = arith.andi %251, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %305 = arith.trunci %304 {async_task_id = array<i32: 2>} : i64 to i1
          %306 = arith.andi %251, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %307 = arith.trunci %306 {async_task_id = array<i32: 2>} : i64 to i1
          %308 = ttg.memdesc_index %arg82[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          %309 = ttg.memdesc_index %arg83[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %310 = arith.extui %307 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %309, %310, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %311 = ttg.memdesc_trans %308 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>
          %312 = ttg.memdesc_index %arg84[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
          %313 = ttg.memdesc_index %arg68[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %314 = ttg.memdesc_index %arg85[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %315 = arith.andi %251, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %316 = arith.trunci %315 {async_task_id = array<i32: 2>} : i64 to i1
          %317 = ttg.memdesc_index %arg86[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %318 = ttg.memdesc_index %arg131[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %319 = arith.xori %316, %true_8 : i1
          %320 = arith.extui %319 : i1 to i32
          ttng.wait_barrier %318, %320, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %321 = arith.andi %251, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %322 = arith.trunci %321 {async_task_id = array<i32: 2>} : i64 to i1
          %323 = ttg.memdesc_index %arg141[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %324 = arith.xori %322, %true_8 : i1
          %325 = arith.extui %324 : i1 to i32
          ttng.wait_barrier %323, %325, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 1>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %326 = ttng.tc_gen5_mma %312, %311, %313[], %false, %true_8, %314[%true_8], %317[%true_8] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %327 = ttg.memdesc_index %arg87[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %328 = ttg.memdesc_index %arg88[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %329 = ttng.tmem_subslice %arg78 {N = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
          %330 = ttg.memdesc_reinterpret %329 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
          %331 = ttg.memdesc_index %330[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %332 = ttg.memdesc_index %arg89[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %333 = ttg.memdesc_index %arg90[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %334 = arith.extui %305 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %333, %334, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %335 = ttg.memdesc_index %arg81[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %336 = ttg.memdesc_index %arg128[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %337 = arith.extui %270 : i1 to i32
          ttng.wait_barrier %336, %337, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %338 = ttng.tc_gen5_mma %331, %328, %327[], %true_8, %true_8, %332[%true_8], %335[%true_8] {async_task_id = array<i32: 2>, is_async, tmem.start = array<i32: 3>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          scf.yield %true_8, %251, %322 : i1, i64, i1
        } {async_task_id = array<i32: 2>, tt.warp_specialize}
        %243 = arith.cmpi sgt, %160, %c0_i32_10 : i32
        %244:3 = scf.if %243 -> (i1, i64, i1) {
          %251 = arith.addi %242#1, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %252 = arith.andi %242#1, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %253 = arith.trunci %252 {async_task_id = array<i32: 2>} : i64 to i1
          %254 = arith.andi %242#1, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %255 = arith.trunci %254 {async_task_id = array<i32: 2>} : i64 to i1
          %256 = ttg.memdesc_index %arg91[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %257 = ttg.memdesc_index %arg92[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %258 = ttng.tmem_subslice %arg68 {N = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
          %259 = ttg.memdesc_reinterpret %258 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
          %260 = ttg.memdesc_index %259[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %261 = ttg.memdesc_index %arg93[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %262 = ttg.memdesc_index %arg94[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %263 = arith.extui %253 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %262, %263, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %264 = ttg.memdesc_index %arg95[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %265 = ttg.memdesc_index %arg134[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %266 = arith.extui %255 : i1 to i32
          ttng.wait_barrier %265, %266 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %267 = ttng.tc_gen5_mma %260, %257, %256[], %242#0, %true_8, %261[%true_8], %264[%true_8] {async_task_id = array<i32: 2>, is_async, tmem.start = array<i32: 4>, tt.autows = "{\22stage\22: \221\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,10\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %268 = arith.andi %242#1, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
          %269 = arith.trunci %268 {async_task_id = array<i32: 2>} : i64 to i1
          %270 = ttg.memdesc_index %arg96[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %271 = ttg.memdesc_index %arg138[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %272 = arith.extui %269 : i1 to i32
          ttng.wait_barrier %271, %272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %273 = ttg.memdesc_trans %270 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared4, #smem, mutable>
          %274 = ttg.memdesc_index %arg97[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %275 = ttng.tmem_subslice %arg68 {N = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %276 = ttg.memdesc_reinterpret %275 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x128xf32, #tmem3, #ttng.tensor_memory, mutable>
          %277 = ttg.memdesc_index %276[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf32, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem4, #ttng.tensor_memory, mutable>
          %278 = ttg.memdesc_index %arg98[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %279 = ttg.memdesc_index %arg69[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %280 = ttg.memdesc_index %arg131[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %281 = arith.extui %242#2 : i1 to i32
          ttng.wait_barrier %280, %281 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %282 = ttng.tc_gen5_mma %273, %274, %277[], %false, %true_8, %278[%true_8], %279[%true_8] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x256xf16, #shared4, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem4, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          scf.yield %true_8, %251, %158 : i1, i64, i1
        } else {
          scf.yield %242#0, %242#1, %242#2 : i1, i64, i1
        }
        %245 = ttg.memdesc_index %arg99[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_commit %245 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %246 = ttg.memdesc_index %arg100[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_commit %246 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %247 = ttg.memdesc_index %arg101[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_commit %247 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %248 = ttg.memdesc_index %arg102[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_commit %248 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %249 = ttg.memdesc_index %arg103[%c0_i32_10] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_commit %249 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %250 = arith.addi %arg143, %c1_i64_6 {async_task_id = array<i32: 2>} : i64
        scf.yield {async_task_id = array<i32: 2>} %250, %244#1 : i64, i64
      } {async_task_id = array<i32: 2>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition2(%arg63: i32, %arg64: i32, %arg65: i32, %arg66: i32, %arg67: i32, %arg68: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg69: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg70: !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, %arg71: !tt.tensordesc<128x16xf32, #shared1>, %arg72: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg73: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg74: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg75: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg76: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg77: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg78: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg79: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg80: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg81: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg82: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg83: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg84: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg85: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg86: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg87: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg88: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg89: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg90: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg91: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg92: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg93: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg94: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg95: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg96: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg97: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg98: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg99: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg100: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg101: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg102: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg103: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg104: !tt.tensordesc<128x128xf16, #shared>, %arg105: !tt.tensordesc<256x64xf16, #shared>, %arg106: !tt.tensordesc<128x128xf16, #shared>, %arg107: !tt.tensordesc<64x128xf16, #shared>, %arg108: !tt.tensordesc<128x64xf16, #shared>, %arg109: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg110: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %arg111: !tt.tensordesc<128xf32, #shared2>, %arg112: !tt.tensordesc<128x64xf16, #shared>, %arg113: !tt.tensordesc<64x128xf16, #shared>, %arg114: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg115: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %arg116: !tt.tensordesc<128xf32, #shared2>, %arg117: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg118: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg119: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg120: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg121: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg122: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg126: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg127: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg128: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg129: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg130: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg131: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg132: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg133: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg134: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg135: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg136: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg137: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg138: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg139: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg140: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg141: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) num_warps(8) {
      %true_6 = arith.constant true
      %c1_i64_7 = arith.constant {async_task_id = array<i32: 3>} 1 : i64
      %c0_i64_8 = arith.constant {async_task_id = array<i32: 3>} 0 : i64
      %c64_i32_9 = arith.constant {async_task_id = array<i32: 3>} 64 : i32
      %c2_i32 = arith.constant {async_task_id = array<i32: 3>} 2 : i32
      %c128_i32_10 = arith.constant {async_task_id = array<i32: 3>} 128 : i32
      %c0_i32_11 = arith.constant {async_task_id = array<i32: 3>} 0 : i32
      %c1_i32_12 = arith.constant {async_task_id = array<i32: 3>} 1 : i32
      %158 = tt.get_program_id z {async_task_id = array<i32: 3>} : i32
      %159 = tt.get_program_id x {async_task_id = array<i32: 3>} : i32
      %160 = tt.get_num_programs y {async_task_id = array<i32: 3>} : i32
      %161 = arith.muli %158, %arg64 {async_task_id = array<i32: 3>} : i32
      %162 = arith.extsi %161 {async_task_id = array<i32: 3>} : i32 to i64
      %163 = arith.extsi %arg63 {async_task_id = array<i32: 3>} : i32 to i64
      %164 = arith.muli %159, %c128_i32_10 {async_task_id = array<i32: 3>} : i32
      %165 = arith.remsi %159, %c2_i32 {async_task_id = array<i32: 3>} : i32
      %166 = arith.extsi %164 {async_task_id = array<i32: 3>} : i32 to i64
      %167 = arith.muli %165, %c128_i32_10 {async_task_id = array<i32: 3>} : i32
      %168 = arith.subi %164, %167 {async_task_id = array<i32: 3>} : i32
      %169 = arith.extsi %168 {async_task_id = array<i32: 3>} : i32 to i64
      %170 = arith.divsi %arg64, %c128_i32_10 {async_task_id = array<i32: 3>} : i32
      %171 = nvg.cluster_id {async_task_id = array<i32: 3>}
      %172 = arith.remsi %171, %c2_i32 {async_task_id = array<i32: 3>} : i32
      %173 = arith.muli %172, %c64_i32_9 {async_task_id = array<i32: 3>} : i32
      %174:2 = scf.for %arg142 = %c0_i32_11 to %160 step %c1_i32_12 iter_args(%arg143 = %c0_i64_8, %arg144 = %c0_i64_8) -> (i64, i64)  : i32 {
        %175 = arith.extui %arg142 {async_task_id = array<i32: 3>} : i32 to i64
        %176 = arith.andi %175, %c1_i64_7 {async_task_id = array<i32: 3>} : i64
        %177 = arith.trunci %176 {async_task_id = array<i32: 3>} : i64 to i1
        %178 = ttg.memdesc_index %arg119[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %179 = arith.xori %177, %true_6 : i1
        %180 = arith.extui %179 : i1 to i32
        ttng.wait_barrier %178, %180 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %181 = arith.remsi %158, %arg65 {async_task_id = array<i32: 3>} : i32
        %182 = arith.muli %arg66, %181 {async_task_id = array<i32: 3>} : i32
        %183 = arith.divsi %158, %arg65 {async_task_id = array<i32: 3>} : i32
        %184 = arith.muli %arg67, %183 {async_task_id = array<i32: 3>} : i32
        %185 = arith.addi %182, %184 {async_task_id = array<i32: 3>} : i32
        %186 = arith.extsi %185 {async_task_id = array<i32: 3>} : i32 to i64
        %187 = arith.divsi %186, %163 {async_task_id = array<i32: 3>} : i64
        %188 = arith.addi %187, %166 {async_task_id = array<i32: 3>} : i64
        %189 = arith.trunci %188 {async_task_id = array<i32: 3>} : i64 to i32
        %190 = arith.andi %arg143, %c1_i64_7 {async_task_id = array<i32: 3>} : i64
        %191 = arith.trunci %190 {async_task_id = array<i32: 3>} : i64 to i1
        %192 = ttg.memdesc_index %arg103[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %193 = arith.xori %191, %true_6 {async_task_id = array<i32: 3>} : i1
        %194 = arith.extui %193 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %192, %194 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %195 = ttg.memdesc_index %arg72[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.barrier_expect %195, 32768 {async_task_id = array<i32: 3>}, %true_6 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %196 = ttg.memdesc_index %arg77[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %arg104[%189, %c0_i32_11] %196, %195, %true_6 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %197 = arith.addi %187, %169 {async_task_id = array<i32: 3>} : i64
        %198 = arith.trunci %197 {async_task_id = array<i32: 3>} : i64 to i32
        %199 = arith.andi %arg143, %c1_i64_7 {async_task_id = array<i32: 3>} : i64
        %200 = arith.trunci %199 {async_task_id = array<i32: 3>} : i64 to i1
        %201 = ttg.memdesc_index %arg102[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %202 = arith.xori %200, %true_6 {async_task_id = array<i32: 3>} : i1
        %203 = arith.extui %202 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %201, %203 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %204 = ttg.memdesc_index %arg73[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.barrier_expect %204, 32768 {async_task_id = array<i32: 3>}, %true_6 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %205 = ttg.memdesc_index %arg97[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %arg105[%198, %173] %205, %204, %true_6 {async_task_id = array<i32: 3>} : !tt.tensordesc<256x64xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        %206 = arith.andi %arg143, %c1_i64_7 {async_task_id = array<i32: 3>} : i64
        %207 = arith.trunci %206 {async_task_id = array<i32: 3>} : i64 to i1
        %208 = ttg.memdesc_index %arg101[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %209 = arith.xori %207, %true_6 {async_task_id = array<i32: 3>} : i1
        %210 = arith.extui %209 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %208, %210 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %211 = ttg.memdesc_index %arg74[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.barrier_expect %211, 32768 {async_task_id = array<i32: 3>}, %true_6 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %212 = ttg.memdesc_index %arg84[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %arg106[%189, %c0_i32_11] %212, %211, %true_6 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %213:2 = scf.for %arg145 = %c0_i32_11 to %170 step %c1_i32_12 iter_args(%arg146 = %c0_i32_11, %arg147 = %arg144) -> (i32, i64)  : i32 {
          %215 = arith.extsi %arg146 {async_task_id = array<i32: 3>} : i32 to i64
          %216 = arith.addi %187, %215 {async_task_id = array<i32: 3>} : i64
          %217 = arith.trunci %216 {async_task_id = array<i32: 3>} : i64 to i32
          %218 = arith.addi %217, %173 {async_task_id = array<i32: 3>} : i32
          %219 = arith.andi %arg147, %c1_i64_7 {async_task_id = array<i32: 3>} : i64
          %220 = arith.trunci %219 {async_task_id = array<i32: 3>} : i64 to i1
          %221 = ttg.memdesc_index %arg79[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %222 = arith.xori %220, %true_6 {async_task_id = array<i32: 3>} : i1
          %223 = arith.extui %222 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %221, %223 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %224 = ttg.memdesc_index %arg76[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.barrier_expect %224, 16384 {async_task_id = array<i32: 3>}, %true_6 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %225 = ttg.memdesc_index %arg75[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg107[%218, %c0_i32_11] %225, %224, %true_6 {async_task_id = array<i32: 3>} : !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          %226 = arith.andi %arg147, %c1_i64_7 {async_task_id = array<i32: 3>} : i64
          %227 = arith.trunci %226 {async_task_id = array<i32: 3>} : i64 to i1
          %228 = ttg.memdesc_index %arg93[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %229 = arith.xori %227, %true_6 {async_task_id = array<i32: 3>} : i1
          %230 = arith.extui %229 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %228, %230 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %231 = ttg.memdesc_index %arg94[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.barrier_expect %231, 16384 {async_task_id = array<i32: 3>}, %true_6 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %232 = ttg.memdesc_index %arg92[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg108[%217, %173] %232, %231, %true_6 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %233 = arith.addi %162, %215 {async_task_id = array<i32: 3>} : i64
          %234 = arith.trunci %233 {async_task_id = array<i32: 3>} : i64 to i32
          %235 = arith.andi %arg147, %c1_i64_7 {async_task_id = array<i32: 3>} : i64
          %236 = arith.trunci %235 {async_task_id = array<i32: 3>} : i64 to i1
          %237 = ttg.memdesc_index %arg125[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %238 = arith.xori %236, %true_6 : i1
          %239 = arith.extui %238 : i1 to i32
          ttng.wait_barrier %237, %239 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %240 = ttg.memdesc_index %arg109[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.barrier_expect %240, 512 {async_task_id = array<i32: 3>}, %true_6 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %241 = ttg.memdesc_index %arg110[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg111[%234] %241, %240, %true_6 {async_task_id = array<i32: 3>} : !tt.tensordesc<128xf32, #shared2>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
          %242 = arith.andi %arg147, %c1_i64_7 {async_task_id = array<i32: 3>} : i64
          %243 = arith.trunci %242 {async_task_id = array<i32: 3>} : i64 to i1
          %244 = ttg.memdesc_index %arg89[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %245 = arith.xori %243, %true_6 {async_task_id = array<i32: 3>} : i1
          %246 = arith.extui %245 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %244, %246 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %247 = ttg.memdesc_index %arg90[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.barrier_expect %247, 16384 {async_task_id = array<i32: 3>}, %true_6 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %248 = ttg.memdesc_index %arg88[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg112[%217, %173] %248, %247, %true_6 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %249 = arith.andi %arg147, %c1_i64_7 {async_task_id = array<i32: 3>} : i64
          %250 = arith.trunci %249 {async_task_id = array<i32: 3>} : i64 to i1
          %251 = ttg.memdesc_index %arg85[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %252 = arith.xori %250, %true_6 {async_task_id = array<i32: 3>} : i1
          %253 = arith.extui %252 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %251, %253 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %254 = ttg.memdesc_index %arg83[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.barrier_expect %254, 16384 {async_task_id = array<i32: 3>}, %true_6 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %255 = ttg.memdesc_index %arg82[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg113[%218, %c0_i32_11] %255, %254, %true_6 {async_task_id = array<i32: 3>} : !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          %256 = arith.andi %arg147, %c1_i64_7 {async_task_id = array<i32: 3>} : i64
          %257 = arith.trunci %256 {async_task_id = array<i32: 3>} : i64 to i1
          %258 = ttg.memdesc_index %arg133[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %259 = arith.xori %257, %true_6 : i1
          %260 = arith.extui %259 : i1 to i32
          ttng.wait_barrier %258, %260 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %261 = ttg.memdesc_index %arg114[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.barrier_expect %261, 512 {async_task_id = array<i32: 3>}, %true_6 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %262 = ttg.memdesc_index %arg115[%c0_i32_11] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg116[%234] %262, %261, %true_6 {async_task_id = array<i32: 3>} : !tt.tensordesc<128xf32, #shared2>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
          %263 = arith.addi %arg146, %c128_i32_10 {async_task_id = array<i32: 3>} : i32
          %264 = arith.addi %arg147, %c1_i64_7 {async_task_id = array<i32: 3>} : i64
          scf.yield {async_task_id = array<i32: 3>} %263, %264 : i32, i64
        } {async_task_id = array<i32: 3>, tt.warp_specialize}
        %214 = arith.addi %arg143, %c1_i64_7 {async_task_id = array<i32: 3>} : i64
        scf.yield {async_task_id = array<i32: 3>} %214, %213#1 : i64, i64
      } {async_task_id = array<i32: 3>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition3(%arg63: i32, %arg64: i32, %arg65: i32, %arg66: i32, %arg67: i32, %arg68: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg69: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg70: !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, %arg71: !tt.tensordesc<128x16xf32, #shared1>, %arg72: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg73: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg74: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg75: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg76: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg77: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg78: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg79: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg80: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg81: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg82: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg83: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg84: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg85: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg86: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg87: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg88: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg89: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg90: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg91: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg92: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg93: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg94: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg95: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg96: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg97: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg98: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg99: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg100: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg101: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg102: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg103: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg104: !tt.tensordesc<128x128xf16, #shared>, %arg105: !tt.tensordesc<256x64xf16, #shared>, %arg106: !tt.tensordesc<128x128xf16, #shared>, %arg107: !tt.tensordesc<64x128xf16, #shared>, %arg108: !tt.tensordesc<128x64xf16, #shared>, %arg109: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg110: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %arg111: !tt.tensordesc<128xf32, #shared2>, %arg112: !tt.tensordesc<128x64xf16, #shared>, %arg113: !tt.tensordesc<64x128xf16, #shared>, %arg114: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg115: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %arg116: !tt.tensordesc<128xf32, #shared2>, %arg117: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg118: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg119: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg120: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg121: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg122: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg126: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg127: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg128: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg129: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg130: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg131: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg132: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg133: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg134: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg135: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg136: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg137: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg138: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg139: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg140: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg141: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) num_warps(8) {
      %c1_i64_6 = arith.constant {async_task_id = array<i32: 4>} 1 : i64
      %c0_i64_7 = arith.constant {async_task_id = array<i32: 4>} 0 : i64
      %c128_i32_8 = arith.constant {async_task_id = array<i32: 4>} 128 : i32
      %c0_i32_9 = arith.constant {async_task_id = array<i32: 4>} 0 : i32
      %c1_i32_10 = arith.constant {async_task_id = array<i32: 4>} 1 : i32
      %158 = tt.get_num_programs y {async_task_id = array<i32: 4>} : i32
      %159 = arith.divsi %arg64, %c128_i32_8 {async_task_id = array<i32: 4>} : i32
      %160 = scf.for %arg142 = %c0_i32_9 to %158 step %c1_i32_10 iter_args(%arg143 = %c0_i64_7) -> (i64)  : i32 {
        %161 = scf.for %arg144 = %c0_i32_9 to %159 step %c1_i32_10 iter_args(%arg145 = %arg143) -> (i64)  : i32 {
          %162 = arith.andi %arg145, %c1_i64_6 {async_task_id = array<i32: 4>} : i64
          %163 = arith.trunci %162 {async_task_id = array<i32: 4>} : i64 to i1
          %164 = ttg.memdesc_index %arg117[%c0_i32_9] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %165 = ttg.memdesc_index %arg136[%c0_i32_9] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %166 = arith.extui %163 : i1 to i32
          ttng.wait_barrier %165, %166 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 3>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.two_cta_peer_relay %164 {async_task_id = array<i32: 4>} : <128x64xf16, #shared, #smem, mutable>
          %167 = ttg.memdesc_index %arg137[%c0_i32_9] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %167, 1 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 3>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %168 = arith.addi %arg145, %c1_i64_6 {async_task_id = array<i32: 4>} : i64
          scf.yield {async_task_id = array<i32: 4>} %168 : i64
        } {async_task_id = array<i32: 4>, tt.warp_specialize}
        scf.yield {async_task_id = array<i32: 4>} %161 : i64
      } {async_task_id = array<i32: 4>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    } : (i32, i32, i32, i32, i32, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !tt.tensordesc<128x128xf16, #shared>, !tt.tensordesc<256x64xf16, #shared>, !tt.tensordesc<128x128xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>, !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, !tt.tensordesc<128xf32, #shared2>, !tt.tensordesc<128x64xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, !tt.tensordesc<128xf32, #shared2>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) -> ()
    %110 = ttg.memdesc_index %4[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %110 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %111 = ttg.memdesc_index %50[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %111 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %112 = ttg.memdesc_index %46[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %112 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %113 = ttg.memdesc_index %42[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %113 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %114 = ttg.memdesc_index %34[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %114 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %115 = ttg.memdesc_index %32[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %115 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %116 = ttg.memdesc_index %24[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %116 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %117 = ttg.memdesc_index %18[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %117 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %118 = ttg.memdesc_index %16[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %118 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %119 = ttg.memdesc_index %14[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %119 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %120 = ttg.memdesc_index %12[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %120 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %121 = ttg.memdesc_index %20[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %121 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %122 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %122 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %123 = ttg.memdesc_index %28[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %123 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %124 = ttg.memdesc_index %30[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %124 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %125 = ttg.memdesc_index %8[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %125 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %126 = ttg.memdesc_index %6[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %126 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %127 = ttg.memdesc_index %36[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %127 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %128 = ttg.memdesc_index %38[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %128 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %129 = ttg.memdesc_index %40[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %129 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %130 = ttg.memdesc_index %44[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %130 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %131 = ttg.memdesc_index %48[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %131 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %132 = ttg.memdesc_index %26[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %132 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %133 = ttg.memdesc_index %10[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %133 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %134 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %134 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %135 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %135 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %136 = ttg.memdesc_index %52[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %136 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %137 = ttg.memdesc_index %53[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %137 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %138 = ttg.memdesc_index %56[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %138 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %139 = ttg.memdesc_index %57[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %139 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %140 = ttg.memdesc_index %60[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %140 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %141 = ttg.memdesc_index %61[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %141 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %142 = ttg.memdesc_index %64[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %142 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %143 = ttg.memdesc_index %65[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %143 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %144 = ttg.memdesc_index %68[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %144 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %145 = ttg.memdesc_index %69[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %145 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %146 = ttg.memdesc_index %72[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %146 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %147 = ttg.memdesc_index %73[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %147 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %148 = ttg.memdesc_index %76[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %148 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %149 = ttg.memdesc_index %77[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %149 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %150 = ttg.memdesc_index %80[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %150 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %151 = ttg.memdesc_index %81[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %151 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %152 = ttg.memdesc_index %84[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %152 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %153 = ttg.memdesc_index %85[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %153 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %154 = ttg.memdesc_index %88[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %154 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %155 = ttg.memdesc_index %89[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %155 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %156 = ttg.memdesc_index %92[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %156 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %157 = ttg.memdesc_index %93[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %157 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    tt.return
  }
}

