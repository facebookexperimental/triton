// RUN: triton-opt %s --tritongpu-allocate-warp-groups | FileCheck %s
// CHECK: "ttg.single-warp-specialize" = true
// CHECK: ttg.warp_specialize
// CHECK-SAME: actualRegisters = array<i32:

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 8]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 32]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [[64, 0], [1, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 64]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [[64, 0], [1, 0]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 1, 0]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0]], block = []}>
#linear9 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1, twoCTAs = true>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2, twoCTAs = true>
#tmem3 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 32, colStride = 1, twoCTAs = true>
#tmem4 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 16, colStride = 1, twoCTAs = true>
#tmem5 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true>
#tmem6 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true, ctaMode = twocta_rhs>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @_attn_bwd_persist(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<64x128xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf16, #shared>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<256x64xf16, #shared>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.tensordesc<128x128xf16, #shared>, %arg21: i32, %arg22: i32, %arg23: i64, %arg24: i64, %arg25: f32, %arg26: !tt.tensordesc<128x64xf16, #shared>, %arg27: i32, %arg28: i32, %arg29: i64, %arg30: i64, %arg31: !tt.tensordesc<64x128xf16, #shared>, %arg32: i32, %arg33: i32, %arg34: i64, %arg35: i64, %arg36: !tt.tensordesc<128x16xf32, #shared1>, %arg37: i32, %arg38: i32, %arg39: i64, %arg40: i64, %arg41: !tt.tensordesc<128x16xf16, #shared2>, %arg42: i32, %arg43: i32, %arg44: i64, %arg45: i64, %arg46: !tt.tensordesc<128x16xf16, #shared2>, %arg47: i32, %arg48: i32, %arg49: i64, %arg50: i64, %arg51: !tt.tensordesc<128xf32, #shared3>, %arg52: i32, %arg53: i64, %arg54: !tt.tensordesc<128xf32, #shared3>, %arg55: i32, %arg56: i64, %arg57: i32 {tt.divisibility = 16 : i32}, %arg58: i32 {tt.divisibility = 16 : i32}, %arg59: i32 {tt.divisibility = 16 : i32}, %arg60: i32, %arg61: i32, %arg62: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c112_i32 = arith.constant 112 : i32
    %c96_i32 = arith.constant 96 : i32
    %c80_i32 = arith.constant 80 : i32
    %c64_i32 = arith.constant 64 : i32
    %c48_i32 = arith.constant 48 : i32
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c15_i32 = arith.constant 15 : i32
    %c256_i32 = arith.constant 256 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %2 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %3 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %4 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %5 = ttg.memdesc_index %4[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %6 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %7 = ttg.memdesc_index %6[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %8 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %9 = ttg.memdesc_index %8[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %10 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %11 = ttg.memdesc_index %10[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %11, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %12 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %13 = ttg.memdesc_index %12[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %13, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %14 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %15 = ttg.memdesc_index %14[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %15, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %16 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %17 = ttg.memdesc_index %16[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %17, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %18 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %19 = ttg.memdesc_index %18[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %19, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %20 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %21 = ttg.memdesc_index %20[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %21, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %22 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %23 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %23, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %24 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %25 = ttg.memdesc_index %24[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %25, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %26 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %27 = ttg.memdesc_index %26[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %27, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %28 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %29 = ttg.memdesc_index %28[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %29, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %30 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %31 = ttg.memdesc_index %30[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %31, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %32 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %33 = ttg.memdesc_index %32[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %33, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %34 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %35 = ttg.memdesc_index %34[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %35, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %36 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %37 = ttg.memdesc_index %36[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %37, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %38 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %39 = ttg.memdesc_index %38[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %39, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %40 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %41 = ttg.memdesc_index %40[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %41, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %42 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %43 = ttg.memdesc_index %42[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %43, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %44 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %45 = ttg.memdesc_index %44[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %45, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %46 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %47 = ttg.memdesc_index %46[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %47, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %48 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %49 = ttg.memdesc_index %48[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %49, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %50 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %51 = ttg.memdesc_index %50[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %51, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %52 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %53 = ttg.memdesc_index %52[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %53, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %54 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %55 = ttg.memdesc_index %54[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %55, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %56 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %57 = ttg.memdesc_index %56[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %57, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %58 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %59 = ttg.memdesc_index %58[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %59, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %60 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %61 = ttg.memdesc_index %60[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %61, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %62 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %63 = ttg.memdesc_index %62[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %63, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %64 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %65 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %66 = ttg.memdesc_index %64[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %66, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %67 = ttg.memdesc_index %65[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %67, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %68 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %69 = ttg.memdesc_index %68[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %69, 2 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %70 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %71 = ttg.memdesc_index %70[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %71, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %72 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %73 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 12 : i32} : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %74 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %result = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 10 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %75 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %76 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 15 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %77 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 16 : i32} : () -> !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>
    %result_1 = ttng.tmem_alloc {allocation.shareGroup = 1 : i32, buffer.copy = 1 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %78 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 17 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %79 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %result_2 = ttng.tmem_alloc {allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %80 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 19 : i32} : () -> !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>
    %81 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 20 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %82 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %83 = ttg.local_alloc {allocation.shareGroup = 3 : i32, buffer.copy = 1 : i32, buffer.id = 22 : i32, buffer.tmaStaging = 2 : i32} : () -> !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>
    %84 = ttg.local_alloc {allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 26 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>
    %85 = ttg.local_alloc {allocation.shareGroup = 2 : i32, buffer.copy = 1 : i32, buffer.id = 34 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>
    %86 = ttg.local_alloc : () -> !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>
    %87 = ttg.memdesc_index %86[%c0_i32] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %87, 2 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %88 = ttg.memdesc_index %86[%c1_i32] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %88, 2 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %89 = ttg.memdesc_index %86[%c2_i32] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %89, 2 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %90 = ttg.memdesc_index %86[%c3_i32] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %90, 2 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %91 = ttg.memdesc_index %86[%c4_i32] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %91, 2 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %92 = ttg.local_alloc : () -> !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>
    %93 = ttg.memdesc_index %92[%c0_i32] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %93, 2 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %94 = ttg.memdesc_index %92[%c1_i32] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %94, 2 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %95 = ttg.memdesc_index %92[%c2_i32] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %95, 2 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %96 = ttg.memdesc_index %92[%c3_i32] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %96, 2 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %97 = ttg.memdesc_index %92[%c4_i32] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %97, 2 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.warp_specialize(%arg62, %arg60, %arg61, %arg59, %arg58, %arg57, %result_2, %0, %83, %arg36, %46, %42, %38, %75, %30, %72, %result_1, %28, %20, %14, %79, %12, %74, %10, %8, %result, %78, %16, %18, %result_0, %76, %24, %26, %4, %82, %73, %2, %32, %34, %36, %40, %44, %arg10, %arg15, %arg20, %arg5, %arg0, %22, %77, %arg51, %arg26, %arg31, %6, %80, %arg54, %48, %50, %52, %54, %56, %58, %60, %62, %64, %65, %68, %70, %86, %92) attributes {requestedRegisters = array<i32: 192, 24, 24, 24>, ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"]}
    default {
      %98 = arith.addi %arg62, %c127_i32 {async_task_id = array<i32: 0>} : i32
      %99 = arith.divsi %98, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %100 = tt.get_program_id x {async_task_id = array<i32: 0>} : i32
      %101 = arith.remsi %100, %c2_i32 {async_task_id = array<i32: 0>} : i32
      %102 = arith.divsi %100, %c2_i32 {async_task_id = array<i32: 0>} : i32
      %103 = tt.get_num_programs x {async_task_id = array<i32: 0>} : i32
      %104 = arith.divsi %103, %c2_i32 {async_task_id = array<i32: 0>} : i32
      %105 = arith.divsi %99, %c2_i32 {async_task_id = array<i32: 0>} : i32
      %106 = arith.muli %105, %arg60 {async_task_id = array<i32: 0>} : i32
      %107 = arith.muli %106, %arg61 {async_task_id = array<i32: 0>} : i32
      %108 = arith.divsi %107, %104 {async_task_id = array<i32: 0>} : i32
      %109 = arith.remsi %107, %104 {async_task_id = array<i32: 0>} : i32
      %110 = arith.cmpi slt, %102, %109 {async_task_id = array<i32: 0>} : i32
      %111 = scf.if %110 -> (i32) {
        %116 = arith.addi %108, %c1_i32 {async_task_id = array<i32: 0>} : i32
        scf.yield {async_task_id = array<i32: 0>} %116 : i32
      } else {
        scf.yield {async_task_id = array<i32: 0>} %108 : i32
      } {async_task_id = array<i32: 0>}
      %112 = arith.extsi %arg59 {async_task_id = array<i32: 0>} : i32 to i64
      %113 = arith.divsi %arg62, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %114 = tt.splat %arg25 : f32 -> tensor<128x16xf32, #linear>
      %115:3 = scf.for %arg63 = %c0_i32 to %111 step %c1_i32 iter_args(%arg64 = %102, %arg65 = %c0_i64, %arg66 = %c0_i64) -> (i32, i64, i64)  : i32 {
        %116 = arith.remsi %arg64, %105 {async_task_id = array<i32: 0>} : i32
        %117 = arith.divsi %arg64, %105 {async_task_id = array<i32: 0>} : i32
        %118 = arith.muli %116, %c2_i32 {async_task_id = array<i32: 0>} : i32
        %119 = arith.addi %118, %101 {async_task_id = array<i32: 0>} : i32
        %120 = arith.remsi %117, %arg61 {async_task_id = array<i32: 0>} : i32
        %121 = arith.muli %arg58, %120 {async_task_id = array<i32: 0>} : i32
        %122 = arith.divsi %117, %arg61 {async_task_id = array<i32: 0>} : i32
        %123 = arith.muli %arg57, %122 {async_task_id = array<i32: 0>} : i32
        %124 = arith.addi %121, %123 {async_task_id = array<i32: 0>} : i32
        %125 = arith.extsi %124 {async_task_id = array<i32: 0>} : i32 to i64
        %126 = arith.divsi %125, %112 {async_task_id = array<i32: 0>} : i64
        %127 = arith.muli %119, %c128_i32 {async_task_id = array<i32: 0>} : i32
        %128 = arith.extsi %127 {async_task_id = array<i32: 0>} : i32 to i64
        %129 = arith.addi %126, %128 {async_task_id = array<i32: 0>} : i64
        %130 = arith.trunci %129 {async_task_id = array<i32: 0>} : i64 to i32
        %131 = arith.andi %arg65, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %132 = arith.trunci %131 {async_task_id = array<i32: 0>} : i64 to i1
        %133 = scf.for %arg67 = %c0_i32 to %113 step %c1_i32 iter_args(%arg68 = %arg66) -> (i64)  : i32 {
          %213 = arith.andi %arg68, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %214 = arith.trunci %213 {async_task_id = array<i32: 0>} : i64 to i1
          %215 = ttg.memdesc_index %77[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128xf32, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
          %216 = ttg.memdesc_index %22[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %217 = arith.extui %214 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %216, %217, %true {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %218 = ttg.local_load %215 : !ttg.memdesc<128xf32, #shared3, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear1}>>
          %219 = ttg.memdesc_index %52[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %219, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %220 = tt.expand_dims %218 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x128xf32, #linear1>
          %221 = tt.broadcast %220 {async_task_id = array<i32: 0>} : tensor<1x128xf32, #linear1> -> tensor<128x128xf32, #linear1>
          %222 = ttg.memdesc_index %result_1[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %223 = ttg.memdesc_index %20[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %223, %217 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %result_19 = ttng.tmem_load %222 {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
          %224 = ttg.memdesc_index %54[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %224, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %225 = arith.subf %result_19, %221 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear1>
          %226 = math.exp2 %225 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear1>
          %227 = arith.truncf %226 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear1> to tensor<128x128xf16, #linear1>
          %228 = ttng.tmem_subslice %result_1 {N = 0 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
          %229 = ttg.memdesc_reinterpret %228 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
          %230 = ttg.memdesc_index %229[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %231 = arith.extui %214 : i1 to i32
          ttng.wait_barrier %224, %231 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {dstTask = 0 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.tmem_store %227, %230, %true {async_task_id = array<i32: 0>} : tensor<128x128xf16, #linear1> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %232 = ttg.memdesc_index %56[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %232, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %233 = ttg.memdesc_index %80[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128xf32, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
          %234 = ttg.memdesc_index %6[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %234, %217, %true {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %235 = ttg.local_load %233 : !ttg.memdesc<128xf32, #shared3, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear1}>>
          %236 = ttg.memdesc_index %60[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %236, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %237 = tt.expand_dims %235 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x128xf32, #linear1>
          %238 = tt.broadcast %237 {async_task_id = array<i32: 0>} : tensor<1x128xf32, #linear1> -> tensor<128x128xf32, #linear1>
          %239 = ttg.memdesc_index %result_2[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %240 = ttg.memdesc_index %8[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %240, %217 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %result_20 = ttng.tmem_load %239 {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
          %241 = ttg.memdesc_index %58[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %241, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %242 = arith.subf %result_20, %238 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear1>
          %243 = arith.mulf %226, %242 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear1>
          %244 = arith.truncf %243 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear1> to tensor<128x128xf16, #linear1>
          %245 = ttng.tmem_subslice %result_2 {N = 0 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
          %246 = ttg.memdesc_reinterpret %245 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
          %247 = ttg.memdesc_index %246[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %248 = ttg.memdesc_index %4[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %249 = arith.xori %214, %true {async_task_id = array<i32: 0>} : i1
          %250 = arith.extui %249 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %248, %250 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.tmem_store %244, %247, %true {async_task_id = array<i32: 0>} : tensor<128x128xf16, #linear1> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %251 = ttg.memdesc_index %62[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %251, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %252 = ttg.memdesc_index %81[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %253 = ttg.memdesc_index %65[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %254 = arith.xori %214, %true : i1
          %255 = arith.extui %254 : i1 to i32
          ttng.wait_barrier %253, %255 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 4>, dstTask = 4 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %256 = ttg.memdesc_index %64[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %257 = ttg.memdesc_index %82[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %258 = ttg.memdesc_index %2[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %258, %250 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %259 = ttng.tmem_subslice %247 {N = 0 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 128x128>
          %result_21 = ttng.tmem_load %259 : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf16, #linear2>
          %260 = ttng.tmem_subslice %247 {N = 64 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 128x128>
          %result_22 = ttng.tmem_load %260 : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf16, #linear2>
          %261 = ttg.memdesc_subslice %257[0, 0] : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64>
          %262 = ttg.memdesc_subslice %257[128, 0] : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64>
          ttng.barrier_expect %256, 16384, %true : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %263 = nvg.cluster_id
          %264 = arith.cmpi eq, %263, %c0_i32 : i32
          scf.if %264 {
            ttg.local_store %result_21, %261 : tensor<128x64xf16, #linear2> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64>
            ttg.local_store %result_22, %252 : tensor<128x64xf16, #linear2> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
            ttng.wait_barrier_named %c15_i32, %c256_i32 : i32, i32
            ttng.fence_async_shared {bCluster = false}
            ttg.async_remote_shmem_copy %252, rank %c1_i32, %261 barrier %256 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64> barrier_ty !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          } else {
            ttg.local_store %result_22, %262 : tensor<128x64xf16, #linear2> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64>
            ttg.local_store %result_21, %252 : tensor<128x64xf16, #linear2> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
            ttng.wait_barrier_named %c15_i32, %c256_i32 : i32, i32
            ttng.fence_async_shared {bCluster = false}
            ttg.async_remote_shmem_copy %252, rank %c0_i32, %262 barrier %256 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64> barrier_ty !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          }
          %265 = ttg.memdesc_index %68[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %265, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %266 = arith.addi %arg68, %c1_i64 {async_task_id = array<i32: 0>} : i64
          scf.yield {async_task_id = array<i32: 0>} %266 : i64
        } {async_task_id = array<i32: 0>, tt.warp_specialize}
        %134 = ttg.memdesc_index %result[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %135 = ttg.memdesc_index %34[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %136 = arith.extui %132 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %135, %136 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %137 = ttng.tmem_subslice %134 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
        %138 = ttng.tmem_subslice %137 {N = 0 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128>
        %139 = ttng.tmem_subslice %138 {N = 0 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %result_3 = ttng.tmem_load %139 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        %140 = ttng.tmem_subslice %138 {N = 16 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %141 = ttng.tmem_subslice %137 {N = 32 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128>
        %142 = ttng.tmem_subslice %141 {N = 0 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %143 = ttng.tmem_subslice %141 {N = 16 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %144 = ttng.tmem_subslice %134 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
        %145 = ttng.tmem_subslice %144 {N = 0 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128>
        %146 = ttng.tmem_subslice %145 {N = 0 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %147 = ttng.tmem_subslice %145 {N = 16 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %148 = ttng.tmem_subslice %144 {N = 32 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128>
        %149 = ttng.tmem_subslice %148 {N = 0 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %150 = ttng.tmem_subslice %148 {N = 16 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %151 = ttg.memdesc_index %48[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %152 = arith.truncf %result_3 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        %result_4 = ttng.tmem_load %140 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        %153 = ttg.memdesc_index %84[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %152, %153 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %154 = ttng.async_tma_copy_local_to_global %arg46[%130, %c0_i32] %153 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %154   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %155 = arith.truncf %result_4 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        %result_5 = ttng.tmem_load %142 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        ttg.local_store %155, %153 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %156 = ttng.async_tma_copy_local_to_global %arg46[%130, %c16_i32] %153 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %156   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %157 = arith.truncf %result_5 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        %result_6 = ttng.tmem_load %143 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        ttg.local_store %157, %153 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %158 = ttng.async_tma_copy_local_to_global %arg46[%130, %c32_i32] %153 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %158   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %159 = arith.truncf %result_6 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        %result_7 = ttng.tmem_load %146 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        ttg.local_store %159, %153 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %160 = ttng.async_tma_copy_local_to_global %arg46[%130, %c48_i32] %153 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %160   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %161 = arith.truncf %result_7 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        %result_8 = ttng.tmem_load %147 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        ttg.local_store %161, %153 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %162 = ttng.async_tma_copy_local_to_global %arg46[%130, %c64_i32] %153 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %162   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %163 = arith.truncf %result_8 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        %result_9 = ttng.tmem_load %149 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        ttg.local_store %163, %153 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %164 = ttng.async_tma_copy_local_to_global %arg46[%130, %c80_i32] %153 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %164   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %165 = arith.truncf %result_9 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        %result_10 = ttng.tmem_load %150 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        ttng.arrive_barrier %151, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttg.local_store %165, %153 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %166 = ttng.async_tma_copy_local_to_global %arg46[%130, %c96_i32] %153 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %166   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %167 = arith.truncf %result_10 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        ttg.local_store %167, %153 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %168 = ttng.async_tma_copy_local_to_global %arg46[%130, %c112_i32] %153 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %168   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %169 = ttg.memdesc_index %result_0[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %170 = ttg.memdesc_index %32[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.wait_barrier %170, %136 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %171 = ttng.tmem_subslice %169 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
        %172 = ttng.tmem_subslice %171 {N = 0 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128>
        %173 = ttng.tmem_subslice %172 {N = 0 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %result_11 = ttng.tmem_load %173 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        %174 = ttng.tmem_subslice %172 {N = 16 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %175 = ttng.tmem_subslice %171 {N = 32 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128>
        %176 = ttng.tmem_subslice %175 {N = 0 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %177 = ttng.tmem_subslice %175 {N = 16 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %178 = ttng.tmem_subslice %169 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128>
        %179 = ttng.tmem_subslice %178 {N = 0 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128>
        %180 = ttng.tmem_subslice %179 {N = 0 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %181 = ttng.tmem_subslice %179 {N = 16 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %182 = ttng.tmem_subslice %178 {N = 32 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128>
        %183 = ttng.tmem_subslice %182 {N = 0 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %184 = ttng.tmem_subslice %182 {N = 16 : i32} : !ttg.memdesc<128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 128x128> -> !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128>
        %185 = ttg.memdesc_index %50[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %186 = arith.mulf %result_11, %114 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear>
        %result_12 = ttng.tmem_load %174 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        %187 = arith.truncf %186 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        %188 = ttg.memdesc_index %85[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %187, %188 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %189 = ttng.async_tma_copy_local_to_global %arg41[%130, %c0_i32] %188 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %189   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %190 = arith.mulf %result_12, %114 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear>
        %result_13 = ttng.tmem_load %176 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        %191 = arith.truncf %190 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        ttg.local_store %191, %188 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %192 = ttng.async_tma_copy_local_to_global %arg41[%130, %c16_i32] %188 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %192   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %193 = arith.mulf %result_13, %114 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear>
        %result_14 = ttng.tmem_load %177 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        %194 = arith.truncf %193 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        ttg.local_store %194, %188 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %195 = ttng.async_tma_copy_local_to_global %arg41[%130, %c32_i32] %188 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %195   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %196 = arith.mulf %result_14, %114 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear>
        %result_15 = ttng.tmem_load %180 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        %197 = arith.truncf %196 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        ttg.local_store %197, %188 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %198 = ttng.async_tma_copy_local_to_global %arg41[%130, %c48_i32] %188 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %198   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %199 = arith.mulf %result_15, %114 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear>
        %result_16 = ttng.tmem_load %181 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        %200 = arith.truncf %199 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        ttg.local_store %200, %188 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %201 = ttng.async_tma_copy_local_to_global %arg41[%130, %c64_i32] %188 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %201   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %202 = arith.mulf %result_16, %114 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear>
        %result_17 = ttng.tmem_load %183 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        %203 = arith.truncf %202 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        ttg.local_store %203, %188 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %204 = ttng.async_tma_copy_local_to_global %arg41[%130, %c80_i32] %188 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %204   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %205 = arith.mulf %result_17, %114 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear>
        %result_18 = ttng.tmem_load %184 : !ttg.memdesc<128x16xf32, #tmem4, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x16xf32, #linear>
        ttng.arrive_barrier %185, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %206 = arith.truncf %205 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        ttg.local_store %206, %188 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %207 = ttng.async_tma_copy_local_to_global %arg41[%130, %c96_i32] %188 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %207   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %208 = arith.mulf %result_18, %114 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear>
        %209 = arith.truncf %208 {async_task_id = array<i32: 0>} : tensor<128x16xf32, #linear> to tensor<128x16xf16, #linear>
        ttg.local_store %209, %188 {async_task_id = array<i32: 0>} : tensor<128x16xf16, #linear> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %210 = ttng.async_tma_copy_local_to_global %arg41[%130, %c112_i32] %188 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %210   {async_task_id = array<i32: 0>, can_rotate_by_buffer_count = 1 : i32} : !ttg.async.token
        %211 = arith.addi %arg64, %104 {async_task_id = array<i32: 0>} : i32
        %212 = arith.addi %arg65, %c1_i64 {async_task_id = array<i32: 0>} : i64
        scf.yield {async_task_id = array<i32: 0>} %211, %212, %133 : i32, i64, i64
      } {async_task_id = array<i32: 0>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_yield
    }
    partition0(%arg63: i32, %arg64: i32, %arg65: i32, %arg66: i32, %arg67: i32, %arg68: i32, %arg69: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg70: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg71: !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, %arg72: !tt.tensordesc<128x16xf32, #shared1>, %arg73: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg74: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg75: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg76: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg77: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg78: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg79: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg80: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg81: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg82: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg83: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg84: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg85: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg86: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg87: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg88: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg89: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg90: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg91: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg92: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg93: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg94: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg95: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg96: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg97: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg98: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg99: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg100: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg101: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg102: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg103: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg104: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg105: !tt.tensordesc<128x128xf16, #shared>, %arg106: !tt.tensordesc<256x64xf16, #shared>, %arg107: !tt.tensordesc<128x128xf16, #shared>, %arg108: !tt.tensordesc<64x128xf16, #shared>, %arg109: !tt.tensordesc<128x64xf16, #shared>, %arg110: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg111: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %arg112: !tt.tensordesc<128xf32, #shared3>, %arg113: !tt.tensordesc<128x64xf16, #shared>, %arg114: !tt.tensordesc<64x128xf16, #shared>, %arg115: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg116: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %arg117: !tt.tensordesc<128xf32, #shared3>, %arg118: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg119: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg120: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg121: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg122: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg126: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg127: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg128: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg129: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg130: !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>, %arg131: !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>) num_warps(4) {
      %cst = arith.constant dense<0.693147182> : tensor<128x16xf32, #linear3>
      %c0_i32_3 = arith.constant 0 : i32
      %c1_i32_4 = arith.constant 1 : i32
      %c2_i32_5 = arith.constant 2 : i32
      %c128_i32_6 = arith.constant 128 : i32
      %c127_i32_7 = arith.constant 127 : i32
      %c2_i64 = arith.constant 2 : i64
      %c16_i32_8 = arith.constant 16 : i32
      %c32_i32_9 = arith.constant 32 : i32
      %c48_i32_10 = arith.constant 48 : i32
      %c64_i32_11 = arith.constant 64 : i32
      %c0_i64_12 = arith.constant 0 : i64
      %c1_i64_13 = arith.constant 1 : i64
      %98 = arith.addi %arg63, %c127_i32_7 {async_task_id = array<i32: 1>} : i32
      %99 = arith.divsi %98, %c128_i32_6 {async_task_id = array<i32: 1>} : i32
      %100 = tt.get_program_id x {async_task_id = array<i32: 1>} : i32
      %101 = arith.remsi %100, %c2_i32_5 {async_task_id = array<i32: 1>} : i32
      %102 = arith.divsi %100, %c2_i32_5 {async_task_id = array<i32: 1>} : i32
      %103 = tt.get_num_programs x {async_task_id = array<i32: 1>} : i32
      %104 = arith.divsi %103, %c2_i32_5 {async_task_id = array<i32: 1>} : i32
      %105 = arith.divsi %99, %c2_i32_5 {async_task_id = array<i32: 1>} : i32
      %106 = arith.muli %105, %arg64 {async_task_id = array<i32: 1>} : i32
      %107 = arith.muli %106, %arg65 {async_task_id = array<i32: 1>} : i32
      %108 = arith.divsi %107, %104 {async_task_id = array<i32: 1>} : i32
      %109 = arith.remsi %107, %104 {async_task_id = array<i32: 1>} : i32
      %110 = arith.cmpi slt, %102, %109 {async_task_id = array<i32: 1>} : i32
      %111 = scf.if %110 -> (i32) {
        %117 = arith.addi %108, %c1_i32_4 {async_task_id = array<i32: 1>} : i32
        scf.yield {async_task_id = array<i32: 1>} %117 : i32
      } else {
        scf.yield {async_task_id = array<i32: 1>} %108 : i32
      } {async_task_id = array<i32: 1>}
      %112 = arith.extsi %arg66 {async_task_id = array<i32: 1>} : i32 to i64
      %113 = arith.divsi %arg63, %c128_i32_6 {async_task_id = array<i32: 1>} : i32
      %114 = arith.muli %101, %c64_i32_11 {async_task_id = array<i32: 1>} : i32
      %115 = arith.extsi %114 {async_task_id = array<i32: 1>} : i32 to i64
      %116:2 = scf.for %arg132 = %c0_i32_3 to %111 step %c1_i32_4 iter_args(%arg133 = %102, %arg134 = %c0_i64_12) -> (i32, i64)  : i32 {
        %117 = arith.divsi %arg133, %105 {async_task_id = array<i32: 1>} : i32
        %118 = arith.remsi %117, %arg65 {async_task_id = array<i32: 1>} : i32
        %119 = arith.muli %arg67, %118 {async_task_id = array<i32: 1>} : i32
        %120 = arith.divsi %117, %arg65 {async_task_id = array<i32: 1>} : i32
        %121 = arith.muli %arg68, %120 {async_task_id = array<i32: 1>} : i32
        %122 = arith.addi %119, %121 {async_task_id = array<i32: 1>} : i32
        %123 = arith.extsi %122 {async_task_id = array<i32: 1>} : i32 to i64
        %124 = arith.divsi %123, %112 {async_task_id = array<i32: 1>} : i64
        %125:2 = scf.for %arg135 = %c0_i32_3 to %113 step %c1_i32_4 iter_args(%arg136 = %c0_i32_3, %arg137 = %arg134) -> (i32, i64)  : i32 {
          %127 = arith.extsi %arg136 {async_task_id = array<i32: 1>} : i32 to i64
          %128 = arith.addi %124, %127 {async_task_id = array<i32: 1>} : i64
          %129 = arith.andi %arg137, %c1_i64_13 {async_task_id = array<i32: 1>} : i64
          %130 = arith.trunci %129 {async_task_id = array<i32: 1>} : i64 to i1
          %131 = ttng.tmem_subslice %arg69 {N = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %132 = ttg.memdesc_reinterpret %131 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x128xf32, #tmem5, #ttng.tensor_memory, mutable>
          %133 = ttg.memdesc_index %132[%c0_i32_3] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x64x128xf32, #tmem5, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem6, #ttng.tensor_memory, mutable>
          %134 = ttg.memdesc_index %arg70[%c0_i32_3] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %135 = arith.extui %130 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %134, %135 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3, 4>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %result_14 = ttng.tmem_load %133 {async_task_id = array<i32: 1>} : !ttg.memdesc<64x128xf32, #tmem6, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear4>
          %136 = ttg.memdesc_index %arg129[%c0_i32_3] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %136, 1 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3, 4>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %137 = tt.reshape %result_14 : tensor<64x128xf32, #linear4> -> tensor<128x2x32xf32, #linear5>
          %138 = tt.trans %137 {async_task_id = array<i32: 1>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
          %outLHS, %outRHS = tt.split %138 : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear7>
          %139 = tt.reshape %outLHS {async_task_id = array<i32: 1>} : tensor<128x32xf32, #linear7> -> tensor<128x2x16xf32, #linear8>
          %140 = tt.trans %139 {async_task_id = array<i32: 1>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear8> -> tensor<128x16x2xf32, #linear9>
          %outLHS_15, %outRHS_16 = tt.split %140 : tensor<128x16x2xf32, #linear9> -> tensor<128x16xf32, #linear3>
          %141 = tt.reshape %outRHS {async_task_id = array<i32: 1>} : tensor<128x32xf32, #linear7> -> tensor<128x2x16xf32, #linear8>
          %142 = tt.trans %141 {async_task_id = array<i32: 1>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear8> -> tensor<128x16x2xf32, #linear9>
          %outLHS_17, %outRHS_18 = tt.split %142 : tensor<128x16x2xf32, #linear9> -> tensor<128x16xf32, #linear3>
          %143 = arith.addi %128, %115 {async_task_id = array<i32: 1>} : i64
          %144 = arith.muli %143, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %145 = arith.mulf %outLHS_15, %cst {async_task_id = array<i32: 1>} : tensor<128x16xf32, #linear3>
          %146 = arith.trunci %144 {async_task_id = array<i32: 1>} : i64 to i32
          %147 = ttg.memdesc_index %arg71[%c0_i32_3] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          ttg.local_store %145, %147 {async_task_id = array<i32: 1>} : tensor<128x16xf32, #linear3> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          ttng.fence_async_shared {bCluster = false}
          %148 = ttng.async_tma_reduce add, %arg72[%146, %c0_i32_3] %147 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          ttng.async_tma_store_token_wait %148   {async_task_id = array<i32: 1>} : !ttg.async.token
          %149 = arith.mulf %outRHS_16, %cst {async_task_id = array<i32: 1>} : tensor<128x16xf32, #linear3>
          ttg.local_store %149, %147 {async_task_id = array<i32: 1>} : tensor<128x16xf32, #linear3> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          ttng.fence_async_shared {bCluster = false}
          %150 = ttng.async_tma_reduce add, %arg72[%146, %c16_i32_8] %147 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          ttng.async_tma_store_token_wait %150   {async_task_id = array<i32: 1>} : !ttg.async.token
          %151 = arith.mulf %outLHS_17, %cst {async_task_id = array<i32: 1>} : tensor<128x16xf32, #linear3>
          ttg.local_store %151, %147 {async_task_id = array<i32: 1>} : tensor<128x16xf32, #linear3> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          ttng.fence_async_shared {bCluster = false}
          %152 = ttng.async_tma_reduce add, %arg72[%146, %c32_i32_9] %147 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          ttng.async_tma_store_token_wait %152   {async_task_id = array<i32: 1>} : !ttg.async.token
          %153 = arith.mulf %outRHS_18, %cst {async_task_id = array<i32: 1>} : tensor<128x16xf32, #linear3>
          ttg.local_store %153, %147 {async_task_id = array<i32: 1>} : tensor<128x16xf32, #linear3> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
          ttng.fence_async_shared {bCluster = false}
          %154 = ttng.async_tma_reduce add, %arg72[%146, %c48_i32_10] %147 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          ttng.async_tma_store_token_wait %154   {async_task_id = array<i32: 1>} : !ttg.async.token
          %155 = arith.addi %arg136, %c128_i32_6 {async_task_id = array<i32: 1>} : i32
          %156 = arith.addi %arg137, %c1_i64_13 {async_task_id = array<i32: 1>} : i64
          scf.yield {async_task_id = array<i32: 1>} %155, %156 : i32, i64
        } {async_task_id = array<i32: 1>, tt.warp_specialize}
        %126 = arith.addi %arg133, %104 {async_task_id = array<i32: 1>} : i32
        scf.yield {async_task_id = array<i32: 1>} %126, %125#1 : i32, i64
      } {async_task_id = array<i32: 1>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition1(%arg63: i32, %arg64: i32, %arg65: i32, %arg66: i32, %arg67: i32, %arg68: i32, %arg69: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg70: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg71: !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, %arg72: !tt.tensordesc<128x16xf32, #shared1>, %arg73: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg74: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg75: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg76: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg77: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg78: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg79: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg80: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg81: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg82: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg83: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg84: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg85: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg86: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg87: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg88: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg89: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg90: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg91: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg92: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg93: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg94: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg95: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg96: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg97: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg98: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg99: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg100: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg101: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg102: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg103: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg104: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg105: !tt.tensordesc<128x128xf16, #shared>, %arg106: !tt.tensordesc<256x64xf16, #shared>, %arg107: !tt.tensordesc<128x128xf16, #shared>, %arg108: !tt.tensordesc<64x128xf16, #shared>, %arg109: !tt.tensordesc<128x64xf16, #shared>, %arg110: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg111: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %arg112: !tt.tensordesc<128xf32, #shared3>, %arg113: !tt.tensordesc<128x64xf16, #shared>, %arg114: !tt.tensordesc<64x128xf16, #shared>, %arg115: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg116: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %arg117: !tt.tensordesc<128xf32, #shared3>, %arg118: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg119: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg120: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg121: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg122: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg126: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg127: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg128: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg129: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg130: !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>, %arg131: !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>) num_warps(1) {
      %false = arith.constant false
      %c0_i32_3 = arith.constant 0 : i32
      %c1_i32_4 = arith.constant 1 : i32
      %c2_i32_5 = arith.constant 2 : i32
      %c128_i32_6 = arith.constant 128 : i32
      %c127_i32_7 = arith.constant 127 : i32
      %true_8 = arith.constant true
      %c0_i64_9 = arith.constant 0 : i64
      %c1_i64_10 = arith.constant 1 : i64
      %c-2_i32 = arith.constant -2 : i32
      %c2_i64 = arith.constant 2 : i64
      %c3_i32_11 = arith.constant 3 : i32
      %c4_i32_12 = arith.constant 4 : i32
      %98 = arith.addi %arg63, %c127_i32_7 {async_task_id = array<i32: 2>} : i32
      %99 = arith.divsi %98, %c128_i32_6 {async_task_id = array<i32: 2>} : i32
      %100 = tt.get_program_id x {async_task_id = array<i32: 2>} : i32
      %101 = arith.divsi %100, %c2_i32_5 {async_task_id = array<i32: 2>} : i32
      %102 = tt.get_num_programs x {async_task_id = array<i32: 2>} : i32
      %103 = arith.divsi %102, %c2_i32_5 {async_task_id = array<i32: 2>} : i32
      %104 = arith.divsi %99, %c2_i32_5 {async_task_id = array<i32: 2>} : i32
      %105 = arith.muli %104, %arg64 {async_task_id = array<i32: 2>} : i32
      %106 = arith.muli %105, %arg65 {async_task_id = array<i32: 2>} : i32
      %107 = arith.divsi %106, %103 {async_task_id = array<i32: 2>} : i32
      %108 = arith.remsi %106, %103 {async_task_id = array<i32: 2>} : i32
      %109 = arith.cmpi slt, %101, %108 {async_task_id = array<i32: 2>} : i32
      %110 = scf.if %109 -> (i32) {
        %113 = arith.addi %107, %c1_i32_4 {async_task_id = array<i32: 2>} : i32
        scf.yield {async_task_id = array<i32: 2>} %113 : i32
      } else {
        scf.yield {async_task_id = array<i32: 2>} %107 : i32
      } {async_task_id = array<i32: 2>}
      %111 = arith.divsi %arg63, %c128_i32_6 {async_task_id = array<i32: 2>} : i32
      %112:2 = scf.for %arg132 = %c0_i32_3 to %110 step %c1_i32_4 iter_args(%arg133 = %c0_i64_9, %arg134 = %c0_i64_9) -> (i64, i64)  : i32 {
        %113 = arith.andi %arg133, %c1_i64_10 {async_task_id = array<i32: 2>} : i64
        %114 = arith.trunci %113 {async_task_id = array<i32: 2>} : i64 to i1
        %115 = ttg.memdesc_index %arg73[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %116 = arith.extui %114 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %115, %116, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %117 = ttg.memdesc_index %arg74[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.wait_barrier %117, %116, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %118 = ttg.memdesc_index %arg75[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.wait_barrier %118, %116, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %119 = ttg.memdesc_index %arg118[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %120 = arith.xori %114, %true_8 : i1
        %121 = arith.extui %120 : i1 to i32
        ttng.wait_barrier %119, %121 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %122 = ttg.memdesc_index %arg119[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.wait_barrier %122, %121 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %123 = arith.cmpi sgt, %111, %c0_i32_3 : i32
        %124 = arith.andi %arg134, %c1_i64_10 {async_task_id = array<i32: 2>} : i64
        %125 = arith.trunci %124 {async_task_id = array<i32: 2>} : i64 to i1
        %126 = ttg.memdesc_index %arg76[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        %127 = ttg.memdesc_trans %126 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared5, #smem, mutable>
        %128 = ttg.memdesc_index %arg77[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %129 = arith.extui %125 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %128, %129, %123 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %130 = ttg.memdesc_index %arg78[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %131 = ttg.memdesc_index %arg79[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %132 = ttg.memdesc_index %arg80[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %133 = ttg.memdesc_index %arg81[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %134 = ttg.memdesc_index %arg121[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %135 = arith.xori %125, %true_8 : i1
        %136 = arith.extui %135 : i1 to i32
        ttng.wait_barrier %134, %136, %123 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %137 = ttg.memdesc_index %arg82[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %138 = arith.xori %125, %true_8 {async_task_id = array<i32: 2>} : i1
        %139 = arith.extui %138 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %137, %139, %123 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {direction = "backward", dstTask = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %140 = ttg.memdesc_index %arg131[%c0_i32_3] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %141 = nvg.cluster_id
        %142 = arith.andi %141, %c-2_i32 : i32
        %143 = ttng.map_to_remote_buffer %140, %142 : !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
        ttng.arrive_barrier %143, 1 : !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
        %144 = arith.extui %arg132 : i32 to i64
        %145 = arith.remui %144, %c2_i64 : i64
        %146 = arith.trunci %145 : i64 to i32
        %147 = arith.remui %141, %c2_i32_5 : i32
        %148 = arith.cmpi eq, %147, %c0_i32_3 : i32
        ttng.wait_barrier %140, %146, %148 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_mma %130, %127, %131, %false, %123, %132[%true_8], %133[%true_8] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,1,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared5, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %149 = ttg.memdesc_index %arg83[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        %150 = ttg.memdesc_trans %149 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared5, #smem, mutable>
        %151 = ttg.memdesc_index %arg84[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.wait_barrier %151, %129, %123 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %152 = ttg.memdesc_index %arg85[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %153 = ttg.memdesc_index %arg69[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %154 = ttg.memdesc_index %arg86[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %155 = ttg.memdesc_index %arg87[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %156 = ttg.memdesc_index %arg123[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.wait_barrier %156, %136, %123 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %157 = ttg.memdesc_index %arg129[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.wait_barrier %157, %136, %123 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 1>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %158 = ttg.memdesc_index %arg131[%c1_i32_4] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %159 = ttng.map_to_remote_buffer %158, %142 : !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
        ttng.arrive_barrier %159, 1 : !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
        ttng.wait_barrier %158, %146, %148 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_mma %152, %150, %153, %false, %123, %154[%true_8], %155[%true_8] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared5, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %160 = ttg.memdesc_index %arg88[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %161 = ttg.memdesc_index %arg89[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %162 = ttng.tmem_subslice %arg79 {N = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
        %163 = ttg.memdesc_reinterpret %162 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        %164 = ttg.memdesc_index %163[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %165 = ttg.memdesc_index %arg90[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %166 = ttg.memdesc_index %arg91[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.wait_barrier %166, %129, %123 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %167 = ttg.memdesc_index %arg122[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %168 = arith.extui %125 : i1 to i32
        ttng.wait_barrier %167, %168, %123 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %169 = ttg.memdesc_index %arg131[%c2_i32_5] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %170 = ttng.map_to_remote_buffer %169, %142 : !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
        ttng.arrive_barrier %170, 1 : !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
        ttng.wait_barrier %169, %146, %148 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_mma %164, %161, %160, %false, %123, %165[%true_8], %137[%true_8] {async_task_id = array<i32: 2>, is_async, tmem.start = array<i32: 3>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %171 = arith.subi %111, %c1_i32_4 : i32
        %172:3 = scf.for %arg135 = %c0_i32_3 to %171 step %c1_i32_4 iter_args(%arg136 = %false, %arg137 = %arg134, %arg138 = %125) -> (i1, i64, i1)  : i32 {
          %180 = arith.addi %arg137, %c1_i64_10 {async_task_id = array<i32: 2>} : i64
          %181 = arith.andi %180, %c1_i64_10 {async_task_id = array<i32: 2>} : i64
          %182 = arith.trunci %181 {async_task_id = array<i32: 2>} : i64 to i1
          %183 = arith.extui %182 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %128, %183, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %184 = arith.andi %arg137, %c1_i64_10 {async_task_id = array<i32: 2>} : i64
          %185 = arith.trunci %184 {async_task_id = array<i32: 2>} : i64 to i1
          %186 = arith.xori %182, %true_8 : i1
          %187 = arith.extui %186 : i1 to i32
          ttng.wait_barrier %134, %187, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %188 = arith.xori %182, %true_8 {async_task_id = array<i32: 2>} : i1
          %189 = arith.extui %188 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %137, %189, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {direction = "backward", dstTask = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %190 = ttg.memdesc_index %arg130[%c0_i32_3] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %191 = ttng.map_to_remote_buffer %190, %142 : !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.arrive_barrier %191, 1 : !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          %192 = arith.extui %arg135 : i32 to i64
          %193 = arith.extui %171 : i32 to i64
          %194 = arith.muli %144, %193 : i64
          %195 = arith.addi %194, %192 : i64
          %196 = arith.remui %195, %c2_i64 : i64
          %197 = arith.trunci %196 : i64 to i32
          ttng.wait_barrier %190, %197, %148 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.tc_gen5_mma %130, %127, %131, %false, %true_8, %132[%true_8], %133[%true_8] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,1,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared5, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %198 = ttg.memdesc_index %arg92[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %199 = ttg.memdesc_index %arg93[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %200 = ttng.tmem_subslice %arg69 {N = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
          %201 = ttg.memdesc_reinterpret %200 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
          %202 = ttg.memdesc_index %201[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %203 = ttg.memdesc_index %arg94[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %204 = ttg.memdesc_index %arg95[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %205 = arith.extui %185 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %204, %205, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %206 = ttg.memdesc_index %arg96[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %207 = ttg.memdesc_index %arg125[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %208 = arith.extui %185 : i1 to i32
          ttng.wait_barrier %207, %208 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %209 = ttg.memdesc_index %arg130[%c1_i32_4] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %210 = ttng.map_to_remote_buffer %209, %142 : !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.arrive_barrier %210, 1 : !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.wait_barrier %209, %197, %148 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.tc_gen5_mma %202, %199, %198, %arg136, %true_8, %203[%true_8], %206[%true_8] {async_task_id = array<i32: 2>, is_async, tmem.start = array<i32: 4>, tt.autows = "{\22stage\22: \221\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,10\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %211 = ttg.memdesc_index %arg97[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %212 = ttg.memdesc_trans %211 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared5, #smem, mutable>
          %213 = ttg.memdesc_index %arg128[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %213, %208 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %214 = ttg.memdesc_index %arg98[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %215 = ttng.tmem_subslice %arg69 {N = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %216 = ttg.memdesc_reinterpret %215 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x128xf32, #tmem5, #ttng.tensor_memory, mutable>
          %217 = ttg.memdesc_index %216[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf32, #tmem5, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem6, #ttng.tensor_memory, mutable>
          %218 = ttg.memdesc_index %arg99[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %219 = ttg.memdesc_index %arg70[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %220 = arith.extui %arg138 : i1 to i32
          ttng.wait_barrier %156, %220 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %221 = ttg.memdesc_index %arg130[%c2_i32_5] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %222 = ttng.map_to_remote_buffer %221, %142 : !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.arrive_barrier %222, 1 : !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.wait_barrier %221, %197, %148 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.fence_async_shared {bCluster = false}
          ttng.tc_gen5_mma %212, %214, %217, %false, %true_8, %218[%true_8], %219[%true_8] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x256xf16, #shared5, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem6, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %151, %183, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %156, %187, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %157, %187, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 1>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %223 = ttg.memdesc_index %arg130[%c3_i32_11] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %224 = ttng.map_to_remote_buffer %223, %142 : !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.arrive_barrier %224, 1 : !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.wait_barrier %223, %197, %148 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.tc_gen5_mma %152, %150, %153, %false, %true_8, %154[%true_8], %155[%true_8] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared5, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %166, %183, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %225 = arith.extui %182 : i1 to i32
          ttng.wait_barrier %167, %225, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %226 = ttg.memdesc_index %arg130[%c4_i32_12] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %227 = ttng.map_to_remote_buffer %226, %142 : !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.arrive_barrier %227, 1 : !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.wait_barrier %226, %197, %148 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.tc_gen5_mma %164, %161, %160, %true_8, %true_8, %165[%true_8], %137[%true_8] {async_task_id = array<i32: 2>, is_async, tmem.start = array<i32: 3>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          scf.yield %true_8, %180, %182 : i1, i64, i1
        } {async_task_id = array<i32: 2>, tt.warp_specialize}
        %173 = scf.if %123 -> (i64) {
          %180 = arith.addi %172#1, %c1_i64_10 {async_task_id = array<i32: 2>} : i64
          %181 = arith.andi %172#1, %c1_i64_10 {async_task_id = array<i32: 2>} : i64
          %182 = arith.trunci %181 {async_task_id = array<i32: 2>} : i64 to i1
          %183 = ttg.memdesc_index %arg92[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %184 = ttg.memdesc_index %arg93[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %185 = ttng.tmem_subslice %arg69 {N = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
          %186 = ttg.memdesc_reinterpret %185 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
          %187 = ttg.memdesc_index %186[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %188 = ttg.memdesc_index %arg94[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %189 = ttg.memdesc_index %arg95[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %190 = arith.extui %182 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %189, %190, %true_8 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %191 = ttg.memdesc_index %arg96[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %192 = ttg.memdesc_index %arg125[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %193 = arith.extui %182 : i1 to i32
          ttng.wait_barrier %192, %193 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %194 = ttg.memdesc_index %arg131[%c3_i32_11] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %195 = ttng.map_to_remote_buffer %194, %142 : !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.arrive_barrier %195, 1 : !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.wait_barrier %194, %146, %148 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.tc_gen5_mma %187, %184, %183, %172#0, %true_8, %188[%true_8], %191[%true_8] {async_task_id = array<i32: 2>, is_async, tmem.start = array<i32: 4>, tt.autows = "{\22stage\22: \221\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,10\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %196 = ttg.memdesc_index %arg97[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %197 = ttg.memdesc_trans %196 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared5, #smem, mutable>
          %198 = ttg.memdesc_index %arg128[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %198, %193 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %199 = ttg.memdesc_index %arg98[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %200 = ttng.tmem_subslice %arg69 {N = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %201 = ttg.memdesc_reinterpret %200 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x128xf32, #tmem5, #ttng.tensor_memory, mutable>
          %202 = ttg.memdesc_index %201[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf32, #tmem5, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem6, #ttng.tensor_memory, mutable>
          %203 = ttg.memdesc_index %arg99[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %204 = ttg.memdesc_index %arg70[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %205 = arith.extui %172#2 : i1 to i32
          ttng.wait_barrier %156, %205 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %206 = ttg.memdesc_index %arg131[%c4_i32_12] : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %207 = ttng.map_to_remote_buffer %206, %142 : !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.arrive_barrier %207, 1 : !ttg.memdesc<1xi64, #shared4, #ttng.shared_cluster_memory, mutable>
          ttng.wait_barrier %206, %146, %148 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.fence_async_shared {bCluster = false}
          ttng.tc_gen5_mma %197, %199, %202, %false, %true_8, %203[%true_8], %204[%true_8] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x256xf16, #shared5, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem6, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          scf.yield %180 : i64
        } else {
          scf.yield %172#1 : i64
        }
        %174 = ttg.memdesc_index %arg100[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_commit %174 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %175 = ttg.memdesc_index %arg101[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_commit %175 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %176 = ttg.memdesc_index %arg102[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_commit %176 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %177 = ttg.memdesc_index %arg103[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_commit %177 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %178 = ttg.memdesc_index %arg104[%c0_i32_3] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_commit %178 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %179 = arith.addi %arg133, %c1_i64_10 {async_task_id = array<i32: 2>} : i64
        scf.yield {async_task_id = array<i32: 2>} %179, %173 : i64, i64
      } {async_task_id = array<i32: 2>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition2(%arg63: i32, %arg64: i32, %arg65: i32, %arg66: i32, %arg67: i32, %arg68: i32, %arg69: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg70: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg71: !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, %arg72: !tt.tensordesc<128x16xf32, #shared1>, %arg73: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg74: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg75: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg76: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg77: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg78: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg79: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg80: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg81: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg82: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg83: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg84: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg85: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg86: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg87: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg88: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg89: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg90: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg91: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg92: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg93: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg94: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg95: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg96: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg97: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg98: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg99: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg100: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg101: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg102: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg103: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg104: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg105: !tt.tensordesc<128x128xf16, #shared>, %arg106: !tt.tensordesc<256x64xf16, #shared>, %arg107: !tt.tensordesc<128x128xf16, #shared>, %arg108: !tt.tensordesc<64x128xf16, #shared>, %arg109: !tt.tensordesc<128x64xf16, #shared>, %arg110: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg111: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %arg112: !tt.tensordesc<128xf32, #shared3>, %arg113: !tt.tensordesc<128x64xf16, #shared>, %arg114: !tt.tensordesc<64x128xf16, #shared>, %arg115: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg116: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %arg117: !tt.tensordesc<128xf32, #shared3>, %arg118: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg119: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg120: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg121: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg122: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg126: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg127: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg128: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg129: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg130: !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>, %arg131: !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>) num_warps(1) {
      %c0_i32_3 = arith.constant 0 : i32
      %c1_i32_4 = arith.constant 1 : i32
      %c2_i32_5 = arith.constant 2 : i32
      %c128_i32_6 = arith.constant 128 : i32
      %c127_i32_7 = arith.constant 127 : i32
      %c64_i32_8 = arith.constant 64 : i32
      %c0_i64_9 = arith.constant 0 : i64
      %c1_i64_10 = arith.constant 1 : i64
      %true_11 = arith.constant true
      %98 = arith.addi %arg63, %c127_i32_7 {async_task_id = array<i32: 3>} : i32
      %99 = arith.divsi %98, %c128_i32_6 {async_task_id = array<i32: 3>} : i32
      %100 = tt.get_program_id x {async_task_id = array<i32: 3>} : i32
      %101 = arith.remsi %100, %c2_i32_5 {async_task_id = array<i32: 3>} : i32
      %102 = arith.divsi %100, %c2_i32_5 {async_task_id = array<i32: 3>} : i32
      %103 = tt.get_num_programs x {async_task_id = array<i32: 3>} : i32
      %104 = arith.divsi %103, %c2_i32_5 {async_task_id = array<i32: 3>} : i32
      %105 = arith.divsi %99, %c2_i32_5 {async_task_id = array<i32: 3>} : i32
      %106 = arith.muli %105, %arg64 {async_task_id = array<i32: 3>} : i32
      %107 = arith.muli %106, %arg65 {async_task_id = array<i32: 3>} : i32
      %108 = arith.divsi %107, %104 {async_task_id = array<i32: 3>} : i32
      %109 = arith.remsi %107, %104 {async_task_id = array<i32: 3>} : i32
      %110 = arith.cmpi slt, %102, %109 {async_task_id = array<i32: 3>} : i32
      %111 = scf.if %110 -> (i32) {
        %119 = arith.addi %108, %c1_i32_4 {async_task_id = array<i32: 3>} : i32
        scf.yield {async_task_id = array<i32: 3>} %119 : i32
      } else {
        scf.yield {async_task_id = array<i32: 3>} %108 : i32
      } {async_task_id = array<i32: 3>}
      %112 = arith.extsi %arg66 {async_task_id = array<i32: 3>} : i32 to i64
      %113 = arith.muli %101, %c128_i32_6 {async_task_id = array<i32: 3>} : i32
      %114 = arith.divsi %arg63, %c128_i32_6 {async_task_id = array<i32: 3>} : i32
      %115 = nvg.cluster_id {async_task_id = array<i32: 3>}
      %116 = arith.remsi %115, %c2_i32_5 {async_task_id = array<i32: 3>} : i32
      %117 = arith.muli %116, %c64_i32_8 {async_task_id = array<i32: 3>} : i32
      %118:3 = scf.for %arg132 = %c0_i32_3 to %111 step %c1_i32_4 iter_args(%arg133 = %102, %arg134 = %c0_i64_9, %arg135 = %c0_i64_9) -> (i32, i64, i64)  : i32 {
        %119 = arith.remsi %arg133, %105 {async_task_id = array<i32: 3>} : i32
        %120 = arith.divsi %arg133, %105 {async_task_id = array<i32: 3>} : i32
        %121 = arith.muli %119, %c2_i32_5 {async_task_id = array<i32: 3>} : i32
        %122 = arith.addi %121, %101 {async_task_id = array<i32: 3>} : i32
        %123 = arith.muli %120, %arg63 {async_task_id = array<i32: 3>} : i32
        %124 = arith.extsi %123 {async_task_id = array<i32: 3>} : i32 to i64
        %125 = arith.remsi %120, %arg65 {async_task_id = array<i32: 3>} : i32
        %126 = arith.muli %arg67, %125 {async_task_id = array<i32: 3>} : i32
        %127 = arith.divsi %120, %arg65 {async_task_id = array<i32: 3>} : i32
        %128 = arith.muli %arg68, %127 {async_task_id = array<i32: 3>} : i32
        %129 = arith.addi %126, %128 {async_task_id = array<i32: 3>} : i32
        %130 = arith.extsi %129 {async_task_id = array<i32: 3>} : i32 to i64
        %131 = arith.divsi %130, %112 {async_task_id = array<i32: 3>} : i64
        %132 = arith.muli %122, %c128_i32_6 {async_task_id = array<i32: 3>} : i32
        %133 = arith.extsi %132 {async_task_id = array<i32: 3>} : i32 to i64
        %134 = arith.addi %131, %133 {async_task_id = array<i32: 3>} : i64
        %135 = arith.trunci %134 {async_task_id = array<i32: 3>} : i64 to i32
        %136 = arith.andi %arg134, %c1_i64_10 {async_task_id = array<i32: 3>} : i64
        %137 = arith.trunci %136 {async_task_id = array<i32: 3>} : i64 to i1
        %138 = ttg.memdesc_index %arg104[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %139 = arith.xori %137, %true_11 {async_task_id = array<i32: 3>} : i1
        %140 = arith.extui %139 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %138, %140 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %141 = ttg.memdesc_index %arg73[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.barrier_expect %141, 32768 {async_task_id = array<i32: 3>}, %true_11 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %142 = ttg.memdesc_index %arg78[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %arg105[%135, %c0_i32_3] %142, %141, %true_11 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %143 = arith.subi %132, %113 {async_task_id = array<i32: 3>} : i32
        %144 = arith.extsi %143 {async_task_id = array<i32: 3>} : i32 to i64
        %145 = arith.addi %131, %144 {async_task_id = array<i32: 3>} : i64
        %146 = arith.trunci %145 {async_task_id = array<i32: 3>} : i64 to i32
        %147 = ttg.memdesc_index %arg103[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.wait_barrier %147, %140 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %148 = ttg.memdesc_index %arg74[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.barrier_expect %148, 32768 {async_task_id = array<i32: 3>}, %true_11 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %149 = ttg.memdesc_index %arg98[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %arg106[%146, %117] %149, %148, %true_11 {async_task_id = array<i32: 3>} : !tt.tensordesc<256x64xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        %150 = ttg.memdesc_index %arg102[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.wait_barrier %150, %140 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %151 = ttg.memdesc_index %arg75[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.barrier_expect %151, 32768 {async_task_id = array<i32: 3>}, %true_11 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %152 = ttg.memdesc_index %arg85[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %arg107[%135, %c0_i32_3] %152, %151, %true_11 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %153:2 = scf.for %arg136 = %c0_i32_3 to %114 step %c1_i32_4 iter_args(%arg137 = %c0_i32_3, %arg138 = %arg135) -> (i32, i64)  : i32 {
          %156 = arith.extsi %arg137 {async_task_id = array<i32: 3>} : i32 to i64
          %157 = arith.addi %131, %156 {async_task_id = array<i32: 3>} : i64
          %158 = arith.trunci %157 {async_task_id = array<i32: 3>} : i64 to i32
          %159 = arith.addi %158, %117 {async_task_id = array<i32: 3>} : i32
          %160 = arith.andi %arg138, %c1_i64_10 {async_task_id = array<i32: 3>} : i64
          %161 = arith.trunci %160 {async_task_id = array<i32: 3>} : i64 to i1
          %162 = ttg.memdesc_index %arg80[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %163 = arith.xori %161, %true_11 {async_task_id = array<i32: 3>} : i1
          %164 = arith.extui %163 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %162, %164 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %165 = ttg.memdesc_index %arg77[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.barrier_expect %165, 16384 {async_task_id = array<i32: 3>}, %true_11 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %166 = ttg.memdesc_index %arg76[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg108[%159, %c0_i32_3] %166, %165, %true_11 {async_task_id = array<i32: 3>} : !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          %167 = ttg.memdesc_index %arg94[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %167, %164 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %168 = ttg.memdesc_index %arg95[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.barrier_expect %168, 16384 {async_task_id = array<i32: 3>}, %true_11 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %169 = ttg.memdesc_index %arg93[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg109[%158, %117] %169, %168, %true_11 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %170 = arith.addi %124, %156 {async_task_id = array<i32: 3>} : i64
          %171 = arith.trunci %170 {async_task_id = array<i32: 3>} : i64 to i32
          %172 = ttg.memdesc_index %arg120[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %173 = arith.xori %161, %true_11 : i1
          %174 = arith.extui %173 : i1 to i32
          ttng.wait_barrier %172, %174 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %175 = ttg.memdesc_index %arg110[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.barrier_expect %175, 512 {async_task_id = array<i32: 3>}, %true_11 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %176 = ttg.memdesc_index %arg111[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128xf32, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg112[%171] %176, %175, %true_11 {async_task_id = array<i32: 3>} : !tt.tensordesc<128xf32, #shared3>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
          %177 = ttg.memdesc_index %arg90[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %177, %164 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %178 = ttg.memdesc_index %arg91[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.barrier_expect %178, 16384 {async_task_id = array<i32: 3>}, %true_11 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %179 = ttg.memdesc_index %arg89[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg113[%158, %117] %179, %178, %true_11 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %180 = ttg.memdesc_index %arg86[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %180, %164 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %181 = ttg.memdesc_index %arg84[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.barrier_expect %181, 16384 {async_task_id = array<i32: 3>}, %true_11 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %182 = ttg.memdesc_index %arg83[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg114[%159, %c0_i32_3] %182, %181, %true_11 {async_task_id = array<i32: 3>} : !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          %183 = ttg.memdesc_index %arg124[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.wait_barrier %183, %174 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %184 = ttg.memdesc_index %arg115[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.barrier_expect %184, 512 {async_task_id = array<i32: 3>}, %true_11 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %185 = ttg.memdesc_index %arg116[%c0_i32_3] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128xf32, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg117[%171] %185, %184, %true_11 {async_task_id = array<i32: 3>} : !tt.tensordesc<128xf32, #shared3>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
          %186 = arith.addi %arg137, %c128_i32_6 {async_task_id = array<i32: 3>} : i32
          %187 = arith.addi %arg138, %c1_i64_10 {async_task_id = array<i32: 3>} : i64
          scf.yield {async_task_id = array<i32: 3>} %186, %187 : i32, i64
        } {async_task_id = array<i32: 3>, tt.warp_specialize}
        %154 = arith.addi %arg133, %104 {async_task_id = array<i32: 3>} : i32
        %155 = arith.addi %arg134, %c1_i64_10 {async_task_id = array<i32: 3>} : i64
        scf.yield {async_task_id = array<i32: 3>} %154, %155, %153#1 : i32, i64, i64
      } {async_task_id = array<i32: 3>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition3(%arg63: i32, %arg64: i32, %arg65: i32, %arg66: i32, %arg67: i32, %arg68: i32, %arg69: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg70: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg71: !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, %arg72: !tt.tensordesc<128x16xf32, #shared1>, %arg73: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg74: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg75: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg76: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg77: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg78: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg79: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg80: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg81: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg82: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg83: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %arg84: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg85: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg86: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg87: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg88: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg89: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg90: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg91: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg92: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg93: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg94: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg95: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg96: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg97: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg98: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %arg99: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg100: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg101: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg102: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg103: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg104: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg105: !tt.tensordesc<128x128xf16, #shared>, %arg106: !tt.tensordesc<256x64xf16, #shared>, %arg107: !tt.tensordesc<128x128xf16, #shared>, %arg108: !tt.tensordesc<64x128xf16, #shared>, %arg109: !tt.tensordesc<128x64xf16, #shared>, %arg110: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg111: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %arg112: !tt.tensordesc<128xf32, #shared3>, %arg113: !tt.tensordesc<128x64xf16, #shared>, %arg114: !tt.tensordesc<64x128xf16, #shared>, %arg115: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg116: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %arg117: !tt.tensordesc<128xf32, #shared3>, %arg118: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg119: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg120: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg121: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg122: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg126: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg127: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg128: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg129: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %arg130: !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>, %arg131: !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>) num_warps(1) {
      %c0_i32_3 = arith.constant 0 : i32
      %c1_i32_4 = arith.constant 1 : i32
      %c2_i32_5 = arith.constant 2 : i32
      %c128_i32_6 = arith.constant 128 : i32
      %c127_i32_7 = arith.constant 127 : i32
      %c0_i64_8 = arith.constant 0 : i64
      %c1_i64_9 = arith.constant 1 : i64
      %98 = arith.addi %arg63, %c127_i32_7 {async_task_id = array<i32: 4>} : i32
      %99 = arith.divsi %98, %c128_i32_6 {async_task_id = array<i32: 4>} : i32
      %100 = tt.get_program_id x {async_task_id = array<i32: 4>} : i32
      %101 = arith.divsi %100, %c2_i32_5 {async_task_id = array<i32: 4>} : i32
      %102 = tt.get_num_programs x {async_task_id = array<i32: 4>} : i32
      %103 = arith.divsi %102, %c2_i32_5 {async_task_id = array<i32: 4>} : i32
      %104 = arith.divsi %99, %c2_i32_5 {async_task_id = array<i32: 4>} : i32
      %105 = arith.muli %104, %arg64 {async_task_id = array<i32: 4>} : i32
      %106 = arith.muli %105, %arg65 {async_task_id = array<i32: 4>} : i32
      %107 = arith.divsi %106, %103 {async_task_id = array<i32: 4>} : i32
      %108 = arith.remsi %106, %103 {async_task_id = array<i32: 4>} : i32
      %109 = arith.cmpi slt, %101, %108 {async_task_id = array<i32: 4>} : i32
      %110 = scf.if %109 -> (i32) {
        %113 = arith.addi %107, %c1_i32_4 {async_task_id = array<i32: 4>} : i32
        scf.yield {async_task_id = array<i32: 4>} %113 : i32
      } else {
        scf.yield {async_task_id = array<i32: 4>} %107 : i32
      } {async_task_id = array<i32: 4>}
      %111 = arith.divsi %arg63, %c128_i32_6 {async_task_id = array<i32: 4>} : i32
      %112 = scf.for %arg132 = %c0_i32_3 to %110 step %c1_i32_4 iter_args(%arg133 = %c0_i64_8) -> (i64)  : i32 {
        %113 = scf.for %arg134 = %c0_i32_3 to %111 step %c1_i32_4 iter_args(%arg135 = %arg133) -> (i64)  : i32 {
          %114 = arith.andi %arg135, %c1_i64_9 {async_task_id = array<i32: 4>} : i64
          %115 = arith.trunci %114 {async_task_id = array<i32: 4>} : i64 to i1
          %116 = ttg.memdesc_index %arg126[%c0_i32_3] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %117 = arith.extui %115 : i1 to i32
          ttng.wait_barrier %116, %117 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 3>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.fence_async_shared {bCluster = false}
          %118 = ttg.memdesc_index %arg128[%c0_i32_3] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %118, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %119 = ttg.memdesc_index %arg127[%c0_i32_3] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %119, 1 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 3>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %120 = arith.addi %arg135, %c1_i64_9 {async_task_id = array<i32: 4>} : i64
          scf.yield {async_task_id = array<i32: 4>} %120 : i64
        } {async_task_id = array<i32: 4>, tt.warp_specialize}
        scf.yield {async_task_id = array<i32: 4>} %113 : i64
      } {async_task_id = array<i32: 4>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    } : (i32, i32, i32, i32, i32, i32, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !tt.tensordesc<128x128xf16, #shared>, !tt.tensordesc<256x64xf16, #shared>, !tt.tensordesc<128x128xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>, !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, !tt.tensordesc<128xf32, #shared3>, !tt.tensordesc<128x64xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, !tt.tensordesc<128xf32, #shared3>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>) -> ()
    ttng.inval_barrier %93 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %94 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %95 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %96 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %97 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.local_dealloc %92 : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %87 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %88 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %89 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %90 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %91 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.local_dealloc %86 : !ttg.memdesc<5x1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %47 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %43 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %39 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %31 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %29 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %21 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %15 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %13 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %11 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %9 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %17 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %19 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %25 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %27 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %5 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %3 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %33 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %35 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %37 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %41 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %45 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %23 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %7 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %49 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %51 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %53 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %55 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %57 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %59 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %61 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %63 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %66 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %67 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %69 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %71 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    tt.return
  }
}
