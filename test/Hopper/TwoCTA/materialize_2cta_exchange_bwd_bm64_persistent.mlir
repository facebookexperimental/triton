// RUN: triton-opt %s --nvgpu-materialize-2cta-exchange | FileCheck %s
// RUN: triton-opt %s --tritongpu-optimize-partition-warps | FileCheck %s --check-prefix=OPT

// Full TTGIR captured immediately after software pipelining and before
// NVGPUMaterialize2CTAExchange from the persistent
// _BWD_DOT_ATTRS_BM64_TMEM 2-CTA configuration.
// CHECK: ttng.init_barrier {{.*}}, 2
// CHECK: ttng.barrier_expect
// CHECK: scf.if
// CHECK: ttg.async_remote_shmem_store
// CHECK: ttg.async_remote_shmem_store
// CHECK-NOT: ttng.two_cta_peer_gather
// OPT: ttng.two_cta_peer_gather {{.*}} split_dim = 0 num_ctas = 2

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [4, 1, 8], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 64], [0, 32], [0, 1], [0, 2], [16, 0]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 64]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [[1, 0, 0], [0, 0, 64]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[0, 0, 1], [0, 64, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [0, 32, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [1, 0, 0], [2, 0, 0]], warp = [[4, 0, 0], [8, 0, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1, 0], [0, 0, 32], [0, 0, 1], [0, 0, 2], [16, 0, 0]], lane = [[0, 0, 4], [0, 0, 8], [0, 0, 16], [1, 0, 0], [2, 0, 0]], warp = [[4, 0, 0], [8, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 0, 1], [0, 32, 0], [0, 1, 0], [0, 2, 0], [16, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [0, 16, 0], [1, 0, 0], [2, 0, 0]], warp = [[4, 0, 0], [8, 0, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 32], [0, 1], [0, 2], [16, 0]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 32]], block = []}>
#linear9 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#linear10 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 1, 0]], block = []}>
#linear11 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 0, 1]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1, twoCTAs = true>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true, ctaMode = twocta_rhs>
#tmem3 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 32, colStride = 1, twoCTAs = true>
#tmem4 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 2, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @_attn_bwd_persist(%arg0: !tt.tensordesc<64x64xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<32x128xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<128x128xf16, #shared>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<128x64xf16, #shared>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.tensordesc<128x128xf16, #shared>, %arg21: i32, %arg22: i32, %arg23: i64, %arg24: i64, %arg25: f32, %arg26: !tt.tensordesc<64x64xf16, #shared>, %arg27: i32, %arg28: i32, %arg29: i64, %arg30: i64, %arg31: !tt.tensordesc<32x128xf16, #shared>, %arg32: i32, %arg33: i32, %arg34: i64, %arg35: i64, %arg36: !tt.tensordesc<32x32xf32, #shared1>, %arg37: i32, %arg38: i32, %arg39: i64, %arg40: i64, %arg41: !tt.tensordesc<128x64xf16, #shared>, %arg42: i32, %arg43: i32, %arg44: i64, %arg45: i64, %arg46: !tt.tensordesc<128x64xf16, #shared>, %arg47: i32, %arg48: i32, %arg49: i64, %arg50: i64, %arg51: !tt.tensordesc<64xf32, #shared2>, %arg52: i32, %arg53: i64, %arg54: !tt.tensordesc<64xf32, #shared2>, %arg55: i32, %arg56: i64, %arg57: i32 {tt.divisibility = 16 : i32}, %arg58: i32 {tt.divisibility = 16 : i32}, %arg59: i32 {tt.divisibility = 16 : i32}, %arg60: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
    %c0_i64 = arith.constant {async_task_id = array<i32: 0>} 0 : i64
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.693147182> : tensor<32x32xf32, #blocked>
    %c96_i32 = arith.constant {async_task_id = array<i32: 0>} 96 : i32
    %c2_i32 = arith.constant {async_task_id = array<i32: 0>} 2 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0>} 128 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0>} 127 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0>} 32 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0>} 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %2 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %3 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %4 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %5 = ttg.memdesc_index %4[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %6 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>
    %7 = ttg.memdesc_index %6[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %8 = ttg.memdesc_index %6[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %8, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %9 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %10 = ttg.memdesc_index %9[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %10, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %11 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %12 = ttg.memdesc_index %11[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %12, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %13 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %14 = ttg.memdesc_index %13[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %14, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %15 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %16 = ttg.memdesc_index %15[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %16, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %17 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>
    %18 = ttg.memdesc_index %17[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %18, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %19 = ttg.memdesc_index %17[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %19, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %20 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>
    %21 = ttg.memdesc_index %20[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %21, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %22 = ttg.memdesc_index %20[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %22, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %23 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %24 = ttg.memdesc_index %23[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %24, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %25 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>
    %26 = ttg.memdesc_index %25[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %26, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %27 = ttg.memdesc_index %25[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %27, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %28 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>
    %29 = ttg.memdesc_index %28[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %29, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %30 = ttg.memdesc_index %28[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %30, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %31 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>
    %32 = ttg.memdesc_index %31[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %32, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %33 = ttg.memdesc_index %31[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %33, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %34 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>
    %35 = ttg.memdesc_index %34[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %35, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %36 = ttg.memdesc_index %34[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %36, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %37 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>
    %38 = ttg.memdesc_index %37[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %38, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %39 = ttg.memdesc_index %37[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
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
    %53 = ttg.memdesc_index %52[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %53, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %54 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %55 = ttg.memdesc_index %54[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %55, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
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
    %64 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>
    %65 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>
    %66 = ttg.memdesc_index %64[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %66, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %67 = ttg.memdesc_index %65[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %67, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %68 = ttg.memdesc_index %64[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %68, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %69 = ttg.memdesc_index %65[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %69, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %70 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %71 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %72 = ttg.memdesc_index %70[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %72, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %73 = ttg.memdesc_index %71[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %73, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %74 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %75 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %76 = ttg.memdesc_index %74[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %76, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %77 = ttg.memdesc_index %75[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %77, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %78 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %79 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %80 = ttg.memdesc_index %78[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %80, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %81 = ttg.memdesc_index %79[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %81, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %82 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>
    %83 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>
    %84 = ttg.memdesc_index %82[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %84, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %85 = ttg.memdesc_index %83[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %85, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %86 = ttg.memdesc_index %82[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %86, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %87 = ttg.memdesc_index %83[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
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
    %96 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %97 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %98 = ttg.memdesc_index %96[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %98, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %99 = ttg.memdesc_index %97[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %99, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %100 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %101 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 13 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %102 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %result, %token = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_0, %token_1 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 10 : i32} : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %103 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 15 : i32} : () -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %104 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable>
    %105 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 17 : i32} : () -> !ttg.memdesc<2x64xf32, #shared2, #smem, mutable>
    %result_2, %token_3 = ttng.tmem_alloc {allocation.shareGroup = 1 : i32, buffer.copy = 1 : i32, buffer.id = 2 : i32} : () -> (!ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %106 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 18 : i32} : () -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %107 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>
    %result_4, %token_5 = ttng.tmem_alloc {allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> (!ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %108 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 20 : i32} : () -> !ttg.memdesc<2x64xf32, #shared2, #smem, mutable>
    %109 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %result_6, %token_7 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 11 : i32} : () -> (!ttg.memdesc<1x64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %110 = ttg.local_alloc {allocation.shareGroup = 3 : i32, buffer.copy = 2 : i32, buffer.id = 22 : i32, buffer.tmaStaging = 2 : i32} : () -> !ttg.memdesc<2x32x32xf32, #shared1, #smem, mutable>
    %111 = ttg.local_alloc {allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 26 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %112 = ttg.local_alloc {allocation.shareGroup = 2 : i32, buffer.copy = 1 : i32, buffer.id = 28 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    ttg.warp_specialize(%arg60, %54, %50, %46, %104, %31, %100, %result_2, %28, %23, %15, %107, %13, %102, %result_4, %11, %9, %4, %result, %106, %17, %20, %result_0, %103, %34, %37, %109, %101, %result_6, %2, %0, %40, %42, %44, %48, %52, %arg59, %arg57, %arg10, %arg15, %arg20, %arg0, %arg5, %25, %105, %arg51, %arg26, %arg31, %6, %108, %arg54, %arg25, %111, %arg46, %112, %arg41, %56, %57, %60, %61, %64, %65, %70, %71, %74, %75, %78, %79, %82, %83, %88, %89, %92, %93, %96, %97) attributes {requestedRegisters = array<i32: 24, 24, 192>, ttg.partition.types = ["reduction", "gemm", "load", "computation"]}
    default {
      %169 = arith.addi %arg60, %c127_i32 {async_task_id = array<i32: 0>} : i32
      %170 = arith.divsi %169, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %171 = tt.get_program_id x {async_task_id = array<i32: 0>} : i32
      %172 = arith.divsi %171, %c2_i32 {async_task_id = array<i32: 0>} : i32
      %173 = tt.get_num_programs x {async_task_id = array<i32: 0>} : i32
      %174 = arith.divsi %173, %c2_i32 {async_task_id = array<i32: 0>} : i32
      %175 = arith.divsi %170, %174 {async_task_id = array<i32: 0>} : i32
      %176 = arith.remsi %170, %174 {async_task_id = array<i32: 0>} : i32
      %177 = arith.cmpi slt, %172, %176 {async_task_id = array<i32: 0>} : i32
      %178 = scf.if %177 -> (i32) {
        %187 = arith.addi %175, %c1_i32 {async_task_id = array<i32: 0>} : i32
        scf.yield {async_task_id = array<i32: 0>} %187 : i32
      } else {
        scf.yield {async_task_id = array<i32: 0>} %175 : i32
      } {async_task_id = array<i32: 0>}
      %179 = arith.extsi %arg59 {async_task_id = array<i32: 0>} : i32 to i64
      %180 = arith.divsi %arg60, %c64_i32 {async_task_id = array<i32: 0>} : i32
      %181 = arith.remsi %171, %c2_i32 {async_task_id = array<i32: 0>} : i32
      %182 = arith.cmpi eq, %181, %c0_i32 {async_task_id = array<i32: 0>} : i32
      %183 = tt.splat %182 {async_task_id = array<i32: 0>} : i1 -> tensor<32x128xi1, #linear>
      %184 = arith.muli %181, %c32_i32 {async_task_id = array<i32: 0>} : i32
      %185 = arith.extsi %184 {async_task_id = array<i32: 0>} : i32 to i64
      %186:2 = scf.for %arg61 = %c0_i32 to %178 step %c1_i32 iter_args(%arg62 = %172, %arg63 = %c0_i64) -> (i32, i64)  : i32 {
        %187 = arith.divsi %arg62, %170 {async_task_id = array<i32: 0>} : i32
        %188 = arith.muli %arg57, %187 {async_task_id = array<i32: 0>} : i32
        %189 = arith.extsi %188 {async_task_id = array<i32: 0>} : i32 to i64
        %190 = arith.divsi %189, %179 {async_task_id = array<i32: 0>} : i64
        %191:2 = scf.for %arg64 = %c0_i32 to %180 step %c1_i32 iter_args(%arg65 = %c0_i32, %arg66 = %arg63) -> (i32, i64)  : i32 {
          %193 = arith.extsi %arg65 {async_task_id = array<i32: 0>} : i32 to i64
          %194 = arith.addi %190, %193 {async_task_id = array<i32: 0>} : i64
          %195 = arith.andi %arg66, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %196 = arith.trunci %195 {async_task_id = array<i32: 0>} : i64 to i1
          %197 = ttg.memdesc_index %result_6[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
          %198 = ttg.memdesc_index %0[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %199 = arith.extui %196 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %198, %199 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %result_8, %token_9 = ttng.tmem_load %197[] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear1>
          %200 = ttg.memdesc_index %97[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %200, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %201 = tt.reshape %result_8 {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear1> -> tensor<2x32x128xf32, #linear2>
          %202 = tt.trans %201 {async_task_id = array<i32: 0>, order = array<i32: 1, 2, 0>} : tensor<2x32x128xf32, #linear2> -> tensor<32x128x2xf32, #linear3>
          %203 = ttg.convert_layout %202 {async_task_id = array<i32: 0>} : tensor<32x128x2xf32, #linear3> -> tensor<32x128x2xf32, #linear4>
          %outLHS, %outRHS = tt.split %203 {async_task_id = array<i32: 0>} : tensor<32x128x2xf32, #linear4> -> tensor<32x128xf32, #linear>
          %204 = arith.select %183, %outLHS, %outRHS {async_task_id = array<i32: 0>} : tensor<32x128xi1, #linear>, tensor<32x128xf32, #linear>
          %205 = tt.reshape %204 {async_task_id = array<i32: 0>} : tensor<32x128xf32, #linear> -> tensor<32x2x64xf32, #linear5>
          %206 = tt.trans %205 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<32x2x64xf32, #linear5> -> tensor<32x64x2xf32, #linear6>
          %outLHS_10, %outRHS_11 = tt.split %206 {async_task_id = array<i32: 0>} : tensor<32x64x2xf32, #linear6> -> tensor<32x64xf32, #linear7>
          %207 = tt.reshape %outLHS_10 {async_task_id = array<i32: 0>} : tensor<32x64xf32, #linear7> -> tensor<32x2x32xf32, #blocked1>
          %208 = tt.trans %207 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<32x2x32xf32, #blocked1> -> tensor<32x32x2xf32, #blocked2>
          %outLHS_12, %outRHS_13 = tt.split %208 {async_task_id = array<i32: 0>} : tensor<32x32x2xf32, #blocked2> -> tensor<32x32xf32, #blocked>
          %209 = tt.reshape %outRHS_11 {async_task_id = array<i32: 0>} : tensor<32x64xf32, #linear7> -> tensor<32x2x32xf32, #blocked1>
          %210 = tt.trans %209 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<32x2x32xf32, #blocked1> -> tensor<32x32x2xf32, #blocked2>
          %outLHS_14, %outRHS_15 = tt.split %210 {async_task_id = array<i32: 0>} : tensor<32x32x2xf32, #blocked2> -> tensor<32x32xf32, #blocked>
          %211 = arith.addi %194, %185 {async_task_id = array<i32: 0>} : i64
          %212 = arith.mulf %outLHS_12, %cst {async_task_id = array<i32: 0>} : tensor<32x32xf32, #blocked>
          %213 = arith.trunci %211 {async_task_id = array<i32: 0>} : i64 to i32
          %214 = ttg.memdesc_index %110[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<2x32x32xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          ttg.local_store %212, %214 {async_task_id = array<i32: 0>} : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          %215 = ttg.memdesc_index %110[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<2x32x32xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          %216 = ttng.async_tma_reduce add, %arg36[%213, %c0_i32] %215 {async_task_id = array<i32: 0>} : !tt.tensordesc<32x32xf32, #shared1>, !ttg.memdesc<32x32xf32, #shared1, #smem, mutable> -> !ttg.async.token
          nvws.tma_store_wait %215   {async_task_id = array<i32: 0>} : !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          %217 = arith.mulf %outRHS_13, %cst {async_task_id = array<i32: 0>} : tensor<32x32xf32, #blocked>
          %218 = ttg.memdesc_index %110[%c1_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<2x32x32xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          ttg.local_store %217, %218 {async_task_id = array<i32: 0>} : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          %219 = ttg.memdesc_index %110[%c1_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<2x32x32xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          %220 = ttng.async_tma_reduce add, %arg36[%213, %c32_i32] %219 {async_task_id = array<i32: 0>} : !tt.tensordesc<32x32xf32, #shared1>, !ttg.memdesc<32x32xf32, #shared1, #smem, mutable> -> !ttg.async.token
          nvws.tma_store_wait %219   {async_task_id = array<i32: 0>} : !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          %221 = arith.mulf %outLHS_14, %cst {async_task_id = array<i32: 0>} : tensor<32x32xf32, #blocked>
          %222 = ttg.memdesc_index %110[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<2x32x32xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          ttg.local_store %221, %222 {async_task_id = array<i32: 0>} : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          %223 = ttg.memdesc_index %110[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<2x32x32xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          %224 = ttng.async_tma_reduce add, %arg36[%213, %c64_i32] %223 {async_task_id = array<i32: 0>} : !tt.tensordesc<32x32xf32, #shared1>, !ttg.memdesc<32x32xf32, #shared1, #smem, mutable> -> !ttg.async.token
          nvws.tma_store_wait %223   {async_task_id = array<i32: 0>} : !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          %225 = arith.mulf %outRHS_15, %cst {async_task_id = array<i32: 0>} : tensor<32x32xf32, #blocked>
          %226 = ttg.memdesc_index %110[%c1_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<2x32x32xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          ttg.local_store %225, %226 {async_task_id = array<i32: 0>} : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          %227 = ttg.memdesc_index %110[%c1_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<2x32x32xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          %228 = ttng.async_tma_reduce add, %arg36[%213, %c96_i32] %227 {async_task_id = array<i32: 0>} : !tt.tensordesc<32x32xf32, #shared1>, !ttg.memdesc<32x32xf32, #shared1, #smem, mutable> -> !ttg.async.token
          nvws.tma_store_wait %227   {async_task_id = array<i32: 0>} : !ttg.memdesc<32x32xf32, #shared1, #smem, mutable>
          %229 = arith.addi %arg65, %c64_i32 {async_task_id = array<i32: 0>} : i32
          %230 = arith.addi %arg66, %c1_i64 {async_task_id = array<i32: 0>} : i64
          scf.yield {async_task_id = array<i32: 0>} %229, %230 : i32, i64
        } {async_task_id = array<i32: 0>, tt.warp_specialize}
        %192 = arith.addi %arg62, %174 {async_task_id = array<i32: 0>} : i32
        scf.yield {async_task_id = array<i32: 0>} %192, %191#1 : i32, i64
      } {async_task_id = array<i32: 0>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_yield
    }
    partition0(%arg61: i32, %arg62: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg63: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg64: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg65: !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable>, %arg66: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg67: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg68: !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, %arg69: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg70: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg71: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg72: !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>, %arg73: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg74: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg75: !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, %arg76: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg77: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg78: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg79: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg80: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg81: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg82: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg83: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg84: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg85: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg86: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg87: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg88: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg89: !ttg.memdesc<1x64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, %arg90: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg91: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg92: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg93: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg94: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg95: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg96: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg97: i32, %arg98: i32, %arg99: !tt.tensordesc<128x128xf16, #shared>, %arg100: !tt.tensordesc<128x64xf16, #shared>, %arg101: !tt.tensordesc<128x128xf16, #shared>, %arg102: !tt.tensordesc<64x64xf16, #shared>, %arg103: !tt.tensordesc<32x128xf16, #shared>, %arg104: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg105: !ttg.memdesc<2x64xf32, #shared2, #smem, mutable>, %arg106: !tt.tensordesc<64xf32, #shared2>, %arg107: !tt.tensordesc<64x64xf16, #shared>, %arg108: !tt.tensordesc<32x128xf16, #shared>, %arg109: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg110: !ttg.memdesc<2x64xf32, #shared2, #smem, mutable>, %arg111: !tt.tensordesc<64xf32, #shared2>, %arg112: f32, %arg113: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg114: !tt.tensordesc<128x64xf16, #shared>, %arg115: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg116: !tt.tensordesc<128x64xf16, #shared>, %arg117: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg118: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg119: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg120: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg121: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg122: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg126: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg127: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg128: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg129: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg130: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg131: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg132: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg133: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg134: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg135: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg136: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) num_warps(1) {
      %169 = ub.poison : i1
      %c2_i64 = arith.constant {async_task_id = array<i32: 1>} 2 : i64
      %c1_i64_8 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
      %c0_i64_9 = arith.constant {async_task_id = array<i32: 1>} 0 : i64
      %true = arith.constant {async_task_id = array<i32: 1>} true
      %c64_i32_10 = arith.constant {async_task_id = array<i32: 1>} 64 : i32
      %c127_i32_11 = arith.constant {async_task_id = array<i32: 1>} 127 : i32
      %c128_i32_12 = arith.constant {async_task_id = array<i32: 1>} 128 : i32
      %c2_i32_13 = arith.constant {async_task_id = array<i32: 1>} 2 : i32
      %c1_i32_14 = arith.constant {async_task_id = array<i32: 1>} 1 : i32
      %c0_i32_15 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
      %false = arith.constant {async_task_id = array<i32: 1>} false
      %170 = arith.addi %arg61, %c127_i32_11 {async_task_id = array<i32: 1>} : i32
      %171 = arith.divsi %170, %c128_i32_12 {async_task_id = array<i32: 1>} : i32
      %172 = tt.get_program_id x {async_task_id = array<i32: 1>} : i32
      %173 = arith.divsi %172, %c2_i32_13 {async_task_id = array<i32: 1>} : i32
      %174 = tt.get_num_programs x {async_task_id = array<i32: 1>} : i32
      %175 = arith.divsi %174, %c2_i32_13 {async_task_id = array<i32: 1>} : i32
      %176 = arith.divsi %171, %175 {async_task_id = array<i32: 1>} : i32
      %177 = arith.remsi %171, %175 {async_task_id = array<i32: 1>} : i32
      %178 = arith.cmpi slt, %173, %177 {async_task_id = array<i32: 1>} : i32
      %179 = scf.if %178 -> (i32) {
        %182 = arith.addi %176, %c1_i32_14 {async_task_id = array<i32: 1>} : i32
        scf.yield {async_task_id = array<i32: 1>} %182 : i32
      } else {
        scf.yield {async_task_id = array<i32: 1>} %176 : i32
      } {async_task_id = array<i32: 1>}
      %180 = arith.divsi %arg61, %c64_i32_10 {async_task_id = array<i32: 1>} : i32
      %181:2 = scf.for %arg137 = %c0_i32_15 to %179 step %c1_i32_14 iter_args(%arg138 = %c0_i64_9, %arg139 = %c0_i64_9) -> (i64, i64)  : i32 {
        %182 = arith.andi %arg138, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        %183 = arith.trunci %182 {async_task_id = array<i32: 1>} : i64 to i1
        %184 = arith.andi %arg138, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        %185 = arith.trunci %184 {async_task_id = array<i32: 1>} : i64 to i1
        %186 = arith.andi %arg138, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        %187 = arith.trunci %186 {async_task_id = array<i32: 1>} : i64 to i1
        %188 = ttg.memdesc_index %arg62[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %189 = arith.extui %183 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %188, %189, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %190 = ttg.memdesc_index %arg63[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %191 = arith.extui %185 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %190, %191, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %192 = ttg.memdesc_index %arg64[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %193 = arith.extui %187 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %192, %193, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %194 = arith.andi %arg138, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        %195 = arith.trunci %194 {async_task_id = array<i32: 1>} : i64 to i1
        %196 = ttg.memdesc_index %arg118[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %197 = arith.xori %195, %true : i1
        %198 = arith.extui %197 : i1 to i32
        ttng.wait_barrier %196, %198 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %199 = arith.andi %arg138, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        %200 = arith.trunci %199 {async_task_id = array<i32: 1>} : i64 to i1
        %201 = ttg.memdesc_index %arg120[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %202 = arith.xori %200, %true : i1
        %203 = arith.extui %202 : i1 to i32
        ttng.wait_barrier %201, %203 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %204 = arith.cmpi sgt, %180, %c0_i32_15 : i32
        %205 = arith.divui %arg139, %c2_i64 {async_task_id = array<i32: 1>} : i64
        %206 = arith.muli %205, %c2_i64 {async_task_id = array<i32: 1>} : i64
        %207 = arith.subi %arg139, %206 {async_task_id = array<i32: 1>} : i64
        %208 = arith.trunci %207 {async_task_id = array<i32: 1>} : i64 to i32
        %209 = arith.andi %205, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        %210 = arith.trunci %209 {async_task_id = array<i32: 1>} : i64 to i1
        %211 = arith.divui %arg139, %c2_i64 {async_task_id = array<i32: 1>} : i64
        %212 = arith.muli %211, %c2_i64 {async_task_id = array<i32: 1>} : i64
        %213 = arith.subi %arg139, %212 {async_task_id = array<i32: 1>} : i64
        %214 = arith.trunci %213 {async_task_id = array<i32: 1>} : i64 to i32
        %215 = ttg.memdesc_index %arg65[%214] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
        %216 = ttg.memdesc_index %arg66[%208] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %217 = arith.extui %210 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %216, %217, %204 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %218 = ttg.memdesc_trans %215 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared4, #smem, mutable>
        %219 = ttg.memdesc_index %arg67[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %220 = ttg.memdesc_index %arg68[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %221 = ttg.memdesc_index %arg69[%208] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %222 = arith.andi %arg139, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        %223 = arith.trunci %222 {async_task_id = array<i32: 1>} : i64 to i1
        %224 = ttg.memdesc_index %arg70[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %225 = ttg.memdesc_index %arg124[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %226 = arith.xori %223, %true : i1
        %227 = arith.extui %226 : i1 to i32
        ttng.wait_barrier %225, %227, %204 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %228 = arith.andi %arg139, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        %229 = arith.trunci %228 {async_task_id = array<i32: 1>} : i64 to i1
        %230 = ttg.memdesc_index %arg71[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %231 = arith.xori %229, %true {async_task_id = array<i32: 1>} : i1
        %232 = arith.extui %231 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %230, %232, %204 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {direction = "backward", dstTask = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %233 = ttng.tc_gen5_mma %219, %218, %220[], %false, %204, %221[%true], %224[%true] {async_task_id = array<i32: 1>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x32xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %234 = arith.divui %arg139, %c2_i64 {async_task_id = array<i32: 1>} : i64
        %235 = arith.muli %234, %c2_i64 {async_task_id = array<i32: 1>} : i64
        %236 = arith.subi %arg139, %235 {async_task_id = array<i32: 1>} : i64
        %237 = arith.trunci %236 {async_task_id = array<i32: 1>} : i64 to i32
        %238 = arith.andi %234, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        %239 = arith.trunci %238 {async_task_id = array<i32: 1>} : i64 to i1
        %240 = arith.andi %arg139, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        %241 = arith.trunci %240 {async_task_id = array<i32: 1>} : i64 to i1
        %242 = ttg.memdesc_index %arg72[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
        %243 = ttg.memdesc_index %arg73[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %244 = arith.extui %241 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %243, %244, %204 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %245 = ttg.memdesc_trans %242 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared4, #smem, mutable>
        %246 = ttg.memdesc_index %arg74[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %247 = ttg.memdesc_index %arg75[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %248 = ttg.memdesc_index %arg76[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %249 = arith.andi %arg139, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        %250 = arith.trunci %249 {async_task_id = array<i32: 1>} : i64 to i1
        %251 = ttg.memdesc_index %arg77[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %252 = ttg.memdesc_index %arg128[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %253 = arith.xori %250, %true : i1
        %254 = arith.extui %253 : i1 to i32
        ttng.wait_barrier %252, %254, %204 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %255 = arith.andi %arg139, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        %256 = arith.trunci %255 {async_task_id = array<i32: 1>} : i64 to i1
        %257 = ttg.memdesc_index %arg78[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %258 = arith.xori %256, %true {async_task_id = array<i32: 1>} : i1
        %259 = arith.extui %258 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %257, %259, %204 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {direction = "backward", dstTask = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %260 = ttng.tc_gen5_mma %246, %245, %247[], %false, %204, %248[%true], %251[%true] {async_task_id = array<i32: 1>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x32xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %261 = ttg.memdesc_index %arg79[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %262 = arith.divui %arg139, %c2_i64 {async_task_id = array<i32: 1>} : i64
        %263 = arith.muli %262, %c2_i64 {async_task_id = array<i32: 1>} : i64
        %264 = arith.subi %arg139, %263 {async_task_id = array<i32: 1>} : i64
        %265 = arith.trunci %264 {async_task_id = array<i32: 1>} : i64 to i32
        %266 = ttg.memdesc_index %arg80[%265] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
        %267 = ttng.tmem_subslice %arg68 {N = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64>
        %268 = ttg.memdesc_reinterpret %267 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64> -> !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable>
        %269 = ttg.memdesc_index %268[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
        %270 = ttg.memdesc_index %arg81[%237] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %271 = ttg.memdesc_index %arg82[%237] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %272 = arith.extui %239 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %271, %272, %204 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %273 = ttg.memdesc_index %arg71[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %274 = ttg.memdesc_index %arg125[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %275 = arith.extui %229 : i1 to i32
        ttng.wait_barrier %274, %275, %204 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %276 = ttng.tc_gen5_mma %269, %266, %261[], %false, %204, %270[%true], %273[%true] {async_task_id = array<i32: 1>, is_async, tmem.start = array<i32: 3>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %277 = arith.subi %180, %c1_i32_14 : i32
        %278:3 = scf.for %arg140 = %c0_i32_15 to %277 step %c1_i32_14 iter_args(%arg141 = %false, %arg142 = %arg139, %arg143 = %256) -> (i1, i64, i1)  : i32 {
          %287 = arith.addi %arg142, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %288 = arith.divui %287, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %289 = arith.muli %288, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %290 = arith.subi %287, %289 {async_task_id = array<i32: 1>} : i64
          %291 = arith.trunci %290 {async_task_id = array<i32: 1>} : i64 to i32
          %292 = arith.andi %288, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %293 = arith.trunci %292 {async_task_id = array<i32: 1>} : i64 to i1
          %294 = arith.divui %287, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %295 = arith.muli %294, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %296 = arith.subi %287, %295 {async_task_id = array<i32: 1>} : i64
          %297 = arith.trunci %296 {async_task_id = array<i32: 1>} : i64 to i32
          %298 = ttg.memdesc_index %arg65[%297] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
          %299 = ttg.memdesc_index %arg66[%291] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %300 = arith.extui %293 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %299, %300, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %301 = ttg.memdesc_trans %298 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared4, #smem, mutable>
          %302 = ttg.memdesc_index %arg67[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
          %303 = ttg.memdesc_index %arg68[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
          %304 = ttg.memdesc_index %arg69[%291] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %305 = arith.andi %287, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %306 = arith.trunci %305 {async_task_id = array<i32: 1>} : i64 to i1
          %307 = ttg.memdesc_index %arg70[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %308 = ttg.memdesc_index %arg124[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %309 = arith.xori %306, %true : i1
          %310 = arith.extui %309 : i1 to i32
          ttng.wait_barrier %308, %310, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %311 = arith.andi %287, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %312 = arith.trunci %311 {async_task_id = array<i32: 1>} : i64 to i1
          %313 = ttg.memdesc_index %arg71[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %314 = arith.xori %312, %true {async_task_id = array<i32: 1>} : i1
          %315 = arith.extui %314 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %313, %315, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {direction = "backward", dstTask = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %316 = ttng.tc_gen5_mma %302, %301, %303[], %false, %true, %304[%true], %307[%true] {async_task_id = array<i32: 1>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x32xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %317 = arith.divui %arg142, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %318 = arith.muli %317, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %319 = arith.subi %arg142, %318 {async_task_id = array<i32: 1>} : i64
          %320 = arith.trunci %319 {async_task_id = array<i32: 1>} : i64 to i32
          %321 = arith.andi %317, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %322 = arith.trunci %321 {async_task_id = array<i32: 1>} : i64 to i1
          %323 = ttg.memdesc_index %arg83[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %324 = arith.divui %arg142, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %325 = arith.muli %324, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %326 = arith.subi %arg142, %325 {async_task_id = array<i32: 1>} : i64
          %327 = arith.trunci %326 {async_task_id = array<i32: 1>} : i64 to i32
          %328 = ttg.memdesc_index %arg84[%327] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
          %329 = ttng.tmem_subslice %arg75 {N = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64>
          %330 = ttg.memdesc_reinterpret %329 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64> -> !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable>
          %331 = ttg.memdesc_index %330[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
          %332 = ttg.memdesc_index %arg85[%320] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %333 = ttg.memdesc_index %arg86[%320] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %334 = arith.extui %322 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %333, %334, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %335 = ttg.memdesc_index %arg78[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %336 = ttg.memdesc_index %arg131[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %337 = arith.extui %arg143 : i1 to i32
          ttng.wait_barrier %336, %337 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %338 = ttng.tc_gen5_mma %331, %328, %323[], %arg141, %true, %332[%true], %335[%true] {async_task_id = array<i32: 1>, is_async, tmem.start = array<i32: 4>, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,tmem,1,5\22, \22opndD,tmem,1,10\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %339 = arith.andi %arg142, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %340 = arith.trunci %339 {async_task_id = array<i32: 1>} : i64 to i1
          %341 = ttg.memdesc_index %arg87[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %342 = ttg.memdesc_index %arg133[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %343 = arith.extui %340 : i1 to i32
          ttng.wait_barrier %342, %343 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %344 = ttg.memdesc_trans %341 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared4, #smem, mutable>
          %345 = ttg.memdesc_index %arg88[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %346 = ttg.memdesc_index %arg89[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
          %347 = ttg.memdesc_index %arg90[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %348 = arith.andi %arg142, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %349 = arith.trunci %348 {async_task_id = array<i32: 1>} : i64 to i1
          %350 = ttg.memdesc_index %arg91[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %351 = ttg.memdesc_index %arg136[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %352 = arith.xori %349, %true : i1
          %353 = arith.extui %352 : i1 to i32
          ttng.wait_barrier %351, %353 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %354 = ttng.tc_gen5_mma %344, %345, %346[], %false, %true, %347[%true], %350[%true] {async_task_id = array<i32: 1>, is_async, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,11\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x128xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %355 = arith.divui %287, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %356 = arith.muli %355, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %357 = arith.subi %287, %356 {async_task_id = array<i32: 1>} : i64
          %358 = arith.trunci %357 {async_task_id = array<i32: 1>} : i64 to i32
          %359 = arith.andi %355, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %360 = arith.trunci %359 {async_task_id = array<i32: 1>} : i64 to i1
          %361 = arith.andi %287, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %362 = arith.trunci %361 {async_task_id = array<i32: 1>} : i64 to i1
          %363 = ttg.memdesc_index %arg72[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
          %364 = ttg.memdesc_index %arg73[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %365 = arith.extui %362 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %364, %365, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %366 = ttg.memdesc_trans %363 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared4, #smem, mutable>
          %367 = ttg.memdesc_index %arg74[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
          %368 = ttg.memdesc_index %arg75[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
          %369 = ttg.memdesc_index %arg76[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %370 = arith.andi %287, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %371 = arith.trunci %370 {async_task_id = array<i32: 1>} : i64 to i1
          %372 = ttg.memdesc_index %arg77[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %373 = ttg.memdesc_index %arg128[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %374 = arith.xori %371, %true : i1
          %375 = arith.extui %374 : i1 to i32
          ttng.wait_barrier %373, %375, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %376 = arith.andi %287, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %377 = arith.trunci %376 {async_task_id = array<i32: 1>} : i64 to i1
          %378 = ttg.memdesc_index %arg78[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %379 = arith.xori %377, %true {async_task_id = array<i32: 1>} : i1
          %380 = arith.extui %379 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %378, %380, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {direction = "backward", dstTask = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %381 = ttng.tc_gen5_mma %367, %366, %368[], %false, %true, %369[%true], %372[%true] {async_task_id = array<i32: 1>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x32xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %382 = ttg.memdesc_index %arg79[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %383 = arith.divui %287, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %384 = arith.muli %383, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %385 = arith.subi %287, %384 {async_task_id = array<i32: 1>} : i64
          %386 = arith.trunci %385 {async_task_id = array<i32: 1>} : i64 to i32
          %387 = ttg.memdesc_index %arg80[%386] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
          %388 = ttng.tmem_subslice %arg68 {N = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64>
          %389 = ttg.memdesc_reinterpret %388 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64> -> !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable>
          %390 = ttg.memdesc_index %389[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
          %391 = ttg.memdesc_index %arg81[%358] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %392 = ttg.memdesc_index %arg82[%358] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %393 = arith.extui %360 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %392, %393, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %394 = ttg.memdesc_index %arg71[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %395 = ttg.memdesc_index %arg125[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %396 = arith.extui %312 : i1 to i32
          ttng.wait_barrier %395, %396, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %397 = ttng.tc_gen5_mma %390, %387, %382[], %true, %true, %391[%true], %394[%true] {async_task_id = array<i32: 1>, is_async, tmem.start = array<i32: 3>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          scf.yield %true, %287, %377 : i1, i64, i1
        } {async_task_id = array<i32: 1>, tt.warp_specialize}
        %279 = arith.cmpi sgt, %180, %c0_i32_15 : i32
        %280:3 = scf.if %279 -> (i1, i64, i1) {
          %287 = arith.addi %278#1, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %288 = arith.divui %278#1, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %289 = arith.muli %288, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %290 = arith.subi %278#1, %289 {async_task_id = array<i32: 1>} : i64
          %291 = arith.trunci %290 {async_task_id = array<i32: 1>} : i64 to i32
          %292 = arith.andi %288, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %293 = arith.trunci %292 {async_task_id = array<i32: 1>} : i64 to i1
          %294 = ttg.memdesc_index %arg83[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %295 = arith.divui %278#1, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %296 = arith.muli %295, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %297 = arith.subi %278#1, %296 {async_task_id = array<i32: 1>} : i64
          %298 = arith.trunci %297 {async_task_id = array<i32: 1>} : i64 to i32
          %299 = ttg.memdesc_index %arg84[%298] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
          %300 = ttng.tmem_subslice %arg75 {N = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64>
          %301 = ttg.memdesc_reinterpret %300 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64> -> !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable>
          %302 = ttg.memdesc_index %301[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
          %303 = ttg.memdesc_index %arg85[%291] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %304 = ttg.memdesc_index %arg86[%291] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %305 = arith.extui %293 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %304, %305, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %306 = ttg.memdesc_index %arg78[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %307 = ttg.memdesc_index %arg131[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %308 = arith.extui %278#2 : i1 to i32
          ttng.wait_barrier %307, %308 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %309 = ttng.tc_gen5_mma %302, %299, %294[], %278#0, %true, %303[%true], %306[%true] {async_task_id = array<i32: 1>, is_async, tmem.start = array<i32: 4>, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,tmem,1,5\22, \22opndD,tmem,1,10\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %310 = arith.andi %278#1, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %311 = arith.trunci %310 {async_task_id = array<i32: 1>} : i64 to i1
          %312 = ttg.memdesc_index %arg87[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %313 = ttg.memdesc_index %arg133[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %314 = arith.extui %311 : i1 to i32
          ttng.wait_barrier %313, %314 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3>, dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %315 = ttg.memdesc_trans %312 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared4, #smem, mutable>
          %316 = ttg.memdesc_index %arg88[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %317 = ttg.memdesc_index %arg89[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
          %318 = ttg.memdesc_index %arg90[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %319 = arith.andi %278#1, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
          %320 = arith.trunci %319 {async_task_id = array<i32: 1>} : i64 to i1
          %321 = ttg.memdesc_index %arg91[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %322 = ttg.memdesc_index %arg136[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %323 = arith.xori %320, %true : i1
          %324 = arith.extui %323 : i1 to i32
          ttng.wait_barrier %322, %324 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %325 = ttng.tc_gen5_mma %315, %316, %317[], %false, %true, %318[%true], %321[%true] {async_task_id = array<i32: 1>, is_async, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,11\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x128xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          scf.yield %true, %287, %169 : i1, i64, i1
        } else {
          scf.yield %278#0, %278#1, %278#2 : i1, i64, i1
        }
        %281 = ttg.memdesc_index %arg92[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_commit %281 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %282 = ttg.memdesc_index %arg93[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_commit %282 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %283 = ttg.memdesc_index %arg94[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_commit %283 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %284 = ttg.memdesc_index %arg95[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_commit %284 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %285 = ttg.memdesc_index %arg96[%c0_i32_15] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_commit %285 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %286 = arith.addi %arg138, %c1_i64_8 {async_task_id = array<i32: 1>} : i64
        scf.yield {async_task_id = array<i32: 1>} %286, %280#1 : i64, i64
      } {async_task_id = array<i32: 1>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition1(%arg61: i32, %arg62: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg63: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg64: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg65: !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable>, %arg66: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg67: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg68: !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, %arg69: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg70: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg71: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg72: !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>, %arg73: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg74: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg75: !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, %arg76: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg77: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg78: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg79: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg80: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg81: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg82: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg83: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg84: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg85: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg86: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg87: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg88: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg89: !ttg.memdesc<1x64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, %arg90: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg91: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg92: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg93: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg94: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg95: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg96: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg97: i32, %arg98: i32, %arg99: !tt.tensordesc<128x128xf16, #shared>, %arg100: !tt.tensordesc<128x64xf16, #shared>, %arg101: !tt.tensordesc<128x128xf16, #shared>, %arg102: !tt.tensordesc<64x64xf16, #shared>, %arg103: !tt.tensordesc<32x128xf16, #shared>, %arg104: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg105: !ttg.memdesc<2x64xf32, #shared2, #smem, mutable>, %arg106: !tt.tensordesc<64xf32, #shared2>, %arg107: !tt.tensordesc<64x64xf16, #shared>, %arg108: !tt.tensordesc<32x128xf16, #shared>, %arg109: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg110: !ttg.memdesc<2x64xf32, #shared2, #smem, mutable>, %arg111: !tt.tensordesc<64xf32, #shared2>, %arg112: f32, %arg113: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg114: !tt.tensordesc<128x64xf16, #shared>, %arg115: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg116: !tt.tensordesc<128x64xf16, #shared>, %arg117: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg118: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg119: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg120: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg121: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg122: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg126: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg127: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg128: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg129: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg130: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg131: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg132: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg133: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg134: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg135: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg136: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) num_warps(1) {
      %c2_i64 = arith.constant {async_task_id = array<i32: 2>} 2 : i64
      %true = arith.constant {async_task_id = array<i32: 2>} true
      %c1_i64_8 = arith.constant {async_task_id = array<i32: 2>} 1 : i64
      %c0_i64_9 = arith.constant {async_task_id = array<i32: 2>} 0 : i64
      %c64_i32_10 = arith.constant {async_task_id = array<i32: 2>} 64 : i32
      %c32_i32_11 = arith.constant {async_task_id = array<i32: 2>} 32 : i32
      %c127_i32_12 = arith.constant {async_task_id = array<i32: 2>} 127 : i32
      %c128_i32_13 = arith.constant {async_task_id = array<i32: 2>} 128 : i32
      %c2_i32_14 = arith.constant {async_task_id = array<i32: 2>} 2 : i32
      %c1_i32_15 = arith.constant {async_task_id = array<i32: 2>} 1 : i32
      %c0_i32_16 = arith.constant {async_task_id = array<i32: 2>} 0 : i32
      %169 = arith.addi %arg61, %c127_i32_12 {async_task_id = array<i32: 2>} : i32
      %170 = arith.divsi %169, %c128_i32_13 {async_task_id = array<i32: 2>} : i32
      %171 = tt.get_program_id x {async_task_id = array<i32: 2>} : i32
      %172 = arith.divsi %171, %c2_i32_14 {async_task_id = array<i32: 2>} : i32
      %173 = tt.get_num_programs x {async_task_id = array<i32: 2>} : i32
      %174 = arith.divsi %173, %c2_i32_14 {async_task_id = array<i32: 2>} : i32
      %175 = arith.divsi %170, %174 {async_task_id = array<i32: 2>} : i32
      %176 = arith.remsi %170, %174 {async_task_id = array<i32: 2>} : i32
      %177 = arith.cmpi slt, %172, %176 {async_task_id = array<i32: 2>} : i32
      %178 = scf.if %177 -> (i32) {
        %186 = arith.addi %175, %c1_i32_15 {async_task_id = array<i32: 2>} : i32
        scf.yield {async_task_id = array<i32: 2>} %186 : i32
      } else {
        scf.yield {async_task_id = array<i32: 2>} %175 : i32
      } {async_task_id = array<i32: 2>}
      %179 = arith.extsi %arg97 {async_task_id = array<i32: 2>} : i32 to i64
      %180 = arith.divsi %arg61, %c64_i32_10 {async_task_id = array<i32: 2>} : i32
      %181 = nvg.cluster_id {async_task_id = array<i32: 2>}
      %182 = arith.remsi %181, %c2_i32_14 {async_task_id = array<i32: 2>} : i32
      %183 = arith.muli %182, %c64_i32_10 {async_task_id = array<i32: 2>} : i32
      %184 = arith.muli %182, %c32_i32_11 {async_task_id = array<i32: 2>} : i32
      %185:3 = scf.for %arg137 = %c0_i32_16 to %178 step %c1_i32_15 iter_args(%arg138 = %172, %arg139 = %c0_i64_9, %arg140 = %c0_i64_9) -> (i32, i64, i64)  : i32 {
        %186 = arith.remsi %arg138, %170 {async_task_id = array<i32: 2>} : i32
        %187 = arith.divsi %arg138, %170 {async_task_id = array<i32: 2>} : i32
        %188 = arith.muli %187, %arg61 {async_task_id = array<i32: 2>} : i32
        %189 = arith.extsi %188 {async_task_id = array<i32: 2>} : i32 to i64
        %190 = arith.muli %arg98, %187 {async_task_id = array<i32: 2>} : i32
        %191 = arith.extsi %190 {async_task_id = array<i32: 2>} : i32 to i64
        %192 = arith.divsi %191, %179 {async_task_id = array<i32: 2>} : i64
        %193 = arith.muli %186, %c128_i32_13 {async_task_id = array<i32: 2>} : i32
        %194 = arith.extsi %193 {async_task_id = array<i32: 2>} : i32 to i64
        %195 = arith.addi %192, %194 {async_task_id = array<i32: 2>} : i64
        %196 = arith.trunci %195 {async_task_id = array<i32: 2>} : i64 to i32
        %197 = arith.andi %arg139, %c1_i64_8 {async_task_id = array<i32: 2>} : i64
        %198 = arith.trunci %197 {async_task_id = array<i32: 2>} : i64 to i1
        %199 = ttg.memdesc_index %arg96[%c0_i32_16] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %200 = arith.xori %198, %true {async_task_id = array<i32: 2>} : i1
        %201 = arith.extui %200 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %199, %201 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "backward", dstTask = 1 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %202 = ttg.memdesc_index %arg62[%c0_i32_16] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.barrier_expect %202, 32768 {async_task_id = array<i32: 2>}, %true : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %203 = ttg.memdesc_index %arg67[%c0_i32_16] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %arg99[%196, %c0_i32_16] %203, %202, %true {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %204 = arith.andi %arg139, %c1_i64_8 {async_task_id = array<i32: 2>} : i64
        %205 = arith.trunci %204 {async_task_id = array<i32: 2>} : i64 to i1
        %206 = ttg.memdesc_index %arg95[%c0_i32_16] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %207 = arith.xori %205, %true {async_task_id = array<i32: 2>} : i1
        %208 = arith.extui %207 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %206, %208 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "backward", dstTask = 1 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %209 = ttg.memdesc_index %arg63[%c0_i32_16] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.barrier_expect %209, 16384 {async_task_id = array<i32: 2>}, %true : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %210 = ttg.memdesc_index %arg88[%c0_i32_16] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %arg100[%196, %183] %210, %209, %true {async_task_id = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %211 = arith.andi %arg139, %c1_i64_8 {async_task_id = array<i32: 2>} : i64
        %212 = arith.trunci %211 {async_task_id = array<i32: 2>} : i64 to i1
        %213 = ttg.memdesc_index %arg94[%c0_i32_16] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %214 = arith.xori %212, %true {async_task_id = array<i32: 2>} : i1
        %215 = arith.extui %214 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %213, %215 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "backward", dstTask = 1 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %216 = ttg.memdesc_index %arg64[%c0_i32_16] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.barrier_expect %216, 32768 {async_task_id = array<i32: 2>}, %true : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %217 = ttg.memdesc_index %arg74[%c0_i32_16] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %arg101[%196, %c0_i32_16] %217, %216, %true {async_task_id = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %218:2 = scf.for %arg141 = %c0_i32_16 to %180 step %c1_i32_15 iter_args(%arg142 = %c0_i32_16, %arg143 = %arg140) -> (i32, i64)  : i32 {
          %221 = arith.extsi %arg142 {async_task_id = array<i32: 2>} : i32 to i64
          %222 = arith.addi %192, %221 {async_task_id = array<i32: 2>} : i64
          %223 = arith.trunci %222 {async_task_id = array<i32: 2>} : i64 to i32
          %224 = arith.divui %arg143, %c2_i64 {async_task_id = array<i32: 2>} : i64
          %225 = arith.muli %224, %c2_i64 {async_task_id = array<i32: 2>} : i64
          %226 = arith.subi %arg143, %225 {async_task_id = array<i32: 2>} : i64
          %227 = arith.trunci %226 {async_task_id = array<i32: 2>} : i64 to i32
          %228 = arith.andi %224, %c1_i64_8 {async_task_id = array<i32: 2>} : i64
          %229 = arith.trunci %228 {async_task_id = array<i32: 2>} : i64 to i1
          %230 = ttg.memdesc_index %arg85[%227] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %231 = arith.xori %229, %true {async_task_id = array<i32: 2>} : i1
          %232 = arith.extui %231 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %230, %232 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %233 = ttg.memdesc_index %arg86[%227] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.barrier_expect %233, 8192 {async_task_id = array<i32: 2>}, %true : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %234 = ttg.memdesc_index %arg84[%227] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg102[%223, %183] %234, %233, %true {async_task_id = array<i32: 2>} : !tt.tensordesc<64x64xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
          %235 = arith.addi %223, %184 {async_task_id = array<i32: 2>} : i32
          %236 = arith.divui %arg143, %c2_i64 {async_task_id = array<i32: 2>} : i64
          %237 = arith.muli %236, %c2_i64 {async_task_id = array<i32: 2>} : i64
          %238 = arith.subi %arg143, %237 {async_task_id = array<i32: 2>} : i64
          %239 = arith.trunci %238 {async_task_id = array<i32: 2>} : i64 to i32
          %240 = arith.andi %236, %c1_i64_8 {async_task_id = array<i32: 2>} : i64
          %241 = arith.trunci %240 {async_task_id = array<i32: 2>} : i64 to i1
          %242 = ttg.memdesc_index %arg69[%239] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %243 = arith.xori %241, %true {async_task_id = array<i32: 2>} : i1
          %244 = arith.extui %243 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %242, %244 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %245 = ttg.memdesc_index %arg66[%239] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.barrier_expect %245, 8192 {async_task_id = array<i32: 2>}, %true : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %246 = ttg.memdesc_index %arg65[%239] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg103[%235, %c0_i32_16] %246, %245, %true {async_task_id = array<i32: 2>} : !tt.tensordesc<32x128xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
          %247 = arith.addi %189, %221 {async_task_id = array<i32: 2>} : i64
          %248 = arith.trunci %247 {async_task_id = array<i32: 2>} : i64 to i32
          %249 = arith.divui %arg143, %c2_i64 {async_task_id = array<i32: 2>} : i64
          %250 = arith.muli %249, %c2_i64 {async_task_id = array<i32: 2>} : i64
          %251 = arith.subi %arg143, %250 {async_task_id = array<i32: 2>} : i64
          %252 = arith.trunci %251 {async_task_id = array<i32: 2>} : i64 to i32
          %253 = arith.andi %249, %c1_i64_8 {async_task_id = array<i32: 2>} : i64
          %254 = arith.trunci %253 {async_task_id = array<i32: 2>} : i64 to i1
          %255 = ttg.memdesc_index %arg122[%252] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %256 = arith.xori %254, %true : i1
          %257 = arith.extui %256 : i1 to i32
          ttng.wait_barrier %255, %257 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %258 = ttg.memdesc_index %arg104[%252] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.barrier_expect %258, 256 {async_task_id = array<i32: 2>}, %true : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %259 = ttg.memdesc_index %arg105[%252] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x64xf32, #shared2, #smem, mutable> -> !ttg.memdesc<64xf32, #shared2, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg106[%248] %259, %258, %true {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared2>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64xf32, #shared2, #smem, mutable>
          %260 = arith.divui %arg143, %c2_i64 {async_task_id = array<i32: 2>} : i64
          %261 = arith.muli %260, %c2_i64 {async_task_id = array<i32: 2>} : i64
          %262 = arith.subi %arg143, %261 {async_task_id = array<i32: 2>} : i64
          %263 = arith.trunci %262 {async_task_id = array<i32: 2>} : i64 to i32
          %264 = arith.andi %260, %c1_i64_8 {async_task_id = array<i32: 2>} : i64
          %265 = arith.trunci %264 {async_task_id = array<i32: 2>} : i64 to i1
          %266 = ttg.memdesc_index %arg81[%263] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %267 = arith.xori %265, %true {async_task_id = array<i32: 2>} : i1
          %268 = arith.extui %267 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %266, %268 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %269 = ttg.memdesc_index %arg82[%263] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.barrier_expect %269, 8192 {async_task_id = array<i32: 2>}, %true : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %270 = ttg.memdesc_index %arg80[%263] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg107[%223, %183] %270, %269, %true {async_task_id = array<i32: 2>} : !tt.tensordesc<64x64xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
          %271 = arith.andi %arg143, %c1_i64_8 {async_task_id = array<i32: 2>} : i64
          %272 = arith.trunci %271 {async_task_id = array<i32: 2>} : i64 to i1
          %273 = ttg.memdesc_index %arg76[%c0_i32_16] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %274 = arith.xori %272, %true {async_task_id = array<i32: 2>} : i1
          %275 = arith.extui %274 {async_task_id = array<i32: 2>} : i1 to i32
          ttng.wait_barrier %273, %275 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %276 = ttg.memdesc_index %arg73[%c0_i32_16] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.barrier_expect %276, 8192 {async_task_id = array<i32: 2>}, %true : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %277 = ttg.memdesc_index %arg72[%c0_i32_16] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg108[%235, %c0_i32_16] %277, %276, %true {async_task_id = array<i32: 2>} : !tt.tensordesc<32x128xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
          %278 = arith.divui %arg143, %c2_i64 {async_task_id = array<i32: 2>} : i64
          %279 = arith.muli %278, %c2_i64 {async_task_id = array<i32: 2>} : i64
          %280 = arith.subi %arg143, %279 {async_task_id = array<i32: 2>} : i64
          %281 = arith.trunci %280 {async_task_id = array<i32: 2>} : i64 to i32
          %282 = arith.andi %278, %c1_i64_8 {async_task_id = array<i32: 2>} : i64
          %283 = arith.trunci %282 {async_task_id = array<i32: 2>} : i64 to i1
          %284 = ttg.memdesc_index %arg130[%281] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %285 = arith.xori %283, %true : i1
          %286 = arith.extui %285 : i1 to i32
          ttng.wait_barrier %284, %286 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %287 = ttg.memdesc_index %arg109[%281] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.barrier_expect %287, 256 {async_task_id = array<i32: 2>}, %true : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %288 = ttg.memdesc_index %arg110[%281] {async_task_id = array<i32: 2>} : !ttg.memdesc<2x64xf32, #shared2, #smem, mutable> -> !ttg.memdesc<64xf32, #shared2, #smem, mutable>
          ttng.async_tma_copy_global_to_local %arg111[%248] %288, %287, %true {async_task_id = array<i32: 2>} : !tt.tensordesc<64xf32, #shared2>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64xf32, #shared2, #smem, mutable>
          %289 = arith.addi %arg142, %c64_i32_10 {async_task_id = array<i32: 2>} : i32
          %290 = arith.addi %arg143, %c1_i64_8 {async_task_id = array<i32: 2>} : i64
          scf.yield {async_task_id = array<i32: 2>} %289, %290 : i32, i64
        } {async_task_id = array<i32: 2>, tt.warp_specialize}
        %219 = arith.addi %arg138, %174 {async_task_id = array<i32: 2>} : i32
        %220 = arith.addi %arg139, %c1_i64_8 {async_task_id = array<i32: 2>} : i64
        scf.yield {async_task_id = array<i32: 2>} %219, %220, %218#1 : i32, i64, i64
      } {async_task_id = array<i32: 2>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition2(%arg61: i32, %arg62: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg63: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg64: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg65: !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable>, %arg66: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg67: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg68: !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, %arg69: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg70: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg71: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg72: !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>, %arg73: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg74: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg75: !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, %arg76: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg77: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg78: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg79: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg80: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg81: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg82: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg83: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg84: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg85: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg86: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg87: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg88: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg89: !ttg.memdesc<1x64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, %arg90: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg91: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg92: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg93: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg94: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg95: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg96: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg97: i32, %arg98: i32, %arg99: !tt.tensordesc<128x128xf16, #shared>, %arg100: !tt.tensordesc<128x64xf16, #shared>, %arg101: !tt.tensordesc<128x128xf16, #shared>, %arg102: !tt.tensordesc<64x64xf16, #shared>, %arg103: !tt.tensordesc<32x128xf16, #shared>, %arg104: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg105: !ttg.memdesc<2x64xf32, #shared2, #smem, mutable>, %arg106: !tt.tensordesc<64xf32, #shared2>, %arg107: !tt.tensordesc<64x64xf16, #shared>, %arg108: !tt.tensordesc<32x128xf16, #shared>, %arg109: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg110: !ttg.memdesc<2x64xf32, #shared2, #smem, mutable>, %arg111: !tt.tensordesc<64xf32, #shared2>, %arg112: f32, %arg113: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg114: !tt.tensordesc<128x64xf16, #shared>, %arg115: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg116: !tt.tensordesc<128x64xf16, #shared>, %arg117: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg118: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg119: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg120: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg121: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg122: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg126: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg127: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg128: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg129: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg130: !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, %arg131: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg132: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg133: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg134: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg135: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg136: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) num_warps(8) {
      %c0_i32_8 = arith.constant {async_task_id = array<i32: 3>} 0 : i32
      %c1_i32_9 = arith.constant {async_task_id = array<i32: 3>} 1 : i32
      %c2_i32_10 = arith.constant {async_task_id = array<i32: 3>} 2 : i32
      %c128_i32_11 = arith.constant {async_task_id = array<i32: 3>} 128 : i32
      %c127_i32_12 = arith.constant {async_task_id = array<i32: 3>} 127 : i32
      %c64_i32_13 = arith.constant {async_task_id = array<i32: 3>} 64 : i32
      %true = arith.constant {async_task_id = array<i32: 3>} true
      %c0_i64_14 = arith.constant {async_task_id = array<i32: 3>} 0 : i64
      %c1_i64_15 = arith.constant {async_task_id = array<i32: 3>} 1 : i64
      %c2_i64 = arith.constant {async_task_id = array<i32: 3>} 2 : i64
      %169 = arith.addi %arg61, %c127_i32_12 {async_task_id = array<i32: 3>} : i32
      %170 = arith.divsi %169, %c128_i32_11 {async_task_id = array<i32: 3>} : i32
      %171 = tt.get_program_id x {async_task_id = array<i32: 3>} : i32
      %172 = arith.divsi %171, %c2_i32_10 {async_task_id = array<i32: 3>} : i32
      %173 = tt.get_num_programs x {async_task_id = array<i32: 3>} : i32
      %174 = arith.divsi %173, %c2_i32_10 {async_task_id = array<i32: 3>} : i32
      %175 = arith.divsi %170, %174 {async_task_id = array<i32: 3>} : i32
      %176 = arith.remsi %170, %174 {async_task_id = array<i32: 3>} : i32
      %177 = arith.cmpi slt, %172, %176 {async_task_id = array<i32: 3>} : i32
      %178 = scf.if %177 -> (i32) {
        %183 = arith.addi %175, %c1_i32_9 {async_task_id = array<i32: 3>} : i32
        scf.yield {async_task_id = array<i32: 3>} %183 : i32
      } else {
        scf.yield {async_task_id = array<i32: 3>} %175 : i32
      } {async_task_id = array<i32: 3>}
      %179 = arith.extsi %arg97 {async_task_id = array<i32: 3>} : i32 to i64
      %180 = arith.divsi %arg61, %c64_i32_13 {async_task_id = array<i32: 3>} : i32
      %181 = tt.splat %arg112 {async_task_id = array<i32: 3>} : f32 -> tensor<128x64xf32, #blocked3>
      %182:3 = scf.for %arg137 = %c0_i32_8 to %178 step %c1_i32_9 iter_args(%arg138 = %172, %arg139 = %c0_i64_14, %arg140 = %c0_i64_14) -> (i32, i64, i64)  : i32 {
        %183 = arith.remsi %arg138, %170 {async_task_id = array<i32: 3>} : i32
        %184 = arith.divsi %arg138, %170 {async_task_id = array<i32: 3>} : i32
        %185 = arith.muli %arg98, %184 {async_task_id = array<i32: 3>} : i32
        %186 = arith.extsi %185 {async_task_id = array<i32: 3>} : i32 to i64
        %187 = arith.divsi %186, %179 {async_task_id = array<i32: 3>} : i64
        %188 = arith.muli %183, %c128_i32_11 {async_task_id = array<i32: 3>} : i32
        %189 = arith.extsi %188 {async_task_id = array<i32: 3>} : i32 to i64
        %190 = arith.addi %187, %189 {async_task_id = array<i32: 3>} : i64
        %191 = arith.trunci %190 {async_task_id = array<i32: 3>} : i64 to i32
        %192 = arith.andi %arg139, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
        %193 = arith.trunci %192 {async_task_id = array<i32: 3>} : i64 to i1
        %194 = arith.andi %arg139, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
        %195 = arith.trunci %194 {async_task_id = array<i32: 3>} : i64 to i1
        %196 = arith.cmpi sgt, %180, %c0_i32_8 : i32
        %197 = arith.divui %arg140, %c2_i64 {async_task_id = array<i32: 3>} : i64
        %198 = arith.muli %197, %c2_i64 {async_task_id = array<i32: 3>} : i64
        %199 = arith.subi %arg140, %198 {async_task_id = array<i32: 3>} : i64
        %200 = arith.trunci %199 {async_task_id = array<i32: 3>} : i64 to i32
        %201 = arith.andi %197, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
        %202 = arith.trunci %201 {async_task_id = array<i32: 3>} : i64 to i1
        %203 = arith.andi %arg140, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
        %204 = arith.trunci %203 {async_task_id = array<i32: 3>} : i64 to i1
        %205 = arith.andi %arg140, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
        %206 = arith.trunci %205 {async_task_id = array<i32: 3>} : i64 to i1
        %207 = arith.divui %arg140, %c2_i64 {async_task_id = array<i32: 3>} : i64
        %208 = arith.muli %207, %c2_i64 {async_task_id = array<i32: 3>} : i64
        %209 = arith.subi %arg140, %208 {async_task_id = array<i32: 3>} : i64
        %210 = arith.trunci %209 {async_task_id = array<i32: 3>} : i64 to i32
        %211 = ttg.memdesc_index %arg105[%210] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x64xf32, #shared2, #smem, mutable> -> !ttg.memdesc<64xf32, #shared2, #smem, mutable>
        %212 = ttg.memdesc_index %arg104[%200] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %213 = arith.extui %202 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %212, %213, %196 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %214 = ttg.local_load %211 : !ttg.memdesc<64xf32, #shared2, #smem, mutable> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear8}>>
        %215 = ttg.memdesc_index %arg122[%200] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %215, 1, %196 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %216 = tt.expand_dims %214 {async_task_id = array<i32: 3>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear8}>> -> tensor<1x64xf32, #linear8>
        %217 = tt.broadcast %216 {async_task_id = array<i32: 3>} : tensor<1x64xf32, #linear8> -> tensor<128x64xf32, #linear8>
        %218 = ttg.memdesc_index %arg68[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %219 = ttg.memdesc_index %arg70[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %220 = arith.extui %204 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %219, %220, %196 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "forward", dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %result_16, %token_17 = ttng.tmem_load %218[] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear8>
        %221 = ttg.memdesc_index %arg124[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %221, 1, %196 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %222 = arith.subf %result_16, %217 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear8>
        %223 = math.exp2 %222 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear8>
        %224 = arith.truncf %223 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear8> to tensor<128x64xf16, #linear8>
        %225 = ttng.tmem_subslice %arg68 {N = 0 : i32, async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64>
        %226 = ttg.memdesc_reinterpret %225 {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64> -> !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable>
        %227 = ttg.memdesc_index %226[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
        %228 = ttg.memdesc_index %arg124[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %229 = arith.extui %206 : i1 to i32
        ttng.wait_barrier %228, %229, %196 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {dstTask = 3 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tmem_store %224, %227, %196 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear8> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
        %230 = ttg.memdesc_index %arg125[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %230, 1, %196 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %231 = arith.subi %180, %c1_i32_9 : i32
        %232:2 = scf.for %arg141 = %c0_i32_8 to %231 step %c1_i32_9 iter_args(%arg142 = %arg140, %arg143 = %223) -> (i64, tensor<128x64xf32, #linear8>)  : i32 {
          %269 = arith.andi %arg142, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          %270 = arith.trunci %269 {async_task_id = array<i32: 3>} : i64 to i1
          %271 = arith.andi %arg142, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          %272 = arith.trunci %271 {async_task_id = array<i32: 3>} : i64 to i1
          %273 = arith.divui %arg142, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %274 = arith.muli %273, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %275 = arith.subi %arg142, %274 {async_task_id = array<i32: 3>} : i64
          %276 = arith.trunci %275 {async_task_id = array<i32: 3>} : i64 to i32
          %277 = arith.andi %273, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          %278 = arith.trunci %277 {async_task_id = array<i32: 3>} : i64 to i1
          %279 = arith.divui %arg142, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %280 = arith.muli %279, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %281 = arith.subi %arg142, %280 {async_task_id = array<i32: 3>} : i64
          %282 = arith.trunci %281 {async_task_id = array<i32: 3>} : i64 to i32
          %283 = ttg.memdesc_index %arg110[%282] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x64xf32, #shared2, #smem, mutable> -> !ttg.memdesc<64xf32, #shared2, #smem, mutable>
          %284 = ttg.memdesc_index %arg109[%276] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %285 = arith.extui %278 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %284, %285, %true {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %286 = ttg.local_load %283 : !ttg.memdesc<64xf32, #shared2, #smem, mutable> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear8}>>
          %287 = ttg.memdesc_index %arg130[%276] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %287, 1 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %288 = tt.expand_dims %286 {async_task_id = array<i32: 3>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear8}>> -> tensor<1x64xf32, #linear8>
          %289 = tt.broadcast %288 {async_task_id = array<i32: 3>} : tensor<1x64xf32, #linear8> -> tensor<128x64xf32, #linear8>
          %290 = ttg.memdesc_index %arg75[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
          %291 = ttg.memdesc_index %arg77[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %292 = arith.extui %270 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %291, %292 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "forward", dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %result_24, %token_25 = ttng.tmem_load %290[] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear8>
          %293 = ttg.memdesc_index %arg128[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %293, 1 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %294 = arith.subf %result_24, %289 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear8>
          %295 = arith.mulf %arg143, %294 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear8>
          %296 = arith.truncf %295 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear8> to tensor<128x64xf16, #linear8>
          %297 = ttg.convert_layout %296 : tensor<128x64xf16, #linear8> -> tensor<128x64xf16, #blocked3>
          %298 = ttng.tmem_subslice %arg75 {N = 0 : i32, async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64>
          %299 = ttg.memdesc_reinterpret %298 {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64> -> !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable>
          %300 = ttg.memdesc_index %299[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
          %301 = ttg.memdesc_index %arg128[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %302 = arith.extui %272 : i1 to i32
          ttng.wait_barrier %301, %302 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {dstTask = 3 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.tmem_store %296, %300, %true {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear8> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
          %303 = ttg.memdesc_index %arg131[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %303, 1 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %304 = ttng.two_cta_peer_gather %297 split_dim = 0 num_ctas = 2 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked3> -> tensor<128x64xf16, #blocked3>
          %305 = ttg.memdesc_index %arg87[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %306 = arith.andi %arg142, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          %307 = arith.trunci %306 {async_task_id = array<i32: 3>} : i64 to i1
          %308 = ttg.memdesc_index %arg90[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %309 = arith.xori %307, %true {async_task_id = array<i32: 3>} : i1
          %310 = arith.extui %309 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %308, %310 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttg.local_store %304, %305 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %311 = ttg.memdesc_index %arg133[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %311, 1 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %312 = arith.addi %arg142, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          %313 = arith.divui %312, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %314 = arith.muli %313, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %315 = arith.subi %312, %314 {async_task_id = array<i32: 3>} : i64
          %316 = arith.trunci %315 {async_task_id = array<i32: 3>} : i64 to i32
          %317 = arith.andi %313, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          %318 = arith.trunci %317 {async_task_id = array<i32: 3>} : i64 to i1
          %319 = arith.andi %312, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          %320 = arith.trunci %319 {async_task_id = array<i32: 3>} : i64 to i1
          %321 = arith.andi %312, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          %322 = arith.trunci %321 {async_task_id = array<i32: 3>} : i64 to i1
          %323 = arith.divui %312, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %324 = arith.muli %323, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %325 = arith.subi %312, %324 {async_task_id = array<i32: 3>} : i64
          %326 = arith.trunci %325 {async_task_id = array<i32: 3>} : i64 to i32
          %327 = ttg.memdesc_index %arg105[%326] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x64xf32, #shared2, #smem, mutable> -> !ttg.memdesc<64xf32, #shared2, #smem, mutable>
          %328 = ttg.memdesc_index %arg104[%316] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %329 = arith.extui %318 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %328, %329, %true {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %330 = ttg.local_load %327 : !ttg.memdesc<64xf32, #shared2, #smem, mutable> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear8}>>
          %331 = ttg.memdesc_index %arg122[%316] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %331, 1, %true {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %332 = tt.expand_dims %330 {async_task_id = array<i32: 3>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear8}>> -> tensor<1x64xf32, #linear8>
          %333 = tt.broadcast %332 {async_task_id = array<i32: 3>} : tensor<1x64xf32, #linear8> -> tensor<128x64xf32, #linear8>
          %334 = ttg.memdesc_index %arg68[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
          %335 = ttg.memdesc_index %arg70[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %336 = arith.extui %320 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %335, %336, %true {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "forward", dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %result_26, %token_27 = ttng.tmem_load %334[] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear8>
          %337 = ttg.memdesc_index %arg124[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %337, 1, %true {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %338 = arith.subf %result_26, %333 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear8>
          %339 = math.exp2 %338 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear8>
          %340 = arith.truncf %339 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear8> to tensor<128x64xf16, #linear8>
          %341 = ttng.tmem_subslice %arg68 {N = 0 : i32, async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64>
          %342 = ttg.memdesc_reinterpret %341 {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64> -> !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable>
          %343 = ttg.memdesc_index %342[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
          %344 = ttg.memdesc_index %arg124[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %345 = arith.extui %322 : i1 to i32
          ttng.wait_barrier %344, %345, %true {async_task_id = array<i32: 3>, constraints = {WSBarrier = {dstTask = 3 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.tmem_store %340, %343, %true {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear8> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
          %346 = ttg.memdesc_index %arg125[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %346, 1, %true {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          scf.yield %312, %339 : i64, tensor<128x64xf32, #linear8>
        } {async_task_id = array<i32: 3>, tt.warp_specialize}
        %233 = arith.cmpi sgt, %180, %c0_i32_8 : i32
        %234 = scf.if %233 -> (i64) {
          %269 = arith.andi %232#0, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          %270 = arith.trunci %269 {async_task_id = array<i32: 3>} : i64 to i1
          %271 = arith.andi %232#0, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          %272 = arith.trunci %271 {async_task_id = array<i32: 3>} : i64 to i1
          %273 = arith.divui %232#0, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %274 = arith.muli %273, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %275 = arith.subi %232#0, %274 {async_task_id = array<i32: 3>} : i64
          %276 = arith.trunci %275 {async_task_id = array<i32: 3>} : i64 to i32
          %277 = arith.andi %273, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          %278 = arith.trunci %277 {async_task_id = array<i32: 3>} : i64 to i1
          %279 = arith.divui %232#0, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %280 = arith.muli %279, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %281 = arith.subi %232#0, %280 {async_task_id = array<i32: 3>} : i64
          %282 = arith.trunci %281 {async_task_id = array<i32: 3>} : i64 to i32
          %283 = ttg.memdesc_index %arg110[%282] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x64xf32, #shared2, #smem, mutable> -> !ttg.memdesc<64xf32, #shared2, #smem, mutable>
          %284 = ttg.memdesc_index %arg109[%276] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %285 = arith.extui %278 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %284, %285, %true {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %286 = ttg.local_load %283 : !ttg.memdesc<64xf32, #shared2, #smem, mutable> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear8}>>
          %287 = ttg.memdesc_index %arg130[%276] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %287, 1 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %288 = tt.expand_dims %286 {async_task_id = array<i32: 3>, axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear8}>> -> tensor<1x64xf32, #linear8>
          %289 = tt.broadcast %288 {async_task_id = array<i32: 3>} : tensor<1x64xf32, #linear8> -> tensor<128x64xf32, #linear8>
          %290 = ttg.memdesc_index %arg75[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
          %291 = ttg.memdesc_index %arg77[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %292 = arith.extui %270 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %291, %292 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "forward", dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %result_24, %token_25 = ttng.tmem_load %290[] {async_task_id = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear8>
          %293 = ttg.memdesc_index %arg128[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %293, 1 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %294 = arith.subf %result_24, %289 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear8>
          %295 = arith.mulf %232#1, %294 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear8>
          %296 = arith.truncf %295 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #linear8> to tensor<128x64xf16, #linear8>
          %297 = ttg.convert_layout %296 : tensor<128x64xf16, #linear8> -> tensor<128x64xf16, #blocked3>
          %298 = ttng.tmem_subslice %arg75 {N = 0 : i32, async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64>
          %299 = ttg.memdesc_reinterpret %298 {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x32xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x64> -> !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable>
          %300 = ttg.memdesc_index %299[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
          %301 = ttg.memdesc_index %arg128[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %302 = arith.extui %272 : i1 to i32
          ttng.wait_barrier %301, %302 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {dstTask = 3 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.tmem_store %296, %300, %true {async_task_id = array<i32: 3>} : tensor<128x64xf16, #linear8> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable>
          %303 = ttg.memdesc_index %arg131[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %303, 1 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %304 = ttng.two_cta_peer_gather %297 split_dim = 0 num_ctas = 2 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked3> -> tensor<128x64xf16, #blocked3>
          %305 = ttg.memdesc_index %arg87[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %306 = arith.andi %232#0, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          %307 = arith.trunci %306 {async_task_id = array<i32: 3>} : i64 to i1
          %308 = ttg.memdesc_index %arg90[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %309 = arith.xori %307, %true {async_task_id = array<i32: 3>} : i1
          %310 = arith.extui %309 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %308, %310 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttg.local_store %304, %305 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %311 = ttg.memdesc_index %arg133[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          ttng.arrive_barrier %311, 1 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
          %312 = arith.addi %232#0, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
          scf.yield %312 : i64
        } else {
          scf.yield %232#0 : i64
        }
        %235 = ttg.memdesc_index %arg79[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %236 = ttg.memdesc_index %arg93[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %237 = arith.extui %193 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %236, %237 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "forward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %result_18, %token_19 = ttng.tmem_load %235[] {async_task_id = array<i32: 3>, tmem.end = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear9>
        %238 = ttg.memdesc_index %arg118[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %238, 1 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %239 = tt.reshape %result_18 : tensor<128x128xf32, #linear9> -> tensor<128x2x64xf32, #linear10>
        %240 = tt.trans %239 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear10> -> tensor<128x64x2xf32, #linear11>
        %241 = ttg.convert_layout %240 : tensor<128x64x2xf32, #linear11> -> tensor<128x64x2xf32, #blocked4>
        %outLHS, %outRHS = tt.split %241 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked3>
        %242 = arith.truncf %outLHS {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked3> to tensor<128x64xf16, #blocked3>
        %243 = ttg.memdesc_index %arg113[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttg.local_store %242, %243 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %244 = ttg.memdesc_index %arg113[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %245 = ttng.async_tma_copy_local_to_global %arg114[%191, %c0_i32_8] %244 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %244   {async_task_id = array<i32: 3>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %246 = arith.truncf %outRHS {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked3> to tensor<128x64xf16, #blocked3>
        %247 = ttg.memdesc_index %arg113[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttg.local_store %246, %247 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %248 = ttg.memdesc_index %arg113[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %249 = ttng.async_tma_copy_local_to_global %arg114[%191, %c64_i32_13] %248 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %248   {async_task_id = array<i32: 3>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %250 = ttg.memdesc_index %arg83[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %251 = ttg.memdesc_index %arg92[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %252 = arith.extui %195 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %251, %252 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, direction = "forward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %result_20, %token_21 = ttng.tmem_load %250[] {async_task_id = array<i32: 3>, tmem.end = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear9>
        %253 = ttg.memdesc_index %arg120[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %253, 1 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %254 = tt.reshape %result_20 : tensor<128x128xf32, #linear9> -> tensor<128x2x64xf32, #linear10>
        %255 = tt.trans %254 {async_task_id = array<i32: 3>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear10> -> tensor<128x64x2xf32, #linear11>
        %256 = ttg.convert_layout %255 : tensor<128x64x2xf32, #linear11> -> tensor<128x64x2xf32, #blocked4>
        %outLHS_22, %outRHS_23 = tt.split %256 {async_task_id = array<i32: 3>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked3>
        %257 = arith.mulf %outLHS_22, %181 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked3>
        %258 = arith.truncf %257 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked3> to tensor<128x64xf16, #blocked3>
        %259 = ttg.memdesc_index %arg115[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttg.local_store %258, %259 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %260 = ttg.memdesc_index %arg115[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %261 = ttng.async_tma_copy_local_to_global %arg116[%191, %c0_i32_8] %260 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %260   {async_task_id = array<i32: 3>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %262 = arith.mulf %outRHS_23, %181 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked3>
        %263 = arith.truncf %262 {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked3> to tensor<128x64xf16, #blocked3>
        %264 = ttg.memdesc_index %arg115[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttg.local_store %263, %264 {async_task_id = array<i32: 3>} : tensor<128x64xf16, #blocked3> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %265 = ttg.memdesc_index %arg115[%c0_i32_8] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %266 = ttng.async_tma_copy_local_to_global %arg116[%191, %c64_i32_13] %265 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %265   {async_task_id = array<i32: 3>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %267 = arith.addi %arg138, %174 {async_task_id = array<i32: 3>} : i32
        %268 = arith.addi %arg139, %c1_i64_15 {async_task_id = array<i32: 3>} : i64
        scf.yield {async_task_id = array<i32: 3>} %267, %268, %234 : i32, i64, i64
      } {async_task_id = array<i32: 3>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    } : (i32, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, i32, i32, !tt.tensordesc<128x128xf16, #shared>, !tt.tensordesc<128x64xf16, #shared>, !tt.tensordesc<128x128xf16, #shared>, !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<32x128xf16, #shared>, !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<2x64xf32, #shared2, #smem, mutable>, !tt.tensordesc<64xf32, #shared2>, !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<32x128xf16, #shared>, !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<2x64xf32, #shared2, #smem, mutable>, !tt.tensordesc<64xf32, #shared2>, f32, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) -> ()
    %113 = ttg.memdesc_index %54[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %113 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %114 = ttg.memdesc_index %50[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %114 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %115 = ttg.memdesc_index %46[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %115 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %116 = ttg.memdesc_index %31[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %116 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %117 = ttg.memdesc_index %31[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %117 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %118 = ttg.memdesc_index %28[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %118 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %119 = ttg.memdesc_index %28[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %119 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %120 = ttg.memdesc_index %23[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %120 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %121 = ttg.memdesc_index %15[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %121 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %122 = ttg.memdesc_index %13[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %122 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %123 = ttg.memdesc_index %11[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %123 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %124 = ttg.memdesc_index %9[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %124 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %125 = ttg.memdesc_index %4[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %125 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %126 = ttg.memdesc_index %17[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %126 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %127 = ttg.memdesc_index %17[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %127 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %128 = ttg.memdesc_index %20[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %128 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %129 = ttg.memdesc_index %20[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %129 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %130 = ttg.memdesc_index %34[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %130 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %131 = ttg.memdesc_index %34[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %131 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %132 = ttg.memdesc_index %37[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %132 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %133 = ttg.memdesc_index %37[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %133 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %134 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %134 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %135 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %135 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %136 = ttg.memdesc_index %40[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %136 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %137 = ttg.memdesc_index %42[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %137 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %138 = ttg.memdesc_index %44[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %138 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %139 = ttg.memdesc_index %48[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %139 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %140 = ttg.memdesc_index %52[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %140 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %141 = ttg.memdesc_index %25[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %141 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %142 = ttg.memdesc_index %25[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %142 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %143 = ttg.memdesc_index %6[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %143 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %144 = ttg.memdesc_index %6[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %144 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %145 = ttg.memdesc_index %56[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %145 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %146 = ttg.memdesc_index %57[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %146 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %147 = ttg.memdesc_index %60[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %147 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %148 = ttg.memdesc_index %61[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %148 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %149 = ttg.memdesc_index %64[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %149 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %150 = ttg.memdesc_index %64[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %150 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %151 = ttg.memdesc_index %65[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %151 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %152 = ttg.memdesc_index %65[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %152 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %153 = ttg.memdesc_index %70[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %153 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %154 = ttg.memdesc_index %71[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %154 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %155 = ttg.memdesc_index %74[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %155 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %156 = ttg.memdesc_index %75[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %156 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %157 = ttg.memdesc_index %78[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %157 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %158 = ttg.memdesc_index %79[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %158 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %159 = ttg.memdesc_index %82[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %159 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %160 = ttg.memdesc_index %82[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %160 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %161 = ttg.memdesc_index %83[%c0_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %161 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %162 = ttg.memdesc_index %83[%c1_i32] : !ttg.memdesc<2x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %162 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %163 = ttg.memdesc_index %88[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %163 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %164 = ttg.memdesc_index %89[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %164 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %165 = ttg.memdesc_index %92[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %165 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %166 = ttg.memdesc_index %93[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %166 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %167 = ttg.memdesc_index %96[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %167 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %168 = ttg.memdesc_index %97[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %168 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    tt.return
  }
}
