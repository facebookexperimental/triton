// RUN: triton-opt %s --nvgpu-materialize-2cta-exchange | FileCheck %s --check-prefix=SHAPE
// RUN: triton-opt %s --nvgpu-materialize-2cta-exchange | FileCheck %s --check-prefix=CLEAN
// RUN: triton-opt %s --tritongpu-optimize-partition-warps | FileCheck %s --check-prefix=WARPS

// Full TTGIR captured immediately before NVGPUMaterialize2CTAExchange from
// the non-causal persistent BM128 2-CTA configuration after AutoWS and
// software pipelining.
// SHAPE-DAG: ttg.memdesc_subslice {{.*}}[0, 0]
// SHAPE-DAG: ttg.memdesc_subslice {{.*}}[128, 0]
// SHAPE-DAG: ttg.local_store {{.*}} tensor<128x64xf16
// SHAPE-DAG: ttg.async_remote_shmem_copy
// Warp-specialize uses IDs 0-6 after padding; the relay takes the next slot.
// SHAPE-DAG: %[[BAR:.+]] = arith.constant 7 : i32
// SHAPE-DAG: ttng.wait_barrier_named %[[BAR]],
// SHAPE-DAG: ttng.fence_async_shared
// SHAPE-DAG: ttng.tmem_subslice {{.*}} {offset = 0 : i32
// SHAPE-DAG: ttng.tmem_subslice {{.*}} {offset = 64 : i32
// SHAPE-COUNT-2: ttng.tmem_load
// CLEAN-NOT: ttng.two_cta_peer_gather
// CLEAN-NOT: ttng.two_cta_peer_relay
// WARPS: partition0({{.*}}) num_warps(1)
// WARPS: partition1({{.*}}) num_warps(1)
// WARPS: partition2({{.*}}) num_warps(1)
// The single-CTA eight-warp computation override changes this kernel's
// layouts and exceeds the Blackwell SMEM limit. Keep its computed four warps.
// WARPS: partition3({{.*}}) num_warps(4)

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [8, 1, 4], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [0, 1, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [0, 0, 1]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 32, 0], [0, 16, 0], [0, 1, 0], [0, 2, 0], [32, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 32], [0, 16], [0, 1], [0, 2], [32, 0]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1, 0], [0, 0, 16], [0, 0, 1], [0, 0, 2], [32, 0, 0]], lane = [[0, 0, 4], [0, 0, 8], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 0, 1], [0, 16, 0], [0, 1, 0], [0, 2, 0], [32, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 16], [0, 1], [0, 2], [32, 0]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear9 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear10 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear11 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear12 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear13 = #ttg.linear<{register = [[0, 128], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#linear14 = #ttg.linear<{register = [[128, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear15 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear16 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear17 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear18 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear19 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true, ctaMode = twocta_rhs>
#tmem3 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1, twoCTAs = true>
#tmem4 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @_attn_bwd_persist(%desc_q: !tt.tensordesc<128x64xf16, #shared>, %desc_q.shape.0: i32, %desc_q.shape.1: i32, %desc_q.stride.0: i64, %desc_q.stride.1: i64, %desc_qt: !tt.tensordesc<64x128xf16, #shared>, %desc_qt.shape.0: i32, %desc_qt.shape.1: i32, %desc_qt.stride.0: i64, %desc_qt.stride.1: i64, %desc_k: !tt.tensordesc<128x128xf16, #shared>, %desc_k.shape.0: i32, %desc_k.shape.1: i32, %desc_k.stride.0: i64, %desc_k.stride.1: i64, %desc_kt: !tt.tensordesc<256x64xf16, #shared>, %desc_kt.shape.0: i32, %desc_kt.shape.1: i32, %desc_kt.stride.0: i64, %desc_kt.stride.1: i64, %desc_v: !tt.tensordesc<128x128xf16, #shared>, %desc_v.shape.0: i32, %desc_v.shape.1: i32, %desc_v.stride.0: i64, %desc_v.stride.1: i64, %sm_scale: f32, %desc_do: !tt.tensordesc<128x64xf16, #shared>, %desc_do.shape.0: i32, %desc_do.shape.1: i32, %desc_do.stride.0: i64, %desc_do.stride.1: i64, %desc_dot: !tt.tensordesc<64x128xf16, #shared>, %desc_dot.shape.0: i32, %desc_dot.shape.1: i32, %desc_dot.stride.0: i64, %desc_dot.stride.1: i64, %desc_dq: !tt.tensordesc<64x16xf32, #shared1>, %desc_dq.shape.0: i32, %desc_dq.shape.1: i32, %desc_dq.stride.0: i64, %desc_dq.stride.1: i64, %desc_dk: !tt.tensordesc<128x16xf16, #shared2>, %desc_dk.shape.0: i32, %desc_dk.shape.1: i32, %desc_dk.stride.0: i64, %desc_dk.stride.1: i64, %desc_dv: !tt.tensordesc<128x16xf16, #shared2>, %desc_dv.shape.0: i32, %desc_dv.shape.1: i32, %desc_dv.stride.0: i64, %desc_dv.stride.1: i64, %desc_m: !tt.tensordesc<128xf32, #shared3>, %desc_m.shape.0: i32, %desc_m.stride.0: i64, %desc_delta: !tt.tensordesc<128xf32, #shared3>, %desc_delta.shape.0: i32, %desc_delta.stride.0: i64, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32 {tt.divisibility = 16 : i32}, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
    %c0_i64 = arith.constant {async_task_id = array<i32: 0>} 0 : i64
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.693147182> : tensor<64x16xf32, #blocked>
    %c2_i32 = arith.constant {async_task_id = array<i32: 0>} 2 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0>} 128 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0>} 127 : i32
    %c16_i32 = arith.constant {async_task_id = array<i32: 0>} 16 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0>} 32 : i32
    %c48_i32 = arith.constant {async_task_id = array<i32: 0>} 48 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0>} 64 : i32
    %c80_i32 = arith.constant {async_task_id = array<i32: 0>} 80 : i32
    %c96_i32 = arith.constant {async_task_id = array<i32: 0>} 96 : i32
    %c112_i32 = arith.constant {async_task_id = array<i32: 0>} 112 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %dq = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dq_0 = ttg.memdesc_index %dq[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dq_0, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT_dq_0 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dsT_dq_0_1 = ttg.memdesc_index %dsT_dq_0[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dsT_dq_0_1, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dsT_2 = ttg.memdesc_index %dsT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dsT_2, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %Di = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %Di_3 = ttg.memdesc_index %Di[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %Di_3, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dpT = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dpT_4 = ttg.memdesc_index %dpT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dpT_4, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dpT_5 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dpT_6 = ttg.memdesc_index %dpT_5[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dpT_6, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dpT_7 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dpT_8 = ttg.memdesc_index %dpT_7[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dpT_8, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %ppT = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %ppT_9 = ttg.memdesc_index %ppT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %ppT_9, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %do = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %do_10 = ttg.memdesc_index %do[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %do_10, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %do_11 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %do_12 = ttg.memdesc_index %do_11[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %do_12, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qkT = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %qkT_13 = ttg.memdesc_index %qkT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %qkT_13, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %m = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %m_14 = ttg.memdesc_index %m[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %m_14, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qT = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared4, #smem, mutable>
    %qT_15 = ttg.memdesc_index %qT[%c0_i32] : !ttg.memdesc<2x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %qT_15, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qT_16 = ttg.memdesc_index %qT[%c1_i32] : !ttg.memdesc<2x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %qT_16, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qT_17 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<2x1xi64, #shared4, #smem, mutable>
    %qT_18 = ttg.memdesc_index %qT_17[%c0_i32] : !ttg.memdesc<2x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %qT_18, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qT_19 = ttg.memdesc_index %qT_17[%c1_i32] : !ttg.memdesc<2x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %qT_19, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %q = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %q_20 = ttg.memdesc_index %q[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %q_20, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %q_21 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %q_22 = ttg.memdesc_index %q_21[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %q_22, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dk = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dk_23 = ttg.memdesc_index %dk[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dk_23, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dv = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dv_24 = ttg.memdesc_index %dv[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dv_24, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %v = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %v_25 = ttg.memdesc_index %v[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %v_25, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %v_26 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %v_27 = ttg.memdesc_index %v_26[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %v_27, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %kt = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %kt_28 = ttg.memdesc_index %kt[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %kt_28, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %kt_29 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %kt_30 = ttg.memdesc_index %kt_29[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %kt_30, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %k = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %k_31 = ttg.memdesc_index %k[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %k_31, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %k_32 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %k_33 = ttg.memdesc_index %k_32[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %k_33, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dv_34 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dv_35 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dv_36 = ttg.memdesc_index %dv_34[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dv_36, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dv_37 = ttg.memdesc_index %dv_35[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dv_37, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %dk_38 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dk_39 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dk_40 = ttg.memdesc_index %dk_38[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dk_40, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dk_41 = ttg.memdesc_index %dk_39[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dk_41, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %m_42 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %m_43 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %m_44 = ttg.memdesc_index %m_42[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %m_44, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %m_45 = ttg.memdesc_index %m_43[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %m_45, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %qkT_46 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %qkT_47 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %qkT_48 = ttg.memdesc_index %qkT_46[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %qkT_48, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qkT_49 = ttg.memdesc_index %qkT_47[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %qkT_49, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %ppT_50 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %ppT_51 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %ppT_52 = ttg.memdesc_index %ppT_50[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %ppT_52, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %ppT_53 = ttg.memdesc_index %ppT_51[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %ppT_53, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %dpT_54 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dpT_55 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dpT_56 = ttg.memdesc_index %dpT_54[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dpT_56, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dpT_57 = ttg.memdesc_index %dpT_55[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dpT_57, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %Di_58 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %Di_59 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %Di_60 = ttg.memdesc_index %Di_58[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %Di_60, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %Di_61 = ttg.memdesc_index %Di_59[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %Di_61, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %dsT_62 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dsT_63 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dsT_64 = ttg.memdesc_index %dsT_62[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dsT_64, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT_65 = ttg.memdesc_index %dsT_63[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dsT_65, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %dsT_dq_1 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dsT_dq_1_66 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dsT_dq_1_67 = ttg.memdesc_index %dsT_dq_1[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dsT_dq_1_67, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT_dq_1_68 = ttg.memdesc_index %dsT_dq_1_66[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dsT_dq_1_68, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %dsT_dq_0_69 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dsT_dq_0_70 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dsT_dq_0_71 = ttg.memdesc_index %dsT_dq_0_69[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dsT_dq_0_71, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT_dq_0_72 = ttg.memdesc_index %dsT_dq_0_70[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dsT_dq_0_72, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %dq_73 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dq_74 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>
    %dq_75 = ttg.memdesc_index %dq_73[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dq_75, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dq_76 = ttg.memdesc_index %dq_74[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.init_barrier %dq_76, 1 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttg.barrier local
    %k_77 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %kt_78 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 12 : i32} : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %v_79 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %dv_80, %dv_81 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dk_82, %dk_83 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 10 : i32} : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %q_84 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 14 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %qT_85 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<2x64x128xf16, #shared, #smem, mutable>
    %m_86 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 16 : i32} : () -> !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>
    %qkT_87, %qkT_88 = ttng.tmem_alloc {allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 2 : i32} : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %do_89 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 17 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %dpT_90 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %dpT_91, %dpT_92 = ttng.tmem_alloc {allocation.shareGroup = 3 : i32, buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %Di_93 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 19 : i32} : () -> !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>
    %dsT_dq_1_94 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 20 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %dsT_dq_0_95 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %desc_dq_reduce_staging = ttg.local_alloc {allocation.shareGroup = 2 : i32, buffer.copy = 1 : i32, buffer.id = 22 : i32, buffer.tmaStaging = 2 : i32} : () -> !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable>
    %desc_dv_staging = ttg.local_alloc {allocation.shareGroup = 1 : i32, buffer.copy = 1 : i32, buffer.id = 30 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>
    %desc_dk_staging = ttg.local_alloc {allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 38 : i32, buffer.tmaStaging = 1 : i32} : () -> !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>
    ttg.warp_specialize(%N_CTX, %BATCH, %H, %k_32, %kt_29, %v_26, %qT_85, %qT_17, %k_77, %qkT_87, %qT, %qkT, %ppT, %dpT_90, %dpT_7, %v_79, %dpT_91, %dpT_5, %dpT, %dv_80, %do_89, %do, %do_11, %dk_82, %q_84, %q, %q_21, %dsT, %dsT_dq_0_95, %kt_78, %dsT_dq_0, %dq, %dk, %dv, %v, %kt, %k, %dsT_dq_1_94, %stride_tok, %stride_h, %stride_z, %desc_k, %desc_kt, %desc_v, %desc_q, %desc_qt, %m, %m_86, %desc_m, %desc_do, %desc_dot, %Di, %Di_93, %desc_delta, %sm_scale, %desc_dv_staging, %desc_dv, %desc_dk_staging, %desc_dk, %dv_34, %dv_35, %dk_38, %dk_39, %m_42, %m_43, %qkT_46, %qkT_47, %ppT_50, %ppT_51, %dpT_54, %dpT_55, %Di_58, %Di_59, %dsT_62, %dsT_63, %dsT_dq_1, %dsT_dq_1_66, %dsT_dq_0_69, %dsT_dq_0_70, %dq_73, %dq_74) attributes {requestedRegisters = array<i32: 24, 24, 24, 192>, ttg.partition.types = ["reduction", "gemm", "relay", "load", "computation"]}
    default {
      %n_tile_num = arith.addi %N_CTX, %c127_i32 {async_task_id = array<i32: 0>} : i32
      %n_tile_num_144 = arith.divsi %n_tile_num, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %cluster_rank = tt.get_program_id x {async_task_id = array<i32: 0>} : i32
      %cluster_rank_145 = arith.remsi %cluster_rank, %c2_i32 {async_task_id = array<i32: 0>} : i32
      %prog_id = arith.divsi %cluster_rank, %c2_i32 {async_task_id = array<i32: 0>} : i32
      %num_progs = tt.get_num_programs x {async_task_id = array<i32: 0>} : i32
      %num_progs_146 = arith.divsi %num_progs, %c2_i32 {async_task_id = array<i32: 0>} : i32
      %scheduled_n_tiles = arith.divsi %n_tile_num_144, %c2_i32 {async_task_id = array<i32: 0>} : i32
      %total_tiles = arith.muli %scheduled_n_tiles, %BATCH {async_task_id = array<i32: 0>} : i32
      %total_tiles_147 = arith.muli %total_tiles, %H {async_task_id = array<i32: 0>} : i32
      %tiles_per_sm = arith.divsi %total_tiles_147, %num_progs_146 {async_task_id = array<i32: 0>} : i32
      %0 = arith.remsi %total_tiles_147, %num_progs_146 {async_task_id = array<i32: 0>} : i32
      %1 = arith.cmpi slt, %prog_id, %0 {async_task_id = array<i32: 0>} : i32
      %2 = scf.if %1 -> (i32) {
        %tiles_per_sm_149 = arith.addi %tiles_per_sm, %c1_i32 {async_task_id = array<i32: 0>} : i32
        scf.yield {async_task_id = array<i32: 0>} %tiles_per_sm_149 : i32
      } else {
        scf.yield {async_task_id = array<i32: 0>} %tiles_per_sm : i32
      } {async_task_id = array<i32: 0>}
      %off_bh = arith.extsi %stride_tok {async_task_id = array<i32: 0>} : i32 to i64
      %num_steps = arith.divsi %N_CTX, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %dq_row = arith.muli %cluster_rank_145, %c64_i32 {async_task_id = array<i32: 0>} : i32
      %dq_row_148 = arith.extsi %dq_row {async_task_id = array<i32: 0>} : i32 to i64
      %tile_idx:2 = scf.for %tile_idx_149 = %c0_i32 to %2 step %c1_i32 iter_args(%prog_id_150 = %prog_id, %tile_idx_151 = %c0_i64) -> (i32, i64)  : i32 {
        %bhid = arith.divsi %prog_id_150, %scheduled_n_tiles {async_task_id = array<i32: 0>} : i32
        %off_bh_152 = arith.remsi %bhid, %H {async_task_id = array<i32: 0>} : i32
        %off_bh_153 = arith.muli %stride_h, %off_bh_152 {async_task_id = array<i32: 0>} : i32
        %off_bh_154 = arith.divsi %bhid, %H {async_task_id = array<i32: 0>} : i32
        %off_bh_155 = arith.muli %stride_z, %off_bh_154 {async_task_id = array<i32: 0>} : i32
        %off_bh_156 = arith.addi %off_bh_153, %off_bh_155 {async_task_id = array<i32: 0>} : i32
        %off_bh_157 = arith.extsi %off_bh_156 {async_task_id = array<i32: 0>} : i32 to i64
        %off_bh_158 = arith.divsi %off_bh_157, %off_bh {async_task_id = array<i32: 0>} : i64
        %curr_m:2 = scf.for %curr_m_160 = %c0_i32 to %num_steps step %c1_i32 iter_args(%arg67 = %c0_i32, %tile_idx_161 = %tile_idx_151) -> (i32, i64)  : i32 {
          %q_162 = arith.extsi %arg67 {async_task_id = array<i32: 0>} : i32 to i64
          %q_163 = arith.addi %off_bh_158, %q_162 {async_task_id = array<i32: 0>} : i64
          %dq_164 = arith.andi %tile_idx_161, %c1_i64 {async_task_id = array<i32: 0>} : i64
          %dq_165 = arith.trunci %dq_164 {async_task_id = array<i32: 0>} : i64 to i1
          %dpT_166 = ttng.tmem_subslice %dpT_91 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %dpT_167 = ttg.memdesc_reinterpret %dpT_166 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x128xf32, #tmem1, #ttng.tensor_memory, mutable>
          %dq_168 = ttg.memdesc_index %dpT_167[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
          %dq_169 = ttg.memdesc_index %dq[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dq_170 = arith.extui %dq_165 {async_task_id = array<i32: 0>} : i1 to i32
          ttng.wait_barrier %dq_169, %dq_170 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3, 4>, direction = "forward", dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dq_171, %dq_172 = ttng.tmem_load %dq_168[] {async_task_id = array<i32: 0>} : !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
          %dq_173 = ttg.memdesc_index %dq_74[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %dq_173, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3, 4>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dqs = tt.reshape %dq_171 {async_task_id = array<i32: 0>} : tensor<64x128xf32, #linear> -> tensor<64x2x64xf32, #linear1>
          %dqs_174 = tt.trans %dqs {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #linear1> -> tensor<64x64x2xf32, #linear2>
          %dqs_175 = ttg.convert_layout %dqs_174 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #linear2> -> tensor<64x64x2xf32, #linear3>
          %dqs_176, %dqs_177 = tt.split %dqs_175 {async_task_id = array<i32: 0>} : tensor<64x64x2xf32, #linear3> -> tensor<64x64xf32, #linear4>
          %dqs_178 = tt.reshape %dqs_176 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #linear4> -> tensor<64x2x32xf32, #linear5>
          %dqs_179 = tt.trans %dqs_178 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x32xf32, #linear5> -> tensor<64x32x2xf32, #linear6>
          %dqs_180, %dqs_181 = tt.split %dqs_179 {async_task_id = array<i32: 0>} : tensor<64x32x2xf32, #linear6> -> tensor<64x32xf32, #linear7>
          %dqs_182 = tt.reshape %dqs_180 {async_task_id = array<i32: 0>} : tensor<64x32xf32, #linear7> -> tensor<64x2x16xf32, #blocked1>
          %dqs_183 = tt.trans %dqs_182 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x16xf32, #blocked1> -> tensor<64x16x2xf32, #blocked2>
          %dqs_184, %dqs_185 = tt.split %dqs_183 {async_task_id = array<i32: 0>} : tensor<64x16x2xf32, #blocked2> -> tensor<64x16xf32, #blocked>
          %dqs_186 = tt.reshape %dqs_181 {async_task_id = array<i32: 0>} : tensor<64x32xf32, #linear7> -> tensor<64x2x16xf32, #blocked1>
          %dqs_187 = tt.trans %dqs_186 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x16xf32, #blocked1> -> tensor<64x16x2xf32, #blocked2>
          %dqs_188, %dqs_189 = tt.split %dqs_187 {async_task_id = array<i32: 0>} : tensor<64x16x2xf32, #blocked2> -> tensor<64x16xf32, #blocked>
          %dqs_190 = tt.reshape %dqs_177 {async_task_id = array<i32: 0>} : tensor<64x64xf32, #linear4> -> tensor<64x2x32xf32, #linear5>
          %dqs_191 = tt.trans %dqs_190 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x32xf32, #linear5> -> tensor<64x32x2xf32, #linear6>
          %dqs_192, %dqs_193 = tt.split %dqs_191 {async_task_id = array<i32: 0>} : tensor<64x32x2xf32, #linear6> -> tensor<64x32xf32, #linear7>
          %dqs_194 = tt.reshape %dqs_192 {async_task_id = array<i32: 0>} : tensor<64x32xf32, #linear7> -> tensor<64x2x16xf32, #blocked1>
          %dqs_195 = tt.trans %dqs_194 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x16xf32, #blocked1> -> tensor<64x16x2xf32, #blocked2>
          %dqs_196, %dqs_197 = tt.split %dqs_195 {async_task_id = array<i32: 0>} : tensor<64x16x2xf32, #blocked2> -> tensor<64x16xf32, #blocked>
          %dqs_198 = tt.reshape %dqs_193 {async_task_id = array<i32: 0>} : tensor<64x32xf32, #linear7> -> tensor<64x2x16xf32, #blocked1>
          %dqs_199 = tt.trans %dqs_198 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<64x2x16xf32, #blocked1> -> tensor<64x16x2xf32, #blocked2>
          %dqs_200, %dqs_201 = tt.split %dqs_199 {async_task_id = array<i32: 0>} : tensor<64x16x2xf32, #blocked2> -> tensor<64x16xf32, #blocked>
          %dq_row_202 = arith.addi %q_163, %dq_row_148 {async_task_id = array<i32: 0>} : i64
          %dqN = arith.mulf %dqs_184, %cst {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked>
          %3 = arith.trunci %dq_row_202 {async_task_id = array<i32: 0>} : i64 to i32
          %desc_dq_reduce_staging_203 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          ttg.local_store %dqN, %desc_dq_reduce_staging_203 {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %desc_dq_reduce_staging_204 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %4 = ttng.async_tma_reduce add, %desc_dq[%3, %c0_i32] %desc_dq_reduce_staging_204 {async_task_id = array<i32: 0>} : !tt.tensordesc<64x16xf32, #shared1>, !ttg.memdesc<64x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          nvws.tma_store_wait %desc_dq_reduce_staging_204   {async_task_id = array<i32: 0>} : !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %dqN_205 = arith.mulf %dqs_185, %cst {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked>
          %desc_dq_reduce_staging_206 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          ttg.local_store %dqN_205, %desc_dq_reduce_staging_206 {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %desc_dq_reduce_staging_207 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %5 = ttng.async_tma_reduce add, %desc_dq[%3, %c16_i32] %desc_dq_reduce_staging_207 {async_task_id = array<i32: 0>} : !tt.tensordesc<64x16xf32, #shared1>, !ttg.memdesc<64x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          nvws.tma_store_wait %desc_dq_reduce_staging_207   {async_task_id = array<i32: 0>} : !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %dqN_208 = arith.mulf %dqs_188, %cst {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked>
          %desc_dq_reduce_staging_209 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          ttg.local_store %dqN_208, %desc_dq_reduce_staging_209 {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %desc_dq_reduce_staging_210 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %6 = ttng.async_tma_reduce add, %desc_dq[%3, %c32_i32] %desc_dq_reduce_staging_210 {async_task_id = array<i32: 0>} : !tt.tensordesc<64x16xf32, #shared1>, !ttg.memdesc<64x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          nvws.tma_store_wait %desc_dq_reduce_staging_210   {async_task_id = array<i32: 0>} : !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %dqN_211 = arith.mulf %dqs_189, %cst {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked>
          %desc_dq_reduce_staging_212 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          ttg.local_store %dqN_211, %desc_dq_reduce_staging_212 {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %desc_dq_reduce_staging_213 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %7 = ttng.async_tma_reduce add, %desc_dq[%3, %c48_i32] %desc_dq_reduce_staging_213 {async_task_id = array<i32: 0>} : !tt.tensordesc<64x16xf32, #shared1>, !ttg.memdesc<64x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          nvws.tma_store_wait %desc_dq_reduce_staging_213   {async_task_id = array<i32: 0>} : !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %dqN_214 = arith.mulf %dqs_196, %cst {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked>
          %desc_dq_reduce_staging_215 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          ttg.local_store %dqN_214, %desc_dq_reduce_staging_215 {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %desc_dq_reduce_staging_216 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %8 = ttng.async_tma_reduce add, %desc_dq[%3, %c64_i32] %desc_dq_reduce_staging_216 {async_task_id = array<i32: 0>} : !tt.tensordesc<64x16xf32, #shared1>, !ttg.memdesc<64x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          nvws.tma_store_wait %desc_dq_reduce_staging_216   {async_task_id = array<i32: 0>} : !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %dqN_217 = arith.mulf %dqs_197, %cst {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked>
          %desc_dq_reduce_staging_218 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          ttg.local_store %dqN_217, %desc_dq_reduce_staging_218 {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %desc_dq_reduce_staging_219 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %9 = ttng.async_tma_reduce add, %desc_dq[%3, %c80_i32] %desc_dq_reduce_staging_219 {async_task_id = array<i32: 0>} : !tt.tensordesc<64x16xf32, #shared1>, !ttg.memdesc<64x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          nvws.tma_store_wait %desc_dq_reduce_staging_219   {async_task_id = array<i32: 0>} : !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %dqN_220 = arith.mulf %dqs_200, %cst {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked>
          %desc_dq_reduce_staging_221 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          ttg.local_store %dqN_220, %desc_dq_reduce_staging_221 {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %desc_dq_reduce_staging_222 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %10 = ttng.async_tma_reduce add, %desc_dq[%3, %c96_i32] %desc_dq_reduce_staging_222 {async_task_id = array<i32: 0>} : !tt.tensordesc<64x16xf32, #shared1>, !ttg.memdesc<64x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          nvws.tma_store_wait %desc_dq_reduce_staging_222   {async_task_id = array<i32: 0>} : !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %dqN_223 = arith.mulf %dqs_201, %cst {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked>
          %desc_dq_reduce_staging_224 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          ttg.local_store %dqN_223, %desc_dq_reduce_staging_224 {async_task_id = array<i32: 0>} : tensor<64x16xf32, #blocked> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %desc_dq_reduce_staging_225 = ttg.memdesc_index %desc_dq_reduce_staging[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x64x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %11 = ttng.async_tma_reduce add, %desc_dq[%3, %c112_i32] %desc_dq_reduce_staging_225 {async_task_id = array<i32: 0>} : !tt.tensordesc<64x16xf32, #shared1>, !ttg.memdesc<64x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
          nvws.tma_store_wait %desc_dq_reduce_staging_225   {async_task_id = array<i32: 0>} : !ttg.memdesc<64x16xf32, #shared1, #smem, mutable>
          %curr_m_226 = arith.addi %arg67, %c128_i32 {async_task_id = array<i32: 0>} : i32
          %accum_cnt = arith.addi %tile_idx_161, %c1_i64 {async_task_id = array<i32: 0>} : i64
          scf.yield {async_task_id = array<i32: 0>} %curr_m_226, %accum_cnt : i32, i64
        } {async_task_id = array<i32: 0>, tt.warp_specialize}
        %tile_idx_159 = arith.addi %prog_id_150, %num_progs_146 {async_task_id = array<i32: 0>} : i32
        scf.yield {async_task_id = array<i32: 0>} %tile_idx_159, %curr_m#1 : i32, i64
      } {async_task_id = array<i32: 0>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "relay", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_yield
    }
    partition0(%N_CTX_144: i32, %BATCH_145: i32, %H_146: i32, %k_147: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %kt_148: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %v_149: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %qT_150: !ttg.memdesc<2x64x128xf16, #shared, #smem, mutable>, %qT_151: !ttg.memdesc<2x1xi64, #shared4, #smem, mutable>, %k_152: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %qkT_153: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %qT_154: !ttg.memdesc<2x1xi64, #shared4, #smem, mutable>, %qkT_155: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %ppT_156: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_157: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %dpT_158: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %v_159: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %dpT_160: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dpT_161: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_162: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dv_163: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %do_164: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %do_165: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %do_166: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_167: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q_168: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %q_169: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %q_170: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_171: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_0_172: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %kt_173: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %dsT_dq_0_174: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dq_175: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_176: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dv_177: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %v_178: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %kt_179: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %k_180: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_1_181: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %stride_tok_182: i32, %stride_h_183: i32, %stride_z_184: i32, %desc_k_185: !tt.tensordesc<128x128xf16, #shared>, %desc_kt_186: !tt.tensordesc<256x64xf16, #shared>, %desc_v_187: !tt.tensordesc<128x128xf16, #shared>, %desc_q_188: !tt.tensordesc<128x64xf16, #shared>, %desc_qt_189: !tt.tensordesc<64x128xf16, #shared>, %m_190: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %m_191: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %desc_m_192: !tt.tensordesc<128xf32, #shared3>, %desc_do_193: !tt.tensordesc<128x64xf16, #shared>, %desc_dot_194: !tt.tensordesc<64x128xf16, #shared>, %Di_195: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %Di_196: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %desc_delta_197: !tt.tensordesc<128xf32, #shared3>, %sm_scale_198: f32, %desc_dv_staging_199: !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>, %desc_dv_200: !tt.tensordesc<128x16xf16, #shared2>, %desc_dk_staging_201: !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>, %desc_dk_202: !tt.tensordesc<128x16xf16, #shared2>, %dv_203: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dv_204: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_205: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_206: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %m_207: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %m_208: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %qkT_209: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %qkT_210: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %ppT_211: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %ppT_212: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_213: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_214: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %Di_215: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %Di_216: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_217: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_218: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_1_219: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_1_220: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_0_221: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_0_222: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dq_223: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dq_224: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>) num_warps(1) {
      %c2_i64 = arith.constant {async_task_id = array<i32: 1>} 2 : i64
      %c1_i64_225 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
      %c0_i64_226 = arith.constant {async_task_id = array<i32: 1>} 0 : i64
      %true = arith.constant {async_task_id = array<i32: 1>} true
      %n_tile_num = arith.constant {async_task_id = array<i32: 1>} 127 : i32
      %c128_i32_227 = arith.constant {async_task_id = array<i32: 1>} 128 : i32
      %kt_228 = arith.constant {async_task_id = array<i32: 1>} 2 : i32
      %c1_i32_229 = arith.constant {async_task_id = array<i32: 1>} 1 : i32
      %c0_i32_230 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
      %false = arith.constant {async_task_id = array<i32: 1>} false
      %n_tile_num_231 = arith.addi %N_CTX_144, %n_tile_num {async_task_id = array<i32: 1>} : i32
      %n_tile_num_232 = arith.divsi %n_tile_num_231, %c128_i32_227 {async_task_id = array<i32: 1>} : i32
      %cluster_rank = tt.get_program_id x {async_task_id = array<i32: 1>} : i32
      %prog_id = arith.divsi %cluster_rank, %kt_228 {async_task_id = array<i32: 1>} : i32
      %num_progs = tt.get_num_programs x {async_task_id = array<i32: 1>} : i32
      %num_progs_233 = arith.divsi %num_progs, %kt_228 {async_task_id = array<i32: 1>} : i32
      %scheduled_n_tiles = arith.divsi %n_tile_num_232, %kt_228 {async_task_id = array<i32: 1>} : i32
      %total_tiles = arith.muli %scheduled_n_tiles, %BATCH_145 {async_task_id = array<i32: 1>} : i32
      %total_tiles_234 = arith.muli %total_tiles, %H_146 {async_task_id = array<i32: 1>} : i32
      %tiles_per_sm = arith.divsi %total_tiles_234, %num_progs_233 {async_task_id = array<i32: 1>} : i32
      %0 = arith.remsi %total_tiles_234, %num_progs_233 {async_task_id = array<i32: 1>} : i32
      %1 = arith.cmpi slt, %prog_id, %0 {async_task_id = array<i32: 1>} : i32
      %2 = scf.if %1 -> (i32) {
        %tiles_per_sm_235 = arith.addi %tiles_per_sm, %c1_i32_229 {async_task_id = array<i32: 1>} : i32
        scf.yield {async_task_id = array<i32: 1>} %tiles_per_sm_235 : i32
      } else {
        scf.yield {async_task_id = array<i32: 1>} %tiles_per_sm : i32
      } {async_task_id = array<i32: 1>}
      %num_steps = arith.divsi %N_CTX_144, %c128_i32_227 {async_task_id = array<i32: 1>} : i32
      %tile_idx:2 = scf.for %tile_idx_235 = %c0_i32_230 to %2 step %c1_i32_229 iter_args(%tile_idx_236 = %c0_i64_226, %tile_idx_237 = %c0_i64_226) -> (i64, i64)  : i32 {
        %curr_m = arith.andi %tile_idx_236, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
        %curr_m_238 = arith.trunci %curr_m {async_task_id = array<i32: 1>} : i64 to i1
        %curr_m_239 = arith.andi %tile_idx_236, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
        %curr_m_240 = arith.trunci %curr_m_239 {async_task_id = array<i32: 1>} : i64 to i1
        %curr_m_241 = arith.andi %tile_idx_236, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
        %curr_m_242 = arith.trunci %curr_m_241 {async_task_id = array<i32: 1>} : i64 to i1
        %k_243 = ttg.memdesc_index %k_147[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %k_244 = arith.extui %curr_m_238 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %k_243, %k_244, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %kt_245 = ttg.memdesc_index %kt_148[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %kt_246 = arith.extui %curr_m_240 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %kt_245, %kt_246, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %v_247 = ttg.memdesc_index %v_149[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %v_248 = arith.extui %curr_m_242 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %v_247, %v_248, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %curr_m_249 = arith.andi %tile_idx_236, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
        %curr_m_250 = arith.trunci %curr_m_249 {async_task_id = array<i32: 1>} : i64 to i1
        %dv_251 = ttg.memdesc_index %dv_204[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dv_252 = arith.xori %curr_m_250, %true : i1
        %dv_253 = arith.extui %dv_252 : i1 to i32
        ttng.wait_barrier %dv_251, %dv_253 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, dstTask = 4 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %curr_m_254 = arith.andi %tile_idx_236, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
        %curr_m_255 = arith.trunci %curr_m_254 {async_task_id = array<i32: 1>} : i64 to i1
        %dk_256 = ttg.memdesc_index %dk_206[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dk_257 = arith.xori %curr_m_255, %true : i1
        %dk_258 = arith.extui %dk_257 : i1 to i32
        ttng.wait_barrier %dk_256, %dk_258 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, dstTask = 4 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %curr_m_259:2 = scf.for %curr_m_265 = %c0_i32_230 to %num_steps step %c1_i32_229 iter_args(%arg148 = %false, %tile_idx_266 = %tile_idx_237) -> (i1, i64)  : i32 {
          %q_267 = arith.andi %tile_idx_266, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
          %q_268 = arith.trunci %q_267 {async_task_id = array<i32: 1>} : i64 to i1
          %qT_269 = arith.divui %tile_idx_266, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %qT_270 = arith.muli %qT_269, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %qT_271 = arith.subi %tile_idx_266, %qT_270 {async_task_id = array<i32: 1>} : i64
          %qT_272 = arith.trunci %qT_271 {async_task_id = array<i32: 1>} : i64 to i32
          %qT_273 = arith.andi %qT_269, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
          %qT_274 = arith.trunci %qT_273 {async_task_id = array<i32: 1>} : i64 to i1
          %qT_275 = arith.divui %tile_idx_266, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %qT_276 = arith.muli %qT_275, %c2_i64 {async_task_id = array<i32: 1>} : i64
          %qT_277 = arith.subi %tile_idx_266, %qT_276 {async_task_id = array<i32: 1>} : i64
          %qT_278 = arith.trunci %qT_277 {async_task_id = array<i32: 1>} : i64 to i32
          %qT_279 = ttg.memdesc_index %qT_150[%qT_278] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          %qT_280 = ttg.memdesc_index %qT_151[%qT_272] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %qT_281 = arith.extui %qT_274 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %qT_280, %qT_281, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %qT_282 = ttg.memdesc_trans %qT_279 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared5, #smem, mutable>
          %k_283 = ttg.memdesc_index %k_152[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
          %qkT_284 = ttg.memdesc_index %qkT_153[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %qT_285 = ttg.memdesc_index %qT_154[%qT_272] {async_task_id = array<i32: 1>} : !ttg.memdesc<2x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %qkT_286 = arith.andi %tile_idx_266, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
          %qkT_287 = arith.trunci %qkT_286 {async_task_id = array<i32: 1>} : i64 to i1
          %qkT_288 = ttg.memdesc_index %qkT_155[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %qkT_289 = ttg.memdesc_index %qkT_210[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %qkT_290 = arith.xori %qkT_287, %true : i1
          %qkT_291 = arith.extui %qkT_290 : i1 to i32
          ttng.wait_barrier %qkT_289, %qkT_291 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, dstTask = 4 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dv_292 = arith.andi %tile_idx_266, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
          %dv_293 = arith.trunci %dv_292 {async_task_id = array<i32: 1>} : i64 to i1
          %ppT_294 = ttg.memdesc_index %ppT_156[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %qkT_295 = arith.xori %dv_293, %true {async_task_id = array<i32: 1>} : i1
          %qkT_296 = arith.extui %qkT_295 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %ppT_294, %qkT_296 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {direction = "backward", dstTask = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %qkT_297 = ttng.tc_gen5_mma %k_283, %qT_282, %qkT_284[], %false, %true, %qT_285[%true], %qkT_288[%true] {async_task_id = array<i32: 1>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared5, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %do_298 = arith.andi %tile_idx_266, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
          %do_299 = arith.trunci %do_298 {async_task_id = array<i32: 1>} : i64 to i1
          %dpT_300 = arith.andi %tile_idx_266, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
          %dpT_301 = arith.trunci %dpT_300 {async_task_id = array<i32: 1>} : i64 to i1
          %dpT_302 = ttg.memdesc_index %dpT_157[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          %dpT_303 = ttg.memdesc_index %dpT_158[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dpT_304 = arith.extui %dpT_301 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %dpT_303, %dpT_304, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dpT_305 = ttg.memdesc_trans %dpT_302 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared5, #smem, mutable>
          %v_306 = ttg.memdesc_index %v_159[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
          %dpT_307 = ttg.memdesc_index %dpT_160[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %dpT_308 = ttg.memdesc_index %dpT_161[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dpT_309 = arith.andi %tile_idx_266, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
          %dpT_310 = arith.trunci %dpT_309 {async_task_id = array<i32: 1>} : i64 to i1
          %dpT_311 = ttg.memdesc_index %dpT_162[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dpT_312 = ttg.memdesc_index %dpT_214[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dpT_313 = arith.xori %dpT_310, %true : i1
          %dpT_314 = arith.extui %dpT_313 : i1 to i32
          ttng.wait_barrier %dpT_312, %dpT_314 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, dstTask = 4 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dq_315 = arith.andi %tile_idx_266, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
          %dq_316 = arith.trunci %dq_315 {async_task_id = array<i32: 1>} : i64 to i1
          %dq_317 = ttg.memdesc_index %dq_224[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dq_318 = arith.xori %dq_316, %true : i1
          %dq_319 = arith.extui %dq_318 : i1 to i32
          ttng.wait_barrier %dq_317, %dq_319 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dpT_320 = ttng.tc_gen5_mma %v_306, %dpT_305, %dpT_307[], %false, %true, %dpT_308[%true], %dpT_311[%true] {async_task_id = array<i32: 1>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared5, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dv_321 = ttg.memdesc_index %dv_163[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %do_322 = ttg.memdesc_index %do_164[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %qkT_323 = ttng.tmem_subslice %qkT_153 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x128>
          %qkT_324 = ttg.memdesc_reinterpret %qkT_323 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem4, #ttng.tensor_memory, mutable>
          %ppT_325 = ttg.memdesc_index %qkT_324[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %do_326 = ttg.memdesc_index %do_165[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %do_327 = ttg.memdesc_index %do_166[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %do_328 = arith.extui %do_299 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %do_327, %do_328, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %ppT_329 = ttg.memdesc_index %ppT_156[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %ppT_330 = ttg.memdesc_index %ppT_211[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dv_331 = arith.extui %dv_293 : i1 to i32
          ttng.wait_barrier %ppT_330, %dv_331 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, dstTask = 4 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dv_332 = ttng.tc_gen5_mma %ppT_325, %do_322, %dv_321[], %arg148, %true, %do_326[%true], %ppT_329[%true] {async_task_id = array<i32: 1>, is_async, tmem.start = array<i32: 3>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dk_333 = arith.andi %tile_idx_266, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
          %dk_334 = arith.trunci %dk_333 {async_task_id = array<i32: 1>} : i64 to i1
          %dk_335 = ttg.memdesc_index %dk_167[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %q_336 = ttg.memdesc_index %q_168[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %dpT_337 = ttng.tmem_subslice %dpT_160 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x128>
          %dpT_338 = ttg.memdesc_reinterpret %dpT_337 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x64xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem4, #ttng.tensor_memory, mutable>
          %dsT_339 = ttg.memdesc_index %dpT_338[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %q_340 = ttg.memdesc_index %q_169[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %q_341 = ttg.memdesc_index %q_170[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %q_342 = arith.extui %q_268 {async_task_id = array<i32: 1>} : i1 to i32
          ttng.wait_barrier %q_341, %q_342, %true {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_343 = ttg.memdesc_index %dsT_171[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_344 = ttg.memdesc_index %dsT_217[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dk_345 = arith.extui %dk_334 : i1 to i32
          ttng.wait_barrier %dsT_344, %dk_345 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, dstTask = 4 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dk_346 = ttng.tc_gen5_mma %dsT_339, %q_336, %dk_335[], %arg148, %true, %q_340[%true], %dsT_343[%true] {async_task_id = array<i32: 1>, is_async, tmem.start = array<i32: 4>, tt.autows = "{\22stage\22: \220\22, \22order\22: \223\22, \22channels\22: [\22opndD,tmem,1,10\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_dq = arith.andi %tile_idx_266, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
          %dsT_dq_347 = arith.trunci %dsT_dq {async_task_id = array<i32: 1>} : i64 to i1
          %dsT_dq_0_348 = ttg.memdesc_index %dsT_dq_0_172[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %dsT_dq_0_349 = ttg.memdesc_index %dsT_dq_0_221[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_dq_350 = arith.extui %dsT_dq_347 : i1 to i32
          ttng.wait_barrier %dsT_dq_0_349, %dsT_dq_350 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, dstTask = 4 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_dq_351 = ttg.memdesc_trans %dsT_dq_0_348 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared5, #smem, mutable>
          %kt_352 = ttg.memdesc_index %kt_173[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %dpT_353 = ttng.tmem_subslice %dpT_160 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %dpT_354 = ttg.memdesc_reinterpret %dpT_353 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x128xf32, #tmem1, #ttng.tensor_memory, mutable>
          %dq_355 = ttg.memdesc_index %dpT_354[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x64x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
          %dsT_dq_0_356 = ttg.memdesc_index %dsT_dq_0_174[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dq_357 = ttg.memdesc_index %dq_175[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dpT_358 = ttg.memdesc_index %dpT_214[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dq_359 = arith.extui %dq_316 : i1 to i32
          ttng.wait_barrier %dpT_358, %dq_359 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 2, 3, 4>, dstTask = 4 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dq_360 = ttng.tc_gen5_mma %dsT_dq_351, %kt_352, %dq_355[], %false, %true, %dsT_dq_0_356[%true], %dq_357[%true] {async_task_id = array<i32: 1>, is_async, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x256xf16, #shared5, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %accum_cnt_361 = arith.addi %tile_idx_266, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
          scf.yield {async_task_id = array<i32: 1>} %true, %accum_cnt_361 : i1, i64
        } {async_task_id = array<i32: 1>, tt.warp_specialize}
        %dk_260 = ttg.memdesc_index %dk_176[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_commit %dk_260 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dv_261 = ttg.memdesc_index %dv_177[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_commit %dv_261 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %v_262 = ttg.memdesc_index %v_178[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_commit %v_262 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %kt_263 = ttg.memdesc_index %kt_179[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_commit %kt_263 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %k_264 = ttg.memdesc_index %k_180[%c0_i32_230] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tc_gen5_commit %k_264 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %accum_cnt = arith.addi %tile_idx_236, %c1_i64_225 {async_task_id = array<i32: 1>} : i64
        scf.yield {async_task_id = array<i32: 1>} %accum_cnt, %curr_m_259#1 : i64, i64
      } {async_task_id = array<i32: 1>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "relay", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition1(%N_CTX_144: i32, %BATCH_145: i32, %H_146: i32, %k_147: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %kt_148: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %v_149: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %qT_150: !ttg.memdesc<2x64x128xf16, #shared, #smem, mutable>, %qT_151: !ttg.memdesc<2x1xi64, #shared4, #smem, mutable>, %k_152: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %qkT_153: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %qT_154: !ttg.memdesc<2x1xi64, #shared4, #smem, mutable>, %qkT_155: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %ppT_156: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_157: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %dpT_158: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %v_159: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %dpT_160: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dpT_161: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_162: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dv_163: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %do_164: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %do_165: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %do_166: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_167: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q_168: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %q_169: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %q_170: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_171: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_0_172: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %kt_173: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %dsT_dq_0_174: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dq_175: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_176: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dv_177: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %v_178: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %kt_179: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %k_180: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_1_181: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %stride_tok_182: i32, %stride_h_183: i32, %stride_z_184: i32, %desc_k_185: !tt.tensordesc<128x128xf16, #shared>, %desc_kt_186: !tt.tensordesc<256x64xf16, #shared>, %desc_v_187: !tt.tensordesc<128x128xf16, #shared>, %desc_q_188: !tt.tensordesc<128x64xf16, #shared>, %desc_qt_189: !tt.tensordesc<64x128xf16, #shared>, %m_190: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %m_191: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %desc_m_192: !tt.tensordesc<128xf32, #shared3>, %desc_do_193: !tt.tensordesc<128x64xf16, #shared>, %desc_dot_194: !tt.tensordesc<64x128xf16, #shared>, %Di_195: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %Di_196: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %desc_delta_197: !tt.tensordesc<128xf32, #shared3>, %sm_scale_198: f32, %desc_dv_staging_199: !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>, %desc_dv_200: !tt.tensordesc<128x16xf16, #shared2>, %desc_dk_staging_201: !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>, %desc_dk_202: !tt.tensordesc<128x16xf16, #shared2>, %dv_203: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dv_204: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_205: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_206: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %m_207: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %m_208: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %qkT_209: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %qkT_210: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %ppT_211: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %ppT_212: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_213: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_214: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %Di_215: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %Di_216: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_217: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_218: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_1_219: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_1_220: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_0_221: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_0_222: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dq_223: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dq_224: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>) num_warps(1) {
      %c1_i64_225 = arith.constant {async_task_id = array<i32: 2>} 1 : i64
      %c0_i64_226 = arith.constant {async_task_id = array<i32: 2>} 0 : i64
      %n_tile_num = arith.constant {async_task_id = array<i32: 2>} 127 : i32
      %c128_i32_227 = arith.constant {async_task_id = array<i32: 2>} 128 : i32
      %kt_228 = arith.constant {async_task_id = array<i32: 2>} 2 : i32
      %c1_i32_229 = arith.constant {async_task_id = array<i32: 2>} 1 : i32
      %c0_i32_230 = arith.constant {async_task_id = array<i32: 2>} 0 : i32
      %n_tile_num_231 = arith.addi %N_CTX_144, %n_tile_num {async_task_id = array<i32: 2>} : i32
      %n_tile_num_232 = arith.divsi %n_tile_num_231, %c128_i32_227 {async_task_id = array<i32: 2>} : i32
      %cluster_rank = tt.get_program_id x {async_task_id = array<i32: 2>} : i32
      %prog_id = arith.divsi %cluster_rank, %kt_228 {async_task_id = array<i32: 2>} : i32
      %num_progs = tt.get_num_programs x {async_task_id = array<i32: 2>} : i32
      %num_progs_233 = arith.divsi %num_progs, %kt_228 {async_task_id = array<i32: 2>} : i32
      %scheduled_n_tiles = arith.divsi %n_tile_num_232, %kt_228 {async_task_id = array<i32: 2>} : i32
      %total_tiles = arith.muli %scheduled_n_tiles, %BATCH_145 {async_task_id = array<i32: 2>} : i32
      %total_tiles_234 = arith.muli %total_tiles, %H_146 {async_task_id = array<i32: 2>} : i32
      %tiles_per_sm = arith.divsi %total_tiles_234, %num_progs_233 {async_task_id = array<i32: 2>} : i32
      %0 = arith.remsi %total_tiles_234, %num_progs_233 {async_task_id = array<i32: 2>} : i32
      %1 = arith.cmpi slt, %prog_id, %0 {async_task_id = array<i32: 2>} : i32
      %2 = scf.if %1 -> (i32) {
        %tiles_per_sm_235 = arith.addi %tiles_per_sm, %c1_i32_229 {async_task_id = array<i32: 2>} : i32
        scf.yield {async_task_id = array<i32: 2>} %tiles_per_sm_235 : i32
      } else {
        scf.yield {async_task_id = array<i32: 2>} %tiles_per_sm : i32
      } {async_task_id = array<i32: 2>}
      %num_steps = arith.divsi %N_CTX_144, %c128_i32_227 {async_task_id = array<i32: 2>} : i32
      %tile_idx = scf.for %tile_idx_235 = %c0_i32_230 to %2 step %c1_i32_229 iter_args(%tile_idx_236 = %c0_i64_226) -> (i64)  : i32 {
        %curr_m = scf.for %curr_m_237 = %c0_i32_230 to %num_steps step %c1_i32_229 iter_args(%tile_idx_238 = %tile_idx_236) -> (i64)  : i32 {
          %dsT_dq = arith.andi %tile_idx_238, %c1_i64_225 {async_task_id = array<i32: 2>} : i64
          %dsT_dq_239 = arith.trunci %dsT_dq {async_task_id = array<i32: 2>} : i64 to i1
          %dsT_dq_1_240 = ttg.memdesc_index %dsT_dq_1_181[%c0_i32_230] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %dsT_dq_1_241 = ttg.memdesc_index %dsT_dq_1_219[%c0_i32_230] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_dq_242 = arith.extui %dsT_dq_239 : i1 to i32
          ttng.wait_barrier %dsT_dq_1_241, %dsT_dq_242 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 4 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.two_cta_peer_relay %dsT_dq_1_240 {async_task_id = array<i32: 2>} : <128x64xf16, #shared, #smem, mutable>
          %dsT_dq_1_243 = ttg.memdesc_index %dsT_dq_1_220[%c0_i32_230] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %dsT_dq_1_243, 1 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 4 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %accum_cnt = arith.addi %tile_idx_238, %c1_i64_225 {async_task_id = array<i32: 2>} : i64
          scf.yield {async_task_id = array<i32: 2>} %accum_cnt : i64
        } {async_task_id = array<i32: 2>, tt.warp_specialize}
        scf.yield {async_task_id = array<i32: 2>} %curr_m : i64
      } {async_task_id = array<i32: 2>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "relay", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition2(%N_CTX_144: i32, %BATCH_145: i32, %H_146: i32, %k_147: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %kt_148: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %v_149: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %qT_150: !ttg.memdesc<2x64x128xf16, #shared, #smem, mutable>, %qT_151: !ttg.memdesc<2x1xi64, #shared4, #smem, mutable>, %k_152: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %qkT_153: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %qT_154: !ttg.memdesc<2x1xi64, #shared4, #smem, mutable>, %qkT_155: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %ppT_156: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_157: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %dpT_158: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %v_159: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %dpT_160: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dpT_161: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_162: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dv_163: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %do_164: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %do_165: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %do_166: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_167: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q_168: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %q_169: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %q_170: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_171: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_0_172: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %kt_173: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %dsT_dq_0_174: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dq_175: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_176: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dv_177: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %v_178: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %kt_179: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %k_180: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_1_181: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %stride_tok_182: i32, %stride_h_183: i32, %stride_z_184: i32, %desc_k_185: !tt.tensordesc<128x128xf16, #shared>, %desc_kt_186: !tt.tensordesc<256x64xf16, #shared>, %desc_v_187: !tt.tensordesc<128x128xf16, #shared>, %desc_q_188: !tt.tensordesc<128x64xf16, #shared>, %desc_qt_189: !tt.tensordesc<64x128xf16, #shared>, %m_190: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %m_191: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %desc_m_192: !tt.tensordesc<128xf32, #shared3>, %desc_do_193: !tt.tensordesc<128x64xf16, #shared>, %desc_dot_194: !tt.tensordesc<64x128xf16, #shared>, %Di_195: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %Di_196: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %desc_delta_197: !tt.tensordesc<128xf32, #shared3>, %sm_scale_198: f32, %desc_dv_staging_199: !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>, %desc_dv_200: !tt.tensordesc<128x16xf16, #shared2>, %desc_dk_staging_201: !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>, %desc_dk_202: !tt.tensordesc<128x16xf16, #shared2>, %dv_203: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dv_204: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_205: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_206: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %m_207: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %m_208: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %qkT_209: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %qkT_210: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %ppT_211: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %ppT_212: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_213: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_214: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %Di_215: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %Di_216: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_217: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_218: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_1_219: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_1_220: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_0_221: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_0_222: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dq_223: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dq_224: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>) num_warps(1) {
      %c2_i64 = arith.constant {async_task_id = array<i32: 3>} 2 : i64
      %true = arith.constant {async_task_id = array<i32: 3>} true
      %c1_i64_225 = arith.constant {async_task_id = array<i32: 3>} 1 : i64
      %c0_i64_226 = arith.constant {async_task_id = array<i32: 3>} 0 : i64
      %kt_227 = arith.constant {async_task_id = array<i32: 3>} 64 : i32
      %n_tile_num = arith.constant {async_task_id = array<i32: 3>} 127 : i32
      %c128_i32_228 = arith.constant {async_task_id = array<i32: 3>} 128 : i32
      %kt_229 = arith.constant {async_task_id = array<i32: 3>} 2 : i32
      %c1_i32_230 = arith.constant {async_task_id = array<i32: 3>} 1 : i32
      %c0_i32_231 = arith.constant {async_task_id = array<i32: 3>} 0 : i32
      %n_tile_num_232 = arith.addi %N_CTX_144, %n_tile_num {async_task_id = array<i32: 3>} : i32
      %n_tile_num_233 = arith.divsi %n_tile_num_232, %c128_i32_228 {async_task_id = array<i32: 3>} : i32
      %cluster_rank = tt.get_program_id x {async_task_id = array<i32: 3>} : i32
      %cluster_rank_234 = arith.remsi %cluster_rank, %kt_229 {async_task_id = array<i32: 3>} : i32
      %prog_id = arith.divsi %cluster_rank, %kt_229 {async_task_id = array<i32: 3>} : i32
      %num_progs = tt.get_num_programs x {async_task_id = array<i32: 3>} : i32
      %num_progs_235 = arith.divsi %num_progs, %kt_229 {async_task_id = array<i32: 3>} : i32
      %scheduled_n_tiles = arith.divsi %n_tile_num_233, %kt_229 {async_task_id = array<i32: 3>} : i32
      %total_tiles = arith.muli %scheduled_n_tiles, %BATCH_145 {async_task_id = array<i32: 3>} : i32
      %total_tiles_236 = arith.muli %total_tiles, %H_146 {async_task_id = array<i32: 3>} : i32
      %tiles_per_sm = arith.divsi %total_tiles_236, %num_progs_235 {async_task_id = array<i32: 3>} : i32
      %0 = arith.remsi %total_tiles_236, %num_progs_235 {async_task_id = array<i32: 3>} : i32
      %1 = arith.cmpi slt, %prog_id, %0 {async_task_id = array<i32: 3>} : i32
      %2 = scf.if %1 -> (i32) {
        %tiles_per_sm_240 = arith.addi %tiles_per_sm, %c1_i32_230 {async_task_id = array<i32: 3>} : i32
        scf.yield {async_task_id = array<i32: 3>} %tiles_per_sm_240 : i32
      } else {
        scf.yield {async_task_id = array<i32: 3>} %tiles_per_sm : i32
      } {async_task_id = array<i32: 3>}
      %off_bh = arith.extsi %stride_tok_182 {async_task_id = array<i32: 3>} : i32 to i64
      %kt_start_n = arith.muli %cluster_rank_234, %c128_i32_228 {async_task_id = array<i32: 3>} : i32
      %num_steps = arith.divsi %N_CTX_144, %c128_i32_228 {async_task_id = array<i32: 3>} : i32
      %kt_237 = nvg.cluster_id {async_task_id = array<i32: 3>}
      %kt_238 = arith.remsi %kt_237, %kt_229 {async_task_id = array<i32: 3>} : i32
      %kt_239 = arith.muli %kt_238, %kt_227 {async_task_id = array<i32: 3>} : i32
      %tile_idx:3 = scf.for %tile_idx_240 = %c0_i32_231 to %2 step %c1_i32_230 iter_args(%prog_id_241 = %prog_id, %tile_idx_242 = %c0_i64_226, %tile_idx_243 = %c0_i64_226) -> (i32, i64, i64)  : i32 {
        %scheduled_pid = arith.remsi %prog_id_241, %scheduled_n_tiles {async_task_id = array<i32: 3>} : i32
        %bhid = arith.divsi %prog_id_241, %scheduled_n_tiles {async_task_id = array<i32: 3>} : i32
        %pid = arith.muli %scheduled_pid, %kt_229 {async_task_id = array<i32: 3>} : i32
        %pid_244 = arith.addi %pid, %cluster_rank_234 {async_task_id = array<i32: 3>} : i32
        %off_chz = arith.muli %bhid, %N_CTX_144 {async_task_id = array<i32: 3>} : i32
        %off_chz_245 = arith.extsi %off_chz {async_task_id = array<i32: 3>} : i32 to i64
        %off_bh_246 = arith.remsi %bhid, %H_146 {async_task_id = array<i32: 3>} : i32
        %off_bh_247 = arith.muli %stride_h_183, %off_bh_246 {async_task_id = array<i32: 3>} : i32
        %off_bh_248 = arith.divsi %bhid, %H_146 {async_task_id = array<i32: 3>} : i32
        %off_bh_249 = arith.muli %stride_z_184, %off_bh_248 {async_task_id = array<i32: 3>} : i32
        %off_bh_250 = arith.addi %off_bh_247, %off_bh_249 {async_task_id = array<i32: 3>} : i32
        %off_bh_251 = arith.extsi %off_bh_250 {async_task_id = array<i32: 3>} : i32 to i64
        %off_bh_252 = arith.divsi %off_bh_251, %off_bh {async_task_id = array<i32: 3>} : i64
        %start_n = arith.muli %pid_244, %c128_i32_228 {async_task_id = array<i32: 3>} : i32
        %k_253 = arith.extsi %start_n {async_task_id = array<i32: 3>} : i32 to i64
        %k_254 = arith.addi %off_bh_252, %k_253 {async_task_id = array<i32: 3>} : i64
        %k_255 = arith.trunci %k_254 {async_task_id = array<i32: 3>} : i64 to i32
        %curr_m = arith.andi %tile_idx_242, %c1_i64_225 {async_task_id = array<i32: 3>} : i64
        %curr_m_256 = arith.trunci %curr_m {async_task_id = array<i32: 3>} : i64 to i1
        %k_257 = ttg.memdesc_index %k_180[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %k_258 = arith.xori %curr_m_256, %true {async_task_id = array<i32: 3>} : i1
        %k_259 = arith.extui %k_258 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %k_257, %k_259 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 1 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %k_260 = ttg.memdesc_index %k_147[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.barrier_expect %k_260, 32768 {async_task_id = array<i32: 3>}, %true : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %k_261 = ttg.memdesc_index %k_152[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc_k_185[%k_255, %c0_i32_231] %k_261, %k_260, %true {async_task_id = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %kt_start_n_262 = arith.subi %start_n, %kt_start_n {async_task_id = array<i32: 3>} : i32
        %kt_263 = arith.extsi %kt_start_n_262 {async_task_id = array<i32: 3>} : i32 to i64
        %kt_264 = arith.addi %off_bh_252, %kt_263 {async_task_id = array<i32: 3>} : i64
        %kt_265 = arith.trunci %kt_264 {async_task_id = array<i32: 3>} : i64 to i32
        %curr_m_266 = arith.andi %tile_idx_242, %c1_i64_225 {async_task_id = array<i32: 3>} : i64
        %curr_m_267 = arith.trunci %curr_m_266 {async_task_id = array<i32: 3>} : i64 to i1
        %kt_268 = ttg.memdesc_index %kt_179[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %kt_269 = arith.xori %curr_m_267, %true {async_task_id = array<i32: 3>} : i1
        %kt_270 = arith.extui %kt_269 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %kt_268, %kt_270 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 1 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %kt_271 = ttg.memdesc_index %kt_148[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.barrier_expect %kt_271, 32768 {async_task_id = array<i32: 3>}, %true : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %kt_272 = ttg.memdesc_index %kt_173[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc_kt_186[%kt_265, %kt_239] %kt_272, %kt_271, %true {async_task_id = array<i32: 3>} : !tt.tensordesc<256x64xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        %curr_m_273 = arith.andi %tile_idx_242, %c1_i64_225 {async_task_id = array<i32: 3>} : i64
        %curr_m_274 = arith.trunci %curr_m_273 {async_task_id = array<i32: 3>} : i64 to i1
        %v_275 = ttg.memdesc_index %v_178[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %v_276 = arith.xori %curr_m_274, %true {async_task_id = array<i32: 3>} : i1
        %v_277 = arith.extui %v_276 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %v_275, %v_277 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 1 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %v_278 = ttg.memdesc_index %v_149[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.barrier_expect %v_278, 32768 {async_task_id = array<i32: 3>}, %true : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %v_279 = ttg.memdesc_index %v_159[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc_v_187[%k_255, %c0_i32_231] %v_279, %v_278, %true {async_task_id = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %curr_m_280:2 = scf.for %curr_m_282 = %c0_i32_231 to %num_steps step %c1_i32_230 iter_args(%arg149 = %c0_i32_231, %tile_idx_283 = %tile_idx_243) -> (i32, i64)  : i32 {
          %q_284 = arith.extsi %arg149 {async_task_id = array<i32: 3>} : i32 to i64
          %q_285 = arith.addi %off_bh_252, %q_284 {async_task_id = array<i32: 3>} : i64
          %q_286 = arith.trunci %q_285 {async_task_id = array<i32: 3>} : i64 to i32
          %q_287 = arith.andi %tile_idx_283, %c1_i64_225 {async_task_id = array<i32: 3>} : i64
          %q_288 = arith.trunci %q_287 {async_task_id = array<i32: 3>} : i64 to i1
          %q_289 = ttg.memdesc_index %q_169[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %q_290 = arith.xori %q_288, %true {async_task_id = array<i32: 3>} : i1
          %q_291 = arith.extui %q_290 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %q_289, %q_291 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %q_292 = ttg.memdesc_index %q_170[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.barrier_expect %q_292, 16384 {async_task_id = array<i32: 3>}, %true : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %q_293 = ttg.memdesc_index %q_168[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %desc_q_188[%q_286, %kt_239] %q_293, %q_292, %true {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %qt = arith.addi %q_286, %kt_239 {async_task_id = array<i32: 3>} : i32
          %qT_294 = arith.divui %tile_idx_283, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %qT_295 = arith.muli %qT_294, %c2_i64 {async_task_id = array<i32: 3>} : i64
          %qT_296 = arith.subi %tile_idx_283, %qT_295 {async_task_id = array<i32: 3>} : i64
          %qT_297 = arith.trunci %qT_296 {async_task_id = array<i32: 3>} : i64 to i32
          %qT_298 = arith.andi %qT_294, %c1_i64_225 {async_task_id = array<i32: 3>} : i64
          %qT_299 = arith.trunci %qT_298 {async_task_id = array<i32: 3>} : i64 to i1
          %qT_300 = ttg.memdesc_index %qT_154[%qT_297] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %qt_301 = arith.xori %qT_299, %true {async_task_id = array<i32: 3>} : i1
          %qt_302 = arith.extui %qt_301 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %qT_300, %qt_302 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %qT_303 = ttg.memdesc_index %qT_151[%qT_297] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.barrier_expect %qT_303, 16384 {async_task_id = array<i32: 3>}, %true : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %qT_304 = ttg.memdesc_index %qT_150[%qT_297] {async_task_id = array<i32: 3>} : !ttg.memdesc<2x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %desc_qt_189[%qt, %c0_i32_231] %qT_304, %qT_303, %true {async_task_id = array<i32: 3>} : !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          %offs_m_start = arith.addi %off_chz_245, %q_284 {async_task_id = array<i32: 3>} : i64
          %m_305 = arith.trunci %offs_m_start {async_task_id = array<i32: 3>} : i64 to i32
          %m_306 = arith.andi %tile_idx_283, %c1_i64_225 {async_task_id = array<i32: 3>} : i64
          %m_307 = arith.trunci %m_306 {async_task_id = array<i32: 3>} : i64 to i1
          %m_308 = ttg.memdesc_index %m_208[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %m_309 = arith.xori %m_307, %true : i1
          %m_310 = arith.extui %m_309 : i1 to i32
          ttng.wait_barrier %m_308, %m_310 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, dstTask = 4 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %m_311 = ttg.memdesc_index %m_190[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.barrier_expect %m_311, 512 {async_task_id = array<i32: 3>}, %true : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %m_312 = ttg.memdesc_index %m_191[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128xf32, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
          ttng.async_tma_copy_global_to_local %desc_m_192[%m_305] %m_312, %m_311, %true {async_task_id = array<i32: 3>} : !tt.tensordesc<128xf32, #shared3>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
          %do_313 = arith.andi %tile_idx_283, %c1_i64_225 {async_task_id = array<i32: 3>} : i64
          %do_314 = arith.trunci %do_313 {async_task_id = array<i32: 3>} : i64 to i1
          %do_315 = ttg.memdesc_index %do_165[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %do_316 = arith.xori %do_314, %true {async_task_id = array<i32: 3>} : i1
          %do_317 = arith.extui %do_316 {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %do_315, %do_317 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %do_318 = ttg.memdesc_index %do_166[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.barrier_expect %do_318, 16384 {async_task_id = array<i32: 3>}, %true : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %do_319 = ttg.memdesc_index %do_164[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %desc_do_193[%q_286, %kt_239] %do_319, %do_318, %true {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %dpT_320 = arith.andi %tile_idx_283, %c1_i64_225 {async_task_id = array<i32: 3>} : i64
          %dpT_321 = arith.trunci %dpT_320 {async_task_id = array<i32: 3>} : i64 to i1
          %dpT_322 = ttg.memdesc_index %dpT_161[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dot = arith.xori %dpT_321, %true {async_task_id = array<i32: 3>} : i1
          %dot_323 = arith.extui %dot {async_task_id = array<i32: 3>} : i1 to i32
          ttng.wait_barrier %dpT_322, %dot_323 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dpT_324 = ttg.memdesc_index %dpT_158[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.barrier_expect %dpT_324, 16384 {async_task_id = array<i32: 3>}, %true : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dpT_325 = ttg.memdesc_index %dpT_157[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          ttng.async_tma_copy_global_to_local %desc_dot_194[%qt, %c0_i32_231] %dpT_325, %dpT_324, %true {async_task_id = array<i32: 3>} : !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
          %Di_326 = arith.andi %tile_idx_283, %c1_i64_225 {async_task_id = array<i32: 3>} : i64
          %Di_327 = arith.trunci %Di_326 {async_task_id = array<i32: 3>} : i64 to i1
          %Di_328 = ttg.memdesc_index %Di_216[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %Di_329 = arith.xori %Di_327, %true : i1
          %Di_330 = arith.extui %Di_329 : i1 to i32
          ttng.wait_barrier %Di_328, %Di_330 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, dstTask = 4 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %Di_331 = ttg.memdesc_index %Di_195[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.barrier_expect %Di_331, 512 {async_task_id = array<i32: 3>}, %true : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %Di_332 = ttg.memdesc_index %Di_196[%c0_i32_231] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128xf32, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
          ttng.async_tma_copy_global_to_local %desc_delta_197[%m_305] %Di_332, %Di_331, %true {async_task_id = array<i32: 3>} : !tt.tensordesc<128xf32, #shared3>, !ttg.memdesc<1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
          %curr_m_333 = arith.addi %arg149, %c128_i32_228 {async_task_id = array<i32: 3>} : i32
          %accum_cnt_334 = arith.addi %tile_idx_283, %c1_i64_225 {async_task_id = array<i32: 3>} : i64
          scf.yield {async_task_id = array<i32: 3>} %curr_m_333, %accum_cnt_334 : i32, i64
        } {async_task_id = array<i32: 3>, tt.warp_specialize}
        %tile_idx_281 = arith.addi %prog_id_241, %num_progs_235 {async_task_id = array<i32: 3>} : i32
        %accum_cnt = arith.addi %tile_idx_242, %c1_i64_225 {async_task_id = array<i32: 3>} : i64
        scf.yield {async_task_id = array<i32: 3>} %tile_idx_281, %accum_cnt, %curr_m_280#1 : i32, i64, i64
      } {async_task_id = array<i32: 3>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "relay", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition3(%N_CTX_144: i32, %BATCH_145: i32, %H_146: i32, %k_147: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %kt_148: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %v_149: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %qT_150: !ttg.memdesc<2x64x128xf16, #shared, #smem, mutable>, %qT_151: !ttg.memdesc<2x1xi64, #shared4, #smem, mutable>, %k_152: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %qkT_153: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %qT_154: !ttg.memdesc<2x1xi64, #shared4, #smem, mutable>, %qkT_155: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %ppT_156: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_157: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %dpT_158: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %v_159: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %dpT_160: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dpT_161: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_162: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dv_163: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %do_164: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %do_165: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %do_166: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_167: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q_168: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %q_169: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %q_170: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_171: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_0_172: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %kt_173: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %dsT_dq_0_174: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dq_175: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_176: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dv_177: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %v_178: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %kt_179: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %k_180: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_1_181: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %stride_tok_182: i32, %stride_h_183: i32, %stride_z_184: i32, %desc_k_185: !tt.tensordesc<128x128xf16, #shared>, %desc_kt_186: !tt.tensordesc<256x64xf16, #shared>, %desc_v_187: !tt.tensordesc<128x128xf16, #shared>, %desc_q_188: !tt.tensordesc<128x64xf16, #shared>, %desc_qt_189: !tt.tensordesc<64x128xf16, #shared>, %m_190: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %m_191: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %desc_m_192: !tt.tensordesc<128xf32, #shared3>, %desc_do_193: !tt.tensordesc<128x64xf16, #shared>, %desc_dot_194: !tt.tensordesc<64x128xf16, #shared>, %Di_195: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %Di_196: !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, %desc_delta_197: !tt.tensordesc<128xf32, #shared3>, %sm_scale_198: f32, %desc_dv_staging_199: !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>, %desc_dv_200: !tt.tensordesc<128x16xf16, #shared2>, %desc_dk_staging_201: !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>, %desc_dk_202: !tt.tensordesc<128x16xf16, #shared2>, %dv_203: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dv_204: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_205: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dk_206: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %m_207: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %m_208: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %qkT_209: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %qkT_210: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %ppT_211: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %ppT_212: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_213: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dpT_214: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %Di_215: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %Di_216: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_217: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_218: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_1_219: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_1_220: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_0_221: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dsT_dq_0_222: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dq_223: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, %dq_224: !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>) num_warps(4) {
      %0 = ub.poison : tensor<128x128xf16, #linear8>
      %c1_i64_225 = arith.constant {async_task_id = array<i32: 4>} 1 : i64
      %c0_i64_226 = arith.constant {async_task_id = array<i32: 4>} 0 : i64
      %true = arith.constant {async_task_id = array<i32: 4>} true
      %c112_i32_227 = arith.constant {async_task_id = array<i32: 4>} 112 : i32
      %c96_i32_228 = arith.constant {async_task_id = array<i32: 4>} 96 : i32
      %c80_i32_229 = arith.constant {async_task_id = array<i32: 4>} 80 : i32
      %kt_230 = arith.constant {async_task_id = array<i32: 4>} 64 : i32
      %c48_i32_231 = arith.constant {async_task_id = array<i32: 4>} 48 : i32
      %c32_i32_232 = arith.constant {async_task_id = array<i32: 4>} 32 : i32
      %c16_i32_233 = arith.constant {async_task_id = array<i32: 4>} 16 : i32
      %n_tile_num = arith.constant {async_task_id = array<i32: 4>} 127 : i32
      %c128_i32_234 = arith.constant {async_task_id = array<i32: 4>} 128 : i32
      %kt_235 = arith.constant {async_task_id = array<i32: 4>} 2 : i32
      %c1_i32_236 = arith.constant {async_task_id = array<i32: 4>} 1 : i32
      %c0_i32_237 = arith.constant {async_task_id = array<i32: 4>} 0 : i32
      %n_tile_num_238 = arith.addi %N_CTX_144, %n_tile_num {async_task_id = array<i32: 4>} : i32
      %n_tile_num_239 = arith.divsi %n_tile_num_238, %c128_i32_234 {async_task_id = array<i32: 4>} : i32
      %cluster_rank = tt.get_program_id x {async_task_id = array<i32: 4>} : i32
      %cluster_rank_240 = arith.remsi %cluster_rank, %kt_235 {async_task_id = array<i32: 4>} : i32
      %prog_id = arith.divsi %cluster_rank, %kt_235 {async_task_id = array<i32: 4>} : i32
      %num_progs = tt.get_num_programs x {async_task_id = array<i32: 4>} : i32
      %num_progs_241 = arith.divsi %num_progs, %kt_235 {async_task_id = array<i32: 4>} : i32
      %scheduled_n_tiles = arith.divsi %n_tile_num_239, %kt_235 {async_task_id = array<i32: 4>} : i32
      %total_tiles = arith.muli %scheduled_n_tiles, %BATCH_145 {async_task_id = array<i32: 4>} : i32
      %total_tiles_242 = arith.muli %total_tiles, %H_146 {async_task_id = array<i32: 4>} : i32
      %tiles_per_sm = arith.divsi %total_tiles_242, %num_progs_241 {async_task_id = array<i32: 4>} : i32
      %1 = arith.remsi %total_tiles_242, %num_progs_241 {async_task_id = array<i32: 4>} : i32
      %2 = arith.cmpi slt, %prog_id, %1 {async_task_id = array<i32: 4>} : i32
      %3 = scf.if %2 -> (i32) {
        %tiles_per_sm_243 = arith.addi %tiles_per_sm, %c1_i32_236 {async_task_id = array<i32: 4>} : i32
        scf.yield {async_task_id = array<i32: 4>} %tiles_per_sm_243 : i32
      } else {
        scf.yield {async_task_id = array<i32: 4>} %tiles_per_sm : i32
      } {async_task_id = array<i32: 4>}
      %off_bh = arith.extsi %stride_tok_182 {async_task_id = array<i32: 4>} : i32 to i64
      %num_steps = arith.divsi %N_CTX_144, %c128_i32_234 {async_task_id = array<i32: 4>} : i32
      %dkN = tt.splat %sm_scale_198 {async_task_id = array<i32: 4>} : f32 -> tensor<128x16xf32, #linear9>
      %tile_idx:3 = scf.for %tile_idx_243 = %c0_i32_237 to %3 step %c1_i32_236 iter_args(%prog_id_244 = %prog_id, %tile_idx_245 = %c0_i64_226, %tile_idx_246 = %c0_i64_226) -> (i32, i64, i64)  : i32 {
        %scheduled_pid = arith.remsi %prog_id_244, %scheduled_n_tiles {async_task_id = array<i32: 4>} : i32
        %bhid = arith.divsi %prog_id_244, %scheduled_n_tiles {async_task_id = array<i32: 4>} : i32
        %pid = arith.muli %scheduled_pid, %kt_235 {async_task_id = array<i32: 4>} : i32
        %pid_247 = arith.addi %pid, %cluster_rank_240 {async_task_id = array<i32: 4>} : i32
        %off_bh_248 = arith.remsi %bhid, %H_146 {async_task_id = array<i32: 4>} : i32
        %off_bh_249 = arith.muli %stride_h_183, %off_bh_248 {async_task_id = array<i32: 4>} : i32
        %off_bh_250 = arith.divsi %bhid, %H_146 {async_task_id = array<i32: 4>} : i32
        %off_bh_251 = arith.muli %stride_z_184, %off_bh_250 {async_task_id = array<i32: 4>} : i32
        %off_bh_252 = arith.addi %off_bh_249, %off_bh_251 {async_task_id = array<i32: 4>} : i32
        %off_bh_253 = arith.extsi %off_bh_252 {async_task_id = array<i32: 4>} : i32 to i64
        %off_bh_254 = arith.divsi %off_bh_253, %off_bh {async_task_id = array<i32: 4>} : i64
        %start_n = arith.muli %pid_247, %c128_i32_234 {async_task_id = array<i32: 4>} : i32
        %k_255 = arith.extsi %start_n {async_task_id = array<i32: 4>} : i32 to i64
        %k_256 = arith.addi %off_bh_254, %k_255 {async_task_id = array<i32: 4>} : i64
        %k_257 = arith.trunci %k_256 {async_task_id = array<i32: 4>} : i64 to i32
        %curr_m = arith.andi %tile_idx_245, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
        %curr_m_258 = arith.trunci %curr_m {async_task_id = array<i32: 4>} : i64 to i1
        %curr_m_259 = arith.andi %tile_idx_245, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
        %curr_m_260 = arith.trunci %curr_m_259 {async_task_id = array<i32: 4>} : i64 to i1
        %curr_m_261 = arith.cmpi sgt, %num_steps, %c0_i32_237 : i32
        %m_262 = arith.andi %tile_idx_246, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
        %m_263 = arith.trunci %m_262 {async_task_id = array<i32: 4>} : i64 to i1
        %qkT_264 = arith.andi %tile_idx_246, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
        %qkT_265 = arith.trunci %qkT_264 {async_task_id = array<i32: 4>} : i64 to i1
        %dv_266 = arith.andi %tile_idx_246, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
        %dv_267 = arith.trunci %dv_266 {async_task_id = array<i32: 4>} : i64 to i1
        %m_268 = ttg.memdesc_index %m_191[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128xf32, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
        %m_269 = ttg.memdesc_index %m_190[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %m_270 = arith.extui %m_263 {async_task_id = array<i32: 4>} : i1 to i32
        ttng.wait_barrier %m_269, %m_270, %curr_m_261 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %m_271 = ttg.local_load %m_268 {async_task_id = array<i32: 4>} : !ttg.memdesc<128xf32, #shared3, #smem, mutable> -> tensor<128xf32, #blocked3>
        %m_272 = ttg.memdesc_index %m_208[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.arrive_barrier %m_272, 1, %curr_m_261 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %pT = ttg.convert_layout %m_271 {async_task_id = array<i32: 4>} : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear8}>>
        %pT_273 = tt.expand_dims %pT {async_task_id = array<i32: 4>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear8}>> -> tensor<1x128xf32, #linear8>
        %pT_274 = tt.broadcast %pT_273 {async_task_id = array<i32: 4>} : tensor<1x128xf32, #linear8> -> tensor<128x128xf32, #linear8>
        %qkT_275 = ttg.memdesc_index %qkT_153[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qkT_276 = ttg.memdesc_index %qkT_155[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %qkT_277 = arith.extui %qkT_265 {async_task_id = array<i32: 4>} : i1 to i32
        ttng.wait_barrier %qkT_276, %qkT_277, %curr_m_261 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "forward", dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %qkT_278, %qkT_279 = ttng.tmem_load %qkT_275[] {async_task_id = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear8>
        %qkT_280 = ttg.memdesc_index %qkT_210[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.arrive_barrier %qkT_280, 1, %curr_m_261 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %pT_281 = arith.subf %qkT_278, %pT_274 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8>
        %pT_282 = math.exp2 %pT_281 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8>
        %ppT_283 = arith.truncf %pT_282 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8> to tensor<128x128xf16, #linear8>
        %qkT_284 = ttng.tmem_subslice %qkT_153 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x128>
        %qkT_285 = ttg.memdesc_reinterpret %qkT_284 {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x64xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem4, #ttng.tensor_memory, mutable>
        %ppT_286 = ttg.memdesc_index %qkT_285[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %qkT_287 = ttg.memdesc_index %qkT_210[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dv_288 = arith.extui %dv_267 : i1 to i32
        ttng.wait_barrier %qkT_287, %dv_288, %curr_m_261 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {dstTask = 4 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tmem_store %ppT_283, %ppT_286, %curr_m_261 {async_task_id = array<i32: 4>} : tensor<128x128xf16, #linear8> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %ppT_289 = ttg.memdesc_index %ppT_211[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.arrive_barrier %ppT_289, 1, %curr_m_261 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dpT_290 = arith.andi %tile_idx_246, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
        %dpT_291 = arith.trunci %dpT_290 {async_task_id = array<i32: 4>} : i64 to i1
        %Di_292 = arith.andi %tile_idx_246, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
        %Di_293 = arith.trunci %Di_292 {async_task_id = array<i32: 4>} : i64 to i1
        %Di_294 = ttg.memdesc_index %Di_196[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128xf32, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
        %Di_295 = ttg.memdesc_index %Di_195[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %Di_296 = arith.extui %Di_293 {async_task_id = array<i32: 4>} : i1 to i32
        ttng.wait_barrier %Di_295, %Di_296, %curr_m_261 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %Di_297 = ttg.local_load %Di_294 {async_task_id = array<i32: 4>} : !ttg.memdesc<128xf32, #shared3, #smem, mutable> -> tensor<128xf32, #blocked3>
        %Di_298 = ttg.memdesc_index %Di_216[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.arrive_barrier %Di_298, 1, %curr_m_261 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dsT_299 = ttg.convert_layout %Di_297 {async_task_id = array<i32: 4>} : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear8}>>
        %dsT_300 = tt.expand_dims %dsT_299 {async_task_id = array<i32: 4>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear8}>> -> tensor<1x128xf32, #linear8>
        %dsT_301 = tt.broadcast %dsT_300 {async_task_id = array<i32: 4>} : tensor<1x128xf32, #linear8> -> tensor<128x128xf32, #linear8>
        %dpT_302 = ttg.memdesc_index %dpT_160[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dpT_303 = ttg.memdesc_index %dpT_162[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dpT_304 = arith.extui %dpT_291 {async_task_id = array<i32: 4>} : i1 to i32
        ttng.wait_barrier %dpT_303, %dpT_304, %curr_m_261 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "forward", dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dpT_305, %dpT_306 = ttng.tmem_load %dpT_302[] {async_task_id = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear8>
        %dpT_307 = ttg.memdesc_index %dpT_214[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.arrive_barrier %dpT_307, 1, %curr_m_261 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dsT_308 = arith.subf %dpT_305, %dsT_301 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8>
        %dsT_309 = arith.mulf %pT_282, %dsT_308 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8>
        %dsT_310 = arith.truncf %dsT_309 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8> to tensor<128x128xf16, #linear8>
        %dpT_311 = ttng.tmem_subslice %dpT_160 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x128>
        %dpT_312 = ttg.memdesc_reinterpret %dpT_311 {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x64xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem4, #ttng.tensor_memory, mutable>
        %dsT_313 = ttg.memdesc_index %dpT_312[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %dk_314 = arith.andi %tile_idx_246, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
        %dk_315 = arith.trunci %dk_314 {async_task_id = array<i32: 4>} : i64 to i1
        %dsT_316 = ttg.memdesc_index %dsT_171[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dk_317 = arith.xori %dk_315, %true {async_task_id = array<i32: 4>} : i1
        %dk_318 = arith.extui %dk_317 {async_task_id = array<i32: 4>} : i1 to i32
        ttng.wait_barrier %dsT_316, %dk_318, %curr_m_261 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.tmem_store %dsT_310, %dsT_313, %curr_m_261 {async_task_id = array<i32: 4>} : tensor<128x128xf16, #linear8> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %dsT_319 = ttg.memdesc_index %dsT_217[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.arrive_barrier %dsT_319, 1, %curr_m_261 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %curr_m_320 = arith.subi %num_steps, %c1_i32_236 : i32
        %curr_m_321:2 = scf.for %curr_m_431 = %c0_i32_237 to %curr_m_320 step %c1_i32_236 iter_args(%tile_idx_432 = %tile_idx_246, %dsT_433 = %dsT_310) -> (i64, tensor<128x128xf16, #linear8>)  : i32 {
          %dsT_dq = tt.reshape %dsT_433 {async_task_id = array<i32: 4>} : tensor<128x128xf16, #linear8> -> tensor<128x2x64xf16, #linear10>
          %dsT_dq_434 = tt.trans %dsT_dq {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf16, #linear10> -> tensor<128x64x2xf16, #linear11>
          %dsT_dq_435, %dsT_dq_436 = tt.split %dsT_dq_434 {async_task_id = array<i32: 4>} : tensor<128x64x2xf16, #linear11> -> tensor<128x64xf16, #linear12>
          %dsT_dq_1_437 = ttg.memdesc_index %dsT_dq_1_181[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %dsT_dq_438 = arith.andi %tile_idx_432, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
          %dsT_dq_439 = arith.trunci %dsT_dq_438 {async_task_id = array<i32: 4>} : i64 to i1
          %dsT_dq_1_440 = ttg.memdesc_index %dsT_dq_1_220[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_dq_441 = arith.xori %dsT_dq_439, %true : i1
          %dsT_dq_442 = arith.extui %dsT_dq_441 : i1 to i32
          ttng.wait_barrier %dsT_dq_1_440, %dsT_dq_442 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttg.local_store %dsT_dq_435, %dsT_dq_1_437 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #linear12> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %dsT_dq_1_443 = ttg.memdesc_index %dsT_dq_1_219[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %dsT_dq_1_443, 1 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_dq_444 = ttng.two_cta_peer_gather %dsT_433 split_dim = 1 num_ctas = 2 {async_task_id = array<i32: 4>} : tensor<128x128xf16, #linear8> -> tensor<64x256xf16, #linear13>
          %dsT_dq_445 = tt.trans %dsT_dq_444 {async_task_id = array<i32: 4>, order = array<i32: 1, 0>} : tensor<64x256xf16, #linear13> -> tensor<256x64xf16, #linear14>
          %dsT_dq_0_446 = ttg.memdesc_index %dsT_dq_0_172[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %dsT_dq_447 = arith.andi %tile_idx_432, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
          %dsT_dq_448 = arith.trunci %dsT_dq_447 {async_task_id = array<i32: 4>} : i64 to i1
          %dsT_dq_0_449 = ttg.memdesc_index %dsT_dq_0_174[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_dq_450 = arith.xori %dsT_dq_448, %true {async_task_id = array<i32: 4>} : i1
          %dsT_dq_451 = arith.extui %dsT_dq_450 {async_task_id = array<i32: 4>} : i1 to i32
          ttng.wait_barrier %dsT_dq_0_449, %dsT_dq_451 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttg.local_store %dsT_dq_445, %dsT_dq_0_446 {async_task_id = array<i32: 4>} : tensor<256x64xf16, #linear14> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %dsT_dq_0_452 = ttg.memdesc_index %dsT_dq_0_221[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %dsT_dq_0_452, 1 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %accum_cnt_453 = arith.addi %tile_idx_432, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
          %m_454 = arith.andi %accum_cnt_453, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
          %m_455 = arith.trunci %m_454 {async_task_id = array<i32: 4>} : i64 to i1
          %qkT_456 = arith.andi %accum_cnt_453, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
          %qkT_457 = arith.trunci %qkT_456 {async_task_id = array<i32: 4>} : i64 to i1
          %dv_458 = arith.andi %accum_cnt_453, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
          %dv_459 = arith.trunci %dv_458 {async_task_id = array<i32: 4>} : i64 to i1
          %m_460 = ttg.memdesc_index %m_191[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128xf32, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
          %m_461 = ttg.memdesc_index %m_190[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %m_462 = arith.extui %m_455 {async_task_id = array<i32: 4>} : i1 to i32
          ttng.wait_barrier %m_461, %m_462, %true {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %m_463 = ttg.local_load %m_460 {async_task_id = array<i32: 4>} : !ttg.memdesc<128xf32, #shared3, #smem, mutable> -> tensor<128xf32, #blocked3>
          %m_464 = ttg.memdesc_index %m_208[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %m_464, 1, %true {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %pT_465 = ttg.convert_layout %m_463 {async_task_id = array<i32: 4>} : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear8}>>
          %pT_466 = tt.expand_dims %pT_465 {async_task_id = array<i32: 4>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear8}>> -> tensor<1x128xf32, #linear8>
          %pT_467 = tt.broadcast %pT_466 {async_task_id = array<i32: 4>} : tensor<1x128xf32, #linear8> -> tensor<128x128xf32, #linear8>
          %qkT_468 = ttg.memdesc_index %qkT_153[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %qkT_469 = ttg.memdesc_index %qkT_155[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %qkT_470 = arith.extui %qkT_457 {async_task_id = array<i32: 4>} : i1 to i32
          ttng.wait_barrier %qkT_469, %qkT_470, %true {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "forward", dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %qkT_471, %qkT_472 = ttng.tmem_load %qkT_468[] {async_task_id = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear8>
          %qkT_473 = ttg.memdesc_index %qkT_210[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %qkT_473, 1, %true {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %pT_474 = arith.subf %qkT_471, %pT_467 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8>
          %pT_475 = math.exp2 %pT_474 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8>
          %ppT_476 = arith.truncf %pT_475 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8> to tensor<128x128xf16, #linear8>
          %qkT_477 = ttng.tmem_subslice %qkT_153 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x128>
          %qkT_478 = ttg.memdesc_reinterpret %qkT_477 {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x64xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem4, #ttng.tensor_memory, mutable>
          %ppT_479 = ttg.memdesc_index %qkT_478[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %qkT_480 = ttg.memdesc_index %qkT_210[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dv_481 = arith.extui %dv_459 : i1 to i32
          ttng.wait_barrier %qkT_480, %dv_481, %true {async_task_id = array<i32: 4>, constraints = {WSBarrier = {dstTask = 4 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.tmem_store %ppT_476, %ppT_479, %true {async_task_id = array<i32: 4>} : tensor<128x128xf16, #linear8> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %ppT_482 = ttg.memdesc_index %ppT_211[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %ppT_482, 1, %true {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dpT_483 = arith.andi %accum_cnt_453, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
          %dpT_484 = arith.trunci %dpT_483 {async_task_id = array<i32: 4>} : i64 to i1
          %Di_485 = arith.andi %accum_cnt_453, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
          %Di_486 = arith.trunci %Di_485 {async_task_id = array<i32: 4>} : i64 to i1
          %Di_487 = ttg.memdesc_index %Di_196[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128xf32, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared3, #smem, mutable>
          %Di_488 = ttg.memdesc_index %Di_195[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %Di_489 = arith.extui %Di_486 {async_task_id = array<i32: 4>} : i1 to i32
          ttng.wait_barrier %Di_488, %Di_489, %true {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %Di_490 = ttg.local_load %Di_487 {async_task_id = array<i32: 4>} : !ttg.memdesc<128xf32, #shared3, #smem, mutable> -> tensor<128xf32, #blocked3>
          %Di_491 = ttg.memdesc_index %Di_216[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %Di_491, 1, %true {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_492 = ttg.convert_layout %Di_490 {async_task_id = array<i32: 4>} : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear8}>>
          %dsT_493 = tt.expand_dims %dsT_492 {async_task_id = array<i32: 4>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear8}>> -> tensor<1x128xf32, #linear8>
          %dsT_494 = tt.broadcast %dsT_493 {async_task_id = array<i32: 4>} : tensor<1x128xf32, #linear8> -> tensor<128x128xf32, #linear8>
          %dpT_495 = ttg.memdesc_index %dpT_160[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %dpT_496 = ttg.memdesc_index %dpT_162[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dpT_497 = arith.extui %dpT_484 {async_task_id = array<i32: 4>} : i1 to i32
          ttng.wait_barrier %dpT_496, %dpT_497, %true {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "forward", dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dpT_498, %dpT_499 = ttng.tmem_load %dpT_495[] {async_task_id = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear8>
          %dpT_500 = ttg.memdesc_index %dpT_214[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %dpT_500, 1, %true {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_501 = arith.subf %dpT_498, %dsT_494 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8>
          %dsT_502 = arith.mulf %pT_475, %dsT_501 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8>
          %dsT_503 = arith.truncf %dsT_502 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8> to tensor<128x128xf16, #linear8>
          %dpT_504 = ttng.tmem_subslice %dpT_160 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x128>
          %dpT_505 = ttg.memdesc_reinterpret %dpT_504 {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x64xf32, #tmem3, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem4, #ttng.tensor_memory, mutable>
          %dsT_506 = ttg.memdesc_index %dpT_505[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf16, #tmem4, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %dk_507 = arith.andi %accum_cnt_453, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
          %dk_508 = arith.trunci %dk_507 {async_task_id = array<i32: 4>} : i64 to i1
          %dsT_509 = ttg.memdesc_index %dsT_171[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dk_510 = arith.xori %dk_508, %true {async_task_id = array<i32: 4>} : i1
          %dk_511 = arith.extui %dk_510 {async_task_id = array<i32: 4>} : i1 to i32
          ttng.wait_barrier %dsT_509, %dk_511, %true {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.tmem_store %dsT_503, %dsT_506, %true {async_task_id = array<i32: 4>} : tensor<128x128xf16, #linear8> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
          %dsT_512 = ttg.memdesc_index %dsT_217[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %dsT_512, 1, %true {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          scf.yield %accum_cnt_453, %dsT_503 : i64, tensor<128x128xf16, #linear8>
        } {async_task_id = array<i32: 4>, tt.warp_specialize}
        %curr_m_322 = arith.cmpi sgt, %num_steps, %c0_i32_237 : i32
        %curr_m_323:2 = scf.if %curr_m_322 -> (i64, tensor<128x128xf16, #linear8>) {
          %dsT_dq = tt.reshape %curr_m_321#1 {async_task_id = array<i32: 4>} : tensor<128x128xf16, #linear8> -> tensor<128x2x64xf16, #linear10>
          %dsT_dq_431 = tt.trans %dsT_dq {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf16, #linear10> -> tensor<128x64x2xf16, #linear11>
          %dsT_dq_432, %dsT_dq_433 = tt.split %dsT_dq_431 {async_task_id = array<i32: 4>} : tensor<128x64x2xf16, #linear11> -> tensor<128x64xf16, #linear12>
          %dsT_dq_1_434 = ttg.memdesc_index %dsT_dq_1_181[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %dsT_dq_435 = arith.andi %curr_m_321#0, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
          %dsT_dq_436 = arith.trunci %dsT_dq_435 {async_task_id = array<i32: 4>} : i64 to i1
          %dsT_dq_1_437 = ttg.memdesc_index %dsT_dq_1_220[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_dq_438 = arith.xori %dsT_dq_436, %true : i1
          %dsT_dq_439 = arith.extui %dsT_dq_438 : i1 to i32
          ttng.wait_barrier %dsT_dq_1_437, %dsT_dq_439 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttg.local_store %dsT_dq_432, %dsT_dq_1_434 {async_task_id = array<i32: 4>} : tensor<128x64xf16, #linear12> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %dsT_dq_1_440 = ttg.memdesc_index %dsT_dq_1_219[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %dsT_dq_1_440, 1 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 2>, dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_dq_441 = ttng.two_cta_peer_gather %curr_m_321#1 split_dim = 1 num_ctas = 2 {async_task_id = array<i32: 4>} : tensor<128x128xf16, #linear8> -> tensor<64x256xf16, #linear13>
          %dsT_dq_442 = tt.trans %dsT_dq_441 {async_task_id = array<i32: 4>, order = array<i32: 1, 0>} : tensor<64x256xf16, #linear13> -> tensor<256x64xf16, #linear14>
          %dsT_dq_0_443 = ttg.memdesc_index %dsT_dq_0_172[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %dsT_dq_444 = arith.andi %curr_m_321#0, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
          %dsT_dq_445 = arith.trunci %dsT_dq_444 {async_task_id = array<i32: 4>} : i64 to i1
          %dsT_dq_0_446 = ttg.memdesc_index %dsT_dq_0_174[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %dsT_dq_447 = arith.xori %dsT_dq_445, %true {async_task_id = array<i32: 4>} : i1
          %dsT_dq_448 = arith.extui %dsT_dq_447 {async_task_id = array<i32: 4>} : i1 to i32
          ttng.wait_barrier %dsT_dq_0_446, %dsT_dq_448 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "backward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttg.local_store %dsT_dq_442, %dsT_dq_0_443 {async_task_id = array<i32: 4>} : tensor<256x64xf16, #linear14> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
          %dsT_dq_0_449 = ttg.memdesc_index %dsT_dq_0_221[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          ttng.arrive_barrier %dsT_dq_0_449, 1 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 1 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 2 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
          %accum_cnt_450 = arith.addi %curr_m_321#0, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
          scf.yield %accum_cnt_450, %0 : i64, tensor<128x128xf16, #linear8>
        } else {
          scf.yield %curr_m_321#0, %curr_m_321#1 : i64, tensor<128x128xf16, #linear8>
        }
        %dv_324 = ttg.memdesc_index %dv_163[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dv_325 = ttg.memdesc_index %dv_177[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dv_326 = arith.extui %curr_m_258 {async_task_id = array<i32: 4>} : i1 to i32
        ttng.wait_barrier %dv_325, %dv_326 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "forward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dv_327, %dv_328 = ttng.tmem_load %dv_324[] {async_task_id = array<i32: 4>, tmem.end = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear8>
        %dv_329 = ttg.memdesc_index %dv_204[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.arrive_barrier %dv_329, 1 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dvs = tt.reshape %dv_327 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8> -> tensor<128x2x64xf32, #linear10>
        %dvs_330 = tt.trans %dvs {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear10> -> tensor<128x64x2xf32, #linear11>
        %dvs_331, %dvs_332 = tt.split %dvs_330 {async_task_id = array<i32: 4>} : tensor<128x64x2xf32, #linear11> -> tensor<128x64xf32, #linear12>
        %dvs_333 = tt.reshape %dvs_331 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #linear12> -> tensor<128x2x32xf32, #linear15>
        %dvs_334 = tt.trans %dvs_333 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear15> -> tensor<128x32x2xf32, #linear16>
        %dvs_335, %dvs_336 = tt.split %dvs_334 {async_task_id = array<i32: 4>} : tensor<128x32x2xf32, #linear16> -> tensor<128x32xf32, #linear17>
        %dvs_337 = tt.reshape %dvs_335 {async_task_id = array<i32: 4>} : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
        %dvs_338 = tt.trans %dvs_337 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
        %dvs_339, %dvs_340 = tt.split %dvs_338 {async_task_id = array<i32: 4>} : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear9>
        %dvs_341 = tt.reshape %dvs_336 {async_task_id = array<i32: 4>} : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
        %dvs_342 = tt.trans %dvs_341 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
        %dvs_343, %dvs_344 = tt.split %dvs_342 {async_task_id = array<i32: 4>} : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear9>
        %dvs_345 = tt.reshape %dvs_332 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #linear12> -> tensor<128x2x32xf32, #linear15>
        %dvs_346 = tt.trans %dvs_345 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear15> -> tensor<128x32x2xf32, #linear16>
        %dvs_347, %dvs_348 = tt.split %dvs_346 {async_task_id = array<i32: 4>} : tensor<128x32x2xf32, #linear16> -> tensor<128x32xf32, #linear17>
        %dvs_349 = tt.reshape %dvs_347 {async_task_id = array<i32: 4>} : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
        %dvs_350 = tt.trans %dvs_349 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
        %dvs_351, %dvs_352 = tt.split %dvs_350 {async_task_id = array<i32: 4>} : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear9>
        %dvs_353 = tt.reshape %dvs_348 {async_task_id = array<i32: 4>} : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
        %dvs_354 = tt.trans %dvs_353 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
        %dvs_355, %dvs_356 = tt.split %dvs_354 {async_task_id = array<i32: 4>} : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear9>
        %4 = arith.truncf %dvs_339 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %5 = ttg.convert_layout %4 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dv_staging_357 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %5, %desc_dv_staging_357 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dv_staging_358 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %6 = ttng.async_tma_copy_local_to_global %desc_dv_200[%k_257, %c0_i32_237] %desc_dv_staging_358 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dv_staging_358   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %7 = arith.truncf %dvs_340 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %8 = ttg.convert_layout %7 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dv_staging_359 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %8, %desc_dv_staging_359 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dv_staging_360 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %9 = ttng.async_tma_copy_local_to_global %desc_dv_200[%k_257, %c16_i32_233] %desc_dv_staging_360 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dv_staging_360   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %10 = arith.truncf %dvs_343 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %11 = ttg.convert_layout %10 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dv_staging_361 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %11, %desc_dv_staging_361 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dv_staging_362 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %12 = ttng.async_tma_copy_local_to_global %desc_dv_200[%k_257, %c32_i32_232] %desc_dv_staging_362 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dv_staging_362   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %13 = arith.truncf %dvs_344 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %14 = ttg.convert_layout %13 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dv_staging_363 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %14, %desc_dv_staging_363 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dv_staging_364 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %15 = ttng.async_tma_copy_local_to_global %desc_dv_200[%k_257, %c48_i32_231] %desc_dv_staging_364 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dv_staging_364   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %16 = arith.truncf %dvs_351 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %17 = ttg.convert_layout %16 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dv_staging_365 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %17, %desc_dv_staging_365 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dv_staging_366 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %18 = ttng.async_tma_copy_local_to_global %desc_dv_200[%k_257, %kt_230] %desc_dv_staging_366 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dv_staging_366   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %19 = arith.truncf %dvs_352 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %20 = ttg.convert_layout %19 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dv_staging_367 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %20, %desc_dv_staging_367 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dv_staging_368 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %21 = ttng.async_tma_copy_local_to_global %desc_dv_200[%k_257, %c80_i32_229] %desc_dv_staging_368 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dv_staging_368   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %22 = arith.truncf %dvs_355 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %23 = ttg.convert_layout %22 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dv_staging_369 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %23, %desc_dv_staging_369 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dv_staging_370 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %24 = ttng.async_tma_copy_local_to_global %desc_dv_200[%k_257, %c96_i32_228] %desc_dv_staging_370 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dv_staging_370   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %25 = arith.truncf %dvs_356 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %26 = ttg.convert_layout %25 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dv_staging_371 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %26, %desc_dv_staging_371 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dv_staging_372 = ttg.memdesc_index %desc_dv_staging_199[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %27 = ttng.async_tma_copy_local_to_global %desc_dv_200[%k_257, %c112_i32_227] %desc_dv_staging_372 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dv_staging_372   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %dk_373 = ttg.memdesc_index %dk_167[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dk_374 = ttg.memdesc_index %dk_176[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dk_375 = arith.extui %curr_m_260 {async_task_id = array<i32: 4>} : i1 to i32
        ttng.wait_barrier %dk_374, %dk_375 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, direction = "forward", dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dk_376, %dk_377 = ttng.tmem_load %dk_373[] {async_task_id = array<i32: 4>, tmem.end = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear8>
        %dk_378 = ttg.memdesc_index %dk_206[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        ttng.arrive_barrier %dk_378, 1 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
        %dks = tt.reshape %dk_376 {async_task_id = array<i32: 4>} : tensor<128x128xf32, #linear8> -> tensor<128x2x64xf32, #linear10>
        %dks_379 = tt.trans %dks {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear10> -> tensor<128x64x2xf32, #linear11>
        %dks_380, %dks_381 = tt.split %dks_379 {async_task_id = array<i32: 4>} : tensor<128x64x2xf32, #linear11> -> tensor<128x64xf32, #linear12>
        %dks_382 = tt.reshape %dks_380 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #linear12> -> tensor<128x2x32xf32, #linear15>
        %dks_383 = tt.trans %dks_382 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear15> -> tensor<128x32x2xf32, #linear16>
        %dks_384, %dks_385 = tt.split %dks_383 {async_task_id = array<i32: 4>} : tensor<128x32x2xf32, #linear16> -> tensor<128x32xf32, #linear17>
        %dks_386 = tt.reshape %dks_384 {async_task_id = array<i32: 4>} : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
        %dks_387 = tt.trans %dks_386 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
        %dks_388, %dks_389 = tt.split %dks_387 {async_task_id = array<i32: 4>} : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear9>
        %dks_390 = tt.reshape %dks_385 {async_task_id = array<i32: 4>} : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
        %dks_391 = tt.trans %dks_390 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
        %dks_392, %dks_393 = tt.split %dks_391 {async_task_id = array<i32: 4>} : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear9>
        %dks_394 = tt.reshape %dks_381 {async_task_id = array<i32: 4>} : tensor<128x64xf32, #linear12> -> tensor<128x2x32xf32, #linear15>
        %dks_395 = tt.trans %dks_394 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear15> -> tensor<128x32x2xf32, #linear16>
        %dks_396, %dks_397 = tt.split %dks_395 {async_task_id = array<i32: 4>} : tensor<128x32x2xf32, #linear16> -> tensor<128x32xf32, #linear17>
        %dks_398 = tt.reshape %dks_396 {async_task_id = array<i32: 4>} : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
        %dks_399 = tt.trans %dks_398 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
        %dks_400, %dks_401 = tt.split %dks_399 {async_task_id = array<i32: 4>} : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear9>
        %dks_402 = tt.reshape %dks_397 {async_task_id = array<i32: 4>} : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
        %dks_403 = tt.trans %dks_402 {async_task_id = array<i32: 4>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
        %dks_404, %dks_405 = tt.split %dks_403 {async_task_id = array<i32: 4>} : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear9>
        %dkN_406 = arith.mulf %dks_388, %dkN {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9>
        %28 = arith.truncf %dkN_406 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %29 = ttg.convert_layout %28 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dk_staging_407 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %29, %desc_dk_staging_407 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dk_staging_408 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %30 = ttng.async_tma_copy_local_to_global %desc_dk_202[%k_257, %c0_i32_237] %desc_dk_staging_408 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dk_staging_408   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %dkN_409 = arith.mulf %dks_389, %dkN {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9>
        %31 = arith.truncf %dkN_409 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %32 = ttg.convert_layout %31 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dk_staging_410 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %32, %desc_dk_staging_410 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dk_staging_411 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %33 = ttng.async_tma_copy_local_to_global %desc_dk_202[%k_257, %c16_i32_233] %desc_dk_staging_411 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dk_staging_411   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %dkN_412 = arith.mulf %dks_392, %dkN {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9>
        %34 = arith.truncf %dkN_412 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %35 = ttg.convert_layout %34 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dk_staging_413 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %35, %desc_dk_staging_413 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dk_staging_414 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %36 = ttng.async_tma_copy_local_to_global %desc_dk_202[%k_257, %c32_i32_232] %desc_dk_staging_414 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dk_staging_414   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %dkN_415 = arith.mulf %dks_393, %dkN {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9>
        %37 = arith.truncf %dkN_415 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %38 = ttg.convert_layout %37 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dk_staging_416 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %38, %desc_dk_staging_416 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dk_staging_417 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %39 = ttng.async_tma_copy_local_to_global %desc_dk_202[%k_257, %c48_i32_231] %desc_dk_staging_417 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dk_staging_417   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %dkN_418 = arith.mulf %dks_400, %dkN {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9>
        %40 = arith.truncf %dkN_418 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %41 = ttg.convert_layout %40 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dk_staging_419 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %41, %desc_dk_staging_419 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dk_staging_420 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %42 = ttng.async_tma_copy_local_to_global %desc_dk_202[%k_257, %kt_230] %desc_dk_staging_420 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dk_staging_420   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %dkN_421 = arith.mulf %dks_401, %dkN {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9>
        %43 = arith.truncf %dkN_421 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %44 = ttg.convert_layout %43 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dk_staging_422 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %44, %desc_dk_staging_422 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dk_staging_423 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %45 = ttng.async_tma_copy_local_to_global %desc_dk_202[%k_257, %c80_i32_229] %desc_dk_staging_423 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dk_staging_423   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %dkN_424 = arith.mulf %dks_404, %dkN {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9>
        %46 = arith.truncf %dkN_424 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %47 = ttg.convert_layout %46 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dk_staging_425 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %47, %desc_dk_staging_425 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dk_staging_426 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %48 = ttng.async_tma_copy_local_to_global %desc_dk_202[%k_257, %c96_i32_228] %desc_dk_staging_426 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dk_staging_426   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %dkN_427 = arith.mulf %dks_405, %dkN {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9>
        %49 = arith.truncf %dkN_427 {async_task_id = array<i32: 4>} : tensor<128x16xf32, #linear9> to tensor<128x16xf16, #linear9>
        %50 = ttg.convert_layout %49 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #linear9> -> tensor<128x16xf16, #blocked4>
        %desc_dk_staging_428 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        ttg.local_store %50, %desc_dk_staging_428 {async_task_id = array<i32: 4>} : tensor<128x16xf16, #blocked4> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %desc_dk_staging_429 = ttg.memdesc_index %desc_dk_staging_201[%c0_i32_237] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %51 = ttng.async_tma_copy_local_to_global %desc_dk_202[%k_257, %c112_i32_227] %desc_dk_staging_429 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<128x16xf16, #shared2, #smem, mutable> -> !ttg.async.token
        nvws.tma_store_wait %desc_dk_staging_429   {async_task_id = array<i32: 4>, can_rotate_by_buffer_count = 1 : i32} : !ttg.memdesc<128x16xf16, #shared2, #smem, mutable>
        %tile_idx_430 = arith.addi %prog_id_244, %num_progs_241 {async_task_id = array<i32: 4>} : i32
        %accum_cnt = arith.addi %tile_idx_245, %c1_i64_225 {async_task_id = array<i32: 4>} : i64
        scf.yield {async_task_id = array<i32: 4>} %tile_idx_430, %accum_cnt, %curr_m_323#0 : i32, i64, i64
      } {async_task_id = array<i32: 4>, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "relay", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    } : (i32, i32, i32, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<2x64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, i32, i32, i32, !tt.tensordesc<128x128xf16, #shared>, !tt.tensordesc<256x64xf16, #shared>, !tt.tensordesc<128x128xf16, #shared>, !tt.tensordesc<128x64xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, !tt.tensordesc<128xf32, #shared3>, !tt.tensordesc<128x64xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x128xf32, #shared3, #smem, mutable>, !tt.tensordesc<128xf32, #shared3>, f32, !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>, !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<1x128x16xf16, #shared2, #smem, mutable>, !tt.tensordesc<128x16xf16, #shared2>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared4, #smem, mutable>) -> ()
    %k_96 = ttg.memdesc_index %k_32[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %k_96 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %kt_97 = ttg.memdesc_index %kt_29[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %kt_97 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %v_98 = ttg.memdesc_index %v_26[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %v_98 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qT_99 = ttg.memdesc_index %qT_17[%c0_i32] : !ttg.memdesc<2x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %qT_99 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qT_100 = ttg.memdesc_index %qT_17[%c1_i32] : !ttg.memdesc<2x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %qT_100 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qT_101 = ttg.memdesc_index %qT[%c0_i32] : !ttg.memdesc<2x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %qT_101 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qT_102 = ttg.memdesc_index %qT[%c1_i32] : !ttg.memdesc<2x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %qT_102 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qkT_103 = ttg.memdesc_index %qkT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %qkT_103 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %ppT_104 = ttg.memdesc_index %ppT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %ppT_104 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dpT_105 = ttg.memdesc_index %dpT_7[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dpT_105 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dpT_106 = ttg.memdesc_index %dpT_5[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dpT_106 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dpT_107 = ttg.memdesc_index %dpT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dpT_107 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %do_108 = ttg.memdesc_index %do[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %do_108 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %do_109 = ttg.memdesc_index %do_11[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %do_109 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %q_110 = ttg.memdesc_index %q[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %q_110 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %q_111 = ttg.memdesc_index %q_21[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %q_111 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT_112 = ttg.memdesc_index %dsT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dsT_112 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT_dq_0_113 = ttg.memdesc_index %dsT_dq_0[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dsT_dq_0_113 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dq_114 = ttg.memdesc_index %dq[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dq_114 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dk_115 = ttg.memdesc_index %dk[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dk_115 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dv_116 = ttg.memdesc_index %dv[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dv_116 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %v_117 = ttg.memdesc_index %v[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %v_117 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %kt_118 = ttg.memdesc_index %kt[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %kt_118 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %k_119 = ttg.memdesc_index %k[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %k_119 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %m_120 = ttg.memdesc_index %m[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %m_120 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %Di_121 = ttg.memdesc_index %Di[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %Di_121 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dv_122 = ttg.memdesc_index %dv_34[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dv_122 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dv_123 = ttg.memdesc_index %dv_35[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dv_123 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dk_124 = ttg.memdesc_index %dk_38[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dk_124 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dk_125 = ttg.memdesc_index %dk_39[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dk_125 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %m_126 = ttg.memdesc_index %m_42[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %m_126 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %m_127 = ttg.memdesc_index %m_43[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %m_127 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qkT_128 = ttg.memdesc_index %qkT_46[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %qkT_128 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %qkT_129 = ttg.memdesc_index %qkT_47[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %qkT_129 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %ppT_130 = ttg.memdesc_index %ppT_50[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %ppT_130 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %ppT_131 = ttg.memdesc_index %ppT_51[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %ppT_131 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dpT_132 = ttg.memdesc_index %dpT_54[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dpT_132 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dpT_133 = ttg.memdesc_index %dpT_55[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dpT_133 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %Di_134 = ttg.memdesc_index %Di_58[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %Di_134 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %Di_135 = ttg.memdesc_index %Di_59[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %Di_135 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT_136 = ttg.memdesc_index %dsT_62[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dsT_136 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT_137 = ttg.memdesc_index %dsT_63[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dsT_137 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT_dq_1_138 = ttg.memdesc_index %dsT_dq_1[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dsT_dq_1_138 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT_dq_1_139 = ttg.memdesc_index %dsT_dq_1_66[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dsT_dq_1_139 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT_dq_0_140 = ttg.memdesc_index %dsT_dq_0_69[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dsT_dq_0_140 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dsT_dq_0_141 = ttg.memdesc_index %dsT_dq_0_70[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dsT_dq_0_141 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dq_142 = ttg.memdesc_index %dq_73[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dq_142 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    %dq_143 = ttg.memdesc_index %dq_74[%c0_i32] : !ttg.memdesc<1x1xi64, #shared4, #smem, mutable> -> !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    ttng.inval_barrier %dq_143 : !ttg.memdesc<1xi64, #shared4, #smem, mutable>
    tt.return
  }
}
