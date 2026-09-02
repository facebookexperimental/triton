#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [8, 1, 4], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 32]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 1, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 0, 1]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 64], [0, 32]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0], [0, 1, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0], [0, 0, 1]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 0, 1], [0, 16, 0], [0, 1, 0], [0, 2, 0], [64, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0], [32, 0, 0]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 16], [0, 1], [0, 2], [64, 0]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [32, 0]], block = []}>
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
  tt.func public @_attn_bwd(%desc_q: !tt.tensordesc<128x64xf16, #shared>, %desc_q.shape.0: i32, %desc_q.shape.1: i32, %desc_q.stride.0: i64, %desc_q.stride.1: i64, %desc_qt: !tt.tensordesc<64x128xf16, #shared>, %desc_qt.shape.0: i32, %desc_qt.shape.1: i32, %desc_qt.stride.0: i64, %desc_qt.stride.1: i64, %desc_k: !tt.tensordesc<128x128xf16, #shared>, %desc_k.shape.0: i32, %desc_k.shape.1: i32, %desc_k.stride.0: i64, %desc_k.stride.1: i64, %desc_kt: !tt.tensordesc<256x64xf16, #shared>, %desc_kt.shape.0: i32, %desc_kt.shape.1: i32, %desc_kt.stride.0: i64, %desc_kt.stride.1: i64, %desc_v: !tt.tensordesc<128x128xf16, #shared>, %desc_v.shape.0: i32, %desc_v.shape.1: i32, %desc_v.stride.0: i64, %desc_v.stride.1: i64, %sm_scale: f32, %desc_do: !tt.tensordesc<128x64xf16, #shared>, %desc_do.shape.0: i32, %desc_do.shape.1: i32, %desc_do.stride.0: i64, %desc_do.stride.1: i64, %desc_dot: !tt.tensordesc<64x128xf16, #shared>, %desc_dot.shape.0: i32, %desc_dot.shape.1: i32, %desc_dot.stride.0: i64, %desc_dot.stride.1: i64, %desc_dq: !tt.tensordesc<128x16xf32, #shared1>, %desc_dq.shape.0: i32, %desc_dq.shape.1: i32, %desc_dq.stride.0: i64, %desc_dq.stride.1: i64, %desc_dk: !tt.tensordesc<128x64xf16, #shared>, %desc_dk.shape.0: i32, %desc_dk.shape.1: i32, %desc_dk.stride.0: i64, %desc_dk.stride.1: i64, %desc_dv: !tt.tensordesc<128x64xf16, #shared>, %desc_dv.shape.0: i32, %desc_dv.shape.1: i32, %desc_dv.stride.0: i64, %desc_dv.stride.1: i64, %desc_m: !tt.tensordesc<128xf32, #shared2>, %desc_m.shape.0: i32, %desc_m.stride.0: i64, %desc_delta: !tt.tensordesc<128xf32, #shared2>, %desc_delta.shape.0: i32, %desc_delta.stride.0: i64, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ub.poison : !ttg.async.token
    %c256_i32 = arith.constant 256 : i32
    %c7_i32 = arith.constant 7 : i32
    %c1_i64 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
    %c0_i64 = arith.constant {async_task_id = array<i32: 0>} 0 : i64
    %true = arith.constant {async_task_id = array<i32: 0>} true
    %c64_i32 = arith.constant {async_task_id = array<i32: 0>} 64 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0>} 128 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0>} 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %1 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %2 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %3 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %4 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %4, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %5 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %6 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %7 = ttg.memdesc_index %5[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %8 = ttg.memdesc_index %6[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %8, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %dq = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dq_0 = ttg.memdesc_index %dq[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dq_0, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT_dq_0 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dsT_dq_0_1 = ttg.memdesc_index %dsT_dq_0[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dsT_dq_0_1, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dsT_2 = ttg.memdesc_index %dsT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dsT_2, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %Di = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %Di_3 = ttg.memdesc_index %Di[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %Di_3, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dpT = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dpT_4 = ttg.memdesc_index %dpT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dpT_4, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dpT_5 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dpT_6 = ttg.memdesc_index %dpT_5[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dpT_6, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %ppT = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %ppT_7 = ttg.memdesc_index %ppT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %ppT_7, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %do = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %do_8 = ttg.memdesc_index %do[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %do_8, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %do_9 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %do_10 = ttg.memdesc_index %do_9[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %do_10, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %m = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %m_11 = ttg.memdesc_index %m[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %m_11, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %q = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %q_12 = ttg.memdesc_index %q[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %q_12, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %q_13 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %q_14 = ttg.memdesc_index %q_13[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %q_14, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %qT = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %qT_15 = ttg.memdesc_index %qT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %qT_15, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %qT_16 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %qT_17 = ttg.memdesc_index %qT_16[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %qT_17, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dk_18 = ttg.memdesc_index %dk[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dk_18, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_19 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dk_20 = ttg.memdesc_index %dk_19[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dk_20, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dv_21 = ttg.memdesc_index %dv[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dv_21, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv_22 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dv_23 = ttg.memdesc_index %dv_22[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dv_23, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dpT_24 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dpT_25 = ttg.memdesc_index %dpT_24[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dpT_25, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %qkT = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %qkT_26 = ttg.memdesc_index %qkT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %qkT_26, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %v = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %v_27 = ttg.memdesc_index %v[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %v_27, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %v_28 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %v_29 = ttg.memdesc_index %v_28[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %v_29, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %kt = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %kt_30 = ttg.memdesc_index %kt[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %kt_30, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %kt_31 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %kt_32 = ttg.memdesc_index %kt_31[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %kt_32, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %k = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %k_33 = ttg.memdesc_index %k[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %k_33, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %k_34 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %k_35 = ttg.memdesc_index %k_34[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %k_35, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %qkT_36 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %qkT_37 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %qkT_38 = ttg.memdesc_index %qkT_36[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %qkT_38, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %qkT_39 = ttg.memdesc_index %qkT_37[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %qkT_39, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %dpT_40 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dpT_41 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dpT_42 = ttg.memdesc_index %dpT_40[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dpT_42, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dpT_43 = ttg.memdesc_index %dpT_41[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dpT_43, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %dv_44 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dv_45 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dv_46 = ttg.memdesc_index %dv_44[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dv_46, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv_47 = ttg.memdesc_index %dv_45[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dv_47, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %dv_48 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dv_49 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dv_50 = ttg.memdesc_index %dv_48[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dv_50, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv_51 = ttg.memdesc_index %dv_49[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dv_51, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %dv_52 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dv_53 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dv_54 = ttg.memdesc_index %dv_52[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dv_54, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv_55 = ttg.memdesc_index %dv_53[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dv_55, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %dk_56 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dk_57 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dk_58 = ttg.memdesc_index %dk_56[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dk_58, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_59 = ttg.memdesc_index %dk_57[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dk_59, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %dk_60 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dk_61 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dk_62 = ttg.memdesc_index %dk_60[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dk_62, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_63 = ttg.memdesc_index %dk_61[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dk_63, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %dk_64 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dk_65 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dk_66 = ttg.memdesc_index %dk_64[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dk_66, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_67 = ttg.memdesc_index %dk_65[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dk_67, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %m_68 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %m_69 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %m_70 = ttg.memdesc_index %m_68[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %m_70, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %m_71 = ttg.memdesc_index %m_69[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %m_71, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %ppT_72 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %ppT_73 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %ppT_74 = ttg.memdesc_index %ppT_72[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %ppT_74, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %ppT_75 = ttg.memdesc_index %ppT_73[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %ppT_75, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %Di_76 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %Di_77 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %Di_78 = ttg.memdesc_index %Di_76[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %Di_78, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %Di_79 = ttg.memdesc_index %Di_77[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %Di_79, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %dsT_80 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dsT_81 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dsT_82 = ttg.memdesc_index %dsT_80[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dsT_82, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT_83 = ttg.memdesc_index %dsT_81[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dsT_83, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %dsT_dq_1 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dsT_dq_1_84 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dsT_dq_1_85 = ttg.memdesc_index %dsT_dq_1[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dsT_dq_1_85, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT_dq_1_86 = ttg.memdesc_index %dsT_dq_1_84[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dsT_dq_1_86, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %dsT_dq_0_87 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dsT_dq_0_88 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dsT_dq_0_89 = ttg.memdesc_index %dsT_dq_0_87[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dsT_dq_0_89, 2 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT_dq_0_90 = ttg.memdesc_index %dsT_dq_0_88[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dsT_dq_0_90, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %dq_91 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dq_92 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %dq_93 = ttg.memdesc_index %dq_91[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dq_93, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dq_94 = ttg.memdesc_index %dq_92[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.init_barrier %dq_94, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttg.barrier local
    %k_95 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %desc_dv_staging = ttg.memdesc_reinterpret %k_95 {allocation.shareGroup = 0 : i32, buffer.copy = 1 : i32, buffer.id = 26 : i32} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %kt_96 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 12 : i32} : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %v_97 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %qkT_98 = ttng.tmem_alloc {allocation.shareGroup = 1 : i32, buffer.copy = 1 : i32, buffer.id = 2 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %dpT_99 = ttng.tmem_alloc {allocation.shareGroup = 4 : i32, buffer.copy = 1 : i32, buffer.id = 5 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %dv_100 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %dk_101 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 10 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %qT_102 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 1 : i32} : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %desc_dk_staging = ttg.memdesc_reinterpret %qT_102 {allocation.shareGroup = 2 : i32, buffer.copy = 1 : i32, buffer.id = 28 : i32} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %q_103 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 15 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %m_104 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 16 : i32} : () -> !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>
    %do_105 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 17 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %dpT_106 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32} : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %Di_107 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 19 : i32} : () -> !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>
    %dsT_dq_1_108 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 20 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %dsT_dq_0_109 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 8 : i32} : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %dk_110 = ttg.memdesc_index %dk_101[%c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %dv_111 = ttg.memdesc_index %dv_100[%c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %desc_dq_reduce_staging = ttg.local_alloc {allocation.shareGroup = 3 : i32, buffer.copy = 1 : i32, buffer.id = 22 : i32, buffer.tmaStaging = 2 : i32} : () -> !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>
    %dv_112 = ttg.memdesc_index %dv_100[%c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %dk_113 = ttg.memdesc_index %dk_101[%c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %desc_dv_staging_114 = ttg.memdesc_index %desc_dv_staging[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %desc_dv_staging_115 = ttg.memdesc_index %desc_dv_staging[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %desc_dv_staging_116 = ttg.memdesc_index %desc_dv_staging[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %desc_dv_staging_117 = ttg.memdesc_index %desc_dv_staging[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %desc_dk_staging_118 = ttg.memdesc_index %desc_dk_staging[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %desc_dk_staging_119 = ttg.memdesc_index %desc_dk_staging[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %desc_dk_staging_120 = ttg.memdesc_index %desc_dk_staging[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %desc_dk_staging_121 = ttg.memdesc_index %desc_dk_staging[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttg.warp_specialize(%H, %stride_h, %stride_z, %stride_tok, %N_CTX, %dk_110, %dv_111, %dpT_99, %dq, %desc_dq_reduce_staging, %desc_dq, %k_34, %kt_31, %v_28, %qT_102, %qT_16, %k_95, %qkT_98, %qT, %qkT, %ppT, %dpT_106, %dpT_5, %v_97, %dpT, %dpT_24, %dv_100, %do_105, %do, %do_9, %dk_101, %q_103, %q, %q_13, %dsT, %dsT_dq_0_109, %kt_96, %dsT_dq_0, %dk, %dk_19, %dv, %dv_22, %v, %kt, %k, %desc_k, %desc_kt, %desc_v, %desc_qt, %desc_q, %m, %m_104, %desc_m, %desc_do, %desc_dot, %Di, %Di_107, %desc_delta, %dsT_dq_1_108, %1, %2, %5, %6, %qkT_36, %qkT_37, %dpT_40, %dpT_41, %dv_44, %dv_45, %dv_48, %dv_49, %dv_52, %dv_53, %dk_56, %dk_57, %dk_60, %dk_61, %dk_64, %dk_65, %m_68, %m_69, %ppT_72, %ppT_73, %Di_76, %Di_77, %dsT_80, %dsT_81, %dsT_dq_1, %dsT_dq_1_84, %dsT_dq_0_87, %dsT_dq_0_88, %dq_91, %dq_92) attributes {requestedRegisters = array<i32: 88, 88, 88, 40>, ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"]}
    default {
      %bhid = tt.get_program_id z {async_task_id = array<i32: 0>} : i32
      %pid = tt.get_program_id x {async_task_id = array<i32: 0>} : i32
      %off_bh = arith.remsi %bhid, %H {async_task_id = array<i32: 0>} : i32
      %off_bh_178 = arith.muli %stride_h, %off_bh {async_task_id = array<i32: 0>} : i32
      %off_bh_179 = arith.divsi %bhid, %H {async_task_id = array<i32: 0>} : i32
      %off_bh_180 = arith.muli %stride_z, %off_bh_179 {async_task_id = array<i32: 0>} : i32
      %off_bh_181 = arith.addi %off_bh_178, %off_bh_180 {async_task_id = array<i32: 0>} : i32
      %off_bh_182 = arith.extsi %off_bh_181 {async_task_id = array<i32: 0>} : i32 to i64
      %off_bh_183 = arith.extsi %stride_tok {async_task_id = array<i32: 0>} : i32 to i64
      %off_bh_184 = arith.divsi %off_bh_182, %off_bh_183 {async_task_id = array<i32: 0>} : i64
      %start_n = arith.muli %pid, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %k_185 = arith.extsi %start_n {async_task_id = array<i32: 0>} : i32 to i64
      %k_186 = arith.addi %off_bh_184, %k_185 {async_task_id = array<i32: 0>} : i64
      %k_187 = arith.trunci %k_186 {async_task_id = array<i32: 0>} : i64 to i32
      %num_steps = arith.divsi %N_CTX, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %curr_m = scf.for %curr_m_208 = %c0_i32 to %num_steps step %c1_i32 iter_args(%curr_m_209 = %c0_i64) -> (i64)  : i32 {
        %m_210 = arith.andi %curr_m_209, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %m_211 = arith.trunci %m_210 {async_task_id = array<i32: 0>} : i64 to i1
        %qkT_212 = arith.andi %curr_m_209, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %qkT_213 = arith.trunci %qkT_212 {async_task_id = array<i32: 0>} : i64 to i1
        %dv_214 = arith.andi %curr_m_209, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %dv_215 = arith.trunci %dv_214 {async_task_id = array<i32: 0>} : i64 to i1
        %m_216 = ttg.memdesc_index %m_104[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
        %m_217 = ttg.memdesc_index %m[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %m_218 = arith.extui %m_211 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %m_217, %m_218, %true {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %m_219 = ttg.local_load %m_216 {async_task_id = array<i32: 0>} : !ttg.memdesc<128xf32, #shared2, #smem, mutable> -> tensor<128xf32, #blocked>
        %m_220 = ttg.memdesc_index %m_69[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %m_220, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %pT = ttg.convert_layout %m_219 {async_task_id = array<i32: 0>} : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
        %pT_221 = tt.expand_dims %pT {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
        %pT_222 = tt.broadcast %pT_221 {async_task_id = array<i32: 0>} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %qkT_223 = ttg.memdesc_index %qkT_98[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qkT_224 = ttg.memdesc_index %qkT[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %qkT_225 = arith.extui %qkT_213 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %qkT_224, %qkT_225 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %qkT_226 = ttng.tmem_load %qkT_223 {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %qkT_227 = ttg.memdesc_index %qkT_37[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %qkT_227, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %pT_228 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            sub.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {async_task_id = array<i32: 0>, constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true} %qkT_226, %pT_222 : tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %pT_229 = math.exp2 %pT_228 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear>
        %ppT_230 = arith.truncf %pT_229 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %qkT_231 = ttng.tmem_subslice %qkT_98 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
        %qkT_232 = ttg.memdesc_reinterpret %qkT_231 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        %ppT_233 = ttg.memdesc_index %qkT_232[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %qkT_234 = ttg.memdesc_index %qkT_37[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dv_235 = arith.extui %dv_215 : i1 to i32
        ttng.wait_barrier %qkT_234, %dv_235 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {dstTask = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tmem_store %ppT_230, %ppT_233, %true {async_task_id = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %ppT_236 = ttg.memdesc_index %ppT_72[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %ppT_236, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dpT_237 = arith.andi %curr_m_209, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %dpT_238 = arith.trunci %dpT_237 {async_task_id = array<i32: 0>} : i64 to i1
        %Di_239 = arith.andi %curr_m_209, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %Di_240 = arith.trunci %Di_239 {async_task_id = array<i32: 0>} : i64 to i1
        %Di_241 = ttg.memdesc_index %Di_107[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
        %Di_242 = ttg.memdesc_index %Di[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %Di_243 = arith.extui %Di_240 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %Di_242, %Di_243, %true {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %Di_244 = ttg.local_load %Di_241 {async_task_id = array<i32: 0>} : !ttg.memdesc<128xf32, #shared2, #smem, mutable> -> tensor<128xf32, #blocked>
        %Di_245 = ttg.memdesc_index %Di_77[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %Di_245, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 3 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_246 = ttg.convert_layout %Di_244 {async_task_id = array<i32: 0>} : tensor<128xf32, #blocked> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>>
        %dsT_247 = tt.expand_dims %dsT_246 {async_task_id = array<i32: 0>, axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xf32, #linear>
        %dsT_248 = tt.broadcast %dsT_247 {async_task_id = array<i32: 0>} : tensor<1x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %dpT_249 = ttg.memdesc_index %dpT_99[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dpT_250 = ttg.memdesc_index %dpT_24[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dpT_251 = arith.extui %dpT_238 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %dpT_250, %dpT_251 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dpT_252 = ttng.tmem_load %dpT_249 {async_task_id = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        %dpT_253 = ttg.memdesc_index %dpT_41[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %dpT_253, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_254 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            sub.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {async_task_id = array<i32: 0>, constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true} %dpT_252, %dsT_248 : tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %dsT_255 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {async_task_id = array<i32: 0>, constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true} %pT_229, %dsT_254 : tensor<128x128xf32, #linear>, tensor<128x128xf32, #linear> -> tensor<128x128xf32, #linear>
        %dsT_256 = arith.truncf %dsT_255 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %dpT_257 = ttng.tmem_subslice %dpT_99 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
        %dpT_258 = ttg.memdesc_reinterpret %dpT_257 {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        %dsT_259 = ttg.memdesc_index %dpT_258[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %dk_260 = arith.andi %curr_m_209, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %dk_261 = arith.trunci %dk_260 {async_task_id = array<i32: 0>} : i64 to i1
        %dsT_262 = ttg.memdesc_index %dsT[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dk_263 = arith.xori %dk_261, %true {async_task_id = array<i32: 0>} : i1
        %dk_264 = arith.extui %dk_263 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %dsT_262, %dk_264 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tmem_store %dsT_256, %dsT_259, %true {async_task_id = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %dsT_265 = ttg.memdesc_index %dsT_80[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %dsT_265, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_1_266 = ttg.memdesc_index %dsT_dq_1_108[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %dsT_dq = arith.andi %curr_m_209, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %dsT_dq_267 = arith.trunci %dsT_dq {async_task_id = array<i32: 0>} : i64 to i1
        %dsT_dq_1_268 = ttg.memdesc_index %dsT_dq_1_84[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_269 = arith.xori %dsT_dq_267, %true : i1
        %dsT_dq_270 = arith.extui %dsT_dq_269 : i1 to i32
        ttng.wait_barrier %dsT_dq_1_268, %dsT_dq_270 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 4>, dstTask = 4 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_1_271 = ttg.memdesc_index %dsT_dq_1[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_0_272 = ttg.memdesc_index %dsT_dq_0_109[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        %dsT_dq_273 = arith.andi %curr_m_209, %c1_i64 {async_task_id = array<i32: 0>} : i64
        %dsT_dq_274 = arith.trunci %dsT_dq_273 {async_task_id = array<i32: 0>} : i64 to i1
        %dsT_dq_0_275 = ttg.memdesc_index %dsT_dq_0[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_276 = arith.xori %dsT_dq_274, %true {async_task_id = array<i32: 0>} : i1
        %dsT_dq_277 = arith.extui %dsT_dq_276 {async_task_id = array<i32: 0>} : i1 to i32
        ttng.wait_barrier %dsT_dq_0_275, %dsT_dq_277 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_278 = ttng.tmem_subslice %dsT_259 {dim = 1 : i32, offset = 0 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 128x128>
        %dsT_dq_279 = ttng.tmem_load %dsT_dq_278 : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf16, #linear1>
        %dsT_dq_280 = ttng.tmem_subslice %dsT_259 {dim = 1 : i32, offset = 64 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 128x128>
        %dsT_dq_281 = ttng.tmem_load %dsT_dq_280 : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf16, #linear1>
        %dsT_dq_282 = ttg.memdesc_subslice %dsT_dq_0_272[0, 0] : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64>
        %dsT_dq_283 = ttg.memdesc_subslice %dsT_dq_0_272[128, 0] : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64>
        ttng.barrier_expect %dsT_dq_1_271, 16384, %true : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_284 = nvg.cluster_id
        %dsT_dq_285 = arith.cmpi eq, %dsT_dq_284, %c0_i32 : i32
        scf.if %dsT_dq_285 {
          ttg.local_store %dsT_dq_279, %dsT_dq_282 : tensor<128x64xf16, #linear1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64>
          ttg.local_store %dsT_dq_281, %dsT_dq_1_266 : tensor<128x64xf16, #linear1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.wait_barrier_named %c7_i32, %c256_i32 : i32, i32
          ttng.fence_async_shared {bCluster = false}
          ttg.async_remote_shmem_copy %dsT_dq_1_266, rank %c1_i32, %dsT_dq_282 barrier %dsT_dq_1_271 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64> barrier_ty !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        } else {
          ttg.local_store %dsT_dq_281, %dsT_dq_283 : tensor<128x64xf16, #linear1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64>
          ttg.local_store %dsT_dq_279, %dsT_dq_1_266 : tensor<128x64xf16, #linear1> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.wait_barrier_named %c7_i32, %c256_i32 : i32, i32
          ttng.fence_async_shared {bCluster = false}
          ttg.async_remote_shmem_copy %dsT_dq_1_266, rank %c0_i32, %dsT_dq_283 barrier %dsT_dq_1_271 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64> barrier_ty !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        }
        %dsT_dq_0_286 = ttg.memdesc_index %dsT_dq_0_87[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %dsT_dq_0_286, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %accum_cnt = arith.addi %curr_m_209, %c1_i64 {async_task_id = array<i32: 0>} : i64
        scf.yield {async_task_id = array<i32: 0>} %accum_cnt : i64
      } {async_task_id = array<i32: 0>, tt.assume_nonempty, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      %dv_188 = ttg.memdesc_index %dv[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %dv_188, %c0_i32 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dv_189 = ttg.memdesc_index %dv_52[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %dv_189, %c0_i32 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dv_190 = ttng.tmem_load %dv_112 {async_task_id = array<i32: 0>, tmem.end = array<i32: 6, 7>, tmem.start = array<i32: 8>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %dv_191 = ttg.memdesc_index %dv_53[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.arrive_barrier %dv_191, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dv_192 = ttg.memdesc_index %dv_49[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.arrive_barrier %dv_192, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %13 = ttg.memdesc_index %6[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.arrive_barrier %13, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dk_193 = ttg.memdesc_index %dk[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %dk_193, %c0_i32 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, direction = "forward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dk_194 = ttg.memdesc_index %dk_64[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %dk_194, %c0_i32 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dk_195 = ttng.tmem_load %dk_113 {async_task_id = array<i32: 0>, tmem.end = array<i32: 10, 11>, tmem.start = array<i32: 12>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      %dk_196 = ttg.memdesc_index %dk_65[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.arrive_barrier %dk_196, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dk_197 = ttg.memdesc_index %dk_61[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.arrive_barrier %dk_197, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 2 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %14 = ttg.memdesc_index %2[%c0_i32] {async_task_id = array<i32: 0>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.arrive_barrier %14, 1 {async_task_id = array<i32: 0>, constraints = {WSBarrier = {channelGraph = array<i32: 1, 2, 3>, dstTask = 1 : i32, maxRegionId = 4 : i32, minRegionId = 4 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dvs = tt.reshape %dv_190 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear2>
      %dvs_198 = tt.trans %dvs {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
      %dvs_199 = ttg.convert_layout %dvs_198 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64x2xf32, #blocked1>
      %dvs_200, %dvs_201 = tt.split %dvs_199 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #blocked1> -> tensor<128x64xf32, #blocked2>
      %15 = arith.truncf %dvs_200 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
      ttg.local_store %15, %desc_dv_staging_114 {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked2> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %16 = ttng.async_tma_copy_local_to_global %desc_dv[%k_187, %c0_i32] %desc_dv_staging_115 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %16   {async_task_id = array<i32: 0>} : !ttg.async.token
      %17 = arith.truncf %dvs_201 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
      ttg.local_store %17, %desc_dv_staging_116 {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked2> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %18 = ttng.async_tma_copy_local_to_global %desc_dv[%k_187, %c64_i32] %desc_dv_staging_117 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %18   {async_task_id = array<i32: 0>} : !ttg.async.token
      %dks = tt.reshape %dk_195 {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> -> tensor<128x2x64xf32, #linear2>
      %dks_202 = tt.trans %dks {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
      %dks_203 = ttg.convert_layout %dks_202 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64x2xf32, #blocked1>
      %dks_204, %dks_205 = tt.split %dks_203 {async_task_id = array<i32: 0>} : tensor<128x64x2xf32, #blocked1> -> tensor<128x64xf32, #blocked2>
      %dkN = tt.splat %sm_scale {async_task_id = array<i32: 0>} : f32 -> tensor<128x64xf32, #blocked2>
      %dkN_206 = arith.mulf %dks_204, %dkN {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2>
      %19 = arith.truncf %dkN_206 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
      ttg.local_store %19, %desc_dk_staging_118 {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked2> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %20 = ttng.async_tma_copy_local_to_global %desc_dk[%k_187, %c0_i32] %desc_dk_staging_119 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %20   {async_task_id = array<i32: 0>} : !ttg.async.token
      %dkN_207 = arith.mulf %dks_205, %dkN {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2>
      %21 = arith.truncf %dkN_207 {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
      ttg.local_store %21, %desc_dk_staging_120 {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked2> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %22 = ttng.async_tma_copy_local_to_global %desc_dk[%k_187, %c64_i32] %desc_dk_staging_121 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %22   {async_task_id = array<i32: 0>} : !ttg.async.token
      ttg.warp_yield
    }
    partition0(%H_178: i32, %stride_h_179: i32, %stride_z_180: i32, %stride_tok_181: i32, %N_CTX_182: i32, %dk_183: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dv_184: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dpT_185: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dq_186: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %desc_dq_reduce_staging_187: !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, %desc_dq_188: !tt.tensordesc<128x16xf32, #shared1>, %k_189: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %kt_190: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %v_191: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qT_192: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %qT_193: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %k_194: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %qkT_195: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %qT_196: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qkT_197: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %ppT_198: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_199: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %dpT_200: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %v_201: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %dpT_202: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_203: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_204: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %do_205: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %do_206: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %do_207: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_208: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q_209: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %q_210: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %q_211: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_212: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_0_213: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %kt_214: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %dsT_dq_0_215: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_216: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_217: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_218: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_219: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %v_220: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %kt_221: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %k_222: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %desc_k_223: !tt.tensordesc<128x128xf16, #shared>, %desc_kt_224: !tt.tensordesc<256x64xf16, #shared>, %desc_v_225: !tt.tensordesc<128x128xf16, #shared>, %desc_qt_226: !tt.tensordesc<64x128xf16, #shared>, %desc_q_227: !tt.tensordesc<128x64xf16, #shared>, %m_228: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %m_229: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %desc_m_230: !tt.tensordesc<128xf32, #shared2>, %desc_do_231: !tt.tensordesc<128x64xf16, #shared>, %desc_dot_232: !tt.tensordesc<64x128xf16, #shared>, %Di_233: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %Di_234: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %desc_delta_235: !tt.tensordesc<128xf32, #shared2>, %dsT_dq_1_236: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg122: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qkT_237: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qkT_238: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_239: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_240: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_241: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_242: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_243: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_244: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_245: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_246: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_247: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_248: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_249: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_250: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_251: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_252: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %m_253: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %m_254: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %ppT_255: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %ppT_256: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %Di_257: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %Di_258: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_259: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_260: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_1_261: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_1_262: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_0_263: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_0_264: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dq_265: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dq_266: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) num_warps(8) {
      %c1_i64_267 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
      %curr_m = arith.constant {async_task_id = array<i32: 1>} 0 : i64
      %c0_i32_268 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
      %cst = arith.constant {async_task_id = array<i32: 1>} dense<0.693147182> : tensor<128x16xf32, #blocked3>
      %c16_i32 = arith.constant {async_task_id = array<i32: 1>} 16 : i32
      %c32_i32 = arith.constant {async_task_id = array<i32: 1>} 32 : i32
      %c48_i32 = arith.constant {async_task_id = array<i32: 1>} 48 : i32
      %c1_i32_269 = arith.constant {async_task_id = array<i32: 1>} 1 : i32
      %c2_i64 = arith.constant {async_task_id = array<i32: 1>} 2 : i64
      %c128_i32_270 = arith.constant {async_task_id = array<i32: 1>} 128 : i32
      %c2_i32 = arith.constant {async_task_id = array<i32: 1>} 2 : i32
      %c64_i32_271 = arith.constant {async_task_id = array<i32: 1>} 64 : i32
      %true_272 = arith.constant {async_task_id = array<i32: 1>} true
      %cst_273 = arith.constant {async_task_id = array<i32: 1>} dense<0.000000e+00> : tensor<128x128xf32, #linear>
      %bhid = tt.get_program_id z {async_task_id = array<i32: 1>} : i32
      %pid = tt.get_program_id x {async_task_id = array<i32: 1>} : i32
      %off_bh = arith.remsi %bhid, %H_178 {async_task_id = array<i32: 1>} : i32
      %off_bh_274 = arith.muli %stride_h_179, %off_bh {async_task_id = array<i32: 1>} : i32
      %off_bh_275 = arith.divsi %bhid, %H_178 {async_task_id = array<i32: 1>} : i32
      %off_bh_276 = arith.muli %stride_z_180, %off_bh_275 {async_task_id = array<i32: 1>} : i32
      %off_bh_277 = arith.addi %off_bh_274, %off_bh_276 {async_task_id = array<i32: 1>} : i32
      %off_bh_278 = arith.extsi %off_bh_277 {async_task_id = array<i32: 1>} : i32 to i64
      %off_bh_279 = arith.extsi %stride_tok_181 {async_task_id = array<i32: 1>} : i32 to i64
      %off_bh_280 = arith.divsi %off_bh_278, %off_bh_279 {async_task_id = array<i32: 1>} : i64
      %cluster_cta_rank = arith.remsi %pid, %c2_i32 {async_task_id = array<i32: 1>} : i32
      %num_steps = arith.divsi %N_CTX_182, %c128_i32_270 {async_task_id = array<i32: 1>} : i32
      %dq_row = arith.muli %cluster_cta_rank, %c64_i32_271 {async_task_id = array<i32: 1>} : i32
      %dq_row_281 = arith.extsi %dq_row {async_task_id = array<i32: 1>} : i32 to i64
      %13 = ttg.memdesc_index %arg123[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %13, %c1_i32_269 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3, 4>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dk_282 = ttg.memdesc_index %dk_252[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %dk_282, %c1_i32_269 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3, 4>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.tmem_store %cst_273, %dk_183, %true_272 {async_task_id = array<i32: 1>, tmem.end = array<i32: 12>, tmem.start = array<i32: 9, 11>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %dk_283 = ttg.memdesc_index %dk_251[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.arrive_barrier %dk_283, 1 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %14 = ttg.memdesc_index %arg125[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %14, %c1_i32_269 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3, 4>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dv_284 = ttg.memdesc_index %dv_246[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %dv_284, %c1_i32_269 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3, 4>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.tmem_store %cst_273, %dv_184, %true_272 {async_task_id = array<i32: 1>, tmem.end = array<i32: 8>, tmem.start = array<i32: 5, 7>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %dv_285 = ttg.memdesc_index %dv_245[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.arrive_barrier %dv_285, 1 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %curr_m_286:2 = scf.for %curr_m_287 = %c0_i32_268 to %num_steps step %c1_i32_269 iter_args(%arg157 = %c0_i32_268, %curr_m_288 = %curr_m) -> (i32, i64)  : i32 {
        %qt = arith.extsi %arg157 {async_task_id = array<i32: 1>} : i32 to i64
        %qt_289 = arith.addi %off_bh_280, %qt {async_task_id = array<i32: 1>} : i64
        %dq_290 = arith.andi %curr_m_288, %c1_i64_267 {async_task_id = array<i32: 1>} : i64
        %dq_291 = arith.trunci %dq_290 {async_task_id = array<i32: 1>} : i64 to i1
        %dpT_292 = ttng.tmem_subslice %dpT_185 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dpT_293 = ttg.memdesc_reinterpret %dpT_292 {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x128xf32, #tmem3, #ttng.tensor_memory, mutable>
        %dq_294 = ttg.memdesc_index %dpT_293[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x64x128xf32, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem4, #ttng.tensor_memory, mutable>
        %dq_295 = ttg.memdesc_index %dq_186[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dq_296 = arith.extui %dq_291 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %dq_295, %dq_296 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3, 4>, direction = "forward", dstTask = 2 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dq_297 = ttng.tmem_load %dq_294 {async_task_id = array<i32: 1>} : !ttg.memdesc<64x128xf32, #tmem4, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear4>
        %dq_298 = ttg.memdesc_index %dq_266[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %dq_298, 1 {async_task_id = array<i32: 1>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 2, 3, 4>, dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dqs = tt.reshape %dq_297 {async_task_id = array<i32: 1>} : tensor<64x128xf32, #linear4> -> tensor<128x2x32xf32, #linear5>
        %dqs_299 = tt.trans %dqs {async_task_id = array<i32: 1>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
        %dqs_300 = ttg.convert_layout %dqs_299 {async_task_id = array<i32: 1>} : tensor<128x32x2xf32, #linear6> -> tensor<128x32x2xf32, #linear7>
        %dqs_301, %dqs_302 = tt.split %dqs_300 {async_task_id = array<i32: 1>} : tensor<128x32x2xf32, #linear7> -> tensor<128x32xf32, #linear8>
        %dqs_303 = tt.reshape %dqs_301 {async_task_id = array<i32: 1>} : tensor<128x32xf32, #linear8> -> tensor<128x2x16xf32, #blocked4>
        %dqs_304 = tt.trans %dqs_303 {async_task_id = array<i32: 1>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked4> -> tensor<128x16x2xf32, #blocked5>
        %dqs_305, %dqs_306 = tt.split %dqs_304 {async_task_id = array<i32: 1>} : tensor<128x16x2xf32, #blocked5> -> tensor<128x16xf32, #blocked3>
        %dqs_307 = tt.reshape %dqs_302 {async_task_id = array<i32: 1>} : tensor<128x32xf32, #linear8> -> tensor<128x2x16xf32, #blocked4>
        %dqs_308 = tt.trans %dqs_307 {async_task_id = array<i32: 1>, order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked4> -> tensor<128x16x2xf32, #blocked5>
        %dqs_309, %dqs_310 = tt.split %dqs_308 {async_task_id = array<i32: 1>} : tensor<128x16x2xf32, #blocked5> -> tensor<128x16xf32, #blocked3>
        %dq_row_311 = arith.addi %qt_289, %dq_row_281 {async_task_id = array<i32: 1>} : i64
        %dq_row_312 = arith.muli %dq_row_311, %c2_i64 {async_task_id = array<i32: 1>} : i64
        %dqN = arith.mulf %dqs_305, %cst {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked3>
        %15 = arith.trunci %dq_row_312 {async_task_id = array<i32: 1>} : i64 to i32
        %desc_dq_reduce_staging_313 = ttg.memdesc_index %desc_dq_reduce_staging_187[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        ttg.local_store %dqN, %desc_dq_reduce_staging_313 {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked3> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        %desc_dq_reduce_staging_314 = ttg.memdesc_index %desc_dq_reduce_staging_187[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        %16 = ttng.async_tma_reduce add, %desc_dq_188[%15, %c0_i32_268] %desc_dq_reduce_staging_314 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %16   {async_task_id = array<i32: 1>} : !ttg.async.token
        %dqN_315 = arith.mulf %dqs_306, %cst {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked3>
        %desc_dq_reduce_staging_316 = ttg.memdesc_index %desc_dq_reduce_staging_187[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        ttg.local_store %dqN_315, %desc_dq_reduce_staging_316 {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked3> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        %desc_dq_reduce_staging_317 = ttg.memdesc_index %desc_dq_reduce_staging_187[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        %17 = ttng.async_tma_reduce add, %desc_dq_188[%15, %c16_i32] %desc_dq_reduce_staging_317 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %17   {async_task_id = array<i32: 1>} : !ttg.async.token
        %dqN_318 = arith.mulf %dqs_309, %cst {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked3>
        %desc_dq_reduce_staging_319 = ttg.memdesc_index %desc_dq_reduce_staging_187[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        ttg.local_store %dqN_318, %desc_dq_reduce_staging_319 {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked3> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        %desc_dq_reduce_staging_320 = ttg.memdesc_index %desc_dq_reduce_staging_187[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        %18 = ttng.async_tma_reduce add, %desc_dq_188[%15, %c32_i32] %desc_dq_reduce_staging_320 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %18   {async_task_id = array<i32: 1>} : !ttg.async.token
        %dqN_321 = arith.mulf %dqs_310, %cst {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked3>
        %desc_dq_reduce_staging_322 = ttg.memdesc_index %desc_dq_reduce_staging_187[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        ttg.local_store %dqN_321, %desc_dq_reduce_staging_322 {async_task_id = array<i32: 1>} : tensor<128x16xf32, #blocked3> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        %desc_dq_reduce_staging_323 = ttg.memdesc_index %desc_dq_reduce_staging_187[%c0_i32_268] {async_task_id = array<i32: 1>} : !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128x16xf32, #shared1, #smem, mutable>
        %19 = ttng.async_tma_reduce add, %desc_dq_188[%15, %c48_i32] %desc_dq_reduce_staging_323 {async_task_id = array<i32: 1>} : !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<128x16xf32, #shared1, #smem, mutable> -> !ttg.async.token
        ttng.async_tma_store_token_wait %19   {async_task_id = array<i32: 1>} : !ttg.async.token
        %curr_m_324 = arith.addi %arg157, %c128_i32_270 {async_task_id = array<i32: 1>} : i32
        %accum_cnt = arith.addi %curr_m_288, %c1_i64_267 {async_task_id = array<i32: 1>} : i64
        scf.yield {async_task_id = array<i32: 1>} %curr_m_324, %accum_cnt : i32, i64
      } {async_task_id = array<i32: 1>, tt.assume_nonempty, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition1(%H_178: i32, %stride_h_179: i32, %stride_z_180: i32, %stride_tok_181: i32, %N_CTX_182: i32, %dk_183: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dv_184: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dpT_185: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dq_186: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %desc_dq_reduce_staging_187: !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, %desc_dq_188: !tt.tensordesc<128x16xf32, #shared1>, %k_189: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %kt_190: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %v_191: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qT_192: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %qT_193: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %k_194: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %qkT_195: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %qT_196: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qkT_197: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %ppT_198: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_199: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %dpT_200: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %v_201: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %dpT_202: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_203: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_204: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %do_205: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %do_206: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %do_207: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_208: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q_209: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %q_210: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %q_211: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_212: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_0_213: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %kt_214: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %dsT_dq_0_215: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_216: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_217: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_218: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_219: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %v_220: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %kt_221: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %k_222: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %desc_k_223: !tt.tensordesc<128x128xf16, #shared>, %desc_kt_224: !tt.tensordesc<256x64xf16, #shared>, %desc_v_225: !tt.tensordesc<128x128xf16, #shared>, %desc_qt_226: !tt.tensordesc<64x128xf16, #shared>, %desc_q_227: !tt.tensordesc<128x64xf16, #shared>, %m_228: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %m_229: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %desc_m_230: !tt.tensordesc<128xf32, #shared2>, %desc_do_231: !tt.tensordesc<128x64xf16, #shared>, %desc_dot_232: !tt.tensordesc<64x128xf16, #shared>, %Di_233: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %Di_234: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %desc_delta_235: !tt.tensordesc<128xf32, #shared2>, %dsT_dq_1_236: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg122: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qkT_237: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qkT_238: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_239: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_240: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_241: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_242: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_243: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_244: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_245: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_246: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_247: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_248: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_249: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_250: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_251: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_252: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %m_253: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %m_254: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %ppT_255: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %ppT_256: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %Di_257: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %Di_258: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_259: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_260: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_1_261: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_1_262: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_0_263: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_0_264: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dq_265: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dq_266: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) num_warps(1) {
      %13 = ub.poison : i1
      %c1_i64_267 = arith.constant {async_task_id = array<i32: 2>} 1 : i64
      %c0_i64_268 = arith.constant {async_task_id = array<i32: 2>} 0 : i64
      %false = arith.constant {async_task_id = array<i32: 2>} false
      %c1_i32_269 = arith.constant {async_task_id = array<i32: 2>} 1 : i32
      %c128_i32_270 = arith.constant {async_task_id = array<i32: 2>} 128 : i32
      %c0_i32_271 = arith.constant {async_task_id = array<i32: 2>} 0 : i32
      %true_272 = arith.constant {async_task_id = array<i32: 2>} true
      %num_steps = arith.divsi %N_CTX_182, %c128_i32_270 {async_task_id = array<i32: 2>} : i32
      %k_273 = ttg.memdesc_index %k_189[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %k_273, %c0_i32_271, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %kt_274 = ttg.memdesc_index %kt_190[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %kt_274, %c0_i32_271, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %v_275 = ttg.memdesc_index %v_191[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %v_275, %c0_i32_271, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dv_276 = ttg.memdesc_index %dv_244[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %dv_276, %c1_i32_269 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dk_277 = ttg.memdesc_index %dk_250[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %dk_277, %c1_i32_269 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %curr_m = arith.cmpi sgt, %num_steps, %c0_i32_271 : i32
      %qT_278 = ttg.memdesc_index %qT_192[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %qT_279 = ttg.memdesc_index %qT_193[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %qT_279, %c0_i32_271, %curr_m {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %qT_280 = ttg.memdesc_trans %qT_278 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>
      %k_281 = ttg.memdesc_index %k_194[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %qkT_282 = ttg.memdesc_index %qkT_195[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %qT_283 = ttg.memdesc_index %qT_196[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %qkT_284 = ttg.memdesc_index %qkT_197[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %qkT_285 = ttg.memdesc_index %qkT_238[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %qkT_285, %c1_i32_269, %curr_m {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %ppT_286 = ttg.memdesc_index %ppT_198[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %ppT_286, %c1_i32_269, %curr_m {async_task_id = array<i32: 2>, constraints = {WSBarrier = {direction = "backward", dstTask = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.tc_gen5_mma %k_281, %qT_280, %qkT_282, %false, %curr_m, %qT_283[%true_272], %qkT_284[%true_272] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,1,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dpT_287 = ttg.memdesc_index %dpT_199[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %dpT_288 = ttg.memdesc_index %dpT_200[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %dpT_288, %c0_i32_271, %curr_m {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dpT_289 = ttg.memdesc_trans %dpT_287 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>
      %v_290 = ttg.memdesc_index %v_201[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %dpT_291 = ttg.memdesc_index %dpT_185[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %dpT_292 = ttg.memdesc_index %dpT_202[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dpT_293 = ttg.memdesc_index %dpT_203[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dpT_294 = ttg.memdesc_index %dpT_240[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %dpT_294, %c1_i32_269, %curr_m {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dq_295 = ttg.memdesc_index %dq_266[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %dq_295, %c1_i32_269, %curr_m {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.tc_gen5_mma %v_290, %dpT_289, %dpT_291, %false, %curr_m, %dpT_292[%true_272], %dpT_293[%true_272] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dv_296 = ttg.memdesc_index %dv_204[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %do_297 = ttg.memdesc_index %do_205[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %qkT_298 = ttng.tmem_subslice %qkT_195 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
      %qkT_299 = ttg.memdesc_reinterpret %qkT_298 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
      %ppT_300 = ttg.memdesc_index %qkT_299[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
      %do_301 = ttg.memdesc_index %do_206[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %do_302 = ttg.memdesc_index %do_207[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %do_302, %c0_i32_271, %curr_m {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %ppT_303 = ttg.memdesc_index %ppT_198[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %ppT_304 = ttg.memdesc_index %ppT_255[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %ppT_304, %c0_i32_271, %curr_m {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.tc_gen5_mma %ppT_300, %do_297, %dv_296, %false, %curr_m, %do_301[%true_272], %ppT_303[%true_272] {async_task_id = array<i32: 2>, is_async, tmem.end = array<i32: 5>, tmem.start = array<i32: 6>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %curr_m_305 = arith.subi %num_steps, %c1_i32_269 : i32
      %curr_m_306:3 = scf.for %curr_m_316 = %c0_i32_271 to %curr_m_305 step %c1_i32_269 iter_args(%arg157 = %false, %curr_m_317 = %c0_i64_268, %dq_318 = %false) -> (i1, i64, i1)  : i32 {
        %accum_cnt = arith.addi %curr_m_317, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %qT_319 = arith.andi %accum_cnt, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %qT_320 = arith.trunci %qT_319 {async_task_id = array<i32: 2>} : i64 to i1
        %qT_321 = ttg.memdesc_index %qT_192[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        %qT_322 = ttg.memdesc_index %qT_193[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %qT_323 = arith.extui %qT_320 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %qT_322, %qT_323, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %qT_324 = ttg.memdesc_trans %qT_321 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>
        %q_325 = arith.andi %curr_m_317, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %q_326 = arith.trunci %q_325 {async_task_id = array<i32: 2>} : i64 to i1
        %k_327 = ttg.memdesc_index %k_194[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qkT_328 = ttg.memdesc_index %qkT_195[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qT_329 = ttg.memdesc_index %qT_196[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %qkT_330 = arith.andi %accum_cnt, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %qkT_331 = arith.trunci %qkT_330 {async_task_id = array<i32: 2>} : i64 to i1
        %qkT_332 = ttg.memdesc_index %qkT_197[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %qkT_333 = ttg.memdesc_index %qkT_238[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %qkT_334 = arith.xori %qkT_331, %true_272 : i1
        %qkT_335 = arith.extui %qkT_334 : i1 to i32
        ttng.wait_barrier %qkT_333, %qkT_335, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dv_336 = arith.andi %accum_cnt, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %dv_337 = arith.trunci %dv_336 {async_task_id = array<i32: 2>} : i64 to i1
        %ppT_338 = ttg.memdesc_index %ppT_198[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %qkT_339 = arith.xori %dv_337, %true_272 {async_task_id = array<i32: 2>} : i1
        %qkT_340 = arith.extui %qkT_339 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %ppT_338, %qkT_340, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {direction = "backward", dstTask = 2 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_mma %k_327, %qT_324, %qkT_328, %false, %true_272, %qT_329[%true_272], %qkT_332[%true_272] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,1,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dk_341 = arith.andi %curr_m_317, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %dk_342 = arith.trunci %dk_341 {async_task_id = array<i32: 2>} : i64 to i1
        %dk_343 = ttg.memdesc_index %dk_208[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %q_344 = ttg.memdesc_index %q_209[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %dpT_345 = ttng.tmem_subslice %dpT_185 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
        %dpT_346 = ttg.memdesc_reinterpret %dpT_345 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        %dsT_347 = ttg.memdesc_index %dpT_346[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %q_348 = ttg.memdesc_index %q_210[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %q_349 = ttg.memdesc_index %q_211[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %q_350 = arith.extui %q_326 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %q_349, %q_350, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_351 = ttg.memdesc_index %dsT_212[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_352 = ttg.memdesc_index %dsT_259[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dk_353 = arith.extui %dk_342 : i1 to i32
        ttng.wait_barrier %dsT_352, %dk_353 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_mma %dsT_347, %q_344, %dk_343, %arg157, %true_272, %q_348[%true_272], %dsT_351[%true_272] {async_task_id = array<i32: 2>, is_async, tmem.end = array<i32: 9>, tmem.start = array<i32: 10>, tt.autows = "{\22stage\22: \221\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,10\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq = arith.andi %curr_m_317, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %dsT_dq_354 = arith.trunci %dsT_dq {async_task_id = array<i32: 2>} : i64 to i1
        %dsT_dq_0_355 = ttg.memdesc_index %dsT_dq_0_213[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        %dsT_dq_0_356 = ttg.memdesc_index %dsT_dq_0_263[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_357 = arith.extui %dsT_dq_354 : i1 to i32
        ttng.wait_barrier %dsT_dq_0_356, %dsT_dq_357 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_358 = ttg.memdesc_trans %dsT_dq_0_355 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared4, #smem, mutable>
        %kt_359 = ttg.memdesc_index %kt_214[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        %dpT_360 = ttng.tmem_subslice %dpT_185 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dpT_361 = ttg.memdesc_reinterpret %dpT_360 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x128xf32, #tmem3, #ttng.tensor_memory, mutable>
        %dq_362 = ttg.memdesc_index %dpT_361[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf32, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem4, #ttng.tensor_memory, mutable>
        %dsT_dq_0_363 = ttg.memdesc_index %dsT_dq_0_215[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dq_364 = ttg.memdesc_index %dq_186[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dpT_365 = ttg.memdesc_index %dpT_240[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dq_366 = arith.extui %dq_318 : i1 to i32
        ttng.wait_barrier %dpT_365, %dq_366 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_mma %dsT_dq_358, %kt_359, %dq_362, %false, %true_272, %dsT_dq_0_363[%true_272], %dq_364[%true_272] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x256xf16, #shared4, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem4, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %do_367 = arith.andi %accum_cnt, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %do_368 = arith.trunci %do_367 {async_task_id = array<i32: 2>} : i64 to i1
        %dpT_369 = arith.andi %accum_cnt, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %dpT_370 = arith.trunci %dpT_369 {async_task_id = array<i32: 2>} : i64 to i1
        %dpT_371 = ttg.memdesc_index %dpT_199[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        %dpT_372 = ttg.memdesc_index %dpT_200[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dpT_373 = arith.extui %dpT_370 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %dpT_372, %dpT_373, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dpT_374 = ttg.memdesc_trans %dpT_371 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>
        %v_375 = ttg.memdesc_index %v_201[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %dpT_376 = ttg.memdesc_index %dpT_185[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dpT_377 = ttg.memdesc_index %dpT_202[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dpT_378 = arith.andi %accum_cnt, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %dpT_379 = arith.trunci %dpT_378 {async_task_id = array<i32: 2>} : i64 to i1
        %dpT_380 = ttg.memdesc_index %dpT_203[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dpT_381 = ttg.memdesc_index %dpT_240[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dpT_382 = arith.xori %dpT_379, %true_272 : i1
        %dpT_383 = arith.extui %dpT_382 : i1 to i32
        ttng.wait_barrier %dpT_381, %dpT_383, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dq_384 = arith.andi %accum_cnt, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %dq_385 = arith.trunci %dq_384 {async_task_id = array<i32: 2>} : i64 to i1
        %dq_386 = ttg.memdesc_index %dq_266[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dq_387 = arith.xori %dq_385, %true_272 : i1
        %dq_388 = arith.extui %dq_387 : i1 to i32
        ttng.wait_barrier %dq_386, %dq_388, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 1 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_mma %v_375, %dpT_374, %dpT_376, %false, %true_272, %dpT_377[%true_272], %dpT_380[%true_272] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared4, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dv_389 = ttg.memdesc_index %dv_204[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %do_390 = ttg.memdesc_index %do_205[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %qkT_391 = ttng.tmem_subslice %qkT_195 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
        %qkT_392 = ttg.memdesc_reinterpret %qkT_391 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        %ppT_393 = ttg.memdesc_index %qkT_392[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %do_394 = ttg.memdesc_index %do_206[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %do_395 = ttg.memdesc_index %do_207[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %do_396 = arith.extui %do_368 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %do_395, %do_396, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %ppT_397 = ttg.memdesc_index %ppT_198[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %ppT_398 = ttg.memdesc_index %ppT_255[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dv_399 = arith.extui %dv_337 : i1 to i32
        ttng.wait_barrier %ppT_398, %dv_399, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_mma %ppT_393, %do_390, %dv_389, %true_272, %true_272, %do_394[%true_272], %ppT_397[%true_272] {async_task_id = array<i32: 2>, is_async, tmem.end = array<i32: 5>, tmem.start = array<i32: 6>, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        scf.yield %true_272, %accum_cnt, %dq_385 : i1, i64, i1
      } {async_task_id = array<i32: 2>, tt.assume_nonempty, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      %curr_m_307 = arith.cmpi sgt, %num_steps, %c0_i32_271 : i32
      %curr_m_308:3 = scf.if %curr_m_307 -> (i1, i64, i1) {
        %accum_cnt = arith.addi %curr_m_306#1, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %q_316 = arith.andi %curr_m_306#1, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %q_317 = arith.trunci %q_316 {async_task_id = array<i32: 2>} : i64 to i1
        %dk_318 = arith.andi %curr_m_306#1, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %dk_319 = arith.trunci %dk_318 {async_task_id = array<i32: 2>} : i64 to i1
        %dk_320 = ttg.memdesc_index %dk_208[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %q_321 = ttg.memdesc_index %q_209[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %dpT_322 = ttng.tmem_subslice %dpT_185 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
        %dpT_323 = ttg.memdesc_reinterpret %dpT_322 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        %dsT_324 = ttg.memdesc_index %dpT_323[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #tmem2, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
        %q_325 = ttg.memdesc_index %q_210[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %q_326 = ttg.memdesc_index %q_211[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %q_327 = arith.extui %q_317 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %q_326, %q_327, %true_272 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, direction = "forward", dstTask = 3 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_328 = ttg.memdesc_index %dsT_212[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_329 = ttg.memdesc_index %dsT_259[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dk_330 = arith.extui %dk_319 : i1 to i32
        ttng.wait_barrier %dsT_329, %dk_330 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_mma %dsT_324, %q_321, %dk_320, %curr_m_306#0, %true_272, %q_325[%true_272], %dsT_328[%true_272] {async_task_id = array<i32: 2>, is_async, tmem.end = array<i32: 9>, tmem.start = array<i32: 10>, tt.autows = "{\22stage\22: \221\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,10\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq = arith.andi %curr_m_306#1, %c1_i64_267 {async_task_id = array<i32: 2>} : i64
        %dsT_dq_331 = arith.trunci %dsT_dq {async_task_id = array<i32: 2>} : i64 to i1
        %dsT_dq_0_332 = ttg.memdesc_index %dsT_dq_0_213[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        %dsT_dq_0_333 = ttg.memdesc_index %dsT_dq_0_263[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_334 = arith.extui %dsT_dq_331 : i1 to i32
        ttng.wait_barrier %dsT_dq_0_333, %dsT_dq_334 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_335 = ttg.memdesc_trans %dsT_dq_0_332 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared4, #smem, mutable>
        %kt_336 = ttg.memdesc_index %kt_214[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        %dpT_337 = ttng.tmem_subslice %dpT_185 {dim = 1 : i32, offset = 0 : i32, async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dpT_338 = ttg.memdesc_reinterpret %dpT_337 {async_task_id = array<i32: 2>} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x128xf32, #tmem3, #ttng.tensor_memory, mutable>
        %dq_339 = ttg.memdesc_index %dpT_338[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x64x128xf32, #tmem3, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem4, #ttng.tensor_memory, mutable>
        %dsT_dq_0_340 = ttg.memdesc_index %dsT_dq_0_215[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dq_341 = ttg.memdesc_index %dq_186[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dpT_342 = ttg.memdesc_index %dpT_240[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dq_343 = arith.extui %curr_m_306#2 : i1 to i32
        ttng.wait_barrier %dpT_342, %dq_343 {async_task_id = array<i32: 2>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 3, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.tc_gen5_mma %dsT_dq_335, %kt_336, %dq_339, %false, %true_272, %dsT_dq_0_340[%true_272], %dq_341[%true_272] {async_task_id = array<i32: 2>, is_async, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x256xf16, #shared4, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf32, #tmem4, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        scf.yield %true_272, %accum_cnt, %13 : i1, i64, i1
      } else {
        scf.yield %curr_m_306#0, %curr_m_306#1, %curr_m_306#2 : i1, i64, i1
      }
      %dk_309 = ttg.memdesc_index %dk_216[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.tc_gen5_commit %dk_309 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dk_310 = ttg.memdesc_index %dk_217[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.tc_gen5_commit %dk_310 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dv_311 = ttg.memdesc_index %dv_218[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.tc_gen5_commit %dv_311 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %dv_312 = ttg.memdesc_index %dv_219[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.tc_gen5_commit %dv_312 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %v_313 = ttg.memdesc_index %v_220[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.tc_gen5_commit %v_313 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %kt_314 = ttg.memdesc_index %kt_221[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.tc_gen5_commit %kt_314 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %k_315 = ttg.memdesc_index %k_222[%c0_i32_271] {async_task_id = array<i32: 2>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.tc_gen5_commit %k_315 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttg.warp_return
    }
    partition2(%H_178: i32, %stride_h_179: i32, %stride_z_180: i32, %stride_tok_181: i32, %N_CTX_182: i32, %dk_183: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dv_184: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dpT_185: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dq_186: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %desc_dq_reduce_staging_187: !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, %desc_dq_188: !tt.tensordesc<128x16xf32, #shared1>, %k_189: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %kt_190: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %v_191: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qT_192: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %qT_193: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %k_194: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %qkT_195: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %qT_196: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qkT_197: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %ppT_198: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_199: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %dpT_200: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %v_201: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %dpT_202: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_203: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_204: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %do_205: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %do_206: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %do_207: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_208: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q_209: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %q_210: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %q_211: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_212: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_0_213: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %kt_214: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %dsT_dq_0_215: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_216: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_217: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_218: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_219: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %v_220: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %kt_221: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %k_222: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %desc_k_223: !tt.tensordesc<128x128xf16, #shared>, %desc_kt_224: !tt.tensordesc<256x64xf16, #shared>, %desc_v_225: !tt.tensordesc<128x128xf16, #shared>, %desc_qt_226: !tt.tensordesc<64x128xf16, #shared>, %desc_q_227: !tt.tensordesc<128x64xf16, #shared>, %m_228: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %m_229: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %desc_m_230: !tt.tensordesc<128xf32, #shared2>, %desc_do_231: !tt.tensordesc<128x64xf16, #shared>, %desc_dot_232: !tt.tensordesc<64x128xf16, #shared>, %Di_233: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %Di_234: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %desc_delta_235: !tt.tensordesc<128xf32, #shared2>, %dsT_dq_1_236: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg122: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qkT_237: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qkT_238: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_239: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_240: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_241: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_242: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_243: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_244: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_245: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_246: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_247: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_248: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_249: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_250: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_251: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_252: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %m_253: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %m_254: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %ppT_255: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %ppT_256: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %Di_257: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %Di_258: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_259: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_260: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_1_261: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_1_262: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_0_263: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_0_264: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dq_265: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dq_266: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) num_warps(1) {
      %c1_i64_267 = arith.constant {async_task_id = array<i32: 3>} 1 : i64
      %curr_m = arith.constant {async_task_id = array<i32: 3>} 0 : i64
      %true_268 = arith.constant {async_task_id = array<i32: 3>} true
      %c1_i32_269 = arith.constant {async_task_id = array<i32: 3>} 1 : i32
      %c128_i32_270 = arith.constant {async_task_id = array<i32: 3>} 128 : i32
      %c0_i32_271 = arith.constant {async_task_id = array<i32: 3>} 0 : i32
      %c2_i32 = arith.constant {async_task_id = array<i32: 3>} 2 : i32
      %c64_i32_272 = arith.constant {async_task_id = array<i32: 3>} 64 : i32
      %bhid = tt.get_program_id z {async_task_id = array<i32: 3>} : i32
      %pid = tt.get_program_id x {async_task_id = array<i32: 3>} : i32
      %off_chz = arith.muli %bhid, %N_CTX_182 {async_task_id = array<i32: 3>} : i32
      %off_chz_273 = arith.extsi %off_chz {async_task_id = array<i32: 3>} : i32 to i64
      %off_bh = arith.remsi %bhid, %H_178 {async_task_id = array<i32: 3>} : i32
      %off_bh_274 = arith.muli %stride_h_179, %off_bh {async_task_id = array<i32: 3>} : i32
      %off_bh_275 = arith.divsi %bhid, %H_178 {async_task_id = array<i32: 3>} : i32
      %off_bh_276 = arith.muli %stride_z_180, %off_bh_275 {async_task_id = array<i32: 3>} : i32
      %off_bh_277 = arith.addi %off_bh_274, %off_bh_276 {async_task_id = array<i32: 3>} : i32
      %off_bh_278 = arith.extsi %off_bh_277 {async_task_id = array<i32: 3>} : i32 to i64
      %off_bh_279 = arith.extsi %stride_tok_181 {async_task_id = array<i32: 3>} : i32 to i64
      %off_bh_280 = arith.divsi %off_bh_278, %off_bh_279 {async_task_id = array<i32: 3>} : i64
      %start_n = arith.muli %pid, %c128_i32_270 {async_task_id = array<i32: 3>} : i32
      %cluster_cta_rank = arith.remsi %pid, %c2_i32 {async_task_id = array<i32: 3>} : i32
      %k_281 = arith.extsi %start_n {async_task_id = array<i32: 3>} : i32 to i64
      %k_282 = arith.addi %off_bh_280, %k_281 {async_task_id = array<i32: 3>} : i64
      %k_283 = arith.trunci %k_282 {async_task_id = array<i32: 3>} : i64 to i32
      %k_284 = ttg.memdesc_index %k_222[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %k_284, %c1_i32_269 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %k_285 = ttg.memdesc_index %k_189[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.barrier_expect %k_285, 32768 {async_task_id = array<i32: 3>}, %true_268 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %k_286 = ttg.memdesc_index %k_194[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttng.async_tma_copy_global_to_local %desc_k_223[%k_283, %c0_i32_271] %k_286, %k_285, %true_268 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %kt_start_n = arith.muli %cluster_cta_rank, %c128_i32_270 {async_task_id = array<i32: 3>} : i32
      %kt_start_n_287 = arith.subi %start_n, %kt_start_n {async_task_id = array<i32: 3>} : i32
      %kt_288 = arith.extsi %kt_start_n_287 {async_task_id = array<i32: 3>} : i32 to i64
      %kt_289 = arith.addi %off_bh_280, %kt_288 {async_task_id = array<i32: 3>} : i64
      %kt_290 = arith.trunci %kt_289 {async_task_id = array<i32: 3>} : i64 to i32
      %kt_291 = nvg.cluster_id {async_task_id = array<i32: 3>}
      %kt_292 = arith.remsi %kt_291, %c2_i32 {async_task_id = array<i32: 3>} : i32
      %kt_293 = arith.muli %kt_292, %c64_i32_272 {async_task_id = array<i32: 3>} : i32
      %kt_294 = ttg.memdesc_index %kt_221[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %kt_294, %c1_i32_269 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %kt_295 = ttg.memdesc_index %kt_190[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.barrier_expect %kt_295, 32768 {async_task_id = array<i32: 3>}, %true_268 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %kt_296 = ttg.memdesc_index %kt_214[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      ttng.async_tma_copy_global_to_local %desc_kt_224[%kt_290, %kt_293] %kt_296, %kt_295, %true_268 {async_task_id = array<i32: 3>} : !tt.tensordesc<256x64xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      %v_297 = ttg.memdesc_index %v_220[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.wait_barrier %v_297, %c1_i32_269 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 3 : i32, minRegionId = 3 : i32, parentId = 0 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %v_298 = ttg.memdesc_index %v_191[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      ttng.barrier_expect %v_298, 32768 {async_task_id = array<i32: 3>}, %true_268 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %v_299 = ttg.memdesc_index %v_201[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttng.async_tma_copy_global_to_local %desc_v_225[%k_283, %c0_i32_271] %v_299, %v_298, %true_268 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %num_steps = arith.divsi %N_CTX_182, %c128_i32_270 {async_task_id = array<i32: 3>} : i32
      %curr_m_300:2 = scf.for %curr_m_301 = %c0_i32_271 to %num_steps step %c1_i32_269 iter_args(%arg157 = %c0_i32_271, %curr_m_302 = %curr_m) -> (i32, i64)  : i32 {
        %qt = arith.extsi %arg157 {async_task_id = array<i32: 3>} : i32 to i64
        %qt_303 = arith.addi %off_bh_280, %qt {async_task_id = array<i32: 3>} : i64
        %qt_304 = arith.trunci %qt_303 {async_task_id = array<i32: 3>} : i64 to i32
        %qt_305 = arith.addi %qt_304, %kt_293 {async_task_id = array<i32: 3>} : i32
        %qT_306 = arith.andi %curr_m_302, %c1_i64_267 {async_task_id = array<i32: 3>} : i64
        %qT_307 = arith.trunci %qT_306 {async_task_id = array<i32: 3>} : i64 to i1
        %qT_308 = ttg.memdesc_index %qT_196[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %qt_309 = arith.xori %qT_307, %true_268 {async_task_id = array<i32: 3>} : i1
        %qt_310 = arith.extui %qt_309 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %qT_308, %qt_310 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %qT_311 = ttg.memdesc_index %qT_193[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.barrier_expect %qT_311, 16384 {async_task_id = array<i32: 3>}, %true_268 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %qT_312 = ttg.memdesc_index %qT_192[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc_qt_226[%qt_305, %c0_i32_271] %qT_312, %qT_311, %true_268 {async_task_id = array<i32: 3>} : !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        %q_313 = arith.andi %curr_m_302, %c1_i64_267 {async_task_id = array<i32: 3>} : i64
        %q_314 = arith.trunci %q_313 {async_task_id = array<i32: 3>} : i64 to i1
        %q_315 = ttg.memdesc_index %q_210[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %q_316 = arith.xori %q_314, %true_268 {async_task_id = array<i32: 3>} : i1
        %q_317 = arith.extui %q_316 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %q_315, %q_317 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %q_318 = ttg.memdesc_index %q_211[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.barrier_expect %q_318, 16384 {async_task_id = array<i32: 3>}, %true_268 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %q_319 = ttg.memdesc_index %q_209[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc_q_227[%qt_304, %kt_293] %q_319, %q_318, %true_268 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %offs_m_start = arith.addi %off_chz_273, %qt {async_task_id = array<i32: 3>} : i64
        %m_320 = arith.trunci %offs_m_start {async_task_id = array<i32: 3>} : i64 to i32
        %m_321 = arith.andi %curr_m_302, %c1_i64_267 {async_task_id = array<i32: 3>} : i64
        %m_322 = arith.trunci %m_321 {async_task_id = array<i32: 3>} : i64 to i1
        %m_323 = ttg.memdesc_index %m_254[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %m_324 = arith.xori %m_322, %true_268 : i1
        %m_325 = arith.extui %m_324 : i1 to i32
        ttng.wait_barrier %m_323, %m_325 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %m_326 = ttg.memdesc_index %m_228[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.barrier_expect %m_326, 512 {async_task_id = array<i32: 3>}, %true_268 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %m_327 = ttg.memdesc_index %m_229[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc_m_230[%m_320] %m_327, %m_326, %true_268 {async_task_id = array<i32: 3>} : !tt.tensordesc<128xf32, #shared2>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
        %do_328 = arith.andi %curr_m_302, %c1_i64_267 {async_task_id = array<i32: 3>} : i64
        %do_329 = arith.trunci %do_328 {async_task_id = array<i32: 3>} : i64 to i1
        %do_330 = ttg.memdesc_index %do_206[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %do_331 = arith.xori %do_329, %true_268 {async_task_id = array<i32: 3>} : i1
        %do_332 = arith.extui %do_331 {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %do_330, %do_332 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %do_333 = ttg.memdesc_index %do_207[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.barrier_expect %do_333, 16384 {async_task_id = array<i32: 3>}, %true_268 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %do_334 = ttg.memdesc_index %do_205[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc_do_231[%qt_304, %kt_293] %do_334, %do_333, %true_268 {async_task_id = array<i32: 3>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
        %dpT_335 = arith.andi %curr_m_302, %c1_i64_267 {async_task_id = array<i32: 3>} : i64
        %dpT_336 = arith.trunci %dpT_335 {async_task_id = array<i32: 3>} : i64 to i1
        %dpT_337 = ttg.memdesc_index %dpT_202[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dot = arith.xori %dpT_336, %true_268 {async_task_id = array<i32: 3>} : i1
        %dot_338 = arith.extui %dot {async_task_id = array<i32: 3>} : i1 to i32
        ttng.wait_barrier %dpT_337, %dot_338 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, direction = "backward", dstTask = 2 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dpT_339 = ttg.memdesc_index %dpT_200[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.barrier_expect %dpT_339, 16384 {async_task_id = array<i32: 3>}, %true_268 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dpT_340 = ttg.memdesc_index %dpT_199[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc_dot_232[%qt_305, %c0_i32_271] %dpT_340, %dpT_339, %true_268 {async_task_id = array<i32: 3>} : !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
        %Di_341 = arith.andi %curr_m_302, %c1_i64_267 {async_task_id = array<i32: 3>} : i64
        %Di_342 = arith.trunci %Di_341 {async_task_id = array<i32: 3>} : i64 to i1
        %Di_343 = ttg.memdesc_index %Di_258[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %Di_344 = arith.xori %Di_342, %true_268 : i1
        %Di_345 = arith.extui %Di_344 : i1 to i32
        ttng.wait_barrier %Di_343, %Di_345 {async_task_id = array<i32: 3>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 4>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %Di_346 = ttg.memdesc_index %Di_233[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.barrier_expect %Di_346, 512 {async_task_id = array<i32: 3>}, %true_268 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %Di_347 = ttg.memdesc_index %Di_234[%c0_i32_271] {async_task_id = array<i32: 3>} : !ttg.memdesc<1x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
        ttng.async_tma_copy_global_to_local %desc_delta_235[%m_320] %Di_347, %Di_346, %true_268 {async_task_id = array<i32: 3>} : !tt.tensordesc<128xf32, #shared2>, !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<128xf32, #shared2, #smem, mutable>
        %curr_m_348 = arith.addi %arg157, %c128_i32_270 {async_task_id = array<i32: 3>} : i32
        %accum_cnt = arith.addi %curr_m_302, %c1_i64_267 {async_task_id = array<i32: 3>} : i64
        scf.yield {async_task_id = array<i32: 3>} %curr_m_348, %accum_cnt : i32, i64
      } {async_task_id = array<i32: 3>, tt.assume_nonempty, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    }
    partition3(%H_178: i32, %stride_h_179: i32, %stride_z_180: i32, %stride_tok_181: i32, %N_CTX_182: i32, %dk_183: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dv_184: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dpT_185: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dq_186: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %desc_dq_reduce_staging_187: !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, %desc_dq_188: !tt.tensordesc<128x16xf32, #shared1>, %k_189: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %kt_190: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %v_191: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qT_192: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %qT_193: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %k_194: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %qkT_195: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %qT_196: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qkT_197: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %ppT_198: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_199: !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, %dpT_200: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %v_201: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %dpT_202: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_203: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_204: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %do_205: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %do_206: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %do_207: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_208: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %q_209: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %q_210: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %q_211: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_212: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_0_213: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %kt_214: !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, %dsT_dq_0_215: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_216: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_217: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_218: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_219: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %v_220: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %kt_221: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %k_222: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %desc_k_223: !tt.tensordesc<128x128xf16, #shared>, %desc_kt_224: !tt.tensordesc<256x64xf16, #shared>, %desc_v_225: !tt.tensordesc<128x128xf16, #shared>, %desc_qt_226: !tt.tensordesc<64x128xf16, #shared>, %desc_q_227: !tt.tensordesc<128x64xf16, #shared>, %m_228: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %m_229: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %desc_m_230: !tt.tensordesc<128xf32, #shared2>, %desc_do_231: !tt.tensordesc<128x64xf16, #shared>, %desc_dot_232: !tt.tensordesc<64x128xf16, #shared>, %Di_233: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %Di_234: !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, %desc_delta_235: !tt.tensordesc<128xf32, #shared2>, %dsT_dq_1_236: !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, %arg122: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg123: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg124: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %arg125: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qkT_237: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %qkT_238: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_239: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dpT_240: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_241: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_242: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_243: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_244: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_245: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dv_246: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_247: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_248: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_249: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_250: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_251: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dk_252: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %m_253: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %m_254: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %ppT_255: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %ppT_256: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %Di_257: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %Di_258: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_259: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_260: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_1_261: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_1_262: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_0_263: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dsT_dq_0_264: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dq_265: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, %dq_266: !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) num_warps(1) {
      %c1_i64_267 = arith.constant {async_task_id = array<i32: 4>} 1 : i64
      %curr_m = arith.constant {async_task_id = array<i32: 4>} 0 : i64
      %c1_i32_268 = arith.constant {async_task_id = array<i32: 4>} 1 : i32
      %c128_i32_269 = arith.constant {async_task_id = array<i32: 4>} 128 : i32
      %c0_i32_270 = arith.constant {async_task_id = array<i32: 4>} 0 : i32
      %num_steps = arith.divsi %N_CTX_182, %c128_i32_269 {async_task_id = array<i32: 4>} : i32
      %curr_m_271 = scf.for %curr_m_272 = %c0_i32_270 to %num_steps step %c1_i32_268 iter_args(%curr_m_273 = %curr_m) -> (i64)  : i32 {
        %dsT_dq = arith.andi %curr_m_273, %c1_i64_267 {async_task_id = array<i32: 4>} : i64
        %dsT_dq_274 = arith.trunci %dsT_dq {async_task_id = array<i32: 4>} : i64 to i1
        %dsT_dq_1_275 = ttg.memdesc_index %dsT_dq_1_261[%c0_i32_270] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_276 = arith.extui %dsT_dq_274 : i1 to i32
        ttng.wait_barrier %dsT_dq_1_275, %dsT_dq_276 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 3>, dstTask = 0 : i32, maxRegionId = 1 : i32, minRegionId = 1 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        %dsT_dq_277 = ttg.memdesc_index %dsT_dq_0_263[%c0_i32_270] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %dsT_dq_277, 1 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %dsT_dq_1_278 = ttg.memdesc_index %dsT_dq_1_262[%c0_i32_270] {async_task_id = array<i32: 4>} : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        ttng.arrive_barrier %dsT_dq_1_278, 1 {async_task_id = array<i32: 4>, constraints = {WSBarrier = {channelGraph = array<i32: 0, 1, 2, 3>, dstTask = 0 : i32, maxRegionId = 2 : i32, minRegionId = 2 : i32, parentId = 1 : i32}}} : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
        %accum_cnt = arith.addi %curr_m_273, %c1_i64_267 {async_task_id = array<i32: 4>} : i64
        scf.yield {async_task_id = array<i32: 4>} %accum_cnt : i64
      } {async_task_id = array<i32: 4>, tt.assume_nonempty, tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "reduction", "gemm", "load", "relay"], ttg.warp_specialize.tag = 0 : i32}
      ttg.warp_return
    } : (i32, i32, i32, i32, i32, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x16xf32, #shared1, #smem, mutable>, !tt.tensordesc<128x16xf32, #shared1>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !tt.tensordesc<128x128xf16, #shared>, !tt.tensordesc<256x64xf16, #shared>, !tt.tensordesc<128x128xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>, !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, !tt.tensordesc<128xf32, #shared2>, !tt.tensordesc<128x64xf16, #shared>, !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x128xf32, #shared2, #smem, mutable>, !tt.tensordesc<128xf32, #shared2>, !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>) -> ()
    %dq_122 = ttg.memdesc_index %dq[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dq_122 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %k_123 = ttg.memdesc_index %k_34[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %k_123 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %kt_124 = ttg.memdesc_index %kt_31[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %kt_124 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %v_125 = ttg.memdesc_index %v_28[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %v_125 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %qT_126 = ttg.memdesc_index %qT_16[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %qT_126 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %qT_127 = ttg.memdesc_index %qT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %qT_127 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %qkT_128 = ttg.memdesc_index %qkT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %qkT_128 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %ppT_129 = ttg.memdesc_index %ppT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %ppT_129 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dpT_130 = ttg.memdesc_index %dpT_5[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dpT_130 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dpT_131 = ttg.memdesc_index %dpT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dpT_131 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dpT_132 = ttg.memdesc_index %dpT_24[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dpT_132 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %do_133 = ttg.memdesc_index %do[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %do_133 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %do_134 = ttg.memdesc_index %do_9[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %do_134 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %q_135 = ttg.memdesc_index %q[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %q_135 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %q_136 = ttg.memdesc_index %q_13[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %q_136 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT_137 = ttg.memdesc_index %dsT[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dsT_137 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT_dq_0_138 = ttg.memdesc_index %dsT_dq_0[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dsT_dq_0_138 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_139 = ttg.memdesc_index %dk[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dk_139 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_140 = ttg.memdesc_index %dk_19[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dk_140 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv_141 = ttg.memdesc_index %dv[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dv_141 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv_142 = ttg.memdesc_index %dv_22[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dv_142 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %v_143 = ttg.memdesc_index %v[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %v_143 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %kt_144 = ttg.memdesc_index %kt[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %kt_144 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %k_145 = ttg.memdesc_index %k[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %k_145 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %m_146 = ttg.memdesc_index %m[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %m_146 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %Di_147 = ttg.memdesc_index %Di[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %Di_147 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %9 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %9 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %10 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %10 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %11 = ttg.memdesc_index %5[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %11 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %12 = ttg.memdesc_index %6[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %12 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %qkT_148 = ttg.memdesc_index %qkT_36[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %qkT_148 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %qkT_149 = ttg.memdesc_index %qkT_37[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %qkT_149 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dpT_150 = ttg.memdesc_index %dpT_40[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dpT_150 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dpT_151 = ttg.memdesc_index %dpT_41[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dpT_151 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv_152 = ttg.memdesc_index %dv_44[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dv_152 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv_153 = ttg.memdesc_index %dv_45[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dv_153 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv_154 = ttg.memdesc_index %dv_48[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dv_154 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv_155 = ttg.memdesc_index %dv_49[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dv_155 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv_156 = ttg.memdesc_index %dv_52[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dv_156 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dv_157 = ttg.memdesc_index %dv_53[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dv_157 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_158 = ttg.memdesc_index %dk_56[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dk_158 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_159 = ttg.memdesc_index %dk_57[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dk_159 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_160 = ttg.memdesc_index %dk_60[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dk_160 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_161 = ttg.memdesc_index %dk_61[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dk_161 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_162 = ttg.memdesc_index %dk_64[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dk_162 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dk_163 = ttg.memdesc_index %dk_65[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dk_163 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %m_164 = ttg.memdesc_index %m_68[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %m_164 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %m_165 = ttg.memdesc_index %m_69[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %m_165 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %ppT_166 = ttg.memdesc_index %ppT_72[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %ppT_166 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %ppT_167 = ttg.memdesc_index %ppT_73[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %ppT_167 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %Di_168 = ttg.memdesc_index %Di_76[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %Di_168 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %Di_169 = ttg.memdesc_index %Di_77[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %Di_169 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT_170 = ttg.memdesc_index %dsT_80[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dsT_170 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT_171 = ttg.memdesc_index %dsT_81[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dsT_171 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT_dq_1_172 = ttg.memdesc_index %dsT_dq_1[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dsT_dq_1_172 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT_dq_1_173 = ttg.memdesc_index %dsT_dq_1_84[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dsT_dq_1_173 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT_dq_0_174 = ttg.memdesc_index %dsT_dq_0_87[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dsT_dq_0_174 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dsT_dq_0_175 = ttg.memdesc_index %dsT_dq_0_88[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dsT_dq_0_175 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dq_176 = ttg.memdesc_index %dq_91[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dq_176 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %dq_177 = ttg.memdesc_index %dq_92[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    ttng.inval_barrier %dq_177 : !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    tt.return
  }
}

// RUN: triton-opt %s --nvgpu-insert-2cta-sync | FileCheck %s

// The full input is the BM128 2-CTA backward TTGIR immediately before
// NVGPUInsert2CTASync. Inner-loop specialization leaves each partition MMA
// outside an scf.for even though the WarpSpecializeOp itself is loop-nested.
// CHECK-LABEL: tt.func public @_attn_bwd
// CHECK: ttng.init_barrier {{.*}}, 2
// CHECK: ttg.warp_specialize(
// CHECK-NOT: ttng.init_barrier
// CHECK: tt.return
