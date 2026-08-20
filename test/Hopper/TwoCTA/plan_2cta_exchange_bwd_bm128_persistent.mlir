// RUN: triton-opt %s --nvgpu-plan-2cta-exchange | FileCheck %s

// Full TTGIR captured immediately before NVGPUPlan2CTAExchange from the
// non-causal persistent BM128 2-CTA configuration with adjacent N ownership.
// CHECK: %[[GATHER:.*]] = ttng.two_cta_peer_gather %{{.*}} split_dim = 1 num_ctas = 2
// CHECK-SAME: tensor<128x128xf16{{.*}}> -> tensor<64x256xf16{{.*}}>
// CHECK: %[[PHYSICAL:.*]] = tt.trans %[[GATHER]]
// CHECK-SAME: tensor<64x256xf16{{.*}}> -> tensor<256x64xf16{{.*}}>
// CHECK: %[[EXCHANGE:.*]] = ttg.local_alloc
// CHECK-SAME: !ttg.memdesc<128x64xf16, #shared, #smem>
// CHECK: %[[ALLOC:.*]] = ttg.local_alloc %[[PHYSICAL]]
// CHECK-SAME: !ttg.memdesc<256x64xf16, #shared, #smem>
// CHECK: ttng.two_cta_peer_relay %[[EXCHANGE]]
// CHECK: %[[VIEW:.*]] = ttg.memdesc_trans %[[ALLOC]]
// CHECK-SAME: !ttg.memdesc<256x64xf16, #shared, #smem> -> !ttg.memdesc<64x256xf16, #shared1, #smem>
// CHECK: ttng.tc_gen5_mma %[[VIEW]],

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [8, 1, 4], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 128], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [0, 1, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [0, 0, 1]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 0, 1], [0, 32, 0], [0, 16, 0], [0, 1, 0], [0, 2, 0], [32, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 32], [0, 16], [0, 1], [0, 2], [32, 0]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
#linear9 = #ttg.linear<{register = [[0, 1, 0], [0, 0, 16], [0, 0, 1], [0, 0, 2], [32, 0, 0]], lane = [[0, 0, 4], [0, 0, 8], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0]], block = []}>
#linear10 = #ttg.linear<{register = [[0, 0, 1], [0, 16, 0], [0, 1, 0], [0, 2, 0], [32, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0]], block = []}>
#linear11 = #ttg.linear<{register = [[0, 16], [0, 1], [0, 2], [32, 0]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
#linear12 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear13 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear14 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear15 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear16 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear17 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear18 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear19 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true, ctaMode = twocta_rhs>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @_attn_bwd_persist(%desc_q: !tt.tensordesc<128x64xf16>, %desc_q.shape.0: i32, %desc_q.shape.1: i32, %desc_q.stride.0: i64, %desc_q.stride.1: i64, %desc_qt: !tt.tensordesc<64x128xf16>, %desc_qt.shape.0: i32, %desc_qt.shape.1: i32, %desc_qt.stride.0: i64, %desc_qt.stride.1: i64, %desc_k: !tt.tensordesc<128x128xf16>, %desc_k.shape.0: i32, %desc_k.shape.1: i32, %desc_k.stride.0: i64, %desc_k.stride.1: i64, %desc_kt: !tt.tensordesc<256x64xf16>, %desc_kt.shape.0: i32, %desc_kt.shape.1: i32, %desc_kt.stride.0: i64, %desc_kt.stride.1: i64, %desc_v: !tt.tensordesc<128x128xf16>, %desc_v.shape.0: i32, %desc_v.shape.1: i32, %desc_v.stride.0: i64, %desc_v.stride.1: i64, %sm_scale: f32, %desc_do: !tt.tensordesc<128x64xf16>, %desc_do.shape.0: i32, %desc_do.shape.1: i32, %desc_do.stride.0: i64, %desc_do.stride.1: i64, %desc_dot: !tt.tensordesc<64x128xf16>, %desc_dot.shape.0: i32, %desc_dot.shape.1: i32, %desc_dot.stride.0: i64, %desc_dot.stride.1: i64, %desc_dq: !tt.tensordesc<64x16xf32>, %desc_dq.shape.0: i32, %desc_dq.shape.1: i32, %desc_dq.stride.0: i64, %desc_dq.stride.1: i64, %desc_dk: !tt.tensordesc<128x16xf16>, %desc_dk.shape.0: i32, %desc_dk.shape.1: i32, %desc_dk.stride.0: i64, %desc_dk.stride.1: i64, %desc_dv: !tt.tensordesc<128x16xf16>, %desc_dv.shape.0: i32, %desc_dv.shape.1: i32, %desc_dv.stride.0: i64, %desc_dv.stride.1: i64, %desc_m: !tt.tensordesc<128xf32>, %desc_m.shape.0: i32, %desc_m.stride.0: i64, %desc_delta: !tt.tensordesc<128xf32>, %desc_delta.shape.0: i32, %desc_delta.stride.0: i64, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32 {tt.divisibility = 16 : i32}, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #linear>
    %cst_0 = arith.constant dense<0.693147182> : tensor<64x16xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %n_tile_num = arith.constant 127 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c48_i32 = arith.constant 48 : i32
    %c64_i32 = arith.constant 64 : i32
    %c80_i32 = arith.constant 80 : i32
    %c96_i32 = arith.constant 96 : i32
    %c112_i32 = arith.constant 112 : i32
    %true = arith.constant true
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear1>
    %n_tile_num_2 = arith.addi %N_CTX, %n_tile_num : i32
    %n_tile_num_3 = arith.divsi %n_tile_num_2, %c128_i32 : i32
    %cluster_rank = tt.get_program_id x : i32
    %cluster_rank_4 = arith.remsi %cluster_rank, %c2_i32 : i32
    %prog_id = arith.divsi %cluster_rank, %c2_i32 : i32
    %num_progs = tt.get_num_programs x : i32
    %num_progs_5 = arith.divsi %num_progs, %c2_i32 : i32
    %scheduled_n_tiles = arith.divsi %n_tile_num_3, %c2_i32 : i32
    %total_tiles = arith.muli %scheduled_n_tiles, %BATCH : i32
    %total_tiles_6 = arith.muli %total_tiles, %H : i32
    %tiles_per_sm = arith.divsi %total_tiles_6, %num_progs_5 : i32
    %0 = arith.remsi %total_tiles_6, %num_progs_5 : i32
    %1 = arith.cmpi slt, %prog_id, %0 : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_15 = arith.addi %tiles_per_sm, %c1_i32 : i32
      scf.yield %tiles_per_sm_15 : i32
    } else {
      scf.yield %tiles_per_sm : i32
    }
    %off_bh = arith.extsi %stride_tok : i32 to i64
    %kt_start_n = arith.muli %cluster_rank_4, %c128_i32 : i32
    %num_steps = arith.divsi %N_CTX, %c128_i32 : i32
    %dq_row = arith.muli %cluster_rank_4, %c64_i32 : i32
    %dq_row_7 = arith.extsi %dq_row : i32 to i64
    %dkN = tt.splat %sm_scale : f32 -> tensor<128x16xf32, #linear2>
    %dkN_8 = tt.splat %sm_scale : f32 -> tensor<128x16xf32, #linear2>
    %dkN_9 = tt.splat %sm_scale : f32 -> tensor<128x16xf32, #linear2>
    %dkN_10 = tt.splat %sm_scale : f32 -> tensor<128x16xf32, #linear2>
    %dkN_11 = tt.splat %sm_scale : f32 -> tensor<128x16xf32, #linear2>
    %dkN_12 = tt.splat %sm_scale : f32 -> tensor<128x16xf32, #linear2>
    %dkN_13 = tt.splat %sm_scale : f32 -> tensor<128x16xf32, #linear2>
    %dkN_14 = tt.splat %sm_scale : f32 -> tensor<128x16xf32, #linear2>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_15 = %prog_id) -> (i32)  : i32 {
      %scheduled_pid = arith.remsi %tile_idx_15, %scheduled_n_tiles : i32
      %bhid = arith.divsi %tile_idx_15, %scheduled_n_tiles : i32
      %pid = arith.muli %scheduled_pid, %c2_i32 : i32
      %pid_16 = arith.addi %pid, %cluster_rank_4 : i32
      %off_chz = arith.muli %bhid, %N_CTX : i32
      %off_chz_17 = arith.extsi %off_chz : i32 to i64
      %off_bh_18 = arith.remsi %bhid, %H : i32
      %off_bh_19 = arith.muli %stride_h, %off_bh_18 : i32
      %off_bh_20 = arith.divsi %bhid, %H : i32
      %off_bh_21 = arith.muli %stride_z, %off_bh_20 : i32
      %off_bh_22 = arith.addi %off_bh_19, %off_bh_21 : i32
      %off_bh_23 = arith.extsi %off_bh_22 : i32 to i64
      %off_bh_24 = arith.divsi %off_bh_23, %off_bh : i64
      %start_n = arith.muli %pid_16, %c128_i32 : i32
      %k = arith.extsi %start_n : i32 to i64
      %k_25 = arith.addi %off_bh_24, %k : i64
      %k_26 = arith.trunci %k_25 : i64 to i32
      %k_27 = tt.descriptor_load %desc_k[%k_26, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked1>
      %k_28 = ttg.local_alloc %k_27 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %kt_start_n_29 = arith.subi %start_n, %kt_start_n : i32
      %kt = arith.extsi %kt_start_n_29 : i32 to i64
      %kt_30 = arith.addi %off_bh_24, %kt : i64
      %kt_31 = arith.trunci %kt_30 : i64 to i32
      %kt_32 = nvg.cluster_id
      %kt_33 = arith.constant 2 : i32
      %kt_34 = arith.remsi %kt_32, %kt_33 : i32
      %kt_35 = arith.constant 64 : i32
      %kt_36 = arith.muli %kt_34, %kt_35 : i32
      %kt_37 = arith.addi %c0_i32, %kt_36 : i32
      %kt_38 = tt.descriptor_load %desc_kt[%kt_31, %kt_37] {two_cta_b} : !tt.tensordesc<256x64xf16> -> tensor<256x64xf16, #blocked2>
      %kt_39 = ttg.local_alloc %kt_38 : (tensor<256x64xf16, #blocked2>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
      %v = tt.descriptor_load %desc_v[%k_26, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked1>
      %v_40 = ttg.local_alloc %v : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %curr_m:3 = scf.for %blk_idx = %c0_i32 to %num_steps step %c1_i32 iter_args(%dk = %cst_1, %dv = %cst_1, %curr_m_104 = %c0_i32) -> (tensor<128x128xf32, #linear1>, tensor<128x128xf32, #linear1>, i32)  : i32 {
        %q = arith.extsi %curr_m_104 : i32 to i64
        %q_105 = arith.addi %off_bh_24, %q : i64
        %q_106 = arith.trunci %q_105 : i64 to i32
        %q_107 = nvg.cluster_id
        %q_108 = arith.constant 2 : i32
        %q_109 = arith.remsi %q_107, %q_108 : i32
        %q_110 = arith.constant 64 : i32
        %q_111 = arith.muli %q_109, %q_110 : i32
        %q_112 = arith.addi %c0_i32, %q_111 : i32
        %q_113 = tt.descriptor_load %desc_q[%q_106, %q_112] {two_cta_b} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked2>
        %q_114 = ttg.local_alloc %q_113 : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %qt = nvg.cluster_id
        %qt_115 = arith.constant 2 : i32
        %qt_116 = arith.remsi %qt, %qt_115 : i32
        %qt_117 = arith.constant 64 : i32
        %qt_118 = arith.muli %qt_116, %qt_117 : i32
        %qt_119 = arith.addi %q_106, %qt_118 : i32
        %qt_120 = tt.descriptor_load %desc_qt[%qt_119, %c0_i32] {two_cta_b} : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked1>
        %qT = ttg.local_alloc %qt_120 : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %qT_121 = ttg.memdesc_trans %qT {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared1, #smem>
        %offs_m_start = arith.addi %off_chz_17, %q : i64
        %m = arith.trunci %offs_m_start : i64 to i32
        %m_122 = tt.descriptor_load %desc_m[%m] : !tt.tensordesc<128xf32> -> tensor<128xf32, #blocked3>
        %qkT, %qkT_123 = ttng.tmem_alloc %cst_1 : (tensor<128x128xf32, #linear1>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %qkT_124 = ttng.tc_gen5_mma %k_28, %qT_121, %qkT[%qkT_123], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qkT_125, %qkT_126 = ttng.tmem_load %qkT[%qkT_124] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %pT = ttg.convert_layout %m_122 : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %pT_127 = tt.expand_dims %pT {axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x128xf32, #linear1>
        %pT_128 = tt.broadcast %pT_127 : tensor<1x128xf32, #linear1> -> tensor<128x128xf32, #linear1>
        %pT_129 = arith.subf %qkT_125, %pT_128 : tensor<128x128xf32, #linear1>
        %pT_130 = math.exp2 %pT_129 : tensor<128x128xf32, #linear1>
        %do = nvg.cluster_id
        %do_131 = arith.constant 2 : i32
        %do_132 = arith.remsi %do, %do_131 : i32
        %do_133 = arith.constant 64 : i32
        %do_134 = arith.muli %do_132, %do_133 : i32
        %do_135 = arith.addi %c0_i32, %do_134 : i32
        %do_136 = tt.descriptor_load %desc_do[%q_106, %do_135] {two_cta_b} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked2>
        %do_137 = ttg.local_alloc %do_136 : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %dot = nvg.cluster_id
        %dot_138 = arith.constant 2 : i32
        %dot_139 = arith.remsi %dot, %dot_138 : i32
        %dot_140 = arith.constant 64 : i32
        %dot_141 = arith.muli %dot_139, %dot_140 : i32
        %dot_142 = arith.addi %q_106, %dot_141 : i32
        %dot_143 = tt.descriptor_load %desc_dot[%dot_142, %c0_i32] {two_cta_b} : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked1>
        %ppT = arith.truncf %pT_130 : tensor<128x128xf32, #linear1> to tensor<128x128xf16, #linear1>
        %ppT_144 = ttg.local_alloc %ppT : (tensor<128x128xf16, #linear1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %dpT = ttg.local_alloc %dot_143 : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %dpT_145 = ttg.memdesc_trans %dpT {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared1, #smem>
        %dpT_146, %dpT_147 = ttng.tmem_alloc %cst_1 : (tensor<128x128xf32, #linear1>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %dpT_148 = ttng.tc_gen5_mma %v_40, %dpT_145, %dpT_146[%dpT_147], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dpT_149, %dpT_150 = ttng.tmem_load %dpT_146[%dpT_148] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %Di = tt.descriptor_load %desc_delta[%m] : !tt.tensordesc<128xf32> -> tensor<128xf32, #blocked3>
        %dv_151, %dv_152 = ttng.tmem_alloc %dv : (tensor<128x128xf32, #linear1>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %dv_153 = ttng.tc_gen5_mma %ppT_144, %do_137, %dv_151[%dv_152], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dv_154, %dv_155 = ttng.tmem_load %dv_151[%dv_153] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %dsT = ttg.convert_layout %Di : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %dsT_156 = tt.expand_dims %dsT {axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x128xf32, #linear1>
        %dsT_157 = tt.broadcast %dsT_156 : tensor<1x128xf32, #linear1> -> tensor<128x128xf32, #linear1>
        %dsT_158 = arith.subf %dpT_149, %dsT_157 : tensor<128x128xf32, #linear1>
        %dsT_159 = arith.mulf %pT_130, %dsT_158 : tensor<128x128xf32, #linear1>
        %dsT_160 = arith.truncf %dsT_159 : tensor<128x128xf32, #linear1> to tensor<128x128xf16, #linear1>
        %dsT_161 = ttg.local_alloc %dsT_160 : (tensor<128x128xf16, #linear1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %dk_162, %dk_163 = ttng.tmem_alloc %dk : (tensor<128x128xf32, #linear1>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %dk_164 = ttng.tc_gen5_mma %dsT_161, %q_114, %dk_162[%dk_163], %true, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndD,tmem,1,10\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dk_165, %dk_166 = ttng.tmem_load %dk_162[%dk_164] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %dsT_dq = tt.trans %dsT_160 {order = array<i32: 1, 0>} : tensor<128x128xf16, #linear1> -> tensor<128x128xf16, #linear3>
        %dsT_dq_167 = tt.reshape %dsT_dq : tensor<128x128xf16, #linear3> -> tensor<64x256xf16, #linear4>
        %dsT_dq_168 = ttg.local_alloc %dsT_dq_167 : (tensor<64x256xf16, #linear4>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        %dq, %dq_169 = ttng.tmem_alloc %cst : (tensor<64x128xf32, #linear>) -> (!ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %dq_170 = ttng.tc_gen5_mma %dsT_dq_168, %kt_39, %dq[%dq_169], %true, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x256xf16, #shared, #smem>, !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %dq_171, %dq_172 = ttng.tmem_load %dq[%dq_170] : !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
        %dqs = tt.reshape %dq_171 : tensor<64x128xf32, #linear> -> tensor<64x2x64xf32, #linear5>
        %dqs_173 = tt.trans %dqs {order = array<i32: 0, 2, 1>} : tensor<64x2x64xf32, #linear5> -> tensor<64x64x2xf32, #linear6>
        %dqs_174 = ttg.convert_layout %dqs_173 : tensor<64x64x2xf32, #linear6> -> tensor<64x64x2xf32, #linear7>
        %dqs_175, %dqs_176 = tt.split %dqs_174 : tensor<64x64x2xf32, #linear7> -> tensor<64x64xf32, #linear8>
        %dqs_177 = tt.reshape %dqs_175 : tensor<64x64xf32, #linear8> -> tensor<64x2x32xf32, #linear9>
        %dqs_178 = tt.trans %dqs_177 {order = array<i32: 0, 2, 1>} : tensor<64x2x32xf32, #linear9> -> tensor<64x32x2xf32, #linear10>
        %dqs_179, %dqs_180 = tt.split %dqs_178 : tensor<64x32x2xf32, #linear10> -> tensor<64x32xf32, #linear11>
        %dqs_181 = tt.reshape %dqs_179 : tensor<64x32xf32, #linear11> -> tensor<64x2x16xf32, #blocked4>
        %dqs_182 = tt.trans %dqs_181 {order = array<i32: 0, 2, 1>} : tensor<64x2x16xf32, #blocked4> -> tensor<64x16x2xf32, #blocked5>
        %dqs_183, %dqs_184 = tt.split %dqs_182 : tensor<64x16x2xf32, #blocked5> -> tensor<64x16xf32, #blocked>
        %dqs_185 = tt.reshape %dqs_180 : tensor<64x32xf32, #linear11> -> tensor<64x2x16xf32, #blocked4>
        %dqs_186 = tt.trans %dqs_185 {order = array<i32: 0, 2, 1>} : tensor<64x2x16xf32, #blocked4> -> tensor<64x16x2xf32, #blocked5>
        %dqs_187, %dqs_188 = tt.split %dqs_186 : tensor<64x16x2xf32, #blocked5> -> tensor<64x16xf32, #blocked>
        %dqs_189 = tt.reshape %dqs_176 : tensor<64x64xf32, #linear8> -> tensor<64x2x32xf32, #linear9>
        %dqs_190 = tt.trans %dqs_189 {order = array<i32: 0, 2, 1>} : tensor<64x2x32xf32, #linear9> -> tensor<64x32x2xf32, #linear10>
        %dqs_191, %dqs_192 = tt.split %dqs_190 : tensor<64x32x2xf32, #linear10> -> tensor<64x32xf32, #linear11>
        %dqs_193 = tt.reshape %dqs_191 : tensor<64x32xf32, #linear11> -> tensor<64x2x16xf32, #blocked4>
        %dqs_194 = tt.trans %dqs_193 {order = array<i32: 0, 2, 1>} : tensor<64x2x16xf32, #blocked4> -> tensor<64x16x2xf32, #blocked5>
        %dqs_195, %dqs_196 = tt.split %dqs_194 : tensor<64x16x2xf32, #blocked5> -> tensor<64x16xf32, #blocked>
        %dqs_197 = tt.reshape %dqs_192 : tensor<64x32xf32, #linear11> -> tensor<64x2x16xf32, #blocked4>
        %dqs_198 = tt.trans %dqs_197 {order = array<i32: 0, 2, 1>} : tensor<64x2x16xf32, #blocked4> -> tensor<64x16x2xf32, #blocked5>
        %dqs_199, %dqs_200 = tt.split %dqs_198 : tensor<64x16x2xf32, #blocked5> -> tensor<64x16xf32, #blocked>
        %dq_row_201 = arith.addi %q_105, %dq_row_7 : i64
        %dqN = arith.mulf %dqs_183, %cst_0 : tensor<64x16xf32, #blocked>
        %35 = arith.trunci %dq_row_201 : i64 to i32
        tt.descriptor_reduce add, %desc_dq[%35, %c0_i32], %dqN : !tt.tensordesc<64x16xf32>, tensor<64x16xf32, #blocked>
        %dqN_202 = arith.mulf %dqs_184, %cst_0 : tensor<64x16xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%35, %c16_i32], %dqN_202 : !tt.tensordesc<64x16xf32>, tensor<64x16xf32, #blocked>
        %dqN_203 = arith.mulf %dqs_187, %cst_0 : tensor<64x16xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%35, %c32_i32], %dqN_203 : !tt.tensordesc<64x16xf32>, tensor<64x16xf32, #blocked>
        %dqN_204 = arith.mulf %dqs_188, %cst_0 : tensor<64x16xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%35, %c48_i32], %dqN_204 : !tt.tensordesc<64x16xf32>, tensor<64x16xf32, #blocked>
        %dqN_205 = arith.mulf %dqs_195, %cst_0 : tensor<64x16xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%35, %c64_i32], %dqN_205 : !tt.tensordesc<64x16xf32>, tensor<64x16xf32, #blocked>
        %dqN_206 = arith.mulf %dqs_196, %cst_0 : tensor<64x16xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%35, %c80_i32], %dqN_206 : !tt.tensordesc<64x16xf32>, tensor<64x16xf32, #blocked>
        %dqN_207 = arith.mulf %dqs_199, %cst_0 : tensor<64x16xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%35, %c96_i32], %dqN_207 : !tt.tensordesc<64x16xf32>, tensor<64x16xf32, #blocked>
        %dqN_208 = arith.mulf %dqs_200, %cst_0 : tensor<64x16xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%35, %c112_i32], %dqN_208 : !tt.tensordesc<64x16xf32>, tensor<64x16xf32, #blocked>
        %curr_m_209 = arith.addi %curr_m_104, %c128_i32 : i32
        scf.yield %dk_165, %dv_154, %curr_m_209 : tensor<128x128xf32, #linear1>, tensor<128x128xf32, #linear1>, i32
      }
      %dvs = tt.reshape %curr_m#1 : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear12>
      %dvs_41 = tt.trans %dvs {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear12> -> tensor<128x64x2xf32, #linear13>
      %dvs_42, %dvs_43 = tt.split %dvs_41 : tensor<128x64x2xf32, #linear13> -> tensor<128x64xf32, #linear14>
      %dvs_44 = tt.reshape %dvs_42 : tensor<128x64xf32, #linear14> -> tensor<128x2x32xf32, #linear15>
      %dvs_45 = tt.trans %dvs_44 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear15> -> tensor<128x32x2xf32, #linear16>
      %dvs_46, %dvs_47 = tt.split %dvs_45 : tensor<128x32x2xf32, #linear16> -> tensor<128x32xf32, #linear17>
      %dvs_48 = tt.reshape %dvs_46 : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
      %dvs_49 = tt.trans %dvs_48 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
      %dvs_50, %dvs_51 = tt.split %dvs_49 : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear2>
      %dvs_52 = tt.reshape %dvs_47 : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
      %dvs_53 = tt.trans %dvs_52 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
      %dvs_54, %dvs_55 = tt.split %dvs_53 : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear2>
      %dvs_56 = tt.reshape %dvs_43 : tensor<128x64xf32, #linear14> -> tensor<128x2x32xf32, #linear15>
      %dvs_57 = tt.trans %dvs_56 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear15> -> tensor<128x32x2xf32, #linear16>
      %dvs_58, %dvs_59 = tt.split %dvs_57 : tensor<128x32x2xf32, #linear16> -> tensor<128x32xf32, #linear17>
      %dvs_60 = tt.reshape %dvs_58 : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
      %dvs_61 = tt.trans %dvs_60 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
      %dvs_62, %dvs_63 = tt.split %dvs_61 : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear2>
      %dvs_64 = tt.reshape %dvs_59 : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
      %dvs_65 = tt.trans %dvs_64 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
      %dvs_66, %dvs_67 = tt.split %dvs_65 : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear2>
      %3 = arith.truncf %dvs_50 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %4 = ttg.convert_layout %3 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dv[%k_26, %c0_i32], %4 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %5 = arith.truncf %dvs_51 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %6 = ttg.convert_layout %5 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dv[%k_26, %c16_i32], %6 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %7 = arith.truncf %dvs_54 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %8 = ttg.convert_layout %7 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dv[%k_26, %c32_i32], %8 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %9 = arith.truncf %dvs_55 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %10 = ttg.convert_layout %9 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dv[%k_26, %c48_i32], %10 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %11 = arith.truncf %dvs_62 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %12 = ttg.convert_layout %11 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dv[%k_26, %c64_i32], %12 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %13 = arith.truncf %dvs_63 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %14 = ttg.convert_layout %13 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dv[%k_26, %c80_i32], %14 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %15 = arith.truncf %dvs_66 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %16 = ttg.convert_layout %15 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dv[%k_26, %c96_i32], %16 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %17 = arith.truncf %dvs_67 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %18 = ttg.convert_layout %17 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dv[%k_26, %c112_i32], %18 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %dks = tt.reshape %curr_m#0 : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear12>
      %dks_68 = tt.trans %dks {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear12> -> tensor<128x64x2xf32, #linear13>
      %dks_69, %dks_70 = tt.split %dks_68 : tensor<128x64x2xf32, #linear13> -> tensor<128x64xf32, #linear14>
      %dks_71 = tt.reshape %dks_69 : tensor<128x64xf32, #linear14> -> tensor<128x2x32xf32, #linear15>
      %dks_72 = tt.trans %dks_71 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear15> -> tensor<128x32x2xf32, #linear16>
      %dks_73, %dks_74 = tt.split %dks_72 : tensor<128x32x2xf32, #linear16> -> tensor<128x32xf32, #linear17>
      %dks_75 = tt.reshape %dks_73 : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
      %dks_76 = tt.trans %dks_75 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
      %dks_77, %dks_78 = tt.split %dks_76 : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear2>
      %dks_79 = tt.reshape %dks_74 : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
      %dks_80 = tt.trans %dks_79 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
      %dks_81, %dks_82 = tt.split %dks_80 : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear2>
      %dks_83 = tt.reshape %dks_70 : tensor<128x64xf32, #linear14> -> tensor<128x2x32xf32, #linear15>
      %dks_84 = tt.trans %dks_83 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear15> -> tensor<128x32x2xf32, #linear16>
      %dks_85, %dks_86 = tt.split %dks_84 : tensor<128x32x2xf32, #linear16> -> tensor<128x32xf32, #linear17>
      %dks_87 = tt.reshape %dks_85 : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
      %dks_88 = tt.trans %dks_87 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
      %dks_89, %dks_90 = tt.split %dks_88 : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear2>
      %dks_91 = tt.reshape %dks_86 : tensor<128x32xf32, #linear17> -> tensor<128x2x16xf32, #linear18>
      %dks_92 = tt.trans %dks_91 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #linear18> -> tensor<128x16x2xf32, #linear19>
      %dks_93, %dks_94 = tt.split %dks_92 : tensor<128x16x2xf32, #linear19> -> tensor<128x16xf32, #linear2>
      %dkN_95 = arith.mulf %dks_77, %dkN_14 : tensor<128x16xf32, #linear2>
      %19 = arith.truncf %dkN_95 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %20 = ttg.convert_layout %19 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dk[%k_26, %c0_i32], %20 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %dkN_96 = arith.mulf %dks_78, %dkN_13 : tensor<128x16xf32, #linear2>
      %21 = arith.truncf %dkN_96 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %22 = ttg.convert_layout %21 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dk[%k_26, %c16_i32], %22 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %dkN_97 = arith.mulf %dks_81, %dkN_12 : tensor<128x16xf32, #linear2>
      %23 = arith.truncf %dkN_97 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %24 = ttg.convert_layout %23 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dk[%k_26, %c32_i32], %24 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %dkN_98 = arith.mulf %dks_82, %dkN_11 : tensor<128x16xf32, #linear2>
      %25 = arith.truncf %dkN_98 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %26 = ttg.convert_layout %25 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dk[%k_26, %c48_i32], %26 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %dkN_99 = arith.mulf %dks_89, %dkN_10 : tensor<128x16xf32, #linear2>
      %27 = arith.truncf %dkN_99 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %28 = ttg.convert_layout %27 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dk[%k_26, %c64_i32], %28 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %dkN_100 = arith.mulf %dks_90, %dkN_9 : tensor<128x16xf32, #linear2>
      %29 = arith.truncf %dkN_100 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %30 = ttg.convert_layout %29 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dk[%k_26, %c80_i32], %30 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %dkN_101 = arith.mulf %dks_93, %dkN_8 : tensor<128x16xf32, #linear2>
      %31 = arith.truncf %dkN_101 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %32 = ttg.convert_layout %31 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dk[%k_26, %c96_i32], %32 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %dkN_102 = arith.mulf %dks_94, %dkN : tensor<128x16xf32, #linear2>
      %33 = arith.truncf %dkN_102 : tensor<128x16xf32, #linear2> to tensor<128x16xf16, #linear2>
      %34 = ttg.convert_layout %33 : tensor<128x16xf16, #linear2> -> tensor<128x16xf16, #blocked6>
      tt.descriptor_store %desc_dk[%k_26, %c112_i32], %34 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked6>
      %tile_idx_103 = arith.addi %tile_idx_15, %num_progs_5 : i32
      scf.yield %tile_idx_103 : i32
    } {tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
