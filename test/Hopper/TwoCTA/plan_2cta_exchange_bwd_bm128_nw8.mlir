// RUN: triton-opt %s --nvgpu-plan-2cta-exchange | FileCheck %s

// Full BM128 2-CTA AutoWS TTGIR captured immediately before
// NVGPUPlan2CTAExchange with the TLX-aligned eight-warp base layout.
// CHECK: ttng.two_cta_peer_gather
// CHECK: ttg.convert_layout

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 2, 4], threadsPerWarp = [8, 1, 4], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 2, 8], threadsPerWarp = [16, 1, 2], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 64], [0, 32]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#linear2 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64], [64, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 128], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64], [32, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0], [0, 1, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], lane = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], warp = [[64, 0, 0], [1, 0, 0], [0, 0, 1]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 0, 1], [0, 16, 0], [0, 1, 0], [0, 2, 0], [64, 0, 0]], lane = [[0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0], [32, 0, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 16], [0, 1], [0, 2], [64, 0]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [32, 0]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 1, 0]], block = []}>
#linear9 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 0, 1]], block = []}>
#linear10 = #ttg.linear<{register = [[0, 0, 1], [0, 32, 0], [0, 16, 0], [0, 1, 0], [0, 2, 0], [0, 4, 0]], lane = [[0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]], warp = [[16, 0, 0], [32, 0, 0], [64, 0, 0]], block = []}>
#linear11 = #ttg.linear<{register = [[0, 32], [0, 16], [0, 1], [0, 2], [0, 4]], lane = [[0, 8], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[16, 0], [32, 0], [64, 0]], block = []}>
#linear12 = #ttg.linear<{register = [[0, 1, 0], [0, 0, 16], [0, 0, 1], [0, 0, 2], [0, 0, 4]], lane = [[0, 0, 8], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]], warp = [[16, 0, 0], [32, 0, 0], [64, 0, 0]], block = []}>
#linear13 = #ttg.linear<{register = [[0, 0, 1], [0, 16, 0], [0, 1, 0], [0, 2, 0], [0, 4, 0]], lane = [[0, 8, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]], warp = [[16, 0, 0], [32, 0, 0], [64, 0, 0]], block = []}>
#linear14 = #ttg.linear<{register = [[0, 16], [0, 1], [0, 2], [0, 4]], lane = [[0, 8], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[16, 0], [32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true, ctaMode = twocta_rhs>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @_attn_bwd_persist(%desc_q: !tt.tensordesc<128x64xf16>, %desc_q.shape.0: i32, %desc_q.shape.1: i32, %desc_q.stride.0: i64, %desc_q.stride.1: i64, %desc_qt: !tt.tensordesc<64x128xf16>, %desc_qt.shape.0: i32, %desc_qt.shape.1: i32, %desc_qt.stride.0: i64, %desc_qt.stride.1: i64, %desc_k: !tt.tensordesc<128x128xf16>, %desc_k.shape.0: i32, %desc_k.shape.1: i32, %desc_k.stride.0: i64, %desc_k.stride.1: i64, %desc_kt: !tt.tensordesc<256x64xf16>, %desc_kt.shape.0: i32, %desc_kt.shape.1: i32, %desc_kt.stride.0: i64, %desc_kt.stride.1: i64, %desc_v: !tt.tensordesc<128x128xf16>, %desc_v.shape.0: i32, %desc_v.shape.1: i32, %desc_v.stride.0: i64, %desc_v.stride.1: i64, %sm_scale: f32, %desc_do: !tt.tensordesc<128x64xf16>, %desc_do.shape.0: i32, %desc_do.shape.1: i32, %desc_do.stride.0: i64, %desc_do.stride.1: i64, %desc_dot: !tt.tensordesc<64x128xf16>, %desc_dot.shape.0: i32, %desc_dot.shape.1: i32, %desc_dot.stride.0: i64, %desc_dot.stride.1: i64, %desc_dq: !tt.tensordesc<128x16xf32>, %desc_dq.shape.0: i32, %desc_dq.shape.1: i32, %desc_dq.stride.0: i64, %desc_dq.stride.1: i64, %desc_dk: !tt.tensordesc<128x16xf16>, %desc_dk.shape.0: i32, %desc_dk.shape.1: i32, %desc_dk.stride.0: i64, %desc_dk.stride.1: i64, %desc_dv: !tt.tensordesc<128x16xf16>, %desc_dv.shape.0: i32, %desc_dv.shape.1: i32, %desc_dv.stride.0: i64, %desc_dv.stride.1: i64, %desc_m: !tt.tensordesc<128xf32>, %desc_m.shape.0: i32, %desc_m.stride.0: i64, %desc_delta: !tt.tensordesc<128xf32>, %desc_delta.shape.0: i32, %desc_delta.stride.0: i64, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #linear>
    %cst_0 = arith.constant dense<0.693147182> : tensor<128x16xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %n_tile_num = arith.constant 127 : i32
    %c2_i64 = arith.constant 2 : i64
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
      %tiles_per_sm_8 = arith.addi %tiles_per_sm, %c1_i32 : i32
      scf.yield %tiles_per_sm_8 : i32
    } else {
      scf.yield %tiles_per_sm : i32
    }
    %off_bh = arith.extsi %stride_tok : i32 to i64
    %kt_start_n = arith.muli %cluster_rank_4, %c128_i32 : i32
    %num_steps = arith.divsi %N_CTX, %c128_i32 : i32
    %dq_row = arith.muli %cluster_rank_4, %c64_i32 : i32
    %dq_row_7 = arith.extsi %dq_row : i32 to i64
    %dkN = tt.splat %sm_scale : f32 -> tensor<128x16xf32, #blocked1>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_8 = %prog_id) -> (i32)  : i32 {
      %scheduled_pid = arith.remsi %tile_idx_8, %scheduled_n_tiles : i32
      %bhid = arith.divsi %tile_idx_8, %scheduled_n_tiles : i32
      %pid = arith.muli %scheduled_pid, %c2_i32 : i32
      %pid_9 = arith.addi %pid, %cluster_rank_4 : i32
      %off_chz = arith.muli %bhid, %N_CTX : i32
      %off_chz_10 = arith.extsi %off_chz : i32 to i64
      %off_bh_11 = arith.remsi %bhid, %H : i32
      %off_bh_12 = arith.muli %stride_h, %off_bh_11 : i32
      %off_bh_13 = arith.divsi %bhid, %H : i32
      %off_bh_14 = arith.muli %stride_z, %off_bh_13 : i32
      %off_bh_15 = arith.addi %off_bh_12, %off_bh_14 : i32
      %off_bh_16 = arith.extsi %off_bh_15 : i32 to i64
      %off_bh_17 = arith.divsi %off_bh_16, %off_bh : i64
      %start_n = arith.muli %pid_9, %c128_i32 : i32
      %k = arith.extsi %start_n : i32 to i64
      %k_18 = arith.addi %off_bh_17, %k : i64
      %k_19 = arith.trunci %k_18 : i64 to i32
      %k_20 = tt.descriptor_load %desc_k[%k_19, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked2>
      %k_21 = ttg.local_alloc %k_20 : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %kt_start_n_22 = arith.subi %start_n, %kt_start_n : i32
      %kt = arith.extsi %kt_start_n_22 : i32 to i64
      %kt_23 = arith.addi %off_bh_17, %kt : i64
      %kt_24 = arith.trunci %kt_23 : i64 to i32
      %kt_25 = nvg.cluster_id
      %kt_26 = arith.constant 2 : i32
      %kt_27 = arith.remsi %kt_25, %kt_26 : i32
      %kt_28 = arith.constant 64 : i32
      %kt_29 = arith.muli %kt_27, %kt_28 : i32
      %kt_30 = arith.addi %c0_i32, %kt_29 : i32
      %kt_31 = tt.descriptor_load %desc_kt[%kt_24, %kt_30] {two_cta_b} : !tt.tensordesc<256x64xf16> -> tensor<256x64xf16, #blocked3>
      %kt_32 = ttg.local_alloc %kt_31 : (tensor<256x64xf16, #blocked3>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
      %v = tt.descriptor_load %desc_v[%k_19, %c0_i32] : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #blocked2>
      %v_33 = ttg.local_alloc %v : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %curr_m:3 = scf.for %blk_idx = %c0_i32 to %num_steps step %c1_i32 iter_args(%dk = %cst_1, %dv = %cst_1, %curr_m_99 = %c0_i32) -> (tensor<128x128xf32, #linear1>, tensor<128x128xf32, #linear1>, i32)  : i32 {
        %qt = arith.extsi %curr_m_99 : i32 to i64
        %qt_100 = arith.addi %off_bh_17, %qt : i64
        %qt_101 = arith.trunci %qt_100 : i64 to i32
        %qt_102 = nvg.cluster_id
        %qt_103 = arith.constant 2 : i32
        %qt_104 = arith.remsi %qt_102, %qt_103 : i32
        %qt_105 = arith.constant 64 : i32
        %qt_106 = arith.muli %qt_104, %qt_105 : i32
        %qt_107 = arith.addi %qt_101, %qt_106 : i32
        %qt_108 = tt.descriptor_load %desc_qt[%qt_107, %c0_i32] {two_cta_b} : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked2>
        %qT = ttg.local_alloc %qt_108 : (tensor<64x128xf16, #blocked2>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %qT_109 = ttg.memdesc_trans %qT {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared1, #smem>
        %q = nvg.cluster_id
        %q_110 = arith.constant 2 : i32
        %q_111 = arith.remsi %q, %q_110 : i32
        %q_112 = arith.constant 64 : i32
        %q_113 = arith.muli %q_111, %q_112 : i32
        %q_114 = arith.addi %c0_i32, %q_113 : i32
        %q_115 = tt.descriptor_load %desc_q[%qt_101, %q_114] {two_cta_b} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked3>
        %q_116 = ttg.local_alloc %q_115 : (tensor<128x64xf16, #blocked3>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %offs_m_start = arith.addi %off_chz_10, %qt : i64
        %m = arith.trunci %offs_m_start : i64 to i32
        %m_117 = tt.descriptor_load %desc_m[%m] : !tt.tensordesc<128xf32> -> tensor<128xf32, #blocked4>
        %qkT, %qkT_118 = ttng.tmem_alloc %cst_1 : (tensor<128x128xf32, #linear1>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %qkT_119 = ttng.tc_gen5_mma %k_21, %qT_109, %qkT[%qkT_118], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,1,1\22, \22opndD,tmem,1,2\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qkT_120, %qkT_121 = ttng.tmem_load %qkT[%qkT_119] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %pT = ttg.convert_layout %m_117 : tensor<128xf32, #blocked4> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %pT_122 = tt.expand_dims %pT {axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x128xf32, #linear1>
        %pT_123 = tt.broadcast %pT_122 : tensor<1x128xf32, #linear1> -> tensor<128x128xf32, #linear1>
        %pT_124 = arith.subf %qkT_120, %pT_123 : tensor<128x128xf32, #linear1>
        %pT_125 = math.exp2 %pT_124 : tensor<128x128xf32, #linear1>
        %do = nvg.cluster_id
        %do_126 = arith.constant 2 : i32
        %do_127 = arith.remsi %do, %do_126 : i32
        %do_128 = arith.constant 64 : i32
        %do_129 = arith.muli %do_127, %do_128 : i32
        %do_130 = arith.addi %c0_i32, %do_129 : i32
        %do_131 = tt.descriptor_load %desc_do[%qt_101, %do_130] {two_cta_b} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked3>
        %do_132 = ttg.local_alloc %do_131 : (tensor<128x64xf16, #blocked3>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %dot = nvg.cluster_id
        %dot_133 = arith.constant 2 : i32
        %dot_134 = arith.remsi %dot, %dot_133 : i32
        %dot_135 = arith.constant 64 : i32
        %dot_136 = arith.muli %dot_134, %dot_135 : i32
        %dot_137 = arith.addi %qt_101, %dot_136 : i32
        %dot_138 = tt.descriptor_load %desc_dot[%dot_137, %c0_i32] {two_cta_b} : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked2>
        %ppT = arith.truncf %pT_125 : tensor<128x128xf32, #linear1> to tensor<128x128xf16, #linear1>
        %ppT_139 = ttg.local_alloc %ppT : (tensor<128x128xf16, #linear1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %dpT = ttg.local_alloc %dot_138 : (tensor<64x128xf16, #blocked2>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %dpT_140 = ttg.memdesc_trans %dpT {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared1, #smem>
        %dpT_141, %dpT_142 = ttng.tmem_alloc %cst_1 : (tensor<128x128xf32, #linear1>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %dpT_143 = ttng.tc_gen5_mma %v_33, %dpT_140, %dpT_141[%dpT_142], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dpT_144, %dpT_145 = ttng.tmem_load %dpT_141[%dpT_143] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %Di = tt.descriptor_load %desc_delta[%m] : !tt.tensordesc<128xf32> -> tensor<128xf32, #blocked4>
        %dv_146, %dv_147 = ttng.tmem_alloc %dv : (tensor<128x128xf32, #linear1>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %dv_148 = ttng.tc_gen5_mma %ppT_139, %do_132, %dv_146[%dv_147], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dv_149, %dv_150 = ttng.tmem_load %dv_146[%dv_148] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %dsT = ttg.convert_layout %Di : tensor<128xf32, #blocked4> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear1}>>
        %dsT_151 = tt.expand_dims %dsT {axis = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x128xf32, #linear1>
        %dsT_152 = tt.broadcast %dsT_151 : tensor<1x128xf32, #linear1> -> tensor<128x128xf32, #linear1>
        %dsT_153 = arith.subf %dpT_144, %dsT_152 : tensor<128x128xf32, #linear1>
        %dsT_154 = arith.mulf %pT_125, %dsT_153 : tensor<128x128xf32, #linear1>
        %dsT_155 = arith.truncf %dsT_154 : tensor<128x128xf32, #linear1> to tensor<128x128xf16, #linear1>
        %dsT_156 = ttg.local_alloc %dsT_155 : (tensor<128x128xf16, #linear1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %dk_157, %dk_158 = ttng.tmem_alloc %dk : (tensor<128x128xf32, #linear1>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %dk_159 = ttng.tc_gen5_mma %dsT_156, %q_116, %dk_157[%dk_158], %true, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \223\22, \22channels\22: [\22opndD,tmem,1,10\22]}", ttng.two_cta_dependency = "collective_contraction", two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dk_160, %dk_161 = ttng.tmem_load %dk_157[%dk_159] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %dsT_dq = tt.trans %dsT_155 {order = array<i32: 1, 0>} : tensor<128x128xf16, #linear1> -> tensor<128x128xf16, #linear2>
        %dsT_dq_162 = tt.reshape %dsT_dq : tensor<128x128xf16, #linear2> -> tensor<64x256xf16, #linear3>
        %dsT_dq_163 = ttg.local_alloc %dsT_dq_162 : (tensor<64x256xf16, #linear3>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        %dq, %dq_164 = ttng.tmem_alloc %cst : (tensor<64x128xf32, #linear>) -> (!ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %dq_165 = ttng.tc_gen5_mma %dsT_dq_163, %kt_32, %dq[%dq_164], %true, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}", ttng.two_cta_dependency = "requires_peer_gather", two_ctas} : !ttg.memdesc<64x256xf16, #shared, #smem>, !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %dq_166, %dq_167 = ttng.tmem_load %dq[%dq_165] : !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear>
        %dqs = tt.reshape %dq_166 : tensor<64x128xf32, #linear> -> tensor<128x2x32xf32, #linear4>
        %dqs_168 = tt.trans %dqs {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear4> -> tensor<128x32x2xf32, #linear5>
        %dqs_169 = ttg.convert_layout %dqs_168 : tensor<128x32x2xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
        %dqs_170, %dqs_171 = tt.split %dqs_169 : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear7>
        %dqs_172 = tt.reshape %dqs_170 : tensor<128x32xf32, #linear7> -> tensor<128x2x16xf32, #blocked5>
        %dqs_173 = tt.trans %dqs_172 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked5> -> tensor<128x16x2xf32, #blocked6>
        %dqs_174, %dqs_175 = tt.split %dqs_173 : tensor<128x16x2xf32, #blocked6> -> tensor<128x16xf32, #blocked>
        %dqs_176 = tt.reshape %dqs_171 : tensor<128x32xf32, #linear7> -> tensor<128x2x16xf32, #blocked5>
        %dqs_177 = tt.trans %dqs_176 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked5> -> tensor<128x16x2xf32, #blocked6>
        %dqs_178, %dqs_179 = tt.split %dqs_177 : tensor<128x16x2xf32, #blocked6> -> tensor<128x16xf32, #blocked>
        %dq_row_180 = arith.addi %qt_100, %dq_row_7 : i64
        %dq_row_181 = arith.muli %dq_row_180, %c2_i64 : i64
        %dqN = arith.mulf %dqs_174, %cst_0 : tensor<128x16xf32, #blocked>
        %19 = arith.trunci %dq_row_181 : i64 to i32
        tt.descriptor_reduce add, %desc_dq[%19, %c0_i32], %dqN : !tt.tensordesc<128x16xf32>, tensor<128x16xf32, #blocked>
        %dqN_182 = arith.mulf %dqs_175, %cst_0 : tensor<128x16xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%19, %c16_i32], %dqN_182 : !tt.tensordesc<128x16xf32>, tensor<128x16xf32, #blocked>
        %dqN_183 = arith.mulf %dqs_178, %cst_0 : tensor<128x16xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%19, %c32_i32], %dqN_183 : !tt.tensordesc<128x16xf32>, tensor<128x16xf32, #blocked>
        %dqN_184 = arith.mulf %dqs_179, %cst_0 : tensor<128x16xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%19, %c48_i32], %dqN_184 : !tt.tensordesc<128x16xf32>, tensor<128x16xf32, #blocked>
        %curr_m_185 = arith.addi %curr_m_99, %c128_i32 : i32
        scf.yield %dk_160, %dv_149, %curr_m_185 : tensor<128x128xf32, #linear1>, tensor<128x128xf32, #linear1>, i32
      }
      %dvs = tt.reshape %curr_m#1 : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear8>
      %dvs_34 = tt.trans %dvs {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear8> -> tensor<128x64x2xf32, #linear9>
      %dvs_35 = ttg.convert_layout %dvs_34 : tensor<128x64x2xf32, #linear9> -> tensor<128x64x2xf32, #linear10>
      %dvs_36, %dvs_37 = tt.split %dvs_35 : tensor<128x64x2xf32, #linear10> -> tensor<128x64xf32, #linear11>
      %dvs_38 = tt.reshape %dvs_36 : tensor<128x64xf32, #linear11> -> tensor<128x2x32xf32, #linear12>
      %dvs_39 = tt.trans %dvs_38 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear12> -> tensor<128x32x2xf32, #linear13>
      %dvs_40, %dvs_41 = tt.split %dvs_39 : tensor<128x32x2xf32, #linear13> -> tensor<128x32xf32, #linear14>
      %dvs_42 = tt.reshape %dvs_40 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked7>
      %dvs_43 = tt.trans %dvs_42 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked7> -> tensor<128x16x2xf32, #blocked8>
      %dvs_44, %dvs_45 = tt.split %dvs_43 : tensor<128x16x2xf32, #blocked8> -> tensor<128x16xf32, #blocked1>
      %dvs_46 = tt.reshape %dvs_41 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked7>
      %dvs_47 = tt.trans %dvs_46 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked7> -> tensor<128x16x2xf32, #blocked8>
      %dvs_48, %dvs_49 = tt.split %dvs_47 : tensor<128x16x2xf32, #blocked8> -> tensor<128x16xf32, #blocked1>
      %dvs_50 = tt.reshape %dvs_37 : tensor<128x64xf32, #linear11> -> tensor<128x2x32xf32, #linear12>
      %dvs_51 = tt.trans %dvs_50 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear12> -> tensor<128x32x2xf32, #linear13>
      %dvs_52, %dvs_53 = tt.split %dvs_51 : tensor<128x32x2xf32, #linear13> -> tensor<128x32xf32, #linear14>
      %dvs_54 = tt.reshape %dvs_52 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked7>
      %dvs_55 = tt.trans %dvs_54 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked7> -> tensor<128x16x2xf32, #blocked8>
      %dvs_56, %dvs_57 = tt.split %dvs_55 : tensor<128x16x2xf32, #blocked8> -> tensor<128x16xf32, #blocked1>
      %dvs_58 = tt.reshape %dvs_53 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked7>
      %dvs_59 = tt.trans %dvs_58 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked7> -> tensor<128x16x2xf32, #blocked8>
      %dvs_60, %dvs_61 = tt.split %dvs_59 : tensor<128x16x2xf32, #blocked8> -> tensor<128x16xf32, #blocked1>
      %3 = arith.truncf %dvs_44 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dv[%k_19, %c0_i32], %3 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %4 = arith.truncf %dvs_45 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dv[%k_19, %c16_i32], %4 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %5 = arith.truncf %dvs_48 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dv[%k_19, %c32_i32], %5 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %6 = arith.truncf %dvs_49 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dv[%k_19, %c48_i32], %6 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %7 = arith.truncf %dvs_56 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dv[%k_19, %c64_i32], %7 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %8 = arith.truncf %dvs_57 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dv[%k_19, %c80_i32], %8 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %9 = arith.truncf %dvs_60 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dv[%k_19, %c96_i32], %9 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %10 = arith.truncf %dvs_61 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dv[%k_19, %c112_i32], %10 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %dks = tt.reshape %curr_m#0 : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear8>
      %dks_62 = tt.trans %dks {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #linear8> -> tensor<128x64x2xf32, #linear9>
      %dks_63 = ttg.convert_layout %dks_62 : tensor<128x64x2xf32, #linear9> -> tensor<128x64x2xf32, #linear10>
      %dks_64, %dks_65 = tt.split %dks_63 : tensor<128x64x2xf32, #linear10> -> tensor<128x64xf32, #linear11>
      %dks_66 = tt.reshape %dks_64 : tensor<128x64xf32, #linear11> -> tensor<128x2x32xf32, #linear12>
      %dks_67 = tt.trans %dks_66 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear12> -> tensor<128x32x2xf32, #linear13>
      %dks_68, %dks_69 = tt.split %dks_67 : tensor<128x32x2xf32, #linear13> -> tensor<128x32xf32, #linear14>
      %dks_70 = tt.reshape %dks_68 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked7>
      %dks_71 = tt.trans %dks_70 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked7> -> tensor<128x16x2xf32, #blocked8>
      %dks_72, %dks_73 = tt.split %dks_71 : tensor<128x16x2xf32, #blocked8> -> tensor<128x16xf32, #blocked1>
      %dks_74 = tt.reshape %dks_69 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked7>
      %dks_75 = tt.trans %dks_74 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked7> -> tensor<128x16x2xf32, #blocked8>
      %dks_76, %dks_77 = tt.split %dks_75 : tensor<128x16x2xf32, #blocked8> -> tensor<128x16xf32, #blocked1>
      %dks_78 = tt.reshape %dks_65 : tensor<128x64xf32, #linear11> -> tensor<128x2x32xf32, #linear12>
      %dks_79 = tt.trans %dks_78 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear12> -> tensor<128x32x2xf32, #linear13>
      %dks_80, %dks_81 = tt.split %dks_79 : tensor<128x32x2xf32, #linear13> -> tensor<128x32xf32, #linear14>
      %dks_82 = tt.reshape %dks_80 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked7>
      %dks_83 = tt.trans %dks_82 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked7> -> tensor<128x16x2xf32, #blocked8>
      %dks_84, %dks_85 = tt.split %dks_83 : tensor<128x16x2xf32, #blocked8> -> tensor<128x16xf32, #blocked1>
      %dks_86 = tt.reshape %dks_81 : tensor<128x32xf32, #linear14> -> tensor<128x2x16xf32, #blocked7>
      %dks_87 = tt.trans %dks_86 {order = array<i32: 0, 2, 1>} : tensor<128x2x16xf32, #blocked7> -> tensor<128x16x2xf32, #blocked8>
      %dks_88, %dks_89 = tt.split %dks_87 : tensor<128x16x2xf32, #blocked8> -> tensor<128x16xf32, #blocked1>
      %dkN_90 = arith.mulf %dks_72, %dkN : tensor<128x16xf32, #blocked1>
      %11 = arith.truncf %dkN_90 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dk[%k_19, %c0_i32], %11 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %dkN_91 = arith.mulf %dks_73, %dkN : tensor<128x16xf32, #blocked1>
      %12 = arith.truncf %dkN_91 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dk[%k_19, %c16_i32], %12 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %dkN_92 = arith.mulf %dks_76, %dkN : tensor<128x16xf32, #blocked1>
      %13 = arith.truncf %dkN_92 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dk[%k_19, %c32_i32], %13 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %dkN_93 = arith.mulf %dks_77, %dkN : tensor<128x16xf32, #blocked1>
      %14 = arith.truncf %dkN_93 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dk[%k_19, %c48_i32], %14 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %dkN_94 = arith.mulf %dks_84, %dkN : tensor<128x16xf32, #blocked1>
      %15 = arith.truncf %dkN_94 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dk[%k_19, %c64_i32], %15 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %dkN_95 = arith.mulf %dks_85, %dkN : tensor<128x16xf32, #blocked1>
      %16 = arith.truncf %dkN_95 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dk[%k_19, %c80_i32], %16 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %dkN_96 = arith.mulf %dks_88, %dkN : tensor<128x16xf32, #blocked1>
      %17 = arith.truncf %dkN_96 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dk[%k_19, %c96_i32], %17 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %dkN_97 = arith.mulf %dks_89, %dkN : tensor<128x16xf32, #blocked1>
      %18 = arith.truncf %dkN_97 : tensor<128x16xf32, #blocked1> to tensor<128x16xf16, #blocked1>
      tt.descriptor_store %desc_dk[%k_19, %c112_i32], %18 : !tt.tensordesc<128x16xf16>, tensor<128x16xf16, #blocked1>
      %tile_idx_98 = arith.addi %tile_idx_8, %num_progs_5 : i32
      scf.yield %tile_idx_98 : i32
    } {tt.merge_epilogue_to_computation = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 180000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
