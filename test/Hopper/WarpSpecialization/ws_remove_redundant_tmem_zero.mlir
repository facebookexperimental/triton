// RUN: triton-opt %s --nvgpu-warp-specialization="capability=100" --mlir-print-debuginfo --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s

// Test: Redundant TMEM zeroing removal for operand D (BWD persistent FA, BLOCK_M=64).
//
// This IR is captured from b64/buffer_creation.prior — the actual BWD
// persistent FA kernel just before NVGPUWarpSpecialization.
// The removeRedundantTmemZeroStores pass should remove the tmem_store
// of dense<0.0> for dk/dv since the MMA's useD=false handles zeroing.
//
// CHECK-LABEL: tt.func public @_attn_bwd_persist
// The tmem_store of zeros for dk/dv should be removed:
// CHECK-NOT: ttng.tmem_store %cst

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1, 64], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 64, 1], threadsPerWarp = [16, 1, 2], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd_persist(%desc_q: !tt.tensordesc<tensor<64x128xf16, #shared>>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %sm_scale: f32, %desc_do: !tt.tensordesc<tensor<64x128xf16, #shared>>, %desc_do_12: i32, %desc_do_13: i32, %desc_do_14: i64, %desc_do_15: i64, %desc_dq: !tt.tensordesc<tensor<64x64xf32, #shared1>>, %desc_dq_16: i32, %desc_dq_17: i32, %desc_dq_18: i64, %desc_dq_19: i64, %desc_dk: !tt.tensordesc<tensor<128x64xf16, #shared>>, %desc_dk_20: i32, %desc_dk_21: i32, %desc_dk_22: i64, %desc_dk_23: i64, %desc_dv: !tt.tensordesc<tensor<128x64xf16, #shared>>, %desc_dv_24: i32, %desc_dv_25: i32, %desc_dv_26: i64, %desc_dv_27: i64, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %D: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32 {tt.divisibility = 16 : i32}, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<0.693147182> : tensor<64x64xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %n_tile_num = arith.constant 127 : i32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %cst_28 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %n_tile_num_29 = arith.addi %N_CTX, %n_tile_num : i32
    %n_tile_num_30 = arith.divsi %n_tile_num_29, %c128_i32 : i32
    %prog_id = tt.get_program_id x : i32
    %num_progs = tt.get_num_programs x : i32
    %total_tiles = arith.muli %n_tile_num_30, %BATCH : i32
    %total_tiles_31 = arith.muli %total_tiles, %H : i32
    %tiles_per_sm = arith.divsi %total_tiles_31, %num_progs : i32
    %0 = arith.remsi %total_tiles_31, %num_progs : i32
    %1 = arith.cmpi slt, %prog_id, %0 : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_32 = arith.addi %tiles_per_sm, %c1_i32 : i32
      scf.yield %tiles_per_sm_32 : i32
    } else {
      scf.yield %tiles_per_sm : i32
    }
    %off_bh = arith.extsi %stride_tok : i32 to i64
    %num_steps = arith.divsi %N_CTX, %c64_i32 : i32
    %offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %dkN = tt.splat %sm_scale : f32 -> tensor<128x64xf32, #blocked2>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_32 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_32, %n_tile_num_30 : i32
      %bhid = arith.divsi %tile_idx_32, %n_tile_num_30 {ttg.partition = array<i32: 0>} : i32
      %off_chz = arith.muli %bhid, %N_CTX {ttg.partition = array<i32: 3>} : i32
      %off_chz_33 = arith.extsi %off_chz {ttg.partition = array<i32: 3>} : i32 to i64
      %off_bh_34 = arith.remsi %bhid, %H {ttg.partition = array<i32: 0>} : i32
      %off_bh_35 = arith.muli %stride_h, %off_bh_34 {ttg.partition = array<i32: 0>} : i32
      %off_bh_36 = arith.divsi %bhid, %H {ttg.partition = array<i32: 0>} : i32
      %off_bh_37 = arith.muli %stride_z, %off_bh_36 {ttg.partition = array<i32: 0>} : i32
      %off_bh_38 = arith.addi %off_bh_35, %off_bh_37 {ttg.partition = array<i32: 0>} : i32
      %off_bh_39 = arith.extsi %off_bh_38 {ttg.partition = array<i32: 0>} : i32 to i64
      %off_bh_40 = arith.divsi %off_bh_39, %off_bh {ttg.partition = array<i32: 0>} : i64
      %M_41 = tt.addptr %M, %off_chz_33 {ttg.partition = array<i32: 3>} : !tt.ptr<f32>, i64
      %D_42 = tt.addptr %D, %off_chz_33 {ttg.partition = array<i32: 3>} : !tt.ptr<f32>, i64
      %start_n = arith.muli %pid, %c128_i32 : i32
      %k = arith.extsi %start_n : i32 to i64
      %k_43 = arith.addi %off_bh_40, %k {ttg.partition = array<i32: 3>} : i64
      %k_44 = arith.trunci %k_43 {ttg.partition = array<i32: 3>} : i64 to i32
      %k_45 = tt.descriptor_load %desc_k[%k_44, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
      %k_46 = ttg.local_alloc %k_45 {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked3>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %v = tt.descriptor_load %desc_v[%k_44, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
      %v_47 = ttg.local_alloc %v {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked3>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %m = tt.splat %M_41 {ttg.partition = array<i32: 3>} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %Di = tt.splat %D_42 {ttg.partition = array<i32: 3>} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %qkT, %qkT_48 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dpT, %dpT_49 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dv, %dv_50 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dq, %dq_51 = ttng.tmem_alloc {ttg.partition = array<i32: 0>} : () -> (!ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dk, %dk_52 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dk_53 = ttng.tmem_store %cst_28, %dk[%dk_52], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      %dv_54 = ttng.tmem_store %cst_28, %dv[%dv_50], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      %curr_m:7 = scf.for %curr_m_68 = %c0_i32 to %num_steps step %c1_i32 iter_args(%arg47 = %c0_i32, %arg48 = %false, %qkT_69 = %qkT_48, %dpT_70 = %dpT_49, %dv_71 = %dv_54, %dq_72 = %dq_51, %dk_73 = %dk_53) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %q = arith.extsi %arg47 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32 to i64
        %q_74 = arith.addi %off_bh_40, %q {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i64
        %q_75 = arith.trunci %q_74 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i64 to i32
        %q_76 = tt.descriptor_load %desc_q[%q_75, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked3>
        %q_77 = ttg.local_alloc %q_76 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %qT = ttg.memdesc_trans %q_77 {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared2, #smem>
        %offs_m_78 = tt.splat %arg47 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %offs_m_79 = arith.addi %offs_m_78, %offs_m {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %m_80 = tt.addptr %m, %offs_m_79 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked2}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %m_81 = tt.load %m_80 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %qkT_82 = ttng.tc_gen5_mma %k_46, %qT, %qkT[%qkT_69], %false, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared2, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %pT = tt.expand_dims %m_81 {axis = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xf32, #blocked2>
        %pT_83 = tt.broadcast %pT {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<1x64xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
        %qkT_84, %qkT_85 = ttng.tmem_load %qkT[%qkT_82] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %pT_86 = arith.subf %qkT_84, %pT_83 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2>
        %pT_87 = math.exp2 %pT_86 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2>
        %do = tt.descriptor_load %desc_do[%q_75, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked3>
        %do_88 = ttg.local_alloc %do {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %ppT = arith.truncf %pT_87 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
        %dv_89 = ttng.tmem_alloc %ppT {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>
        %dpT_90 = ttg.memdesc_trans %do_88 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared2, #smem>
        %dpT_91 = ttng.tc_gen5_mma %v_47, %dpT_90, %dpT[%dpT_70], %false, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared2, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %Di_92 = tt.addptr %Di, %offs_m_79 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked2}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %Di_93 = tt.load %Di_92 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %dv_94 = ttng.tc_gen5_mma %dv_89, %do_88, %dv[%dv_71], %arg48, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %dsT = tt.expand_dims %Di_93 {axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xf32, #blocked2>
        %dsT_95 = tt.broadcast %dsT {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : tensor<1x64xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
        %dpT_96, %dpT_97 = ttng.tmem_load %dpT[%dpT_91] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %dsT_98 = arith.subf %dpT_96, %dsT_95 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2>
        %dsT_99 = arith.mulf %pT_87, %dsT_98 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2>
        %dsT_100 = arith.truncf %dsT_99 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
        %dsT_101 = ttg.local_alloc %dsT_100 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %dq_102 = ttg.memdesc_trans %dsT_101 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared2, #smem>
        %dq_103 = ttng.tc_gen5_mma %dq_102, %k_46, %dq[%dq_72], %false, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,11\22]}", ttg.partition = array<i32: 1>} : !ttg.memdesc<64x128xf16, #shared2, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>
        %dk_104 = ttng.tc_gen5_mma %dsT_101, %q_77, %dk[%dk_73], %arg48, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndD,tmem,1,10\22]}", tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %dq_105, %dq_106 = ttng.tmem_load %dq[%dq_103] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked4>
        %dqs = tt.reshape %dq_105 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<64x128xf32, #blocked4> -> tensor<64x2x64xf32, #blocked5>
        %dqs_107 = tt.trans %dqs {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<64x2x64xf32, #blocked5> -> tensor<64x64x2xf32, #blocked6>
        %dqs_108 = ttg.convert_layout %dqs_107 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<64x64x2xf32, #blocked6> -> tensor<64x64x2xf32, #blocked7>
        %dqs_109, %dqs_110 = tt.split %dqs_108 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<64x64x2xf32, #blocked7> -> tensor<64x64xf32, #blocked>
        %dqN = arith.mulf %dqs_109, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%q_75, %c0_i32], %dqN {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<64x64xf32, #shared1>>, tensor<64x64xf32, #blocked>
        %dqN_111 = arith.mulf %dqs_110, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<64x64xf32, #blocked>
        tt.descriptor_reduce add, %desc_dq[%q_75, %c64_i32], %dqN_111 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<64x64xf32, #shared1>>, tensor<64x64xf32, #blocked>
        %curr_m_112 = arith.addi %arg47, %c64_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : i32
        scf.yield %curr_m_112, %true, %qkT_85, %dpT_97, %dv_94, %dq_106, %dk_104 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 3>}
      %dv_55, %dv_56 = ttng.tmem_load %dv[%curr_m#4] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %dvs = tt.reshape %dv_55 {ttg.partition = array<i32: 3>} : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked8>
      %dvs_57 = tt.trans %dvs {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 3>} : tensor<128x2x64xf32, #blocked8> -> tensor<128x64x2xf32, #blocked9>
      %dvs_58, %dvs_59 = tt.split %dvs_57 {ttg.partition = array<i32: 3>} : tensor<128x64x2xf32, #blocked9> -> tensor<128x64xf32, #blocked2>
      %3 = arith.truncf %dvs_58 {ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
      %4 = ttg.convert_layout %3 {ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked2> -> tensor<128x64xf16, #blocked10>
      tt.descriptor_store %desc_dv[%k_44, %c0_i32], %4 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked10>
      %5 = arith.truncf %dvs_59 {ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
      %6 = ttg.convert_layout %5 {ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked2> -> tensor<128x64xf16, #blocked10>
      tt.descriptor_store %desc_dv[%k_44, %c64_i32], %6 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked10>
      %dk_60, %dk_61 = ttng.tmem_load %dk[%curr_m#6] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %dks = tt.reshape %dk_60 {ttg.partition = array<i32: 3>} : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked8>
      %dks_62 = tt.trans %dks {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 3>} : tensor<128x2x64xf32, #blocked8> -> tensor<128x64x2xf32, #blocked9>
      %dks_63, %dks_64 = tt.split %dks_62 {ttg.partition = array<i32: 3>} : tensor<128x64x2xf32, #blocked9> -> tensor<128x64xf32, #blocked2>
      %dkN_65 = arith.mulf %dks_63, %dkN {ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2>
      %7 = arith.truncf %dkN_65 {ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
      %8 = ttg.convert_layout %7 {ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked2> -> tensor<128x64xf16, #blocked10>
      tt.descriptor_store %desc_dk[%k_44, %c0_i32], %8 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked10>
      %dkN_66 = arith.mulf %dks_64, %dkN {ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2>
      %9 = arith.truncf %dkN_66 {ttg.partition = array<i32: 3>} : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
      %10 = ttg.convert_layout %9 {ttg.partition = array<i32: 3>} : tensor<128x64xf16, #blocked2> -> tensor<128x64xf16, #blocked10>
      tt.descriptor_store %desc_dk[%k_44, %c64_i32], %10 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked10>
      %tile_idx_67 = arith.addi %tile_idx_32, %num_progs : i32
      scf.yield %tile_idx_67 : i32
    } {tt.merge_epilogue = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["reduction", "gemm", "load", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
