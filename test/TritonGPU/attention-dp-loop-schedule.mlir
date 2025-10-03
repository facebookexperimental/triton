// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-schedule-loops | FileCheck %s


#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>

// Note: There is no cluster 3 in the generated IR. This is fine as the relative
// ordering is all that matters for the IR.

// CHECK: tt.descriptor_load %{{.*}} {loop.cluster = 6 : i32, loop.stage = 0 : i32}
// CHECK: tt.descriptor_load %{{.*}} {loop.cluster = 6 : i32, loop.stage = 0 : i32}
// CHECK: ttng.tc_gen5_mma %{{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32}
// CHECK: ttng.tc_gen5_mma %{{.*}} {loop.cluster = 4 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32}
// CHECK: ttng.tc_gen5_mma %{{.*}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32}
// CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABLE: @_dp_attn_peristent
  tt.func public @_dp_attn_peristent(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %desc_o: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %desc_o_12: i32, %desc_o_13: i32, %desc_o_14: i64, %desc_o_15: i64, %N_CTX: i32) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %n_tile_num = arith.constant 255 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %cst_16 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_17 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_18 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %n_tile_num_19 = arith.addi %N_CTX, %n_tile_num : i32
    %n_tile_num_20 = arith.divsi %n_tile_num_19, %c256_i32 : i32
    %prog_id = tt.get_program_id x : i32
    %num_progs = tt.get_num_programs x : i32
    %total_tiles = arith.muli %n_tile_num_20, %Z : i32
    %total_tiles_21 = arith.muli %total_tiles, %H : i32
    %tiles_per_sm = arith.divsi %total_tiles_21, %num_progs : i32
    %0 = arith.remsi %total_tiles_21, %num_progs : i32
    %1 = arith.cmpi slt, %prog_id, %0 : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_22 = arith.addi %tiles_per_sm, %c1_i32 : i32
      scf.yield %tiles_per_sm_22 : i32
    } else {
      scf.yield %tiles_per_sm : i32
    }
    %offset_y = arith.muli %N_CTX, %H : i32
    %offs_m0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %offs_m1 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked1>
    %qk_scale = arith.mulf %sm_scale, %cst : f32
    %m_ij = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %qk = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #blocked>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_22 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_22, %n_tile_num_20 : i32
      %off_hz = arith.divsi %tile_idx_22, %n_tile_num_20 : i32
      %off_z = arith.divsi %off_hz, %H : i32
      %off_h = arith.remsi %off_hz, %H : i32
      %offset_y_23 = arith.muli %off_z, %offset_y : i32
      %offset_y_24 = arith.muli %off_h, %N_CTX : i32
      %offset_y_25 = arith.addi %offset_y_23, %offset_y_24 : i32
      %qo_offset_y = arith.muli %pid, %c256_i32 : i32
      %qo_offset_y_26 = arith.addi %offset_y_25, %qo_offset_y : i32
      %offs_m0_27 = tt.splat %qo_offset_y : i32 -> tensor<128xi32, #blocked1>
      %offs_m0_28 = arith.addi %offs_m0_27, %offs_m0 : tensor<128xi32, #blocked1>
      %offs_m1_29 = arith.addi %offs_m0_27, %offs_m1 : tensor<128xi32, #blocked1>
      %q0 = tt.descriptor_load %desc_q[%qo_offset_y_26, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
      %q0_30 = ttg.local_alloc %q0 : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %q1 = arith.addi %qo_offset_y_26, %c128_i32 : i32
      %q1_31 = tt.descriptor_load %desc_q[%q1, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
      %q1_32 = ttg.local_alloc %q1_31 : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %qk_33, %qk_34 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc, %acc_35 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %qk_36, %qk_37 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_38, %acc_39 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_40 = ttng.tmem_store %cst_16, %acc_38[%acc_39], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_41 = ttng.tmem_store %cst_16, %acc[%acc_35], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %offsetkv_y:10 = scf.for %offsetkv_y_56 = %c0_i32 to %N_CTX step %c128_i32 iter_args(%arg28 = %cst_18, %arg29 = %cst_18, %arg30 = %cst_17, %arg31 = %cst_17, %offset_y_57 = %offset_y_25, %arg33 = %false, %qk_58 = %qk_34, %acc_59 = %acc_41, %qk_60 = %qk_37, %acc_61 = %acc_40) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %k = tt.descriptor_load %desc_k[%offset_y_57, %c0_i32] {tt.latency = 2 : i32} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
        %k_62 = ttg.local_alloc %k : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %k_63 = ttg.memdesc_trans %k_62 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
        %v = tt.descriptor_load %desc_v[%offset_y_57, %c0_i32] {tt.latency = 1 : i32} : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked2>
        %v_64 = ttg.local_alloc %v : (tensor<128x128xbf16, #blocked2>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %qk_65 = ttng.tc_gen5_mma %q0_30, %k_63, %qk_33[%qk_58], %false, %true {tt.latency = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_66, %qk_67 = ttng.tmem_load %qk_33[%qk_65] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %m_ij_68 = "tt.reduce"(%qk_66) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_124: f32, %m_ij_125: f32):
          %m_ij_126 = arith.maxnumf %m_ij_124, %m_ij_125 : f32
          tt.reduce.return %m_ij_126 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_69 = arith.mulf %m_ij_68, %m_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_70 = arith.maxnumf %arg30, %m_ij_69 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %qk_71 = arith.mulf %qk_66, %qk : tensor<128x128xf32, #blocked>
        %qk_72 = tt.expand_dims %m_ij_70 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %qk_73 = tt.broadcast %qk_72 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qk_74 = arith.subf %qk_71, %qk_73 : tensor<128x128xf32, #blocked>
        %p = math.exp2 %qk_74 : tensor<128x128xf32, #blocked>
        %alpha = arith.subf %arg30, %m_ij_70 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_75 = math.exp2 %alpha : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_124: f32, %l_ij_125: f32):
          %l_ij_126 = arith.addf %l_ij_124, %l_ij_125 : f32
          tt.reduce.return %l_ij_126 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc_76, %acc_77 = ttng.tmem_load %acc[%acc_59] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %9 = tt.reshape %acc_76 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
        %10 = tt.trans %9 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
        %outLHS, %outRHS = tt.split %10 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5>
        %acc0_78 = tt.expand_dims %alpha_75 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc0_79 = ttg.convert_layout %acc0_78 : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked5>
        %acc0_80 = tt.broadcast %acc0_79 : tensor<128x1xf32, #blocked5> -> tensor<128x64xf32, #blocked5>
        %acc0_81 = arith.mulf %outLHS, %acc0_80 : tensor<128x64xf32, #blocked5>
        %acc1_82 = arith.mulf %outRHS, %acc0_80 : tensor<128x64xf32, #blocked5>
        %acc_83 = tt.join %acc0_81, %acc1_82 : tensor<128x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked4>
        %acc_84 = tt.trans %acc_83 {order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x2x64xf32, #blocked3>
        %acc_85 = tt.reshape %acc_84 : tensor<128x2x64xf32, #blocked3> -> tensor<128x128xf32, #blocked>
        %p_86 = arith.truncf %p : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %acc_87 = ttng.tmem_alloc %p_86 : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>
        %acc_88 = ttng.tmem_store %acc_85, %acc[%acc_77], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_89 = ttng.tc_gen5_mma %acc_87, %v_64, %acc[%acc_88], %arg33, %true {tt.latency = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_i = arith.mulf %arg28, %alpha_75 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i_90 = arith.addf %l_i, %l_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %qk_91 = ttng.tc_gen5_mma %q1_32, %k_63, %qk_36[%qk_60], %false, %true {tt.latency = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_92, %qk_93 = ttng.tmem_load %qk_36[%qk_91] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %m_ij_94 = "tt.reduce"(%qk_92) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_124: f32, %m_ij_125: f32):
            %m_ij_126 = arith.maxnumf %m_ij_124, %m_ij_125 : f32
            tt.reduce.return %m_ij_126 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_95 = arith.mulf %m_ij_94, %m_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_ij_96 = arith.maxnumf %arg31, %m_ij_95 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %qk_97 = arith.mulf %qk_92, %qk : tensor<128x128xf32, #blocked>
        %qk_98 = tt.expand_dims %m_ij_96 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %qk_99 = tt.broadcast %qk_98 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qk_100 = arith.subf %qk_97, %qk_99 : tensor<128x128xf32, #blocked>
        %p_101 = math.exp2 %qk_100 : tensor<128x128xf32, #blocked>
        %alpha_102 = arith.subf %arg31, %m_ij_96 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %alpha_103 = math.exp2 %alpha_102 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_ij_104 = "tt.reduce"(%p_101) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_124: f32, %l_ij_125: f32):
            %l_ij_126 = arith.addf %l_ij_124, %l_ij_125 : f32
            tt.reduce.return %l_ij_126 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc_105, %acc_106 = ttng.tmem_load %acc_38[%acc_61] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %11 = tt.reshape %acc_105 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
        %12 = tt.trans %11 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
        %outLHS_107, %outRHS_108 = tt.split %12 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5>
        %acc0_109 = tt.expand_dims %alpha_103 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc0_110 = ttg.convert_layout %acc0_109 : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked5>
        %acc0_111 = tt.broadcast %acc0_110 : tensor<128x1xf32, #blocked5> -> tensor<128x64xf32, #blocked5>
        %acc0_112 = arith.mulf %outLHS_107, %acc0_111 : tensor<128x64xf32, #blocked5>
        %acc1_113 = arith.mulf %outRHS_108, %acc0_111 : tensor<128x64xf32, #blocked5>
        %acc_114 = tt.join %acc0_112, %acc1_113 : tensor<128x64xf32, #blocked5> -> tensor<128x64x2xf32, #blocked4>
        %acc_115 = tt.trans %acc_114 {order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked4> -> tensor<128x2x64xf32, #blocked3>
        %acc_116 = tt.reshape %acc_115 : tensor<128x2x64xf32, #blocked3> -> tensor<128x128xf32, #blocked>
        %p_117 = arith.truncf %p_101 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %acc_118 = ttng.tmem_alloc %p_117 : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>
        %acc_119 = ttng.tmem_store %acc_116, %acc_38[%acc_106], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_120 = ttng.tc_gen5_mma %acc_118, %v_64, %acc_38[%acc_119], %arg33, %true {tt.latency = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_i_121 = arith.mulf %arg29, %alpha_103 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %l_i_122 = arith.addf %l_i_121, %l_ij_104 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %offsetkv_y_123 = arith.addi %offset_y_57, %c128_i32 : i32
        scf.yield %l_i_90, %l_i_122, %m_ij_70, %m_ij_96, %offsetkv_y_123, %true, %qk_67, %acc_89, %qk_93, %acc_120 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
        } {tt.disallow_acc_multi_buffer}
        %m_i0 = math.log2 %offsetkv_y#0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_i0_42 = arith.addf %offsetkv_y#2, %m_i0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc0 = tt.expand_dims %offsetkv_y#0 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc0_43 = tt.broadcast %acc0 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %acc_44, %acc_45 = ttng.tmem_load %acc[%offsetkv_y#7] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %acc0_46 = arith.divf %acc_44, %acc0_43 : tensor<128x128xf32, #blocked>
        %m_ptrs0 = arith.muli %off_hz, %N_CTX : i32
        %m_ptrs0_47 = tt.addptr %M, %m_ptrs0 : !tt.ptr<f32>, i32
        %m_ptrs0_48 = tt.splat %m_ptrs0_47 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
        %m_ptrs0_49 = tt.addptr %m_ptrs0_48, %offs_m0_28 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
        %3 = ttg.convert_layout %m_i0_42 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
        tt.store %m_ptrs0_49, %3 : tensor<128x!tt.ptr<f32>, #blocked1>
        %4 = arith.truncf %acc0_46 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %5 = ttg.convert_layout %4 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked2>
        tt.descriptor_store %desc_o[%qo_offset_y_26, %c0_i32], %5 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, tensor<128x128xbf16, #blocked2>
        %m_i1 = math.log2 %offsetkv_y#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %m_i1_50 = arith.addf %offsetkv_y#3, %m_i1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %acc1 = tt.expand_dims %offsetkv_y#1 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %acc1_51 = tt.broadcast %acc1 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %acc_52, %acc_53 = ttng.tmem_load %acc_38[%offsetkv_y#9] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %acc1_54 = arith.divf %acc_52, %acc1_51 : tensor<128x128xf32, #blocked>
        %m_ptrs1 = tt.addptr %m_ptrs0_48, %offs_m1_29 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
        %6 = ttg.convert_layout %m_i1_50 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
        tt.store %m_ptrs1, %6 : tensor<128x!tt.ptr<f32>, #blocked1>
        %7 = arith.truncf %acc1_54 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
        %8 = ttg.convert_layout %7 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked2>
        tt.descriptor_store %desc_o[%q1, %c0_i32], %8 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, tensor<128x128xbf16, #blocked2>
        %tile_idx_55 = arith.addi %tile_idx_22, %num_progs : i32
        scf.yield %tile_idx_55 : i32
      } {tt.warp_specialize}
    tt.return
  }
}
