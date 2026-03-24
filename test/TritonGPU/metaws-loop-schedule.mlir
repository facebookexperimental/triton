// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-schedule-loops=use-meta-ws=true | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 168 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LAEBL: @_attn_fwd
  tt.func public @_attn_fwd(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<tensor<64x128xf16, #shared>>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<tensor<64x128xf16, #shared>>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %desc_o: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_o_12: i32, %desc_o_13: i32, %desc_o_14: i64, %desc_o_15: i64, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %l_i = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %m_i = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant 1.44269502 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_16 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %cst_17 = arith.constant dense<-1.000000e+06> : tensor<128x64xf32, #blocked>
    %start_m = tt.get_program_id x : i32
    %off_hz = tt.get_program_id y : i32
    %off_z = arith.divsi %off_hz, %H : i32
    %off_h = arith.remsi %off_hz, %H : i32
    %offset_y = arith.muli %N_CTX, %H : i32
    %offset_y_18 = arith.muli %off_z, %offset_y : i32
    %offset_y_19 = arith.muli %off_h, %N_CTX : i32
    %offset_y_20 = arith.addi %offset_y_18, %offset_y_19 : i32
    %qo_offset_y = arith.muli %start_m, %c128_i32 : i32
    %qo_offset_y_21 = arith.addi %offset_y_20, %qo_offset_y : i32
    %offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %offs_m_23 = tt.splat %qo_offset_y : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_24 = tt.splat %qo_offset_y : i32 -> tensor<128xi32, #blocked2>
    %offs_m_25 = arith.addi %offs_m_23, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_26 = arith.addi %offs_m_24, %offs_m_22 : tensor<128xi32, #blocked2>
    %qk_scale = arith.mulf %sm_scale, %cst : f32
    %q = tt.descriptor_load %desc_q[%qo_offset_y_21, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
    %q_27 = ttg.local_alloc %q : (tensor<128x128xf16, #blocked3>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %m_ij = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %qk = tt.splat %qk_scale : f32 -> tensor<128x64xf32, #blocked>
    %qk_28, %qk_29 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_30, %acc_31 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_32 = ttng.tmem_store %acc, %acc_30[%acc_31], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: scf.for {{.*}}
    %offsetv_y:6 = scf.for %offsetv_y_56 = %c0_i32 to %qo_offset_y step %c64_i32 iter_args(%l_i_57 = %l_i, %m_i_58 = %m_i, %offset_y_59 = %offset_y_20, %arg29 = %false, %qk_60 = %qk_29, %acc_61 = %acc_32) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i1, !ttg.async.token, !ttg.async.token)  : i32 {
      // CHECK: tt.descriptor_load {{.*}} {loop.cluster = [[CLUSTER1:[0-9]+]] : i32, loop.stage = 0 : i32} {{.*}}
      %k = tt.descriptor_load %desc_k[%offset_y_59, %c0_i32] {tt.latency = 1 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked3>
      %k_62 = ttg.local_alloc %k : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %k_63 = ttg.memdesc_trans %k_62 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared1, #smem>
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = [[CLUSTER1]] : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} {{.*}}
      %qk_64 = ttng.tc_gen5_mma %q_27, %k_63, %qk_28[%qk_60], %false, %true {tt.latency = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared1, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %qk_65, %qk_66 = ttng.tmem_load %qk_28[%qk_64] : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
      %m_ij_67 = "tt.reduce"(%qk_65) <{axis = 1 : i32}> ({
      ^bb0(%m_ij_90: f32, %m_ij_91: f32):
        %m_ij_92 = arith.maxnumf %m_ij_90, %m_ij_91 : f32
        tt.reduce.return %m_ij_92 : f32
      }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_ij_68 = arith.mulf %m_ij_67, %m_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_ij_69 = arith.maxnumf %m_i_58, %m_ij_68 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %qk_70 = arith.mulf %qk_65, %qk : tensor<128x64xf32, #blocked>
      %qk_71 = tt.expand_dims %m_ij_69 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %qk_72 = tt.broadcast %qk_71 : tensor<128x1xf32, #blocked> -> tensor<128x64xf32, #blocked>
      %qk_73 = arith.subf %qk_70, %qk_72 : tensor<128x64xf32, #blocked>
      %p = math.exp2 %qk_73 : tensor<128x64xf32, #blocked>
      %alpha = arith.subf %m_i_58, %m_ij_69 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_74 = math.exp2 %alpha : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
      ^bb0(%l_ij_90: f32, %l_ij_91: f32):
        %l_ij_92 = arith.addf %l_ij_90, %l_ij_91 : f32
        tt.reduce.return %l_ij_92 : f32
      }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %acc_75, %acc_76 = ttng.tmem_load %acc_30[%acc_61] : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %6 = tt.reshape %acc_75 : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked4>
      %7 = tt.trans %6 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %outLHS, %outRHS = tt.split %7 : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked>
      %acc0 = tt.expand_dims %alpha_74 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %acc0_77 = tt.broadcast %acc0 : tensor<128x1xf32, #blocked> -> tensor<128x64xf32, #blocked>
      %acc0_78 = arith.mulf %outLHS, %acc0_77 : tensor<128x64xf32, #blocked>
      %acc1 = arith.mulf %outRHS, %acc0_77 : tensor<128x64xf32, #blocked>
      %acc_79 = tt.join %acc0_78, %acc1 : tensor<128x64xf32, #blocked> -> tensor<128x64x2xf32, #blocked5>
      %acc_80 = tt.trans %acc_79 {order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked5> -> tensor<128x2x64xf32, #blocked4>
      %acc_81 = tt.reshape %acc_80 : tensor<128x2x64xf32, #blocked4> -> tensor<128x128xf32, #blocked1>
      // CHECK: tt.descriptor_load {{.*}} {loop.cluster = [[CLUSTER2:[0-9]+]] : i32, loop.stage = 2 : i32} {{.*}}
      %v = tt.descriptor_load %desc_v[%offset_y_59, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked3>
      %v_82 = ttg.local_alloc %v : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %p_83 = arith.truncf %p : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
      %acc_84 = ttng.tmem_alloc %p_83 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #tmem2, #ttng.tensor_memory>
      %acc_85 = ttng.tmem_store %acc_81, %acc_30[%acc_76], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = [[CLUSTER2]] : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} {{.*}}
      %acc_86 = ttng.tc_gen5_mma %acc_84, %v_82, %acc_30[%acc_85], %arg29, %true {tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #tmem2, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      %l_i_87 = arith.mulf %l_i_57, %alpha_74 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_i_88 = arith.addf %l_i_87, %l_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %offsetk_y_89 = arith.addi %offset_y_59, %c64_i32 : i32
      // CHECK: scf.yield {{.*}}
      scf.yield %l_i_88, %m_ij_69, %offsetk_y_89, %true, %qk_66, %acc_86 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i1, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize}
    %acc_33, %acc_34 = ttng.tmem_load %acc_30[%offsetv_y#5] : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %0 = arith.muli %start_m, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
    %1 = arith.addi %start_m, %c1_i32 : i32
    %2 = arith.muli %1, %c128_i32 : i32
    %offsetk_y = arith.addi %offset_y_20, %0 : i32
    %mask = tt.expand_dims %offs_m_25 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %mask_35 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_36 = tt.expand_dims %mask_35 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %mask_37 = tt.broadcast %mask : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %qk_38 = tt.splat %qk_scale : f32 -> tensor<128x64xf32, #blocked>
    %qk_39, %qk_40 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_41, %acc_42 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_43 = ttng.tmem_store %acc_33, %acc_41[%acc_42], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: scf.for {{.*}}
    %offsetv_y_44:5 = scf.for %offsetv_y_56 = %0 to %2 step %c64_i32 iter_args(%offsetv_y_57 = %offsetv_y#0, %offsetv_y_58 = %offsetv_y#1, %offsetk_y_59 = %offsetk_y, %qk_60 = %qk_40, %acc_61 = %acc_43) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, !ttg.async.token, !ttg.async.token)  : i32 {
      // CHECK: tt.descriptor_load {{.*}} {loop.cluster = [[CLUSTER3:[0-9]+]] : i32, loop.stage = 0 : i32} {{.*}}
      %k = tt.descriptor_load %desc_k[%offsetk_y_59, %c0_i32] {tt.latency = 1 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked3>
      %k_62 = ttg.local_alloc %k : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %k_63 = ttg.memdesc_trans %k_62 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xf16, #shared, #smem> -> !ttg.memdesc<128x64xf16, #shared1, #smem>
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = [[CLUSTER3]] : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} {{.*}}
      %qk_64 = ttng.tc_gen5_mma %q_27, %k_63, %qk_39[%qk_60], %false, %true {tt.latency = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x64xf16, #shared1, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %mask_65 = tt.splat %offsetv_y_56 : i32 -> tensor<1x64xi32, #blocked>
      %mask_66 = arith.addi %mask_65, %mask_36 : tensor<1x64xi32, #blocked>
      %mask_67 = tt.broadcast %mask_66 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
      %mask_68 = arith.cmpi sge, %mask_37, %mask_67 : tensor<128x64xi32, #blocked>
      %qk_69, %qk_70 = ttng.tmem_load %qk_39[%qk_64] : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
      %qk_71 = arith.mulf %qk_69, %qk_38 : tensor<128x64xf32, #blocked>
      %qk_72 = arith.select %mask_68, %cst_16, %cst_17 : tensor<128x64xi1, #blocked>, tensor<128x64xf32, #blocked>
      %qk_73 = arith.addf %qk_71, %qk_72 : tensor<128x64xf32, #blocked>
      %m_ij_74 = "tt.reduce"(%qk_73) <{axis = 1 : i32}> ({
      ^bb0(%m_ij_95: f32, %m_ij_96: f32):
        %m_ij_97 = arith.maxnumf %m_ij_95, %m_ij_96 : f32
        tt.reduce.return %m_ij_97 : f32
      }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_ij_75 = arith.maxnumf %offsetv_y_58, %m_ij_74 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %qk_76 = tt.expand_dims %m_ij_75 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %qk_77 = tt.broadcast %qk_76 : tensor<128x1xf32, #blocked> -> tensor<128x64xf32, #blocked>
      %qk_78 = arith.subf %qk_73, %qk_77 : tensor<128x64xf32, #blocked>
      %p = math.exp2 %qk_78 : tensor<128x64xf32, #blocked>
      %alpha = arith.subf %offsetv_y_58, %m_ij_75 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_79 = math.exp2 %alpha : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
      ^bb0(%l_ij_95: f32, %l_ij_96: f32):
        %l_ij_97 = arith.addf %l_ij_95, %l_ij_96 : f32
        tt.reduce.return %l_ij_97 : f32
      }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %acc_80, %acc_81 = ttng.tmem_load %acc_41[%acc_61] : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %6 = tt.reshape %acc_80 : tensor<128x128xf32, #blocked1> -> tensor<128x2x64xf32, #blocked4>
      %7 = tt.trans %6 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %outLHS, %outRHS = tt.split %7 : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked>
      %acc0 = tt.expand_dims %alpha_79 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %acc0_82 = tt.broadcast %acc0 : tensor<128x1xf32, #blocked> -> tensor<128x64xf32, #blocked>
      %acc0_83 = arith.mulf %outLHS, %acc0_82 : tensor<128x64xf32, #blocked>
      %acc1 = arith.mulf %outRHS, %acc0_82 : tensor<128x64xf32, #blocked>
      %acc_84 = tt.join %acc0_83, %acc1 : tensor<128x64xf32, #blocked> -> tensor<128x64x2xf32, #blocked5>
      %acc_85 = tt.trans %acc_84 {order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #blocked5> -> tensor<128x2x64xf32, #blocked4>
      %acc_86 = tt.reshape %acc_85 : tensor<128x2x64xf32, #blocked4> -> tensor<128x128xf32, #blocked1>
      // CHECK: tt.descriptor_load {{.*}} {loop.cluster = [[CLUSTER4:[0-9]+]] : i32, loop.stage = {{[0-9]+}} : i32} {{.*}}
      %v = tt.descriptor_load %desc_v[%offsetk_y_59, %c0_i32] {tt.latency = 1 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked3>
      %v_87 = ttg.local_alloc %v : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %p_88 = arith.truncf %p : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
      %acc_89 = ttng.tmem_alloc %p_88 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #tmem2, #ttng.tensor_memory>
      %acc_90 = ttng.tmem_store %acc_86, %acc_41[%acc_81], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = [[CLUSTER5:[0-9]+]] : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} {{.*}}
      %acc_91 = ttng.tc_gen5_mma %acc_89, %v_87, %acc_41[%acc_90], %true, %true {tt.self_latency = 1 : i32} : !ttg.memdesc<128x64xf16, #tmem2, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      %l_i_92 = arith.mulf %offsetv_y_57, %alpha_79 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_i_93 = arith.addf %l_i_92, %l_ij : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %offsetk_y_94 = arith.addi %offsetk_y_59, %c64_i32 : i32
      // CHECK: scf.yield {{.*}}
      scf.yield %l_i_93, %m_ij_75, %offsetk_y_94, %qk_70, %acc_91 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize}
    %acc_45, %acc_46 = ttng.tmem_load %acc_41[%offsetv_y_44#4] : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %m_i_47 = math.log2 %offsetv_y_44#0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %m_i_48 = arith.addf %offsetv_y_44#1, %m_i_47 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %acc_49 = tt.expand_dims %offsetv_y_44#0 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %acc_50 = ttg.convert_layout %acc_49 : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked1>
    %acc_51 = tt.broadcast %acc_50 : tensor<128x1xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
    %acc_52 = arith.divf %acc_45, %acc_51 : tensor<128x128xf32, #blocked1>
    %m_ptrs = arith.muli %off_hz, %N_CTX : i32
    %m_ptrs_53 = tt.addptr %M, %m_ptrs : !tt.ptr<f32>, i32
    %m_ptrs_54 = tt.splat %m_ptrs_53 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
    %m_ptrs_55 = tt.addptr %m_ptrs_54, %offs_m_26 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
    %3 = ttg.convert_layout %m_i_48 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked2>
    tt.store %m_ptrs_55, %3 : tensor<128x!tt.ptr<f32>, #blocked2>
    %4 = arith.truncf %acc_52 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    %5 = ttg.convert_layout %4 : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #blocked3>
    tt.descriptor_store %desc_o[%qo_offset_y_21, %c0_i32], %5 : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked3>
    tt.return
  }
}
