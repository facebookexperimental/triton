// RUN: triton-opt %s -split-input-file --nvgpu-warp-specialization="capability=100" | FileCheck %s
// CHECK-LABEL: _attn_fwd_persist
// CHECK: ttg.warp_specialize
// default: Accumulator correction (tmem_load acc, expand_dims alpha, broadcast, mulf for acc scaling, tmem_store acc)
// CHECK: default
// CHECK: ttng.tmem_load
// CHECK: ttng.tmem_load
// CHECK: ttng.tmem_store
// CHECK: ttng.tmem_store
// partition0: MMA operations (tc_gen5_mma)
// CHECK: partition0
// CHECK: ttng.tc_gen5_mma
// CHECK: ttng.tc_gen5_mma
// CHECK: ttng.tc_gen5_mma
// CHECK: ttng.tc_gen5_mma
// partition1: Descriptor loads (Q, K, V loads and local_alloc)
// CHECK: partition1
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.async_tma_copy_global_to_local
// partition2: Output TMA store (convert_layout, descriptor_store for output)
// CHECK: partition2
// CHECK: ttg.convert_layout
// CHECK: tt.descriptor_store
// CHECK: ttg.convert_layout
// CHECK: tt.descriptor_store
// partition3: Softmax 1 (tmem_load qk, reduce max/sum, exp2, truncf, tmem_alloc p)
// CHECK: partition3
// CHECK: ttng.tmem_load
// CHECK: tt.reduce
// CHECK: math.exp2
// CHECK: tt.reduce
// CHECK: arith.truncf
// partition4: Softmax 2 (tmem_load qk, reduce max/sum, exp2, truncf, tmem_alloc p)
// CHECK: partition4
// CHECK: ttng.tmem_load
// CHECK: tt.reduce
// CHECK: math.exp2
// CHECK: tt.reduce
// CHECK: arith.truncf

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 128 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_persist(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_o: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %n_tile_num = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant 1.44269502 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %prog_id = tt.get_program_id x : i32
    %num_progs = tt.get_num_programs x : i32
    %total_tiles = arith.muli %Z, %n_tile_num : i32
    %total_tiles_3 = arith.muli %total_tiles, %H : i32
    %tiles_per_sm = arith.divsi %total_tiles_3, %num_progs : i32
    %0 = arith.remsi %total_tiles_3, %num_progs : i32
    %1 = arith.cmpi slt, %prog_id, %0 : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_15 = arith.addi %tiles_per_sm, %c1_i32 : i32
      scf.yield %tiles_per_sm_15 : i32
    } else {
      scf.yield %tiles_per_sm : i32
    }
    %desc_q_4 = arith.muli %Z, %H : i32
    %desc_q_5 = arith.muli %desc_q_4, %c1024_i32 : i32
    %desc_q_6 = tt.make_tensor_descriptor %desc_q, [%desc_q_5, %c64_i32], [%c64_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
    %desc_q_7 = tt.make_tensor_descriptor %desc_q, [%desc_q_5, %c64_i32], [%c64_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
    %desc_k_8 = tt.make_tensor_descriptor %desc_k, [%desc_q_5, %c64_i32], [%c64_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
    %desc_v_9 = tt.make_tensor_descriptor %desc_v, [%desc_q_5, %c64_i32], [%c64_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
    %desc_o_10 = tt.make_tensor_descriptor %desc_o, [%desc_q_5, %c64_i32], [%c64_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
    %desc_o_11 = tt.make_tensor_descriptor %desc_o, [%desc_q_5, %c64_i32], [%c64_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
    %offset_y = arith.muli %H, %c1024_i32 : i32
    %offs_m0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %offs_m0_12 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked2>
    %qk_scale = arith.mulf %sm_scale, %cst : f32
    %m_ij = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %m_ij_13 = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %qk = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #blocked1>
    %qk_14 = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #blocked1>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_15 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_15, %n_tile_num : i32
      %off_hz = arith.divsi %tile_idx_15, %n_tile_num : i32
      %off_z = arith.divsi %off_hz, %H : i32
      %off_h = arith.remsi %off_hz, %H : i32
      %offset_y_16 = arith.muli %off_z, %offset_y : i32
      %offset_y_17 = arith.muli %off_h, %c1024_i32 : i32
      %offset_y_18 = arith.addi %offset_y_16, %offset_y_17 : i32
      %qo_offset_y = arith.muli %pid, %c256_i32 : i32
      %qo_offset_y_19 = arith.addi %offset_y_18, %qo_offset_y : i32
      %3 = arith.addi %qo_offset_y_19, %c128_i32 : i32
      %q0 = arith.addi %qo_offset_y_19, %c128_i32 : i32
      %offs_m0_20 = tt.splat %qo_offset_y : i32 -> tensor<128xi32, #blocked2>
      %offs_m0_21 = tt.splat %qo_offset_y : i32 -> tensor<128xi32, #blocked2>
      %offs_m0_22 = arith.addi %offs_m0_20, %offs_m0 : tensor<128xi32, #blocked2>
      %offs_m0_23 = arith.addi %offs_m0_21, %offs_m0_12 : tensor<128xi32, #blocked2>
      %q0_24 = tt.descriptor_load %desc_q_6[%qo_offset_y_19, %c0_i32] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked3>
      %q0_25 = tt.descriptor_load %desc_q_7[%q0, %c0_i32] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked3>
      %q0_26 = ttg.local_alloc %q0_24 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked3>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %q0_27 = ttg.local_alloc %q0_25 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked3>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %qk_28, %qk_29 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %qk_30, %qk_31 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc, %acc_32 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_33, %acc_34 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_35 = ttng.tmem_store %cst_0, %acc[%acc_32], %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
      %acc_36 = ttng.tmem_store %cst_0, %acc_33[%acc_34], %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
      %offsetkv_y:10 = scf.for %offsetkv_y_57 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%offset_y_58 = %offset_y_18, %arg12 = %false, %arg13 = %cst_2, %arg14 = %cst_1, %qk_59 = %qk_29, %acc_60 = %acc_35, %arg17 = %cst_2, %arg18 = %cst_1, %qk_61 = %qk_31, %acc_62 = %acc_36) -> (i32, i1, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        %acc_63, %acc_64 = ttng.tmem_load %acc[%acc_60] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
        %acc_65, %acc_66 = ttng.tmem_load %acc_33[%acc_62] {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
        %10 = ttg.convert_layout %acc_63 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked1>
        %11 = ttg.convert_layout %acc_65 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked1>
        %k = tt.descriptor_load %desc_k_8[%offset_y_58, %c0_i32] {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked3>
        %k_67 = ttg.local_alloc %k {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked3>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %k_68 = ttg.memdesc_trans %k_67 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %v = tt.descriptor_load %desc_v_9[%offset_y_58, %c0_i32] {loop.cluster = 6 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked3>
        %v_69 = ttg.local_alloc %v {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked3>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %qk_70 = ttng.tc_gen5_mma %q0_26, %k_68, %qk_28[%qk_59], %false, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_71 = ttng.tc_gen5_mma %q0_27, %k_68, %qk_30[%qk_61], %false, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_72, %qk_73 = ttng.tmem_load %qk_28[%qk_70] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
        %qk_74, %qk_75 = ttng.tmem_load %qk_30[%qk_71] {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
        %m_ij_76 = "tt.reduce"(%qk_72) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_117: f32, %m_ij_118: f32):
          %m_ij_119 = arith.maxnumf %m_ij_117, %m_ij_118 : f32
          tt.reduce.return %m_ij_119 : f32
        }) {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : (tensor<128x128xf32, #blocked1>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %m_ij_77 = "tt.reduce"(%qk_74) <{axis = 1 : i32}> ({
        ^bb0(%m_ij_117: f32, %m_ij_118: f32):
          %m_ij_119 = arith.maxnumf %m_ij_117, %m_ij_118 : f32
          tt.reduce.return %m_ij_119 : f32
        }) {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : (tensor<128x128xf32, #blocked1>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %m_ij_78 = arith.mulf %m_ij_76, %m_ij {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %m_ij_79 = arith.mulf %m_ij_77, %m_ij_13 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %m_ij_80 = arith.maxnumf %arg14, %m_ij_78 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %m_ij_81 = arith.maxnumf %arg18, %m_ij_79 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %qk_82 = arith.mulf %qk_72, %qk {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : tensor<128x128xf32, #blocked1>
        %qk_83 = arith.mulf %qk_74, %qk_14 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128x128xf32, #blocked1>
        %qk_84 = tt.expand_dims %m_ij_80 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
        %qk_85 = tt.expand_dims %m_ij_81 {axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
        %qk_86 = tt.broadcast %qk_84 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : tensor<128x1xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
        %qk_87 = tt.broadcast %qk_85 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128x1xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
        %qk_88 = arith.subf %qk_82, %qk_86 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : tensor<128x128xf32, #blocked1>
        %qk_89 = arith.subf %qk_83, %qk_87 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128x128xf32, #blocked1>
        %p = math.exp2 %qk_88 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : tensor<128x128xf32, #blocked1>
        %p_90 = math.exp2 %qk_89 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128x128xf32, #blocked1>
        %alpha = arith.subf %arg14, %m_ij_80 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %alpha_91 = arith.subf %arg18, %m_ij_81 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %alpha_92 = math.exp2 %alpha {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %alpha_93 = math.exp2 %alpha_91 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_117: f32, %l_ij_118: f32):
          %l_ij_119 = arith.addf %l_ij_117, %l_ij_118 : f32
          tt.reduce.return %l_ij_119 : f32
        }) {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 5 : i32} : (tensor<128x128xf32, #blocked1>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %l_ij_94 = "tt.reduce"(%p_90) <{axis = 1 : i32}> ({
        ^bb0(%l_ij_117: f32, %l_ij_118: f32):
          %l_ij_119 = arith.addf %l_ij_117, %l_ij_118 : f32
          tt.reduce.return %l_ij_119 : f32
        }) {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : (tensor<128x128xf32, #blocked1>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %acc_95 = tt.expand_dims %alpha_92 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
        %acc_96 = tt.expand_dims %alpha_93 {axis = 1 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
        %acc_97 = tt.broadcast %acc_95 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : tensor<128x1xf32, #blocked1> -> tensor<128x64xf32, #blocked1>
        %acc_98 = tt.broadcast %acc_96 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<128x1xf32, #blocked1> -> tensor<128x64xf32, #blocked1>
        %acc_99 = arith.mulf %10, %acc_97 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked1>
        %acc_100 = arith.mulf %11, %acc_98 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked1>
        %p_101 = arith.truncf %p {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
        %p_102 = arith.truncf %p_90 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
        %acc_103 = ttg.convert_layout %p_101 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #blocked1>
        %acc_104 = ttng.tmem_alloc %acc_103 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 5 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory>
        %acc_105 = ttg.convert_layout %p_102 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #blocked1>
        %acc_106 = ttng.tmem_alloc %acc_105 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory>
        %acc_107 = ttg.convert_layout %acc_99 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #blocked>
        %acc_108 = ttg.convert_layout %acc_100 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #blocked>
        %acc_109 = ttng.tmem_store %acc_107, %acc[%acc_64], %true {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %acc_110 = ttng.tmem_store %acc_108, %acc_33[%acc_66], %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %acc_111 = ttng.tc_gen5_mma %acc_104, %v_69, %acc[%acc_109], %arg12, %true {loop.cluster = 4 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory>, !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %acc_112 = ttng.tc_gen5_mma %acc_106, %v_69, %acc_33[%acc_110], %arg12, %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory>, !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %l_i0 = arith.mulf %arg13, %alpha_92 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 5 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %l_i0_113 = arith.mulf %arg17, %alpha_93 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %l_i0_114 = arith.addf %l_i0, %l_ij {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = 5 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %l_i0_115 = arith.addf %l_i0_113, %l_ij_94 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %offsetkv_y_116 = arith.addi %offset_y_58, %c128_i32 {loop.cluster = 5 : i32, loop.stage = 1 : i32} : i32
        scf.yield %offsetkv_y_116, %true, %l_i0_114, %m_ij_80, %qk_73, %acc_111, %l_i0_115, %m_ij_81, %qk_75, %acc_112 : i32, i1, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, !ttg.async.token, !ttg.async.token
      } {tt.data_partition_factor = 2 : i32, tt.disallow_acc_multi_buffer, tt.scheduled_max_stage = 2 : i32, ttg.partition = 0 : i32}
      %acc_37, %acc_38 = ttng.tmem_load %acc[%offsetkv_y#5] {ttg.partition = 0 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
      %acc_39, %acc_40 = ttng.tmem_load %acc_33[%offsetkv_y#9] {ttg.partition = 0 : i32} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
      %offsetkv_y_41 = ttg.convert_layout %acc_37 {ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked1>
      %offsetkv_y_42 = ttg.convert_layout %acc_39 {ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked1>
      %m_i0 = math.log2 %offsetkv_y#2 {ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %m_i0_43 = math.log2 %offsetkv_y#6 {ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %m_i0_44 = arith.addf %offsetkv_y#3, %m_i0 {ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %m_i0_45 = arith.addf %offsetkv_y#7, %m_i0_43 {ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %acc0 = tt.expand_dims %offsetkv_y#2 {axis = 1 : i32, ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
      %acc0_46 = tt.expand_dims %offsetkv_y#6 {axis = 1 : i32, ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
      %acc0_47 = tt.broadcast %acc0 {ttg.partition = 0 : i32} : tensor<128x1xf32, #blocked1> -> tensor<128x64xf32, #blocked1>
      %acc0_48 = tt.broadcast %acc0_46 {ttg.partition = 0 : i32} : tensor<128x1xf32, #blocked1> -> tensor<128x64xf32, #blocked1>
      %acc0_49 = arith.divf %offsetkv_y_41, %acc0_47 {ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked1>
      %acc0_50 = arith.divf %offsetkv_y_42, %acc0_48 {ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked1>
      %m_ptrs0 = arith.muli %off_hz, %c1024_i32 : i32
      %m_ptrs0_51 = tt.addptr %M, %m_ptrs0 : !tt.ptr<f32>, i32
      %m_ptrs0_52 = tt.splat %m_ptrs0_51 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %m_ptrs0_53 = tt.splat %m_ptrs0_51 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %m_ptrs0_54 = tt.addptr %m_ptrs0_52, %offs_m0_22 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
      %m_ptrs0_55 = tt.addptr %m_ptrs0_53, %offs_m0_23 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
      %4 = ttg.convert_layout %m_i0_44 {ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128xf32, #blocked2>
      %5 = ttg.convert_layout %m_i0_45 {ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128xf32, #blocked2>
      tt.store %m_ptrs0_54, %4 {ttg.partition = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
      tt.store %m_ptrs0_55, %5 {ttg.partition = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
      %6 = arith.truncf %acc0_49 {ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
      %7 = arith.truncf %acc0_50 {ttg.partition = 0 : i32} : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
      %8 = ttg.convert_layout %6 {ttg.partition = 3 : i32} : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #blocked3>
      %9 = ttg.convert_layout %7 {ttg.partition = 3 : i32} : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #blocked3>
      tt.descriptor_store %desc_o_10[%qo_offset_y_19, %c0_i32], %8 {ttg.partition = 3 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked3>
      tt.descriptor_store %desc_o_11[%3, %c0_i32], %9 {ttg.partition = 3 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked3>
      %tile_idx_56 = arith.addi %tile_idx_15, %num_progs : i32
      scf.yield %tile_idx_56 : i32
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
