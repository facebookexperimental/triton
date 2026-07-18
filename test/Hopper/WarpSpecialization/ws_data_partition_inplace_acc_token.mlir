// RUN: triton-opt %s -split-input-file --nvgpu-ws-data-partition=num-warp-groups=2 | FileCheck %s

// HSTU self-attention forward: the PV (output) MMA is a loop-carried in-place
// TMEM accumulator that is read only AFTER the loop. Data partitioning (DP=2)
// must split its full [256,128] accumulator into two per-partition [128,128]
// tiles, each with its OWN loop-carried token — mirroring the QK accumulator.
//
// Regression: DP used to slice the MMAs but leave the original full [256,128]
// accumulator's zero-init store (and hence its loop-carried token) un-sliced.
// The split MMAs then shared the full accumulator's token, keeping the full
// (dead-data) [256,128] TMEM tile live via the token cycle -> 2x TMEM -> OOM.
// After the fix no full-tile TMEM accumulator may survive.

// CHECK-LABEL: @_hstu_attn_fwd
// A per-partition [128,128] accumulator must exist (DP actually fired) ...
// CHECK: ttng.tmem_alloc {{.*}}128x128xf32{{.*}}#ttng.tensor_memory
// ... and NO full-tile [256,x] TMEM accumulator may survive the split.
// CHECK-NOT: 256x128xf32, {{.+}}#ttng.tensor_memory
// CHECK-NOT: 256x128xbf16, {{.+}}#ttng.tensor_memory

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [128, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_hstu_attn_fwd(%Q: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %K: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %V: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %workspace_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %seq_offsets: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %seq_offsets_q: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %Out: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %alpha: f32, %stride_qm: i32 {tt.divisibility = 16 : i32}, %stride_qh: i32 {tt.divisibility = 16 : i32}, %stride_kn: i32 {tt.divisibility = 16 : i32}, %stride_kh: i32 {tt.divisibility = 16 : i32}, %stride_vn: i32 {tt.divisibility = 16 : i32}, %stride_vh: i32 {tt.divisibility = 16 : i32}, %stride_om: i32 {tt.divisibility = 16 : i32}, %stride_oh: i32 {tt.divisibility = 16 : i32}, %attn_scale: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %AUTOTUNE_MAX_SEQ_LEN: i32 {tt.divisibility = 16 : i32}, %DimQ: i32 {tt.divisibility = 16 : i32}, %DimV: i32 {tt.divisibility = 16 : i32}, %max_attn_len: i32 {tt.divisibility = 16 : i32}, %contextual_seq_len: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #linear>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<256x128xf32, #linear>
    %true = arith.constant true
    %c383_i32 = arith.constant 383 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2_i64 = arith.constant 2 : i64
    %device_desc_o = arith.constant 384 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0> : tensor<256x128xi32, #linear>
    %off_hz = tt.get_program_id y : i32
    %pid = tt.get_program_id x : i32
    %off_z = arith.extsi %off_hz : i32 to i64
    %seq_start_kv = tt.addptr %seq_offsets, %off_z : !tt.ptr<i64>, i64
    %seq_start_kv_2 = tt.load %seq_start_kv : !tt.ptr<i64>
    %seq_end_kv = tt.addptr %seq_start_kv, %c1_i32 : !tt.ptr<i64>, i32
    %seq_end_kv_3 = tt.load %seq_end_kv : !tt.ptr<i64>
    %seq_start_q = tt.addptr %seq_offsets_q, %off_z : !tt.ptr<i64>, i64
    %seq_start_q_4 = tt.load %seq_start_q : !tt.ptr<i64>
    %seq_end_q = tt.addptr %seq_start_q, %c1_i32 : !tt.ptr<i64>, i32
    %seq_end_q_5 = tt.load %seq_end_q : !tt.ptr<i64>
    %seq_len_q = arith.subi %seq_end_q_5, %seq_start_q_4 : i64
    %seq_len_q_6 = arith.trunci %seq_len_q : i64 to i32
    %workspace_base = tt.get_num_programs y : i32
    %workspace_base_7 = arith.muli %pid, %workspace_base : i32
    %workspace_base_8 = arith.addi %off_hz, %workspace_base_7 : i32
    %workspace_base_9 = arith.muli %workspace_base_8, %c512_i32 : i32
    %workspace_base_10 = tt.addptr %workspace_ptr, %workspace_base_9 : !tt.ptr<i8>, i32
    %device_desc_k = tt.addptr %workspace_base_10, %c128_i32 : !tt.ptr<i8>, i32
    %device_desc_v = tt.addptr %workspace_base_10, %c256_i32 : !tt.ptr<i8>, i32
    %device_desc_o_11 = tt.addptr %workspace_base_10, %device_desc_o : !tt.ptr<i8>, i32
    %0 = arith.trunci %seq_end_kv_3 : i64 to i32
    %1 = arith.extsi %DimQ : i32 to i64
    %2 = arith.muli %1, %c2_i64 : i64
    ttng.tensormap_create %device_desc_k, %K, [%c64_i32, %c128_i32], [%DimQ, %0], [%2], [%c1_i32, %c1_i32] {elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    %3 = arith.extsi %DimV : i32 to i64
    %4 = arith.muli %3, %c2_i64 : i64
    ttng.tensormap_create %device_desc_v, %V, [%c64_i32, %c128_i32], [%DimV, %0], [%4], [%c1_i32, %c1_i32] {elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %device_desc_k : !tt.ptr<i8>
    ttng.tensormap_fenceproxy_acquire %device_desc_v : !tt.ptr<i8>
    %start_m = arith.muli %pid, %c256_i32 : i32
    %5 = arith.cmpi slt, %start_m, %seq_len_q_6 : i32
    scf.if %5 {
      %offs_m = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %offs_m_12 = tt.splat %start_m : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %offs_m_13 = arith.addi %offs_m_12, %offs_m : tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %offs_n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
      %scale = tt.load %attn_scale : !tt.ptr<f32>
      %6 = arith.trunci %seq_end_q_5 : i64 to i32
      ttng.tensormap_create %workspace_base_10, %Q, [%c64_i32, %c256_i32], [%DimQ, %6], [%2], [%c1_i32, %c1_i32] {elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
      ttng.tensormap_fenceproxy_acquire %workspace_base_10 : !tt.ptr<i8>
      %q = arith.extsi %start_m : i32 to i64
      %q_14 = arith.addi %seq_start_q_4, %q : i64
      %q_15 = arith.trunci %q_14 : i64 to i32
      %q_16 = ttng.reinterpret_tensor_descriptor %workspace_base_10 : !tt.ptr<i8> to !tt.tensordesc<tensor<256x128xbf16, #shared>>
      %q_17 = tt.descriptor_load %q_16[%q_15, %c0_i32] : !tt.tensordesc<tensor<256x128xbf16, #shared>> -> tensor<256x128xbf16, #blocked>
      %q_18 = ttg.local_alloc %q_17 : (tensor<256x128xbf16, #blocked>) -> !ttg.memdesc<256x128xbf16, #shared, #smem>
      %high = arith.addi %start_m, %c256_i32 : i32
      %n_uih = arith.addi %start_m, %c383_i32 : i32
      %n_uih_19 = arith.divsi %n_uih, %c128_i32 : i32
      %k = ttng.reinterpret_tensor_descriptor %device_desc_k : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16, #shared>>
      %valid_mask = tt.expand_dims %offs_m_13 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<256x1xi32, #linear>
      %valid_mask_20 = tt.broadcast %valid_mask : tensor<256x1xi32, #linear> -> tensor<256x128xi32, #linear>
      %qk = tt.splat %alpha : f32 -> tensor<256x128xf32, #linear>
      %silu = tt.splat %scale : f32 -> tensor<256x128xf32, #linear>
      %v = ttng.reinterpret_tensor_descriptor %device_desc_v : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16, #shared>>
      %qk_21, %qk_22 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc, %acc_23 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_24 = ttng.tmem_store %cst, %acc[%acc_23], %true : tensor<256x128xf32, #linear> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_25:3 = scf.for %it = %c0_i32 to %n_uih_19 step %c1_i32 iter_args(%acc_29 = %false, %qk_30 = %qk_22, %acc_31 = %acc_24) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
        %is_tgt = arith.cmpi sge, %it, %n_uih_19 : i32
        %blk = arith.subi %it, %n_uih_19 : i32
        %blk_32 = arith.select %is_tgt, %blk, %it : i32
        %start_n = arith.muli %blk_32, %c128_i32 : i32
        %start_n_33 = arith.addi %high, %start_n : i32
        %start_n_34 = arith.select %is_tgt, %start_n_33, %start_n : i32
        %acc_35 = tt.splat %start_n_34 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %acc_36 = arith.addi %offs_n, %acc_35 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %k_37 = arith.extsi %start_n_34 : i32 to i64
        %k_38 = arith.addi %seq_start_kv_2, %k_37 : i64
        %k_39 = arith.trunci %k_38 : i64 to i32
        %k_40 = tt.descriptor_load %k[%k_39, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked>
        %qk_41 = ttg.local_alloc %k_40 : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %qk_42 = ttg.memdesc_trans %qk_41 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
        %qk_43 = ttng.tc_gen5_mma %q_18, %qk_42, %qk_21[%qk_30], %false, %true : !ttg.memdesc<256x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %valid_mask_44 = tt.expand_dims %acc_36 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xi32, #linear>
        %valid_mask_45 = tt.broadcast %valid_mask_44 : tensor<1x128xi32, #linear> -> tensor<256x128xi32, #linear>
        %valid_mask_46 = arith.cmpi eq, %valid_mask_20, %valid_mask_45 : tensor<256x128xi32, #linear>
        %offs_m_minus_n = arith.subi %valid_mask_20, %valid_mask_45 : tensor<256x128xi32, #linear>
        %valid_mask_47 = arith.cmpi sgt, %offs_m_minus_n, %cst_1 : tensor<256x128xi32, #linear>
        %valid_mask_48 = arith.ori %valid_mask_46, %valid_mask_47 : tensor<256x128xi1, #linear>
        %qk_49, %qk_50 = ttng.tmem_load %qk_21[%qk_43] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear>
        %qk_51 = arith.mulf %qk_49, %qk : tensor<256x128xf32, #linear>
        %silu_52 = arith.subf %cst, %qk_51 : tensor<256x128xf32, #linear>
        %silu_53 = math.exp %silu_52 : tensor<256x128xf32, #linear>
        %silu_54 = arith.addf %silu_53, %cst_0 : tensor<256x128xf32, #linear>
        %silu_55 = tt.extern_elementwise %qk_51, %silu_54 {libname = "", libpath = "", pure = true, symbol = "__nv_fast_fdividef"} : (tensor<256x128xf32, #linear>, tensor<256x128xf32, #linear>) -> tensor<256x128xf32, #linear>
        %silu_56 = arith.mulf %silu_55, %silu : tensor<256x128xf32, #linear>
        %act_qk = arith.select %valid_mask_48, %silu_56, %cst : tensor<256x128xi1, #linear>, tensor<256x128xf32, #linear>
        %v_57 = tt.descriptor_load %v[%k_39, %c0_i32] : !tt.tensordesc<tensor<128x128xbf16, #shared>> -> tensor<128x128xbf16, #blocked>
        %v_58 = ttg.local_alloc %v_57 : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
        %act_qk_59 = arith.truncf %act_qk : tensor<256x128xf32, #linear> to tensor<256x128xbf16, #linear>
        %acc_60 = ttng.tmem_alloc %act_qk_59 : (tensor<256x128xbf16, #linear>) -> !ttg.memdesc<256x128xbf16, #tmem, #ttng.tensor_memory>
        %acc_61 = ttng.tc_gen5_mma %acc_60, %v_58, %acc[%acc_31], %acc_29, %true : !ttg.memdesc<256x128xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %qk_50, %acc_61 : i1, !ttg.async.token, !ttg.async.token
      } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}
      %acc_26, %acc_27 = ttng.tmem_load %acc[%acc_25#2] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear>
      %acc_28 = arith.truncf %acc_26 : tensor<256x128xf32, #linear> to tensor<256x128xbf16, #linear>
      ttng.tensormap_create %device_desc_o_11, %Out, [%c64_i32, %c256_i32], [%DimV, %6], [%4], [%c1_i32, %c1_i32] {elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
      ttng.tensormap_fenceproxy_acquire %device_desc_o_11 : !tt.ptr<i8>
      %7 = ttng.reinterpret_tensor_descriptor %device_desc_o_11 : !tt.ptr<i8> to !tt.tensordesc<tensor<256x128xbf16, #shared>>
      %8 = ttg.convert_layout %acc_28 : tensor<256x128xbf16, #linear> -> tensor<256x128xbf16, #blocked>
      tt.descriptor_store %7[%q_15, %c0_i32], %8 : !tt.tensordesc<tensor<256x128xbf16, #shared>>, tensor<256x128xbf16, #blocked>
    }
    tt.return
  }
}


