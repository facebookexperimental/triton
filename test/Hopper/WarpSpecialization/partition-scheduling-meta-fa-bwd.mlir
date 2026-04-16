// RUN: triton-opt %s --nvgpu-partition-scheduling-meta="merge-epilogue-to-computation" | FileCheck %s

// Tests that the full FA BWD persistent kernel (bwd.part.prior) gets the correct
// 4-partition layout: reduction + gemm + load + computation.
// This is a real BWD FA kernel dumped from fused-attention-ws-device-tma.py.
//
// Partition structure:
//   0 = reduction: dq tmem_load, reshape/split, descriptor_reduce, dk/dv init
//   1 = gemm:      all 5 MMAs (QK, dpT, dv, dq, dk) + memdesc_trans
//   2 = load:      descriptor_load (K, V, Q, dO) + local_alloc
//   3 = computation: QK tmem_load, softmax, dpT tmem_load, dsT computation,
//                    p tmem_alloc, post-loop tmem_load/reshape/split/descriptor_store

// CHECK-LABEL: @_attn_bwd_persist
//
// --- Pre-loop: address computation -> reduction partition ---
// (scalar ops may be unscheduled since they can be rematerialized)
// CHECK: arith.divsi {{.*}}ttg.partition = array<i32: [[RED:[0-9]+]]>
// CHECK: arith.remsi {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: arith.muli {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: arith.divsi {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: arith.muli {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: arith.addi {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: arith.extsi {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: arith.divsi {{.*}}ttg.partition = array<i32: [[RED]]>
// --- Pre-loop: K, V descriptor_load -> load partition ---
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD:[0-9]+]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: tt.splat {{.*}}ttg.partition = array<i32: [[COMP:[0-9]+]]>
// CHECK: tt.splat {{.*}}ttg.partition = array<i32: [[COMP]]>
// --- Pre-loop: dq tmem_alloc, dk/dv init → reduction partition ---
// CHECK: ttng.tmem_alloc {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: ttng.tmem_store {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: ttng.tmem_store {{.*}}ttg.partition = array<i32: [[RED]]>
// --- In-loop: address computation → reduction partition ---
// CHECK: arith.extsi {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: arith.addi {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: arith.trunci {{.*}}ttg.partition = array<i32: [[RED]]>
// --- In-loop: Q descriptor_load, local_alloc → load partition ---
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[LOAD]]>
// --- In-loop: Q memdesc_trans → gemm partition ---
// CHECK: ttg.memdesc_trans {{.*}}ttg.partition = array<i32: [[GEMM:[0-9]+]]>
// --- In-loop: QK MMA → gemm partition ---
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[GEMM]]>
// --- In-loop: QK tmem_load, softmax → computation partition ---
// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.subf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: math.exp2 {{.*}}ttg.partition = array<i32: [[COMP]]>
// --- In-loop: dO descriptor_load, local_alloc → load partition ---
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[LOAD]]>
// --- In-loop: ppT truncf, tmem_alloc → computation partition ---
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttng.tmem_alloc {{.*}}ttg.partition = array<i32: [[COMP]]>
// --- In-loop: dO memdesc_trans → gemm partition ---
// CHECK: ttg.memdesc_trans {{.*}}ttg.partition = array<i32: [[GEMM]]>
// --- In-loop: dpT MMA, dv MMA → gemm partition ---
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[GEMM]]>
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[GEMM]]>
// --- In-loop: dpT tmem_load, dsT computation → computation partition ---
// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.subf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[COMP]]>
// --- In-loop: dsT memdesc_trans → gemm partition ---
// CHECK: ttg.memdesc_trans {{.*}}ttg.partition = array<i32: [[GEMM]]>
// --- In-loop: dq MMA, dk MMA → gemm partition ---
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[GEMM]]>
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[GEMM]]>
// --- In-loop: dq tmem_load, reshape/split → reduction partition ---
// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.reshape {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.trans {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.split {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.reshape {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.trans {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.split {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.reshape {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.trans {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.split {{.*}}ttg.partition = array<i32: [[RED]]>
// --- In-loop: dq descriptor_reduce (×4) → reduction partition ---
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.descriptor_reduce {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.descriptor_reduce {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.descriptor_reduce {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[RED]]>
// CHECK: tt.descriptor_reduce {{.*}}ttg.partition = array<i32: [[RED]]>
//
// --- Post-loop: dv tmem_load, reshape/split → computation partition (via mergeEpilogueToComputation) ---
// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.reshape {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.trans {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.split {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.reshape {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.trans {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.split {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.reshape {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.trans {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.split {{.*}}ttg.partition = array<i32: [[COMP]]>
// --- Post-loop: dv truncf, convert, descriptor_store (×4) → computation partition (via mergeEpilogueToComputation) ---
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.descriptor_store {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.descriptor_store {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.descriptor_store {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.descriptor_store {{.*}}ttg.partition = array<i32: [[COMP]]>
// --- Post-loop: dk tmem_load, reshape/split → computation partition (via mergeEpilogueToComputation) ---
// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.reshape {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.trans {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.split {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.reshape {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.trans {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.split {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.reshape {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.trans {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.split {{.*}}ttg.partition = array<i32: [[COMP]]>
// --- Post-loop: dk mulf, truncf, convert, descriptor_store (×4) → computation partition (via mergeEpilogueToComputation) ---
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.descriptor_store {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.descriptor_store {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.descriptor_store {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: arith.truncf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.descriptor_store {{.*}}ttg.partition = array<i32: [[COMP]]>
//
// --- Partition types ---
// CHECK: tt.warp_specialize
// CHECK-SAME: ttg.partition.types = ["reduction", "gemm", "load", "computation"]

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 2, 32], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 32, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 192 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd_persist(%desc_q: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %sm_scale: f32, %desc_do: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_do_12: i32, %desc_do_13: i32, %desc_do_14: i64, %desc_do_15: i64, %desc_dq: !tt.tensordesc<tensor<128x32xf32, #shared1>>, %desc_dq_16: i32, %desc_dq_17: i32, %desc_dq_18: i64, %desc_dq_19: i64, %desc_dk: !tt.tensordesc<tensor<128x32xf16, #shared2>>, %desc_dk_20: i32, %desc_dk_21: i32, %desc_dk_22: i64, %desc_dk_23: i64, %desc_dv: !tt.tensordesc<tensor<128x32xf16, #shared2>>, %desc_dv_24: i32, %desc_dv_25: i32, %desc_dv_26: i64, %desc_dv_27: i64, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %D: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %stride_z: i32 {tt.divisibility = 16 : i32}, %stride_h: i32 {tt.divisibility = 16 : i32}, %stride_tok: i32 {tt.divisibility = 16 : i32}, %BATCH: i32, %H: i32 {tt.divisibility = 16 : i32}, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %n_tile_num = arith.constant 127 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c96_i32 = arith.constant 96 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_28 = arith.constant dense<0.693147182> : tensor<128x32xf32, #blocked1>
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
    %num_steps = arith.divsi %N_CTX, %c128_i32 : i32
    %offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %dkN = tt.splat %sm_scale : f32 -> tensor<128x32xf32, #blocked1>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_32 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_32, %n_tile_num_30 : i32
      %bhid = arith.divsi %tile_idx_32, %n_tile_num_30 : i32
      %off_chz = arith.muli %bhid, %N_CTX : i32
      %off_chz_33 = arith.extsi %off_chz : i32 to i64
      %off_bh_34 = arith.remsi %bhid, %H : i32
      %off_bh_35 = arith.muli %stride_h, %off_bh_34 : i32
      %off_bh_36 = arith.divsi %bhid, %H : i32
      %off_bh_37 = arith.muli %stride_z, %off_bh_36 : i32
      %off_bh_38 = arith.addi %off_bh_35, %off_bh_37 : i32
      %off_bh_39 = arith.extsi %off_bh_38 : i32 to i64
      %off_bh_40 = arith.divsi %off_bh_39, %off_bh : i64
      %M_41 = tt.addptr %M, %off_chz_33 : !tt.ptr<f32>, i64
      %D_42 = tt.addptr %D, %off_chz_33 : !tt.ptr<f32>, i64
      %start_n = arith.muli %pid, %c128_i32 : i32
      %k = arith.extsi %start_n : i32 to i64
      %k_43 = arith.addi %off_bh_40, %k : i64
      %k_44 = arith.trunci %k_43 : i64 to i32
      %k_45 = tt.descriptor_load %desc_k[%k_44, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
      %k_46 = ttg.local_alloc %k_45 : (tensor<128x128xf16, #blocked3>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %v = tt.descriptor_load %desc_v[%k_44, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
      %v_47 = ttg.local_alloc %v : (tensor<128x128xf16, #blocked3>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %m = tt.splat %M_41 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %Di = tt.splat %D_42 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %qkT, %qkT_48 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dpT, %dpT_49 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dv, %dv_50 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dq, %dq_51 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dk, %dk_52 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dk_53 = ttng.tmem_store %cst, %dk[%dk_52], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %dv_54 = ttng.tmem_store %cst, %dv[%dv_50], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %curr_m:7 = scf.for %curr_m_86 = %c0_i32 to %num_steps step %c1_i32 iter_args(%arg47 = %c0_i32, %arg48 = %false, %qkT_87 = %qkT_48, %dpT_88 = %dpT_49, %dv_89 = %dv_54, %dq_90 = %dq_51, %dk_91 = %dk_53) -> (i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %q = arith.extsi %arg47 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 to i64
        %q_92 = arith.addi %off_bh_40, %q {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %q_93 = arith.trunci %q_92 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64 to i32
        %q_94 = tt.descriptor_load %desc_q[%q_93, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
        %q_95 = ttg.local_alloc %q_94 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<128x128xf16, #blocked3>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %qT = ttg.memdesc_trans %q_95 {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared3, #smem>
        %offs_m_96 = tt.splat %arg47 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, #blocked2>
        %offs_m_97 = arith.addi %offs_m_96, %offs_m {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128xi32, #blocked2>
        %m_98 = tt.addptr %m, %offs_m_97 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %m_99 = tt.load %m_98 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
        %qkT_100 = ttng.tc_gen5_mma %k_46, %qT, %qkT[%qkT_87], %false, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndA,smem,1,0\22, \22opndB,smem,2,1\22, \22opndD,tmem,1,2\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared3, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %pT = ttg.convert_layout %m_99 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %pT_101 = tt.expand_dims %pT {axis = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xf32, #blocked>
        %pT_102 = tt.broadcast %pT_101 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<1x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %qkT_103, %qkT_104 = ttng.tmem_load %qkT[%qkT_100] {loop.cluster = 4 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %pT_105 = arith.subf %qkT_103, %pT_102 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked>
        %pT_106 = math.exp2 %pT_105 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked>
        %do = tt.descriptor_load %desc_do[%q_93, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked3>
        %do_107 = ttg.local_alloc %do {loop.cluster = 4 : i32, loop.stage = 0 : i32} : (tensor<128x128xf16, #blocked3>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %ppT = arith.truncf %pT_106 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %dv_108 = ttng.tmem_alloc %ppT {loop.cluster = 4 : i32, loop.stage = 0 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
        %dpT_109 = ttg.memdesc_trans %do_107 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared3, #smem>
        %dpT_110 = ttng.tc_gen5_mma %v_47, %dpT_109, %dpT[%dpT_88], %false, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,smem,1,3\22, \22opndB,smem,1,4\22, \22opndD,tmem,1,5\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared3, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %Di_111 = tt.addptr %Di, %offs_m_97 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
        %Di_112 = tt.load %Di_111 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x!tt.ptr<f32>, #blocked2>
        %dv_113 = ttng.tc_gen5_mma %dv_108, %do_107, %dv[%dv_89], %arg48, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dsT = ttg.convert_layout %Di_112 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128xf32, #blocked2> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %dsT_114 = tt.expand_dims %dsT {axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xf32, #blocked>
        %dsT_115 = tt.broadcast %dsT_114 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<1x128xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %dpT_116, %dpT_117 = ttng.tmem_load %dpT[%dpT_110] {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %dsT_118 = arith.subf %dpT_116, %dsT_115 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %dsT_119 = arith.mulf %pT_106, %dsT_118 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked>
        %dsT_120 = arith.truncf %dsT_119 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %dsT_121 = ttg.local_alloc %dsT_120 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %dq_122 = ttg.memdesc_trans %dsT_121 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared3, #smem>
        %dq_123 = ttng.tc_gen5_mma %dq_122, %k_46, %dq[%dq_90], %false, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,5\22]}"} : !ttg.memdesc<128x128xf16, #shared3, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dk_124 = ttng.tc_gen5_mma %dsT_121, %q_95, %dk[%dk_91], %arg48, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndD,tmem,1,10\22]}", tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %dq_125, %dq_126 = ttng.tmem_load %dq[%dq_123] {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %dqs = tt.reshape %dq_125 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked4>
        %dqs_127 = tt.trans %dqs {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
        %dqs_128, %dqs_129 = tt.split %dqs_127 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
        %dqs_130 = tt.reshape %dqs_128 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
        %dqs_131 = tt.trans %dqs_130 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
        %dqs_132, %dqs_133 = tt.split %dqs_131 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
        %dqs_134 = tt.reshape %dqs_129 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
        %dqs_135 = tt.trans %dqs_134 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
        %dqs_136, %dqs_137 = tt.split %dqs_135 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
        %dqN = arith.mulf %dqs_132, %cst_28 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1>
        %dqN_138 = ttg.convert_layout %dqN {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq[%q_93, %c0_i32], %dqN_138 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
        %dqN_139 = arith.mulf %dqs_133, %cst_28 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1>
        %dqN_140 = ttg.convert_layout %dqN_139 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq[%q_93, %c32_i32], %dqN_140 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
        %dqN_141 = arith.mulf %dqs_136, %cst_28 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1>
        %dqN_142 = ttg.convert_layout %dqN_141 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq[%q_93, %c64_i32], %dqN_142 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
        %dqN_143 = arith.mulf %dqs_137, %cst_28 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1>
        %dqN_144 = ttg.convert_layout %dqN_143 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked9>
        tt.descriptor_reduce add, %desc_dq[%q_93, %c96_i32], %dqN_144 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !tt.tensordesc<tensor<128x32xf32, #shared1>>, tensor<128x32xf32, #blocked9>
        %curr_m_145 = arith.addi %arg47, %c128_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
        scf.yield %curr_m_145, %true, %qkT_104, %dpT_117, %dv_113, %dq_126, %dk_124 : i32, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {tt.scheduled_max_stage = 1 : i32}
      %dv_55, %dv_56 = ttng.tmem_load %dv[%curr_m#4] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %dvs = tt.reshape %dv_55 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked4>
      %dvs_57 = tt.trans %dvs {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %dvs_58, %dvs_59 = tt.split %dvs_57 : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
      %dvs_60 = tt.reshape %dvs_58 : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dvs_61 = tt.trans %dvs_60 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dvs_62, %dvs_63 = tt.split %dvs_61 : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
      %dvs_64 = tt.reshape %dvs_59 : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dvs_65 = tt.trans %dvs_64 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dvs_66, %dvs_67 = tt.split %dvs_65 : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
      %3 = arith.truncf %dvs_62 : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %4 = ttg.convert_layout %3 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dv[%k_44, %c0_i32], %4 : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %5 = arith.truncf %dvs_63 : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %6 = ttg.convert_layout %5 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dv[%k_44, %c32_i32], %6 : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %7 = arith.truncf %dvs_66 : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %8 = ttg.convert_layout %7 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dv[%k_44, %c64_i32], %8 : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %9 = arith.truncf %dvs_67 : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %10 = ttg.convert_layout %9 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dv[%k_44, %c96_i32], %10 : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %dk_68, %dk_69 = ttng.tmem_load %dk[%curr_m#6] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %dks = tt.reshape %dk_68 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked4>
      %dks_70 = tt.trans %dks {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %dks_71, %dks_72 = tt.split %dks_70 : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked6>
      %dks_73 = tt.reshape %dks_71 : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dks_74 = tt.trans %dks_73 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dks_75, %dks_76 = tt.split %dks_74 : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
      %dks_77 = tt.reshape %dks_72 : tensor<128x64xf32, #blocked6> -> tensor<128x2x32xf32, #blocked7>
      %dks_78 = tt.trans %dks_77 {order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #blocked7> -> tensor<128x32x2xf32, #blocked8>
      %dks_79, %dks_80 = tt.split %dks_78 : tensor<128x32x2xf32, #blocked8> -> tensor<128x32xf32, #blocked1>
      %dkN_81 = arith.mulf %dks_75, %dkN : tensor<128x32xf32, #blocked1>
      %11 = arith.truncf %dkN_81 : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %12 = ttg.convert_layout %11 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dk[%k_44, %c0_i32], %12 : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %dkN_82 = arith.mulf %dks_76, %dkN : tensor<128x32xf32, #blocked1>
      %13 = arith.truncf %dkN_82 : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %14 = ttg.convert_layout %13 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dk[%k_44, %c32_i32], %14 : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %dkN_83 = arith.mulf %dks_79, %dkN : tensor<128x32xf32, #blocked1>
      %15 = arith.truncf %dkN_83 : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %16 = ttg.convert_layout %15 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dk[%k_44, %c64_i32], %16 : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %dkN_84 = arith.mulf %dks_80, %dkN : tensor<128x32xf32, #blocked1>
      %17 = arith.truncf %dkN_84 : tensor<128x32xf32, #blocked1> to tensor<128x32xf16, #blocked1>
      %18 = ttg.convert_layout %17 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blocked10>
      tt.descriptor_store %desc_dk[%k_44, %c96_i32], %18 : !tt.tensordesc<tensor<128x32xf16, #shared2>>, tensor<128x32xf16, #blocked10>
      %tile_idx_85 = arith.addi %tile_idx_32, %num_progs : i32
      scf.yield %tile_idx_85 : i32
    } {tt.merge_epilogue = true, tt.smem_alloc_algo = 1 : i32, tt.smem_budget = 200000 : i32, tt.tmem_alloc_algo = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
