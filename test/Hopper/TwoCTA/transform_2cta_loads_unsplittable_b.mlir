// RUN: not triton-opt %s -split-input-file --nvgpu-2cta-transform-loads 2>&1 | FileCheck %s

// A 2-CTA MMA reads half of B (along N) from each CTA of the pair. When the
// pass cannot give each CTA its half, compiling on with the full-width B
// silently computes with the wrong half, so it must be a hard error.

// BLOCK_N = 16: the per-CTA half (8) is below the minimum MMA N.
// CHECK: error: two_ctas dot: cannot split the B operand across the CTA pair

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 16, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @k(%A: !tt.ptr<bf16>, %B: !tt.ptr<bf16>, %B2: !tt.ptr<f32>, %E1: !tt.ptr<f32>, %C: !tt.ptr<f32>, %M: i32, %N: i32, %K: i32) attributes {noinline = false} {
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %num_pid_m = arith.constant 127 : i32
    %num_pid_n = arith.constant 15 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #linear>
    %a_desc = arith.extsi %K : i32 to i64
    %a_desc_0 = tt.make_tensor_descriptor %A, [%M, %K], [%a_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16>
    %b_desc = arith.extsi %N : i32 to i64
    %b_desc_1 = tt.make_tensor_descriptor %B, [%K, %N], [%b_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<64x16xbf16>
    %c_desc = tt.make_tensor_descriptor %C, [%M, %N], [%b_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x16xf32>
    %num_pid_m_2 = arith.addi %M, %num_pid_m : i32
    %num_pid_m_3 = arith.divsi %num_pid_m_2, %c128_i32 : i32
    %num_pid_n_4 = arith.addi %N, %num_pid_n : i32
    %num_pid_n_5 = arith.divsi %num_pid_n_4, %c16_i32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %num_pid_m_3, %num_pid_n_5 : i32
    %2 = arith.addi %K, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    scf.for %t = %0 to %1 step %c148_i32  : i32 {
      %pid_m = arith.remsi %t, %num_pid_m_3 : i32
      %pid_n = arith.divsi %t, %num_pid_m_3 : i32
      %a = arith.muli %pid_m, %c128_i32 : i32
      %b = arith.muli %pid_n, %c16_i32 : i32
      %acc = scf.for %ki = %c0_i32 to %3 step %c1_i32 iter_args(%acc_6 = %cst) -> (tensor<128x16xf32, #linear>)  : i32 {
        %a_7 = arith.muli %ki, %c64_i32 : i32
        %a_8 = tt.descriptor_load %a_desc_0[%a, %a_7] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
        %a_9 = ttg.local_alloc %a_8 : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %b_10 = tt.descriptor_load %b_desc_1[%a_7, %b] : !tt.tensordesc<64x16xbf16> -> tensor<64x16xbf16, #blocked1>
        %b_11 = ttg.local_alloc %b_10 : (tensor<64x16xbf16, #blocked1>) -> !ttg.memdesc<64x16xbf16, #shared1, #smem>
        %acc_12, %acc_13 = ttng.tmem_alloc %acc_6 : (tensor<128x16xf32, #linear>) -> (!ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %acc_14 = ttng.tc_gen5_mma %a_9, %b_11, %acc_12[%acc_13], %true, %true {two_ctas} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x16xbf16, #shared1, #smem>, !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_15, %acc_16 = ttng.tmem_load %acc_12[%acc_14] : !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x16xf32, #linear>
        scf.yield %acc_15 : tensor<128x16xf32, #linear>
      }
      %4 = arith.muli %pid_m, %c128_i32 : i32
      %5 = arith.muli %pid_n, %c16_i32 : i32
      %6 = ttg.convert_layout %acc : tensor<128x16xf32, #linear> -> tensor<128x16xf32, #blocked2>
      tt.descriptor_store %c_desc[%4, %5], %6 : !tt.tensordesc<128x16xf32>, tensor<128x16xf32, #blocked2>
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

// B loaded through pointers, not a descriptor.
// CHECK: error: two_ctas dot: cannot split the B operand across the CTA pair

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @k(%A: !tt.ptr<bf16>, %B: !tt.ptr<bf16>, %B2: !tt.ptr<f32>, %E1: !tt.ptr<f32>, %C: !tt.ptr<f32>, %M: i32, %N: i32, %K: i32) attributes {noinline = false} {
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %a_desc = arith.extsi %K : i32 to i64
    %a_desc_0 = tt.make_tensor_descriptor %A, [%M, %K], [%a_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16>
    %c_desc = arith.extsi %N : i32 to i64
    %c_desc_1 = tt.make_tensor_descriptor %C, [%M, %N], [%c_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x128xf32>
    %num_pid_m = arith.addi %M, %c127_i32 : i32
    %num_pid_m_2 = arith.divsi %num_pid_m, %c128_i32 : i32
    %num_pid_n = arith.addi %N, %c127_i32 : i32
    %num_pid_n_3 = arith.divsi %num_pid_n, %c128_i32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %num_pid_m_2, %num_pid_n_3 : i32
    %2 = arith.addi %K, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    %rk = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %b = tt.splat %N : i32 -> tensor<64x1xi32, #blocked>
    %b_4 = tt.splat %B : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
    scf.for %t = %0 to %1 step %c148_i32  : i32 {
      %pid_m = arith.remsi %t, %num_pid_m_2 : i32
      %pid_n = arith.divsi %t, %num_pid_m_2 : i32
      %a = arith.muli %pid_m, %c128_i32 : i32
      %rn_5 = arith.muli %pid_n, %c128_i32 : i32
      %rn_6 = tt.splat %rn_5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %rn_7 = arith.addi %rn_6, %rn : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %b_8 = tt.expand_dims %rn_7 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %b_9 = tt.broadcast %b_8 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
      %acc = scf.for %ki = %c0_i32 to %3 step %c1_i32 iter_args(%acc_10 = %cst) -> (tensor<128x128xf32, #linear>)  : i32 {
        %a_11 = arith.muli %ki, %c64_i32 : i32
        %a_12 = tt.descriptor_load %a_desc_0[%a, %a_11] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked1>
        %a_13 = ttg.local_alloc %a_12 : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %rk_14 = tt.splat %a_11 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %rk_15 = arith.addi %rk_14, %rk : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %b_16 = tt.expand_dims %rk_15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
        %b_17 = arith.muli %b_16, %b : tensor<64x1xi32, #blocked>
        %b_18 = tt.addptr %b_4, %b_17 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
        %b_19 = tt.broadcast %b_18 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
        %b_20 = tt.addptr %b_19, %b_9 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
        %b_21 = tt.load %b_20 : tensor<64x128x!tt.ptr<bf16>, #blocked>
        %b_22 = ttg.local_alloc %b_21 : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %acc_23, %acc_24 = ttng.tmem_alloc %acc_10 : (tensor<128x128xf32, #linear>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %acc_25 = ttng.tc_gen5_mma %a_13, %b_22, %acc_23[%acc_24], %true, %true {two_ctas} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_26, %acc_27 = ttng.tmem_load %acc_23[%acc_25] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        scf.yield %acc_26 : tensor<128x128xf32, #linear>
      }
      %4 = arith.muli %pid_m, %c128_i32 : i32
      %5 = arith.muli %pid_n, %c128_i32 : i32
      %6 = ttg.convert_layout %acc : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked2>
      tt.descriptor_store %c_desc_1[%4, %5], %6 : !tt.tensordesc<128x128xf32>, tensor<128x128xf32, #blocked2>
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

// A source-level split of one V tile whose descriptor argument also feeds an
// unrelated load: retyping the argument in place would hand that load a
// half-width descriptor, so the chain cannot be split.
// CHECK: error: two_ctas dot: cannot split the B operand across the CTA pair

#v2_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#v2_reshape = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#v2_trans = #ttg.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [2, 16, 1], warpsPerCTA = [4, 1, 1], order = [1, 0, 2]}>
#v2_leaf = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [8, 0], [16, 0], [32, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0]], warp = [[2, 0], [4, 0]], block = []}>
#v2_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#v2_smem = #ttg.shared_memory
#v2_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fa_split_v_load_shared_desc(
      %v_desc: !tt.tensordesc<128x128xbf16>,
      %a0: !ttg.memdesc<256x64xbf16, #v2_shared, #v2_smem>,
      %a1: !ttg.memdesc<256x64xbf16, #v2_shared, #v2_smem>,
      %acc0: !ttg.memdesc<256x128xf32, #v2_tmem, #ttng.tensor_memory, mutable>,
      %acc1: !ttg.memdesc<256x128xf32, #v2_tmem, #ttng.tensor_memory, mutable>,
      %token0: !ttg.async.token,
      %token1: !ttg.async.token) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32

    %v = tt.descriptor_load %v_desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xbf16> -> tensor<128x128xbf16, #v2_blocked>
    %v_reshape_value = tt.reshape %v : tensor<128x128xbf16, #v2_blocked> -> tensor<2x64x128xbf16, #v2_reshape>
    %v_trans_value = tt.trans %v_reshape_value {order = array<i32: 1, 2, 0>} : tensor<2x64x128xbf16, #v2_reshape> -> tensor<64x128x2xbf16, #v2_trans>
    %v0, %v1 = tt.split %v_trans_value : tensor<64x128x2xbf16, #v2_trans> -> tensor<64x128xbf16, #v2_leaf>
    %v0_smem = ttg.local_alloc %v0 : (tensor<64x128xbf16, #v2_leaf>) -> !ttg.memdesc<64x128xbf16, #v2_shared, #v2_smem>
    %v1_smem = ttg.local_alloc %v1 : (tensor<64x128xbf16, #v2_leaf>) -> !ttg.memdesc<64x128xbf16, #v2_shared, #v2_smem>

    // Second consumer of the same descriptor.
    %v_other = tt.descriptor_load %v_desc[%c128_i32, %c0_i32] : !tt.tensordesc<128x128xbf16> -> tensor<128x128xbf16, #v2_blocked>
    %v_other_smem = ttg.local_alloc %v_other : (tensor<128x128xbf16, #v2_blocked>) -> !ttg.memdesc<128x128xbf16, #v2_shared, #v2_smem>

    %mma0 = ttng.tc_gen5_mma %a0, %v0_smem, %acc0[%token0], %true, %true {two_ctas} : !ttg.memdesc<256x64xbf16, #v2_shared, #v2_smem>, !ttg.memdesc<64x128xbf16, #v2_shared, #v2_smem>, !ttg.memdesc<256x128xf32, #v2_tmem, #ttng.tensor_memory, mutable>
    %mma1 = ttng.tc_gen5_mma %a1, %v1_smem, %acc1[%token1], %true, %true {two_ctas} : !ttg.memdesc<256x64xbf16, #v2_shared, #v2_smem>, !ttg.memdesc<64x128xbf16, #v2_shared, #v2_smem>, !ttg.memdesc<256x128xf32, #v2_tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// B built in registers (fp32 tile scaled by a vector, then cast) is not a
// descriptor load the pass can split.
// CHECK: error: two_ctas dot: cannot split the B operand across the CTA pair

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @k(%A: !tt.ptr<bf16>, %B: !tt.ptr<f32>, %B2: !tt.ptr<f32>, %E1: !tt.ptr<f32>, %C: !tt.ptr<f32>, %M: i32, %N: i32, %K: i32) attributes {noinline = false} {
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %a_desc = arith.extsi %K : i32 to i64
    %a_desc_0 = tt.make_tensor_descriptor %A, [%M, %K], [%a_desc, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<128x64xbf16>
    %b_desc = arith.extsi %N : i32 to i64
    %b_desc_1 = tt.make_tensor_descriptor %B, [%K, %N], [%b_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
    %e1_desc = tt.make_tensor_descriptor %E1, [%K], [%c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64xf32>
    %c_desc = tt.make_tensor_descriptor %C, [%M, %N], [%b_desc, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<128x128xf32>
    %num_pid_m = arith.addi %M, %c127_i32 : i32
    %num_pid_m_2 = arith.divsi %num_pid_m, %c128_i32 : i32
    %num_pid_n = arith.addi %N, %c127_i32 : i32
    %num_pid_n_3 = arith.divsi %num_pid_n, %c128_i32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %num_pid_m_2, %num_pid_n_3 : i32
    %2 = arith.addi %K, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    scf.for %t = %0 to %1 step %c148_i32  : i32 {
      %pid_m = arith.remsi %t, %num_pid_m_2 : i32
      %pid_n = arith.divsi %t, %num_pid_m_2 : i32
      %a = arith.muli %pid_m, %c128_i32 : i32
      %b = arith.muli %pid_n, %c128_i32 : i32
      %acc = scf.for %ki = %c0_i32 to %3 step %c1_i32 iter_args(%acc_4 = %cst) -> (tensor<128x128xf32, #linear>)  : i32 {
        %a_5 = arith.muli %ki, %c64_i32 : i32
        %a_6 = tt.descriptor_load %a_desc_0[%a, %a_5] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked>
        %a_7 = ttg.local_alloc %a_6 : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
        %b_8 = tt.descriptor_load %b_desc_1[%a_5, %b] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked1>
        %b_9 = tt.descriptor_load %e1_desc[%a_5] : !tt.tensordesc<64xf32> -> tensor<64xf32, #blocked2>
        %b_10 = ttg.convert_layout %b_9 : tensor<64xf32, #blocked2> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %b_11 = tt.expand_dims %b_10 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xf32, #blocked1>
        %b_12 = tt.broadcast %b_11 : tensor<64x1xf32, #blocked1> -> tensor<64x128xf32, #blocked1>
        %b_13 = arith.mulf %b_8, %b_12 : tensor<64x128xf32, #blocked1>
        %b_14 = arith.truncf %b_13 : tensor<64x128xf32, #blocked1> to tensor<64x128xbf16, #blocked1>
        %b_15 = ttg.local_alloc %b_14 : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %acc_16, %acc_17 = ttng.tmem_alloc %acc_4 : (tensor<128x128xf32, #linear>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
        %acc_18 = ttng.tc_gen5_mma %a_7, %b_15, %acc_16[%acc_17], %true, %true {two_ctas} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %acc_19, %acc_20 = ttng.tmem_load %acc_16[%acc_18] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
        scf.yield %acc_19 : tensor<128x128xf32, #linear>
      }
      %4 = arith.muli %pid_m, %c128_i32 : i32
      %5 = arith.muli %pid_n, %c128_i32 : i32
      %6 = ttg.convert_layout %acc : tensor<128x128xf32, #linear> -> tensor<128x128xf32, #blocked1>
      tt.descriptor_store %c_desc[%4, %5], %6 : !tt.tensordesc<128x128xf32>, tensor<128x128xf32, #blocked1>
    } {tt.warp_specialize}
    tt.return
  }
}
