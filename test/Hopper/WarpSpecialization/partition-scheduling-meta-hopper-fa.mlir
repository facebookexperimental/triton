// RUN: triton-opt %s --nvgpu-partition-scheduling-meta="merge-correction merge-epilogue-into-computation" | FileCheck %s

// Tests that Hopper FA forward (dpFactor=2, warp_group_dot, mergeCorrection +
// mergeEpilogue) gets 3 partitions: load + computation×2.
//
// Key differences from Blackwell FA:
// - Uses warp_group_dot (not MMAv5/tc_gen5_mma) → no gemm partition
// - mergeCorrection: correction ops → computation[dpId]
// - mergeEpilogue: epilogue ops → computation[dpId]
// - Result: load + comp×2 = 3 partitions

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {

// CHECK-LABEL: @hopper_fa_forward_3_partitions
//
// --- Partition types: load + two computation partitions ---
// CHECK: tt.warp_specialize
// CHECK-SAME: ttg.partition.types =
// CHECK-SAME: "load"
// CHECK-SAME: "computation"
// CHECK-SAME: "computation"

tt.func public @hopper_fa_forward_3_partitions(
  %Q: !tt.ptr<f16> {tt.divisibility = 16 : i32},
  %K: !tt.ptr<f16> {tt.divisibility = 16 : i32},
  %V: !tt.ptr<f16> {tt.divisibility = 16 : i32},
  %Out: !tt.ptr<f16> {tt.divisibility = 16 : i32},
  %stride_qm: i32 {tt.divisibility = 16 : i32},
  %stride_kn: i32 {tt.divisibility = 16 : i32},
  %stride_vn: i32 {tt.divisibility = 16 : i32},
  %stride_om: i32 {tt.divisibility = 16 : i32},
  %Q_LEN: i32 {tt.divisibility = 16 : i32},
  %KV_LEN: i32 {tt.divisibility = 16 : i32},
  %SM_SCALE: f32
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %c128_i64 = arith.constant 128 : i64
  %cst_neg_inf = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  %cst_one = arith.constant dense<1.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  %cst_zero_2d = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #mma>
  %cst_scale = arith.constant dense<1.44269502> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  %cst_scale_2d = arith.constant dense<1.44269502> : tensor<64x128xf32, #mma>
  %n_iters = arith.constant 8 : i32

  // Q descriptor and loads for two data partitions
  %desc_q_stride = arith.extsi %stride_qm : i32 to i64
  %desc_q = tt.make_tensor_descriptor %Q, [%Q_LEN, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<64x128xf16, #shared>>
  %desc_q_2 = tt.make_tensor_descriptor %Q, [%Q_LEN, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<64x128xf16, #shared>>
  %q_0_data = tt.descriptor_load %desc_q[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked>
  %q_1_data = tt.descriptor_load %desc_q_2[%c64_i32, %c0_i32] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked>
  %q_0 = ttg.local_alloc %q_0_data : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
  %q_1 = ttg.local_alloc %q_1_data : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

  // K/V descriptors
  %desc_k = tt.make_tensor_descriptor %K, [%KV_LEN, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
  %desc_v = tt.make_tensor_descriptor %V, [%KV_LEN, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>

  // Output descriptor (TMA store — epilogue)
  %desc_o = tt.make_tensor_descriptor %Out, [%Q_LEN, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<64x128xf16, #shared>>
  %desc_o_2 = tt.make_tensor_descriptor %Out, [%Q_LEN, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<64x128xf16, #shared>>

  // Main attention loop — uses warp_group_dot (Hopper MMA, not MMAv5)
  %loop:6 = scf.for %i = %c0_i32 to %n_iters step %c1_i32
      iter_args(
        %acc_0 = %cst_zero_2d, %l_i_0 = %cst_one, %m_i_0 = %cst_neg_inf,
        %acc_1 = %cst_zero_2d, %l_i_1 = %cst_one, %m_i_1 = %cst_neg_inf
      ) -> (
        tensor<64x128xf32, #mma>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>,
        tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>,
        tensor<64x128xf32, #mma>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>,
        tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      ) : i32 {

    // Load K and V
    %kv_offset = arith.muli %i, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
    %k_data = tt.descriptor_load %desc_k[%kv_offset, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked>
    %v_data = tt.descriptor_load %desc_v[%kv_offset, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked>
    %k_smem = ttg.local_alloc %k_data {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %k_trans = ttg.memdesc_trans %k_smem {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared1, #smem>
    %v_smem = ttg.local_alloc %v_data {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>

    // QK warp_group_dot for both data partitions (Hopper MMA)
    %qk_0 = ttng.warp_group_dot %q_0, %k_trans, %cst_zero_2d {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x128xf16, #shared, #smem> * !ttg.memdesc<128x128xf16, #shared1, #smem> -> tensor<64x128xf32, #mma>
    %qk_1 = ttng.warp_group_dot %q_1, %k_trans, %cst_zero_2d {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x128xf16, #shared, #smem> * !ttg.memdesc<128x128xf16, #shared1, #smem> -> tensor<64x128xf32, #mma>

    // Online softmax
    %m_ij_0 = "tt.reduce"(%qk_0) <{axis = 1 : i32}> ({
    ^bb0(%a0: f32, %b0: f32):
      %max0 = arith.maxnumf %a0, %b0 : f32
      tt.reduce.return %max0 : f32
    }) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<64x128xf32, #mma>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %m_ij_1 = "tt.reduce"(%qk_1) <{axis = 1 : i32}> ({
    ^bb0(%a1: f32, %b1: f32):
      %max1 = arith.maxnumf %a1, %b1 : f32
      tt.reduce.return %max1 : f32
    }) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<64x128xf32, #mma>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>

    %m_scaled_0 = arith.mulf %m_ij_0, %cst_scale {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %m_scaled_1 = arith.mulf %m_ij_1, %cst_scale {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %new_m_0 = arith.maxnumf %m_i_0, %m_scaled_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %new_m_1 = arith.maxnumf %m_i_1, %m_scaled_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>

    // Scale QK and compute p
    %scores_0 = arith.mulf %qk_0, %cst_scale_2d {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #mma>
    %scores_1 = arith.mulf %qk_1, %cst_scale_2d {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #mma>
    %m_bcast_0 = tt.expand_dims %new_m_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xf32, #mma>
    %m_bcast2d_0 = tt.broadcast %m_bcast_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x1xf32, #mma> -> tensor<64x128xf32, #mma>
    %m_bcast_1 = tt.expand_dims %new_m_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32, axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xf32, #mma>
    %m_bcast2d_1 = tt.broadcast %m_bcast_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x1xf32, #mma> -> tensor<64x128xf32, #mma>
    %p_sub_0 = arith.subf %scores_0, %m_bcast2d_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #mma>
    %p_sub_1 = arith.subf %scores_1, %m_bcast2d_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #mma>
    %p_0 = math.exp2 %p_sub_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #mma>
    %p_1 = math.exp2 %p_sub_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #mma>

    // alpha = exp2(m_i - new_m)
    %alpha_0 = arith.subf %m_i_0, %new_m_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %alpha_1 = arith.subf %m_i_1, %new_m_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %alpha_exp_0 = math.exp2 %alpha_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %alpha_exp_1 = math.exp2 %alpha_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>

    // Rescale acc
    %alpha_1d_0 = tt.expand_dims %alpha_exp_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xf32, #mma>
    %alpha_2d_0 = tt.broadcast %alpha_1d_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x1xf32, #mma> -> tensor<64x128xf32, #mma>
    %alpha_1d_1 = tt.expand_dims %alpha_exp_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32, axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xf32, #mma>
    %alpha_2d_1 = tt.broadcast %alpha_1d_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x1xf32, #mma> -> tensor<64x128xf32, #mma>
    %acc_scaled_0 = arith.mulf %acc_0, %alpha_2d_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #mma>
    %acc_scaled_1 = arith.mulf %acc_1, %alpha_2d_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #mma>

    // p → f16 for PV dot
    %p_f16_0 = arith.truncf %p_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
    %p_f16_1 = arith.truncf %p_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
    %p_dot_0 = ttg.convert_layout %p_f16_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %p_dot_1 = ttg.convert_layout %p_f16_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>

    // PV warp_group_dot
    %pv_0 = ttng.warp_group_dot %p_dot_0, %v_smem, %acc_scaled_0 {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>
    %pv_1 = ttng.warp_group_dot %p_dot_1, %v_smem, %acc_scaled_1 {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<64x128xf32, #mma>

    // l_i update
    %l_ij_0 = "tt.reduce"(%p_0) <{axis = 1 : i32}> ({
    ^bb0(%a2: f32, %b2: f32):
      %s0 = arith.addf %a2, %b2 : f32
      tt.reduce.return %s0 : f32
    }) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<64x128xf32, #mma>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %l_ij_1 = "tt.reduce"(%p_1) <{axis = 1 : i32}> ({
    ^bb0(%a3: f32, %b3: f32):
      %s1 = arith.addf %a3, %b3 : f32
      tt.reduce.return %s1 : f32
    }) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<64x128xf32, #mma>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %l_scaled_0 = arith.mulf %l_i_0, %alpha_exp_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %l_scaled_1 = arith.mulf %l_i_1, %alpha_exp_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %new_l_0 = arith.addf %l_scaled_0, %l_ij_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %new_l_1 = arith.addf %l_scaled_1, %l_ij_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>

    scf.yield %pv_0, %new_l_0, %new_m_0, %pv_1, %new_l_1, %new_m_1
      : tensor<64x128xf32, #mma>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>,
        tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>,
        tensor<64x128xf32, #mma>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>,
        tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
  } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}

  // Post-loop: normalize and store with descriptor_store (epilogue)
  %l_bcast_0 = tt.expand_dims %loop#1 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xf32, #mma>
  %l_bcast2d_0 = tt.broadcast %l_bcast_0 : tensor<64x1xf32, #mma> -> tensor<64x128xf32, #mma>
  %l_bcast_1 = tt.expand_dims %loop#4 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xf32, #mma>
  %l_bcast2d_1 = tt.broadcast %l_bcast_1 : tensor<64x1xf32, #mma> -> tensor<64x128xf32, #mma>
  %acc_norm_0 = arith.divf %loop#0, %l_bcast2d_0 : tensor<64x128xf32, #mma>
  %acc_norm_1 = arith.divf %loop#3, %l_bcast2d_1 : tensor<64x128xf32, #mma>
  %out_f16_0 = arith.truncf %acc_norm_0 : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
  %out_f16_1 = arith.truncf %acc_norm_1 : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
  %out_conv_0 = ttg.convert_layout %out_f16_0 : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked>
  %out_conv_1 = ttg.convert_layout %out_f16_1 : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked>
  tt.descriptor_store %desc_o[%c0_i32, %c0_i32], %out_conv_0 : !tt.tensordesc<tensor<64x128xf16, #shared>>, tensor<64x128xf16, #blocked>
  tt.descriptor_store %desc_o_2[%c64_i32, %c0_i32], %out_conv_1 : !tt.tensordesc<tensor<64x128xf16, #shared>>, tensor<64x128xf16, #blocked>

  tt.return
}

}
