// RUN: not triton-opt %s --nvgpu-2cta-transform-loads 2>&1 | FileCheck %s

// Retyping the shared descriptor argument would break the second load, so B
// cannot be split; that must be an error, not a silent full-width B.
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
