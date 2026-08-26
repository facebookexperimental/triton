// RUN: triton-opt %s --nvgpu-analyze-2cta-dependencies | FileCheck %s

// Classification of dependent 2-CTA MMAs, reduced from the BM128 persistent
// backward kernel to just the operand chains the pass inspects.
//
// The pass annotates a two_ctas MMA only when its A operand traces back to
// another two_ctas MMA. It then splits on whether that chain contains a
// rank-two transpose: a plain contraction stays collective, while the
// transposed dQ-style operand needs a peer gather because the value is an MMA
// result already distributed across the CTA pair and cannot be re-loaded.
//
// %root is not annotated: its A comes straight from a block argument.

// CHECK-DAG: ttng.tc_gen5_mma {{.*}}ttng.two_cta_dependency = "collective_contraction"
// CHECK-DAG: ttng.tc_gen5_mma {{.*}}ttng.two_cta_dependency = "collective_contraction"
// CHECK-DAG: ttng.tc_gen5_mma {{.*}}ttng.two_cta_dependency = "requires_peer_gather"{{.*}}!ttg.memdesc<64x256xf16{{.*}}!ttg.memdesc<256x128xf16{{.*}}!ttg.memdesc<64x128xf32

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 128], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, twoCTAs = true, ctaMode = twocta_rhs>

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @two_cta_dependency_kinds(
      %a: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %bT: !ttg.memdesc<128x128xf16, #shared1, #smem>,
      %kt: !ttg.memdesc<256x128xf16, #shared, #smem>) attributes {noinline = false} {
    %true = arith.constant true
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear1>
    %zero_dq = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #linear>

    // Root collective MMA. Its A operand is a block argument, so it is left
    // unannotated and only serves as the producer the dependents trace back to.
    %acc, %acc_tok = ttng.tmem_alloc %zero : (tensor<128x128xf32, #linear1>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %root = ttng.tc_gen5_mma %a, %bT, %acc[%acc_tok], %true, %true {two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %rootv, %rootv_tok = ttng.tmem_load %acc[%root] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
    %p = arith.truncf %rootv : tensor<128x128xf32, #linear1> to tensor<128x128xf16, #linear1>
    %p_smem = ttg.local_alloc %p : (tensor<128x128xf16, #linear1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>

    // Two dependents whose A chain has no rank-two transpose.
    %acc_dv, %acc_dv_tok = ttng.tmem_alloc %zero : (tensor<128x128xf32, #linear1>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dv = ttng.tc_gen5_mma %p_smem, %bT, %acc_dv[%acc_dv_tok], %true, %true {two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    %acc_dk, %acc_dk_tok = ttng.tmem_alloc %zero : (tensor<128x128xf32, #linear1>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dk = ttng.tc_gen5_mma %p_smem, %bT, %acc_dk[%acc_dk_tok], %true, %true {two_ctas} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // dQ-style dependent: the A chain carries the rank-two transpose plus the
    // reshape into TLX's physical [M/2, 2*N] operand layout.
    %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<128x128xf16, #linear1> -> tensor<128x128xf16, #linear3>
    %pT_packed = tt.reshape %pT : tensor<128x128xf16, #linear3> -> tensor<64x256xf16, #linear4>
    %pT_smem = ttg.local_alloc %pT_packed : (tensor<64x256xf16, #linear4>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
    %acc_dq, %acc_dq_tok = ttng.tmem_alloc %zero_dq : (tensor<64x128xf32, #linear>) -> (!ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %dq = ttng.tc_gen5_mma %pT_smem, %kt, %acc_dq[%acc_dq_tok], %true, %true {two_ctas} : !ttg.memdesc<64x256xf16, #shared, #smem>, !ttg.memdesc<256x128xf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #tmem1, #ttng.tensor_memory, mutable>
    tt.return
  }
}
