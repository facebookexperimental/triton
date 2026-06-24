// RUN: triton-opt %s -split-input-file --triton-nvidia-check-matmul-two-cta --verify-diagnostics | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

// CHECK-LABEL: module
// CHECK-SAME: "ttng.two-ctas" = true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @independent_two_cta_mmas(
      %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %a1: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %acc0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc1: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %acc_tok0: !ttg.async.token,
      %acc_tok1: !ttg.async.token) {
    %true = arith.constant true
    %tok0 = ttng.tc_gen5_mma %a0, %b, %acc0[%acc_tok0], %true, %true {two_ctas} :
      !ttg.memdesc<128x64xf16, #shared, #smem>,
      !ttg.memdesc<64x128xf16, #shared1, #smem>,
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %tok1 = ttng.tc_gen5_mma %a1, %b, %acc1[%acc_tok1], %true, %true {two_ctas} :
      !ttg.memdesc<128x64xf16, #shared, #smem>,
      !ttg.memdesc<64x128xf16, #shared1, #smem>,
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @dependent_two_cta_dot_chain(
      %q: tensor<128x64xf16>,
      %k: tensor<64x128xf16>,
      %v: tensor<128x128xf16>,
      %acc: tensor<128x128xf32>) {
    // expected-note @+1 {{producer 2-CTA dot result is consumed by this dot.}}
    %qk = tt.dot %q, %k, %acc {two_ctas} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
    %qk_f16 = arith.truncf %qk : tensor<128x128xf32> to tensor<128x128xf16>
    // expected-error @+1 {{two_ctas=True does not currently support dependent matmul chains}}
    %pv = tt.dot %qk_f16, %v, %acc {two_ctas} : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
    tt.return
  }
}
