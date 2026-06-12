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

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @dependent_two_cta_mma_chain(
      %q: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %k: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %v: !ttg.memdesc<128x128xf16, #shared1, #smem>,
      %qk_acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %pv_a: !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>,
      %out_acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %qk_tok: !ttg.async.token,
      %out_tok: !ttg.async.token) {
    %true = arith.constant true
    // expected-note @+1 {{producer 2-CTA MMA result is consumed by this matmul.}}
    %qk_tok_out = ttng.tc_gen5_mma %q, %k, %qk_acc[%qk_tok], %true, %true {two_ctas} :
      !ttg.memdesc<128x64xf16, #shared, #smem>,
      !ttg.memdesc<64x128xf16, #shared1, #smem>,
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %qk, %qk_load_tok = ttng.tmem_load %qk_acc[%qk_tok_out] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    %qk_bf16 = arith.truncf %qk : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
    ttng.tmem_store %qk_bf16, %pv_a, %true : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{two_ctas=True does not currently support dependent matmul chains}}
    %pv_tok_out = ttng.tc_gen5_mma %pv_a, %v, %out_acc[%out_tok], %true, %true {two_ctas} :
      !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<128x128xf16, #shared1, #smem>,
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32} {
  tt.func @dependent_two_cta_tmem_alloc_operand(
      %q: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %k: !ttg.memdesc<64x128xf16, #shared1, #smem>,
      %v: !ttg.memdesc<128x128xf16, #shared1, #smem>,
      %qk_acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %out_acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %qk_tok: !ttg.async.token,
      %out_tok: !ttg.async.token) {
    %true = arith.constant true
    // expected-note @+1 {{producer 2-CTA MMA result is consumed by this matmul.}}
    %qk_tok_out = ttng.tc_gen5_mma %q, %k, %qk_acc[%qk_tok], %true, %true {two_ctas} :
      !ttg.memdesc<128x64xf16, #shared, #smem>,
      !ttg.memdesc<64x128xf16, #shared1, #smem>,
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %qk, %qk_load_tok = ttng.tmem_load %qk_acc[%qk_tok_out] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    %qk_f16 = arith.truncf %qk : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
    %pv_a = ttng.tmem_alloc %qk_f16 : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
    // expected-error @+1 {{two_ctas=True does not currently support dependent matmul chains}}
    %pv_tok_out = ttng.tc_gen5_mma %pv_a, %v, %out_acc[%out_tok], %true, %true {two_ctas} :
      !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>,
      !ttg.memdesc<128x128xf16, #shared1, #smem>,
      !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
