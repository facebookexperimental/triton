// RUN: not triton-opt %s --nvgpu-2cta-transform-loads 2>&1 | FileCheck %s

// A host descriptor is retyped in place when its B tile is split between two
// CTAs. Reject a truncated B path if another load still needs the original
// descriptor shape.
// CHECK: error: 2-CTA truncated B requires an unshared host descriptor

#src = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dst = #ttg.blocked<{sizePerThread = [64, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @shared_host_descriptor_truncated_b(
      %b_desc: !tt.tensordesc<128x64xf32>,
      %a_smem: !ttg.memdesc<128x64xbf16, #shared_a, #smem>,
      %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %token: !ttg.async.token) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32

    %b = tt.descriptor_load %b_desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x64xf32> -> tensor<128x64xf32, #src>
    %b_trans = tt.trans %b {order = array<i32: 1, 0>} : tensor<128x64xf32, #src> -> tensor<64x128xf32, #dst>
    %b_trunc = arith.truncf %b_trans : tensor<64x128xf32, #dst> to tensor<64x128xbf16, #dst>
    %b_smem = ttg.local_alloc %b_trunc : (tensor<64x128xbf16, #dst>) -> !ttg.memdesc<64x128xbf16, #shared_b, #smem>
    %mma = ttng.tc_gen5_mma %a_smem, %b_smem, %acc[%token], %true, %true {two_ctas} : !ttg.memdesc<128x64xbf16, #shared_a, #smem>, !ttg.memdesc<64x128xbf16, #shared_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // This second use requires the original full-width host descriptor.
    %unused = tt.descriptor_load %b_desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x64xf32> -> tensor<128x64xf32, #src>
    tt.return
  }
}
