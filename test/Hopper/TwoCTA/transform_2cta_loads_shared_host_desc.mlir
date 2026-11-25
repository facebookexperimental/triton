// RUN: not triton-opt %s --nvgpu-2cta-transform-loads 2>&1 | FileCheck %s

// A host descriptor argument is retyped in place to the half-width block. Two
// 2-CTA MMAs whose B tiles come from the same argument but split different
// descriptor dims (B = W tile, B = trans(W tile)) cannot both be satisfied:
// the second retype would leave the first MMA's half-width load reading a box
// of the wrong shape. That must be an error, not a wrong tile.
// CHECK: error: 2-CTA B shares a host descriptor with another 2-CTA MMA that splits it along a different dimension
// CHECK-NEXT: ttng.tc_gen5_mma %a, %s2

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @shared_host_desc_two_split_dims(%a: !ttg.memdesc<128x128xbf16, #shared, #smem>, %w_desc: !tt.tensordesc<128x128xbf16>, %acc0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %acc1: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %i: i32, %j: i32) {
    %true = arith.constant true
    %w = tt.descriptor_load %w_desc[%i, %j] : !tt.tensordesc<128x128xbf16> -> tensor<128x128xbf16, #blocked>
    %s = ttg.local_alloc %w : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    ttng.tc_gen5_mma %a, %s, %acc0, %true, %true {two_ctas} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %w2 = tt.descriptor_load %w_desc[%j, %i] : !tt.tensordesc<128x128xbf16> -> tensor<128x128xbf16, #blocked>
    %wt = tt.trans %w2 {order = array<i32: 1, 0>} : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #blocked_t>
    %s2 = ttg.local_alloc %wt : (tensor<128x128xbf16, #blocked_t>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    ttng.tc_gen5_mma %a, %s2, %acc1, %true, %true {two_ctas} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
