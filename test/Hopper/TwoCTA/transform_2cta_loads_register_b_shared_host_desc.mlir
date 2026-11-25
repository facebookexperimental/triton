// RUN: not triton-opt %s --split-input-file --nvgpu-2cta-transform-loads 2>&1 | FileCheck %s

// Register-built B: two 2-CTA MMAs whose B comes from one host descriptor
// argument split along different descriptor dims (B = W*W and B = trans(W*W)).
// The argument is retyped in place, so the second split must be an error.
// CHECK: error: 2-CTA B shares a host descriptor with another 2-CTA MMA that splits it along a different dimension
// CHECK-NEXT: ttng.tc_gen5_mma %a, %s2

#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedT = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @xmma(%a: !ttg.memdesc<128x128xbf16, #shared, #smem>, %bd: !tt.tensordesc<128x128xf32>, %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %acc2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %i: i32, %j: i32) {
    %true = arith.constant true
    %b = tt.descriptor_load %bd[%i, %j] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32, #blocked1>
    %m = arith.mulf %b, %b : tensor<128x128xf32, #blocked1>
    %t = arith.truncf %m : tensor<128x128xf32, #blocked1> to tensor<128x128xbf16, #blocked1>
    %s = ttg.local_alloc %t : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    ttng.tc_gen5_mma %a, %s, %acc, %true, %true {two_ctas} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %b2 = tt.descriptor_load %bd[%j, %i] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32, #blocked1>
    %m2 = arith.mulf %b2, %b2 : tensor<128x128xf32, #blocked1>
    %t2 = arith.truncf %m2 : tensor<128x128xf32, #blocked1> to tensor<128x128xbf16, #blocked1>
    %tt = tt.trans %t2 {order = array<i32: 1, 0>} : tensor<128x128xbf16, #blocked1> -> tensor<128x128xbf16, #blockedT>
    %s2 = ttg.local_alloc %tt : (tensor<128x128xbf16, #blockedT>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    ttng.tc_gen5_mma %a, %s2, %acc2, %true, %true {two_ctas} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Same conflict with the register-built split first and a direct descriptor
// load B second (B = trans(W*W), then B = W).
// CHECK: error: 2-CTA B shares a host descriptor with another 2-CTA MMA that splits it along a different dimension
// CHECK-NEXT: ttng.tc_gen5_mma %a, %s2

#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedT = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @register_then_direct(%a: !ttg.memdesc<128x128xbf16, #shared, #smem>, %bd: !tt.tensordesc<128x128xbf16>, %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %acc2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %i: i32, %j: i32) {
    %true = arith.constant true
    %b = tt.descriptor_load %bd[%j, %i] : !tt.tensordesc<128x128xbf16> -> tensor<128x128xbf16, #blocked1>
    %m = arith.mulf %b, %b : tensor<128x128xbf16, #blocked1>
    %t = tt.trans %m {order = array<i32: 1, 0>} : tensor<128x128xbf16, #blocked1> -> tensor<128x128xbf16, #blockedT>
    %s = ttg.local_alloc %t : (tensor<128x128xbf16, #blockedT>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    ttng.tc_gen5_mma %a, %s, %acc, %true, %true {two_ctas} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %b2 = tt.descriptor_load %bd[%i, %j] : !tt.tensordesc<128x128xbf16> -> tensor<128x128xbf16, #blocked1>
    %s2 = ttg.local_alloc %b2 : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    ttng.tc_gen5_mma %a, %s2, %acc2, %true, %true {two_ctas} : !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
