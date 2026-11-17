// RUN: triton-opt %s --nvgpu-2cta-transform-loads | FileCheck %s

// B built in registers from a descriptor load must still be halved along N
// per CTA; a full-width B makes each CTA feed the wrong half.
// CHECK-LABEL: @register_b
// CHECK: nvg.cluster_id
// CHECK: tt.descriptor_load {{.*}}{two_cta_b{{.*}}} : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32
// CHECK: tt.broadcast {{.*}} -> tensor<64x64xf32
// CHECK: arith.mulf {{.*}} : tensor<64x64xf32
// CHECK: arith.truncf {{.*}} to tensor<64x64xbf16
// CHECK: ttng.tc_gen5_mma {{.*}} {two_cta_register_b, two_ctas} : !ttg.memdesc<128x64xbf16, {{.*}}>, !ttg.memdesc<64x64xbf16
// CHECK-NOT: tensor<64x128xf32

#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @register_b(%a: !ttg.memdesc<128x64xbf16, #shared, #smem>, %bp: !tt.ptr<f32>, %n: i32, %s64: i64, %scale: tensor<64x1xf32, #blocked1>, %acc: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %i: i32, %j: i32) {
    %true = arith.constant true
    %c1 = arith.constant 1 : i64
    %bd = tt.make_tensor_descriptor %bp, [%n, %n], [%s64, %c1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
    %b = tt.descriptor_load %bd[%i, %j] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked1>
    %sb = tt.broadcast %scale : tensor<64x1xf32, #blocked1> -> tensor<64x128xf32, #blocked1>
    %m = arith.mulf %b, %sb : tensor<64x128xf32, #blocked1>
    %t = arith.truncf %m : tensor<64x128xf32, #blocked1> to tensor<64x128xbf16, #blocked1>
    %s = ttg.local_alloc %t : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
    ttng.tc_gen5_mma %a, %s, %acc, %true, %true {two_ctas} : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
