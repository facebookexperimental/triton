// RUN: env TRITON_USE_META_WS=1 triton-opt %s '--tritongpu-pipeline=num-stages=3' | FileCheck %s

// TRITON_USE_META_WS=1 with no ttg.warp_specialize (AutoWS bailed out). The
// metaWS epilogue peeling must not apply: peeling this dynamic-trip-count loop
// would predicate the last-stage ttng.warp_group_dot, which the pipeliner
// cannot predicate.

#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @bailout_dynamic_wgmma_loop
  // CHECK: scf.for
  // CHECK: ttng.warp_group_dot {{.*}}isAsync = true
  // CHECK: scf.yield
  // CHECK-NOT: ttng.warp_group_dot {{.*}}->
  // CHECK: tt.return
  tt.func public @bailout_dynamic_wgmma_loop(%a_desc: !tt.tensordesc<128x64xf16, #shared>, %b_desc: !tt.tensordesc<128x64xf16, #shared>, %k_tiles: i32, %offs_am: i32, %offs_bn: i32) -> tensor<128x128xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %acc = scf.for %ki = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc_arg = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
      %offs_k = arith.muli %ki, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %a = tt.descriptor_load %a_desc[%offs_am, %offs_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %a_smem = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%offs_bn, %offs_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %b_smem = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_trans = ttg.memdesc_trans %b_smem {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %dot = ttng.warp_group_dot %a_smem, %b_trans, %acc_arg {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared1, #smem> -> tensor<128x128xf32, #mma>
      scf.yield %dot : tensor<128x128xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
    tt.return %acc : tensor<128x128xf32, #mma>
  }
}
