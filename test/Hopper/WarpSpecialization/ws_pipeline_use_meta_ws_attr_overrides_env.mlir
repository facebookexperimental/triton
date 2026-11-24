// RUN: env TRITON_USE_META_WS=1 triton-opt %s -tritongpu-pipeline | FileCheck %s

// Variant of ws_pipeline_predicate_warp_group_dot.mlir with the per-module
// ttg.use_meta_ws = false attribute set by make_ttgir. The attribute takes
// precedence over TRITON_USE_META_WS=1, so the Meta WS epilogue peeling (an
// scf.if-predicated warp_group_dot after the loop) must not happen.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 8]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {ttg.use_meta_ws = false, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The module attribute wins over TRITON_USE_META_WS=1: the loop is pipelined
  // as with the knob off, so no epilogue is peeled after it.
  // CHECK-LABEL: @pipeline_predicate_warp_group_dot
  // CHECK: scf.for {{.*}} to %c128_i32
  // CHECK: ttng.warp_group_dot {{.*}}isAsync = true
  // CHECK: scf.yield
  // CHECK-NOT: scf.if
  // CHECK-NOT: ttng.warp_group_dot %
  // CHECK: tt.return
  tt.func public @pipeline_predicate_warp_group_dot(%arg0: tensor<256x128xf32, #mma>, %arg1: tensor<256x32x!tt.ptr<f32>, #blocked>, %arg2: tensor<32x128x!tt.ptr<f32>, #blocked1>, %arg3: tensor<256x32xi32, #blocked>, %arg4: tensor<32x128xi32, #blocked1>) -> (tensor<256x128xf32, #mma>, tensor<256x32x!tt.ptr<f32>, #blocked>, tensor<32x128x!tt.ptr<f32>, #blocked1>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %0:3 = scf.for %arg5 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %arg2) -> (tensor<256x128xf32, #mma>, tensor<256x32x!tt.ptr<f32>, #blocked>, tensor<32x128x!tt.ptr<f32>, #blocked1>)  : i32 {
      %1 = tt.load %arg7 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<256x32x!tt.ptr<f32>, #blocked>
      %2 = ttg.local_alloc %1 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<256x32xf32, #blocked>) -> !ttg.memdesc<256x32xf32, #shared, #smem>
      %3 = tt.load %arg8 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<32x128x!tt.ptr<f32>, #blocked1>
      %4 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<32x128xf32, #blocked1>) -> !ttg.memdesc<32x128xf32, #shared1, #smem>
      %5 = ttng.warp_group_dot %2, %4, %arg6 {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32} : !ttg.memdesc<256x32xf32, #shared, #smem> * !ttg.memdesc<32x128xf32, #shared1, #smem> -> tensor<256x128xf32, #mma>
      %6 = tt.addptr %arg7, %arg3 {loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<256x32x!tt.ptr<f32>, #blocked>, tensor<256x32xi32, #blocked>
      %7 = tt.addptr %arg8, %arg4 {loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<32x128x!tt.ptr<f32>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %5, %6, %7 : tensor<256x128xf32, #mma>, tensor<256x32x!tt.ptr<f32>, #blocked>, tensor<32x128x!tt.ptr<f32>, #blocked1>
    } {tt.num_stages = 4 : i32, tt.scheduled_max_stage = 1 : i32}
    tt.return %0#0, %0#1, %0#2 : tensor<256x128xf32, #mma>, tensor<256x32x!tt.ptr<f32>, #blocked>, tensor<32x128x!tt.ptr<f32>, #blocked1>
  }
}
