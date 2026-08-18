// RUN: triton-opt %s --nvgpu-warp-specialization="capability=90 num-stages=2" | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @native_ws_marker_without_tasks
  // CHECK: tt.descriptor_load
  // CHECK-NOT: nvws.descriptor_load
  // CHECK-NOT: tt.warp_specialize
  tt.func public @native_ws_marker_without_tasks(
      %src: !tt.tensordesc<128x64xf16, #shared>,
      %dst: !tt.tensordesc<128x64xf16, #shared>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c1 step %c1 : i32 {
      %tile = tt.descriptor_load %src[%c0, %c0] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      tt.descriptor_store %dst[%c0, %c0], %tile : !tt.tensordesc<128x64xf16, #shared>, tensor<128x64xf16, #blocked>
    } {tt.warp_specialize}
    tt.return
  }
}
