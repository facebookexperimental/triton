
// RUN: triton-opt %s | FileCheck %s

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 0}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @require_layout
  tt.func @require_layout(%arg0: !ttg.memdesc<128x64xf16, #shared1, #smem>) {
    // CHECK: tlx.require_layout
    %0 = tlx.require_layout %arg0 : !ttg.memdesc<128x64xf16, #shared1, #smem> -> !ttg.memdesc<128x64xf16, #shared2, #smem>
    tt.return
  }
}
