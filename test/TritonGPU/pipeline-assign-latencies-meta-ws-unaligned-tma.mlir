// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-assign-latencies="num-stages=3 use-meta-ws=true" | FileCheck %s

// Under Meta autoWS an unaligned TMA load in a warp-specialized loop still gets a latency.

#tiny_blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tiny_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @unaligned_tma_ws_loop
  // CHECK: tt.descriptor_load {{.*}}tt.latency = 2
  tt.func @unaligned_tma_ws_loop(%desc: !tt.tensordesc<4xf32, #tiny_shared>,
                                 %lb: index, %ub: index, %step: index) {
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      %load = tt.descriptor_load %desc[%c0] : !tt.tensordesc<4xf32, #tiny_shared> -> tensor<4xf32, #tiny_blocked>
      "use"(%load) : (tensor<4xf32, #tiny_blocked>) -> ()
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    tt.return
  }
}
