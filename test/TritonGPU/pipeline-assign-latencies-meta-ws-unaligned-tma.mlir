// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-assign-latencies="num-stages=3 use-meta-ws=true" | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-assign-latencies="num-stages=3 use-meta-ws=false" | FileCheck %s --check-prefix=NOMETA

// Under Meta autoWS an unaligned TMA load in a warp-specialized loop still gets a latency.

#tiny_blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tiny_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#scale_blocked = #ttg.slice<{dim = 0, parent = #blocked}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#scale_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16, rank = 1}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @unaligned_tma_ws_loop
  // CHECK: tt.descriptor_load {{.*}}tt.latency = 2
  // Without Meta autoWS it stays unpipelined even in a warp-specialized loop.
  // NOMETA-LABEL: @unaligned_tma_ws_loop
  // NOMETA: tt.descriptor_load
  // NOMETA-NOT: tt.latency
  // NOMETA: "use"
  tt.func @unaligned_tma_ws_loop(%desc: !tt.tensordesc<4xf32, #tiny_shared>,
                                 %lb: index, %ub: index, %step: index) {
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      %load = tt.descriptor_load %desc[%c0] : !tt.tensordesc<4xf32, #tiny_shared> -> tensor<4xf32, #tiny_blocked>
      "use"(%load) : (tensor<4xf32, #tiny_blocked>) -> ()
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    tt.return
  }

  // Without warp specialization the unaligned load stays unpipelined.
  // CHECK-LABEL: @unaligned_tma_no_ws_loop
  // CHECK: tt.descriptor_load
  // CHECK-NOT: tt.latency
  // CHECK: "use"
  tt.func @unaligned_tma_no_ws_loop(%desc: !tt.tensordesc<4xf32, #tiny_shared>,
                                    %lb: index, %ub: index, %step: index) {
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      %load = tt.descriptor_load %desc[%c0] : !tt.tensordesc<4xf32, #tiny_shared> -> tensor<4xf32, #tiny_blocked>
      "use"(%load) : (tensor<4xf32, #tiny_blocked>) -> ()
    } {tt.num_stages = 3 : i32}
    tt.return
  }

  // An unaligned 1-D load that scales MMA operand A gets the same latency as
  // the 2-D operand loads, so the producer loads share a stage.
  // CHECK-LABEL: @unaligned_tma_scales_mma_operand
  // CHECK: tt.descriptor_load {{.*}} {tt.latency = 2 : i32} : !tt.tensordesc<128x16xf16,
  // CHECK: tt.descriptor_load {{.*}} {tt.latency = 2 : i32} : !tt.tensordesc<16xf16,
  // CHECK: tt.descriptor_load {{.*}} {tt.latency = 2 : i32} : !tt.tensordesc<16x128xf16,
  tt.func @unaligned_tma_scales_mma_operand(%a_desc: !tt.tensordesc<128x16xf16, #shared>,
                                            %b_desc: !tt.tensordesc<16x128xf16, #shared>,
                                            %s_desc: !tt.tensordesc<16xf16, #scale_shared>,
                                            %n: i32) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i32
    %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %r = scf.for %k = %c0 to %n step %c16 iter_args(%t = %tok) -> (!ttg.async.token) : i32 {
      %a = tt.descriptor_load %a_desc[%c0, %k] : !tt.tensordesc<128x16xf16, #shared> -> tensor<128x16xf16, #blocked>
      %s = tt.descriptor_load %s_desc[%k] : !tt.tensordesc<16xf16, #scale_shared> -> tensor<16xf16, #scale_blocked>
      %s2 = tt.expand_dims %s {axis = 0 : i32} : tensor<16xf16, #scale_blocked> -> tensor<1x16xf16, #blocked>
      %sb = tt.broadcast %s2 : tensor<1x16xf16, #blocked> -> tensor<128x16xf16, #blocked>
      %as = arith.mulf %a, %sb : tensor<128x16xf16, #blocked>
      %a_sh = ttg.local_alloc %as : (tensor<128x16xf16, #blocked>) -> !ttg.memdesc<128x16xf16, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%k, %c0] : !tt.tensordesc<16x128xf16, #shared> -> tensor<16x128xf16, #blocked>
      %b_sh = ttg.local_alloc %b : (tensor<16x128xf16, #blocked>) -> !ttg.memdesc<16x128xf16, #shared, #smem>
      %nt = ttng.tc_gen5_mma %a_sh, %b_sh, %acc[%t], %true, %true : !ttg.memdesc<128x16xf16, #shared, #smem>, !ttg.memdesc<16x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %nt : !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    tt.return
  }
}
