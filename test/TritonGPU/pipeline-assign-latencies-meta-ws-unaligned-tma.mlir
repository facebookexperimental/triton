// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-assign-latencies="num-stages=3 use-meta-ws=true" | FileCheck %s

// Under Meta autoWS the memory planner keeps an unaligned TMA landing slot
// single-buffered, so a TMA load whose per-stage allocation is not aligned is
// still given a latency in a warp-specialized loop. Otherwise it lands at its
// consumer's stage, after the other (prefetched) loads of the same producer
// partition, and a single-buffered prefetched operand deadlocks the producer.

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

  // Without warp specialization the pipeliner allocates the stages itself, so
  // the unaligned load stays unpipelined.
  // CHECK-LABEL: @unaligned_tma_no_ws
  // CHECK: tt.descriptor_load
  // CHECK-NOT: tt.latency
  // CHECK: "use"
  tt.func @unaligned_tma_no_ws(%desc: !tt.tensordesc<4xf32, #tiny_shared>,
                               %lb: index, %ub: index, %step: index) {
    %c0 = arith.constant 0 : i32
    scf.for %iv = %lb to %ub step %step {
      %load = tt.descriptor_load %desc[%c0] : !tt.tensordesc<4xf32, #tiny_shared> -> tensor<4xf32, #tiny_blocked>
      "use"(%load) : (tensor<4xf32, #tiny_blocked>) -> ()
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

// The real shape: an unaligned 1-D operand tile scales a 2-D TMA tile that
// feeds the MMA. Both loads get the same latency in a warp-specialized loop.
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#vshared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @unaligned_tma_feeds_mma_ws_loop
  // CHECK: tt.descriptor_load %{{.*}} {tt.latency = 2 : i32} : !tt.tensordesc<128x16xf16
  // CHECK: tt.descriptor_load %{{.*}} {tt.latency = 2 : i32} : !tt.tensordesc<16xf32
  tt.func @unaligned_tma_feeds_mma_ws_loop(%a_desc: !tt.tensordesc<128x16xf16, #shared>,
                                           %v_desc: !tt.tensordesc<16xf32, #vshared>,
                                           %b: !ttg.memdesc<16x128xf16, #shared, #ttg.shared_memory>,
                                           %lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %r = scf.for %iv = %lb to %ub step %step iter_args(%t = %tok) -> (!ttg.async.token) : i32 {
      %a = tt.descriptor_load %a_desc[%c0, %iv] : !tt.tensordesc<128x16xf16, #shared> -> tensor<128x16xf16, #blocked>
      %v = tt.descriptor_load %v_desc[%iv] : !tt.tensordesc<16xf32, #vshared> -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %vh = arith.truncf %v : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<16xf16, #ttg.slice<{dim = 0, parent = #blocked}>>
      %ve = tt.expand_dims %vh {axis = 0 : i32} : tensor<16xf16, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xf16, #blocked>
      %vb = tt.broadcast %ve : tensor<1x16xf16, #blocked> -> tensor<128x16xf16, #blocked>
      %as = arith.mulf %a, %vb : tensor<128x16xf16, #blocked>
      %a_sh = ttg.local_alloc %as : (tensor<128x16xf16, #blocked>) -> !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>
      %nt = ttng.tc_gen5_mma %a_sh, %b, %acc[%t], %true, %true : !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<16x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %nt : !ttg.async.token
    } {tt.warp_specialize}
    tt.return
  }
}
