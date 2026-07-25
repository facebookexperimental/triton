// RUN: triton-opt %s -tritongpu-optimize-accumulator-init | FileCheck %s

// Two chained accumulating MMAv5 dots into one logical accumulator (the HSTU
// reduce_dq "compute fold" shape: dk = dot(a0,b0,dk); dk = dot(a1,b1,dk)).
// AccelerateMatmul lowers each dot into its own tmem_alloc + tc_gen5_mma +
// tmem_load, threading the accumulator through registers, so dot2's TMEM tile is
// initialized from dot1's TMEM read-out. ChainAccumulatorInPlace must coalesce
// them into a single in-place accumulation on ONE TMEM tile (no intermediate
// tile / read-out bridge), matching the TLX single-dk_tiles shape. Left alone,
// the two tiles become a TMEM<->TMEM load/store bridge that blocks TMEM slot
// reuse downstream (the WSCodePartition reuse-order check aborts).

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @chain_accumulator_in_place
// One accumulator tile; both MMAs accumulate into it; the second reads the
// first's completion token in place; a single read-out; no bridge tile/load.
// CHECK:       scf.for
// CHECK:         %[[ACC:.*]], %[[TK0:.*]] = ttng.tmem_alloc %arg
// CHECK:         %[[TK1:.*]] = ttng.tc_gen5_mma {{.*}} %[[ACC]][%[[TK0]]]
// CHECK-NOT:     ttng.tmem_alloc
// CHECK-NOT:     ttng.tmem_load
// CHECK:         %[[TK2:.*]] = ttng.tc_gen5_mma {{.*}} %[[ACC]][%[[TK1]]], %true
// CHECK:         ttng.tmem_load %[[ACC]][%[[TK2]]]
tt.func public @chain_accumulator_in_place(
    %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b0: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %a1: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b1: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %n: i32) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>) : i32 {
    // dot1 tile
    %t1, %tok1 = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m1 = ttng.tc_gen5_mma %a0, %b0, %t1[%tok1], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v1, %ld1 = ttng.tmem_load %t1[%m1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // dot2 tile, initialized from dot1's read-out
    %t2, %tok2 = ttng.tmem_alloc %v1 : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m2 = ttng.tc_gen5_mma %a1, %b1, %t2[%tok2], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v2, %ld2 = ttng.tmem_load %t2[%m2] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    scf.yield %v2 : tensor<128x128xf32, #blocked>
  }
  tt.return
}

}
