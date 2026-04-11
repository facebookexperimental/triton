// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -nvgpu-modulo-schedule -nvgpu-modulo-ws-partition | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// Verify that Pass B assigns utilization-driven ttg.partition attrs on a
// persistent kernel with a WS outer loop containing an inner K-loop.
// Expected partitions: MEM=0, TC=1, CUDA(tmem_load)=2.
// Shared/scalar ops get allParts [0,1,2].
//
// CHECK-LABEL: @persistent_gemm_ws_partition
// MEM ops (descriptor_load, local_alloc) → partition 0
// CHECK: tt.descriptor_load {{.*}} ttg.partition = array<i32: 0>
// CHECK: tt.descriptor_load {{.*}} ttg.partition = array<i32: 0>
// CHECK: ttg.local_alloc {{.*}} ttg.partition = array<i32: 0>
// CHECK: ttg.local_alloc {{.*}} ttg.partition = array<i32: 0>
// TC ops (tc_gen5_mma) → partition 1
// CHECK: ttng.tc_gen5_mma {{.*}} ttg.partition = array<i32: 1>
// CUDA ops (tmem_load) → partition 2
// CHECK: ttng.tmem_load {{.*}} ttg.partition = array<i32: 2>
tt.func @persistent_gemm_ws_partition(
  %a_desc: !tt.tensordesc<tensor<128x64xf16>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16>>,
  %num_tiles: i32
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %k_tiles = arith.constant 32 : i32
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

  // Outer tile loop with tt.warp_specialize — triggers partition assignment
  scf.for %tile = %c0_i32 to %num_tiles step %c1_i32 : i32 {
    // Inner K-loop (GEMM accumulation)
    scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> (tensor<128x128xf32, #acc_layout>) : i32 {
      %off_k = arith.muli %k, %c1_i32 : i32

      %a = tt.descriptor_load %a_desc[%c0_i32, %off_k] : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked>
      %b = tt.descriptor_load %b_desc[%off_k, %c0_i32] : !tt.tensordesc<tensor<64x128xf16>> -> tensor<64x128xf16, #blocked>

      %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

      %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
      %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

      scf.yield %c : tensor<128x128xf32, #acc_layout>
    }

    scf.yield
  } {tt.warp_specialize}

  tt.return
}

}
