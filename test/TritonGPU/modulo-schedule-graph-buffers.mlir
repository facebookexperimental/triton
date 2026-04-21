// REQUIRES: asserts
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule -debug-only=nvgpu-modulo-schedule 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: Buffer allocations and barrier pairing
//   SMEM buffers for A (128x64xf16) and B (64x128xf16) tiles,
//   TMEM buffer for accumulator (128x128xf32), each with paired barriers.
//===----------------------------------------------------------------------===//

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// --- SMEM buffers: count=2, shapes match tiles, live=[start, end) per
//     design doc §215 worked example. ---
// CHECK: %buf0 = modulo.alloc SMEM [2 x 128x64 x f16]
// CHECK-SAME: live=[
// CHECK-SAME: bytes total
// CHECK: %buf1 = modulo.alloc SMEM [2 x 64x128 x f16]
// CHECK-SAME: live=[
// CHECK-SAME: bytes total
//
// --- TMEM buffer: count=3 for accumulator ---
// CHECK: %buf2 = modulo.alloc TMEM [3 x 128x128 x f32]
// CHECK-SAME: live=[
// CHECK-SAME: 196608 bytes total
//
// --- Paired barriers carry the same live interval as their data buffer ---
// CHECK: %bar3 = modulo.alloc BARRIER [2] for buf0
// CHECK-SAME: live=[
// CHECK: %bar4 = modulo.alloc BARRIER [2] for buf1
// CHECK-SAME: live=[
// CHECK: %bar5 = modulo.alloc BARRIER [3] for buf2
// CHECK-SAME: live=[
//
// --- Producers: local_alloc → ->buf ---
// CHECK: ttg.local_alloc  {pipe: MEM, {{.*}}->buf0}
// CHECK: ttg.local_alloc  {pipe: MEM, {{.*}}->buf1}
//
// --- Consumer: MMA consumes all three buffers ---
// CHECK: ttng.tc_gen5_mma  {pipe: TC, {{.*}}<-buf0, <-buf1, <-buf2}
//
// --- tmem_load consumes TMEM buffer ---
// CHECK: ttng.tmem_load  {pipe: CUDA, {{.*}}<-buf2}
tt.func @test_buffers(
  %a_desc: !tt.tensordesc<tensor<128x64xf16>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %k_tiles = arith.constant 32 : i32
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

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

  tt.return
}

}
