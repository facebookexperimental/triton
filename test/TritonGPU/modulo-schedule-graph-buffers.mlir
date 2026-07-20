// REQUIRES: asserts
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule -debug-only=nvgpu-modulo-schedule 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: Buffer allocations and barrier pairing
//   SMEM buffers for A (128x64xf16) and B (64x128xf16) tiles,
//   TMEM buffer for accumulator (128x128xf32) with a paired barrier record;
//   the SMEM tiles' producer/consumer sync surfaces as cross-group mbarriers
//   (record consolidation folded their BARRIER alloc records away).
//===----------------------------------------------------------------------===//

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// --- SMEM buffers: count=1 under the honest latency model — the SMEM live
//     ranges ([557,1146) and [587,1146)) fit inside a single II=1091, so no
//     double-buffering is needed. Shapes match tiles, live=[start, end)
//     (design doc §215 worked example predates the honest latency model). ---
// CHECK: %buf0 = modulo.alloc SMEM [1 x 128x64 x f16]
// CHECK-SAME: live=[
// CHECK-SAME: bytes total
// CHECK: %buf1 = modulo.alloc SMEM [1 x 64x128 x f16]
// CHECK-SAME: live=[
// CHECK-SAME: bytes total
//
// --- TMEM buffer: count=2 for accumulator (live=[0,1678) spans two IIs of
//     1091; 2 x 128x128 x f32 = 131072 bytes) ---
// CHECK: %buf2 = modulo.alloc TMEM [2 x 128x128 x f32]
// CHECK-SAME: live=[0, 1678)
// CHECK-SAME: 131072 bytes total
//
// --- Paired barrier carries the same live interval as its data buffer
//     (live=[0, 1678) matches buf2 above; barrier slot count 2 matches the
//     buffer count). Record consolidation removed the per-SMEM-buffer
//     BARRIER records — SMEM sync is checked as cross-group mbarriers at
//     the bottom of this file. ---
// CHECK: %bar3 = modulo.alloc BARRIER [2] for buf2
// CHECK-SAME: live=[0, 1678)
//
// --- Producers: local_alloc → ->buf (local_alloc is an infra op on the NONE
//     pipe now; the TMA pipe is carried by the tt.descriptor_load) ---
// CHECK: ttg.local_alloc  {pipe: NONE, {{.*}}->buf0}
// CHECK: ttg.local_alloc  {pipe: NONE, {{.*}}->buf1}
//
// --- Consumer: MMA consumes all three buffers ---
// CHECK: ttng.tc_gen5_mma  {pipe: TC, {{.*}}<-buf0, <-buf1, <-buf2}
//
// --- tmem_load consumes TMEM buffer ---
// CHECK: ttng.tmem_load  {pipe: CUDA, {{.*}}<-buf2}
//
// --- SMEM buffers' producer→consumer pairing now surfaces as cross-group
//     mbarriers: N3/N4 are the local_allocs (->buf0/->buf1), N6 the MMA;
//     depth=1 matches the buffer count, expect = full tile bytes (16384).
//     Warp-group ids are incidental to this test (near-tied partition
//     candidates), so they are regexed. ---
// CHECK: [PassB.2] Barrier: N3(wg{{[0-9]+}}) → N6(wg{{[0-9]+}}) mbarrier depth=1 {{.*}}expect=16384B
// CHECK: [PassB.2] Barrier: N4(wg{{[0-9]+}}) → N6(wg{{[0-9]+}}) mbarrier depth=1 {{.*}}expect=16384B
// CHECK: [PassB.2] Total cross-group barriers: 2
tt.func @test_buffers(
  %a_desc: !tt.tensordesc<128x64xf16>,
  %b_desc: !tt.tensordesc<64x128xf16>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %k_tiles = arith.constant 32 : i32
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> (tensor<128x128xf32, #acc_layout>) : i32 {
    %off_k = arith.muli %k, %c1_i32 : i32

    %a = tt.descriptor_load %a_desc[%c0_i32, %off_k] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
    %b = tt.descriptor_load %b_desc[%off_k, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked>

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
