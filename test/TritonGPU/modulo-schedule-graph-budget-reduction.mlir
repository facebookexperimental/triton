// REQUIRES: asserts
// RUN: TRITON_MODULO_SMEM_BUDGET_KB=448 triton-opt %s -allow-unregistered-dialect -split-input-file -nvgpu-modulo-schedule -debug-only=nvgpu-modulo-schedule 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Regression tests for Step 4.6 budget reduction (reduceBufferGroup +
// post-reduction fixed-point II recompute). Both cases run with an inflated
// SMEM budget (TRITON_MODULO_SMEM_BUDGET_KB=448 = 458752 B) so the reduction
// path fires at a controlled, self-documenting threshold instead of relying on
// the hardware default.
//===----------------------------------------------------------------------===//

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

//===----------------------------------------------------------------------===//
// Issue 1: loop-carried II under budget reduction.
//
// A software-prefetch GEMM: the A tile loaded in iteration i is consumed by the
// MMA in iteration i+1, so buf2 (the 128x512xf16 A ring) has a loop-carried
// consumer edge (N7 -> N4, dist=1). Its lifetime therefore grows with II:
//   lifetime(II) = base + distance*II,   base = 1714 (intra-iteration span).
// At the initial II=1166 the ring wants count=3 (393216 B); together with the
// 512x64 B ring and barriers the loop needs 524344 B > 458752, so the reducer
// drops buf2 to count=2. buf2 is the binding buffer for the recomputed II:
//   - single-pass (the OLD bug) would use the lifetime measured at the old II:
//       ceil(lifetime(1166)/2) = ceil((1714+1166)/2) = ceil(2880/2) = 1440,
//     an UNDER-estimate: at II=1440, depth*II = 2*1440 = 2880 < lifetime(1440)
//     = 1714+1440 = 3154, so the loader would reclaim a slot before the
//     loop-carried consumer finished.
//   - the fixed-point recompute solves depth*II >= base + distance*II, i.e.
//       2*II >= 1714 + II  =>  II >= 1714,
//     converging to II = 1714, where 2*1714 = 3428 >= 1714+1714 = 3428.
//
// CHECK: [Step4.6] Reduced SMEM buf2 (+ co-consumed/merge peers) to count=2
// CHECK: [Step4.6] Raising II from 1166 to 1714 due to buffer depth reduction
// The schedule graph carries the fixed-point II, and the loop-carried operand
// ring is double-buffered (count=2) at that raised II:
// CHECK: ii = 1714, max_stage = 1
// CHECK: %buf2 = modulo.alloc SMEM [2 x 128x512 x f16]
tt.func @prefetch_carried_ii(
  %a_desc: !tt.tensordesc<128x512xf16>,
  %b_desc: !tt.tensordesc<512x64xf16>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %k_tiles = arith.constant 32 : i32
  %zero = arith.constant dense<0.0> : tensor<128x64xf32, #acc_layout>

  // Prefetch iteration 0's A tile before the loop.
  %a0 = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x512xf16> -> tensor<128x512xf16, #blocked>
  %a0_sh = ttg.local_alloc %a0 : (tensor<128x512xf16, #blocked>) -> !ttg.memdesc<128x512xf16, #shared, #smem>

  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%a_prev = %a0_sh, %acc = %zero) -> (!ttg.memdesc<128x512xf16, #shared, #smem>, tensor<128x64xf32, #acc_layout>) : i32 {
    %off_k = arith.muli %k, %c1_i32 : i32

    %b = tt.descriptor_load %b_desc[%off_k, %c0_i32] : !tt.tensordesc<512x64xf16> -> tensor<512x64xf16, #blocked>
    %b_sh = ttg.local_alloc %b : (tensor<512x64xf16, #blocked>) -> !ttg.memdesc<512x64xf16, #shared, #smem>

    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x64xf32, #acc_layout>) -> (!ttg.memdesc<128x64xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // Consume the PREVIOUS iteration's prefetched A tile.
    %mma_tok = ttng.tc_gen5_mma %a_prev, %b_sh, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x512xf16, #shared, #smem>, !ttg.memdesc<512x64xf16, #shared, #smem>, !ttg.memdesc<128x64xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x64xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #acc_layout>

    // Prefetch the next iteration's A tile.
    %a_next = tt.descriptor_load %a_desc[%c0_i32, %off_k] : !tt.tensordesc<128x512xf16> -> tensor<128x512xf16, #blocked>
    %a_next_sh = ttg.local_alloc %a_next : (tensor<128x512xf16, #blocked>) -> !ttg.memdesc<128x512xf16, #shared, #smem>

    scf.yield %a_next_sh, %c : !ttg.memdesc<128x512xf16, #shared, #smem>, tensor<128x64xf32, #acc_layout>
  }
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

//===----------------------------------------------------------------------===//
// Issue 2: co-consumed / merge-group peers reduced together.
//
// A and B both feed the same MMA, so they form one co-consumed ring group that
// must stay equal depth (and, once merged, one physical allocation whose size
// is max(member)). At the initial II both want count=3:
//   buf0 (128x512xf16) = 3*131072 = 393216 B,  buf1 (512x64xf16) = 3*65536 =
//   196608 B, so the loop needs ~589896 B > 458752.
// reduceBufferGroup must drop the WHOLE group by one step and refresh the
// physical buffers so computeTotalSmem sees the smaller footprint. At count=2
// the group is 262144 + 131072 = 393216 B (+ barriers) = 393272 B <= 458752, so
// the reducer stops — capping the ring at the largest depth that fits.
//
// The regression guard: BOTH members drop together to the SAME count (2), and
// the reported footprint actually falls below budget. Before the fix (reduce a
// single member, no physical refresh) one member would be hammered down to
// count=1 while its peer stayed high and the shared physical footprint never
// dropped.
//
// CHECK: [Step4.6] Reduced SMEM buf1 (+ co-consumed/merge peers) to count=2
// Footprint dropped below the 458752 B budget after refreshing physical buffers:
// CHECK: [Step4.6] Budget: SMEM 393272/458752 OK
// Both co-consumed operands end at the SAME count=2 (not one collapsed to 1):
// CHECK-DAG: %buf0 = modulo.alloc SMEM [2 x 128x512 x f16]
// CHECK-DAG: %buf1 = modulo.alloc SMEM [2 x 512x64 x f16]
tt.func @coconsumed_group_reduce(
  %a_desc: !tt.tensordesc<128x512xf16>,
  %b_desc: !tt.tensordesc<512x64xf16>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %k_tiles = arith.constant 32 : i32
  %zero = arith.constant dense<0.0> : tensor<128x64xf32, #acc_layout>

  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> (tensor<128x64xf32, #acc_layout>) : i32 {
    %off_k = arith.muli %k, %c1_i32 : i32

    %a = tt.descriptor_load %a_desc[%c0_i32, %off_k] : !tt.tensordesc<128x512xf16> -> tensor<128x512xf16, #blocked>
    %b = tt.descriptor_load %b_desc[%off_k, %c0_i32] : !tt.tensordesc<512x64xf16> -> tensor<512x64xf16, #blocked>

    %a_sh = ttg.local_alloc %a : (tensor<128x512xf16, #blocked>) -> !ttg.memdesc<128x512xf16, #shared, #smem>
    %b_sh = ttg.local_alloc %b : (tensor<512x64xf16, #blocked>) -> !ttg.memdesc<512x64xf16, #shared, #smem>

    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x64xf32, #acc_layout>) -> (!ttg.memdesc<128x64xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %a_sh, %b_sh, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x512xf16, #shared, #smem>, !ttg.memdesc<512x64xf16, #shared, #smem>, !ttg.memdesc<128x64xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x64xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #acc_layout>

    scf.yield %c : tensor<128x64xf32, #acc_layout>
  }
  tt.return
}

}
