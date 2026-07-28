// REQUIRES: asserts
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -nvgpu-modulo-schedule -nvgpu-modulo-ws-partition | FileCheck %s
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -nvgpu-modulo-schedule -debug-only=modulo-scheduling-rau,nvgpu-modulo-schedule 2>&1 | FileCheck %s --check-prefix=JOINT

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// Verify that the modulo schedule pass annotates the inner loop and the
// ws-partition pass preserves the outer WS loop's tt.warp_specialize. The
// inner loop gets tt.modulo_ii / tt.scheduled_max_stage from modulo (exact
// values depend on the latency model, so we only check presence).
//
// CHECK-LABEL: @persistent_gemm_ws_partition
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tt.modulo_ii = {{[0-9]+}} : i32
// CHECK-SAME: tt.scheduled_max_stage = {{[0-9]+}} : i32
// CHECK: tt.warp_specialize
//
// The scheduler constructs one `(cycle, warp group)` assignment directly for
// this leaf/control case. This preserves coverage for candidate-free scheduling
// output but does not directly exercise register hand-off rejection. The real
// nested test covers non-register separation across a TMEM hand-off.
// N0-N4 fit on wg0 at their globally ready cycles. N5 cannot issue on wg0 at
// its required cycle, so the scheduler creates wg1 and places the remaining
// chain there after accounting for cross-warp synchronization latency.
// JOINT: Placed N4 {{.*}} at cycle=653 stage=0 wg=0
// JOINT-NEXT: {{.*}} Placed N5 {{.*}} at cycle=0 stage=0 wg=1
// JOINT-NEXT: {{.*}} Placed N6 {{.*}} at cycle=713 stage=0 wg=1
// JOINT-NEXT: {{.*}} Placed N7 {{.*}} at cycle=1272 stage=1 wg=1
// JOINT-NEXT: {{.*}} SUCCESS at II=1091 wgs=2 warps=9
// JOINT-NOT: [nested-wg]
// JOINT-NOT: cand[
tt.func @persistent_gemm_ws_partition(
  %a_desc: !tt.tensordesc<128x64xf16>,
  %b_desc: !tt.tensordesc<64x128xf16>,
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

      %a = tt.descriptor_load %a_desc[%c0_i32, %off_k] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
      %b = tt.descriptor_load %b_desc[%off_k, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #blocked>

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
