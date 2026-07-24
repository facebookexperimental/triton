// REQUIRES: asserts
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule -debug-only=nvgpu-modulo-schedule 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: Basic ScheduleGraph — graph structure, nodes, and edges
//===----------------------------------------------------------------------===//

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// --- Graph structure: II=1091, max_stage=1, trip_count=32 ---
// With load selfLatency=30, loads issue every 30 cycles (not every 556 cycles),
// so II is driven by RecMII (loop-carried dep: MMA→tmem_load→tmem_alloc→MMA).
// CHECK: [PASS-A] === Inner ScheduleGraph ===
// CHECK-NEXT: modulo.schedule @loop0 {
// CHECK-NEXT:   ii = 1091, max_stage = 1, prologue_latency = 587, trip_count = 32
//
// --- Nodes: loads+allocs+MMA@s0, tmem_load@s1 ---
// CHECK: modulo.stage @s0 {
// CHECK:   %N1 = tt.descriptor_load  {pipe: TMA, cycle: 1, cluster: 1, latency: 556, selfLatency: 30}
// CHECK:   %N2 = tt.descriptor_load  {pipe: TMA, cycle: 31, cluster: 2, latency: 556, selfLatency: 30}
// CHECK:   %N3 = ttg.local_alloc  {pipe: NONE, cycle: 557, cluster: 3, ->buf0}
// CHECK:   %N4 = ttg.local_alloc  {pipe: NONE, cycle: 587, cluster: 4, ->buf1}
// CHECK:   %N6 = ttng.tc_gen5_mma  {pipe: TC, cycle: 587, cluster: 4, latency: 559, selfLatency: 30, <-buf0, <-buf1, <-buf2}
// CHECK: }
// CHECK: modulo.stage @s1 {
// CHECK:   %N7 = ttng.tmem_load  {pipe: CUDA, cycle: 1146, cluster: 0, latency: 532, selfLatency: 256, <-buf2}
// CHECK: }
//
// --- Edges: SSA + loop-carried ---
// CHECK: edges {
// CHECK-DAG: N0 -> N1  lat=1  dist=0
// CHECK-DAG: N0 -> N2  lat=1  dist=0
// CHECK-DAG: N1 -> N3  lat=556  dist=0
// CHECK-DAG: N2 -> N4  lat=556  dist=0
// CHECK-DAG: N3 -> N6  lat=0  dist=0
// CHECK-DAG: N4 -> N6  lat=0  dist=0
// CHECK-DAG: N5 -> N6  lat=0  dist=0
// CHECK-DAG: N5 -> N7  lat=0  dist=0
// CHECK-DAG: N6 -> N7  lat=559  dist=0
// CHECK-DAG: N7 -> N5  lat=532  dist=1
// CHECK: }
// CHECK: }
tt.func @test_basic_graph(
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
