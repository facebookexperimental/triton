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

// --- Graph structure: II=1038, max_stage=2, trip_count=32 ---
// CHECK: [PASS-A] === Inner Loop ScheduleGraph ===
// CHECK-NEXT: modulo.schedule @loop0 {
// CHECK-NEXT:   ii = 1038, max_stage = 2, prologue_latency = 1038, trip_count = 32
//
// --- Nodes: loads+allocs@s0, MMA@s1, tmem_load@s2 with cluster IDs ---
// CHECK: modulo.stage @s0 {
// CHECK:   tt.descriptor_load  {pipe: MEM, cycle: 0, cluster: 0, latency: 1218, selfLatency: 518}
// CHECK:   tt.descriptor_load  {pipe: MEM, cycle: 518, cluster: 1, latency: 1218, selfLatency: 518}
// CHECK:   ttg.local_alloc  {pipe: MEM, cycle: 1036, cluster: 2, latency: 700
// CHECK:   ttg.local_alloc  {pipe: MEM, cycle: 1037, cluster: 3, latency: 700
// CHECK: }
// CHECK: modulo.stage @s1 {
// CHECK:   ttng.tc_gen5_mma  {pipe: TC, cycle: 1737, cluster: 0, latency: 900, selfLatency: 900
// CHECK: }
// CHECK: modulo.stage @s2 {
// CHECK:   ttng.tmem_load  {pipe: CUDA, cycle: 2637, cluster: 0, latency: 130, selfLatency: 130
// CHECK: }
//
// --- Edges: SSA + loop-carried ---
// CHECK: edges {
// CHECK-DAG: N0 -> N1  lat=0  dist=0
// CHECK-DAG: N0 -> N2  lat=0  dist=0
// CHECK-DAG: N1 -> N3  lat=518  dist=0
// CHECK-DAG: N2 -> N4  lat=518  dist=0
// CHECK-DAG: N3 -> N6  lat=700  dist=0
// CHECK-DAG: N4 -> N6  lat=700  dist=0
// CHECK-DAG: N5 -> N6  lat=0  dist=0
// CHECK-DAG: N5 -> N7  lat=0  dist=0
// CHECK-DAG: N6 -> N7  lat=900  dist=0
// CHECK-DAG: N7 -> N5  lat=130  dist=1
// CHECK: }
// CHECK: }
tt.func @test_basic_graph(
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
