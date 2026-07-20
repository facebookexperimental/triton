// RUN: env TRITON_DATA_PARTITION_N=auto TRITON_MODULO_DUMP_SCHEDULE=%t.json triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule -o /dev/null && FileCheck %s --check-prefix=ATTR < %t.json

//===----------------------------------------------------------------------===//
// Test: an explicit per-loop tt.data_partition_factor attr wins over the
// TRITON_DATA_PARTITION_N=auto search, per-MMA.
//
// Same kernel as modulo-data-partition-auto.mlir, where the auto search
// keeps N=1 (all score terms tie). Here the loop carries
// tt.data_partition_factor = 2. The search now runs (an attr no longer
// disables it module-wide), but it resolves each MMA on its own terms: a
// pinned MMA keeps its factor in every variant AND in the baseline, so the
// search can only ever return N=2 for it.
//===----------------------------------------------------------------------===//

// This check is a true behavioral differential: on this kernel the un-pinned
// auto search keeps N=1 (see modulo-data-partition-auto.mlir), so applied_n = 2
// can only come from the pin winning — i.e. the attr overrode the search for
// this MMA.
// ATTR: "op_kind": "ttng.tc_gen5_mma", "dim": 0, "applied_n": 2, "factors": [{"n": 2, "m_size": 64}]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#acc_layout = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

tt.func @test_dp_auto_attr_bypass(
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
  } {tt.data_partition_factor = 2 : i32}

  tt.return
}

}
