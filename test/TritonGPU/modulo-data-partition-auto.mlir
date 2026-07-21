// RUN: env TRITON_MODULO_DUMP_SCHEDULE=%t.json triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule -o /dev/null && FileCheck %s --check-prefix=SURFACE < %t.json
// RUN: env TRITON_DATA_PARTITION_N=auto TRITON_MODULO_DUMP_SCHEDULE=%t2.json triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule -o /dev/null && FileCheck %s --check-prefix=AUTO < %t2.json

//===----------------------------------------------------------------------===//
// Test: A.5 auto factor search — the tie-keeps-baseline path.
//
// The accumulator is 128 rows with blockM = 64, so BOTH configurations are
// TMEM-legal: baseline (128 rows <= 128 lanes) and the N=2 split
// (m_size = 64 >= minM = blockM = 64). The M-split conserves MAC area
// (occupancy = max(occ_full, N x issue floor) stays occ_full), the loop
// schedules either way, and no SMEM ring is budget-reduced — so all four
// score terms tie and the search must keep the incumbent baseline (N=1).
// This is the ONLY corpus input that exercises the tie chain: the example
// cases are either candidate-free (BM=128, blockM=128) or decided at the
// legality term (case2's BM=256).
//===----------------------------------------------------------------------===//

// --- Candidate surface: the N=2 split is legal and reported, not applied.
// --- This run also guards the dump-mode self-trigger: TRITON_MODULO_DUMP_SCHEDULE
// --- is set but no TRITON_DATA_PARTITION_N, and the baseline (128 rows) is
// --- TLX-legal, so the self-trigger must NOT fire — applied_n stays 1. ---
// SURFACE: "op_kind": "ttng.tc_gen5_mma", "dim": 0, "applied_n": 1, "factors": [{"n": 2, "m_size": 64}]

// --- Under auto: same dump — the search ran and kept N=1. Non-vacuous by
// --- construction: the SURFACE check proves the factor set is non-empty,
// --- and with env=auto and no explicit attr the search's only
// --- exit-before-evaluate is an empty factor set — so N=2 was evaluated
// --- and lost the tie. ---
// AUTO: "op_kind": "ttng.tc_gen5_mma", "dim": 0, "applied_n": 1, "factors": [{"n": 2, "m_size": 64}]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#acc_layout = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

tt.func @test_dp_auto_tie(
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
