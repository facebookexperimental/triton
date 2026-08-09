// REQUIRES: z3-joint-solver
//===----------------------------------------------------------------------===//
// Joint-solver fallback policy, APPLY stage (docs/Diff11FallbackPolicy.md).
//
// The companion file (modulo-schedule-joint-fallback.mlir) fails the schedule
// solve, which the old in-place Rau fall-through already handled. These cases
// are the ones the hoist was for: the schedule solve genuinely SUCCEEDS and
// the warp-group partition fails afterwards. Before the policy moved above
// both stages, that fell through to a heuristic partition sitting on a joint
// MinII schedule — the mixed state — and produced IR matching neither
// reference.
//
// EVERY RUN LINE HERE NEEDS A LIVE Z3 BACKEND, which is why they are split out
// into their own file and gated: the `partition:` fault scope deliberately
// routes the schedule-stage request (joint-solver-0.1) to the real solver, and
// the clean run needs a real solve to succeed at all.
//
// `z3-joint-solver` is declared by test/lit.cfg.py from the CMake option
// (default OFF) and by BUCK.template's lit rule for the Buck build, which
// always compiles the solver in — so this file runs there and is skipped in a
// stock CMake build.
//===----------------------------------------------------------------------===//

// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule -o %t.baseline.mlir
// RUN: FileCheck %s --check-prefix=BASE --input-file=%t.baseline.mlir

// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule=force-joint-solver-fault=partition:unavailable -o %t.partition.mlir
// RUN: diff %t.baseline.mlir %t.partition.mlir
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule=force-joint-solver-fault=partition:unavailable 2>&1 | FileCheck %s --check-prefix=PART-REMARK

// A healthy solve must not trip the policy, even though the II sweep proves
// candidate IIs infeasible along the way — a per-II UNSAT is the search making
// progress, not a terminal outcome. Under strict-error that distinction is the
// difference between compiling and failing.
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule 2>&1 | FileCheck %s --check-prefix=CLEAN --implicit-check-not="fell back to the baseline"
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule="strict-error=true" 2>&1 | FileCheck %s --check-prefix=CLEAN --implicit-check-not="error:"

// BASE: loop.stage
// BASE: ttg.partition
// BASE: tt.modulo_ii

// PART-REMARK: remark: joint scheduling fell back to the baseline schedule and partition path (partition-solve)
// CLEAN: tt.modulo_ii

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

tt.func @gemm_inner_loop(
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
