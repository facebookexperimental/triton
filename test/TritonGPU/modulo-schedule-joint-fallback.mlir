//===----------------------------------------------------------------------===//
// Joint-solver terminal fallback policy (docs/Diff11FallbackPolicy.md).
//
// The load-bearing test here is the byte-identical comparison: for every
// terminal trigger, the fallback compile must produce exactly the IR a
// flag-off compile produces. That one property subsumes "fallback is atomic"
// and "no solver-owned state survives fallback" — if a single joint-solver
// cycle, warp group, buffer depth, or barrier had leaked into the rerun, the
// files would differ.
//
// Faults are injected deterministically at the backend seam
// (JointSolverScheduler.h) rather than by racing a timeout. Every RUN line
// below fails the SCHEDULE stage, so none of them needs a working Z3 backend;
// the apply-stage half, which does, lives in
// modulo-schedule-joint-fallback-apply-stage.mlir.
//===----------------------------------------------------------------------===//

// The flag-off reference. Checked to be a real, fully annotated schedule so
// the diffs below cannot pass by comparing two unscheduled files.
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-modulo-schedule -o %t.baseline.mlir
// RUN: FileCheck %s --check-prefix=BASE --input-file=%t.baseline.mlir

// Schedule stage: backend unavailable / timeout / UNKNOWN / proven global
// UNSAT / malformed model / re-verification failure.
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule=force-joint-solver-fault=schedule:unavailable -o %t.unavailable.mlir
// RUN: diff %t.baseline.mlir %t.unavailable.mlir
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule=force-joint-solver-fault=schedule:timeout -o %t.timeout.mlir
// RUN: diff %t.baseline.mlir %t.timeout.mlir
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule=force-joint-solver-fault=schedule:unknown -o %t.unknown.mlir
// RUN: diff %t.baseline.mlir %t.unknown.mlir
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule=force-joint-solver-fault=schedule:global-unsat -o %t.unsat.mlir
// RUN: diff %t.baseline.mlir %t.unsat.mlir
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule=force-joint-solver-fault=schedule:malformed -o %t.malformed.mlir
// RUN: diff %t.baseline.mlir %t.malformed.mlir
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule=force-joint-solver-fault=schedule:illegal-schedule -o %t.illegal.mlir
// RUN: diff %t.baseline.mlir %t.illegal.mlir

// The fallback is announced, and as a remark: a successful baseline compile
// must not become an error under any -Werror-like diagnostic policy.
// RUN: triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule=force-joint-solver-fault=schedule:unavailable 2>&1 | FileCheck %s --check-prefix=SCHED-REMARK

// strict-error turns the same trigger into a compilation error and does NOT
// fall back. This is the mode the golden and determinism lanes run in.
// RUN: not triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule="force-joint-solver-fault=schedule:unavailable strict-error=true" 2>&1 | FileCheck %s --check-prefix=STRICT
// RUN: not env TRITON_MODULO_STRICT_ERROR=1 triton-opt %s -allow-unregistered-dialect -nvgpu-joint-solver-schedule=force-joint-solver-fault=schedule:unavailable 2>&1 | FileCheck %s --check-prefix=STRICT

// BASE: loop.stage
// BASE: ttg.partition
// BASE: tt.modulo_ii

// SCHED-REMARK: remark: joint scheduling fell back to the baseline schedule and partition path (schedule-solve)
// STRICT: error: joint scheduling failed (schedule-solve) and the strict-error terminal policy is enabled

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
