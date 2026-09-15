// RUN: triton-opt --tlx-print-ttgir-to-tlx %s | FileCheck %s

// Test that ttng.vote_ballot_sync round-trips to its TLX spelling.
//
// Operands match, so only the name needs mapping -- and it has to be the name the
// DSL exports, not the MLIR op name with the dialect prefix swapped.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: def ballot_scalar(
  // CHECK: {{[a-zA-Z_0-9]+}} = tlx.vote_ballot_sync(-1, {{[a-zA-Z_0-9]+}})
  tt.func public @ballot_scalar(%arg0: f32) attributes {noinline = false} {
    %cst = arith.constant 1.000000e+00 : f32
    %mask = arith.constant -1 : i32
    %pred = arith.cmpf olt, %arg0, %cst : f32
    %0 = ttng.vote_ballot_sync %mask, %pred : i1 -> i32
    tt.return
  }

  // The mask is an ordinary operand, so a narrower one is carried through rather
  // than assumed to be all-lanes.
  // CHECK-LABEL: def ballot_partial_mask(
  // CHECK: tlx.vote_ballot_sync(255, {{[a-zA-Z_0-9]+}})
  tt.func public @ballot_partial_mask(%arg0: f32) attributes {noinline = false} {
    %cst = arith.constant 1.000000e+00 : f32
    %mask = arith.constant 255 : i32
    %pred = arith.cmpf olt, %arg0, %cst : f32
    %0 = ttng.vote_ballot_sync %mask, %pred : i1 -> i32
    tt.return
  }

  // The predicate may be a tensor, in which case the result is one too.
  // CHECK-LABEL: def ballot_tensor(
  // CHECK: {{[a-zA-Z_0-9]+}} = tlx.vote_ballot_sync(-1, {{[a-zA-Z_0-9]+}})
  tt.func public @ballot_tensor() attributes {noinline = false} {
    %mask = arith.constant -1 : i32
    %cst = arith.constant dense<0> : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %pred = arith.cmpi slt, %cst, %cst : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %0 = ttng.vote_ballot_sync %mask, %pred : tensor<128xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
}
