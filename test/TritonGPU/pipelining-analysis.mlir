// RUN: triton-opt %s -split-input-file -tritongpu-pipelining-analysis=num-stages=3 -verify-diagnostics=only-expected -o /dev/null

// Test: outer loop (contains nested loop) should emit "outer loop" remark.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @outer_loop(%lb : index, %ub : index, %step : index,
                    %ptr : tensor<128x32x!tt.ptr<f16>, #blocked>) {
  %c = arith.constant dense<4> : tensor<128x32xi32, #blocked>
  // expected-remark @below {{pipelining not applied: outer loop (contains nested loop)}}
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%p = %ptr) -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
    %inner = scf.for %jv = %lb to %ub step %step iter_args(%ip = %p) -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
      %v = tt.load %ip : tensor<128x32x!tt.ptr<f16>, #blocked>
      %np = tt.addptr %ip, %c : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      scf.yield %np : tensor<128x32x!tt.ptr<f16>, #blocked>
    }
    scf.yield %inner : tensor<128x32x!tt.ptr<f16>, #blocked>
  } {tt.num_stages = 3 : i32}
  tt.return
}
}

// -----

// Test: non-outer loop with num_stages=3 should NOT emit any remark.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @simple_loop(%lb : index, %ub : index, %step : index,
                     %ptr : tensor<128x32x!tt.ptr<f16>, #blocked>) {
  %c = arith.constant dense<4> : tensor<128x32xi32, #blocked>
  // No remark expected here.
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%p = %ptr) -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
    %v = tt.load %p : tensor<128x32x!tt.ptr<f16>, #blocked>
    %np = tt.addptr %p, %c : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
    scf.yield %np : tensor<128x32x!tt.ptr<f16>, #blocked>
  } {tt.num_stages = 3 : i32}
  tt.return
}
}

// -----

// Test: loop with num_stages=1 should NOT emit any remark (pipelining not requested).

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @num_stages_one(%lb : index, %ub : index, %step : index,
                        %ptr : tensor<128x32x!tt.ptr<f16>, #blocked>) {
  %c = arith.constant dense<4> : tensor<128x32xi32, #blocked>
  // No remark expected here.
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%p = %ptr) -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
    %v = tt.load %p : tensor<128x32x!tt.ptr<f16>, #blocked>
    %np = tt.addptr %p, %c : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
    scf.yield %np : tensor<128x32x!tt.ptr<f16>, #blocked>
  } {tt.num_stages = 1 : i32}
  tt.return
}
}
