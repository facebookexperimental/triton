// RUN: triton-opt %s -split-input-file -tritongpu-pipelining-analysis=num-stages=3 -verify-diagnostics=only-expected -o /dev/null

// Test: loop with num_stages=1 should emit "no pipelining requested" remark.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @num_stages_one(%lb : index, %ub : index, %step : index,
                        %ptr : tensor<128x32x!tt.ptr<f16>, #blocked>) {
  %c = arith.constant dense<4> : tensor<128x32xi32, #blocked>
  // expected-remark @below {{pipelining not applied: num_stages is 1, no pipelining requested}}
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%p = %ptr) -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
    %v = tt.load %p : tensor<128x32x!tt.ptr<f16>, #blocked>
    %np = tt.addptr %p, %c : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
    scf.yield %np : tensor<128x32x!tt.ptr<f16>, #blocked>
  } {tt.num_stages = 1 : i32}
  tt.return
}
}

// -----

// Test: successfully pipelined loop (has tt.scheduled_max_stage) should NOT emit any remark.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @already_pipelined(%lb : index, %ub : index, %step : index,
                           %ptr : tensor<128x32x!tt.ptr<f16>, #blocked>) {
  %c = arith.constant dense<4> : tensor<128x32xi32, #blocked>
  // No remark expected here.
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%p = %ptr) -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
    %v = tt.load %p : tensor<128x32x!tt.ptr<f16>, #blocked>
    %np = tt.addptr %p, %c : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
    scf.yield %np : tensor<128x32x!tt.ptr<f16>, #blocked>
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

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
  }
  tt.return
}
}

// -----

// Test: loop with no latency-assigned ops should emit "no latency assigned" remark.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
tt.func @no_latency(%lb : index, %ub : index, %step : index,
                    %ptr : tensor<128x32x!tt.ptr<f16>, #blocked>) {
  %c = arith.constant dense<4> : tensor<128x32xi32, #blocked>
  // expected-remark @below {{pipelining not applied: no latency assigned to any op in loop}}
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%p = %ptr) -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
    %v = tt.load %p : tensor<128x32x!tt.ptr<f16>, #blocked>
    %np = tt.addptr %p, %c : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
    scf.yield %np : tensor<128x32x!tt.ptr<f16>, #blocked>
  }
  tt.return
}
}
