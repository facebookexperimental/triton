// RUN: triton-opt --tlx-print-ttgir-to-tlx -split-input-file %s | FileCheck %s

// Test that tt.map_elementwise bodies are inlined as elementwise tensor ops.
//
// map_elementwise is a scheduling hint: the body is a scalar computation
// applied over the operand tensors. tl.map_elementwise takes a callable, which
// is gone once the body has been inlined into TTGIR, so emitting a flat call
// would not recompile; the inlined elementwise form is equivalent.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // The body ops are emitted in place and the op's result is bound to the
  // value the region returned.
  // CHECK-LABEL: def single_result_map(
  // CHECK: [[SUM:[a-zA-Z_0-9]+]] = arg0 + arg1
  // CHECK: {{[a-zA-Z_0-9]+}} = [[SUM]]
  // CHECK-NOT: tl.map_elementwise(
  tt.func public @single_result_map(%arg0: tensor<128xf32, #blocked>, %arg1: tensor<128xf32, #blocked>) attributes {noinline = false} {
    %r = "tt.map_elementwise"(%arg0, %arg1) <{pack = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.map_elementwise.return %s : f32
    }) : (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

// The op is variadic in its results, so every result is bound, as one tuple.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: def multi_result_map(
  // CHECK: [[SUM:[a-zA-Z_0-9]+]] = arg0 + arg1
  // CHECK: [[DIFF:[a-zA-Z_0-9]+]] = arg0 - arg1
  // CHECK: {{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}} = [[SUM]], [[DIFF]]
  tt.func public @multi_result_map(%arg0: tensor<128xf32, #blocked>, %arg1: tensor<128xf32, #blocked>) attributes {noinline = false} {
    %lo, %hi = "tt.map_elementwise"(%arg0, %arg1) <{pack = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      %d = arith.subf %a, %b : f32
      tt.map_elementwise.return %s, %d : f32, f32
    }) : (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) -> (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>)
    tt.return
  }
}

// -----

// With pack > 1 the body's block arguments are num_operands * pack scalars
// rather than one per operand, so substituting the operands for them would
// misrepresent the op. Those are left on the existing non-inlined path.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: def packed_map(
  // CHECK: tl.map_elementwise(
  tt.func public @packed_map(%arg0: tensor<128xf32, #blocked>, %arg1: tensor<128xf32, #blocked>) attributes {noinline = false} {
    %lo, %hi = "tt.map_elementwise"(%arg0, %arg1) <{pack = 2 : i32}> ({
    ^bb0(%a0: f32, %a1: f32, %b0: f32, %b1: f32):
      %s0 = arith.addf %a0, %b0 : f32
      %s1 = arith.addf %a1, %b1 : f32
      %d0 = arith.subf %a0, %b0 : f32
      %d1 = arith.subf %a1, %b1 : f32
      tt.map_elementwise.return %s0, %s1, %d0, %d1 : f32, f32, f32, f32
    }) : (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) -> (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>)
    tt.return
  }
}
