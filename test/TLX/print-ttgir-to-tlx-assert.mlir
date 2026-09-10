// RUN: triton-opt --tlx-print-ttgir-to-tlx %s | FileCheck %s

// Test that tt.assert round-trips to tl.device_assert with its message.
//
// Unmapped, this prints as `tt.assert(cond)`, and `assert` being a Python keyword
// makes that a syntax error that takes the whole file down, not one bad name.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: def assert_scalar(
  // CHECK: tl.device_assert([[C:[a-zA-Z_0-9]+]], "index out of range")
  // CHECK-NOT: tt.assert
  tt.func public @assert_scalar(%arg0: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %cond = arith.cmpi sge, %arg0, %c0_i32 : i32
    tt.assert %cond, "index out of range" : i1
    tt.return
  }

  // A message carrying characters that are not legal inside a Python string
  // literal has to be escaped, not passed through. DEL is a control character
  // above the C0 range, so it needs escaping too.
  // CHECK-LABEL: def assert_quoted_message(
  // CHECK: tl.device_assert({{[a-zA-Z_0-9]+}}, "bad \"x\":\\y\nz\x7f")
  tt.func public @assert_quoted_message(%arg0: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %cond = arith.cmpi sge, %arg0, %c0_i32 : i32
    tt.assert %cond, "bad \22x\22:\5Cy\0Az\7F" : i1
    tt.return
  }

  // The condition is commonly a tensor rather than a scalar.
  // CHECK-LABEL: def assert_tensor(
  // CHECK: tl.device_assert({{[a-zA-Z_0-9]+}}, "mask must hold")
  tt.func public @assert_tensor() attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %cond = arith.cmpi sge, %cst, %cst : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    tt.assert %cond, "mask must hold" : tensor<128xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
}
