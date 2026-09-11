// RUN: triton-opt --tlx-print-ttgir-to-tlx %s | FileCheck %s

// Test that tt.elementwise_inline_asm round-trips with its attributes.
//
// Four of the six arguments tl.inline_asm_elementwise takes are attributes on the
// op, so the generic printer emits only the operands and produces a call that
// looks mapped but cannot compile.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // The operands become one `args` list, and asm text, constraints, purity and
  // packing are recovered from the attributes.
  // CHECK-LABEL: def asm_single(
  // CHECK: {{[a-zA-Z_0-9]+}} = tl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [{{[a-zA-Z_0-9]+}}], tl.float32, True, 1)
  tt.func public @asm_single() attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #blocked>
    %0 = tt.elementwise_inline_asm "mov.b32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %cst : tensor<128xf32, #blocked> -> tensor<128xf32, #blocked>
    tt.return
  }

  // Multiple results bind as a tuple and dtype becomes a list.
  // CHECK-LABEL: def asm_multi(
  // CHECK: {{[a-zA-Z_0-9]+}}, {{[a-zA-Z_0-9]+}} = tl.inline_asm_elementwise({{.*}}, [tl.float32, tl.int32], True, 2)
  tt.func public @asm_multi() attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #blocked>
    %0:2 = tt.elementwise_inline_asm "mov.b32 $0, $2;" {constraints = "=r,=r,r", packed_element = 2 : i32, pure = true} %cst : tensor<128xf32, #blocked> -> tensor<128xf32, #blocked>, tensor<128xi32, #blocked>
    tt.return
  }

  // Impure asm has to stay impure, or the round-tripped kernel is free to have it
  // hoisted or eliminated.
  // CHECK-LABEL: def asm_impure(
  // CHECK: tl.inline_asm_elementwise({{.*}}, tl.float32, False, 1)
  tt.func public @asm_impure() attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #blocked>
    %0 = tt.elementwise_inline_asm "mov.b32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %cst : tensor<128xf32, #blocked> -> tensor<128xf32, #blocked>
    tt.return
  }

  // Real PTX is multi-line and quoted, so the asm text has to be escaped rather
  // than pasted into the string literal.
  // CHECK-LABEL: def asm_multiline(
  // CHECK: tl.inline_asm_elementwise("\n  .reg .b32 r;\n  mov.b32 $0, \"x\";\n", "=r,r", {{.*}})
  tt.func public @asm_multiline() attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #blocked>
    %0 = tt.elementwise_inline_asm "\0A  .reg .b32 r;\0A  mov.b32 $0, \22x\22;\0A" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %cst : tensor<128xf32, #blocked> -> tensor<128xf32, #blocked>
    tt.return
  }
}
