// RUN: triton-opt --verify-each=false %s -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=4 target=cuda:100 num-ctas=1 threads-per-warp=32})' | FileCheck %s

// The Python frontend constructs helper bodies with encoding-free tensor types.
// A concrete call operand specializes the helper ABI during TLX fixup, so the
// existing reshape and transpose results must be re-inferred before verification.

// CHECK-DAG: #[[$SRC:.*]] = #ttg.linear<{{.*}}>
// CHECK-LABEL: tt.func private @restructure(
// CHECK-SAME: %[[ARG:.*]]: tensor<4x256xi8, #tlx.no_verify_layout<#[[$SRC]]>>)
// CHECK: %[[RESHAPE:.*]] = tt.reshape %[[ARG]]
// CHECK-SAME: -> tensor<4x4x16x2x2xi8, #tlx.no_verify_layout<#[[$RESHAPED:.*]]>>
// CHECK: %[[TRANSPOSE:.*]] = tt.trans %[[RESHAPE]]
// CHECK-SAME: -> tensor<4x2x16x2x4xi8, #tlx.no_verify_layout<#[[$TRANSPOSED:.*]]>>
// CHECK: tt.return %[[TRANSPOSE]] : tensor<4x2x16x2x4xi8, #tlx.no_verify_layout<#[[$TRANSPOSED]]>>
// CHECK-LABEL: tt.func public @kernel
// CHECK: %[[PINNED:.*]] = tlx.require_layout
// CHECK: %[[RESULT:.*]] = tt.call @restructure(%[[PINNED]])
// CHECK-SAME: -> tensor<4x2x16x2x4xi8, #tlx.no_verify_layout<#[[$TRANSPOSED]]>>

#physical = #ttg.linear<{
  register = [[0, 1], [0, 2], [0, 4]],
  lane = [[0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
  warp = [[1, 0], [2, 0]],
  block = []
}>
#pin = #tlx.no_verify_layout<#physical>

"builtin.module"() ({
  "tt.func"() <{
    function_type = (tensor<4x256xi8, #pin>) -> tensor<4x2x16x2x4xi8>,
    sym_name = "restructure",
    sym_visibility = "private"
  }> ({
  ^bb0(%arg0: tensor<4x256xi8, #pin>):
    %0 = "tt.reshape"(%arg0) <{allow_reorder}> :
      (tensor<4x256xi8, #pin>) -> tensor<4x4x16x2x2xi8>
    %1 = "tt.trans"(%0) <{order = array<i32: 0, 4, 2, 3, 1>}> :
      (tensor<4x4x16x2x2xi8>) -> tensor<4x2x16x2x4xi8>
    "tt.return"(%1) : (tensor<4x2x16x2x4xi8>) -> ()
  }) : () -> ()
  "tt.func"() <{
    function_type = (tensor<4x256xi8>) -> (),
    sym_name = "kernel",
    sym_visibility = "public"
  }> ({
  ^bb0(%arg0: tensor<4x256xi8>):
    %0 = "tlx.require_layout"(%arg0) :
      (tensor<4x256xi8>) -> tensor<4x256xi8, #pin>
    %1 = "tt.call"(%0) <{callee = @restructure}> :
      (tensor<4x256xi8, #pin>) -> tensor<4x2x16x2x4xi8>
    "tt.return"() : () -> ()
  }) : () -> ()
}) {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} : () -> ()
