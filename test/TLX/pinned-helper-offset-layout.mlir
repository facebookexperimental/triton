// RUN: triton-opt --verify-each=false %s -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=4 target=hip:gfx950 num-ctas=1 threads-per-warp=64})' | FileCheck %s
//
// Run manually:
//   build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt \
//     --verify-each=false test/TLX/pinned-helper-offset-layout.mlir \
//     -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=4 target=hip:gfx950 num-ctas=1 threads-per-warp=64})'
//
// The Python frontend normally enters this pass with an encoding-stripped
// helper signature and a temporarily mismatched pinned call operand. Textual
// MLIR verifies call signatures before running passes, so this reproducer uses
// the equivalent post-specialization helper signature. Verification between
// passes is disabled because the current bug makes the call invalid again by
// releasing only its operand.
//
// CHECK-DAG: #[[$TRANSPOSE_SRC_LINEAR:.*]] = #ttg.linear<{{.*}}register = {{\[\[0, 1\], \[8, 0\], \[0, 8\], \[0, 16\], \[0, 32\]\]}}
// CHECK-DAG: #[[$TRANSPOSE_SRC_USER:.*]] = #tlx.user_layout<#[[$TRANSPOSE_SRC_LINEAR]]>
// The helper restructures only the loaded value. The pinned offset flows into
// amdg.buffer_load, which is an ownership boundary for argument dataflow.
//
// CHECK-LABEL: tt.func private @load_then_restructure(
// CHECK-SAME: %[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<f8E4M3FN>, %[[OFFSETS:[a-zA-Z0-9_]+]]: tensor<4x256xi32, #[[PIN:tlx.no_verify_layout<#shared[0-9]*>]]>
// CHECK: %[[LOADED:.*]] = amdg.buffer_load %[[BASE]][%[[OFFSETS]]] : tensor<4x256xf8E4M3FN, #[[PIN]]>
// CHECK: %[[RESHAPED:.*]] = tt.reshape %[[LOADED]]
// CHECK-SAME: -> tensor<4x4x16x2x2xf8E4M3FN, #[[RESHAPE_LAYOUT:.*]]>
// CHECK: %[[TRANSPOSED:.*]] = tt.trans %[[RESHAPED]]
// CHECK-SAME: -> tensor<4x2x16x2x4xf8E4M3FN, #[[TRANS_LAYOUT:.*]]>
// CHECK: tt.return {{.*}} : tensor<128x8xf8E4M3FN, #[[RETURN_LAYOUT:.*]]>
// CHECK-LABEL: tt.func public @kernel
// CHECK: %[[PINNED:.*]] = tlx.require_layout
// CHECK-NOT: tlx.release_layout
// CHECK: tt.call @load_then_restructure(%{{.*}}, %[[PINNED]])
// CHECK-LABEL: tt.func private @explicit_release
// CHECK: %[[RELEASED:.*]] = tlx.release_layout
// CHECK: tt.return %[[RELEASED]] : tensor<4x256xi32>
// CHECK-LABEL: tt.func public @explicit_release_kernel
// CHECK: %[[EXPLICIT_PIN:.*]] = tlx.require_layout
// CHECK: %[[UNPINNED:.*]] = tt.call @explicit_release(%[[EXPLICIT_PIN]])
// CHECK-SAME: -> tensor<4x256xi32>
// CHECK: tt.return
// CHECK-LABEL: tt.func public @transpose_result_pin
// CHECK: %[[TRANSPOSE_VALUE:.*]] = arith.constant
// CHECK-SAME: tensor<128x64xf32, #tlx.no_verify_layout<#[[$TRANSPOSE_SRC_USER]]>>
// CHECK: tt.trans %[[TRANSPOSE_VALUE]]
// CHECK: tt.return
//
// Full post-fixup snapshots follow. The current module uses generic syntax
// because releasing only the call operand makes the call temporarily invalid.
// The desired example uses custom syntax and normalized SSA names for clarity.

// ============================== PRE-FIX GENERATED OUTPUT ==============================
//
// #linear = #ttg.linear<{register = [[0, 1], [0, 2], [1, 0], [2, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], warp = [[0, 0], [0, 0]], block = []}>
// #shared = #tlx.user_layout<#tlx.no_verify_layout<#linear>>
// "builtin.module"() ({
//   "tt.func"() <{function_type = (!tt.ptr<f8E4M3FN>, tensor<4x256xi32, #tlx.no_verify_layout<#shared>>) -> tensor<128x8xf8E4M3FN>, sym_name = "load_then_restructure", sym_visibility = "private"}> ({
//   ^bb0(%arg2: !tt.ptr<f8E4M3FN>, %arg3: tensor<4x256xi32, #tlx.no_verify_layout<#shared>>):
//     %3 = "amdg.buffer_load"(%arg2, %arg3) <{cache = 1 : i32, contiguity = 1 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (!tt.ptr<f8E4M3FN>, tensor<4x256xi32, #tlx.no_verify_layout<#shared>>) -> tensor<4x256xf8E4M3FN, #tlx.no_verify_layout<#shared>>
//     %4 = "tt.reshape"(%3) <{allow_reorder}> : (tensor<4x256xf8E4M3FN, #tlx.no_verify_layout<#shared>>) -> tensor<4x4x16x2x2xf8E4M3FN>
//     %5 = "tt.trans"(%4) <{order = array<i32: 0, 4, 2, 3, 1>}> : (tensor<4x4x16x2x2xf8E4M3FN>) -> tensor<4x2x16x2x4xf8E4M3FN>
//     %6 = "tt.reshape"(%5) <{allow_reorder}> : (tensor<4x2x16x2x4xf8E4M3FN>) -> tensor<128x8xf8E4M3FN>
//     "tt.return"(%6) : (tensor<128x8xf8E4M3FN>) -> ()
//   }) : () -> ()
//   "tt.func"() <{function_type = (!tt.ptr<f8E4M3FN>, tensor<4x256xi32>) -> (), sym_name = "kernel", sym_visibility = "public"}> ({
//   ^bb0(%arg0: !tt.ptr<f8E4M3FN>, %arg1: tensor<4x256xi32>):
//     %0 = "tlx.require_layout"(%arg1) : (tensor<4x256xi32>) -> tensor<4x256xi32, #tlx.no_verify_layout<#shared>>
//     %1 = "tlx.release_layout"(%0) : (tensor<4x256xi32, #tlx.no_verify_layout<#shared>>) -> tensor<4x256xi32>
//     %2 = "tt.call"(%arg0, %1) <{callee = @load_then_restructure}> : (!tt.ptr<f8E4M3FN>, tensor<4x256xi32>) -> tensor<128x8xf8E4M3FN>
//     "tt.return"() : () -> ()
//   }) : () -> ()
// }) {tlx.has_tlx_ops = true, triton.skip_generic_pipeline, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} : () -> ()
//
// ================================= FIXED OUTPUT =================================
//
// #linear = #ttg.linear<{register = [[0, 1], [0, 2], [1, 0], [2, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], warp = [[0, 0], [0, 0]], block = []}>
// #shared = #tlx.user_layout<#tlx.no_verify_layout<#linear>>
// module attributes {tlx.has_tlx_ops = true, triton.skip_generic_pipeline, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
//   tt.func private @load_then_restructure(
//       %base: !tt.ptr<f8E4M3FN>,
//       %pinned_offsets: tensor<4x256xi32, #tlx.no_verify_layout<#shared>>)
//       -> tensor<128x8xf8E4M3FN> {
//     %loaded = amdg.buffer_load %base[%pinned_offsets]
//         : tensor<4x256xf8E4M3FN, #tlx.no_verify_layout<#shared>>
//     %reshaped = tt.reshape %loaded allow_reorder
//         : tensor<4x256xf8E4M3FN, #tlx.no_verify_layout<#shared>>
//           -> tensor<4x4x16x2x2xf8E4M3FN>
//     %transposed = tt.trans %reshaped {order = array<i32: 0, 4, 2, 3, 1>}
//         : tensor<4x4x16x2x2xf8E4M3FN> -> tensor<4x2x16x2x4xf8E4M3FN>
//     %result = tt.reshape %transposed allow_reorder
//         : tensor<4x2x16x2x4xf8E4M3FN> -> tensor<128x8xf8E4M3FN>
//     tt.return %result : tensor<128x8xf8E4M3FN>
//   }
//
//   tt.func public @kernel(
//       %base: !tt.ptr<f8E4M3FN>,
//       %offsets: tensor<4x256xi32>) {
//     %pinned_offsets = tlx.require_layout %offsets
//         : tensor<4x256xi32>
//           -> tensor<4x256xi32, #tlx.no_verify_layout<#shared>>
//     %result = tt.call @load_then_restructure(%base, %pinned_offsets)
//         : (!tt.ptr<f8E4M3FN>, tensor<4x256xi32, #tlx.no_verify_layout<#shared>>)
//           -> tensor<128x8xf8E4M3FN>
//     tt.return
//   }
// }
//
// The semantic difference is the absent tlx.release_layout in @kernel. The load
// is the ownership boundary: its result retains the same-shaped pin, while only
// that loaded result is subsequently reshaped and transposed.

#physical = #ttg.linear<{
  register = [[0, 1], [0, 2], [1, 0], [2, 0]],
  lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
  warp = [[0, 0], [0, 0]],
  block = []
}>
#pin = #tlx.no_verify_layout<#tlx.user_layout<#tlx.no_verify_layout<#physical>>>
#transpose_physical = #ttg.linear<{
  register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0]],
  lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]],
  warp = [[0, 32], [0, 64], [0, 16]],
  block = []
}>
#transpose_pin = #tlx.no_verify_layout<#tlx.user_layout<#transpose_physical>>
#transpose_wrong_src_pin = #tlx.no_verify_layout<#tlx.user_layout<#transpose_physical>>

module {
  tt.func private @load_then_restructure(
      %base: !tt.ptr<f8E4M3FN>,
      %offsets: tensor<4x256xi32, #pin>) -> tensor<128x8xf8E4M3FN> {
    %loaded = amdg.buffer_load %base[%offsets] : tensor<4x256xf8E4M3FN, #pin>
    %reshaped = tt.reshape %loaded allow_reorder : tensor<4x256xf8E4M3FN, #pin> -> tensor<4x4x16x2x2xf8E4M3FN>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 4, 2, 3, 1>} : tensor<4x4x16x2x2xf8E4M3FN> -> tensor<4x2x16x2x4xf8E4M3FN>
    %result = tt.reshape %transposed allow_reorder : tensor<4x2x16x2x4xf8E4M3FN> -> tensor<128x8xf8E4M3FN>
    tt.return %result : tensor<128x8xf8E4M3FN>
  }

  tt.func public @kernel(
      %base: !tt.ptr<f8E4M3FN>,
      %offsets: tensor<4x256xi32>) {
    %pinned_offsets = tlx.require_layout %offsets
        : tensor<4x256xi32> -> tensor<4x256xi32, #pin>
    %result = tt.call @load_then_restructure(%base, %pinned_offsets)
        : (!tt.ptr<f8E4M3FN>, tensor<4x256xi32, #pin>) -> tensor<128x8xf8E4M3FN>
    tt.return
  }

  tt.func private @explicit_release(
      %value: tensor<4x256xi32, #pin>) -> tensor<4x256xi32> {
    %released = tlx.release_layout %value
        : tensor<4x256xi32, #pin> -> tensor<4x256xi32>
    tt.return %released : tensor<4x256xi32>
  }

  tt.func public @explicit_release_kernel(%value: tensor<4x256xi32>) {
    %pinned = tlx.require_layout %value
        : tensor<4x256xi32> -> tensor<4x256xi32, #pin>
    %released = tt.call @explicit_release(%pinned)
        : (tensor<4x256xi32, #pin>) -> tensor<4x256xi32>
    tt.return
  }

  tt.func public @transpose_result_pin() {
    %value = arith.constant dense<0.0> : tensor<128x64xf32, #transpose_wrong_src_pin>
    %transposed = tt.trans %value {order = array<i32: 1, 0>}
        : tensor<128x64xf32, #transpose_wrong_src_pin>
          -> tensor<64x128xf32, #transpose_pin>
    tt.return
  }
}
