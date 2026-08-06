// RUN: triton-opt --verify-each=false %s -pass-pipeline='builtin.module(triton-tlx-fixup{num-warps=4 target=hip:gfx950 num-ctas=1 threads-per-warp=64})' | FileCheck %s --check-prefix=CURRENT
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
// Current behavior: the helper-wide scan sees tt.reshape/tt.trans and releases the
// pinned offset even though that offset flows only into amdg.buffer_load.
//
// CURRENT: %[[PINNED:.*]] = "tlx.require_layout"
// CURRENT-NEXT: %[[RELEASED:.*]] = "tlx.release_layout"(%[[PINNED]])
// CURRENT-NEXT: {{.*}} = "tt.call"({{.*}}, %[[RELEASED]])
//
// FIXED should instead preserve the pinned offset through the call and attach
// its encoding to the same-shaped load result before restructuring that result:
//
//   %pinned = tlx.require_layout %offsets
//   %result = tt.call @load_then_restructure(%base, %pinned)
//
//   // Inside @load_then_restructure:
//   %loaded = amdg.buffer_load %base[%pinned_arg]
//       : tensor<4x256xf8E4M3FN, #pin>
//   %reshaped = tt.reshape %loaded ...
//
// In particular, fixed output must have no tlx.release_layout between
// %pinned_offsets and amdg.buffer_load.

#physical = #ttg.linear<{
  register = [[0, 1], [0, 2], [1, 0], [2, 0]],
  lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]],
  warp = [[0, 0], [0, 0]],
  block = []
}>
#pin = #tlx.no_verify_layout<#tlx.user_layout<#tlx.no_verify_layout<#physical>>>

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
}
