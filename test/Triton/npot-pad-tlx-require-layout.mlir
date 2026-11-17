// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad='num-warps=4' -verify-diagnostics | FileCheck %s

// A layout anchor forwards its shape: an NPOT tile through
// tlx.require_layout keeps its exact size instead of padding the whole
// connected component.
// CHECK-LABEL: @require_layout_96
tt.func @require_layout_96(%pa: !tt.ptr<f16>, %po: !tt.ptr<f16>) {
  // CHECK: tlx.require_layout %{{.*}} : tensor<96x32xf16> -> tensor<96x32xf16>
  // CHECK: tt.store %{{.*}} : tensor<96x32x!tt.ptr<f16>>
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<96x32x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<96x32x!tt.ptr<f16>>
  %r = tlx.require_layout %a : tensor<96x32xf16> -> tensor<96x32xf16>
  %so = tt.splat %po : !tt.ptr<f16> -> tensor<96x32x!tt.ptr<f16>>
  tt.store %so, %r : tensor<96x32x!tt.ptr<f16>>
  tt.return
}

// -----

// Control: a power-of-two tile is unchanged.
// CHECK-LABEL: @require_layout_128
tt.func @require_layout_128(%pa: !tt.ptr<f16>, %po: !tt.ptr<f16>) {
  // CHECK: tlx.require_layout %{{.*}} : tensor<128x32xf16> -> tensor<128x32xf16>
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<128x32x!tt.ptr<f16>>
  %r = tlx.require_layout %a : tensor<128x32xf16> -> tensor<128x32xf16>
  %so = tt.splat %po : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
  tt.store %so, %r : tensor<128x32x!tt.ptr<f16>>
  tt.return
}

// -----

// A strict release anchor forwards its shape the same way.
// CHECK-LABEL: @release_layout_96
tt.func @release_layout_96(%pa: !tt.ptr<f16>, %po: !tt.ptr<f16>) {
  // CHECK: tlx.release_layout %{{.*}} : tensor<96x32xf16> -> tensor<96x32xf16>
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<96x32x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<96x32x!tt.ptr<f16>>
  %r = tlx.release_layout %a : tensor<96x32xf16> -> tensor<96x32xf16>
  %so = tt.splat %po : !tt.ptr<f16> -> tensor<96x32x!tt.ptr<f16>>
  tt.store %so, %r : tensor<96x32x!tt.ptr<f16>>
  tt.return
}

// -----

// An anchor whose encoded result is legal for its NPOT shape (wgmma:
// M % 64 == 0, N % 8 == 0) is admitted and keeps its exact size.
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8, 16]}>
tt.func @require_layout_mma_104() {
  %c = arith.constant dense<0.0> : tensor<64x104xf32>
  // CHECK: tlx.require_layout %{{.*}} : tensor<64x104xf32> -> tensor<64x104xf32, #mma>
  %r = tlx.require_layout %c : tensor<64x104xf32> -> tensor<64x104xf32, #mma>
  tt.return
}

// -----

// Control: N=100 is not a multiple of 8, so the MMA encoding is
// illegal for the shape and the anchor still rejects.
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8, 16]}>
tt.func @require_layout_mma_100() {
  %c = arith.constant dense<0.0> : tensor<64x100xf32>
  // expected-remark @+2 {{NPOT mixed-radix layout is not yet supported for tensor<64x100xf32}}
  // expected-error @+1 {{on an NPOT pointer-to-tensor, descriptor, memory descriptor or encoded tensor is unsupported}}
  %r = tlx.require_layout %c : tensor<64x100xf32> -> tensor<64x100xf32, #mma>
  tt.return
}
