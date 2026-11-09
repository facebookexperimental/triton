// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -triton-npot-pad=target=hip:gfx950 \
// RUN:   -convert-triton-to-tritongpu="target=hip:gfx950 num-warps=4 threads-per-warp=64" \
// RUN:   -tritongpu-coalesce -tritonamdgpu-coalesce-buffer-ops \
// RUN:   -tritongpu-remove-layout-conversions -tritongpu-F32DotTC \
// RUN:   -tritongpu-remove-layout-conversions -tritongpu-optimize-thread-locality \
// RUN:   -tritonamdgpu-lower-barrier-ops -tritonamdgpu-accelerate-matmul="gfx-arch=gfx950" \
// RUN:   -tlx-insert-require-layout -tlx-propagate-layout -tlx-storage-alias-lowering \
// RUN:   -tlx-resolve-placeholder-layouts -tlx-rewrite-local-alias \
// RUN:   -tritongpu-remove-layout-conversions -tritonamdgpu-optimize-epilogue \
// RUN:   -tritonamdgpu-optimize-dot-operands="gfx-arch=gfx950" \
// RUN:   -tritonamdgpu-hoist-layout-conversions -tritonamdgpu-sink-layout-conversions \
// RUN:   -tritongpu-fuse-nested-loops -canonicalize -triton-licm -canonicalize \
// RUN:   -tritonamdgpu-optimize-descriptor-encoding -tritonamdgpu-schedule-loops="num_stages=2" \
// RUN:   -tritonamdgpu-pipeline="use_async_copy=true use_pingpong=true" \
// RUN:   -tritonamdgpu-coalesce-async-copy="gfx-arch=gfx950 use_async_copy=true" \
// RUN:   -tritonamdgpu-convert-tensor-ops -canonicalize -tritongpu-remove-layout-conversions \
// RUN:   -tritongpu-reduce-data-duplication -tlx-finalize-user-layouts \
// RUN:   -tritonamdgpu-move-up-prologue-loads -tritonamdgpu-block-pingpong="num-stages=2" \
// RUN:   -tritonamdgpu-canonicalize-pointers -canonicalize \
// RUN:   -tritonamdgpu-convert-buffer-ops="gfx-arch=gfx950" -tritonamdgpu-optimize-buffer-op-ptr \
// RUN:   -tritonamdgpu-prepare-if-combining -canonicalize -tlx-propagate-layout -cse -symbol-dce \
// RUN:   -tritonamdgpu-normalize-warp-pipeline -tritonamdgpu-warp-pipeline -tlx-dump-layout \
// RUN:   | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -triton-npot-pad=target=hip:gfx950 \
// RUN:   -convert-triton-to-tritongpu="target=hip:gfx950 num-warps=4 threads-per-warp=64" \
// RUN:   -tritonamdgpu-accelerate-matmul="gfx-arch=gfx950" -tlx-insert-require-layout \
// RUN:   | FileCheck %s --check-prefix=INSERT
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -triton-npot-pad=target=hip:gfx950 \
// RUN:   -tritonamdgpu-schedule-loops -verify-diagnostics -o /dev/null

// make_ttgir's gfx950 order minus intra-wave-pipeline (not in triton-opt).
// TLX insert keeps the exact NPOT operand convert, not a require_layout the
// verifier rejects. K=96 uses the padded 32x32x16 MFMA; no module attribute.
// CHECK: #mma = #ttg.amd_mfma<{version = 4, {{.*}}instrShape = [32, 32, 16]
// CHECK: module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
// CHECK-LABEL: @dot_k96
// CHECK: tt.dot {{.*}} : tensor<128x96xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<96x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
// INSERT-LABEL: @dot_k96
// INSERT-NOT: tlx.require_layout
// INSERT: ttg.convert_layout {{.*}} -> tensor<128x96xbf16, #ttg.dot_op<{opIdx = 0
// INSERT: ttg.convert_layout {{.*}} -> tensor<96x128xbf16, #ttg.dot_op<{opIdx = 1
// INSERT-NOT: tlx.require_layout
// INSERT: tt.dot
tt.func @dot_k96(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>, %pc: !tt.ptr<f32>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x96x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x96x!tt.ptr<bf16>>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<96x128x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<96x128x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x96xbf16> * tensor<96x128xbf16> -> tensor<128x128xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x128x!tt.ptr<f32>>
  tt.return
}

// fp8 K=96 pads: its pow2-padded K's f8f6f4 MFMA has kDim 64.
// CHECK-LABEL: @dot_k96_fp8
// CHECK: tt.dot_scaled {{.*}} : tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma{{.*}}, kWidth = 16}>>
tt.func @dot_k96_fp8(%pa: !tt.ptr<f8E4M3FN>, %pb: !tt.ptr<f8E4M3FN>, %pc: !tt.ptr<f32>) {
  %sa = tt.splat %pa : !tt.ptr<f8E4M3FN> -> tensor<128x96x!tt.ptr<f8E4M3FN>>
  %a = tt.load %sa : tensor<128x96x!tt.ptr<f8E4M3FN>>
  %sb = tt.splat %pb : !tt.ptr<f8E4M3FN> -> tensor<96x128x!tt.ptr<f8E4M3FN>>
  %b = tt.load %sb : tensor<96x128x!tt.ptr<f8E4M3FN>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  // expected-remark @+1 {{the pow2-padded K's MFMA (rocdl.mfma.scale.f32.32x32x64.f8f6f4, kDim 64) doesn't divide K=96}}
  %d = tt.dot %a, %b, %c : tensor<128x96xf8E4M3FN> * tensor<96x128xf8E4M3FN> -> tensor<128x128xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x128x!tt.ptr<f32>>
  tt.return
}

// Control: a pow2 K still gets the require_layout anchors.
// INSERT-LABEL: @dot_k128
// INSERT-COUNT-2: tlx.require_layout
// INSERT: tt.dot
tt.func @dot_k128(%pa: !tt.ptr<bf16>, %pb: !tt.ptr<bf16>, %pc: !tt.ptr<f32>) {
  %sa = tt.splat %pa : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x128x!tt.ptr<bf16>>
  %sb = tt.splat %pb : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>>
  %b = tt.load %sb : tensor<128x128x!tt.ptr<bf16>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
  %sc = tt.splat %pc : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x128x!tt.ptr<f32>>
  tt.return
}
