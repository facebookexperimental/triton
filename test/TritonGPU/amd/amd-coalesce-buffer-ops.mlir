// RUN: triton-opt %s -split-input-file --tritonamdgpu-coalesce-buffer-ops | FileCheck %s

// Test 1: 1D buffer_load f32 with suboptimal sizePerThread=1 is upgraded to 4.
// ptr divisibility = 16 bytes → ptrAlign = 16 / 4 = 4
// offset divisibility = 16 bytes → offsetAlign = 16 / 4 = 4
// maxVec = 128 / 32 = 4; perThread = min(4, 4, 4) = 4
// fair-share cap = 1024 / (4 warps * 64 threads) = 4 → final perThread = 4

// CHECK: #[[$BLOCKED4:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: buffer_load_f32_vectorize
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_f32_vectorize(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<1024xi32, #blocked1> {tt.divisibility = 16 : i32}) {
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<1024xi32, #blocked{{.*}}> -> tensor<1024xi32, #[[$BLOCKED4]]>
    // CHECK: %{{.*}} = amdg.buffer_load %{{.*}}[%{{.*}}] : tensor<1024xf32, #[[$BLOCKED4]]>
    %result = amdg.buffer_load %ptr[%offsets] : tensor<1024xf32, #blocked1>
    tt.return
  }
}

// -----

// Test 2: 1D buffer_load f16 capped by fair-share, not by maxVec.
// ptr divisibility = 16 bytes → ptrAlign = 16 / 2 = 8
// offset divisibility = 16 bytes → offsetAlign = 16 / 2 = 8
// maxVec = 128 / 16 = 8; perThread = min(8, 8, 8) = 8
// fair-share cap = 1024 / (4 * 64) = 4 → final perThread = 4

// CHECK: #[[$BLOCKED4_F16:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: buffer_load_f16_fairshare_cap
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_f16_fairshare_cap(
      %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %offsets: tensor<1024xi32, #blocked2> {tt.divisibility = 16 : i32}) {
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<1024xi32, #blocked{{.*}}> -> tensor<1024xi32, #[[$BLOCKED4_F16]]>
    // CHECK: %{{.*}} = amdg.buffer_load %{{.*}}[%{{.*}}] : tensor<1024xf16, #[[$BLOCKED4_F16]]>
    %result = amdg.buffer_load %ptr[%offsets] : tensor<1024xf16, #blocked2>
    tt.return
  }
}

// -----

// Test 3: 1D buffer_load f16 reaches the 128-bit max vector width (8 elements).
// ptr divisibility = 16 bytes → ptrAlign = 16 / 2 = 8
// offset divisibility = 16 bytes → offsetAlign = 16 / 2 = 8
// maxVec = 128 / 16 = 8; perThread = min(8, 8, 8) = 8
// fair-share cap = 16384 / (4 * 64) = 64 → final perThread = 8

// CHECK: #[[$BLOCKED8:.*]] = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: buffer_load_f16_max_vec
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_f16_max_vec(
      %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %offsets: tensor<16384xi32, #blocked3> {tt.divisibility = 16 : i32}) {
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<16384xi32, #blocked{{.*}}> -> tensor<16384xi32, #[[$BLOCKED8]]>
    // CHECK: %{{.*}} = amdg.buffer_load %{{.*}}[%{{.*}}] : tensor<16384xf16, #[[$BLOCKED8]]>
    %result = amdg.buffer_load %ptr[%offsets] : tensor<16384xf16, #blocked3>
    tt.return
  }
}

// -----

// Test 4: 1D buffer_store f32 — value, offsets, and mask all get convert_layout.
// ptr divisibility = 16 bytes → ptrAlign = 4
// offset divisibility = 16 bytes → offsetAlign = 4
// perThread = 4; fair-share cap = 1024 / 256 = 4 → final perThread = 4

// CHECK: #[[$BLOCKED4_ST:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: buffer_store_f32_with_mask
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_store_f32_with_mask(
      %value: tensor<1024xf32, #blocked4>,
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<1024xi32, #blocked4> {tt.divisibility = 16 : i32},
      %mask: tensor<1024xi1, #blocked4>) {
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<1024xf32, #blocked{{.*}}> -> tensor<1024xf32, #[[$BLOCKED4_ST]]>
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<1024xi32, #blocked{{.*}}> -> tensor<1024xi32, #[[$BLOCKED4_ST]]>
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<1024xi1, #blocked{{.*}}> -> tensor<1024xi1, #[[$BLOCKED4_ST]]>
    // CHECK: amdg.buffer_store %{{.*}}, %{{.*}}[%{{.*}}], %{{.*}} : tensor<1024xf32, #[[$BLOCKED4_ST]]>
    amdg.buffer_store %value, %ptr[%offsets], %mask : tensor<1024xf32, #blocked4>
    tt.return
  }
}

// -----

// Test 5: 2D buffer_load f32, inner dimension is fast-varying (order = [1, 0]).
// ptr divisibility = 16 bytes → ptrAlign = 16 / 4 = 4
// offset divisibility (innermost dim 1) = 16 bytes → offsetAlign = 16 / 4 = 4
// maxVec = 4; fair-share cap = (128 * 64) / 256 = 32 → final perThread = 4
// New: sizePerThread = [1, 4], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1]

// CHECK: #[[$BLOCKED_2D:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: buffer_load_2d_inner_contiguous
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_2d_inner_contiguous(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<128x64xi32, #blocked5> {tt.divisibility = 16 : i32}) {
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<128x64xi32, #blocked{{.*}}> -> tensor<128x64xi32, #[[$BLOCKED_2D]]>
    // CHECK: %{{.*}} = amdg.buffer_load %{{.*}}[%{{.*}}] : tensor<128x64xf32, #[[$BLOCKED_2D]]>
    %result = amdg.buffer_load %ptr[%offsets] : tensor<128x64xf32, #blocked5>
    tt.return
  }
}

// -----

// Test 6: Encoding is already optimal — no convert_layout inserted.
// ptr divisibility = 16 bytes → ptrAlign = 4; offset div = 16 → offsetAlign = 4
// perThread = 4; cap = 4 → new encoding matches existing sizePerThread = [4]; no rewrite.

// CHECK-LABEL: buffer_load_already_optimal
#blocked6 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_already_optimal(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<1024xi32, #blocked6> {tt.divisibility = 16 : i32}) {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: amdg.buffer_load %{{.*}}[%{{.*}}] : tensor<1024xf32, #blocked{{.*}}>
    %result = amdg.buffer_load %ptr[%offsets] : tensor<1024xf32, #blocked6>
    tt.return
  }
}

// -----

// Test 7: Low pointer alignment limits vectorization to 1 element — no rewrite.
// ptr divisibility = 4 bytes → ptrAlign = 4 / 4 = 1
// offset divisibility = 16 bytes → offsetAlign = 4
// perThread = min(1, 4, 4) = 1; encoding already has sizePerThread = [1] → no rewrite.

// CHECK-LABEL: buffer_load_low_ptr_align
#blocked7 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_low_ptr_align(
      %ptr: !tt.ptr<f32> {tt.divisibility = 4 : i32},
      %offsets: tensor<1024xi32, #blocked7> {tt.divisibility = 16 : i32}) {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: amdg.buffer_load %{{.*}}[%{{.*}}] : tensor<1024xf32, #blocked{{.*}}>
    %result = amdg.buffer_load %ptr[%offsets] : tensor<1024xf32, #blocked7>
    tt.return
  }
}

// -----

// Test 8: buffer_load_to_local is excluded from the pass — encoding not modified.
// The pass explicitly skips BufferLoadToLocalOp regardless of alignment information.
// ptr div = 16 bytes would allow perThread=4 for f32, but the op is not touched.

// CHECK-LABEL: buffer_load_to_local_excluded
#blocked8 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared8 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_to_local_excluded(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<1024xi32, #blocked8> {tt.divisibility = 16 : i32},
      %dst: !ttg.memdesc<1024xf32, #shared8, #smem, mutable>) {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: amdg.buffer_load_to_local %{{.*}}[%{{.*}}] into %{{.*}}
    %token = amdg.buffer_load_to_local %ptr[%offsets] into %dst : !tt.ptr<f32>[tensor<1024xi32, #blocked8>] -> <1024xf32, #shared8, #smem, mutable>
    tt.return
  }
}
