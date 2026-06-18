// RUN: triton-opt %s -split-input-file --tritonamdgpu-coalesce-async-copy=gfx-arch=gfx950 | FileCheck %s

// Test 1: 1D buffer_load_to_local f32 with sizePerThread=[1].
// ptr divisibility = 16 bytes -> ptrAlign = 16 / 4 = 4
// offset divisibility = 16 bytes -> offsetAlign = 16 / 4 = 4
// maxVec (128-bit / 32-bit) = 4; loadContig = min(4,4,4) = 4
// Expected: offsets converted to sizePerThread=[4] blocked encoding.
// CHECK: #[[$BLOCKED4:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: buffer_load_to_local_f32_vectorize
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_to_local_f32_vectorize(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<1024xi32, #blocked> {tt.divisibility = 16 : i32},
      %dst: !ttg.memdesc<1024xf32, #shared, #smem, mutable>) {
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<1024xi32, #blocked{{.*}}> -> tensor<1024xi32, #[[$BLOCKED4]]>
    // CHECK: %{{.*}} = amdg.buffer_load_to_local %{{.*}}[%{{.*}}] into %{{.*}} : <f32>[tensor<1024xi32, #[[$BLOCKED4]]>]
    %token = amdg.buffer_load_to_local %ptr[%offsets] into %dst : !tt.ptr<f32>[tensor<1024xi32, #blocked>] -> <1024xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test 2: 1D buffer_load_to_local f16, reaches the 128-bit max vector width (8 elements).
// ptr divisibility = 16 bytes -> ptrAlign = 16 / 2 = 8
// offset divisibility = 16 bytes -> offsetAlign = 16 / 2 = 8
// maxVec (128-bit / 16-bit) = 8; loadContig = 8 (large tensor, no fair-share cap here)
// CHECK: #[[$BLOCKED8:.*]] = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: buffer_load_to_local_f16_max_vec
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_to_local_f16_max_vec(
      %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %offsets: tensor<16384xi32, #blocked> {tt.divisibility = 16 : i32},
      %dst: !ttg.memdesc<16384xf16, #shared, #smem, mutable>) {
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<16384xi32, #blocked{{.*}}> -> tensor<16384xi32, #[[$BLOCKED8]]>
    // CHECK: %{{.*}} = amdg.buffer_load_to_local %{{.*}}[%{{.*}}] into %{{.*}} : <f16>[tensor<16384xi32, #[[$BLOCKED8]]>]
    %token = amdg.buffer_load_to_local %ptr[%offsets] into %dst : !tt.ptr<f16>[tensor<16384xi32, #blocked>] -> <16384xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test 3: 2D buffer_load_to_local f32, fast-varying inner dim (order=[1,0]).
// ptr divisibility = 16 bytes -> ptrAlign = 16 / 4 = 4
// offset divisibility (innermost dim 1) = 16 bytes -> offsetAlign = 4; maxVec = 4
// Expected: sizePerThread = [1, 4], threadsPerWarp redistributed.
// CHECK: #[[$BLOCKED_2D:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: buffer_load_to_local_2d_inner_contiguous
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_to_local_2d_inner_contiguous(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<128x64xi32, #blocked> {tt.divisibility = dense<[2, 16]> : tensor<2xi32>},
      %dst: !ttg.memdesc<128x64xf32, #shared, #smem, mutable>) {
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<128x64xi32, #blocked{{.*}}> -> tensor<128x64xi32, #[[$BLOCKED_2D]]>
    // CHECK: %{{.*}} = amdg.buffer_load_to_local %{{.*}}[%{{.*}}] into %{{.*}} : <f32>[tensor<128x64xi32, #[[$BLOCKED_2D]]>]
    %token = amdg.buffer_load_to_local %ptr[%offsets] into %dst : !tt.ptr<f32>[tensor<128x64xi32, #blocked>] -> <128x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test 4: buffer_load_to_local with mask and other — all three tensors converted.
// ptr div=16 -> ptrAlign=4; offset div=16 -> offsetAlign=4; loadContig=4.
// CHECK: #[[$BLOCKED4:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: buffer_load_to_local_with_mask_and_other
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_to_local_with_mask_and_other(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<1024xi32, #blocked> {tt.divisibility = 16 : i32},
      %mask: tensor<1024xi1, #blocked>,
      %other: tensor<1024xf32, #blocked>,
      %dst: !ttg.memdesc<1024xf32, #shared, #smem, mutable>) {
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<1024xi32, #blocked{{.*}}> -> tensor<1024xi32, #[[$BLOCKED4]]>
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<1024xi1, #blocked{{.*}}> -> tensor<1024xi1, #[[$BLOCKED4]]>
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<1024xf32, #blocked{{.*}}> -> tensor<1024xf32, #[[$BLOCKED4]]>
    // CHECK: %{{.*}} = amdg.buffer_load_to_local %{{.*}}[%{{.*}}] mask = %{{.*}} other = %{{.*}} into %{{.*}}
    %token = amdg.buffer_load_to_local %ptr[%offsets] mask = %mask other = %other into %dst : !tt.ptr<f32>[tensor<1024xi32, #blocked>] tensor<1024xf32, #blocked> -> <1024xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test 5: Already-optimal encoding — no convert_layout inserted.
// sizePerThread=[4] already matches loadContig=4 -> no rewrite.
// CHECK-LABEL: buffer_load_to_local_already_optimal
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_to_local_already_optimal(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<1024xi32, #blocked> {tt.divisibility = 16 : i32},
      %dst: !ttg.memdesc<1024xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: amdg.buffer_load_to_local %{{.*}}[%{{.*}}] into %{{.*}}
    %token = amdg.buffer_load_to_local %ptr[%offsets] into %dst : !tt.ptr<f32>[tensor<1024xi32, #blocked>] -> <1024xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test 6: Low pointer alignment limits vectorization to 1 — no rewrite.
// ptr divisibility = 4 bytes -> ptrAlign = 4 / 4 = 1; loadContig = 1.
// sizePerThread=[1] already matches -> no rewrite.
// CHECK-LABEL: buffer_load_to_local_low_ptr_align
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_to_local_low_ptr_align(
      %ptr: !tt.ptr<f32> {tt.divisibility = 4 : i32},
      %offsets: tensor<1024xi32, #blocked> {tt.divisibility = 16 : i32},
      %dst: !ttg.memdesc<1024xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: amdg.buffer_load_to_local %{{.*}}[%{{.*}}] into %{{.*}}
    %token = amdg.buffer_load_to_local %ptr[%offsets] into %dst : !tt.ptr<f32>[tensor<1024xi32, #blocked>] -> <1024xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test 7: buffer_load_to_local with padded_shared encoding (linear layout rewrite).
// ptr div=16 -> ptrAlign=4; offset div=16 -> offsetAlign=4; loadContig=4 (fit to valid = 4).
// The padded layout forces a linear encoding for the src so each warp writes coalesced.
// CHECK: #[[$NEW_SRC_ENC:.*]] = #ttg.linear
// CHECK-SAME{LITERAL}: register = [], lane = [[1], [2], [4], [8], [64], [128]], warp = [[16], [32]], block = []
// CHECK-LABEL: buffer_load_to_local_padded_shared
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.padded_shared<[64:+4] {offset = [[1], [2], [4], [8], [64], [128], [16], [32]], block = []}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_to_local_padded_shared(
      %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %offsets: tensor<256xi32, #blocked> {tt.divisibility = 16 : i32},
      %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : tensor<256xi32, #blocked{{.*}}> -> tensor<256xi32, #[[$NEW_SRC_ENC]]>
    // CHECK: %{{.*}} = amdg.buffer_load_to_local %{{.*}}[%{{.*}}] into %{{.*}} : <f32>[tensor<256xi32, #[[$NEW_SRC_ENC]]>]
    %token = amdg.buffer_load_to_local %ptr[%offsets] into %dst : !tt.ptr<f32>[tensor<256xi32, #blocked>] -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}
