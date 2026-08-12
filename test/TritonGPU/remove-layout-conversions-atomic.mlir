// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s

// The producer layout has more total elements per thread, but its contiguous
// register dimension is strided for the atomic pointer. The atomic layout must
// win because it can issue eight contiguous operations instead of one.

// CHECK: #[[$ATOMIC:.*]] = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK-LABEL: @multidimensional_atomic_layout
// CHECK: tt.atomic_rmw fadd, relaxed, gpu, %{{.*}}, %{{.*}}, %{{.*}} : (tensor<128x32x!tt.ptr<bf16>, #[[$ATOMIC]]>, tensor<128x32xbf16, #[[$ATOMIC]]>, tensor<128x32xi1, #[[$ATOMIC]]>) -> tensor<128x32xbf16, #[[$ATOMIC]]>

#atomic = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#producer = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  tt.func @multidimensional_atomic_layout(
      %ptr: tensor<128x32x!tt.ptr<bf16>, #atomic> {tt.contiguity = dense<[128, 1]> : tensor<2xi32>, tt.divisibility = dense<[16, 2]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
      %value: tensor<128x32xbf16, #producer>,
      %mask: tensor<128x32xi1, #atomic> {tt.constancy = dense<[128, 1]> : tensor<2xi32>}) {
    %value_atomic = ttg.convert_layout %value : tensor<128x32xbf16, #producer> -> tensor<128x32xbf16, #atomic>
    %result = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %value_atomic, %mask : (tensor<128x32x!tt.ptr<bf16>, #atomic>, tensor<128x32xbf16, #atomic>, tensor<128x32xi1, #atomic>) -> tensor<128x32xbf16, #atomic>
    tt.return
  }
}

// -----

// Both layouts provide at least 128 bits of contiguous fp32 elements. Once
// their vector widths are capped at four, prefer the layout with more total
// elements per thread using the existing score as a tiebreaker.

// CHECK: #[[$BALANCED:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK-LABEL: @atomic_vector_width_tie
// CHECK: tt.atomic_rmw fadd, relaxed, gpu, %{{.*}}, %{{.*}}, %{{.*}} : (tensor<128x32x!tt.ptr<f32>, #[[$BALANCED]]>, tensor<128x32xf32, #[[$BALANCED]]>, tensor<128x32xi1, #[[$BALANCED]]>) -> tensor<128x32xf32, #[[$BALANCED]]>

#wide = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#balanced = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  tt.func @atomic_vector_width_tie(
      %ptr: tensor<128x32x!tt.ptr<f32>, #wide> {tt.contiguity = dense<[128, 1]> : tensor<2xi32>, tt.divisibility = dense<[32, 4]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
      %value: tensor<128x32xf32, #balanced>,
      %mask: tensor<128x32xi1, #wide> {tt.constancy = dense<[128, 1]> : tensor<2xi32>}) {
    %value_wide = ttg.convert_layout %value : tensor<128x32xf32, #balanced> -> tensor<128x32xf32, #wide>
    %result = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %value_wide, %mask : (tensor<128x32x!tt.ptr<f32>, #wide>, tensor<128x32xf32, #wide>, tensor<128x32xi1, #wide>) -> tensor<128x32xf32, #wide>
    tt.return
  }
}
