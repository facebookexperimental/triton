// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -tritongpu-remove-layout-conversions | FileCheck %s

// A release is a durable boundary between an upstream layout and the
// independently selected layout of its result. Running layout removal again
// must not make the source layout propagate through the boundary.

#src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#pinned = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#released = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-DAG: #[[$PINNED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-DAG: #[[$RELEASED_LAYOUT:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-LABEL: tt.func @release_starts_independent_layout_region
  tt.func @release_starts_independent_layout_region(
      %src: tensor<64x32xf32, #src>,
      %other: tensor<64x32xf32, #released>)
      -> tensor<64x32xf32, #released> {
    // CHECK: %[[REQUIRED:.*]] = ttg.require_layout %{{.*}} : tensor<64x32xf32, #{{.*}}> -> tensor<64x32xf32, #[[$PINNED]]>
    %required = ttg.require_layout %src : tensor<64x32xf32, #src> -> tensor<64x32xf32, #pinned>
    // CHECK: %[[RELEASED:.*]] = ttg.release_layout %[[REQUIRED]] : tensor<64x32xf32, #[[$PINNED]]> -> tensor<64x32xf32, #[[$RELEASED_LAYOUT]]>
    %released = ttg.release_layout %required : tensor<64x32xf32, #pinned> -> tensor<64x32xf32, #released>
    // CHECK: arith.addf %[[RELEASED]], %{{.*}} : tensor<64x32xf32, #[[$RELEASED_LAYOUT]]>
    %sum = arith.addf %released, %other : tensor<64x32xf32, #released>
    tt.return %sum : tensor<64x32xf32, #released>
  }
}

// -----

// An identity release still carries semantic information and must survive RLC.

#layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @identity_release_survives
  tt.func @identity_release_survives(
      %src: tensor<64x32xf32, #layout>) -> tensor<64x32xf32, #layout> {
    // CHECK: %[[RELEASED:.*]] = ttg.release_layout %{{.*}} : tensor<64x32xf32, #{{.*}}> -> tensor<64x32xf32, #{{.*}}>
    %released = ttg.release_layout %src : tensor<64x32xf32, #layout> -> tensor<64x32xf32, #layout>
    // CHECK: tt.return %[[RELEASED]]
    tt.return %released : tensor<64x32xf32, #layout>
  }
}

// -----

// A conversion after release may move through downstream elementwise work, but
// neither rematerialization nor hoisting may cross the release itself.

#before = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#after = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#consumer = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @rematerialization_stops_at_release
  tt.func @rematerialization_stops_at_release(
      %src: tensor<64x32xf16, #before>) -> tensor<64x32xf32, #consumer> {
    // CHECK: %[[RELEASED:.*]] = ttg.release_layout %{{.*}}
    %released = ttg.release_layout %src : tensor<64x32xf16, #before> -> tensor<64x32xf16, #after>
    // CHECK: %[[AFTER_RELEASE:.*]] = ttg.convert_layout %[[RELEASED]]
    // CHECK: %[[EXTENDED:.*]] = arith.extf %[[AFTER_RELEASE]]
    %extended = arith.extf %released : tensor<64x32xf16, #after> to tensor<64x32xf32, #after>
    %converted = ttg.convert_layout %extended : tensor<64x32xf32, #after> -> tensor<64x32xf32, #consumer>
    // CHECK-NOT: arith.extf %src
    // CHECK: tt.return %[[EXTENDED]]
    tt.return %converted : tensor<64x32xf32, #consumer>
  }
}
