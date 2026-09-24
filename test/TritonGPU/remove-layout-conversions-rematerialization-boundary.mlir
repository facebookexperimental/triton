// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -tritongpu-remove-layout-conversions -tritongpu-remove-layout-conversions | FileCheck %s

// An identity conversion carrying coordinate-rematerialization metadata is a
// semantic backend boundary. Its block-argument source has no producer to
// clone, so backward rematerialization must leave the conversion intact.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @identity_rematerialization_boundary
  tt.func @identity_rematerialization_boundary(
      %src: tensor<64x2xf32, #blocked>) -> tensor<64x2xf32, #blocked> {
    // CHECK: %[[BOUNDARY:.*]] = ttg.convert_layout %arg0 {tlx.rematerialize_coordinates_group = 21 : i32} : tensor<64x2xf32, #{{.*}}> -> tensor<64x2xf32, #{{.*}}>
    %boundary = ttg.convert_layout %src {tlx.rematerialize_coordinates_group = 21 : i32} : tensor<64x2xf32, #blocked> -> tensor<64x2xf32, #blocked>
    // CHECK: %[[RESULT:.*]] = arith.addf %[[BOUNDARY]], %[[BOUNDARY]]
    %result = arith.addf %boundary, %boundary : tensor<64x2xf32, #blocked>
    // CHECK: tt.return %[[RESULT]]
    tt.return %result : tensor<64x2xf32, #blocked>
  }
}
