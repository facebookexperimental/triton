// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu="target=hip:gfx1250 num-warps=4 threads-per-warp=32" | FileCheck %s

// Unencoded scaled fp4 casts get a layout that keeps each v_cvt_scale*_pk8
// group of 8 fp4 values in consecutive registers along the scaled axis, and a
// scale layout derived from the output layout.

// CHECK-DAG: #[[$UP_PACKED:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$UP_UNPACKED:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$UP_SCALE:.+]] = #ttg.linear<{register = {{\[}}[8, 0], [16, 0]], lane = {{\[}}[0, 0], [0, 0], [0, 1], [0, 2], [1, 0]], warp = {{\[}}[2, 0], [4, 0]], block = []}>
// CHECK-LABEL: tt.func @upcast
// CHECK: %[[X:.+]] = ttg.convert_layout %arg0 : {{.*}} -> tensor<32x64xi8, #[[$UP_PACKED]]>
// CHECK: %[[S:.+]] = ttg.convert_layout %arg1 : {{.*}} -> tensor<32x4xi8, #[[$UP_SCALE]]>
// CHECK: amdg.scaled_upcast_fp4 %[[X]] scale %[[S]] {axis = 1 : i32} : {{.*}} -> tensor<32x128xbf16, #[[$UP_UNPACKED]]>
module {
  tt.func @upcast(%x: tensor<32x64xi8>, %s: tensor<32x4xi8>) -> tensor<32x128xbf16> {
    %y = amdg.scaled_upcast_fp4 %x scale %s {axis = 1 : i32} : tensor<32x64xi8>, tensor<32x4xi8> -> tensor<32x128xbf16>
    tt.return %y : tensor<32x128xbf16>
  }
}

// -----

// CHECK-DAG: #[[$DN_UNPACKED:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$DN_PACKED:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$DN_SCALE:.+]] = #ttg.linear<{register = {{\[}}[8, 0], [16, 0]], lane = {{\[}}[0, 0], [0, 0], [0, 1], [0, 2], [1, 0]], warp = {{\[}}[2, 0], [4, 0]], block = []}>
// CHECK-LABEL: tt.func @downcast
// CHECK: %[[X:.+]] = ttg.convert_layout %arg0 : {{.*}} -> tensor<32x128xf32, #[[$DN_UNPACKED]]>
// CHECK: %[[S:.+]] = ttg.convert_layout %arg1 : {{.*}} -> tensor<32x4xi8, #[[$DN_SCALE]]>
// CHECK: amdg.scaled_downcast_fp4 %[[X]] scale %[[S]] {axis = 1 : i32} : {{.*}} -> tensor<32x64xi8, #[[$DN_PACKED]]>
module {
  tt.func @downcast(%x: tensor<32x128xf32>, %s: tensor<32x4xi8>) -> tensor<32x64xi8> {
    %y = amdg.scaled_downcast_fp4 %x scale %s {axis = 1 : i32} : tensor<32x128xf32>, tensor<32x4xi8> -> tensor<32x64xi8>
    tt.return %y : tensor<32x64xi8>
  }
}

// -----

// A scaled axis that holds a single pk8 group spreads lanes and warps over the
// other dimensions instead of replicating the group.

// CHECK-DAG: #[[$NA_UNPACKED:.+]] = #ttg.blocked<{sizePerThread = [1, 1, 1, 8], threadsPerWarp = [4, 4, 2, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 2, 1, 0]}>
// CHECK-LABEL: tt.func @downcast_narrow_axis
// CHECK: %[[X:.+]] = ttg.convert_layout %arg0 : {{.*}} -> tensor<64x4x2x8xbf16, #[[$NA_UNPACKED]]>
// CHECK: amdg.scaled_downcast_fp4 %[[X]] scale {{.*}} {axis = 3 : i32} : tensor<64x4x2x8xbf16, #[[$NA_UNPACKED]]>
module {
  tt.func @downcast_narrow_axis(%x: tensor<64x4x2x8xbf16>, %s: tensor<64x4x2x1xi8>) -> tensor<64x4x2x4xi8> {
    %y = amdg.scaled_downcast_fp4 %x scale %s {axis = 3 : i32} : tensor<64x4x2x8xbf16>, tensor<64x4x2x1xi8> -> tensor<64x4x2x4xi8>
    tt.return %y : tensor<64x4x2x4xi8>
  }
}
