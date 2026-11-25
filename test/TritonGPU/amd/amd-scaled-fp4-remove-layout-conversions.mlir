// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s

// Rematerializing the downcast in the store layout would split each
// v_cvt_scalef32_pk8 group of 8 values across threads, so the conversion after
// the downcast must stay.

#ld = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#ldscale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#packed = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#store = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#scale = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[0, 0], [0, 0], [0, 1], [0, 2], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
// CHECK-DAG: #[[$PACKED:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK-DAG: #[[$STORE:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @downcast_keeps_pk8_groups
  tt.func @downcast_keeps_pk8_groups(%px: tensor<32x128x!tt.ptr<f32>, #ld>, %ps: tensor<32x4x!tt.ptr<i8>, #ldscale>, %p: tensor<32x64x!tt.ptr<i8>, #store>) {
    %x = tt.load %px : tensor<32x128x!tt.ptr<f32>, #ld>
    %s = tt.load %ps : tensor<32x4x!tt.ptr<i8>, #ldscale>
    %xc = ttg.convert_layout %x : tensor<32x128xf32, #ld> -> tensor<32x128xf32, #unpacked>
    %sc = ttg.convert_layout %s : tensor<32x4xi8, #ldscale> -> tensor<32x4xi8, #scale>
    // CHECK: %[[Y:.+]] = amdg.scaled_downcast_fp4 {{.*}} -> tensor<32x64xi8, #[[$PACKED]]>
    // CHECK: ttg.convert_layout %[[Y]] : tensor<32x64xi8, #[[$PACKED]]> -> tensor<32x64xi8, #[[$STORE]]>
    %y = amdg.scaled_downcast_fp4 %xc scale %sc {axis = 1 : i32} : tensor<32x128xf32, #unpacked>, tensor<32x4xi8, #scale> -> tensor<32x64xi8, #packed>
    %c = ttg.convert_layout %y : tensor<32x64xi8, #packed> -> tensor<32x64xi8, #store>
    tt.store %p, %c : tensor<32x64x!tt.ptr<i8>, #store>
    tt.return
  }
}
