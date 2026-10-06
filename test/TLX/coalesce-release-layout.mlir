// RUN: triton-opt %s -tritongpu-coalesce='max-vec-bits=128' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// CHECK: #[[$WIDE:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @tlx_release
  tt.func @tlx_release(
      %ptrs: tensor<512x!tt.ptr<i8>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 4 : i32, tt.constancy = 2 : i32})
      -> tensor<512xi8, #blocked> {
    // CHECK: %[[TLX_RELEASED:.*]] = tlx.release_layout
    %released = tlx.release_layout %ptrs : tensor<512x!tt.ptr<i8>, #blocked> -> tensor<512x!tt.ptr<i8>, #blocked>
    // CHECK: %[[TLX_WIDE_PTRS:.*]] = ttg.convert_layout %[[TLX_RELEASED]] {{.*}} -> tensor<512x!tt.ptr<i8>, #[[$WIDE]]>
    // CHECK: %[[TLX_VALUE:.*]] = tt.load %[[TLX_WIDE_PTRS]] : tensor<512x!tt.ptr<i8>, #[[$WIDE]]>
    %value = tt.load %released : tensor<512x!tt.ptr<i8>, #blocked>
    tt.return %value : tensor<512xi8, #blocked>
  }

  // CHECK-LABEL: tt.func @ttg_release
  tt.func @ttg_release(
      %ptrs: tensor<512x!tt.ptr<i8>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 4 : i32, tt.constancy = 2 : i32})
      -> tensor<512xi8, #blocked> {
    // CHECK: %[[TTG_RELEASED:.*]] = ttg.release_layout
    %released = ttg.release_layout %ptrs : tensor<512x!tt.ptr<i8>, #blocked> -> tensor<512x!tt.ptr<i8>, #blocked>
    // CHECK: %[[TTG_WIDE_PTRS:.*]] = ttg.convert_layout %[[TTG_RELEASED]] {{.*}} -> tensor<512x!tt.ptr<i8>, #[[$WIDE]]>
    // CHECK: %[[TTG_VALUE:.*]] = tt.load %[[TTG_WIDE_PTRS]] : tensor<512x!tt.ptr<i8>, #[[$WIDE]]>
    %value = tt.load %released : tensor<512x!tt.ptr<i8>, #blocked>
    tt.return %value : tensor<512xi8, #blocked>
  }
}
