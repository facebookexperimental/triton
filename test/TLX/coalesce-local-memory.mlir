
// RUN: triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s

// Test that local_load gets coalesced encoding for vectorized access

// CHECK-DAG: #[[$UNCOALESCED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK-DAG: #[[$COALESCED:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: @local_load_coalesce
// CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<128x64xf16, {{.*}}> -> tensor<128x64xf16, #[[$COALESCED]]>
// CHECK: ttg.convert_layout %{{.*}} : tensor<128x64xf16, #[[$COALESCED]]> -> tensor<128x64xf16, #[[$UNCOALESCED]]>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func @local_load_coalesce(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem>) -> tensor<128x64xf16, #blocked> {
  %0 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked>
  tt.return %0 : tensor<128x64xf16, #blocked>
}

}

// -----

// A tt.store whose value is pinned via tlx.require_layout (wrapped
// #tlx.no_verify_layout<#tlx.user_layout<...>>) keeps the user's layout through
// coalesce: coalesce commits the store to the peeled concrete layout and converts
// ptr/mask to match, instead of choosing its own (contiguity-coalesced) layout.
#pinned = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#user = #tlx.user_layout<#pinned>
#nv = #tlx.no_verify_layout<#user>
// CHECK-DAG: #[[$PIN:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pinned_store
  // CHECK: tt.store {{.*}} : tensor<128x64x!tt.ptr<f16>, #[[$PIN]]>
  tt.func @pinned_store(%ptr: tensor<128x64x!tt.ptr<f16>, #nv>, %val: tensor<128x64xf16, #nv>) {
    tt.store %ptr, %val : tensor<128x64x!tt.ptr<f16>, #nv>
    tt.return
  }
}

// -----

// A pinned local_load result is likewise not re-coalesced: coalesce leaves the
// pinned layout in place (peeled later by resolve-placeholder-layouts).
#pinned = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#user = #tlx.user_layout<#pinned>
#nv = #tlx.no_verify_layout<#user>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pinned_load
  // CHECK: ttg.local_load {{.*}} -> tensor<128x64xf16, #tlx.no_verify_layout<{{.+}}>>
  tt.func @pinned_load(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem>) -> tensor<128x64xf16, #nv> {
    %0 = ttg.local_load %arg0 : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #nv>
    tt.return %0 : tensor<128x64xf16, #nv>
  }
}
