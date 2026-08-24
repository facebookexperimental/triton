// RUN: triton-opt -split-input-file -tritongpu-coalesce='max-vec-bits=256' --tlx-propagate-layout %s | FileCheck %s

// A user-pinned *register* layout on a local_load (#tlx.user_layout carrying
// PinnedEncodingTrait) must not be coalesced to a "full contiguity" blocked
// layout: the user chose this register layout on purpose. It stays wrapped
// through the layout-optimization passes (anchored via the trait) and is
// unwrapped later by tlx-finalize-user-layouts.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 32]], block = []}>
#user = #tlx.user_layout<#linear>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-DAG: #[[$USER:.*]] = #tlx.user_layout<
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @reg_pin
  tt.func @reg_pin(%buf: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>) -> tensor<128x128xf16, #user> {
    // Coalesce leaves the pinned result untouched (would otherwise force #blocked).
    // CHECK: ttg.local_load {{.*}} -> tensor<128x128xf16, #[[$USER]]>
    %y = ttg.local_load %buf : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #user>
    tt.return %y : tensor<128x128xf16, #user>
  }
}

// -----

// Blackwell's 256-bit vector width applies to global memory coalescing, not
// shared-memory local loads. TLX layout propagation runs after coalescing and
// must preserve both choices.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// CHECK-DAG: #[[$GLOBAL:.*]] = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-DAG: #[[$LOCAL:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @blackwell_global_and_local_vector_widths
  tt.func @blackwell_global_and_local_vector_widths(
      %ptr: !tt.ptr<f32> {tt.divisibility = 32 : i32},
      %buf: !ttg.memdesc<1024xf32, #shared, #smem, mutable>)
      -> (tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>) {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %offsets = tt.addptr %ptrs, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, #[[$GLOBAL]]>
    %global = tt.load %offsets : tensor<1024x!tt.ptr<f32>, #blocked>
    // Force tlx-propagate-layout to run; the identity constraint is removed.
    // CHECK-NOT: tlx.require_layout
    %required = tlx.require_layout %global : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked>
    // CHECK: ttg.local_load {{.*}} -> tensor<1024xf32, #[[$LOCAL]]>
    %local = ttg.local_load %buf : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
    tt.return %required, %local : tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>
  }
}
