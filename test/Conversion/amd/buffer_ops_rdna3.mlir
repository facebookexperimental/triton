// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1170 | FileCheck %s

// RDNA3/RDNA4m cache policy bits: GLC(1) | SLC(2) | DLC(4). An eviction
// policy steers SLC/DLC but never weakens a .cv/.wt cache bypass.

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: buffer_load_eviction
  tt.func @buffer_load_eviction(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}) {
    // evict_last with .cs clears SLC/DLC, keeps the GLC L1 bypass: 1
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 1
    %ret0 = amdg.buffer_load %arg0[%offset] cacheModifier = cs evictionPolicy = evict_last : !tt.ptr<f32> -> tensor<128xf32, #blocked0>
    // evict_last never weakens the .cv bypass: GLC|SLC|DLC stays 7.
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 7
    %ret1 = amdg.buffer_load %arg0[%offset] cacheModifier = cv evictionPolicy = evict_last : !tt.ptr<f32> -> tensor<128xf32, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: buffer_store_eviction
  tt.func @buffer_store_eviction(%value : tensor<128xf32, #blocked0>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}) {
    // evict_first with no modifier sets SLC/DLC streaming bits: 6
    // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, 6
    amdg.buffer_store %value, %arg0[%offset] evictionPolicy = evict_first : !tt.ptr<f32> -> tensor<128xf32, #blocked0>
    // evict_last never weakens the .wt bypass: GLC|SLC|DLC stays 7.
    // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, 7
    amdg.buffer_store %value, %arg0[%offset] cacheModifier = wt evictionPolicy = evict_last : !tt.ptr<f32> -> tensor<128xf32, #blocked0>
    tt.return
  }
}
