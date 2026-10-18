// RUN: triton-opt %s -split-input-file -tritonamdgpu-sink-layout-conversions | FileCheck %s

//   CHECK-LABEL: sink_layout_conversion
// CHECK-COUNT-2: ttg.local_dealloc %{{.+}} : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
//         CHECK: ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_layout_conversion(%arg0: tensor<32x32xf32, #blocked>, %arg1: tensor<32x32xf32, #blocked1>, %arg2: tensor<32x32x!tt.ptr<f32>, #blocked1>) {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %2 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    ttg.local_dealloc %0 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %3 = arith.addf %2, %arg1 : tensor<32x32xf32, #blocked1>
    tt.store %arg2, %3 : tensor<32x32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

//   CHECK-LABEL: sink_past_cond_barrier
//         CHECK: amdg.cond_barrier %arg3
//         CHECK: scf.for
//         CHECK: amdg.cond_barrier %arg4
//    CHECK-NEXT: ttg.convert_layout
//         CHECK: amdg.cond_barrier %arg3
//    CHECK-NEXT: ttg.convert_layout %arg0
//         CHECK: arith.addf
//         CHECK: amdg.cond_barrier %arg4
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_past_cond_barrier(%arg0: tensor<32x32xf32, #blocked>, %arg1: tensor<32x32xf32, #blocked1>, %arg2: tensor<32x32x!tt.ptr<f32>, #blocked1>, %arg3: i1, %arg4: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    amdg.cond_barrier %arg3
    %0 = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> (tensor<32x32xf32, #blocked>) : i32 {
      %s = arith.addf %acc, %acc : tensor<32x32xf32, #blocked>
      scf.yield %s : tensor<32x32xf32, #blocked>
    }
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    amdg.cond_barrier %arg4
    tt.store %arg2, %1 : tensor<32x32x!tt.ptr<f32>, #blocked1>
    // A conversion whose use lies inside the window stays in place.
    amdg.cond_barrier %arg3
    %2 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    %3 = arith.addf %2, %arg1 : tensor<32x32xf32, #blocked1>
    amdg.cond_barrier %arg4
    tt.store %arg2, %3 : tensor<32x32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
