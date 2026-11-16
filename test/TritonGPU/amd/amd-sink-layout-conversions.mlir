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

// Windows are cond_barrier(p) ... cond_barrier(~p). Both the canonical
// inverted-cmpi form and an explicit xori with true close a window.
//   CHECK-LABEL: sink_past_cond_barrier
//         CHECK: amdg.cond_barrier %[[GE:.+]]
//         CHECK: scf.for
//         CHECK: amdg.cond_barrier %[[LT:.+]]
//    CHECK-NEXT: ttg.convert_layout
//         CHECK: amdg.cond_barrier %[[GE]]
//    CHECK-NEXT: ttg.convert_layout %arg0
//         CHECK: arith.addf
//         CHECK: amdg.cond_barrier %[[LT]]
//         CHECK: amdg.cond_barrier %arg3
//         CHECK: amdg.cond_barrier %[[NOT:.+]]
//    CHECK-NEXT: ttg.convert_layout %arg0
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_past_cond_barrier(%arg0: tensor<32x32xf32, #blocked>, %arg1: tensor<32x32xf32, #blocked1>, %arg2: tensor<32x32x!tt.ptr<f32>, #blocked1>, %arg3: i1, %tid: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c64 = arith.constant 64 : i32
    %true = arith.constant true
    %ge = arith.cmpi sge, %tid, %c64 : i32
    %lt = arith.cmpi slt, %tid, %c64 : i32
    %not = arith.xori %arg3, %true : i1
    amdg.cond_barrier %ge
    %0 = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> (tensor<32x32xf32, #blocked>) : i32 {
      %s = arith.addf %acc, %acc : tensor<32x32xf32, #blocked>
      scf.yield %s : tensor<32x32xf32, #blocked>
    }
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    amdg.cond_barrier %lt
    tt.store %arg2, %1 : tensor<32x32x!tt.ptr<f32>, #blocked1>
    // A conversion whose use lies inside the window stays in place.
    amdg.cond_barrier %ge
    %2 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    %3 = arith.addf %2, %arg1 : tensor<32x32xf32, #blocked1>
    amdg.cond_barrier %lt
    tt.store %arg2, %3 : tensor<32x32x!tt.ptr<f32>, #blocked1>
    amdg.cond_barrier %arg3
    %4 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    amdg.cond_barrier %not
    tt.store %arg2, %4 : tensor<32x32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// Barriers are only paired when the predicates are provably complementary.
// An unpaired barrier before a window does not shift the pairing (pairing by
// position would take %arg2 ... %ge as the window and leave %1 inside the real
// one), and barriers on unrelated predicates form no window.
//   CHECK-LABEL: cond_barrier_pairing
//         CHECK: amdg.cond_barrier %arg2
//    CHECK-NEXT: ttg.convert_layout %arg0
//    CHECK-NEXT: amdg.cond_barrier %[[GE:.+]]
//    CHECK-NEXT: amdg.cond_barrier %[[LT:.+]]
//    CHECK-NEXT: ttg.convert_layout %arg0
//    CHECK-NEXT: amdg.cond_barrier %arg2
//    CHECK-NEXT: ttg.convert_layout %arg0
//    CHECK-NEXT: amdg.cond_barrier %arg3
//    CHECK-NEXT: tt.store
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cond_barrier_pairing(%arg0: tensor<32x32xf32, #blocked>, %arg1: tensor<32x32x!tt.ptr<f32>, #blocked1>, %arg2: i1, %arg3: i1, %arg4: i32) {
    %c64 = arith.constant 64 : i32
    %ge = arith.cmpi sge, %arg4, %c64 : i32
    %lt = arith.cmpi slt, %arg4, %c64 : i32
    amdg.cond_barrier %arg2
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    amdg.cond_barrier %ge
    %1 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    amdg.cond_barrier %lt
    amdg.cond_barrier %arg2
    %2 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    amdg.cond_barrier %arg3
    tt.store %arg1, %0 : tensor<32x32x!tt.ptr<f32>, #blocked1>
    tt.store %arg1, %1 : tensor<32x32x!tt.ptr<f32>, #blocked1>
    tt.store %arg1, %2 : tensor<32x32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
