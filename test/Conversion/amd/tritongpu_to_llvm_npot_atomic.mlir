// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s

// #p broadcasts its two low register digits, so each thread holds 6 elements in
// 24 registers.
#p = #ttg.linear<{register = [[0, 0], [0, 0], [8, 0], [16, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], warp = [[2, 0], [4, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [48, 1]}>
#b = #ttg.linear<{register = [[0, 0], [0, 0], [8, 0], [16, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], warp = [[2, 0], [4, 0]], block = [], outDimSizes = [32, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @mixed_radix_48x1
  // CHECK-COUNT-6: llvm.atomicrmw add
  // CHECK-NOT: llvm.atomicrmw
  // CHECK-COUNT-6: llvm.cmpxchg
  // CHECK-NOT: llvm.cmpxchg
  tt.func @mixed_radix_48x1(%p: !tt.ptr<i32>) {
    %ps = tt.splat %p : !tt.ptr<i32> -> tensor<48x1x!tt.ptr<i32>, #p>
    %v = arith.constant dense<1> : tensor<48x1xi32, #p>
    %r = tt.atomic_rmw add, relaxed, gpu, %ps, %v : (tensor<48x1x!tt.ptr<i32>, #p>, tensor<48x1xi32, #p>) -> tensor<48x1xi32, #p>
    %c = tt.atomic_cas relaxed, gpu, %ps, %v, %v : (tensor<48x1x!tt.ptr<i32>, #p>, tensor<48x1xi32, #p>, tensor<48x1xi32, #p>) -> tensor<48x1xi32, #p>
    tt.return
  }

  // Control: the pow2 [32, 1] with the same broadcast issues one RMW per
  // element, 4 for its 4 elements (the flag-off behaviour).
  // CHECK-LABEL: llvm.func @pow2_32x1
  // CHECK-COUNT-4: llvm.atomicrmw add
  // CHECK-NOT: llvm.atomicrmw
  tt.func @pow2_32x1(%p: !tt.ptr<i32>) {
    %ps = tt.splat %p : !tt.ptr<i32> -> tensor<32x1x!tt.ptr<i32>, #b>
    %v = arith.constant dense<1> : tensor<32x1xi32, #b>
    %r = tt.atomic_rmw add, relaxed, gpu, %ps, %v : (tensor<32x1x!tt.ptr<i32>, #b>, tensor<32x1xi32, #b>) -> tensor<32x1xi32, #b>
    tt.return
  }
}
