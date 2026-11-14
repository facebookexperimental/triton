// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s

// Each thread holds 8 contiguous f16 on dim 1, rows +32/+64, and an odd digit
// (columns +32/+64): 96 registers.
#p = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [64, 0], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#s8 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mixed_radix_local_load
  tt.func @mixed_radix_local_load(%p: !tt.ptr<f16>, %i: i32) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x128x96xf16, #s8, #smem, mutable>
    %s = ttg.memdesc_index %a[%i] : !ttg.memdesc<2x128x96xf16, #s8, #smem, mutable> -> !ttg.memdesc<128x96xf16, #s8, #smem, mutable>
    // CHECK-COUNT-12: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    // CHECK-NOT: llvm.load {{.*}}ptr<3>
    %v = ttg.local_load %s : !ttg.memdesc<128x96xf16, #s8, #smem, mutable> -> tensor<128x96xf16, #p>
    %q = tt.splat %p : !tt.ptr<f16> -> tensor<128x96x!tt.ptr<f16>, #p>
    tt.store %q, %v : tensor<128x96x!tt.ptr<f16>, #p>
    tt.return
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#s8 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pow2_control
  tt.func @pow2_control(%p: !tt.ptr<f16>, %i: i32) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #s8, #smem, mutable>
    %s = ttg.memdesc_index %a[%i] : !ttg.memdesc<2x128x128xf16, #s8, #smem, mutable> -> !ttg.memdesc<128x128xf16, #s8, #smem, mutable>
    // CHECK-COUNT-16: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<4xi32>
    // CHECK-NOT: llvm.load {{.*}}ptr<3>
    %v = ttg.local_load %s : !ttg.memdesc<128x128xf16, #s8, #smem, mutable> -> tensor<128x128xf16, #b>
    %q = tt.splat %p : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #b>
    tt.store %q, %v : tensor<128x128x!tt.ptr<f16>, #b>
    tt.return
  }
}

// -----

// A 16x96 view at row 16 of a 64x96 mixed-radix buffer: each run XORs in the
// view's origin, which adds because the subslice extent, 16, is pow2.
#p = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[8, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [16, 96]}>
#s8 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mixed_radix_view
  tt.func @mixed_radix_view(%p: !tt.ptr<f16>) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<64x96xf16, #s8, #smem, mutable>
    %s = ttg.memdesc_subslice %a [16, 0] : !ttg.memdesc<64x96xf16, #s8, #smem, mutable> -> !ttg.memdesc<16x96xf16, #s8, #smem, mutable, 64x96>
    // CHECK: llvm.mlir.constant(16 : i32)
    // CHECK: %[[O:.*]] = llvm.or disjoint
    // CHECK-NEXT: %[[ORIGIN:.*]] = llvm.xor %{{.*}}, %[[O]] : i32
    // CHECK: llvm.xor %{{.*}}, %[[ORIGIN]] : i32
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    // CHECK: llvm.xor %{{.*}}, %[[ORIGIN]] : i32
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    // CHECK: llvm.xor %{{.*}}, %[[ORIGIN]] : i32
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    // CHECK-NOT: llvm.load {{.*}}ptr<3>
    %v = ttg.local_load %s : !ttg.memdesc<16x96xf16, #s8, #smem, mutable, 64x96> -> tensor<16x96xf16, #p>
    %q = tt.splat %p : !tt.ptr<f16> -> tensor<16x96x!tt.ptr<f16>, #p>
    tt.store %q, %v : tensor<16x96x!tt.ptr<f16>, #p>
    tt.return
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#s8 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pow2_view
  tt.func @pow2_view(%p: !tt.ptr<f16>) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #s8, #smem, mutable>
    %s = ttg.memdesc_subslice %a [16, 0] : !ttg.memdesc<64x128xf16, #s8, #smem, mutable> -> !ttg.memdesc<16x128xf16, #s8, #smem, mutable, 64x128>
    // CHECK-COUNT-4: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<4xi32>
    // CHECK-NOT: llvm.load {{.*}}ptr<3>
    %v = ttg.local_load %s : !ttg.memdesc<16x128xf16, #s8, #smem, mutable, 64x128> -> tensor<16x128xf16, #b>
    %q = tt.splat %p : !tt.ptr<f16> -> tensor<16x128x!tt.ptr<f16>, #b>
    tt.store %q, %v : tensor<16x128x!tt.ptr<f16>, #b>
    tt.return
  }
}
