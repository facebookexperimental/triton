// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tlx-propagate-layout -verify-diagnostics | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tlx-propagate-layout -tlx-resolve-placeholder-layouts -tlx-finalize-user-layouts -verify-diagnostics | FileCheck %s --check-prefix=CHAIN

#b = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [32, 96]}>

#c = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32]], lane = [[0, 8], [0, 16], [2, 0], [1, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [32, 96]}>

// Pinned require (lowered to a conversion) and strict release anchors on
// an NPOT tile propagate with the exact shape intact.
// CHECK-LABEL: @anchors_96
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  tt.func @anchors_96(%pa: !tt.ptr<f16>, %po: !tt.ptr<f16>) {
    // CHECK: ttg.convert_layout %{{.*}} : tensor<32x96xf16, #{{.*}}> -> tensor<32x96xf16, #{{.*}}>
    // CHECK: ttg.release_layout %{{.*}} : tensor<32x96xf16, #{{.*}}> -> tensor<32x96xf16, #{{.*}}>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x96xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
    %ac = ttg.local_load %alloc : !ttg.memdesc<32x96xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> tensor<32x96xf16, #b>
    %r = tlx.require_layout %ac : tensor<32x96xf16, #b> -> tensor<32x96xf16, #c>
    %s = tlx.release_layout %r : tensor<32x96xf16, #c> -> tensor<32x96xf16, #b>
    %so = tt.splat %po : !tt.ptr<f16> -> tensor<32x96x!tt.ptr<f16>, #b>
    tt.store %so, %s : tensor<32x96x!tt.ptr<f16>, #b>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

// Control: a power-of-two tile propagates the same way.
// CHECK-LABEL: @anchors_128
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  tt.func @anchors_128(%pa: !tt.ptr<f16>, %po: !tt.ptr<f16>) {
    // CHECK: ttg.release_layout %{{.*}} : tensor<32x128xf16, #{{.*}}> -> tensor<32x128xf16, #{{.*}}>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
    %ac = ttg.local_load %alloc : !ttg.memdesc<32x128xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> tensor<32x128xf16, #blocked>
    %r = tlx.require_layout %ac : tensor<32x128xf16, #blocked> -> tensor<32x128xf16, #blocked>
    %s = tlx.release_layout %r : tensor<32x128xf16, #blocked> -> tensor<32x128xf16, #blocked>
    %so = tt.splat %po : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.store %so, %s : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// The full anchor chain (propagate, resolve, finalize) is accepted:
// both anchors become converts carrying the anchor's pair.
#b = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [32, 96]}>
#c = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32]], lane = [[0, 8], [0, 16], [2, 0], [1, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [32, 96]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHAIN-LABEL: @anchors_legal_pair
  tt.func @anchors_legal_pair(%po: !tt.ptr<f16>) {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x96xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
    %ac = ttg.local_load %alloc : !ttg.memdesc<32x96xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> tensor<32x96xf16, #b>
    // CHAIN: ttg.convert_layout %{{.*}} : tensor<32x96xf16, #{{.*}}> -> tensor<32x96xf16, #{{.*}}>
    %r = tlx.require_layout %ac : tensor<32x96xf16, #b> -> tensor<32x96xf16, #c>
    // CHAIN: ttg.convert_layout %{{.*}} : tensor<32x96xf16, #{{.*}}> -> tensor<32x96xf16, #{{.*}}>
    %s = tlx.release_layout %r : tensor<32x96xf16, #c> -> tensor<32x96xf16, #b>
    %so = tt.splat %po : !tt.ptr<f16> -> tensor<32x96x!tt.ptr<f16>, #b>
    tt.store %so, %s : tensor<32x96x!tt.ptr<f16>, #b>
    tt.return
  }
}
