// RUN: TRITON_ALLOW_NPOT=0 triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts > %t.off
// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts > %t.on
// RUN: diff %t.off %t.on

// Enabling NPOT support must not change pow2 store lowering byte-for-byte.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:80"} {
  tt.func @pow2_store_64(%ptr: tensor<64x!tt.ptr<i32>, #blocked>, %val: tensor<64xi32, #blocked>) {
    tt.store %ptr, %val : tensor<64x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

#blocked2d = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "cuda:80"} {
  tt.func @pow2_store_16x64(%ptr: tensor<16x64x!tt.ptr<i32>, #blocked2d>, %val: tensor<16x64xi32, #blocked2d>) {
    tt.store %ptr, %val : tensor<16x64x!tt.ptr<i32>, #blocked2d>
    tt.return
  }
}
