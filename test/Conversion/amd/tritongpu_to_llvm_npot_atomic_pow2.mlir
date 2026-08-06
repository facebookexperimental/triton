// RUN: TRITON_ALLOW_NPOT=0 triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 > %t.off
// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 > %t.on
// RUN: diff %t.off %t.on

// Enabling NPOT support must not change pow2 atomic lowering byte-for-byte.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx950"} {
  tt.func @pow2_atomic_rmw_64(%ptr: tensor<64x!tt.ptr<i32>, #blocked>, %val: tensor<64xi32, #blocked>) {
    %0 = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (tensor<64x!tt.ptr<i32>, #blocked>, tensor<64xi32, #blocked>) -> tensor<64xi32, #blocked>
    tt.return
  }
}
