// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s

// The transposed LDS read composes layouts, which needs radix 2, so a
// mixed-radix local_load takes the generic run lowering: 48 f16 per thread, 8
// per run.
#p = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [64, 0], [0, 32]], lane = [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[16, 0], [32, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#s8 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @mixed_radix_local_load
  tt.func @mixed_radix_local_load(%p: !tt.ptr<f16>, %i: i32) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x128x96xf16, #s8, #smem, mutable>
    %s = ttg.memdesc_index %a[%i] : !ttg.memdesc<2x128x96xf16, #s8, #smem, mutable> -> !ttg.memdesc<128x96xf16, #s8, #smem, mutable>
    // CHECK-COUNT-6: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    // CHECK-NOT: llvm.load {{.*}}ptr<3>
    %v = ttg.local_load %s : !ttg.memdesc<128x96xf16, #s8, #smem, mutable> -> tensor<128x96xf16, #p>
    %q = tt.splat %p : !tt.ptr<f16> -> tensor<128x96x!tt.ptr<f16>, #p>
    tt.store %q, %v : tensor<128x96x!tt.ptr<f16>, #p>
    tt.return
  }
}
