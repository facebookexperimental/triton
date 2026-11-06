// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 -verify-diagnostics

// gfx950 rejects a mixed-radix async copy, so it stays illegal.
#p = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
#s1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @mixed_radix_hip(%p: !tt.ptr<f16>, %d: !ttg.memdesc<192xf16, #s1, #smem, mutable>) {
    %a = tt.splat %p : !tt.ptr<f16> -> tensor<192x!tt.ptr<f16>, #p>
    // expected-error @+2 {{NPOT async copy is not yet supported on hip:gfx950}}
    // expected-error @+1 {{failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal}}
    %t = ttg.async_copy_global_to_local %a, %d : tensor<192x!tt.ptr<f16>, #p> -> <192xf16, #s1, #smem, mutable>
    tt.return
  }
}
