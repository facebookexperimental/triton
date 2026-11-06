// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --allocate-amdgpu-shared-memory=arch=gfx950 -verify-diagnostics
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --convert-warp-pipeline=gfx-arch=gfx950 -verify-diagnostics

// convert-warp-pipeline builds its own ModuleAllocation, so it rejects too.

#sw = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#rot = #ttg.amd_rotating_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @rotating_96x64() {
    // expected-error @+1 {{NPOT shared memory descriptors are not yet supported}}
    %a = ttg.local_alloc : () -> !ttg.memdesc<96x64xf16, #rot, #smem, mutable>
    tt.return
  }

  tt.func public @swizzled_96x64_control() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<96x64xf16, #sw, #smem, mutable>
    tt.return
  }
}
