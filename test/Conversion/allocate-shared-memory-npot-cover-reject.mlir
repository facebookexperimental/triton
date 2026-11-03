// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --allocate-shared-memory-nv -verify-diagnostics
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --allocate-shared-memory -verify-diagnostics

// Allocation rejects an NPOT shared memdesc smaller than its pow2 cover.

#sw = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#rot = #ttg.amd_rotating_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rotating_96x64() {
    // expected-error @+1 {{NPOT shared memory descriptors are not yet supported}}
    %a = ttg.local_alloc : () -> !ttg.memdesc<96x64xf16, #rot, #smem, mutable>
    tt.return
  }

  // The swizzled allocation already rounds 96 up to the cover.
  tt.func public @swizzled_96x64_control() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<96x64xf16, #sw, #smem, mutable>
    tt.return
  }
}
