// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-amdgpu-shared-memory=arch=gfx950 | FileCheck %s

// A K96 bf16 tile is three 32-column panels, each 128 * 32 elements apart.

#sw = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: module @k96_swizzled
// CHECK-SAME: ttg.shared = 24576 : i32
module @k96_swizzled attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @k96_swizzled() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x96xbf16, #sw, #smem, mutable>
    tt.return
  }
}

// -----

#rot = #ttg.amd_rotating_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: module @k96_rotating
// CHECK-SAME: ttg.shared = 24576 : i32
module @k96_rotating attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @k96_rotating() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x96xbf16, #rot, #smem, mutable>
    tt.return
  }
}

// -----

// A 64-column swizzle does not fit a 32-column panel, so the tile keeps its
// pow2 size.
#sw = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: module @k96_wide_swizzle
// CHECK-SAME: ttg.shared = 32768 : i32
module @k96_wide_swizzle attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @k96_wide_swizzle() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x96xbf16, #sw, #smem, mutable>
    tt.return
  }
}
