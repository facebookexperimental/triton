// With the flag on, verifyMemDescLayout rejects a CTA split of an NPOT shared
// extent: the layout is built on the pow2 alloc shape. With the flag off the
// allocations verify, as before this diff.
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt --split-input-file %s --verify-diagnostics
// RUN: env -u TRITON_ALLOW_NPOT triton-opt --split-input-file %s | FileCheck %s --check-prefix=OFF

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // OFF-LABEL: tt.func @swizzled_alloc_extent6_split4
  // OFF: ttg.local_alloc
  tt.func @swizzled_alloc_extent6_split4() {
    // expected-error @+1 {{NPOT memdesc extent split across CTAs is not yet supported}}
    %0 = ttg.local_alloc : () -> !ttg.memdesc<6x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A partitioned layout builds its inner layout over one 6x32 piece.
#inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#shared = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 1, partitionDim = 1, partitionLayout = #inner}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // OFF-LABEL: tt.func @partitioned_piece_extent6_split4
  // OFF: ttg.local_alloc
  tt.func @partitioned_piece_extent6_split4() {
    // expected-error @+1 {{NPOT memdesc extent split across CTAs is not yet supported}}
    %0 = ttg.local_alloc : () -> !ttg.memdesc<6x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// 4 CTAs split the 24 rows evenly, but the 32-row pow2 alloc shape gives each
// CTA 8 rows, not 6.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // OFF-LABEL: tt.func @swizzled_alloc_extent24_split4
  // OFF: ttg.local_alloc
  tt.func @swizzled_alloc_extent24_split4() {
    // expected-error @+1 {{NPOT memdesc extent split across CTAs is not yet supported}}
    %0 = ttg.local_alloc : () -> !ttg.memdesc<24x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// The 4-row pow2 alloc shape of extent 3 is split 4 ways, so nothing
// broadcasts.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // OFF-LABEL: tt.func @swizzled_alloc_extent3_split4
  // OFF: ttg.local_alloc
  tt.func @swizzled_alloc_extent3_split4() {
    // expected-error @+1 {{NPOT memdesc extent split across CTAs is not yet supported}}
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// The NPOT pipelining dimension is dropped before the check.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // OFF-LABEL: tt.func @swizzled_alloc_three_buffers_split4
  // OFF: ttg.local_alloc
  tt.func @swizzled_alloc_three_buffers_split4() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
