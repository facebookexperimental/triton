// RUN: triton-opt --split-input-file %s --tlx-rewrite-local-alias --verify-diagnostics

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @non_divisor_alias() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x32x32xbf16, #shared, #smem, mutable>
    // expected-error @+1 {{TLXRewriteLocalAlias cannot view a 49152-bit allocation as a 32768-bit alias}}
    %1 = tlx.local_alias %0 : !ttg.memdesc<3x32x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<1x32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @multi_slot_size_mismatch_alias() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<8x32x32xf32, #shared, #smem, mutable>
    // expected-error @+1 {{TLXRewriteLocalAlias cannot shrink a size-mismatched alias with leading batch dim 4 (only unit batch dim is supported)}}
    %1 = tlx.local_alias %0 : !ttg.memdesc<8x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<4x32x32xbf16, #shared, #smem, mutable>
    tt.return
  }
}
