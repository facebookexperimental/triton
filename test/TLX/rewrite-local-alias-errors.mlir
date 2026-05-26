// RUN: triton-opt --split-input-file %s --tlx-rewrite-local-alias --verify-diagnostics

// Non-divisor sizes: the bf16 backing has 49152 bits (3 buffers), but the
// f32 view asks for 32768 bits and 49152 is not a whole multiple of 32768,
// so the size-mismatched alias recipe does not apply.
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

// Tensor memory size-mismatched alias: the pass only knows how to expand
// shared memory views. The alias is larger than the base, so the pass first
// creates a new tmem_alloc with the alias's f32 type and then needs to view
// it back as the original f16 base type, which is where the size mismatch
// surfaces.
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @tmem_alias_size_mismatch() {
    // expected-error @+1 {{TLXRewriteLocalAlias only supports size-mismatched aliases for shared memory}}
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x32xf16, #tmem1, #ttng.tensor_memory, mutable>
    %1 = tlx.local_alias %0 : !ttg.memdesc<1x64x32xf16, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Non-unit leading batch dim on a size-mismatched alias: the size-aware
// expansion produces a fresh single-slot descriptor via memdesc_index[0]
// and cannot reproduce a multi-slot view onto a partial backing buffer.
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
