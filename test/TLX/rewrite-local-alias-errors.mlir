// RUN: triton-opt --split-input-file %s --tlx-rewrite-local-alias --verify-diagnostics

#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
#tmem_space = #ttng.tensor_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @non_divisor_alias() {
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<3x64x32xbf16, #tmem, #tmem_space, mutable>
    // expected-error @+1 {{TLXRewriteLocalAlias cannot view a}}
    %1 = tlx.local_alias %0 : !ttg.memdesc<3x64x32xbf16, #tmem, #tmem_space, mutable> -> !ttg.memdesc<1x64x32xf32, #tmem, #tmem_space, mutable>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
#tmem_space = #ttng.tensor_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @multi_slot_size_mismatch_alias() {
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<8x64x32xf32, #tmem, #tmem_space, mutable>
    // expected-error @+1 {{TLXRewriteLocalAlias cannot shrink a size-mismatched alias with leading batch dim 4 (only unit batch dim is supported)}}
    %1 = tlx.local_alias %0 : !ttg.memdesc<8x64x32xf32, #tmem, #tmem_space, mutable> -> !ttg.memdesc<4x64x32xbf16, #tmem, #tmem_space, mutable>
    tt.return
  }
}
