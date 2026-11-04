// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics

// A width some tcgen05 form can issue is admitted; its consumer checks the form.
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subslice_n96(%c: !ttg.memdesc<128x192xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>) {
    %s = ttng.tmem_subslice %c {offset = 96 : i32} : !ttg.memdesc<128x192xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subslice_n100(%c: !ttg.memdesc<128x192xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>) {
    // expected-error @+1 {{NPOT tmem_subslice width 100 is unsupported: tcgen05.mma kind f16/tf32 SS K-major B cta_group::1 per-CTA M=128 requires N to be a multiple of 8 in [8, 256], got N=100}}
    %s = ttng.tmem_subslice %c {offset = 0 : i32} : !ttg.memdesc<128x192xf32, #tmem, #ttng.tensor_memory, mutable, 128x256> -> !ttg.memdesc<128x100xf32, #tmem, #ttng.tensor_memory, mutable, 128x256>
    tt.return
  }
}

// -----

// Columns 64..127 sit 16 lanes down, so this offset-32 slice cannot be a column translation.
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subslice_m64_carry(%c: !ttg.memdesc<64x192xf32, #tmem, #ttng.tensor_memory, mutable, 64x256>) {
    // expected-error @+1 {{The split offset may not touch the tile}}
    %s = ttng.tmem_subslice %c {offset = 32 : i32} : !ttg.memdesc<64x192xf32, #tmem, #ttng.tensor_memory, mutable, 64x256> -> !ttg.memdesc<64x96xf32, #tmem, #ttng.tensor_memory, mutable, 64x256>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subslice_scales(%c: !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory, mutable>) {
    // expected-error @+1 {{NPOT tmem_subslice extent 6 in dim 1 is not yet supported}}
    %s = ttng.tmem_subslice %c {offset = 0 : i32} : !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x6xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory, mutable, 128x8>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subslice_rows96(%c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
    // expected-error @+1 {{NPOT tmem_subslice extent 96 in dim 0 is not yet supported}}
    %s = ttng.tmem_subslice %c {offset = 0 : i32, dim = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<96x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}
