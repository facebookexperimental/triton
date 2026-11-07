// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics

// Legal NPOT N, unscaled and scaled; the pow2 N128 MMA is the control.
#a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#a8 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#b8 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @legal(%a: !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>,
                 %b: !ttg.memdesc<64x96xf16, #b, #ttg.shared_memory, 64x128>,
                 %c: !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>,
                 %b128: !ttg.memdesc<64x128xf16, #b, #ttg.shared_memory>,
                 %c128: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                 %a8: !ttg.memdesc<128x64xf8E4M3FN, #a8, #ttg.shared_memory>,
                 %b8: !ttg.memdesc<64x96xf8E4M3FN, #b8, #ttg.shared_memory, 64x128>,
                 %sa: !ttg.memdesc<128x2xi8, #scales, #ttng.tensor_memory>,
                 %sb: !ttg.memdesc<96x2xi8, #scales, #ttng.tensor_memory, 128x2>,
                 %p: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>, !ttg.memdesc<64x96xf16, #b, #ttg.shared_memory, 64x128>, !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    ttng.tc_gen5_mma %a, %b128, %c128, %p, %p : !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>, !ttg.memdesc<64x128xf16, #b, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma_scaled %a8, %b8, %c, %sa, %sb, %p, %p lhs = e4m3 rhs = e4m3 : !ttg.memdesc<128x64xf8E4M3FN, #a8, #ttg.shared_memory>, !ttg.memdesc<64x96xf8E4M3FN, #b8, #ttg.shared_memory, 64x128>, !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>, !ttg.memdesc<128x2xi8, #scales, #ttng.tensor_memory>, !ttg.memdesc<96x2xi8, #scales, #ttng.tensor_memory, 128x2>
    tt.return
  }
}

// -----

#a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @n100(%a: !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>,
                %b: !ttg.memdesc<64x100xf16, #b, #ttg.shared_memory, 64x128>,
                %c: !ttg.memdesc<128x100xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>,
                %p: i1) {
    // expected-error @+1 {{NPOT MMAv5 shape 128x100 is unsupported: tcgen05.mma kind f16/tf32 SS K-major B cta_group::1 per-CTA M=128 requires N to be a multiple of 8 in [8, 256], got N=100}}
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>, !ttg.memdesc<64x100xf16, #b, #ttg.shared_memory, 64x128>, !ttg.memdesc<128x100xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

#b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmema = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @ts_n104(%a: !ttg.memdesc<128x64xf16, #tmema, #ttng.tensor_memory>,
                   %b: !ttg.memdesc<64x104xf16, #b, #ttg.shared_memory, 64x128>,
                   %c: !ttg.memdesc<128x104xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>,
                   %p: i1) {
    // expected-error @+1 {{NPOT MMAv5 shape 128x104 is unsupported: tcgen05.mma kind f16/tf32 TS MN-major B cta_group::1 per-CTA M=128 requires N to be a multiple of 16}}
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x64xf16, #tmema, #ttng.tensor_memory>, !ttg.memdesc<64x104xf16, #b, #ttg.shared_memory, 64x128>, !ttg.memdesc<128x104xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

#a = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @f8_mn_major_n40(%a: !ttg.memdesc<128x64xf8E4M3FN, #a, #ttg.shared_memory>,
                           %b: !ttg.memdesc<64x40xf8E4M3FN, #b, #ttg.shared_memory, 64x64>,
                           %c: !ttg.memdesc<128x40xf32, #tmem, #ttng.tensor_memory, mutable, 128x64>,
                           %p: i1) {
    // expected-error @+1 {{NPOT MMAv5 shape 128x40 is unsupported: tcgen05.mma kind f8f6f4 SS MN-major B cta_group::1 per-CTA M=128 requires N to be a multiple of 16}}
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x64xf8E4M3FN, #a, #ttg.shared_memory>, !ttg.memdesc<64x40xf8E4M3FN, #b, #ttg.shared_memory, 64x64>, !ttg.memdesc<128x40xf32, #tmem, #ttng.tensor_memory, mutable, 128x64>
    tt.return
  }
}

// -----

#a = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @i8_n36(%a: !ttg.memdesc<128x64xi8, #a, #ttg.shared_memory>,
                  %b: !ttg.memdesc<64x36xi8, #b, #ttg.shared_memory, 64x64>,
                  %c: !ttg.memdesc<128x36xi32, #tmem, #ttng.tensor_memory, mutable, 128x64>,
                  %p: i1) {
    // expected-error @+1 {{NPOT MMAv5 shape 128x36 is unsupported: tcgen05.mma kind i8 SS K-major B cta_group::1 per-CTA M=128 requires N to be a multiple of 16}}
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x64xi8, #a, #ttg.shared_memory>, !ttg.memdesc<64x36xi8, #b, #ttg.shared_memory, 64x64>, !ttg.memdesc<128x36xi32, #tmem, #ttng.tensor_memory, mutable, 128x64>
    tt.return
  }
}

// -----

// No multiple of 8 divides N300, so blockN 256 leaves a 44-column tail.
#a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {
  tt.func @ragged_n300(%a: !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>,
                       %b: !ttg.memdesc<64x300xf16, #b, #ttg.shared_memory, 64x512>,
                       %c: !ttg.memdesc<128x300xf32, #tmem, #ttng.tensor_memory, mutable, 128x512>,
                       %p: i1) {
    // expected-error @+1 {{NPOT MMAv5 N that is not a multiple of the instruction N is not yet supported}}
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>, !ttg.memdesc<64x300xf16, #b, #ttg.shared_memory, 64x512>, !ttg.memdesc<128x300xf32, #tmem, #ttng.tensor_memory, mutable, 128x512>
    tt.return
  }
}

// -----

#a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @m192(%a: !ttg.memdesc<192x64xf16, #a, #ttg.shared_memory>,
                %b: !ttg.memdesc<64x128xf16, #b, #ttg.shared_memory>,
                %c: !ttg.memdesc<192x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>,
                %p: i1) {
    // expected-error @+1 {{NPOT MMAv5 M is not yet supported}}
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<192x64xf16, #a, #ttg.shared_memory>, !ttg.memdesc<64x128xf16, #b, #ttg.shared_memory>, !ttg.memdesc<192x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    tt.return
  }
}

// -----

#a = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @k40(%a: !ttg.memdesc<128x40xf16, #a, #ttg.shared_memory, 128x64>,
               %b: !ttg.memdesc<40x128xf16, #b, #ttg.shared_memory>,
               %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
               %p: i1) {
    // expected-error @+1 {{NPOT MMAv5 K=40 is unsupported: tcgen05.mma takes K in steps of 16}}
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x40xf16, #a, #ttg.shared_memory, 128x64>, !ttg.memdesc<40x128xf16, #b, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// A compact NPOT shared operand reaches the same check.
#a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @compact_n100(%a: !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>,
                        %b: !ttg.memdesc<64x100xf16, #b, #ttg.shared_memory>,
                        %c: !ttg.memdesc<128x100xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>,
                        %p: i1) {
    // expected-error @+1 {{NPOT MMAv5 shape 128x100 is unsupported: tcgen05.mma kind f16/tf32 SS K-major B cta_group::1 per-CTA M=128 requires N to be a multiple of 8 in [8, 256], got N=100}}
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x64xf16, #a, #ttg.shared_memory>, !ttg.memdesc<64x100xf16, #b, #ttg.shared_memory>, !ttg.memdesc<128x100xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

#a = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @compact_k40(%a: !ttg.memdesc<128x40xf16, #a, #ttg.shared_memory>,
                       %b: !ttg.memdesc<40x128xf16, #b, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %p: i1) {
    // expected-error @+1 {{NPOT MMAv5 K=40 is unsupported: tcgen05.mma takes K in steps of 16}}
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x40xf16, #a, #ttg.shared_memory>, !ttg.memdesc<40x128xf16, #b, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @m64_n96(%a: !ttg.memdesc<64x64xf16, #a, #ttg.shared_memory>,
                   %b: !ttg.memdesc<64x96xf16, #b, #ttg.shared_memory, 64x128>,
                   %c: !ttg.memdesc<64x96xf32, #tmem, #ttng.tensor_memory, mutable, 64x128>,
                   %p: i1) {
    // expected-error @+1 {{NPOT MMAv5 N with blockM=64 is not yet supported}}
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<64x64xf16, #a, #ttg.shared_memory>, !ttg.memdesc<64x96xf16, #b, #ttg.shared_memory, 64x128>, !ttg.memdesc<64x96xf32, #tmem, #ttng.tensor_memory, mutable, 64x128>
    tt.return
  }
}
