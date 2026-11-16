// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=88' | FileCheck %s --check-prefixes=CHECK,SM100
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=107 ptx-version=94' | FileCheck %s --check-prefixes=CHECK,SM107

// K=96 issues six K16 MMAs, from shared or tensor memory A.
#a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @k96_f16
  // CHECK-COUNT-6: tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2,
  // CHECK-NOT: tcgen05.mma
  tt.func @k96_f16(%a: !ttg.memdesc<128x96xf16, #a, #ttg.shared_memory>,
                   %b: !ttg.memdesc<96x128xf16, #b, #ttg.shared_memory>,
                   %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                   %p: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x96xf16, #a, #ttg.shared_memory>, !ttg.memdesc<96x128xf16, #b, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // A's 48 packed columns are read 8 per MMA.
  // CHECK-LABEL: @k96_tmem_a
  // CHECK: kind::f16 [ $0 + 0 ], [ $1 + 0 ], $2,
  // CHECK: kind::f16 [ $0 + 0 ], [ $1 + 8 ], $2,
  // CHECK: kind::f16 [ $0 + 0 ], [ $1 + 16 ], $2,
  // CHECK: kind::f16 [ $0 + 0 ], [ $1 + 24 ], $2,
  // CHECK: kind::f16 [ $0 + 0 ], [ $1 + 32 ], $2,
  // CHECK: kind::f16 [ $0 + 0 ], [ $1 + 40 ], $2,
  // CHECK-NOT: tcgen05.mma
  tt.func @k96_tmem_a(%a: !ttg.memdesc<128x96xf16, #tmem, #ttng.tensor_memory, 128x128>,
                      %b: !ttg.memdesc<96x128xf16, #b, #ttg.shared_memory>,
                      %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                      %p: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x96xf16, #tmem, #ttng.tensor_memory, 128x128>, !ttg.memdesc<96x128xf16, #b, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// The 2x fp8 K64 MMA can't tile K=96, so it takes three K32 MMAs; K=128 is
// the control.
#a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @k96_fp8
  // CHECK-COUNT-3: tcgen05.mma.cta_group::1.kind::f8f6f4
  // CHECK-NOT: tcgen05.mma
  tt.func @k96_fp8(%a: !ttg.memdesc<128x96xf8E4M3FN, #a, #ttg.shared_memory>,
                   %b: !ttg.memdesc<96x128xf8E4M3FN, #b, #ttg.shared_memory>,
                   %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                   %p: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x96xf8E4M3FN, #a, #ttg.shared_memory>, !ttg.memdesc<96x128xf8E4M3FN, #b, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @k128_fp8
  // SM100-COUNT-4: tcgen05.mma.cta_group::1.kind::f8f6f4
  // SM107-COUNT-2: tcgen05.mma.cta_group::1.kind::f8f6f4
  // CHECK-NOT: tcgen05.mma
  tt.func @k128_fp8(%a: !ttg.memdesc<128x128xf8E4M3FN, #a, #ttg.shared_memory>,
                    %b: !ttg.memdesc<128x128xf8E4M3FN, #b, #ttg.shared_memory>,
                    %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                    %p: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x128xf8E4M3FN, #a, #ttg.shared_memory>, !ttg.memdesc<128x128xf8E4M3FN, #b, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, twoCTAs = true>
module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttng.two-ctas" = true} {
  // CHECK-LABEL: @k96_2cta
  // CHECK-COUNT-6: tcgen05.mma.cta_group::2.kind::f16
  // CHECK-NOT: tcgen05.mma
  tt.func @k96_2cta(%a: !ttg.memdesc<128x96xf16, #a, #ttg.shared_memory>,
                    %b: !ttg.memdesc<96x128xf16, #b, #ttg.shared_memory>,
                    %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                    %p: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %p, %p {two_ctas} : !ttg.memdesc<128x96xf16, #a, #ttg.shared_memory>, !ttg.memdesc<96x128xf16, #b, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
