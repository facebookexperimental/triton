// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' | FileCheck %s

// Under a 256-column TMEM block, N=320 issues the largest N whose B tiles
// start on a swizzle pattern: two n160 MMAs for a K-major B, the second at
// TMEM column 160, and five n64 MMAs for an N-major B.
#a = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#bk = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#bn = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @k_major_b_n320
  // CHECK: tcgen05.mma.cta_group::1.kind::f16{{.*}} [ $0 + 0 ]
  // CHECK: tcgen05.mma.cta_group::1.kind::f16{{.*}} [ $0 + 160 ]
  // CHECK-NOT: tcgen05.mma.cta_group
  tt.func @k_major_b_n320(%a: !ttg.memdesc<128x16xf16, #a, #ttg.shared_memory>,
                          %b: !ttg.memdesc<16x320xf16, #bk, #ttg.shared_memory, 16x512>,
                          %c: !ttg.memdesc<128x320xf32, #tmem, #ttng.tensor_memory, mutable, 128x512>, %p: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x16xf16, #a, #ttg.shared_memory>, !ttg.memdesc<16x320xf16, #bk, #ttg.shared_memory, 16x512>, !ttg.memdesc<128x320xf32, #tmem, #ttng.tensor_memory, mutable, 128x512>
    tt.return
  }

  // CHECK-LABEL: @n_major_b_n320
  // CHECK: tcgen05.mma.cta_group::1.kind::f16{{.*}} [ $0 + 0 ]
  // CHECK: tcgen05.mma.cta_group::1.kind::f16{{.*}} [ $0 + 64 ]
  // CHECK: tcgen05.mma.cta_group::1.kind::f16{{.*}} [ $0 + 128 ]
  // CHECK: tcgen05.mma.cta_group::1.kind::f16{{.*}} [ $0 + 192 ]
  // CHECK: tcgen05.mma.cta_group::1.kind::f16{{.*}} [ $0 + 256 ]
  // CHECK-NOT: tcgen05.mma.cta_group
  tt.func @n_major_b_n320(%a: !ttg.memdesc<128x16xf16, #a, #ttg.shared_memory>,
                          %b: !ttg.memdesc<16x320xf16, #bn, #ttg.shared_memory, 16x512>,
                          %c: !ttg.memdesc<128x320xf32, #tmem, #ttng.tensor_memory, mutable, 128x512>, %p: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %p, %p : !ttg.memdesc<128x16xf16, #a, #ttg.shared_memory>, !ttg.memdesc<16x320xf16, #bn, #ttg.shared_memory, 16x512>, !ttg.memdesc<128x320xf32, #tmem, #ttng.tensor_memory, mutable, 128x512>
    tt.return
  }
}
