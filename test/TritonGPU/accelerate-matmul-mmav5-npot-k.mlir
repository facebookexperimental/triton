// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s

// MMAv5 needs a power-of-two K (A sits in an NVMMA shared memdesc, where only
// the leading dim may be NPOT), so an NPOT K falls through: K24 is not whole
// 32-byte K steps and stays on FMA, K48 gets MMAv2. Pow2 K32 keeps MMAv5.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @k24_f16_row_major
  // CHECK-NOT: ttg.local_alloc
  // CHECK-NOT: ttng.tc_gen5_mma
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #blocked{{[0-9]*}}>
  // CHECK-NOT: ttng.tc_gen5_mma
  // CHECK: tt.return
  tt.func @k24_f16_row_major(%a: tensor<64x24xf16, #blocked1>, %b: tensor<24x64xf16, #blocked1>, %c: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
    %ad = ttg.convert_layout %a : tensor<64x24xf16, #blocked1> -> tensor<64x24xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bd = ttg.convert_layout %b : tensor<24x64xf16, #blocked1> -> tensor<24x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %d = tt.dot %ad, %bd, %c : tensor<64x24xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<24x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: #[[$MMA:.+]] = #ttg.nvidia_mma<{versionMajor = 2
  // CHECK-LABEL: tt.func @k48_f16_aligned
  // CHECK-NOT: ttng.tc_gen5_mma
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$MMA]]>
  // CHECK-NOT: ttng.tc_gen5_mma
  // CHECK: tt.return
  tt.func @k48_f16_aligned(%a: tensor<64x48xf16, #blocked1>, %b: tensor<48x64xf16, #blocked1>, %c: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
    %ad = ttg.convert_layout %a : tensor<64x48xf16, #blocked1> -> tensor<64x48xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bd = ttg.convert_layout %b : tensor<48x64xf16, #blocked1> -> tensor<48x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %d = tt.dot %ad, %bd, %c : tensor<64x48xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<48x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @k32_f16_pow2
  // CHECK: ttng.tc_gen5_mma
  tt.func @k32_f16_pow2(%a: tensor<64x32xf16, #blocked1>, %b: tensor<32x64xf16, #blocked1>, %c: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
    %ad = ttg.convert_layout %a : tensor<64x32xf16, #blocked1> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bd = ttg.convert_layout %b : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %d = tt.dot %ad, %bd, %c : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }
}
