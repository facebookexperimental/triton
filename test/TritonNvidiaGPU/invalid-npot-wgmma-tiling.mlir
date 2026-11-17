// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics

// The second n160 tile starts halfway into a 64-column swizzle pattern of B.
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 160, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wgmma_n320(%a: !ttg.memdesc<128x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x320xf16, #shared, #smem>) {
    %c = arith.constant dense<0.0> : tensor<128x320xf32, #mma>
    // expected-error @+1 {{WGMMA B operand has an n160 tile that starts inside a swizzle pattern}}
    %d = ttng.warp_group_dot %a, %b, %c : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x320xf16, #shared, #smem> -> tensor<128x320xf32, #mma>
    tt.return
  }
}

// -----

// Control: the second n192 tile starts on a swizzle pattern.
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 192, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wgmma_n384(%a: !ttg.memdesc<128x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x384xf16, #shared, #smem>) {
    %c = arith.constant dense<0.0> : tensor<128x384xf32, #mma>
    %d = ttng.warp_group_dot %a, %b, %c : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x384xf16, #shared, #smem> -> tensor<128x384xf32, #mma>
    tt.return
  }
}
