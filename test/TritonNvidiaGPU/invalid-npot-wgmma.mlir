// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics

// i8 K48's power-of-two part, 16, holds no whole k32 atom.
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>
#a = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @k48_i8(%a: !ttg.memdesc<128x48xi8, #a, #smem>, %b: !ttg.memdesc<48x128xi8, #b, #smem>, %c: tensor<128x128xi32, #mma>) {
    // expected-error @+1 {{NPOT K 48 doesn't hold whole k32 WGMMA atoms}}
    %d = ttng.warp_group_dot %a, %b, %c : !ttg.memdesc<128x48xi8, #a, #smem> * !ttg.memdesc<48x128xi8, #b, #smem> -> tensor<128x128xi32, #mma>
    tt.return
  }
}

// -----

// Control: f16 K96's power-of-two part, 32, holds whole k16 atoms.
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#a = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @k96_f16(%a: !ttg.memdesc<128x96xf16, #a, #smem>, %b: !ttg.memdesc<96x128xf16, #b, #smem>, %c: tensor<128x128xf32, #mma>) {
    %d = ttng.warp_group_dot %a, %b, %c : !ttg.memdesc<128x96xf16, #a, #smem> * !ttg.memdesc<96x128xf16, #b, #smem> -> tensor<128x128xf32, #mma>
    tt.return
  }
}
