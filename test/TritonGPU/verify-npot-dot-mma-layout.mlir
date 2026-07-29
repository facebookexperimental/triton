// Flag ON: NPOT dot-operand / MMA tensors PASS verifyTensorLayout and the
// module round-trips cleanly (FileCheck the op labels are printed back).
// RUN: TRITON_ALLOW_NPOT=1 triton-opt --split-input-file %s | FileCheck %s
//
// Flag OFF (default): the same NPOT tensors are REJECTED by verifyTensorLayout
// with the "not a power of two" diagnostic.
// RUN: triton-opt --split-input-file %s --verify-diagnostics
//
// The NPOT allow-set admits DotOperand/NvidiaMma/AMDMfma/AMDWmma under the flag
// so an NPOT tl.dot's tensors verify; parents keep pow2 sizePerThread/
// threadsPerWarp, only the shape is NPOT. Verifier only, not dot codegen.

// NPOT dot-operand (opIdx = 0). Shape N=96 is not a power of two.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @npot_dot_operand
  tt.func public @npot_dot_operand(%arg0: tensor<16x96xf16, #dot0>) {
    // expected-error @+1 {{which is not a power of two}}
    %0 = ttg.convert_layout %arg0 : tensor<16x96xf16, #dot0> -> tensor<16x96xf16, #blocked>
    tt.return
  }
}

// -----

// NPOT Nvidia MMA (MMAv2) result tensor. Shape N=96 is not a power of two.
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @npot_nvidia_mma
  tt.func public @npot_nvidia_mma(%arg0: tensor<128x96xf16, #mma>) {
    // expected-error @+1 {{which is not a power of two}}
    %0 = ttg.convert_layout %arg0 : tensor<128x96xf16, #mma> -> tensor<128x96xf16, #mma>
    tt.return
  }
}

// -----

// NPOT AMD MFMA (wave64) tensor. Shape N=96 is not a power of two. This is the
// encoding the native AMD MFMA NPOT GEMM path depends on being admitted.
#mfma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @npot_amd_mfma
  tt.func public @npot_amd_mfma(%arg0: tensor<128x96xf16, #mfma>) {
    // expected-error @+1 {{which is not a power of two}}
    %0 = ttg.convert_layout %arg0 : tensor<128x96xf16, #mfma> -> tensor<128x96xf16, #mfma>
    tt.return
  }
}
