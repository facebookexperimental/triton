// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 -verify-diagnostics | FileCheck %s --enable-var-scope

// An NPOT K's odd part is a register digit of the operands, so each warp
// issues K/16 32x32x16 MFMAs and reads only K's 96 columns from LDS.
#mfma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>
#sa = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#sb = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @dot_k96
  tt.func @dot_k96(%c: tensor<64x64xf32, #mfma>, %p: tensor<64x64x!tt.ptr<f32>, #mfma>) {
    %sa = ttg.local_alloc : () -> !ttg.memdesc<64x96xbf16, #sa, #smem, mutable>
    %sb = ttg.local_alloc : () -> !ttg.memdesc<96x64xbf16, #sb, #smem, mutable>
    // CHECK-COUNT-12: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xbf16>
    // CHECK-NOT: llvm.load {{.*}}ptr<3>
    %a = ttg.local_load %sa : !ttg.memdesc<64x96xbf16, #sa, #smem, mutable> -> tensor<64x96xbf16, #a>
    %b = ttg.local_load %sb : !ttg.memdesc<96x64xbf16, #sb, #smem, mutable> -> tensor<96x64xbf16, #b>
    // CHECK-COUNT-6: rocdl.mfma.f32.32x32x16.bf16
    // CHECK-NOT: rocdl.mfma
    %d = tt.dot %a, %b, %c : tensor<64x96xbf16, #a> * tensor<96x64xbf16, #b> -> tensor<64x64xf32, #mfma>
    tt.store %p, %d : tensor<64x64x!tt.ptr<f32>, #mfma>
    tt.return
  }
}

// -----

// Control: K=64 issues 4 MFMAs.
#mfma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @dot_k64
  tt.func @dot_k64(%a: tensor<64x64xbf16, #a>, %b: tensor<64x64xbf16, #b>, %c: tensor<64x64xf32, #mfma>, %p: tensor<64x64x!tt.ptr<f32>, #mfma>) {
    // CHECK-COUNT-4: rocdl.mfma.f32.32x32x16.bf16
    // CHECK-NOT: rocdl.mfma
    %d = tt.dot %a, %b, %c : tensor<64x64xbf16, #a> * tensor<64x64xbf16, #b> -> tensor<64x64xf32, #mfma>
    tt.store %p, %d : tensor<64x64x!tt.ptr<f32>, #mfma>
    tt.return
  }
}

// -----

// K=48's two-part 16 holds no whole 32-wide mfma16 K tile, so the operand
// keeps its pow2-padded K and the verifier rejects it.
#mfma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @operand_k48_mfma16(%x: bf16) {
    // expected-error @+1 {{Number of elements must be power-of-two}}
    %0 = tt.splat %x : bf16 -> tensor<64x48xbf16, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
    tt.return
  }
}
