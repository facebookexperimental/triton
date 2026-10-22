// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --convert-triton-gpu-to-llvm=compute-capability=80 --canonicalize | FileCheck %s

// An NPOT K gives modular MMAv2 operands holding PowerOf2Ceil(K) / kTile K-reps
// per outer rep (2 M-reps, 2 N-reps here). Outer rep 1 must start after all the
// padded K-reps of rep 0, and the wrapped K-reps are never extracted.

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // f16 K80, kTile 16: 5 of 8 K-reps. A rep = 8 elements, B rep = 4.
  // CHECK-LABEL: llvm.func @mmav2_modular_k80_f16
  // CHECK: llvm.extractvalue %arg0[39]
  // CHECK-NOT: llvm.extractvalue %arg0[
  // CHECK: llvm.extractvalue %arg0[64]
  // CHECK: llvm.extractvalue %arg0[103]
  // CHECK-NOT: llvm.extractvalue %arg0[
  // CHECK: llvm.extractvalue %arg1[19]
  // CHECK-NOT: llvm.extractvalue %arg1[
  // CHECK: llvm.extractvalue %arg1[32]
  // CHECK: llvm.extractvalue %arg1[51]
  // CHECK-NOT: llvm.extractvalue
  // CHECK-COUNT-20: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
  // CHECK-NOT: mma.sync
  // CHECK: llvm.return
  tt.func @mmav2_modular_k80_f16(%a: tensor<32x80xf16, #dot_a>, %b: tensor<80x16xf16, #dot_b>) {
    %zero = arith.constant dense<0.0> : tensor<32x16xf32, #mma>
    %r = tt.dot %a, %b, %zero : tensor<32x80xf16, #dot_a> * tensor<80x16xf16, #dot_b> -> tensor<32x16xf32, #mma>
    tt.return
  }
}
