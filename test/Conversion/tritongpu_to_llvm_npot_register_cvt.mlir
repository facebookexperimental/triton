// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s
//
// NPOT register-only convert_layout (order swap, same lanes/warps). In #regB
// the pow2 M register bit (128,0) sits between the NPOT dim1 = 6 bits, which
// the modular invertAndCompose remap alone scrambled (12 of 16 registers).

#regA = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#regB = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // regB -> regA: the remap must be repaired.
  // CHECK-LABEL: llvm.func @npot_register_cvt_regB_to_regA
  // CHECK: %[[S0:[0-9]+]] = llvm.extractvalue %arg0[0]
  // CHECK-NEXT: %[[S1:[0-9]+]] = llvm.extractvalue %arg0[1]
  // CHECK-NEXT: %[[S2:[0-9]+]] = llvm.extractvalue %arg0[2]
  // CHECK-NEXT: %[[S3:[0-9]+]] = llvm.extractvalue %arg0[3]
  // CHECK-NEXT: %[[S4:[0-9]+]] = llvm.extractvalue %arg0[4]
  // CHECK-NEXT: %[[S5:[0-9]+]] = llvm.extractvalue %arg0[5]
  // CHECK-NEXT: %[[S6:[0-9]+]] = llvm.extractvalue %arg0[6]
  // CHECK-NEXT: %[[S7:[0-9]+]] = llvm.extractvalue %arg0[7]
  // CHECK-NEXT: %[[S8:[0-9]+]] = llvm.extractvalue %arg0[8]
  // CHECK-NEXT: %[[S9:[0-9]+]] = llvm.extractvalue %arg0[9]
  // CHECK-NEXT: %[[S10:[0-9]+]] = llvm.extractvalue %arg0[10]
  // CHECK-NEXT: %[[S11:[0-9]+]] = llvm.extractvalue %arg0[11]
  // CHECK-NEXT: %[[S12:[0-9]+]] = llvm.extractvalue %arg0[12]
  // CHECK-NEXT: %[[S13:[0-9]+]] = llvm.extractvalue %arg0[13]
  // CHECK-NEXT: %[[S14:[0-9]+]] = llvm.extractvalue %arg0[14]
  // CHECK-NEXT: %[[S15:[0-9]+]] = llvm.extractvalue %arg0[15]
  // CHECK-NEXT: llvm.mlir.undef
  // CHECK-NEXT: llvm.insertvalue %[[S0]], %{{[0-9]+}}[0]
  // CHECK-NEXT: llvm.insertvalue %[[S1]], %{{[0-9]+}}[1]
  // CHECK-NEXT: llvm.insertvalue %[[S4]], %{{[0-9]+}}[2]
  // CHECK-NEXT: llvm.insertvalue %[[S5]], %{{[0-9]+}}[3]
  // CHECK-NEXT: llvm.insertvalue %[[S8]], %{{[0-9]+}}[4]
  // CHECK-NEXT: llvm.insertvalue %[[S9]], %{{[0-9]+}}[5]
  // CHECK-NEXT: llvm.insertvalue %[[S0]], %{{[0-9]+}}[6]
  // CHECK-NEXT: llvm.insertvalue %[[S1]], %{{[0-9]+}}[7]
  // CHECK-NEXT: llvm.insertvalue %[[S2]], %{{[0-9]+}}[8]
  // CHECK-NEXT: llvm.insertvalue %[[S3]], %{{[0-9]+}}[9]
  // CHECK-NEXT: llvm.insertvalue %[[S6]], %{{[0-9]+}}[10]
  // CHECK-NEXT: llvm.insertvalue %[[S7]], %{{[0-9]+}}[11]
  // CHECK-NEXT: llvm.insertvalue %[[S10]], %{{[0-9]+}}[12]
  // CHECK-NEXT: llvm.insertvalue %[[S11]], %{{[0-9]+}}[13]
  // CHECK-NEXT: llvm.insertvalue %[[S2]], %{{[0-9]+}}[14]
  // CHECK-NEXT: llvm.insertvalue %[[S3]], %{{[0-9]+}}[15]
  tt.func @npot_register_cvt_regB_to_regA(%arg0: tensor<256x6xf16, #regB>) {
    %0 = ttg.convert_layout %arg0 : tensor<256x6xf16, #regB> -> tensor<256x6xf16, #regA>
    tt.return
  }
}

// -----

#regA = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#regB = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // regA -> regB: the NPOT register bits of #regA are contiguous, the remap was already right and is kept.
  // CHECK-LABEL: llvm.func @npot_register_cvt_regA_to_regB
  // CHECK: %[[S0:[0-9]+]] = llvm.extractvalue %arg0[0]
  // CHECK-NEXT: %[[S1:[0-9]+]] = llvm.extractvalue %arg0[1]
  // CHECK-NEXT: %[[S2:[0-9]+]] = llvm.extractvalue %arg0[2]
  // CHECK-NEXT: %[[S3:[0-9]+]] = llvm.extractvalue %arg0[3]
  // CHECK-NEXT: %[[S4:[0-9]+]] = llvm.extractvalue %arg0[4]
  // CHECK-NEXT: %[[S5:[0-9]+]] = llvm.extractvalue %arg0[5]
  // CHECK-NEXT: %[[S6:[0-9]+]] = llvm.extractvalue %arg0[6]
  // CHECK-NEXT: %[[S7:[0-9]+]] = llvm.extractvalue %arg0[7]
  // CHECK-NEXT: %[[S8:[0-9]+]] = llvm.extractvalue %arg0[8]
  // CHECK-NEXT: %[[S9:[0-9]+]] = llvm.extractvalue %arg0[9]
  // CHECK-NEXT: %[[S10:[0-9]+]] = llvm.extractvalue %arg0[10]
  // CHECK-NEXT: %[[S11:[0-9]+]] = llvm.extractvalue %arg0[11]
  // CHECK-NEXT: %[[S12:[0-9]+]] = llvm.extractvalue %arg0[12]
  // CHECK-NEXT: %[[S13:[0-9]+]] = llvm.extractvalue %arg0[13]
  // CHECK-NEXT: %[[S14:[0-9]+]] = llvm.extractvalue %arg0[14]
  // CHECK-NEXT: %[[S15:[0-9]+]] = llvm.extractvalue %arg0[15]
  // CHECK-NEXT: llvm.mlir.undef
  // CHECK-NEXT: llvm.insertvalue %[[S0]], %{{[0-9]+}}[0]
  // CHECK-NEXT: llvm.insertvalue %[[S1]], %{{[0-9]+}}[1]
  // CHECK-NEXT: llvm.insertvalue %[[S8]], %{{[0-9]+}}[2]
  // CHECK-NEXT: llvm.insertvalue %[[S9]], %{{[0-9]+}}[3]
  // CHECK-NEXT: llvm.insertvalue %[[S2]], %{{[0-9]+}}[4]
  // CHECK-NEXT: llvm.insertvalue %[[S3]], %{{[0-9]+}}[5]
  // CHECK-NEXT: llvm.insertvalue %[[S10]], %{{[0-9]+}}[6]
  // CHECK-NEXT: llvm.insertvalue %[[S11]], %{{[0-9]+}}[7]
  // CHECK-NEXT: llvm.insertvalue %[[S4]], %{{[0-9]+}}[8]
  // CHECK-NEXT: llvm.insertvalue %[[S5]], %{{[0-9]+}}[9]
  // CHECK-NEXT: llvm.insertvalue %[[S12]], %{{[0-9]+}}[10]
  // CHECK-NEXT: llvm.insertvalue %[[S13]], %{{[0-9]+}}[11]
  // CHECK-NEXT: llvm.insertvalue %[[S6]], %{{[0-9]+}}[12]
  // CHECK-NEXT: llvm.insertvalue %[[S7]], %{{[0-9]+}}[13]
  // CHECK-NEXT: llvm.insertvalue %[[S14]], %{{[0-9]+}}[14]
  // CHECK-NEXT: llvm.insertvalue %[[S15]], %{{[0-9]+}}[15]
  tt.func @npot_register_cvt_regA_to_regB(%arg0: tensor<256x6xf16, #regA>) {
    %0 = ttg.convert_layout %arg0 : tensor<256x6xf16, #regA> -> tensor<256x6xf16, #regB>
    tt.return
  }
}

// -----

// Broadcast registers: dim0 = 2 < sizePerThread[0] = 4, so the remap indexes
// the 8 registers left after removeZeroBasesAlongDim(register), not 16. The
// modular invertAndCompose remap alone was [0,2,1,3,0,2,1,3].
#brA = #ttg.blocked<{sizePerThread = [4, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#brB = #ttg.blocked<{sizePerThread = [4, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @npot_register_cvt_broadcast_brA_to_brB(%arg0: !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
  // CHECK: %[[S0:[0-9]+]] = llvm.extractvalue %arg0[0]
  // CHECK-NEXT: %[[S1:[0-9]+]] = llvm.extractvalue %arg0[1]
  // CHECK-NEXT: %[[S2:[0-9]+]] = llvm.extractvalue %arg0[2]
  // CHECK-NEXT: %[[S3:[0-9]+]] = llvm.extractvalue %arg0[3]
  // CHECK-NEXT: %[[S4:[0-9]+]] = llvm.extractvalue %arg0[4]
  // CHECK-NEXT: %[[S5:[0-9]+]] = llvm.extractvalue %arg0[5]
  // CHECK-NEXT: %[[S6:[0-9]+]] = llvm.extractvalue %arg0[6]
  // CHECK-NEXT: %[[S7:[0-9]+]] = llvm.extractvalue %arg0[7]
  // CHECK-NEXT: llvm.mlir.undef
  // CHECK-NEXT: llvm.insertvalue %[[S0]], %{{[0-9]+}}[0]
  // CHECK-NEXT: llvm.insertvalue %[[S2]], %{{[0-9]+}}[1]
  // CHECK-NEXT: llvm.insertvalue %[[S1]], %{{[0-9]+}}[2]
  // CHECK-NEXT: llvm.insertvalue %[[S3]], %{{[0-9]+}}[3]
  // CHECK-NEXT: llvm.insertvalue %[[S4]], %{{[0-9]+}}[4]
  // CHECK-NEXT: llvm.insertvalue %[[S6]], %{{[0-9]+}}[5]
  // CHECK-NEXT: llvm.insertvalue %[[S5]], %{{[0-9]+}}[6]
  // CHECK-NEXT: llvm.insertvalue %[[S7]], %{{[0-9]+}}[7]
  // CHECK-NEXT: llvm.return
  tt.func @npot_register_cvt_broadcast_brA_to_brB(%arg0: tensor<2x384xf16, #brA>) {
    %0 = ttg.convert_layout %arg0 : tensor<2x384xf16, #brA> -> tensor<2x384xf16, #brB>
    tt.return
  }
}
