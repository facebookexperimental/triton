// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s
//
// FMA tt.dot, NPOT K: a K=24 operand has a modular digit of 32 registers
// (register k >= 24 holds element k % 24). Row m must multiply A registers
// [32m, 32m + 24) with B registers [0, 24); the 8 duplicates are never read.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @fma_dot_modular_k24
  // CHECK-DAG: %[[A0:.*]] = llvm.extractvalue %arg0[0] :
  // CHECK-DAG: %[[A23:.*]] = llvm.extractvalue %arg0[23] :
  // CHECK-DAG: %[[A32:.*]] = llvm.extractvalue %arg0[32] :
  // CHECK-DAG: %[[A55:.*]] = llvm.extractvalue %arg0[55] :
  // CHECK-DAG: %[[B0:.*]] = llvm.extractvalue %arg1[0] :
  // CHECK-DAG: %[[B23:.*]] = llvm.extractvalue %arg1[23] :
  // CHECK: llvm.intr.fmuladd(%[[A0]], %[[B0]], %{{.*}})
  // CHECK-COUNT-22: llvm.intr.fmuladd
  // CHECK: llvm.intr.fmuladd(%[[A23]], %[[B23]], %{{.*}})
  // CHECK-NEXT: llvm.intr.fmuladd(%[[A32]], %[[B0]], %{{.*}})
  // CHECK-COUNT-22: llvm.intr.fmuladd
  // CHECK: llvm.intr.fmuladd(%[[A55]], %[[B23]], %{{.*}})
  // CHECK-NOT: llvm.intr.fmuladd
  // CHECK: llvm.return
  tt.func @fma_dot_modular_k24(%a: tensor<2x24xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<24x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %c: tensor<2x32xf32, #blocked>, %out: tensor<2x32x!tt.ptr<f32>, #blocked>) {
    %d = tt.dot %a, %b, %c, inputPrecision = ieee : tensor<2x24xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<24x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<2x32xf32, #blocked>
    tt.store %out, %d : tensor<2x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Power-of-two control (K=32): the operand register count equals K, so row 1
// starts right after row 0's last register (A register 32), as on master.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @fma_dot_pow2_k32
  // CHECK-DAG: %[[A0:.*]] = llvm.extractvalue %arg0[0] :
  // CHECK-DAG: %[[A31:.*]] = llvm.extractvalue %arg0[31] :
  // CHECK-DAG: %[[A32:.*]] = llvm.extractvalue %arg0[32] :
  // CHECK-DAG: %[[A63:.*]] = llvm.extractvalue %arg0[63] :
  // CHECK-DAG: %[[B0:.*]] = llvm.extractvalue %arg1[0] :
  // CHECK-DAG: %[[B31:.*]] = llvm.extractvalue %arg1[31] :
  // CHECK: llvm.intr.fmuladd(%[[A0]], %[[B0]], %{{.*}})
  // CHECK-COUNT-30: llvm.intr.fmuladd
  // CHECK: llvm.intr.fmuladd(%[[A31]], %[[B31]], %{{.*}})
  // CHECK-NEXT: llvm.intr.fmuladd(%[[A32]], %[[B0]], %{{.*}})
  // CHECK-COUNT-30: llvm.intr.fmuladd
  // CHECK: llvm.intr.fmuladd(%[[A63]], %[[B31]], %{{.*}})
  // CHECK-NOT: llvm.intr.fmuladd
  // CHECK: llvm.return
  tt.func @fma_dot_pow2_k32(%a: tensor<2x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %c: tensor<2x32xf32, #blocked>, %out: tensor<2x32x!tt.ptr<f32>, #blocked>) {
    %d = tt.dot %a, %b, %c, inputPrecision = ieee : tensor<2x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<2x32xf32, #blocked>
    tt.store %out, %d : tensor<2x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
