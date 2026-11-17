// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics --convert-triton-gpu-to-llvm="compute-capability=100 ptx-version=86" | FileCheck %s

// With TRITON_ALLOW_NPOT=1, gather4/scatter4 split an NPOT row into whole
// boxes, one message each. A box that doesn't tile the row and an NPOT row
// count stay rejected.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#sw128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#sw64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // 128 columns: two 64-column boxes.
  // CHECK-LABEL: llvm.func @gather_8x128
  // CHECK-COUNT-2: tile::gather4
  // CHECK-NOT: tile::gather4
  tt.func @gather_8x128(%desc: !tt.tensordesc<1x128xf16, #sw128>, %x: tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, %y: i32, %alloc: !ttg.memdesc<8x128xf16, #sw128, #smem, mutable>, %barrier: !ttg.memdesc<1xi64, #shared0, #smem, mutable>, %pred: i1) {
    ttng.async_tma_gather %desc[%x, %y] %alloc, %barrier, %pred : !tt.tensordesc<1x128xf16, #sw128>, tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<1xi64, #shared0, #smem, mutable>, !ttg.memdesc<8x128xf16, #sw128, #smem, mutable>, i1
    tt.return
  }

  // 96 columns: three 32-column boxes at y, y + 32 and y + 64.
  // CHECK-LABEL: llvm.func @gather_8x96
  // CHECK-DAG: %[[C1:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-DAG: %[[C2:.*]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK: %[[Y:.*]] = llvm.add %arg2
  // CHECK: tile::gather4
  // CHECK: %[[Y1:.*]] = llvm.add %[[Y]], %[[C1]] : i32
  // CHECK-NEXT: tile::gather4{{.*}}, %[[Y1]],
  // CHECK: %[[Y2:.*]] = llvm.add %[[Y]], %[[C2]] : i32
  // CHECK-NEXT: tile::gather4{{.*}}, %[[Y2]],
  // CHECK-NOT: tile::gather4
  tt.func @gather_8x96(%desc: !tt.tensordesc<1x96xf16, #sw64>, %x: tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, %y: i32, %alloc: !ttg.memdesc<8x96xf16, #sw64, #smem, mutable, 8x128>, %barrier: !ttg.memdesc<1xi64, #shared0, #smem, mutable>, %pred: i1) {
    ttng.async_tma_gather %desc[%x, %y] %alloc, %barrier, %pred : !tt.tensordesc<1x96xf16, #sw64>, tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<1xi64, #shared0, #smem, mutable>, !ttg.memdesc<8x96xf16, #sw64, #smem, mutable, 8x128>, i1
    tt.return
  }

  // CHECK-LABEL: llvm.func @scatter_8x96
  // CHECK-COUNT-3: tile::scatter4
  // CHECK-NOT: tile::scatter4
  tt.func @scatter_8x96(%desc: !tt.tensordesc<1x96xf16, #sw64>, %x: tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, %y: i32, %alloc: !ttg.memdesc<8x96xf16, #sw64, #smem, mutable, 8x128>) {
    ttng.async_tma_scatter %desc[%x, %y] %alloc : !tt.tensordesc<1x96xf16, #sw64>, tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<8x96xf16, #sw64, #smem, mutable, 8x128>
    tt.return
  }
}

// -----

// A 64-column box doesn't tile 96 columns.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#sw128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @gather_8x96_sw128(%desc: !tt.tensordesc<1x96xf16, #sw128>, %x: tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, %y: i32, %alloc: !ttg.memdesc<8x96xf16, #sw128, #smem, mutable, 8x128>, %barrier: !ttg.memdesc<1xi64, #shared0, #smem, mutable>, %pred: i1) {
    // expected-error @+1 {{NPOT TMA box 64 does not divide extent 96}}
    ttng.async_tma_gather %desc[%x, %y] %alloc, %barrier, %pred : !tt.tensordesc<1x96xf16, #sw128>, tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<1xi64, #shared0, #smem, mutable>, !ttg.memdesc<8x96xf16, #sw128, #smem, mutable, 8x128>, i1
    tt.return
  }
}

// -----

// 24 rows: the x offsets take an NPOT mixed-radix layout, which gather rejects.
#x24 = #ttg.linear<{register = [[1], [2], [8]], lane = [[0], [0], [0], [0], [0]], warp = [[4], [0]], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [24]}>
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#sw128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @gather_24x64(%desc: !tt.tensordesc<1x64xf16, #sw128>, %i: i32, %y: i32, %alloc: !ttg.memdesc<24x64xf16, #sw128, #smem, mutable, 32x64>, %barrier: !ttg.memdesc<1xi64, #shared0, #smem, mutable>, %pred: i1) {
    %x = tt.splat %i : i32 -> tensor<24xi32, #x24>
    // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on ttng.async_tma_gather}}
    ttng.async_tma_gather %desc[%x, %y] %alloc, %barrier, %pred : !tt.tensordesc<1x64xf16, #sw128>, tensor<24xi32, #x24>, i32, !ttg.memdesc<1xi64, #shared0, #smem, mutable>, !ttg.memdesc<24x64xf16, #sw128, #smem, mutable, 32x64>, i1
    tt.return
  }
}
