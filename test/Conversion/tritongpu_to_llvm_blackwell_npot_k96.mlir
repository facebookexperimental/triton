// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --triton-tensor-memory-allocation --convert-triton-gpu-to-llvm='compute-capability=103 ptx-version=88' | FileCheck %s

// FP4 K=96 is one MMA; bit 31 of the instruction descriptor selects K96.
#a = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#b = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#s = #ttg.linear<{register = [[0, 1], [32, 0], [64, 0], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 6]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103"} {
  // CHECK-LABEL: @nvfp4_k96
  // CHECK: tcgen05.st.sync.aligned.32x32b.x8.b32
  // CHECK: llvm.mlir.constant(-2011167616 : i32)
  // CHECK-NEXT: tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16
  // CHECK-NOT: tcgen05.mma
  tt.func @nvfp4_k96(%a: !ttg.memdesc<128x48xi8, #a, #ttg.shared_memory>, %b: !ttg.memdesc<48x128xi8, #b, #ttg.shared_memory>, %p: i1) {
    %s = arith.constant dense<1.0> : tensor<128x6xf8E4M3FN, #s>
    %c = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %sm = ttng.tmem_alloc %s : (tensor<128x6xf8E4M3FN, #s>) -> !ttg.memdesc<128x6xf8E4M3FN, #scales, #ttng.tensor_memory, 128x8>
    ttng.tc_gen5_mma_scaled %a, %b, %c, %sm, %sm, %p, %p lhs = e2m1 rhs = e2m1 : !ttg.memdesc<128x48xi8, #a, #ttg.shared_memory>, !ttg.memdesc<48x128xi8, #b, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x6xf8E4M3FN, #scales, #ttng.tensor_memory, 128x8>, !ttg.memdesc<128x6xf8E4M3FN, #scales, #ttng.tensor_memory, 128x8>
    tt.return
  }
}

// -----

// At 8 warps the second warpgroup stores the scales of rows 64..127.
#s = #ttg.linear<{register = [[0, 1], [32, 0], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0], [64, 0]], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2, 2], block = []}, outDimSizes = [128, 6]}>
#scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:103"} {
  // CHECK-LABEL: @scales_8w
  // CHECK-COUNT-2: tcgen05.st.sync.aligned.32x32b.x2.b32
  // CHECK-NOT: tcgen05.st
  tt.func @scales_8w() {
    %s = arith.constant dense<1.0> : tensor<128x6xf8E4M3FN, #s>
    %sm = ttng.tmem_alloc %s : (tensor<128x6xf8E4M3FN, #s>) -> !ttg.memdesc<128x6xf8E4M3FN, #scales, #ttng.tensor_memory, 128x8>
    tt.return
  }
}
