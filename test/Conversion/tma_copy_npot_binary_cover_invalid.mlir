// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics --convert-triton-gpu-to-llvm="compute-capability=90 ptx-version=84" | FileCheck %s
// RUN: env -u TRITON_ALLOW_NPOT not triton-opt %s -split-input-file 2>&1 | FileCheck %s --check-prefix=OFF --implicit-check-not="op NPOT"

// TMA lowering splits a tile into strided1D(shapePerCTA / box, box) messages.
// With TRITON_ALLOW_NPOT=1 the op verifier rejects a tile that is not a
// power-of-two count of power-of-two boxes; flag off, the tiles verify.

// 48 rows: the box itself is not a power of two.
// OFF-LABEL: tt.func @tma_load_48x64
// OFF: ttng.async_tma_copy_global_to_local
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_load_48x64(%desc: !tt.tensordesc<48x64xf16, #nvmma>, %alloc: !ttg.memdesc<48x64xf16, #nvmma, #smem, mutable, 64x64>, %x: i32, %barrier: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    // expected-error @+1 {{NPOT TMA tile is not yet supported}}
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %alloc, %barrier, %pred : !tt.tensordesc<48x64xf16, #nvmma>, !ttg.memdesc<1xi64, #shared0, #smem> -> !ttg.memdesc<48x64xf16, #nvmma, #smem, mutable, 64x64>
    tt.return
  }
}

// -----

// 768 rows: three 256-row boxes.
// OFF-LABEL: tt.func @tma_store_768x64
// OFF: ttng.async_tma_copy_local_to_global
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_store_768x64(%desc: !tt.tensordesc<768x64xf16, #nvmma>, %alloc: !ttg.memdesc<768x64xf16, #nvmma, #smem, 1024x64>, %x: i32) {
    // expected-error @+1 {{NPOT TMA tile is not yet supported}}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc : !tt.tensordesc<768x64xf16, #nvmma>, !ttg.memdesc<768x64xf16, #nvmma, #smem, 1024x64>
    tt.return
  }
}

// -----

// Other encodings keep the existing TMA error.
#sw = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_store_swizzled_48x64(%desc: !tt.tensordesc<48x64xf16, #nvmma>, %alloc: !ttg.memdesc<48x64xf16, #sw, #smem, 64x64>, %x: i32) {
    // expected-error @+1 {{TMA descriptor must have NVMMA shared layout}}
    ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc : !tt.tensordesc<48x64xf16, #nvmma>, !ttg.memdesc<48x64xf16, #sw, #smem, 64x64>
    tt.return
  }
}

// -----

// Power-of-two control: 512 rows are two 256-row boxes and lower.
// CHECK-LABEL: llvm.func @tma_load_512x64
// CHECK-COUNT-1: cp.async.bulk.tensor.2d.shared::cluster.global
// OFF-LABEL: tt.func @tma_load_512x64
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_load_512x64(%desc: !tt.tensordesc<512x64xf16, #nvmma>, %alloc: !ttg.memdesc<512x64xf16, #nvmma, #smem, mutable>, %x: i32, %barrier: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %alloc, %barrier, %pred : !tt.tensordesc<512x64xf16, #nvmma>, !ttg.memdesc<1xi64, #shared0, #smem> -> !ttg.memdesc<512x64xf16, #nvmma, #smem, mutable>
    tt.return
  }
}
