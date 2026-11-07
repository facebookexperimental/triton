// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -verify-diagnostics

// Flag off, this tile aborts in its shared layout, so it has no OFF run.
// A 4-row pow2 alloc shape is smaller than the 8-row swizzle atom.
#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tma_load_3x64(%desc: !tt.tensordesc<3x64xf16, #nvmma>, %alloc: !ttg.memdesc<3x64xf16, #nvmma, #smem, mutable, 4x64>, %x: i32, %barrier: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    // expected-error @+1 {{NPOT TMA tile has no NVMMA shared layout}}
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %alloc, %barrier, %pred : !tt.tensordesc<3x64xf16, #nvmma>, !ttg.memdesc<1xi64, #shared0, #smem> -> !ttg.memdesc<3x64xf16, #nvmma, #smem, mutable, 4x64>
    tt.return
  }
}
