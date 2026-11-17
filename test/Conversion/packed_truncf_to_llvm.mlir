// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm="compute-capability=90 ptx-version=93" -reconcile-unrealized-casts 2>/dev/null | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 104, 16]}>
#s2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: truncf_f32_to_bf16_paired
  tt.func @truncf_f32_to_bf16_paired() {
    // The exact-layout accumulator convert pairs into one packed
    // convert per element pair. PTX puts the first source in the
    // upper half, so the asm passes element 1 ($2) before element
    // 0 ($1).
    // CHECK: llvm.inline_asm {{.*}}"cvt.rn.bf16x2.f32 $0, $2, $1;"
    %src = ttg.local_alloc : () -> !ttg.memdesc<64x104xf32, #s2, #smem, mutable>
    %acc = ttg.local_load %src : !ttg.memdesc<64x104xf32, #s2, #smem, mutable> -> tensor<64x104xf32, #mma>
    %t = arith.truncf %acc : tensor<64x104xf32, #mma> to tensor<64x104xbf16, #mma>
    tt.print "converted: " {hex = false, isSigned = array<i32: 0>} : %t : tensor<64x104xbf16, #mma>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: truncf_f32_to_bf16_scalar
  tt.func @truncf_f32_to_bf16_scalar(%in: tensor<64xf32, #blocked1>, %po: !tt.ptr<bf16>) {
    // A pow2 layout keeps the generic path even with two elements
    // per thread, so its PTX is unchanged.
    // CHECK-NOT: llvm.inline_asm
    // CHECK: llvm.fptrunc
    // CHECK: llvm.fptrunc
    %out = arith.truncf %in : tensor<64xf32, #blocked1> to tensor<64xbf16, #blocked1>
    %sp = tt.splat %po : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>, #blocked1>
    tt.store %sp, %out : tensor<64x!tt.ptr<bf16>, #blocked1>
    tt.return
  }
}
