// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=80 --convert-triton-gpu-to-llvm=compute-capability=80 | FileCheck %s

// Only the MMA tile count of a mixed-radix operand is NPOT. N=96 issues 3/4 of
// the 64x128 control's mma.sync.
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#s = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mma_64x96
  // CHECK-COUNT-48: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
  // CHECK-NOT: mma.sync
  tt.func @mma_64x96(%am: !ttg.memdesc<64x64xf16, #s, #smem>, %bm: !ttg.memdesc<64x96xf16, #s, #smem>) {
    %a = ttg.local_load %am : !ttg.memdesc<64x64xf16, #s, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %b = ttg.local_load %bm : !ttg.memdesc<64x96xf16, #s, #smem> -> tensor<64x96xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %c = arith.constant dense<0.0> : tensor<64x96xf32, #mma>
    %d = tt.dot %a, %b, %c : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x96xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x96xf32, #mma>
    tt.return
  }
  // CHECK-LABEL: @mma_64x128
  // CHECK-COUNT-64: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
  // CHECK-NOT: mma.sync
  tt.func @mma_64x128(%am: !ttg.memdesc<64x64xf16, #s, #smem>, %bm: !ttg.memdesc<64x128xf16, #s, #smem>) {
    %a = ttg.local_load %am : !ttg.memdesc<64x64xf16, #s, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %b = ttg.local_load %bm : !ttg.memdesc<64x128xf16, #s, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %c = arith.constant dense<0.0> : tensor<64x128xf32, #mma>
    %d = tt.dot %a, %b, %c : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x128xf32, #mma>
    tt.return
  }
}
