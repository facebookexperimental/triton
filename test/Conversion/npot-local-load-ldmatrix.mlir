// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=80 --convert-triton-gpu-to-llvm=compute-capability=80 | FileCheck %s

// A mixed-radix dot operand loads its pow2 chunks by ldmatrix, as the padded
// operand loads its own: 3/4 of the 64x128 control's ldmatrix at N 96.
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#s = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
#k96 = #ttg.linear<{register = [[0, 1], [0, 8], [0, 16], [0, 32]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [32, 96]}>
#k128 = #ttg.linear<{register = [[0, 1], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // Each chunk adds its start, column 32 or 64, to the view's column offset.
  // CHECK-LABEL: @b_64x96
  // CHECK-DAG: %[[S1:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-DAG: %[[S2:.*]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK: %[[COL:.*]] = llvm.extractvalue %arg0[2]
  // CHECK-COUNT-4: nvvm.ldmatrix {{.*}}layout = <col>
  // CHECK: llvm.add %[[COL]], %[[S1]] : i32
  // CHECK-COUNT-4: nvvm.ldmatrix {{.*}}layout = <col>
  // CHECK: llvm.add %[[COL]], %[[S2]] : i32
  // CHECK-COUNT-4: nvvm.ldmatrix {{.*}}layout = <col>
  tt.func @b_64x96(%m: !ttg.memdesc<64x96xf16, #s, #smem, mutable>) {
    // CHECK-NOT: nvvm.ldmatrix
    // CHECK-NOT: llvm.load {{.*}}ptr<3>
    %v = ttg.local_load %m : !ttg.memdesc<64x96xf16, #s, #smem, mutable> -> tensor<64x96xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    tt.return
  }
  // CHECK-LABEL: @b_64x128
  tt.func @b_64x128(%m: !ttg.memdesc<64x128xf16, #s, #smem, mutable>) {
    // CHECK-COUNT-16: nvvm.ldmatrix {{.*}}layout = <col>
    // CHECK-NOT: nvvm.ldmatrix
    %v = ttg.local_load %m : !ttg.memdesc<64x128xf16, #s, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    tt.return
  }
  // A dense buffer's 64-column panels sit 64 x 64 elements apart, so row 32 is
  // 4096 bytes on, not the 16384 of the pow2 cover's 256-element rows.
  // CHECK-LABEL: @b_64x192
  // CHECK-DAG: %[[ROW:.*]] = llvm.mlir.constant(4096 : i32) : i32
  // CHECK: nvvm.ldmatrix {{.*}}layout = <col>
  // CHECK: llvm.add %{{.*}}, %[[ROW]] : i32
  // CHECK-NEXT: llvm.getelementptr
  // CHECK-NEXT: nvvm.ldmatrix {{.*}}layout = <col>
  // CHECK-COUNT-22: nvvm.ldmatrix {{.*}}layout = <col>
  tt.func @b_64x192(%m: !ttg.memdesc<64x192xf16, #s, #smem, mutable>) {
    // CHECK-NOT: nvvm.ldmatrix
    // CHECK-NOT: llvm.load {{.*}}ptr<3>
    %v = ttg.local_load %m : !ttg.memdesc<64x192xf16, #s, #smem, mutable> -> tensor<64x192xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    tt.return
  }
  // With 96 rows each 32-row block of a panel is a pow2 region: 32 x 64
  // elements down the panel, 96 x 64 per panel.
  // CHECK-LABEL: @b_96x192
  // CHECK-COUNT-12: nvvm.ldmatrix {{.*}}layout = <col>
  // CHECK: llvm.getelementptr %{{.*}}[6144] : {{.*}}, f16
  // CHECK: nvvm.ldmatrix {{.*}}layout = <col>
  // CHECK: llvm.getelementptr %{{.*}}[8192] : {{.*}}, f16
  // CHECK-COUNT-23: nvvm.ldmatrix {{.*}}layout = <col>
  tt.func @b_96x192(%m: !ttg.memdesc<96x192xf16, #s, #smem, mutable>) {
    // CHECK-NOT: nvvm.ldmatrix
    // CHECK-NOT: llvm.load {{.*}}ptr<3>
    %v = ttg.local_load %m : !ttg.memdesc<96x192xf16, #s, #smem, mutable> -> tensor<96x192xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    tt.return
  }
  // CHECK-LABEL: @a_96x64
  tt.func @a_96x64(%m: !ttg.memdesc<96x64xf16, #s, #smem, mutable>) {
    // CHECK-COUNT-12: nvvm.ldmatrix {{.*}}layout = <row>
    // CHECK-NOT: nvvm.ldmatrix
    // CHECK-NOT: llvm.load {{.*}}ptr<3>
    %v = ttg.local_load %m : !ttg.memdesc<96x64xf16, #s, #smem, mutable> -> tensor<96x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    tt.return
  }
  // tf32 has no transposed ldmatrix, so its chunks take the radix-2 path,
  // as the pow2 load does.
  // CHECK-LABEL: @b_tf32
  tt.func @b_tf32(%m: !ttg.memdesc<32x96xf32, #s, #smem, mutable>, %q: !tt.ptr<f32>) {
    // CHECK-NOT: nvvm.ldmatrix
    // CHECK-COUNT-48: llvm.load {{.*}}ptr<3> -> i32
    // CHECK-NOT: llvm.load
    %v = ttg.local_load %m : !ttg.memdesc<32x96xf32, #s, #smem, mutable> -> tensor<32x96xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %qs = tt.splat %q : !tt.ptr<f32> -> tensor<32x96x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    tt.store %qs, %v : tensor<32x96x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    tt.return
  }
  // CHECK-LABEL: @b_tf32_128
  tt.func @b_tf32_128(%m: !ttg.memdesc<32x128xf32, #s, #smem, mutable>, %q: !tt.ptr<f32>) {
    // CHECK-COUNT-64: llvm.load {{.*}}ptr<3> -> i32
    // CHECK-NOT: llvm.load
    %v = ttg.local_load %m : !ttg.memdesc<32x128xf32, #s, #smem, mutable> -> tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %qs = tt.splat %q : !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    tt.store %qs, %v : tensor<32x128x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    tt.return
  }
  // A K tile loads in its pre-trans layout, not a dot operand, and takes the
  // same chunks: 3/4 of the 32x128 control's ldmatrix.
  // CHECK-LABEL: @k_32x96
  tt.func @k_32x96(%m: !ttg.memdesc<32x96xf16, #s, #smem, mutable>) {
    // CHECK-COUNT-3: nvvm.ldmatrix {{.*}}layout = <row>
    // CHECK-NOT: nvvm.ldmatrix
    // CHECK-NOT: llvm.load {{.*}}ptr<3>
    %v = ttg.local_load %m : !ttg.memdesc<32x96xf16, #s, #smem, mutable> -> tensor<32x96xf16, #k96>
    tt.return
  }
  // CHECK-LABEL: @k_32x128
  tt.func @k_32x128(%m: !ttg.memdesc<32x128xf16, #s, #smem, mutable>) {
    // CHECK-COUNT-4: nvvm.ldmatrix {{.*}}layout = <row>
    // CHECK-NOT: nvvm.ldmatrix
    %v = ttg.local_load %m : !ttg.memdesc<32x128xf16, #s, #smem, mutable> -> tensor<32x128xf16, #k128>
    tt.return
  }
}
