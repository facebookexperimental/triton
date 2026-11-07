// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s

// [32, 96] on four warps: each thread owns 24 distinct elements, none
// replicated, so it issues 24 atomics and a used result needs no shared memory.
#p = #ttg.linear<{register = [[0, 1], [0, 2], [16, 0], [0, 32]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [32, 96]}>
#b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#r = #ttg.linear<{register = [[0, 0], [0, 0], [8, 0], [16, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], warp = [[2, 0], [4, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [48, 1]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @atomics_32x96
  // CHECK-COUNT-24: red.global.gpu.relaxed.add.u32
  // CHECK-NOT: red.global
  // CHECK-COUNT-24: atom.global.acq_rel.gpu.cas.b32
  // CHECK-NOT: atom.global
  // CHECK-NOT: !llvm.ptr<3>
  tt.func public @atomics_32x96(%p: !tt.ptr<i32>) {
    %ps = tt.splat %p : !tt.ptr<i32> -> tensor<32x96x!tt.ptr<i32>, #p>
    %c = arith.constant dense<0> : tensor<32x96xi32, #p>
    %v = arith.constant dense<1> : tensor<32x96xi32, #p>
    %r = tt.atomic_rmw add, relaxed, gpu, %ps, %v : (tensor<32x96x!tt.ptr<i32>, #p>, tensor<32x96xi32, #p>) -> tensor<32x96xi32, #p>
    %o = tt.atomic_cas acq_rel, gpu, %ps, %c, %v : (tensor<32x96x!tt.ptr<i32>, #p>, tensor<32x96xi32, #p>, tensor<32x96xi32, #p>) -> tensor<32x96xi32, #p>
    tt.store %ps, %o : tensor<32x96x!tt.ptr<i32>, #p>
    tt.return
  }

  // Control: the pow2 [32, 128] on four warps.
  // CHECK-LABEL: llvm.func @atomics_32x128
  // CHECK-COUNT-32: red.global.gpu.relaxed.add.u32
  // CHECK-NOT: red.global
  // CHECK-COUNT-32: atom.global.acq_rel.gpu.cas.b32
  // CHECK-NOT: atom.global
  tt.func public @atomics_32x128(%p: !tt.ptr<i32>) {
    %ps = tt.splat %p : !tt.ptr<i32> -> tensor<32x128x!tt.ptr<i32>, #b>
    %c = arith.constant dense<0> : tensor<32x128xi32, #b>
    %v = arith.constant dense<1> : tensor<32x128xi32, #b>
    %r = tt.atomic_rmw add, relaxed, gpu, %ps, %v : (tensor<32x128x!tt.ptr<i32>, #b>, tensor<32x128xi32, #b>) -> tensor<32x128xi32, #b>
    %o = tt.atomic_cas acq_rel, gpu, %ps, %c, %v : (tensor<32x128x!tt.ptr<i32>, #b>, tensor<32x128xi32, #b>, tensor<32x128xi32, #b>) -> tensor<32x128xi32, #b>
    tt.store %ps, %o : tensor<32x128x!tt.ptr<i32>, #b>
    tt.return
  }

  // [48, 1] with two broadcast register digits: 6 elements in 24 registers.
  // CHECK-LABEL: llvm.func @broadcast_registers_48x1
  // CHECK-COUNT-6: red.global.gpu.relaxed.add.u32
  // CHECK-NOT: red.global
  tt.func public @broadcast_registers_48x1(%p: !tt.ptr<i32>) {
    %ps = tt.splat %p : !tt.ptr<i32> -> tensor<48x1x!tt.ptr<i32>, #r>
    %v = arith.constant dense<1> : tensor<48x1xi32, #r>
    %r = tt.atomic_rmw add, relaxed, gpu, %ps, %v : (tensor<48x1x!tt.ptr<i32>, #r>, tensor<48x1xi32, #r>) -> tensor<48x1xi32, #r>
    tt.return
  }
}
