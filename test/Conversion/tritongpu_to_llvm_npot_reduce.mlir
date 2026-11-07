// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm='compute-capability=90 enable-tree-reduction=true' | FileCheck %s

// [32, 96] along 96: the radix-3 register digit folds in-thread with the binary
// ones (12 values per row), and shuffles span only the three lane bits on 96.
#p = #ttg.linear<{register = [[0, 1], [0, 2], [16, 0], [0, 32]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [32, 96]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @odd_on_axis
  // CHECK-COUNT-22: llvm.fadd %{{.*}} : f32
  // CHECK-NOT: llvm.fadd
  // CHECK: llvm.mlir.constant(31 : i32)
  // CHECK: %[[X4:.+]] = llvm.mlir.constant(4 : i32)
  // CHECK: nvvm.shfl.sync bfly %{{.+}}, %{{.+}}, %[[X4]],
  // CHECK: %[[X2:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK: nvvm.shfl.sync bfly %{{.+}}, %{{.+}}, %[[X2]],
  // CHECK: %[[X1:.+]] = llvm.mlir.constant(1 : i32)
  // CHECK: nvvm.shfl.sync bfly %{{.+}}, %{{.+}}, %[[X1]],
  // CHECK-COUNT-3: nvvm.shfl.sync bfly
  // CHECK-NOT: nvvm.shfl.sync
  // CHECK-NOT: ptr<3>
  // CHECK: llvm.return
  tt.func public @odd_on_axis(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>) {
    %ps = tt.splat %p : !tt.ptr<f32> -> tensor<32x96x!tt.ptr<f32>, #p>
    %x = tt.load %ps : tensor<32x96x!tt.ptr<f32>, #p>
    %s = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %c = arith.addf %a, %b : f32
      tt.reduce.return %c : f32
    }) : (tensor<32x96xf32, #p>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #p}>>
    %qs = tt.splat %q : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #p}>>
    tt.store %qs, %s : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #p}>>
    tt.return
  }
}

// -----

// Control: the pow2 [32, 128] reduce.
#b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @pow2_control
  // CHECK-COUNT-30: llvm.fadd %{{.*}} : f32
  // CHECK-NOT: llvm.fadd
  // CHECK: llvm.mlir.constant(31 : i32)
  // CHECK: %[[X4:.+]] = llvm.mlir.constant(4 : i32)
  // CHECK: nvvm.shfl.sync bfly %{{.+}}, %{{.+}}, %[[X4]],
  // CHECK: %[[X2:.+]] = llvm.mlir.constant(2 : i32)
  // CHECK: nvvm.shfl.sync bfly %{{.+}}, %{{.+}}, %[[X2]],
  // CHECK: %[[X1:.+]] = llvm.mlir.constant(1 : i32)
  // CHECK: nvvm.shfl.sync bfly %{{.+}}, %{{.+}}, %[[X1]],
  // CHECK-COUNT-3: nvvm.shfl.sync bfly
  // CHECK-NOT: nvvm.shfl.sync
  // CHECK-NOT: ptr<3>
  // CHECK: llvm.return
  tt.func public @pow2_control(%p: !tt.ptr<f32>, %q: !tt.ptr<f32>) {
    %ps = tt.splat %p : !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>, #b>
    %x = tt.load %ps : tensor<32x128x!tt.ptr<f32>, #b>
    %s = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %c = arith.addf %a, %b : f32
      tt.reduce.return %c : f32
    }) : (tensor<32x128xf32, #b>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #b}>>
    %qs = tt.splat %q : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #b}>>
    tt.store %qs, %s : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #b}>>
    tt.return
  }
}

// -----

// A register-only radix-5 digit: f16 pairs can't pack, so the five fold as
// scalars.
#p5 = #ttg.linear<{register = [[1]], lane = [[0], [0], [0], [0], [0]], warp = [], block = [], inputRadices = {register = [5], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [5]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @pure_odd
  // CHECK-COUNT-4: llvm.fadd %{{.*}} : f16
  // CHECK-NOT: llvm.fadd
  // CHECK-NOT: vector<2xf16>
  // CHECK: llvm.return
  tt.func public @pure_odd(%p: !tt.ptr<f16>, %q: !tt.ptr<f16>) {
    %ps = tt.splat %p : !tt.ptr<f16> -> tensor<5x!tt.ptr<f16>, #p5>
    %x = tt.load %ps : tensor<5x!tt.ptr<f16>, #p5>
    %s = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %c = arith.addf %a, %b : f16
      tt.reduce.return %c : f16
    }) : (tensor<5xf16, #p5>) -> f16
    tt.store %q, %s : !tt.ptr<f16>
    tt.return
  }
}
