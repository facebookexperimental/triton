// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-optimize-thread-locality | FileCheck %s --check-prefix=LOCAL

#p = #ttg.linear<{register = [[32, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [], block = [], inputRadices = {register = [3, 2, 2, 2, 2, 2, 2], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96, 64]}>
// A mixed-radix gather goes through shared memory with no warp shuffles, and
// the thread-locality pass keeps its layout.
// CHECK: ttg.shared = 24576 : i32
// CHECK-LABEL: llvm.func @gather_96x64
// CHECK-NOT: nvvm.shfl.sync
// CHECK: nvvm.barrier
// CHECK-NOT: nvvm.shfl.sync
// LOCAL-LABEL: @gather_96x64
// LOCAL: tt.gather {{.*}} {axis = 0 : i32} : (tensor<96x64xf32, #linear>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gather_96x64(%x: !tt.ptr<f32>, %i: !tt.ptr<i32>, %y: !tt.ptr<f32>) {
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<96x64x!tt.ptr<f32>, #p>
    %src = tt.load %xs : tensor<96x64x!tt.ptr<f32>, #p>
    %is = tt.splat %i : !tt.ptr<i32> -> tensor<96x64x!tt.ptr<i32>, #p>
    %idx = tt.load %is : tensor<96x64x!tt.ptr<i32>, #p>
    %g = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<96x64xf32, #p>, tensor<96x64xi32, #p>) -> tensor<96x64xf32, #p>
    %ys = tt.splat %y : !tt.ptr<f32> -> tensor<96x64x!tt.ptr<f32>, #p>
    tt.store %ys, %g : tensor<96x64x!tt.ptr<f32>, #p>
    tt.return
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
// Power-of-two control: warp-local, and thread locality marks it efficient.
// CHECK: ttg.shared = 0 : i32
// CHECK-LABEL: llvm.func @gather_128x64
// CHECK: nvvm.shfl.sync idx
// LOCAL-LABEL: @gather_128x64
// LOCAL: tt.gather {{.*}} efficient_layout
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gather_128x64(%src: tensor<128x64xf32, #b>, %idx: tensor<128x64xi32, #b>, %y: !tt.ptr<f32>) {
    %g = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<128x64xf32, #b>, tensor<128x64xi32, #b>) -> tensor<128x64xf32, #b>
    %ys = tt.splat %y : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>, #b>
    tt.store %ys, %g : tensor<128x64x!tt.ptr<f32>, #b>
    tt.return
  }
}
