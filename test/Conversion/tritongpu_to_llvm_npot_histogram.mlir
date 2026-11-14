// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s

// [768] on one warp: no element has a second holder, so no atomic needs an
// owner mask. The 48 bins are a mixed-radix layout too: each thread reads bins
// lane % 16 + {0, 16, 32}.
#p = #ttg.linear<{register = [[32], [64], [128], [256]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [768]}>
#pb = #ttg.linear<{register = [[16]], lane = [[1], [2], [4], [8], [0]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [48]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @hist_768
  // CHECK-COUNT-24: llvm.atomicrmw add
  // CHECK-NOT: llvm.atomicrmw
  // CHECK-COUNT-3: llvm.getelementptr %{{.+}}[%{{.+}}] : (!llvm.ptr<3>, i32)
  tt.func public @hist_768(%p: !tt.ptr<i32>, %q: !tt.ptr<i32>) {
    %ps = tt.splat %p : !tt.ptr<i32> -> tensor<768x!tt.ptr<i32>, #p>
    %x = tt.load %ps : tensor<768x!tt.ptr<i32>, #p>
    %h = tt.histogram %x : tensor<768xi32, #p> -> tensor<48xi32, #pb>
    %qs = tt.splat %q : !tt.ptr<i32> -> tensor<48x!tt.ptr<i32>, #pb>
    tt.store %qs, %h : tensor<48x!tt.ptr<i32>, #pb>
    tt.return
  }
}

// -----

// Control: the pow2 [1024] on one warp.
#b1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @hist_1024
  // CHECK-COUNT-32: llvm.atomicrmw add
  // CHECK-NOT: llvm.atomicrmw
  tt.func public @hist_1024(%p: !tt.ptr<i32>, %q: !tt.ptr<i32>) {
    %ps = tt.splat %p : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #b1>
    %x = tt.load %ps : tensor<1024x!tt.ptr<i32>, #b1>
    %h = tt.histogram %x : tensor<1024xi32, #b1> -> tensor<16xi32, #b1>
    %qs = tt.splat %q : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #b1>
    tt.store %qs, %h : tensor<16x!tt.ptr<i32>, #b1>
    tt.return
  }
}

// -----

// [96] on four warps with lane bit 4 and both warp bits broadcast: eight
// threads hold each element, so only the holder with those bits clear may count
// it, or each bin would be eight times too large.
#h = #ttg.linear<{register = [[1], [32]], lane = [[2], [4], [8], [16], [0]], warp = [[0], [0]], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [96]}>
#b4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @hist_96_broadcast
  // CHECK-DAG: %[[LANE_MASK:.+]] = llvm.mlir.constant(16 : i32)
  // CHECK-DAG: %[[WARP_MASK:.+]] = llvm.mlir.constant(3 : i32)
  // CHECK: %[[LANE_BIT:.+]] = llvm.and %{{.+}}, %[[LANE_MASK]] : i32
  // CHECK: %[[LANE_OK:.+]] = llvm.icmp "eq" %[[LANE_BIT]], %{{.+}} : i32
  // CHECK: %[[WARP_BITS:.+]] = llvm.and %{{.+}}, %[[WARP_MASK]] : i32
  // CHECK: %[[WARP_OK:.+]] = llvm.icmp "eq" %[[WARP_BITS]], %{{.+}} : i32
  // CHECK: %[[OWNER:.+]] = llvm.and %[[LANE_OK]], %[[WARP_OK]] : i1
  // CHECK: llvm.and %{{.+}}, %[[OWNER]] : i1
  // CHECK: llvm.atomicrmw add
  // CHECK: llvm.and %{{.+}}, %[[OWNER]] : i1
  // CHECK: llvm.atomicrmw add
  // CHECK: llvm.and %{{.+}}, %[[OWNER]] : i1
  // CHECK: llvm.atomicrmw add
  // CHECK: llvm.and %{{.+}}, %[[OWNER]] : i1
  // CHECK: llvm.atomicrmw add
  // CHECK: llvm.and %{{.+}}, %[[OWNER]] : i1
  // CHECK: llvm.atomicrmw add
  // CHECK: llvm.and %{{.+}}, %[[OWNER]] : i1
  // CHECK: llvm.atomicrmw add
  // CHECK-NOT: llvm.atomicrmw
  tt.func public @hist_96_broadcast(%p: !tt.ptr<i32>, %q: !tt.ptr<i32>) {
    %ps = tt.splat %p : !tt.ptr<i32> -> tensor<96x!tt.ptr<i32>, #h>
    %x = tt.load %ps : tensor<96x!tt.ptr<i32>, #h>
    %h = tt.histogram %x : tensor<96xi32, #h> -> tensor<16xi32, #b4>
    %qs = tt.splat %q : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #b4>
    tt.store %qs, %h : tensor<16x!tt.ptr<i32>, #b4>
    tt.return
  }
}
