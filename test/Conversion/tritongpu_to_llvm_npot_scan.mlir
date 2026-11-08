// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s

// [32, 96] appends radix 3 on 96 as registers 8-23, so sizePerThread block 1
// along 96 is registers 8-11, not 4-7.
#p = #ttg.linear<{register = [[0, 1], [0, 2], [16, 0], [0, 32]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [32, 96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @scan_32x96
  // CHECK: %[[E0:.+]] = llvm.extractvalue %[[V:.+]][0] : !llvm.struct<(i32,
  // CHECK-NEXT: %[[E1:.+]] = llvm.extractvalue %[[V]][1] :
  // CHECK: %[[E8:.+]] = llvm.extractvalue %[[V]][8] :
  // CHECK-NEXT: %[[E9:.+]] = llvm.extractvalue %[[V]][9] :
  // CHECK: llvm.add %[[E0]], %[[E1]] : i32
  // CHECK-COUNT-2: llvm.add
  // CHECK-NEXT: llvm.add %[[E8]], %[[E9]] : i32
  tt.func @scan_32x96(%q: !tt.ptr<i32>) {
    %qs = tt.splat %q : !tt.ptr<i32> -> tensor<32x96x!tt.ptr<i32>, #p>
    %x = tt.load %qs : tensor<32x96x!tt.ptr<i32>, #p>
    %s = "tt.scan"(%x) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %t = arith.addi %a, %b : i32
      tt.scan.return %t : i32
    }) : (tensor<32x96xi32, #p>) -> tensor<32x96xi32, #p>
    tt.store %qs, %s : tensor<32x96x!tt.ptr<i32>, #p>
    tt.return
  }
}

// -----

// [96,80] with a zero register basis: the scan walks the 60 distinct
// registers, so sizePerThread block 1 along 80 is registers 12-15.
#z = #ttg.linear<{register = [[0, 0], [0, 1], [0, 2], [32, 0], [0, 16]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = [], inputRadices = {register = [2, 2, 2, 3, 5], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [96, 80]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @scan_broadcast_register
  // CHECK: %[[E0:.+]] = llvm.extractvalue %[[V:.+]][0] : !llvm.struct<(i32,
  // CHECK-NEXT: %[[E1:.+]] = llvm.extractvalue %[[V]][1] :
  // CHECK: %[[E12:.+]] = llvm.extractvalue %[[V]][12] :
  // CHECK-NEXT: %[[E13:.+]] = llvm.extractvalue %[[V]][13] :
  // CHECK: llvm.add %[[E0]], %[[E1]] : i32
  // CHECK-COUNT-2: llvm.add
  // CHECK-NEXT: llvm.add %[[E12]], %[[E13]] : i32
  tt.func @scan_broadcast_register(%q: !tt.ptr<i32>) {
    %qs = tt.splat %q : !tt.ptr<i32> -> tensor<96x80x!tt.ptr<i32>, #z>
    %x = tt.load %qs : tensor<96x80x!tt.ptr<i32>, #z>
    %s = "tt.scan"(%x) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %t = arith.addi %a, %b : i32
      tt.scan.return %t : i32
    }) : (tensor<96x80xi32, #z>) -> tensor<96x80xi32, #z>
    tt.store %qs, %s : tensor<96x80x!tt.ptr<i32>, #z>
    tt.return
  }
}

// -----

// Control: the same scan on a blocked layout.
#b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @scan_32x128
  // CHECK: %[[E0:.+]] = llvm.extractvalue %[[V:.+]][0] : !llvm.struct<(i32,
  // CHECK-NEXT: %[[E1:.+]] = llvm.extractvalue %[[V]][1] :
  // CHECK: %[[E4:.+]] = llvm.extractvalue %[[V]][4] :
  // CHECK-NEXT: %[[E5:.+]] = llvm.extractvalue %[[V]][5] :
  // CHECK: llvm.add %[[E0]], %[[E1]] : i32
  // CHECK-COUNT-2: llvm.add
  // CHECK-NEXT: llvm.add %[[E4]], %[[E5]] : i32
  tt.func @scan_32x128(%q: !tt.ptr<i32>) {
    %qs = tt.splat %q : !tt.ptr<i32> -> tensor<32x128x!tt.ptr<i32>, #b>
    %x = tt.load %qs : tensor<32x128x!tt.ptr<i32>, #b>
    %s = "tt.scan"(%x) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %t = arith.addi %a, %b : i32
      tt.scan.return %t : i32
    }) : (tensor<32x128xi32, #b>) -> tensor<32x128xi32, #b>
    tt.store %qs, %s : tensor<32x128x!tt.ptr<i32>, #b>
    tt.return
  }
}
