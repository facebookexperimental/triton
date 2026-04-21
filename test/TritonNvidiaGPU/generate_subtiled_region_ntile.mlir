// RUN: triton-opt %s --triton-nvidia-gpu-test-generate-subtiled-region | FileCheck %s

// Note: N-tile tests are in a separate file from the 2-tile tests to avoid
// heap corruption from split-input-file when inner splits are erased.

// Test: 4-tile subtiling via nested splits.

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#blocked3d = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked_full = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2d = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3db = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_permb = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked2db = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @four_tile_nested_split
  // CHECK: ttng.subtiled_region
  // CHECK-SAME: tile_mappings = [array<i32: 0,
  // CHECK-SAME: array<i32: 1,
  // CHECK-SAME: array<i32: 2,
  // CHECK-SAME: array<i32: 3,
  // CHECK:   setup {
  // CHECK:     tt.split
  // CHECK:     tt.split
  // CHECK:     tt.split
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   } tile{
  // CHECK:     arith.truncf
  // CHECK:     tt.descriptor_store
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  // CHECK-NOT: tt.split
  tt.func @four_tile_nested_split(
      %buf: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
      %tok: !ttg.async.token,
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %m: i32, %n: i32, %c64: i32, %c128: i32, %c192: i32) {
    %l:2 = ttng.tmem_load %buf[%tok] : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked_full>
    %r1 = tt.reshape %l#0 : tensor<128x256xf32, #blocked_full> -> tensor<128x2x128xf32, #blocked3d>
    %t1 = tt.trans %r1 {order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked3d> -> tensor<128x128x2xf32, #blocked3d_perm>
    %a, %b = tt.split %t1 : tensor<128x128x2xf32, #blocked3d_perm> -> tensor<128x128xf32, #blocked2d>
    %r2a = tt.reshape %a : tensor<128x128xf32, #blocked2d> -> tensor<128x2x64xf32, #blocked3db>
    %t2a = tt.trans %r2a {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3db> -> tensor<128x64x2xf32, #blocked3d_permb>
    %c, %d = tt.split %t2a : tensor<128x64x2xf32, #blocked3d_permb> -> tensor<128x64xf32, #blocked2db>
    %r2b = tt.reshape %b : tensor<128x128xf32, #blocked2d> -> tensor<128x2x64xf32, #blocked3db>
    %t2b = tt.trans %r2b {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3db> -> tensor<128x64x2xf32, #blocked3d_permb>
    %e, %f = tt.split %t2b : tensor<128x64x2xf32, #blocked3d_permb> -> tensor<128x64xf32, #blocked2db>
    %x0 = arith.truncf %c : tensor<128x64xf32, #blocked2db> to tensor<128x64xf16, #blocked2db>
    tt.descriptor_store %desc[%m, %n], %x0 : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked2db>
    %x1 = arith.truncf %d : tensor<128x64xf32, #blocked2db> to tensor<128x64xf16, #blocked2db>
    %n1 = arith.addi %n, %c64 : i32
    tt.descriptor_store %desc[%m, %n1], %x1 : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked2db>
    %x2 = arith.truncf %e : tensor<128x64xf32, #blocked2db> to tensor<128x64xf16, #blocked2db>
    %n2 = arith.addi %n, %c128 : i32
    tt.descriptor_store %desc[%m, %n2], %x2 : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked2db>
    %x3 = arith.truncf %f : tensor<128x64xf32, #blocked2db> to tensor<128x64xf16, #blocked2db>
    %n3 = arith.addi %n, %c192 : i32
    tt.descriptor_store %desc[%m, %n3], %x3 : !tt.tensordesc<tensor<128x64xf16, #shared>>, tensor<128x64xf16, #blocked2db>
    tt.return
  }
}
