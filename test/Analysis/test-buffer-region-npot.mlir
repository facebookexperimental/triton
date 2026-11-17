// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -mlir-disable-threading -allow-unregistered-dialect -test-buffer-region-alias -o /dev/null 2>&1 | FileCheck %s --check-prefixes=CHECK,POT
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -mlir-disable-threading -allow-unregistered-dialect -test-print-buffer-region -o /dev/null 2>&1 | FileCheck %s --check-prefixes=REGION,POT-REGION
// RUN: env -u TRITON_ALLOW_NPOT not triton-opt %s -split-input-file -mlir-disable-threading -allow-unregistered-dialect -test-buffer-region-alias -o /dev/null 2>&1 | FileCheck %s --check-prefixes=OFF,POT

// An NPOT memdesc's footprint is the leading box of its pow2 alloc shape: a
// subview inside the box aliases it, one past the box does not. With the flag
// off the verifier rejects the NPOT types and the pow2 control is unchanged.

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// REGION: remark: Buffers: [64, 32]
// REGION-NEXT: test.region_name = "tmem_col64"
// REGION: remark: Buffers: [96, 32]
// REGION-NEXT: test.region_name = "tmem_col96"
// REGION: remark: Buffers: [0, 128]
// REGION-NEXT: test.region_name = "tmem_zfull"
// CHECK-LABEL: tmem_col64 vs tmem_col64
// CHECK: tmem_col64 vs tmem_col96: alias=false
// CHECK: tmem_col64 vs tmem_pow2_alloc: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: tmem_col64 vs tmem_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: tmem_col96 vs tmem_pow2_alloc: alias=false
// CHECK: tmem_col96 vs tmem_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: tmem_pow2_alloc vs tmem_zfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// OFF: shape must have power-of-2 and non-zero dimensions; got 128, 96
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @tmem_pow2_alloc() {
    %pow2_alloc = ttng.tmem_alloc {test.region_name = "tmem_pow2_alloc", tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    %full = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %c64 = ttng.tmem_subslice %full {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x32xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    %c96 = ttng.tmem_subslice %full {offset = 96 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x32xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    %1 = ttng.tmem_load %c64 {test.region_name = "tmem_col64"} : !ttg.memdesc<128x32xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x32xf32>
    %2 = ttng.tmem_load %c96 {test.region_name = "tmem_col96"} : !ttg.memdesc<128x32xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x32xf32>
    %3 = ttng.tmem_load %full {test.region_name = "tmem_zfull"} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32>
    tt.return
  }
}

// -----

// Each page of a multibuffered TMEM pow2 alloc shape is a 128x96 box of its own
// 128x128 pow2 alloc shape.
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

// REGION: remark: Buffers: [128, 128]
// REGION-NEXT: test.region_name = "tpage1_pow2_alloc"
// REGION: remark: Buffers: [224, 32]
// REGION-NEXT: test.region_name = "tpage1_tail"
// CHECK-LABEL: tpage0 vs tpage0
// CHECK: tpage0 vs tpage1: alias=false
// CHECK: tpage0 vs tpage1_pow2_alloc: alias=false
// CHECK: tpage0 vs tpage1_tail: alias=false
// CHECK: tpage1 vs tpage1_pow2_alloc: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: tpage1 vs tpage1_tail: alias=false
// OFF: shape must have power-of-2 and non-zero dimensions; got 2, 128, 96
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 256 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @tmem_pages() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %buf = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<2x128x96xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %full1 = ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %tail1 = ttng.tmem_subslice %full1 {offset = 96 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x32xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    %p0 = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<2x128x96xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    %p1 = ttg.memdesc_index %buf[%c1] : !ttg.memdesc<2x128x96xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    "test.use"(%p0) {test.region_name = "tpage0"} : (!ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>) -> ()
    "test.use"(%p1) {test.region_name = "tpage1"} : (!ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>) -> ()
    %2 = ttng.tmem_load %full1 {test.region_name = "tpage1_pow2_alloc"} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32>
    %3 = ttng.tmem_load %tail1 {test.region_name = "tpage1_tail"} : !ttg.memdesc<128x32xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x32xf32>
    tt.return
  }
}

// -----

// An SMEM pow2 alloc shape is sized through its whole power-of-two allocation.
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: s_pow2_alloc vs s_pow2_alloc
// CHECK: s_pow2_alloc vs sfull: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: s_pow2_alloc case [0, 512]
// OFF: shape must have power-of-2 and non-zero dimensions; got 8, 48
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 1024 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, test.print_state_plan} {
  tt.func public @smem_pow2_alloc() {
    %pow2_alloc = ttg.local_alloc {allocation.offset = 0 : i32, test.region_name = "s_pow2_alloc"} : () -> !ttg.memdesc<8x48xi8, #shared, #smem, mutable, 8x64>
    %full = ttg.local_alloc {allocation.offset = 0 : i32, test.region_name = "sfull"} : () -> !ttg.memdesc<8x64xi8, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Power-of-two control: identical with the flag unset and set.
#sw4 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

// POT-REGION: remark: Buffers: [0, 16384]
// POT-REGION-NEXT: test.region_name = "pot_page0"
// POT-REGION: remark: Buffers: [16384, 16384]
// POT-REGION-NEXT: test.region_name = "pot_page1"
// POT-REGION: remark: Buffers: [16512, 16192]
// POT-REGION-NEXT: test.region_name = "pot_col64"
// POT-REGION: remark: Buffers: [16576, 16192]
// POT-REGION-NEXT: test.region_name = "pot_col96"
// POT-LABEL: pot_col64 vs pot_col64
// POT: pot_col64 vs pot_col96: alias=false
// POT: pot_col64 vs pot_page1: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// POT: pot_col96 vs pot_page1: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// POT: pot_page0 vs pot_page1: alias=false
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 32768 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @pot_pages() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x128xf16, #sw4, #smem, mutable>
    %p0 = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<2x64x128xf16, #sw4, #smem, mutable> -> !ttg.memdesc<64x128xf16, #sw4, #smem, mutable>
    %p1 = ttg.memdesc_index %buf[%c1] : !ttg.memdesc<2x64x128xf16, #sw4, #smem, mutable> -> !ttg.memdesc<64x128xf16, #sw4, #smem, mutable>
    %c64 = ttg.memdesc_subslice %p1 [0, 64] : !ttg.memdesc<64x128xf16, #sw4, #smem, mutable> -> !ttg.memdesc<64x32xf16, #sw4, #smem, mutable, 64x128>
    %c96 = ttg.memdesc_subslice %p1 [0, 96] : !ttg.memdesc<64x128xf16, #sw4, #smem, mutable> -> !ttg.memdesc<64x32xf16, #sw4, #smem, mutable, 64x128>
    %0 = ttg.local_load %p0 {test.region_name = "pot_page0"} : !ttg.memdesc<64x128xf16, #sw4, #smem, mutable> -> tensor<64x128xf16>
    %1 = ttg.local_load %p1 {test.region_name = "pot_page1"} : !ttg.memdesc<64x128xf16, #sw4, #smem, mutable> -> tensor<64x128xf16>
    %2 = ttg.local_load %c64 {test.region_name = "pot_col64"} : !ttg.memdesc<64x32xf16, #sw4, #smem, mutable, 64x128> -> tensor<64x32xf16>
    %3 = ttg.local_load %c96 {test.region_name = "pot_col96"} : !ttg.memdesc<64x32xf16, #sw4, #smem, mutable, 64x128> -> tensor<64x32xf16>
    tt.return
  }
}

// -----

// A multibuffered NPOT NVMMA tile whose 160 columns aren't whole 64-column
// panels: each page is its own 64x256 pow2 alloc shape.
#nv = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

// REGION: remark: Buffers: [0, 32768]
// REGION-NEXT: test.region_name = "npage0"
// REGION: remark: Buffers: [32768, 32768]
// REGION-NEXT: test.region_name = "npage1"
// CHECK-LABEL: npage0 vs npage0
// CHECK: npage0 vs npage1: alias=false
// CHECK: npage0 vs npages: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// OFF: shape must have power-of-2 and non-zero dimensions; got 3, 64, 160
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 98304 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @nvmma_pages() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %buf = ttg.local_alloc {allocation.offset = 0 : i32, test.region_name = "npages"} : () -> !ttg.memdesc<3x64x160xf16, #nv, #smem, mutable>
    %p0 = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<3x64x160xf16, #nv, #smem, mutable> -> !ttg.memdesc<64x160xf16, #nv, #smem, mutable>
    %p1 = ttg.memdesc_index %buf[%c1] : !ttg.memdesc<3x64x160xf16, #nv, #smem, mutable> -> !ttg.memdesc<64x160xf16, #nv, #smem, mutable>
    %0 = ttg.local_load %p0 {test.region_name = "npage0"} : !ttg.memdesc<64x160xf16, #nv, #smem, mutable> -> tensor<64x160xf16>
    %1 = ttg.local_load %p1 {test.region_name = "npage1"} : !ttg.memdesc<64x160xf16, #nv, #smem, mutable> -> tensor<64x160xf16>
    tt.return
  }
}

// -----

// A multibuffered dense NPOT tile: each 128x96 page is allocated as exactly
// 128x96 elements, so its pages are exact and disjoint.
#nv = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

// REGION: remark: Buffers: [0, 24576]
// REGION-NEXT: test.region_name = "dpage0"
// REGION: remark: Buffers: [24576, 24576]
// REGION-NEXT: test.region_name = "dpage1"
// CHECK-LABEL: dpage0 vs dpage0
// CHECK: dpage0 vs dpage1: alias=false
// CHECK: dpage0 vs dpages: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// CHECK: dpage1 vs dpages: alias=true, lhs_contains_rhs=false, rhs_contains_lhs=true
// OFF: shape must have power-of-2 and non-zero dimensions; got 3, 128, 96
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 73728 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dense_pages() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %buf = ttg.local_alloc {allocation.offset = 0 : i32, test.region_name = "dpages"} : () -> !ttg.memdesc<3x128x96xf16, #nv, #smem, mutable>
    %p0 = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<3x128x96xf16, #nv, #smem, mutable> -> !ttg.memdesc<128x96xf16, #nv, #smem, mutable>
    %p1 = ttg.memdesc_index %buf[%c1] : !ttg.memdesc<3x128x96xf16, #nv, #smem, mutable> -> !ttg.memdesc<128x96xf16, #nv, #smem, mutable>
    %0 = ttg.local_load %p0 {test.region_name = "dpage0"} : !ttg.memdesc<128x96xf16, #nv, #smem, mutable> -> tensor<128x96xf16>
    %1 = ttg.local_load %p1 {test.region_name = "dpage1"} : !ttg.memdesc<128x96xf16, #nv, #smem, mutable> -> tensor<128x96xf16>
    tt.return
  }
}
