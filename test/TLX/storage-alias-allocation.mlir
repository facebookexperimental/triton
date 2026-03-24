// RUN: triton-opt --split-input-file %s --tlx-storage-alias-lowering | FileCheck %s

// Test that allocation pass creates correct size for single f32 buffer
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_single_f32_buffer
  tt.func @alloc_single_f32_buffer() {
    // 2 * 64 * 64 * 4 bytes (f32) = 32768 bytes
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<32768xi8
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test that allocation pass creates correct size for single f16 buffer
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_single_f16_buffer
  tt.func @alloc_single_f16_buffer() {
    // 2 * 64 * 64 * 2 bytes (f16) = 16384 bytes
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<16384xi8
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test that allocation pass creates correct size for single bf16 buffer
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_single_bf16_buffer
  tt.func @alloc_single_bf16_buffer() {
    // 4 * 128 * 32 * 2 bytes (bf16) = 32768 bytes
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<32768xi8
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<4x128x32xbf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test that allocation pass creates correct size for single i8 buffer
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_single_i8_buffer
  tt.func @alloc_single_i8_buffer() {
    // 8 * 16 * 16 * 1 byte (i8) = 2048 bytes
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2048xi8
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<8x16x16xi8, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test that allocation pass creates correct size for pointer type (8 bytes per pointer)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_pointer_buffer
  tt.func @alloc_pointer_buffer() {
    // 2 * 8 * 8 * 8 bytes (pointer) = 1024 bytes
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<1024xi8
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x8x8x!tt.ptr<f32>, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test that allocation pass picks max size when multiple allocations reference same spec
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_multiple_users_picks_max
  tt.func @alloc_multiple_users_picks_max() {
    // First alloc: 2 * 64 * 64 * 4 bytes (f32) = 32768 bytes
    // Second alloc: 2 * 64 * 64 * 2 bytes (bf16) = 16384 bytes
    // Max = 32768 bytes
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<32768xi8
    // CHECK: tlx.local_alias
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test that allocation pass handles multiple storage_alias_specs independently
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_independent_specs
  tt.func @alloc_independent_specs() {
    // First spec: 2 * 64 * 64 * 4 bytes (f32) = 32768 bytes
    // Second spec: 4 * 32 * 32 * 2 bytes (f16) = 8192 bytes
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<32768xi8
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<8192xi8
    // CHECK: tlx.local_alias
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.storage_alias_local_alloc %1 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<4x32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test that allocation pass respects explicit size when it's larger than needed
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_explicit_size_larger
  tt.func @alloc_explicit_size_larger() {
    // Explicit size 65536, required = 2 * 64 * 64 * 4 = 32768
    // Should use explicit size 65536
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<65536xi8
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = smem, size = 65536 : !tlx.storage_alias_spec<smem, 65536>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem, 65536> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test f8E5M2 (fp8) type allocation
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_fp8_buffer
  tt.func @alloc_fp8_buffer() {
    // 4 * 128 * 64 * 1 byte (f8E5M2) = 32768 bytes
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<32768xi8
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<4x128x64xf8E5M2, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test i32 type allocation
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_i32_buffer
  tt.func @alloc_i32_buffer() {
    // 2 * 32 * 32 * 4 bytes (i32) = 8192 bytes
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<8192xi8
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x32x32xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test i64 type allocation
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_i64_buffer
  tt.func @alloc_i64_buffer() {
    // 2 * 16 * 16 * 8 bytes (i64) = 4096 bytes
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<4096xi8
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x16x16xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test f64 type allocation
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_f64_buffer
  tt.func @alloc_f64_buffer() {
    // 1 * 32 * 32 * 8 bytes (f64) = 8192 bytes
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<8192xi8
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<1x32x32xf64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test TMEM allocation creates TMEMAllocOp with tensor_memory_encoding
// TMEM uses max blockM and blockN from user allocations (2D layout assumption),
// with blockN scaled down for smaller element types (divided by 4/elementBytes).
#tmem_enc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 2>
#tmem = #ttng.tensor_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_tmem_buffer
  tt.func @alloc_tmem_buffer() {
    // 128 * 64 * 2 bytes (f16) = 16384 bytes
    // blockN scaled: 64 / (4/2) = 64 / 2 = 32
    // CHECK: ttng.tmem_alloc : () -> !ttg.memdesc<128x32xi32
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = tmem : !tlx.storage_alias_spec<tmem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<128x64xf16, #tmem_enc, #tmem, mutable>
    tt.return
  }
}

// -----

// Test TMEM allocation respects explicit size when it's larger than needed
// The blockN should be padded to accommodate the larger explicit size
#tmem_enc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 2>
#tmem = #ttng.tensor_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @alloc_tmem_explicit_size_larger
  tt.func @alloc_tmem_explicit_size_larger() {
    // Explicit size 65536, required = 128 * 64 * 4 = 32768 bytes
    // requiredBlockN = 65536 / (128 * 4) = 128
    // Should pad blockN to 128 to accommodate explicit size
    // CHECK: ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32
    // CHECK: tlx.local_alias
    %0 = tlx.storage_alias_spec storage = tmem, size = 65536 : !tlx.storage_alias_spec<tmem, 65536>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem, 65536> -> !ttg.memdesc<128x64xf16, #tmem_enc, #tmem, mutable>
    tt.return
  }
}
