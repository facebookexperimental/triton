// RUN: triton-opt --split-input-file %s --tlx-storage-alias-lowering --verify-each=false 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Buffer Offset Calculation Pass Tests
//===----------------------------------------------------------------------===//

// Test: Basic shared reuse group with two allocations of different sizes
// shared(f32[2,64,64], f16[2,64,64])
// bytes_between_buffers = max(16384, 8192) = 16384
// For f32: scale = 16384/16384 = 1, offset = 0, shape unchanged
// For f16: scale = 16384/8192 = 2, offset = 0, shape expands 2->3
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: shared_reuse_group_basic
  tt.func @shared_reuse_group_basic() {
    // For shared reuse group: total size = max(16384, 8192) * 2 = 32768 bytes
    // CHECK: memdesc<32768xi8
    // f32 allocation: no expansion needed (scale=1, offset=0)
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf32
    // f16 allocation: expanded from 2 to 3 (scale=2, offset=0)
    // CHECK: local_alias{{.*}}memdesc<3x64x64xf16
    // CHECK-NOT: reuse_group
    // CHECK-NOT: set_buffer_overlap
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    tt.return
  }
}

// -----

// Test: Basic distinct reuse group with two allocations
// distinct(f32[2,64,64], f32[2,64,64])
// bytes_between_buffers = 16384 + 16384 = 32768
// For first: scale = 32768/16384 = 2, offset = 0, shape: 2 -> 3
// For second: scale = 32768/16384 = 2, offset = 16384/16384 = 1, shape: 2 -> 4
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: distinct_reuse_group_basic
  tt.func @distinct_reuse_group_basic() {
    // For distinct reuse group: total size = (16384 + 16384) * 2 = 65536 bytes
    // CHECK: memdesc<65536xi8
    // First allocation: scale=2, offset=0, shape: 2 -> 3
    // CHECK: local_alias{{.*}}memdesc<3x64x64xf32
    // Second allocation: scale=2, offset=1, shape: 2 -> 4
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf32
    // CHECK-NOT: reuse_group
    // CHECK-NOT: set_buffer_overlap
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
}

// -----

// Test: Nested shared(distinct) reuse group
// P: scale = 16384/8192 = 2, offset = 0, shape: 2 -> 3
// alpha: scale = 16384/256 = 64, offset = 8192/256 = 32, shape: 2 -> 97
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: nested_shared_distinct
  tt.func @nested_shared_distinct() {
    // CHECK: memdesc<32768xi8
    // QK: no expansion (scale=1, offset=0)
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf32
    // P: scale=2, offset=0, shape: 2 -> 3
    // CHECK: local_alias{{.*}}memdesc<3x64x64xf16
    // alpha: scale=64, offset=32, shape: 2 -> 97
    // CHECK: local_alias{{.*}}memdesc<97x64xf32
    // CHECK-NOT: set_buffer_overlap
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %inner_distinct = tlx.reuse_group(%2, %3) group_kind = distinct : (!ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    %outer_shared = tlx.reuse_group(%1, %inner_distinct) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !tlx.reuse_group<distinct>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %outer_shared) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    tt.return
  }
}

// -----

// Test: Nested distinct(shared) reuse group
// distinct(A, shared(B, C))
// A at offset 0, scale = 8192/4096 = 2, shape: 2 -> 3
// B at offset 4096, scale = 8192/4096 = 2, offset = 4096/4096 = 1, shape: 2 -> 4
// C shares with B, scale = 8192/2048 = 4, offset = 4096/2048 = 2, shape: 2 -> 7
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: nested_distinct_shared
  tt.func @nested_distinct_shared() {
    // CHECK: memdesc<16384xi8
    // A at offset 0, scale = 8192/4096 = 2, shape: 2 -> 3
    // CHECK: local_alias{{.*}}memdesc<3x32x32xf32
    // B at offset 4096, scale = 8192/4096 = 2, offset = 4096/4096 = 1, shape: 2 -> 4
    // CHECK: local_alias{{.*}}memdesc<4x32x32xf32
    // C shares with B, same offset, scale = 8192/2048 = 4, offset = 4096/2048 = 2, shape: 2 -> 7
    // CHECK: local_alias{{.*}}memdesc<7x32x32xf16
    // CHECK-NOT: set_buffer_overlap
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %3 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x32x32xf16, #shared, #smem, mutable>
    %inner_shared = tlx.reuse_group(%2, %3) group_kind = shared : (!ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>, !ttg.memdesc<2x32x32xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    %outer_distinct = tlx.reuse_group(%1, %inner_shared) group_kind = distinct : (!ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>, !tlx.reuse_group<shared>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %outer_distinct) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
}

// -----

// Test: Index rewriting with scale only (first allocation in distinct)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: index_rewriting_scale_only
  tt.func @index_rewriting_scale_only(%idx: i32) {
    // CHECK: memdesc<65536xi8
    // CHECK: local_alias{{.*}}memdesc<3x64x64xf32
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli
    // CHECK: memdesc_index
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    %4 = ttg.memdesc_index %1[%idx] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: Index rewriting with both scale and offset
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: index_rewriting_scale_and_offset
  tt.func @index_rewriting_scale_and_offset(%idx: i32) {
    // CHECK: memdesc<65536xi8
    // CHECK: local_alias{{.*}}memdesc<3x64x64xf32
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf32
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli
    // CHECK: arith.constant 1 : i32
    // CHECK: arith.addi
    // CHECK: memdesc_index
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    %4 = ttg.memdesc_index %2[%idx] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: No set_buffer_overlap -> no expansion
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: no_set_buffer_overlap
  tt.func @no_set_buffer_overlap() {
    // CHECK: memdesc<32768xi8
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf32
    // CHECK-NOT: arith.muli
    // CHECK-NOT: arith.addi
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: Single allocation in reuse group -> no expansion
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: single_allocation_reuse_group
  tt.func @single_allocation_reuse_group() {
    // CHECK: memdesc<32768xi8
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf32
    // CHECK-NOT: reuse_group
    // CHECK-NOT: set_buffer_overlap
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.reuse_group(%1) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %2) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    tt.return
  }
}

// -----

// Test: Shared reuse group with different sizes but same element type
// Small: scale = 8192/2048 = 4, offset = 0, shape: 2 -> 5
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: shared_different_sizes_same_type
  tt.func @shared_different_sizes_same_type() {
    // CHECK: memdesc<16384xi8
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf16
    // CHECK: local_alias{{.*}}memdesc<5x32x32xf16
    // CHECK-NOT: set_buffer_overlap
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x32x32xf16, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x32x32xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    tt.return
  }
}

// -----

// Test: Index rewriting with constant index (second allocation)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: index_rewriting_constant_index
  tt.func @index_rewriting_constant_index() {
    // CHECK: memdesc<65536xi8
    // CHECK: local_alias{{.*}}memdesc<3x64x64xf32
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf32
    // CHECK: arith.constant 0 : i32
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli
    // CHECK: arith.constant 1 : i32
    // CHECK: arith.addi
    // CHECK: memdesc_index
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    %c0 = arith.constant 0 : i32
    %4 = ttg.memdesc_index %2[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: Index rewriting with dynamic function argument index
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: index_rewriting_dynamic_index
  tt.func @index_rewriting_dynamic_index(%idx: i32) {
    // CHECK: memdesc<65536xi8
    // CHECK: local_alias{{.*}}memdesc<3x64x64xf32
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf32
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli %arg0
    // CHECK: arith.constant 1 : i32
    // CHECK: arith.addi
    // CHECK: memdesc_index
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    %4 = ttg.memdesc_index %2[%idx] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: Index rewriting with computed index (add of two args)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: index_rewriting_computed_index
  tt.func @index_rewriting_computed_index(%a: i32, %b: i32) {
    // CHECK: memdesc<65536xi8
    // CHECK: local_alias{{.*}}memdesc<3x64x64xf32
    // CHECK: arith.addi %arg0, %arg1
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli
    // CHECK: memdesc_index
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    %sum = arith.addi %a, %b : i32
    %4 = ttg.memdesc_index %1[%sum] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: Multiple index uses of the same allocation
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: multiple_index_uses
  tt.func @multiple_index_uses(%idx0: i32, %idx1: i32) {
    // CHECK: memdesc<65536xi8
    // CHECK: local_alias{{.*}}memdesc<3x64x64xf32
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli %arg0
    // CHECK: memdesc_index
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli %arg1
    // CHECK: memdesc_index
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    %4 = ttg.memdesc_index %1[%idx0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %5 = ttg.memdesc_index %1[%idx1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: No index rewriting for the largest allocation (scale=1, offset=0)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: no_index_rewriting_for_largest_alloc
  tt.func @no_index_rewriting_for_largest_alloc(%idx: i32) {
    // CHECK: memdesc<32768xi8
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf32
    // CHECK: memdesc_index %{{.*}}[%arg0]
    // CHECK-NOT: arith.muli %arg0
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    %4 = ttg.memdesc_index %1[%idx] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: Index rewriting for the smaller allocation (scale=2)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: index_rewriting_for_smaller_alloc
  tt.func @index_rewriting_for_smaller_alloc(%idx: i32) {
    // CHECK: memdesc<32768xi8
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf32
    // CHECK: local_alias{{.*}}memdesc<3x64x64xf16
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli
    // CHECK: memdesc_index
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    %4 = ttg.memdesc_index %2[%idx] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: Warp specialize with shared reuse group
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: warp_specialize_shared_reuse_group
  tt.func @warp_specialize_shared_reuse_group(%idx: i32) {
    // CHECK: memdesc<32768xi8
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // f32: no expansion (scale=1, offset=0)
    // CHECK: %[[ALIAS0:.*]] = tlx.local_alias{{.*}}memdesc<2x64x64xf32
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // f16: expanded from 2 to 3 (scale=2, offset=0)
    // CHECK: %[[ALIAS1:.*]] = tlx.local_alias{{.*}}memdesc<3x64x64xf16
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    // CHECK: ttg.warp_specialize(%[[ALIAS0]], %[[ALIAS1]],
    ttg.warp_specialize(%1, %2, %idx)
    default {
      ttg.warp_yield
    }
    // CHECK: partition0(%{{.*}}: !ttg.memdesc<2x64x64xf32, {{.*}}>, %{{.*}}: !ttg.memdesc<3x64x64xf16, {{.*}}>
    partition0(%arg0: !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, %arg1: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg_idx: i32) num_warps(1) {
      // CHECK: arith.constant 2 : i32
      // CHECK: arith.muli
      // CHECK: memdesc_index
      %4 = ttg.memdesc_index %arg1[%arg_idx] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, i32) -> ()
    tt.return
  }
}

// -----

// Test: Warp specialize with distinct reuse group
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: warp_specialize_distinct_reuse_group
  tt.func @warp_specialize_distinct_reuse_group(%idx: i32) {
    // CHECK: memdesc<65536xi8
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // First: scale=2, offset=0, shape: 2->3
    // CHECK: %[[ALIAS0:.*]] = tlx.local_alias{{.*}}memdesc<3x64x64xf32
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // Second: scale=2, offset=1, shape: 2->4
    // CHECK: %[[ALIAS1:.*]] = tlx.local_alias{{.*}}memdesc<4x64x64xf32
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    // CHECK: ttg.warp_specialize(%[[ALIAS0]], %[[ALIAS1]],
    ttg.warp_specialize(%1, %2, %idx)
    default {
      ttg.warp_yield
    }
    // CHECK: partition0(%{{.*}}: !ttg.memdesc<3x64x64xf32, {{.*}}>, %{{.*}}: !ttg.memdesc<4x64x64xf32, {{.*}}>
    partition0(%arg0: !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, %arg1: !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, %arg_idx: i32) num_warps(1) {
      // CHECK: arith.constant 2 : i32
      // CHECK: arith.muli
      // CHECK: memdesc_index
      %4 = ttg.memdesc_index %arg0[%arg_idx] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
      // CHECK: arith.constant 2 : i32
      // CHECK: arith.muli
      // CHECK: arith.constant 1 : i32
      // CHECK: arith.addi
      // CHECK: memdesc_index
      %5 = ttg.memdesc_index %arg1[%arg_idx] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, i32) -> ()
    tt.return
  }
}

// -----

// Test: Shared reuse group with 3 elements
// A: scale=1, offset=0 (no expansion)
// B: scale = 16384/4096 = 4, offset = 0, shape: 2 -> 5
// C: scale = 16384/1024 = 16, offset = 0, shape: 2 -> 17
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: shared_reuse_group_three_elements
  tt.func @shared_reuse_group_three_elements() {
    // CHECK: memdesc<32768xi8
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf32
    // CHECK: local_alias{{.*}}memdesc<5x32x32xf32
    // CHECK: local_alias{{.*}}memdesc<17x16x16xf32
    // CHECK-NOT: reuse_group
    // CHECK-NOT: set_buffer_overlap
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %3 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable>
    %4 = tlx.reuse_group(%1, %2, %3) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>, !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %4) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    tt.return
  }
}

// -----

// Test: Distinct reuse group with 3 elements
// A: scale=3, offset=0, shape: 2 -> 4
// B: scale=3, offset=1, shape: 2 -> 5
// C: scale=3, offset=2, shape: 2 -> 6
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: distinct_reuse_group_three_elements
  tt.func @distinct_reuse_group_three_elements() {
    // CHECK: memdesc<98304xi8
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf32
    // CHECK: local_alias{{.*}}memdesc<5x64x64xf32
    // CHECK: local_alias{{.*}}memdesc<6x64x64xf32
    // CHECK-NOT: reuse_group
    // CHECK-NOT: set_buffer_overlap
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %4 = tlx.reuse_group(%1, %2, %3) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %4) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
}
