// RUN: triton-opt --split-input-file %s --tlx-storage-alias-lowering --verify-each=false 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Buffer Offset Calculation Pass Tests
//===----------------------------------------------------------------------===//

// Test: Basic shared reuse group - both allocations get same shape expansion
// Two allocations: 2x64x64xf32 (16384 bytes per buffer) and 2x64x64xf16 (8192 bytes)
// max size = 16384 bytes * 2 buffers = 32768 bytes total
// For f32: bytes_between_buffers = 16384, buffer_size = 16384, scale = 1, no expansion needed
// For f16: bytes_between_buffers = 16384, buffer_size = 8192, scale = 2, shape expands 2->4
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: shared_reuse_group_basic
  tt.func @shared_reuse_group_basic() {
    // For shared reuse group: total size = max(16384, 8192) * 2 = 32768 bytes
    // CHECK: memdesc<32768xi8
    // f32 allocation: no expansion needed (scale=1, offset=0)
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf32
    // f16 allocation: expanded from 2 to 4 buffers (scale=2, offset=0)
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf16
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

// Test: Basic distinct reuse group - allocations placed sequentially with shape expansion
// Two allocations: 2x64x64xf32 (16384 bytes per buffer) each
// total = (16384 + 16384) * 2 = 65536 bytes
// bytes_between_buffers = 32768 for both
// For first: scale = 32768/16384 = 2, offset = 0, shape: 2 -> 4
// For second: scale = 32768/16384 = 2, offset = 16384/16384 = 1, shape: 2 -> 2*2+1 = 5
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: distinct_reuse_group_basic
  tt.func @distinct_reuse_group_basic() {
    // For distinct reuse group: total size = (16384 + 16384) * 2 = 65536 bytes
    // CHECK: memdesc<65536xi8
    // First allocation: scale=2, offset=0, shape: 2 -> 4
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf32
    // Second allocation: scale=2, offset=1, shape: 2 -> 5
    // CHECK: local_alias{{.*}}memdesc<5x64x64xf32
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

// Test: Nested reuse groups - shared containing distinct
// shared(QK, distinct(P, alpha))
// QK: 2x64x64xf32 = 16384 bytes per buffer
// P: 2x64x64xf16 = 8192 bytes per buffer
// alpha: 2x64xf32 = 256 bytes per buffer
// distinct(P, alpha) size = 8192 + 256 = 8448 bytes
// shared max = max(16384, 8448) = 16384 bytes * 2 = 32768 bytes total
// bytes_between_buffers = 16384 for all
// QK: scale = 16384/16384 = 1, offset = 0, no expansion
// P: scale = 16384/8192 = 2, offset = 0, shape: 2 -> 4
// alpha: scale = 16384/256 = 64, offset = 8192/256 = 32, shape: 2 -> 2*64+32 = 160
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: nested_shared_distinct
  tt.func @nested_shared_distinct() {
    // CHECK: memdesc<32768xi8
    // QK: no expansion (scale=1, offset=0)
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf32
    // P: scale=2, offset=0, shape: 2 -> 4
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf16
    // alpha: scale=64, offset=32, shape: 2 -> 160
    // CHECK: local_alias{{.*}}memdesc<160x64xf32
    // CHECK-NOT: set_buffer_overlap
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %4 = tlx.reuse_group(%2, %3) group_kind = distinct : (!ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    %5 = tlx.reuse_group(%1, %4) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !tlx.reuse_group<distinct>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %5) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    tt.return
  }
}

// -----

// Test: Distinct containing shared
// distinct(A, shared(B, C))
// A: 2x64x64xf32 = 16384 bytes per buffer
// B: 2x64x64xf16 = 8192 bytes per buffer
// C: 2x32x32xf32 = 4096 bytes per buffer
// shared(B, C) size = max(8192, 4096) = 8192 bytes
// distinct total = 16384 + 8192 = 24576 bytes * 2 = 49152 bytes
// bytes_between_buffers = 24576 for all
// A: scale = 24576/16384 = 1.5 -> NOT EVENLY DIVISIBLE - should use shape = 2 * 24576/16384
// Actually let's recalculate: 24576 is not evenly divisible by 16384
// We need sizes that are evenly divisible. Let me use different shapes.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: nested_distinct_shared
  tt.func @nested_distinct_shared() {
    // A: 2x32x32xf32 = 4096 bytes per buffer
    // B: 2x32x32xf16 = 2048 bytes per buffer
    // C: 2x16x16xf32 = 1024 bytes per buffer
    // shared(B,C) = max(2048, 1024) = 2048
    // distinct = 4096 + 2048 = 6144 bytes per buffer * 2 = 12288 total
    // bytes_between_buffers = 6144
    // A: scale = 6144/4096 = 1.5 -> still not evenly divisible
    // Let's use sizes that work: A=4096, B=2048, C=2048
    // distinct = 4096 + 2048 = 6144, but 6144/4096 = 1.5 still
    // Need: A=2048, B=1024, C=1024 -> distinct = 2048+1024 = 3072
    // 3072/2048 = 1.5 still...
    // Let's try: A=4096, B=4096 -> distinct = 8192, 8192/4096 = 2 ✓
    // CHECK: memdesc<16384xi8
    // A at offset 0, scale = 8192/4096 = 2, shape: 2 -> 4
    // CHECK: local_alias{{.*}}memdesc<4x32x32xf32
    // B at offset 4096, scale = 8192/4096 = 2, offset = 4096/4096 = 1, shape: 2 -> 5
    // CHECK: local_alias{{.*}}memdesc<5x32x32xf32
    // C shares with B, same offset, scale = 8192/2048 = 4, offset = 4096/2048 = 2, shape: 2 -> 10
    // CHECK: local_alias{{.*}}memdesc<10x32x32xf16
    // CHECK-NOT: set_buffer_overlap
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %3 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x32x32xf16, #shared, #smem, mutable>
    %4 = tlx.reuse_group(%2, %3) group_kind = shared : (!ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>, !ttg.memdesc<2x32x32xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    %5 = tlx.reuse_group(%1, %4) group_kind = distinct : (!ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>, !tlx.reuse_group<shared>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %5) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
}

// -----

// Test: Index rewriting for MemDescIndexOp
// When scale > 1 or offset > 0, MemDescIndexOp indices must be rewritten
// This test verifies the arithmetic operations are generated
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: index_rewriting_scale_only
  tt.func @index_rewriting_scale_only(%idx: i32) {
    // Two f32 allocations in distinct: each 16384 bytes per buffer
    // bytes_between_buffers = 32768, scale = 2
    // First: offset = 0, scale = 2 -> index rewritten: new_idx = 2 * idx
    // CHECK: memdesc<65536xi8
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf32
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli
    // CHECK: memdesc_index
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    // Use the first allocation with memdesc_index
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
    // Two f32 allocations in distinct: each 16384 bytes per buffer
    // bytes_between_buffers = 32768, scale = 2
    // Second allocation: offset = 16384, scale = 2
    // offset_slots = 16384/16384 = 1
    // new_idx = 2 * idx + 1
    // CHECK: memdesc<65536xi8
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf32
    // CHECK: local_alias{{.*}}memdesc<5x64x64xf32
    // For second allocation's index:
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
    // Use the second allocation with memdesc_index (it has scale=2 and offset=1)
    %4 = ttg.memdesc_index %2[%idx] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: No set_buffer_overlap - allocations should NOT have expanded shapes
// 2x64x64xf32 = 32768 bytes total
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

// Test: Single allocation in reuse group (degenerate case) - no expansion needed
// shared(A) - just A with scale=1, offset=0
// A: 2x64x64xf32 = 16384 bytes per buffer * 2 = 32768 bytes
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: single_allocation_reuse_group
  tt.func @single_allocation_reuse_group() {
    // CHECK: memdesc<32768xi8
    // No expansion needed since scale=1 and offset=0
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

// Test: Shape expansion with same element type different sizes
// Two f16 allocations: 2x64x64xf16 (8192 bytes) and 2x32x32xf16 (2048 bytes)
// shared group: max = 8192 bytes per buffer
// First (8192): scale = 1, no expansion
// Second (2048): scale = 8192/2048 = 4, shape: 2 -> 8
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: shared_different_sizes_same_type
  tt.func @shared_different_sizes_same_type() {
    // CHECK: memdesc<16384xi8
    // Large allocation: no expansion
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf16
    // Small allocation: scale=4, shape: 2 -> 8
    // CHECK: local_alias{{.*}}memdesc<8x32x32xf16
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

// Test: Index rewriting with constant index value
// The constant index should be transformed: new_idx = 2 * 0 + 1 = 1
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: index_rewriting_constant_index
  tt.func @index_rewriting_constant_index() {
    // Two f32 allocations in distinct: each 16384 bytes per buffer
    // bytes_between_buffers = 32768, scale = 2
    // Second allocation: offset = 16384, offset_slots = 1
    // For constant index 0: new_idx = 2 * 0 + 1 = 1
    // CHECK: memdesc<65536xi8
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf32
    // CHECK: local_alias{{.*}}memdesc<5x64x64xf32
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
    // Use constant index 0
    %c0 = arith.constant 0 : i32
    %4 = ttg.memdesc_index %2[%c0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: Index rewriting with non-constant (dynamic) index
// The dynamic index should be transformed: new_idx = 2 * idx + 1
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: index_rewriting_dynamic_index
  tt.func @index_rewriting_dynamic_index(%idx: i32) {
    // Two f32 allocations in distinct: each 16384 bytes per buffer
    // bytes_between_buffers = 32768, scale = 2
    // Second allocation: offset = 16384, offset_slots = 1
    // For dynamic index %idx: new_idx = 2 * %idx + 1
    // CHECK: memdesc<65536xi8
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf32
    // CHECK: local_alias{{.*}}memdesc<5x64x64xf32
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
    // Use dynamic function argument index
    %4 = ttg.memdesc_index %2[%idx] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: Index rewriting with computed (arithmetic result) index
// Tests that index rewriting works with values computed from arithmetic
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: index_rewriting_computed_index
  tt.func @index_rewriting_computed_index(%a: i32, %b: i32) {
    // Two f32 allocations in distinct: each 16384 bytes per buffer
    // bytes_between_buffers = 32768, scale = 2
    // First allocation: offset = 0, scale = 2
    // For computed index %a + %b: new_idx = 2 * (%a + %b)
    // CHECK: memdesc<65536xi8
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf32
    // CHECK: arith.addi %arg0, %arg1
    // Then the index rewriting:
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli
    // CHECK: memdesc_index
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    // Use computed index from arithmetic
    %computed = arith.addi %a, %b : i32
    %4 = ttg.memdesc_index %1[%computed] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: Multiple MemDescIndexOp uses of the same alias with different indices
// Each index should be independently rewritten
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: multiple_index_uses
  tt.func @multiple_index_uses(%idx0: i32, %idx1: i32) {
    // Two f32 allocations in distinct: each 16384 bytes per buffer
    // bytes_between_buffers = 32768, scale = 2
    // First allocation: offset = 0, scale = 2
    // Both indices should be scaled by 2
    // CHECK: memdesc<65536xi8
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf32
    // First index rewriting:
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli %arg0
    // CHECK: memdesc_index
    // Second index rewriting:
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli %arg1
    // CHECK: memdesc_index
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    // Multiple uses with different indices
    %4 = ttg.memdesc_index %1[%idx0] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %5 = ttg.memdesc_index %1[%idx1] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: No index rewriting when scale=1 and offset=0 (no expansion needed)
// When the largest allocation is used, no transformation is needed
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: no_index_rewriting_for_largest_alloc
  tt.func @no_index_rewriting_for_largest_alloc(%idx: i32) {
    // f32 (16384 bytes) and f16 (8192 bytes) in shared
    // bytes_between_buffers = 16384 (max size)
    // f32 allocation: scale = 1, offset = 0 -> no rewriting needed
    // CHECK: memdesc<32768xi8
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf32
    // No multiplication or addition for the f32 allocation's index
    // CHECK: memdesc_index %{{.*}}[%arg0]
    // CHECK-NOT: arith.muli %arg0
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    // Use the largest allocation (f32) - should NOT have index rewriting
    %4 = ttg.memdesc_index %1[%idx] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: Index rewriting for smaller allocation in shared group
// The smaller allocation should have its index scaled
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: index_rewriting_for_smaller_alloc
  tt.func @index_rewriting_for_smaller_alloc(%idx: i32) {
    // f32 (16384 bytes) and f16 (8192 bytes) in shared
    // bytes_between_buffers = 16384 (max size)
    // f16 allocation: scale = 16384/8192 = 2, offset = 0 -> multiply by 2
    // CHECK: memdesc<32768xi8
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf32
    // CHECK: local_alias{{.*}}memdesc<4x64x64xf16
    // For f16 allocation:
    // CHECK: arith.constant 2 : i32
    // CHECK: arith.muli
    // CHECK: memdesc_index
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    // Use the smaller allocation (f16) - should have index rewriting
    %4 = ttg.memdesc_index %2[%idx] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test: Warp specialization with shared reuse group - partition args must have expanded types
// Two allocations in shared group: f32 (16384 bytes) and f16 (8192 bytes)
// f16 gets expanded from 2 to 4 buffers (scale=2)
// The warp_specialize captures both allocations; partition args must reflect the new types
// Additionally, memdesc_index inside the partition region must have index rewriting
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
    // f16: expanded from 2 to 4 (scale=2, offset=0)
    // CHECK: %[[ALIAS1:.*]] = tlx.local_alias{{.*}}memdesc<4x64x64xf16
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    // Captures should have the expanded types
    // CHECK: ttg.warp_specialize(%[[ALIAS0]], %[[ALIAS1]],
    ttg.warp_specialize(%1, %2, %idx)
    default {
      ttg.warp_yield
    }
    // Partition args must reflect the expanded types
    // CHECK: partition0(%{{.*}}: !ttg.memdesc<2x64x64xf32, {{.*}}>, %{{.*}}: !ttg.memdesc<4x64x64xf16, {{.*}}>
    partition0(%arg0: !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, %arg1: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg_idx: i32) num_warps(1) {
      // Index rewriting for the f16 block arg (scale=2)
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

// Test: Warp specialization with distinct reuse group - partition args must have expanded types
// Two f32 allocations in distinct group: each 16384 bytes per buffer
// First: scale=2, offset=0, shape: 2->4
// Second: scale=2, offset=1, shape: 2->5
// Both are captured by warp_specialize; partition args must reflect expanded types
// Additionally, memdesc_index inside the partition must have index rewriting
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: warp_specialize_distinct_reuse_group
  tt.func @warp_specialize_distinct_reuse_group(%idx: i32) {
    // CHECK: memdesc<65536xi8
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // First: scale=2, offset=0, shape: 2->4
    // CHECK: %[[ALIAS0:.*]] = tlx.local_alias{{.*}}memdesc<4x64x64xf32
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // Second: scale=2, offset=1, shape: 2->5
    // CHECK: %[[ALIAS1:.*]] = tlx.local_alias{{.*}}memdesc<5x64x64xf32
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    // Captures should have the expanded types
    // CHECK: ttg.warp_specialize(%[[ALIAS0]], %[[ALIAS1]],
    ttg.warp_specialize(%1, %2, %idx)
    default {
      ttg.warp_yield
    }
    // Partition args must reflect the expanded types
    // CHECK: partition0(%{{.*}}: !ttg.memdesc<4x64x64xf32, {{.*}}>, %{{.*}}: !ttg.memdesc<5x64x64xf32, {{.*}}>
    partition0(%arg0: !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, %arg1: !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, %arg_idx: i32) num_warps(1) {
      // Index rewriting for first block arg (scale=2, offset=0)
      // CHECK: arith.constant 2 : i32
      // CHECK: arith.muli
      // CHECK: memdesc_index
      %4 = ttg.memdesc_index %arg0[%arg_idx] : !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
      // Index rewriting for second block arg (scale=2, offset=1)
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
// shared(A, B, C) where:
// A: 2x64x64xf32 = 16384 bytes per buffer
// B: 2x32x32xf32 = 4096 bytes per buffer
// C: 2x16x16xf32 = 1024 bytes per buffer
// shared max = max(16384, 4096, 1024) = 16384
// bytes_between_buffers = alignUp(16384, 128) = 16384
// A: scale = 16384/16384 = 1, offset = 0, no expansion
// B: scale = 16384/4096 = 4, offset = 0, shape: 2 -> 8
// C: scale = 16384/1024 = 16, offset = 0, shape: 2 -> 32
// Total backing alloc = 16384 * 2 = 32768
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: shared_reuse_group_three_elements
  tt.func @shared_reuse_group_three_elements() {
    // CHECK: memdesc<32768xi8
    // A: no expansion (scale=1, offset=0)
    // CHECK: local_alias{{.*}}memdesc<2x64x64xf32
    // B: scale=4, offset=0, shape: 2 -> 8
    // CHECK: local_alias{{.*}}memdesc<8x32x32xf32
    // C: scale=16, offset=0, shape: 2 -> 32
    // CHECK: local_alias{{.*}}memdesc<32x16x16xf32
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
// distinct(A, B, C) where:
// A: 2x64x64xf32 = 16384 bytes per buffer
// B: 2x64x64xf32 = 16384 bytes per buffer
// C: 2x64x64xf32 = 16384 bytes per buffer
// alignment = 128
// distinct total = 16384 + 16384 + 16384 = 49152 (all already 128-aligned)
// bytes_between_buffers = alignUp(49152, 128) = 49152
// A: scale = 49152/16384 = 3, offset_slots = 0/16384 = 0, shape: 2 -> 6
// B: scale = 49152/16384 = 3, offset_slots = 16384/16384 = 1, shape: 2 -> 2*3+1 = 7
// C: scale = 49152/16384 = 3, offset_slots = 32768/16384 = 2, shape: 2 -> 2*3+2 = 8
// Total backing alloc = 49152 * 2 = 98304
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: distinct_reuse_group_three_elements
  tt.func @distinct_reuse_group_three_elements() {
    // CHECK: memdesc<98304xi8
    // A: scale=3, offset=0, shape: 2 -> 6
    // CHECK: local_alias{{.*}}memdesc<6x64x64xf32
    // B: scale=3, offset=1, shape: 2 -> 7
    // CHECK: local_alias{{.*}}memdesc<7x64x64xf32
    // C: scale=3, offset=2, shape: 2 -> 8
    // CHECK: local_alias{{.*}}memdesc<8x64x64xf32
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
