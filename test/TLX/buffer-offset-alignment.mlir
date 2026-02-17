// RUN: triton-opt --split-input-file %s --tlx-storage-alias-lowering | FileCheck %s

// Test SMEM alignment (128-byte) with nested reuse group tree:
//   distinct(shared(A, distinct(B, C)), D)
// where A, B, D are f32 [4,2] and C is bf16 [1,1]
//
// Per-buffer sizes:
//   A = 2*4 = 8 bytes, B = 2*4 = 8 bytes, C = 1*2 = 2 bytes, D = 2*4 = 8 bytes
//
// Alignment = max(128, max_elem_bytes) = 128 for all (SMEM)
//
// getElementSize (alignment=128):
//   distinct(B, C):    alignUp(0,128) + 8 = 8;  alignUp(8,128) + 2 = 130
//   shared(A, distinct(B,C)):  max(8, 130) = 130
//   distinct(shared(..), D):   alignUp(0,128) + 130 = 130;  alignUp(130,128) + 8 = 264
//
// sizePerBuffer = 264, bytesBetweenBuffers = alignUp(264, 128) = 384
// totalSizeBytes = 384 * 4 = 1536
//
// Offsets:
//   A: offset=0,   bytesBetweenBuffers=384 → scale=48, offSlots=0  → [192, 2]
//   B: offset=0,   bytesBetweenBuffers=384 → scale=48, offSlots=0  → [192, 2]
//   C: offset=128, bytesBetweenBuffers=384 → scale=192, offSlots=64 → [256, 1]
//   D: offset=256, bytesBetweenBuffers=384 → scale=48, offSlots=32 → [224, 2]
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @smem_distinct_shared_distinct_alignment
  tt.func @smem_distinct_shared_distinct_alignment() {
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<1536xi8
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<192x2xf32
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<192x2xf32
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<256x1xbf16
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<224x2xf32
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %A = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<4x2xf32, #shared, #smem, mutable>
    %B = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<4x2xf32, #shared, #smem, mutable>
    %C = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<1x1xbf16, #shared, #smem, mutable>
    %D = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<4x2xf32, #shared, #smem, mutable>
    %inner_distinct = tlx.reuse_group(%B, %C) group_kind = distinct : (!ttg.memdesc<4x2xf32, #shared, #smem, mutable>, !ttg.memdesc<1x1xbf16, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    %inner_shared = tlx.reuse_group(%A, %inner_distinct) group_kind = shared : (!ttg.memdesc<4x2xf32, #shared, #smem, mutable>, !tlx.reuse_group<distinct>) -> !tlx.reuse_group<shared>
    %outer_distinct = tlx.reuse_group(%inner_shared, %D) group_kind = distinct : (!tlx.reuse_group<shared>, !ttg.memdesc<4x2xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %outer_distinct) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
}

// -----

// Test TMEM alignment (32-byte) with nested reuse group tree:
//   distinct(shared(A, distinct(B, C)), D)
// where A, B, D are f32 [4,2] and C is bf16 [1,1]
//
// Per-buffer sizes:
//   A = 2*4 = 8 bytes, B = 2*4 = 8 bytes, C = 1*2 = 2 bytes, D = 2*4 = 8 bytes
//
// Alignment = max(32, max_elem_bytes) = 32 for all (TMEM)
//
// getElementSize (alignment=32):
//   distinct(B, C):    alignUp(0,32) + 8 = 8;  alignUp(8,32) + 2 = 34
//   shared(A, distinct(B,C)):  max(8, 34) = 34
//   distinct(shared(..), D):   alignUp(0,32) + 34 = 34;  alignUp(34,32) + 8 = 72
//
// sizePerBuffer = 72, bytesBetweenBuffers = alignUp(72, 32) = 96
// totalSizeBytes = 96 * 4 = 384
//
// Offsets:
//   A: offset=0,  bytesBetweenBuffers=96 → scale=12, offSlots=0  → [48, 2]
//   B: offset=0,  bytesBetweenBuffers=96 → scale=12, offSlots=0  → [48, 2]
//   C: offset=32, bytesBetweenBuffers=96 → scale=48, offSlots=16 → [64, 1]
//   D: offset=64, bytesBetweenBuffers=96 → scale=12, offSlots=8  → [56, 2]
#dummy_tmem_layout = #tlx.dummy_tmem_layout<>
#tmem = #ttng.tensor_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_distinct_shared_distinct_alignment
  tt.func @tmem_distinct_shared_distinct_alignment() {
    // CHECK: ttng.tmem_alloc
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<48x2xf32
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<48x2xf32
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<64x1xbf16
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<56x2xf32
    %0 = tlx.storage_alias_spec storage = tmem : !tlx.storage_alias_spec<tmem>
    %A = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<4x2xf32, #dummy_tmem_layout, #tmem, mutable>
    %B = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<4x2xf32, #dummy_tmem_layout, #tmem, mutable>
    %C = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<1x1xbf16, #dummy_tmem_layout, #tmem, mutable>
    %D = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<4x2xf32, #dummy_tmem_layout, #tmem, mutable>
    %inner_distinct = tlx.reuse_group(%B, %C) group_kind = distinct : (!ttg.memdesc<4x2xf32, #dummy_tmem_layout, #tmem, mutable>, !ttg.memdesc<1x1xbf16, #dummy_tmem_layout, #tmem, mutable>) -> !tlx.reuse_group<distinct>
    %inner_shared = tlx.reuse_group(%A, %inner_distinct) group_kind = shared : (!ttg.memdesc<4x2xf32, #dummy_tmem_layout, #tmem, mutable>, !tlx.reuse_group<distinct>) -> !tlx.reuse_group<shared>
    %outer_distinct = tlx.reuse_group(%inner_shared, %D) group_kind = distinct : (!tlx.reuse_group<shared>, !ttg.memdesc<4x2xf32, #dummy_tmem_layout, #tmem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %outer_distinct) : (!tlx.storage_alias_spec<tmem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
}
