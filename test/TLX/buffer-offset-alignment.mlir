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
// Offsets (using new formula: newBufferDim = scale * lastIdx + offset + 1):
//   A: offset=0,   bytesBetweenBuffers=384 → scale=48, offSlots=0  → [48*3+0+1, 2] = [145, 2]
//   B: offset=0,   bytesBetweenBuffers=384 → scale=48, offSlots=0  → [48*3+0+1, 2] = [145, 2]
//   C: offset=128, bytesBetweenBuffers=384 → scale=192, offSlots=64 → [192*0+64+1, 1] = [65, 1]
//   D: offset=256, bytesBetweenBuffers=384 → scale=48, offSlots=32 → [48*3+32+1, 2] = [177, 2]
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @smem_distinct_shared_distinct_alignment
  tt.func @smem_distinct_shared_distinct_alignment() {
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<1536xi8
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<145x2xf32
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<145x2xf32
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<65x1xbf16
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<177x2xf32
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

// Test TMEM alignment (column-based) with nested reuse group tree:
//   distinct(shared(A, distinct(B, C)), D)
// where A, B, D are f32 [4,32,8] and C is bf16 [1,32,4]
//
// Per-buffer TMEM columns (DummyTMEMLayout: ceil(m/32)*ceil(k/4)):
//   A = ceil(32/32)*ceil(8/4) = 2, B = 2, C = ceil(32/32)*ceil(4/4) = 1, D = 2
//
// Alignment (useTmemColumns): max of all leaf column counts = 2
//
// getElementSize (useTmemColumns=true):
//   distinct(B, C):    alignUp(0,2) + 2 = 2;  alignUp(2,1) + 1 = 3
//   shared(A, distinct(B,C)):  max(2, 3) = 3
//   distinct(shared(..), D):   alignUp(0,2) + 3 = 3;  alignUp(3,2) + 2 = 6
//
// columnsPerBufferGroup = 6, columnsBetweenBufferGroups = alignUp(6, 2) = 6
//
// Offsets (using formula: newBufferDim = scale * lastIdx + offset + 1):
//   A: offset=0, colsBetween=6 → scale=6/2=3, offSlots=0  → [3*3+0+1, 32, 8] = [10, 32, 8]
//   B: offset=0, colsBetween=6 → scale=6/2=3, offSlots=0  → [3*3+0+1, 32, 8] = [10, 32, 8]
//   C: offset=2, colsBetween=6 → scale=6/1=6, offSlots=2  → [6*0+2+1, 32, 4] = [3, 32, 4]
//   D: offset=4, colsBetween=6 → scale=6/2=3, offSlots=2  → [3*3+2+1, 32, 8] = [12, 32, 8]
#dummy_tmem_layout = #tlx.dummy_tmem_layout<>
#tmem = #ttng.tensor_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_distinct_shared_distinct_alignment
  tt.func @tmem_distinct_shared_distinct_alignment() {
    // CHECK: ttng.tmem_alloc
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<10x32x8xf32
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<10x32x8xf32
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<3x32x4xbf16
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<12x32x8xf32
    %0 = tlx.storage_alias_spec storage = tmem : !tlx.storage_alias_spec<tmem>
    %A = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<4x32x8xf32, #dummy_tmem_layout, #tmem, mutable>
    %B = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<4x32x8xf32, #dummy_tmem_layout, #tmem, mutable>
    %C = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<1x32x4xbf16, #dummy_tmem_layout, #tmem, mutable>
    %D = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<4x32x8xf32, #dummy_tmem_layout, #tmem, mutable>
    %inner_distinct = tlx.reuse_group(%B, %C) group_kind = distinct : (!ttg.memdesc<4x32x8xf32, #dummy_tmem_layout, #tmem, mutable>, !ttg.memdesc<1x32x4xbf16, #dummy_tmem_layout, #tmem, mutable>) -> !tlx.reuse_group<distinct>
    %inner_shared = tlx.reuse_group(%A, %inner_distinct) group_kind = shared : (!ttg.memdesc<4x32x8xf32, #dummy_tmem_layout, #tmem, mutable>, !tlx.reuse_group<distinct>) -> !tlx.reuse_group<shared>
    %outer_distinct = tlx.reuse_group(%inner_shared, %D) group_kind = distinct : (!tlx.reuse_group<shared>, !ttg.memdesc<4x32x8xf32, #dummy_tmem_layout, #tmem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %outer_distinct) : (!tlx.storage_alias_spec<tmem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
}

// -----

// Test TMEM distinct reuse between f32 and i8 buffers (different
// bytes-per-column ratios). This is the key case where column-based reuse
// differs from byte-based reuse.
//   distinct(A, B) where A is f32 [4,32,8] and B is i8 [4,32,4]
//
// Per-buffer TMEM columns (DummyTMEMLayout: ceil(m/32)*ceil(k/4)):
//   A = ceil(32/32)*ceil(8/4) = 2, B = ceil(32/32)*ceil(4/4) = 1
//
// Alignment (useTmemColumns): max(2, 1) = 2
//
// getElementSize (useTmemColumns=true):
//   distinct(A, B):  alignUp(0,2) + 2 = 2;  alignUp(2,1) + 1 = 3
//
// columnsPerBufferGroup = 3, columnsBetweenBufferGroups = alignUp(3, 2) = 4
//
// Offsets (using formula: newBufferDim = scale * lastIdx + offset + 1):
//   A: offset=0, colsBetween=4 → scale=4/2=2, offSlots=0  → [2*3+0+1, 32, 8] = [7, 32, 8]
//   B: offset=2, colsBetween=4 → scale=4/1=4, offSlots=2  → [4*3+2+1, 32, 4] = [15, 32, 4]
#dummy_tmem_layout = #tlx.dummy_tmem_layout<>
#tmem = #ttng.tensor_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_distinct_f32_i8
  tt.func @tmem_distinct_f32_i8() {
    // CHECK: ttng.tmem_alloc
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<7x32x8xf32
    // CHECK: tlx.local_alias {{.*}} -> !ttg.memdesc<15x32x4xi8
    %0 = tlx.storage_alias_spec storage = tmem : !tlx.storage_alias_spec<tmem>
    %A = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<4x32x8xf32, #dummy_tmem_layout, #tmem, mutable>
    %B = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<4x32x4xi8, #dummy_tmem_layout, #tmem, mutable>
    %distinct = tlx.reuse_group(%A, %B) group_kind = distinct : (!ttg.memdesc<4x32x8xf32, #dummy_tmem_layout, #tmem, mutable>, !ttg.memdesc<4x32x4xi8, #dummy_tmem_layout, #tmem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %distinct) : (!tlx.storage_alias_spec<tmem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
}
