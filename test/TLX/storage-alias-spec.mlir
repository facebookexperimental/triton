// RUN: triton-opt --split-input-file %s | FileCheck %s
// RUN: triton-opt --split-input-file %s --verify-diagnostics

// Test basic storage_alias_spec with smem storage (unsized)
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @storage_alias_spec_smem_unsized
  tt.func @storage_alias_spec_smem_unsized() {
    // CHECK: %{{.*}} = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    tt.return
  }
}

// -----

// Test storage_alias_spec with tmem storage (unsized)
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @storage_alias_spec_tmem_unsized
  tt.func @storage_alias_spec_tmem_unsized() {
    // CHECK: %{{.*}} = tlx.storage_alias_spec storage = tmem : !tlx.storage_alias_spec<tmem>
    %0 = tlx.storage_alias_spec storage = tmem : !tlx.storage_alias_spec<tmem>
    tt.return
  }
}

// -----

// Test storage_alias_spec with smem storage and explicit size
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @storage_alias_spec_smem_sized
  tt.func @storage_alias_spec_smem_sized() {
    // CHECK: %{{.*}} = tlx.storage_alias_spec storage = smem, size = 16384 : !tlx.storage_alias_spec<smem, 16384>
    %0 = tlx.storage_alias_spec storage = smem, size = 16384 : !tlx.storage_alias_spec<smem, 16384>
    tt.return
  }
}

// -----

// Test storage_alias_spec with tmem storage and explicit size
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @storage_alias_spec_tmem_sized
  tt.func @storage_alias_spec_tmem_sized() {
    // CHECK: %{{.*}} = tlx.storage_alias_spec storage = tmem, size = 32768 : !tlx.storage_alias_spec<tmem, 32768>
    %0 = tlx.storage_alias_spec storage = tmem, size = 32768 : !tlx.storage_alias_spec<tmem, 32768>
    tt.return
  }
}

// -----

// Test multiple storage_alias_spec in same function
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @multiple_storage_alias_specs
  tt.func @multiple_storage_alias_specs() {
    // CHECK: %{{.*}} = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // CHECK: %{{.*}} = tlx.storage_alias_spec storage = tmem, size = 8192 : !tlx.storage_alias_spec<tmem, 8192>
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_spec storage = tmem, size = 8192 : !tlx.storage_alias_spec<tmem, 8192>
    tt.return
  }
}

// -----

// Test storage_alias_local_alloc with smem storage
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @storage_alias_local_alloc_smem
  tt.func @storage_alias_local_alloc_smem() {
    // CHECK: %[[ALIAS:.*]] = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // CHECK: %[[BUF:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test multiple storage_alias_local_alloc referencing same storage_alias_spec
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @multiple_allocs_same_storage_alias
  tt.func @multiple_allocs_same_storage_alias() {
    // CHECK: %[[ALIAS:.*]] = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // CHECK: %[[A:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // CHECK: %[[B:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Test storage_alias_local_alloc with pointer element type (8 bytes per pointer)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @storage_alias_local_alloc_pointer_type
  tt.func @storage_alias_local_alloc_pointer_type() {
    // CHECK: %[[ALIAS:.*]] = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // CHECK: %[[BUF:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64x!tt.ptr<f32>, #shared, #smem, mutable>
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64x!tt.ptr<f32>, #shared, #smem, mutable>
    tt.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Reuse Group Tests
//===----------------------------------------------------------------------===//

// Test basic reuse_group with shared group_kind and smem storage
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @reuse_group_shared_smem
  tt.func @reuse_group_shared_smem() {
    // CHECK: %[[ALIAS:.*]] = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // CHECK: %[[A:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // CHECK: %[[B:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    // CHECK: %[[GROUP:.*]] = tlx.reuse_group(%[[A]], %[[B]]) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tt.return
  }
}

// -----

// Test reuse_group with distinct group_kind
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @reuse_group_distinct_smem
  tt.func @reuse_group_distinct_smem() {
    // CHECK: %[[ALIAS:.*]] = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // CHECK: %[[A:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // CHECK: %[[B:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // CHECK: %[[GROUP:.*]] = tlx.reuse_group(%[[A]], %[[B]]) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tt.return
  }
}

// -----

// Test nested reuse_group (shared containing distinct)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @nested_reuse_group_shared_distinct
  tt.func @nested_reuse_group_shared_distinct() {
    // CHECK: %[[ALIAS:.*]] = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // CHECK: %[[QK:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // CHECK: %[[P:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    // CHECK: %[[ALPHA:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    // CHECK: %[[INNER:.*]] = tlx.reuse_group(%[[P]], %[[ALPHA]]) group_kind = distinct : (!ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    // CHECK: %[[OUTER:.*]] = tlx.reuse_group(%[[QK]], %[[INNER]]) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !tlx.reuse_group<distinct>) -> !tlx.reuse_group<shared>
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %4 = tlx.reuse_group(%2, %3) group_kind = distinct : (!ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    %5 = tlx.reuse_group(%1, %4) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !tlx.reuse_group<distinct>) -> !tlx.reuse_group<shared>
    tt.return
  }
}

// -----

// Test deeply nested reuse_group (3 levels)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @deeply_nested_reuse_group
  tt.func @deeply_nested_reuse_group() {
    // CHECK: %[[ALIAS:.*]] = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // CHECK: %[[A:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // CHECK: %[[B:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    // CHECK: %[[C:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    // CHECK: %[[D:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    // CHECK: %[[INNER:.*]] = tlx.reuse_group(%[[C]], %[[D]]) group_kind = shared : (!ttg.memdesc<2x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    // CHECK: %[[MIDDLE:.*]] = tlx.reuse_group(%[[B]], %[[INNER]]) group_kind = distinct : (!ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !tlx.reuse_group<shared>) -> !tlx.reuse_group<distinct>
    // CHECK: %[[OUTER:.*]] = tlx.reuse_group(%[[A]], %[[MIDDLE]]) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !tlx.reuse_group<distinct>) -> !tlx.reuse_group<shared>
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %4 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %5 = tlx.reuse_group(%3, %4) group_kind = shared : (!ttg.memdesc<2x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    %6 = tlx.reuse_group(%2, %5) group_kind = distinct : (!ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !tlx.reuse_group<shared>) -> !tlx.reuse_group<distinct>
    %7 = tlx.reuse_group(%1, %6) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !tlx.reuse_group<distinct>) -> !tlx.reuse_group<shared>
    tt.return
  }
}

// -----

// Test reuse_group with single element
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @reuse_group_single_element
  tt.func @reuse_group_single_element() {
    // CHECK: %[[ALIAS:.*]] = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // CHECK: %[[A:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // CHECK: %[[GROUP:.*]] = tlx.reuse_group(%[[A]]) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.reuse_group(%1) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tt.return
  }
}

// -----

// Test reuse_group with multiple elements (more than 2)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @reuse_group_multiple_elements
  tt.func @reuse_group_multiple_elements() {
    // CHECK: %[[ALIAS:.*]] = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // CHECK: %[[A:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // CHECK: %[[B:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    // CHECK: %[[C:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    // CHECK: %[[D:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    // CHECK: %[[GROUP:.*]] = tlx.reuse_group(%[[A]], %[[B]], %[[C]], %[[D]]) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %4 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %5 = tlx.reuse_group(%1, %2, %3, %4) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tt.return
  }
}

// -----

// Test reuse_group with tmem storage
// Note: #tmem binds to tensor_memory_encoding, memory space is #ttng.tensor_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @reuse_group_shared_tmem
  tt.func @reuse_group_shared_tmem() {
    // CHECK: %[[ALIAS:.*]] = tlx.storage_alias_spec storage = tmem : !tlx.storage_alias_spec<tmem>
    // CHECK: %[[A:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: %[[B:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<2x64x64xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: %[[GROUP:.*]] = tlx.reuse_group(%[[A]], %[[B]]) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x64x64xf16, #tmem, #ttng.tensor_memory, mutable>) -> !tlx.reuse_group<shared>
    %0 = tlx.storage_alias_spec storage = tmem : !tlx.storage_alias_spec<tmem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<tmem> -> !ttg.memdesc<2x64x64xf16, #tmem, #ttng.tensor_memory, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x64x64xf16, #tmem, #ttng.tensor_memory, mutable>) -> !tlx.reuse_group<shared>
    tt.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// set_buffer_overlap Tests
//===----------------------------------------------------------------------===//

// Test basic set_buffer_overlap with smem storage
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @set_buffer_overlap_basic
  tt.func @set_buffer_overlap_basic() {
    // CHECK: %[[ALIAS:.*]] = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // CHECK: %[[A:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    // CHECK: %[[B:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]] : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    // CHECK: %[[GROUP:.*]] = tlx.reuse_group(%[[A]], %[[B]]) group_kind = shared
    // CHECK: tlx.set_buffer_overlap(%[[ALIAS]], %[[GROUP]])
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    tt.return
  }
}

// -----

// Test set_buffer_overlap with nested reuse_group
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @set_buffer_overlap_nested
  tt.func @set_buffer_overlap_nested() {
    // CHECK: %[[ALIAS:.*]] = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // CHECK: %[[QK:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]]
    // CHECK: %[[P:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]]
    // CHECK: %[[ALPHA:.*]] = tlx.storage_alias_local_alloc %[[ALIAS]]
    // CHECK: %[[INNER:.*]] = tlx.reuse_group(%[[P]], %[[ALPHA]]) group_kind = distinct
    // CHECK: %[[OUTER:.*]] = tlx.reuse_group(%[[QK]], %[[INNER]]) group_kind = shared
    // CHECK: tlx.set_buffer_overlap(%[[ALIAS]], %[[OUTER]])
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
