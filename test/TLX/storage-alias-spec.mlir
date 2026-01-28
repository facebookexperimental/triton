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
