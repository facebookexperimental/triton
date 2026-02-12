// RUN: triton-opt --split-input-file %s --verify-diagnostics

//===----------------------------------------------------------------------===//
// set_buffer_overlap Verifier Error Tests
//===----------------------------------------------------------------------===//

// Test: duplicate element in reuse_group tree (same allocation appears twice via nesting)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @set_buffer_overlap_duplicate_element() {
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    // Create a nested group that includes %1 twice (once directly, once via inner group)
    %inner = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    %outer = tlx.reuse_group(%1, %inner) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !tlx.reuse_group<shared>) -> !tlx.reuse_group<distinct>
    // expected-error @+1 {{reuse_group tree contains duplicate elements}}
    tlx.set_buffer_overlap(%0, %outer) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
}

// -----

// Test: allocations in reuse_group must all reference the same storage_alias_spec
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @set_buffer_overlap_mismatched_spec() {
    %spec1 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %spec2 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // Allocate from different specs
    %1 = tlx.storage_alias_local_alloc %spec1 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %spec2 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %group = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    // expected-error @+1 {{all allocations in the reuse_group must reference the same storage_alias_spec}}
    tlx.set_buffer_overlap(%spec1, %group) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    tt.return
  }
}
