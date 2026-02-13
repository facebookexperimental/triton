// RUN: triton-opt --split-input-file %s --tlx-storage-alias-lowering --verify-diagnostics

//===----------------------------------------------------------------------===//
// Buffer Offset Calculation Error Tests
//===----------------------------------------------------------------------===//

// Test: Duplicate set_buffer_overlap on same spec
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @duplicate_set_buffer_overlap() {
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    %4 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    // expected-error @+1 {{storage_alias_spec already has a set_buffer_overlap defined}}
    tlx.set_buffer_overlap(%0, %4) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
}
