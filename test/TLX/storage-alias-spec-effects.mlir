// RUN: triton-opt --split-input-file %s --cse | FileCheck %s

// A dead storage_alias_spec must be erased by CSE/DCE.
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @dead_spec_erased
  // CHECK-NOT: tlx.storage_alias_spec
  // CHECK: tt.return
  tt.func @dead_spec_erased() {
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %1 = tlx.storage_alias_spec storage = tmem : !tlx.storage_alias_spec<tmem>
    tt.return
  }
}

// -----

// Two live specs with identical attributes must NOT be merged by CSE.
// The set_buffer_overlap sinks keep every spec and alloc live.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @identical_specs_not_merged
  // CHECK-COUNT-2: tlx.storage_alias_spec storage = smem
  tt.func @identical_specs_not_merged() {
    %s0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %s1 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %a0 = tlx.storage_alias_local_alloc %s0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %a1 = tlx.storage_alias_local_alloc %s1 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %g0 = tlx.reuse_group(%a0) group_kind = shared : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    %g1 = tlx.reuse_group(%a1) group_kind = shared : (!ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>) -> !tlx.reuse_group<shared>
    tlx.set_buffer_overlap(%s0, %g0) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    tlx.set_buffer_overlap(%s1, %g1) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<shared>) -> ()
    tt.return
  }
}
