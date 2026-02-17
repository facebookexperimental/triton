// RUN: triton-opt --split-input-file %s --tlx-storage-alias-lowering --verify-diagnostics

//===----------------------------------------------------------------------===//
// Buffer Layout Error Tests (during TLXStorageAliasLowering)
//===----------------------------------------------------------------------===//

// Test: bytes_between_buffers not evenly divisible by buffer size
// Two allocations in distinct with power-of-2 shapes that don't divide evenly
// A: 2x64x64xf32 = 16384 bytes per buffer
// B: 2x64x32xf32 = 8192 bytes per buffer
// distinct total = 16384 + 8192 = 24576 bytes per buffer
// For A: 24576 % 16384 = 8192 (NOT divisible)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @bytes_between_not_divisible_error() {
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // expected-error @+1 {{bytes_between_buffers (24576) must be a multiple of the original buffer size (16384)}}
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x32xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x32xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
}

// -----

// Test: Another case where bytes_between_buffers is not evenly divisible
// A: 2x128x64xf32 = 32768 bytes per buffer
// B: 2x64x64xf32 = 16384 bytes per buffer
// distinct total = 32768 + 16384 = 49152 bytes per buffer
// For A: 49152 % 32768 = 16384 (not divisible)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @bytes_between_not_divisible_error_2() {
    %0 = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    // expected-error @+1 {{bytes_between_buffers (49152) must be a multiple of the original buffer size (32768)}}
    %1 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x128x64xf32, #shared, #smem, mutable>
    %2 = tlx.storage_alias_local_alloc %0 : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
    %3 = tlx.reuse_group(%1, %2) group_kind = distinct : (!ttg.memdesc<2x128x64xf32, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%0, %3) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
}
