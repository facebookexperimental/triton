// RUN: triton-opt -split-input-file --allocate-shared-memory-nv %s | FileCheck %s

// A 2-CTA tcgen05.alloc writes the TMEM base address into its 4-byte SMEM slot
// in both CTAs of the pair, unordered with the other CTA's shared memory uses.
// The slot must not share an offset with TMA descriptor staging scratch,
// whether the staging comes before the alloc (TLX paired MMA) or after it
// (the alloc is placed at kernel start otherwise).

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @two_cta_alloc_slot_not_aliased
  // CHECK: ttng.tensormap_create
  // CHECK-SAME: allocation.offset = 0 : i32
  // CHECK: ttng.tcgen5_global_alloc {allocation.offset = {{[1-9][0-9]*}} : i32
  // CHECK-SAME: two_ctas = true
  tt.func public @two_cta_alloc_slot_not_aliased(%arg0: !tt.ptr<i16>, %arg1: !tt.ptr<i8>, %arg2: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c256_i32 = arith.constant 256 : i32
    ttng.tensormap_create %arg1, %arg0, [%c256_i32], [%arg2], [], [%c1_i32] {elem_type = 3 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 2 : i32} : (!tt.ptr<i8>, !tt.ptr<i16>, i32, i32, i32) -> ()
    ttng.tcgen5_global_alloc {tensor_memory_size = 128 : i32, two_ctas = true}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  // CHECK-LABEL: @two_cta_alloc_slot_not_aliased_by_later_scratch
  // CHECK: ttng.tcgen5_global_alloc {allocation.offset = {{[1-9][0-9]*}} : i32
  // CHECK-SAME: two_ctas = true
  // CHECK: ttng.tensormap_create
  // CHECK-SAME: allocation.offset = 0 : i32
  tt.func public @two_cta_alloc_slot_not_aliased_by_later_scratch(%arg0: !tt.ptr<i16>, %arg1: !tt.ptr<i8>, %arg2: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c256_i32 = arith.constant 256 : i32
    ttng.tcgen5_global_alloc {tensor_memory_size = 128 : i32, two_ctas = true}
    ttng.tensormap_create %arg1, %arg0, [%c256_i32], [%arg2], [], [%c1_i32] {elem_type = 3 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 2 : i32} : (!tt.ptr<i8>, !tt.ptr<i16>, i32, i32, i32) -> ()
    tt.return
  }
}

// -----

// A 1-CTA alloc only writes its own CTA's slot, so it may reuse the offset of
// scratch that is dead by the time the alloc runs.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @one_cta_alloc_slot_may_reuse
  // CHECK: ttng.tensormap_create
  // CHECK-SAME: allocation.offset = [[OFF:[0-9]+]] : i32
  // CHECK: ttng.tcgen5_global_alloc {allocation.offset = [[OFF]] : i32
  tt.func public @one_cta_alloc_slot_may_reuse(%arg0: !tt.ptr<i16>, %arg1: !tt.ptr<i8>, %arg2: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c256_i32 = arith.constant 256 : i32
    ttng.tensormap_create %arg1, %arg0, [%c256_i32], [%arg2], [], [%c1_i32] {elem_type = 3 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 2 : i32} : (!tt.ptr<i8>, !tt.ptr<i16>, i32, i32, i32) -> ()
    ttng.tcgen5_global_alloc {tensor_memory_size = 128 : i32, two_ctas = false}
    tt.return
  }
}
