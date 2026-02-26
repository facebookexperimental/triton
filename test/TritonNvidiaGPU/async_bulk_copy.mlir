// RUN: triton-opt --split-input-file %s | FileCheck %s
// RUN: triton-opt --split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm=compute-capability=90 --convert-nv-gpu-to-llvm %s | FileCheck %s --check-prefix=CHECK-LLVM

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: async_bulk_copy_local_to_global
  // CHECK-LLVM-LABEL: llvm.func @async_bulk_copy_local_to_global
  tt.func @async_bulk_copy_local_to_global(%dst: !tt.ptr<i8>, %size: i32) {
    %src = ttg.local_alloc : () -> !ttg.memdesc<1024xi8, #shared, #smem, mutable>
    // CHECK: ttng.async_bulk_copy_local_to_global
    // CHECK-SAME: !ttg.memdesc<1024xi8, #shared, #smem, mutable>, !tt.ptr<i8>
    // CHECK-LLVM: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-LLVM-SAME: cp.async.bulk.global.shared::cta.bulk_group
    // CHECK-LLVM: nvvm.cp.async.bulk.commit.group
    ttng.async_bulk_copy_local_to_global %src, %dst, %size : !ttg.memdesc<1024xi8, #shared, #smem, mutable>, !tt.ptr<i8>
    tt.return
  }
}
