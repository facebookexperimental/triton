// RUN: triton-opt --split-input-file %s | FileCheck %s
// RUN: triton-opt --split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm=compute-capability=90 --convert-nv-gpu-to-llvm %s | FileCheck %s --check-prefix=CHECK-LLVM
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: async_store
  // CHECK-LLVM-LABEL: llvm.func @async_store
  tt.func @async_store(%dst: !tt.ptr<i8>, %size: i32) {
    %src = ttg.local_alloc : () -> !ttg.memdesc<1024xi8, #shared, #smem, mutable>
    // CHECK: ttng.async_store
    // CHECK-SAME: !ttg.memdesc<1024xi8, #shared, #smem, mutable>, !tt.ptr<i8>
    // CHECK-LLVM: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-LLVM-SAME: cp.async.bulk.global.shared::cta.bulk_group
    // CHECK-LLVM: nvvm.cp.async.bulk.commit.group
    ttng.async_store %src, %dst, %size : !ttg.memdesc<1024xi8, #shared, #smem, mutable>, !tt.ptr<i8>
    tt.return
  }
}

// -----

// Test async_store with data originating from a register layout (blocked).
// tl.arange creates a blocked layout in registers; local_alloc writes it to SMEM;
// async_store bulk-copies from SMEM to global memory.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem1 = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: async_store_from_registers
  // CHECK-LLVM-LABEL: llvm.func @async_store_from_registers
  tt.func @async_store_from_registers(%dst: !tt.ptr<f32>) {
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %data = arith.sitofp %range : tensor<128xi32, #blocked> to tensor<128xf32, #blocked>
    %smem = ttg.local_alloc %data : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared1, #smem1, mutable>
    %size = arith.constant 512 : i32
    // CHECK: ttng.async_store
    // CHECK-SAME: !ttg.memdesc<128xf32, #{{.*}}, #{{.*}}, mutable>, !tt.ptr<f32>
    // CHECK-LLVM: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-LLVM-SAME: cp.async.bulk.global.shared::cta.bulk_group
    // CHECK-LLVM: nvvm.cp.async.bulk.commit.group
    ttng.async_store %smem, %dst, %size : !ttg.memdesc<128xf32, #shared1, #smem1, mutable>, !tt.ptr<f32>
    tt.return
  }
}
