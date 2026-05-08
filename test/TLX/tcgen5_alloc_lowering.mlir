// RUN: triton-opt -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm %s | FileCheck %s

// Test that TCGen5AllocOp is lowered to tcgen05.alloc PTX at its position,
// and TensorMemoryBaseAddress ops from tmem_alloc are replaced with the
// alloc result.

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.tensor_memory_size" = 128 : i32} {
  // CHECK-LABEL: @tcgen5_alloc_lowering
  // CHECK: tcgen05.alloc.cta_group::2
  // CHECK: nvvm.barrier0
  // CHECK: tcgen05.relinquish_alloc_permit.cta_group::2
  tt.func public @tcgen5_alloc_lowering() attributes {noinline = false} {
    ttng.tcgen5_alloc {tensor_memory_size = 128 : i32, two_ctas = true}
    %alloc = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Test that tcgen05.alloc uses a SMEM offset that doesn't overlap with other
// shared memory allocations. The local_alloc occupies SMEM at offset 0, so
// the tcgen05.alloc buffer must be at a different offset.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 136 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.tensor_memory_size" = 128 : i32, "ttg.tmem_alloc_smem_offset" = 128 : i32} {
  // CHECK-LABEL: @tcgen5_alloc_no_smem_overlap
  // The local_alloc uses SMEM at offset 0. The tcgen05.alloc buffer must
  // use a different offset (128) to avoid clobbering the local_alloc data.
  // CHECK: llvm.mlir.constant(128 : i32)
  // CHECK: llvm.getelementptr
  // CHECK: tcgen05.alloc.cta_group::2
  tt.func public @tcgen5_alloc_no_smem_overlap() attributes {noinline = false} {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
    ttng.tcgen5_alloc {tensor_memory_size = 128 : i32, two_ctas = true}
    %alloc = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
