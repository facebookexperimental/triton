// RUN: triton-opt -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm %s | FileCheck %s
// RUN: triton-opt -split-input-file --allocate-shared-memory-nv %s | FileCheck %s --check-prefix=ALLOC

// Test that TCGen5GlobalAllocOp is lowered to tcgen05.alloc PTX at its position,
// and TensorMemoryBaseAddress ops from tmem_alloc are replaced with the
// alloc result.

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
// ALLOC: ttg.shared = 4 : i32
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.tensor_memory_size" = 128 : i32} {
  // CHECK-LABEL: @tcgen5_global_alloc_lowering
  // CHECK: tcgen05.alloc.cta_group::2
  // CHECK: nvvm.barrier0
  // CHECK: tcgen05.relinquish_alloc_permit.cta_group::2
  // ALLOC-LABEL: @tcgen5_global_alloc_lowering
  tt.func public @tcgen5_global_alloc_lowering() attributes {noinline = false} {
    // ALLOC-NEXT: ttng.tcgen5_global_alloc
    // ALLOC-SAME: allocation.offset = 0 : i32
    ttng.tcgen5_global_alloc {tensor_memory_size = 128 : i32, two_ctas = true}
    %alloc = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Test that the allocation pass assigns an offset for TCGen5GlobalAllocOp's scratch
// buffer that does not overlap with a live local_alloc buffer.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {tlx.enable_paired_cta_mma = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.tensor_memory_size" = 128 : i32} {
  // CHECK-LABEL: @tcgen5_global_alloc_no_smem_overlap
  // The local_alloc is live across the tcgen5_global_alloc, so the allocation pass
  // must assign a non-zero offset for the scratch buffer (via GEP).
  // CHECK: llvm.getelementptr
  // CHECK: tcgen05.alloc.cta_group::2
  // ALLOC-LABEL: @tcgen5_global_alloc_no_smem_overlap
  tt.func public @tcgen5_global_alloc_no_smem_overlap(%bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) attributes {noinline = false} {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
    // ALLOC: ttng.tcgen5_global_alloc
    // ALLOC-SAME: allocation.offset = 256 : i32
    ttng.tcgen5_global_alloc {tensor_memory_size = 128 : i32, two_ctas = true}
    %alloc = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // Use %buf after tcgen5_global_alloc to keep it live across the alloc.
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.local_dealloc %buf : !ttg.memdesc<1x128xf16, #shared, #smem, mutable>
    tt.return
  }
}
