// RUN: triton-opt --split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm=compute-capability=100 %s | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Test that async_remote_shmem_copy generates cp.async.bulk and
// elects only one thread (warp 0) to issue the copy.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @async_remote_shmem_copy_basic
  tt.func @async_remote_shmem_copy_basic(%arg0: i32) {
    %src = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: nvvm.mapa
    // CHECK: nvvm.mapa
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att{{.*}}cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
    ttg.async_remote_shmem_copy %src, rank %arg0, %dst barrier %bar : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// Test that async_remote_shmem_copy with a subsliced destination
// includes the subslice offset in the address passed to mapa.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @async_remote_shmem_copy_subslice
  tt.func @async_remote_shmem_copy_subslice(%arg0: i32) {
    %src = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %parent = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %dst = ttg.memdesc_subslice %parent[128, 0] : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64>
    // The subslice at [128, 0] adds an offset to the base pointer.
    // Verify GEP (offset computation) appears before mapa for the dst.
    // CHECK: llvm.getelementptr
    // CHECK: nvvm.mapa
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att{{.*}}cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
    ttg.async_remote_shmem_copy %src, rank %arg0, %dst barrier %bar : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 256x64> barrier_ty !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}
