// RUN: triton-opt --split-input-file %s | FileCheck %s
// RUN: triton-opt --split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm=compute-capability=100 %s | FileCheck %s --check-prefix=CHECK-LLVM

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: async_remote_shmem_store
  // CHECK-LLVM-LABEL: llvm.func @async_remote_shmem_store
  tt.func @async_remote_shmem_store(%arg0: tensor<1x1xf32, #blocked>, %arg1: i32) {
    // CHECK: %c0_i32 = arith.constant 0 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: %0 = ttg.local_alloc : () -> !ttg.memdesc<1x1xf32, #shared, #smem, mutable>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x1xf32, #shared, #smem, mutable>
    // CHECK: %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: ttg.async_remote_shmem_store %arg0, rank %arg1, %0 barrier %1 : tensor<1x1xf32, #blocked> -> !ttg.memdesc<1x1xf32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK-LLVM: nvvm.mapa
    // CHECK-LLVM: nvvm.mapa
    // CHECK-LLVM: llvm.inline_asm has_side_effects asm_dialect = att{{.*}}st.async.shared::cluster.mbarrier::complete_tx::bytes
    ttg.async_remote_shmem_store %arg0, rank %arg1, %0 barrier %1 : tensor<1x1xf32, #blocked> -> !ttg.memdesc<1x1xf32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: remote_shmem_store_no_barrier
  // CHECK-LLVM-LABEL: llvm.func @remote_shmem_store_no_barrier
  tt.func @remote_shmem_store_no_barrier(%arg0: tensor<1x1xf32, #blocked>, %arg1: i32) {
    // CHECK: %c0_i32 = arith.constant 0 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: %0 = ttg.local_alloc : () -> !ttg.memdesc<1x1xf32, #shared, #smem, mutable>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x1xf32, #shared, #smem, mutable>
    // CHECK: ttg.remote_shmem_store %arg0, rank %arg1, %0 : tensor<1x1xf32, #blocked> -> !ttg.memdesc<1x1xf32, #shared, #smem, mutable>
    // CHECK-LLVM: nvvm.mapa
    // CHECK-LLVM-NOT: llvm.inline_asm{{.*}}st.async.shared::cluster.mbarrier
    ttg.remote_shmem_store %arg0, rank %arg1, %0 : tensor<1x1xf32, #blocked> -> !ttg.memdesc<1x1xf32, #shared, #smem, mutable>
    tt.return
  }
}
