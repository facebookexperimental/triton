// RUN: triton-opt %s --canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#converted = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @preserve_add
  // CHECK-SAME: %[[SRC:[a-zA-Z0-9_]+]]: tensor<64xi32,
  tt.func @preserve_add(%src: tensor<64xi32, #blocked>, %rank: i32) {
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %converted = ttg.convert_layout %src : tensor<64xi32, #blocked> -> tensor<64xi32, #converted>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: ttg.async_remote_shmem_store %[[SRC]], rank {{.*}} {store_reduce = 1 : i32} : tensor<64xi32,
    ttg.async_remote_shmem_store %converted, rank %rank, %dst barrier %bar {store_reduce = 1 : i32} : tensor<64xi32, #converted> -> !ttg.memdesc<64xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @preserve_signed_min
  // CHECK-SAME: %[[MIN_SRC:[a-zA-Z0-9_]+]]: tensor<64xi32,
  tt.func @preserve_signed_min(%src: tensor<64xi32, #blocked>, %rank: i32) {
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %converted = ttg.convert_layout %src : tensor<64xi32, #blocked> -> tensor<64xi32, #converted>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: ttg.async_remote_shmem_store %[[MIN_SRC]], rank {{.*}} {store_reduce = 2 : i32, store_reduce_signed = true}
    ttg.async_remote_shmem_store %converted, rank %rank, %dst barrier %bar {store_reduce = 2 : i32, store_reduce_signed = true} : tensor<64xi32, #converted> -> !ttg.memdesc<64xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @preserve_signed_max
  // CHECK-SAME: %[[MAX_SRC:[a-zA-Z0-9_]+]]: tensor<64xi32,
  tt.func @preserve_signed_max(%src: tensor<64xi32, #blocked>, %rank: i32) {
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %converted = ttg.convert_layout %src : tensor<64xi32, #blocked> -> tensor<64xi32, #converted>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: ttg.async_remote_shmem_store %[[MAX_SRC]], rank {{.*}} {store_reduce = 3 : i32, store_reduce_signed = true}
    ttg.async_remote_shmem_store %converted, rank %rank, %dst barrier %bar {store_reduce = 3 : i32, store_reduce_signed = true} : tensor<64xi32, #converted> -> !ttg.memdesc<64xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
