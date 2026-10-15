// RUN: triton-opt --split-input-file %s | FileCheck %s
// RUN: triton-opt --split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm=compute-capability=100 %s | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: triton-opt --split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm=compute-capability=90 %s | FileCheck %s --check-prefix=CHECK-LLVM

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
    // CHECK-LLVM: mapa.shared::cluster.u32
    // CHECK-LLVM: mapa.shared::cluster.u32
    // CHECK-LLVM: llvm.inline_asm has_side_effects asm_dialect = att{{.*}}st.async.shared::cluster.mbarrier::complete_tx::bytes
    ttg.async_remote_shmem_store %arg0, rank %arg1, %0 barrier %1 : tensor<1x1xf32, #blocked> -> !ttg.memdesc<1x1xf32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: async_remote_shmem_store_f16
  // CHECK-LLVM-LABEL: llvm.func @async_remote_shmem_store_f16
  tt.func @async_remote_shmem_store_f16(%arg0: tensor<1x64xf16, #blocked>, %arg1: i32) {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK-LLVM: llvm.inline_asm has_side_effects asm_dialect = att{{.*}}st.async.shared::cluster.mbarrier::complete_tx::bytes{{(\.v4)?}}.b32
    // CHECK-LLVM-NOT: st.async.shared::cluster.mbarrier::complete_tx::bytes{{.*}}b16
    ttg.async_remote_shmem_store %arg0, rank %arg1, %0 barrier %1 : tensor<1x64xf16, #blocked> -> !ttg.memdesc<1x64xf16, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared1, #smem, mutable>
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
    // CHECK-LLVM: mapa.shared::cluster.u32
    // CHECK-LLVM-NOT: llvm.inline_asm{{.*}}st.async.shared::cluster.mbarrier
    ttg.remote_shmem_store %arg0, rank %arg1, %0 : tensor<1x1xf32, #blocked> -> !ttg.memdesc<1x1xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

// A physical cluster does not require a multi-CTA logical tensor layout.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: async_remote_shmem_reduce_replicated
  // CHECK-LLVM-LABEL: llvm.func @async_remote_shmem_reduce_replicated
  tt.func @async_remote_shmem_reduce_replicated(%src: tensor<1x4xi32, #blocked>, %rank: i32) {
    %dst = ttg.local_alloc : () -> !ttg.memdesc<1x4xi32, #shared, #smem, mutable>
    %barriers = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrier, #smem, mutable>
    %bar = ttg.memdesc_subslice %barriers[1] : !ttg.memdesc<2xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable, 2>
    // CHECK: ttg.async_remote_shmem_store {{.*}} {store_reduce = 1 : i32}
    // Both lane and warp replicas are masked; each register emits scalar ADD.
    // CHECK-LLVM: llvm.icmp "eq"
    // CHECK-LLVM: llvm.icmp "eq"
    // CHECK-LLVM: llvm.and
    // CHECK-LLVM: llvm.inline_asm has_side_effects asm_dialect = att{{.*}}@$3 red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32 [ $0 + 0 ], $1, [ $2 + 0 ];{{.*}}"r,r,r,b"
    // CHECK-LLVM-COUNT-3: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32
    // CHECK-LLVM-NOT: red.async
    // CHECK-LLVM-NOT: st.async
    // CHECK-LLVM-NOT: mbarrier.arrive
    // CHECK-LLVM: llvm.return
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 1 : i32} : tensor<1x4xi32, #blocked> -> !ttg.memdesc<1x4xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #barrier, #smem, mutable, 2>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CGALayout = [[0]]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: async_remote_shmem_reduce_self
  // CHECK-LLVM-LABEL: llvm.func @async_remote_shmem_reduce_self
  tt.func @async_remote_shmem_reduce_self(%src: tensor<64xi32, #blocked>) {
    %rank = tti.experimental_cluster_cta_id : i32
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: ttg.async_remote_shmem_store {{.*}} {store_reduce = 1 : i32}
    // Self rank gives each CTA a different destination, even with replicated CGA.
    // CHECK-LLVM-NOT: llvm.icmp
    // CHECK-LLVM: %[[SELF:[a-zA-Z0-9_]+]] = nvg.cluster_id
    // CHECK-LLVM-NOT: llvm.icmp
    // CHECK-LLVM: nvvm.mapa {{.*}}, %[[SELF]] : !llvm.ptr<3> -> !llvm.ptr<7>
    // CHECK-LLVM-NOT: llvm.icmp
    // CHECK-LLVM: nvvm.mapa {{.*}}, %[[SELF]] : !llvm.ptr<3> -> !llvm.ptr<7>
    // CHECK-LLVM-NOT: llvm.icmp
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32
    // CHECK-LLVM-NOT: llvm.icmp
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32
    // CHECK-LLVM-NOT: red.async
    // CHECK-LLVM-NOT: st.async
    // CHECK-LLVM-NOT: mbarrier.arrive
    // CHECK-LLVM: llvm.return
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 1 : i32} : tensor<64xi32, #blocked> -> !ttg.memdesc<64xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CGALayout = [[0]]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // A constant target collapses replicated logical CTA issuers. Without the
  // block predicate, both CTAs add the same element and complete twice its bytes.
  // CHECK-LABEL: async_remote_shmem_reduce_constant_rank
  // CHECK-LLVM-LABEL: llvm.func @async_remote_shmem_reduce_constant_rank
  tt.func @async_remote_shmem_reduce_constant_rank(%src: tensor<32xi32, #blocked>) {
    %rank = arith.constant 0 : i32
    %dst = ttg.local_alloc : () -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: ttg.async_remote_shmem_store {{.*}} {store_reduce = 1 : i32}
    // CHECK-LLVM: %[[MASK:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32)
    // CHECK-LLVM: %[[MASKED:[a-zA-Z0-9_]+]] = llvm.and {{.*}}, %[[MASK]] : i32
    // CHECK-LLVM: %[[BLOCKPRED:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[MASKED]], {{.*}} : i32
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32
    // CHECK-LLVM-NOT: red.async
    // CHECK-LLVM-NOT: mbarrier.arrive
    // CHECK-LLVM: llvm.return
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 1 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // Kernel-entry scalar arguments are common to all logical CTAs as well.
  // CHECK-LABEL: async_remote_shmem_reduce_uniform_rank
  // CHECK-LLVM-LABEL: llvm.func @async_remote_shmem_reduce_uniform_rank
  tt.func public @async_remote_shmem_reduce_uniform_rank(%src: tensor<32xi32, #blocked>, %rank: i32) {
    %dst = ttg.local_alloc : () -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: ttg.async_remote_shmem_store {{.*}} {store_reduce = 1 : i32}
    // CHECK-LLVM: llvm.icmp "eq"
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32
    // CHECK-LLVM-NOT: red.async
    // CHECK-LLVM: llvm.return
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 1 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: async_remote_shmem_reduce_operators
  // CHECK-LLVM-LABEL: llvm.func @async_remote_shmem_reduce_operators
  tt.func @async_remote_shmem_reduce_operators(%src: tensor<32xi32, #blocked>, %rank: i32) {
    %dst = ttg.local_alloc : () -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // Signless i32 needs explicit metadata to choose signed MIN/MAX.
    // CHECK: {store_reduce = 2 : i32}
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.u32
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 2 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: {store_reduce = 2 : i32, store_reduce_signed = true}
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.min.s32
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 2 : i32, store_reduce_signed = true} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: {store_reduce = 3 : i32}
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.u32
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 3 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: {store_reduce = 3 : i32, store_reduce_signed = true}
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.max.s32
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 3 : i32, store_reduce_signed = true} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: {store_reduce = 6 : i32}
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.and.b32
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 6 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: {store_reduce = 7 : i32}
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.or.b32
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 7 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: {store_reduce = 8 : i32}
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.xor.b32
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 8 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // The source register is the bound for unsigned INC/DEC.
    // CHECK: {store_reduce = 4 : i32}
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.inc.u32
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 4 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: {store_reduce = 5 : i32}
    // CHECK-LLVM: red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.dec.u32
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 5 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK-LLVM-NOT: red.async
    // CHECK-LLVM-NOT: st.async
    // CHECK-LLVM: llvm.return
    tt.return
  }
}
