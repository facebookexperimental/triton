// RUN: triton-opt %s --split-input-file --verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90"} {
  tt.func @invalid_float(%src: tensor<32xf32, #blocked>, %rank: i32, %dst: !ttg.memdesc<32xf32, #shared, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @+1 {{32-bit integer}}
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 1 : i32} : tensor<32xf32, #blocked> -> !ttg.memdesc<32xf32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90"} {
  tt.func @invalid_i64(%src: tensor<32xi64, #blocked>, %rank: i32, %dst: !ttg.memdesc<32xi64, #shared, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @+1 {{32-bit integer}}
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 1 : i32} : tensor<32xi64, #blocked> -> !ttg.memdesc<32xi64, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90"} {
  tt.func @invalid_kind(%src: tensor<32xi32, #blocked>, %rank: i32, %dst: !ttg.memdesc<32xi32, #shared, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @+1 {{store_reduce}}
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 9 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90"} {
  tt.func @invalid_single_cta(%src: tensor<32xi32, #blocked>, %rank: i32, %dst: !ttg.memdesc<32xi32, #shared, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @+1 {{at least two physical CTAs}}
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 1 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80"} {
  tt.func @invalid_arch(%src: tensor<32xi32, #blocked>, %rank: i32, %dst: !ttg.memdesc<32xi32, #shared, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @+1 {{NVIDIA SM90}}
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 1 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950"} {
  tt.func @invalid_backend(%src: tensor<32xi32, #blocked>, %rank: i32, %dst: !ttg.memdesc<32xi32, #shared, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @+1 {{NVIDIA SM90}}
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 1 : i32} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90"} {
  tt.func @invalid_signed_inc(%src: tensor<32xi32, #blocked>, %rank: i32, %dst: !ttg.memdesc<32xi32, #shared, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @+1 {{unsigned}}
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 4 : i32, store_reduce_signed = true} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90"} {
  tt.func @invalid_signed_dec(%src: tensor<32xi32, #blocked>, %rank: i32, %dst: !ttg.memdesc<32xi32, #shared, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @+1 {{unsigned}}
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 5 : i32, store_reduce_signed = true} : tensor<32xi32, #blocked> -> !ttg.memdesc<32xi32, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.cluster-dim-x" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90"} {
  tt.func @invalid_i16_bitwise(%src: tensor<32xi16, #blocked>, %rank: i32, %dst: !ttg.memdesc<32xi16, #shared, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @+1 {{32-bit integer}}
    ttg.async_remote_shmem_store %src, rank %rank, %dst barrier %bar {store_reduce = 8 : i32} : tensor<32xi16, #blocked> -> !ttg.memdesc<32xi16, #shared, #smem, mutable> barrier_ty !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
