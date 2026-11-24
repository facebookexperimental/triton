// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=100' --triton-nvidia-gpu-proxy-fence-insertion='compute-capability=100' --triton-nvidia-gpu-membar='compute-capability=100 ptx-version=87' --triton-nvidia-gpu-tmem-barrier-insertion --triton-nvidia-gpu-optimize-mbarrier-arrivals --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --convert-nv-gpu-to-llvm --convert-warp-specialize-to-llvm -verify-diagnostics -o /dev/null

// Warps that skip a predicated region never reach a barrier inside it. The
// synchronization passes walk into the region, so a barrier or fence its body
// would need lands inside it, and the region is rejected rather than the
// synchronization dropped.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear64 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 256 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // The store writes rows that other threads of the same warp loaded, so TMEM
  // barrier insertion puts a CTA barrier inside the region.
  tt.func @cross_thread_hazard(%pred: i1, %data: tensor<64x128xf32, #linear64>) {
    %true = arith.constant true
    %x = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %x2 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 16 : i32} : () -> !ttg.memdesc<64x128xf32, #tmem64, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{region may not contain CTA barriers}}
    ttg.warp_predicate %pred () {
      %0 = ttng.tmem_load %x : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      ttng.tmem_store %data, %x2, %true : tensor<64x128xf32, #linear64> -> !ttg.memdesc<64x128xf32, #tmem64, #ttng.tensor_memory, mutable>
      ttg.predicate_yield
    } {wave_uniform} : (i1) -> ()
    tt.return
  }
}

// -----

// A per-warp rescale whose layout gives warps w and w+4 the same words. Every
// op in the body is one a region admits, but the store must wait for the other
// warp's load: TMEM barrier insertion puts a CTA barrier inside the region,
// which a warp that skips the region would never reach. Kept opaque instead,
// the region would compile with no wait and no barrier, and race.
#bcast = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @cross_warp_rescale(%pred: i1, %alpha: tensor<128x1xf32, #bcast>) {
    %true = arith.constant true
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{region may not contain CTA barriers}}
    ttg.warp_predicate %pred () {
      %0 = ttng.tmem_load %mem : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #bcast>
      %1 = arith.mulf %0, %alpha : tensor<128x1xf32, #bcast>
      ttng.tmem_store %1, %mem, %true : tensor<128x1xf32, #bcast> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      ttg.predicate_yield
    } {wave_uniform} : (i1) -> ()
    tt.return
  }
}

// -----

// Proxy fence insertion walks into the region, so the fence that orders the
// load before it with the TMA copy inside it lands in the body. The CTA barrier
// before the region keeps membar from needing one inside. A region does not
// support fences, so it is rejected instead of losing the fence.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @proxy_fence(%pred: i1, %desc: !tt.tensordesc<64x64xf32, #shared>, %bar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> tensor<64x64xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    %v = ttg.local_load %buf : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    ttg.barrier local
    // expected-error @+1 {{operation ttng.fence_async_shared is not yet supported in predicated regions on NVIDIA}}
    ttg.warp_predicate %pred () {
      ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %buf, %bar, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
      ttg.predicate_yield
    } {wave_uniform} : (i1) -> ()
    tt.return %v : tensor<64x64xf32, #blocked>
  }
}

// -----

// Cluster barrier insertion walks into the region, so the cluster barrier
// between two conversions through cross-CTA scratch lands in the body: the
// second reuses scratch that the other CTA may still be reading for the first.
// Warps that skip the region would never arrive, so it is rejected instead of
// losing the barrier.
#blockedSplitM = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#blockedSplitN = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 1]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @cluster_barrier(%pred: i1, %x: tensor<256x128xf16, #blockedSplitM>) {
    // expected-error @+1 {{operation ttng.cluster_arrive is not yet supported in predicated regions on NVIDIA}}
    ttg.warp_predicate %pred () {
      %0 = ttg.convert_layout %x : tensor<256x128xf16, #blockedSplitM> -> tensor<256x128xf16, #blockedSplitN>
      %1 = ttg.convert_layout %0 : tensor<256x128xf16, #blockedSplitN> -> tensor<256x128xf16, #blockedSplitM>
      ttg.predicate_yield
    } {wave_uniform} : (i1) -> ()
    tt.return
  }
}

// -----

// The lanes of a warp may disagree on a predicate that is not wave-uniform,
// which warp-collective instructions such as tcgen05.ld do not allow.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @not_wave_uniform(%pred: i1) {
    // expected-error @+1 {{per-lane and non-wave-uniform predicates are not yet supported in predicated regions on NVIDIA}}
    ttg.warp_predicate %pred () {
      ttg.predicate_yield
    } : (i1) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tensor_predicate(%pred: tensor<128xi1, #blocked>) {
    // expected-error @+1 {{per-lane and non-wave-uniform predicates are not yet supported in predicated regions on NVIDIA}}
    ttg.warp_predicate %pred () {
      ttg.predicate_yield
    } {wave_uniform} : (tensor<128xi1, #blocked>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @carried_value(%pred: i1, %init: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    // expected-error @+1 {{carried values are not yet supported in predicated regions on NVIDIA}}
    %r = ttg.warp_predicate %pred (%init) {
      %0 = arith.addf %init, %init : tensor<128xf32, #blocked>
      ttg.predicate_yield %0 : tensor<128xf32, #blocked>
    } {wave_uniform} : (i1, tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    tt.return %r : tensor<128xf32, #blocked>
  }
}

// -----

// Mbarrier waits are not yet supported in predicated regions. Publish the
// barrier before the region so this tests the operation restriction directly.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mbarrier_wait(%pred: i1) {
    %c0 = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.barrier local
    // expected-error @+1 {{operation ttng.wait_barrier is not yet supported in predicated regions on NVIDIA}}
    ttg.warp_predicate %pred () {
      ttng.wait_barrier %bar, %c0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.predicate_yield
    } {wave_uniform} : (i1) -> ()
    tt.return
  }
}

// -----

// A leader arrive lowers to a CTA barrier and then an arrive by thread 0
// alone. Warps that skip the region would miss the barrier, and whether
// thread 0 arrives would depend on warp 0.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @leader_arrive(%pred: i1) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // expected-error @+1 {{operation ttng.arrive_barrier is not yet supported in predicated regions on NVIDIA}}
    ttg.warp_predicate %pred () {
      ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.predicate_yield
    } {wave_uniform} : (i1) -> ()
    tt.return
  }
}

// -----

// A per-thread arrive is rejected too. The FA rescale does not need one, and
// membar would put a CTA barrier before one that follows a memory access.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @per_thread_arrive(%pred: i1) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 128 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.barrier local
    // expected-error @+1 {{operation ttng.arrive_barrier is not yet supported in predicated regions on NVIDIA}}
    ttg.warp_predicate %pred () {
      ttng.arrive_barrier %bar, 1 {perThread} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.predicate_yield
    } {wave_uniform} : (i1) -> ()
    tt.return
  }
}

// -----

// Lowering deduplicates a store to one thread per stored value, here thread 0
// for the scalar. Whether anything is stored would then depend on warp 0
// entering the region, not on the warps that ran the body. Tensor stores
// replicated across warps are deduplicated the same way, so every tt.store
// is rejected.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @scalar_store(%pred: i1, %ptr: !tt.ptr<f32>, %v: f32) {
    // expected-error @+1 {{operation tt.store is not yet supported in predicated regions on NVIDIA}}
    ttg.warp_predicate %pred () {
      tt.store %ptr, %v : !tt.ptr<f32>
      ttg.predicate_yield
    } {wave_uniform} : (i1) -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @callee() attributes {noinline = true} {
    tt.return
  }
  tt.func @call(%pred: i1) {
    // expected-error @+1 {{operation tt.call is not yet supported in predicated regions on NVIDIA}}
    ttg.warp_predicate %pred () {
      tt.call @callee() : () -> ()
      ttg.predicate_yield
    } {wave_uniform} : (i1) -> ()
    tt.return
  }
}

// -----

// The TTGIR verifier cannot see a barrier inside inline asm.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @inline_asm_barrier(%pred: i1, %x: tensor<128xi32, #blocked>) {
    // expected-error @+1 {{lowered body may not contain CTA barriers}}
    ttg.warp_predicate %pred () {
      %0 = tt.elementwise_inline_asm "bar.sync 0; mov.b32 $0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %x : tensor<128xi32, #blocked> -> tensor<128xi32, #blocked>
      ttg.predicate_yield
    } {wave_uniform} : (i1) -> ()
    tt.return
  }
}
