// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=100' --triton-nvidia-gpu-membar='compute-capability=100 ptx-version=87' --triton-nvidia-gpu-tmem-barrier-insertion --triton-nvidia-gpu-optimize-mbarrier-arrivals --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --convert-nv-gpu-to-llvm --convert-warp-specialize-to-llvm | FileCheck %s

// A per-warp rescale of a TMEM accumulator, like the FA correction's. Each
// thread loads and stores its own rows, so the region needs a wait between
// the load and the store but no CTA barrier. The store completes before the
// region exits, as on any CFG edge, and the region lowers to a conditional
// branch around its body.
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 256 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @rescale_per_warp
  // CHECK: llvm.cond_br %arg0, ^[[BODY:bb[0-9]+]], ^[[MERGE:bb[0-9]+]]
  // CHECK: ^[[BODY]]:
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.tcgen05.ld {{.*}}shape = shape_32x32b : vector<128xi32>
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.tcgen05.wait <load>
  // CHECK-NOT: nvvm.barrier
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NEXT: nvvm.tcgen05.wait <store>
  // CHECK-NEXT: llvm.br ^[[MERGE]]
  // CHECK: ^[[MERGE]]:
  // CHECK-NOT: ttg.warp_predicate
  tt.func @rescale_per_warp(%pred: i1, %alpha: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %buf = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_predicate %pred () {
      %0 = ttng.tmem_load %acc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %1 = arith.mulf %0, %alpha : tensor<128x128xf32, #blocked>
      ttng.tmem_store %1, %acc, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      ttg.predicate_yield
    } {wave_uniform} : (i1) -> ()
    tt.return
  }
}
