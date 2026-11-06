// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics -triton-nvidia-optimize-tmem-layouts --allocate-shared-memory-nv=compute-capability=100 --convert-triton-gpu-to-llvm=compute-capability=100 | FileCheck %s

// N=96 moves as three 32-column chunks at column offsets 0, 32 and 64.
#tl = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @ld_128x96_4w
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x32.b32 {{.*}}[$32 + 0]
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x32.b32 {{.*}}[$32 + 32]
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x32.b32 {{.*}}[$32 + 64]
  // CHECK-NOT: tcgen05.ld
  // CHECK: tcgen05.st.sync.aligned.32x32b.x32.b32 [$1 + 0]
  // CHECK: tcgen05.st.sync.aligned.32x32b.x32.b32 [$1 + 32]
  // CHECK: tcgen05.st.sync.aligned.32x32b.x32.b32 [$1 + 64]
  tt.func public @ld_128x96_4w() {
    %true = arith.constant true
    %c = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    %v = ttng.tmem_load %c : !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x96xf32, #tl>
    ttng.tmem_store %v, %c, %true : tensor<128x96xf32, #tl> -> !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

// At 8 warps each chunk splits into two 16-column halves, one per warp group;
// the alloc and the store both issue three messages.
#tl = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 16]], block = [], inputRadices = {register = [2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2, 2], block = []}, outDimSizes = [128, 96]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @st_128x96_8w
  // CHECK-COUNT-6: tcgen05.st.sync.aligned.32x32b.x16.b32
  // CHECK-NOT: tcgen05.st
  tt.func public @st_128x96_8w() {
    %true = arith.constant true
    %v = arith.constant dense<1.0> : tensor<128x96xf32, #tl>
    %c = ttng.tmem_alloc %v {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x96xf32, #tl>) -> !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    ttng.tmem_store %v, %c, %true : tensor<128x96xf32, #tl> -> !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

// An N reduction at 8 warps moves the second warp group to rows 16..31 of each
// 32-row block (16x32bx2), and the NPOT layout keeps that split.
#tl = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 16]], block = [], inputRadices = {register = [2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2, 2], block = []}, outDimSizes = [128, 96]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @ld_red_128x96_8w
  // CHECK-COUNT-3: tcgen05.ld.sync.aligned.16x32bx2.x16.b32
  // CHECK-NOT: tcgen05.ld
  tt.func public @ld_red_128x96_8w(%p: !tt.ptr<f32>) {
    %c = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    %v = ttng.tmem_load %c : !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x96xf32, #tl>
    %r = "tt.reduce"(%v) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) : (tensor<128x96xf32, #tl>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #tl}>>
    %s = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #tl}>>
    tt.store %s, %r : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #tl}>>
    tt.return
  }
}

// -----

// The odd digit must be the outermost register digit.
#tl = #ttg.linear<{register = [[0, 32], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = [], inputRadices = {register = [3, 2, 2, 2, 2, 2], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ld_odd_inner(%c: !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>) {
    // expected-error @+1 {{NPOT TMEM load/store is not yet supported for register layout}}
    %v = ttng.tmem_load %c : !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x96xf32, #tl>
    tt.return
  }
}

// -----

// blockM=64 has no NPOT plan yet.
#tl = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [64, 96]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ld_block_m64(%c: !ttg.memdesc<64x96xf32, #tmem, #ttng.tensor_memory, mutable, 64x128>) {
    // expected-error @+1 {{NPOT TMEM load/store is not yet supported for}}
    %v = ttng.tmem_load %c : !ttg.memdesc<64x96xf32, #tmem, #ttng.tensor_memory, mutable, 64x128> -> tensor<64x96xf32, #tl>
    tt.return
  }
}

// -----

#tl = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = [], inputRadices = {register = [2, 2, 2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 96]}>
#red = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ld_red(%c: !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>) {
    // expected-error @+1 {{NPOT tmem_load reduction is not yet supported}}
    %v, %r = ttng.tmem_load %c {redOp = #ttng.redOp<min>} : !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x96xf32, #tl>, tensor<128xf32, #red>
    tt.return
  }
}
