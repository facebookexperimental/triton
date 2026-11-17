// RUN: triton-opt %s -split-input-file --nvgpu-ws-data-partition=num-warp-groups=1 | FileCheck %s
// RUN: triton-opt %s -split-input-file --nvgpu-ws-data-partition=num-warp-groups=1 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN

// The A tile feeds both the MMA (sliced along M) and a row reduction. Slicing
// the descriptor block for the MMA must also slice the reduction chain, or the
// rewrite keeps a full-size load of a descriptor whose block type is already
// sliced, which fails verification.

// CHECK-LABEL: @descriptor_load_feeds_mma_and_row_reduce
// CHECK-NOT: tensor<256x64xbf16
// CHECK-COUNT-2: tt.descriptor_load {{.*}} -> tensor<128x64xbf16
// CHECK-NOT: tensor<256x64xbf16
// CHECK: }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32
// CHECK: ttng.tc_gen5_mma {{.*}}: !ttg.memdesc<128x64xbf16
// CHECK: }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32
// CHECK: ttng.tc_gen5_mma {{.*}}: !ttg.memdesc<128x64xbf16
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [128, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @descriptor_load_feeds_mma_and_row_reduce(
      %a_desc: !tt.tensordesc<256x64xbf16, #shared>,
      %b_desc: !tt.tensordesc<128x64xbf16, #shared>,
      %rstd_out: !tt.ptr<f32>, %k_tiles: i32, %m_off: i32, %n_off: i32) {
    %false = arith.constant false
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %zero_rows = arith.constant dense<0.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #linear>
    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %zero_acc, %acc[%acc_tok], %true : tensor<256x128xf32, #linear> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %loop:3 = scf.for %ki = %c0 to %k_tiles step %c1 iter_args(%sumsq = %zero_rows, %use_acc = %false, %tok = %init_tok) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i1, !ttg.async.token) : i32 {
      %k = arith.muli %ki, %c64 : i32
      %a = tt.descriptor_load %a_desc[%m_off, %k] : !tt.tensordesc<256x64xbf16, #shared> -> tensor<256x64xbf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<256x64xbf16, #blocked>) -> !ttg.memdesc<256x64xbf16, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%n_off, %k] : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
      %a_f32 = arith.extf %a : tensor<256x64xbf16, #blocked> to tensor<256x64xf32, #blocked>
      %sq = arith.mulf %a_f32, %a_f32 : tensor<256x64xf32, #blocked>
      %row = "tt.reduce"(%sq) <{axis = 1 : i32}> ({
      ^bb0(%x: f32, %y: f32):
        %s = arith.addf %x, %y : f32
        tt.reduce.return %s : f32
      }) : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %next = arith.addf %sumsq, %row : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %b_smem = ttg.local_alloc %b : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
      %b_t = ttg.memdesc_trans %b_smem {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem> -> !ttg.memdesc<64x128xbf16, #shared2, #smem>
      %mma_tok = ttng.tc_gen5_mma %a_smem, %b_t, %acc[%tok], %use_acc, %true : !ttg.memdesc<256x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared2, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %next, %true, %mma_tok : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i1, !ttg.async.token
    } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}
    %rstd = math.rsqrt %loop#0 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %res, %res_tok = ttng.tmem_load %acc[%loop#2] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear>
    %rstd_2d = tt.expand_dims %rstd {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
    %rstd_b = tt.broadcast %rstd_2d : tensor<256x1xf32, #blocked> -> tensor<256x128xf32, #blocked>
    %rstd_l = ttg.convert_layout %rstd_b : tensor<256x128xf32, #blocked> -> tensor<256x128xf32, #linear>
    %scaled = arith.mulf %res, %rstd_l : tensor<256x128xf32, #linear>
    %ptrs = tt.splat %rstd_out : !tt.ptr<f32> -> tensor<256x128x!tt.ptr<f32>, #linear>
    tt.store %ptrs, %scaled : tensor<256x128x!tt.ptr<f32>, #linear>
    tt.return
  }
}

// -----

// The A tile is also operand B of a second MMA, which contracts over the rows
// being sliced, so its users cannot be partitioned with the first MMA. Data
// partitioning is skipped rather than slicing the descriptor block alone.

// WARN: warning: skipping data partitioning because a register user of this descriptor load cannot be partitioned with it
// CHECK-LABEL: @descriptor_load_feeds_mma_and_contraction_over_m(%arg0: !tt.tensordesc<256x64xbf16
// CHECK: tt.descriptor_load {{.*}} -> tensor<256x64xbf16
// CHECK: ttng.tc_gen5_mma {{.*}}!ttg.memdesc<256x128xf32
// CHECK: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [128, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @descriptor_load_feeds_mma_and_contraction_over_m(
      %a_desc: !tt.tensordesc<256x64xbf16, #shared>,
      %b_desc: !tt.tensordesc<128x64xbf16, #shared>,
      %x: !ttg.memdesc<128x256xbf16, #shared, #smem>,
      %out: !tt.ptr<f32>, %k_tiles: i32, %m_off: i32, %n_off: i32) {
    %false = arith.constant false
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #linear>
    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc2, %acc2_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %zero_acc, %acc[%acc_tok], %true : tensor<256x128xf32, #linear> -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %loop:3 = scf.for %ki = %c0 to %k_tiles step %c1 iter_args(%use_acc = %false, %tok = %init_tok, %tok2 = %acc2_tok) -> (i1, !ttg.async.token, !ttg.async.token) : i32 {
      %k = arith.muli %ki, %c64 : i32
      %a = tt.descriptor_load %a_desc[%m_off, %k] : !tt.tensordesc<256x64xbf16, #shared> -> tensor<256x64xbf16, #blocked>
      %a_smem = ttg.local_alloc %a : (tensor<256x64xbf16, #blocked>) -> !ttg.memdesc<256x64xbf16, #shared, #smem>
      %a_rhs = ttg.local_alloc %a : (tensor<256x64xbf16, #blocked>) -> !ttg.memdesc<256x64xbf16, #shared, #smem>
      %b = tt.descriptor_load %b_desc[%n_off, %k] : !tt.tensordesc<128x64xbf16, #shared> -> tensor<128x64xbf16, #blocked>
      %b_smem = ttg.local_alloc %b : (tensor<128x64xbf16, #blocked>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
      %b_t = ttg.memdesc_trans %b_smem {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem> -> !ttg.memdesc<64x128xbf16, #shared2, #smem>
      %mma_tok = ttng.tc_gen5_mma %a_smem, %b_t, %acc[%tok], %use_acc, %true : !ttg.memdesc<256x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared2, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %mma2_tok = ttng.tc_gen5_mma %x, %a_rhs, %acc2[%tok2], %use_acc, %true : !ttg.memdesc<128x256xbf16, #shared, #smem>, !ttg.memdesc<256x64xbf16, #shared, #smem>, !ttg.memdesc<128x64xf32, #tmem2, #ttng.tensor_memory, mutable>
      scf.yield %true, %mma_tok, %mma2_tok : i1, !ttg.async.token, !ttg.async.token
    } {tt.data_partition_factor = 2 : i32, tt.warp_specialize}
    %res, %res_tok = ttng.tmem_load %acc[%loop#1] : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<256x128x!tt.ptr<f32>, #linear>
    tt.store %ptrs, %res : tensor<256x128x!tt.ptr<f32>, #linear>
    tt.return
  }
}
