// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s

// A user layout requirement is a semantic boundary, not a transparent
// convert_layout.  RemoveLayoutConversions may insert physical conversions on
// either side of the boundary, but it must not propagate the TMEM-store layout
// backward through the user requirement.

#src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>
#user_col = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#tmem_compatible = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], warp = [[16, 0], [32, 0], [0, 16]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-DAG: #[[$USER_COL:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
  // CHECK-DAG: #[[$TMEM_COMPAT:.*]] = #ttg.linear
  // CHECK-LABEL: tt.func @require_layout_blocks_tmem_store_propagation
  tt.func @require_layout_blocks_tmem_store_propagation(
      %arg0: tensor<64x32xf32, #src>,
      %acc: !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    // CHECK: %[[TO_USER:.*]] = ttg.convert_layout %{{.*}} : tensor<64x32xf32, #{{.*}}> -> tensor<64x32xf32, #[[$USER_COL]]>
    // CHECK: %[[REQ:.*]] = ttg.require_layout %[[TO_USER]] : tensor<64x32xf32, #[[$USER_COL]]> -> tensor<64x32xf32, #[[$USER_COL]]>
    %required = ttg.require_layout %arg0 : tensor<64x32xf32, #src> -> tensor<64x32xf32, #user_col>
    // CHECK: %[[TO_TMEM:.*]] = ttg.convert_layout %[[REQ]] : tensor<64x32xf32, #[[$USER_COL]]> -> tensor<64x32xf32, #[[$TMEM_COMPAT]]>
    %for_store = ttg.convert_layout %required : tensor<64x32xf32, #user_col> -> tensor<64x32xf32, #tmem_compatible>
    // CHECK: ttng.tmem_store %[[TO_TMEM]], %{{.*}}, %{{.*}} : tensor<64x32xf32, #[[$TMEM_COMPAT]]> -> !ttg.memdesc<64x32xf32, #{{.*}}, #ttng.tensor_memory, mutable>
    ttng.tmem_store %for_store, %acc, %true : tensor<64x32xf32, #tmem_compatible> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// AMD FA backward epilogues also use the opposite packing direction:
// join two D-slices, permute the packed axis, then reshape back to 2D.  Both
// join operands may carry user requirements and must not be rewritten to their
// original source layouts.

#join_src_piece = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>
#join_user_piece = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#join_packed = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
#join_trans = #ttg.blocked<{sizePerThread = [1, 2, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [4, 1, 2], order = [1, 2, 0]}>
#join_flat = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-DAG: #[[$JOIN_USER_PIECE:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
  // CHECK-DAG: #[[$JOIN_PACKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
  // CHECK-LABEL: tt.func @require_layout_blocks_join_trans_reshape_rewrite
  tt.func @require_layout_blocks_join_trans_reshape_rewrite(
      %lo_src: tensor<128x64xf32, #join_src_piece>,
      %hi_src: tensor<128x64xf32, #join_src_piece>)
      -> tensor<128x128xf32, #join_flat> {
    // CHECK-DAG: %[[JOIN_LO_CVT:.*]] = ttg.convert_layout %arg0 : tensor<128x64xf32, #{{.*}}> -> tensor<128x64xf32, #[[$JOIN_USER_PIECE]]>
    // CHECK-DAG: %[[JOIN_HI_CVT:.*]] = ttg.convert_layout %arg1 : tensor<128x64xf32, #{{.*}}> -> tensor<128x64xf32, #[[$JOIN_USER_PIECE]]>
    // CHECK: %[[JOIN_LO_REQ:.*]] = ttg.require_layout %[[JOIN_LO_CVT]] : tensor<128x64xf32, #[[$JOIN_USER_PIECE]]> -> tensor<128x64xf32, #[[$JOIN_USER_PIECE]]>
    %lo = ttg.require_layout %lo_src : tensor<128x64xf32, #join_src_piece> -> tensor<128x64xf32, #join_user_piece>
    // CHECK: %[[JOIN_HI_REQ:.*]] = ttg.require_layout %[[JOIN_HI_CVT]] : tensor<128x64xf32, #[[$JOIN_USER_PIECE]]> -> tensor<128x64xf32, #[[$JOIN_USER_PIECE]]>
    %hi = ttg.require_layout %hi_src : tensor<128x64xf32, #join_src_piece> -> tensor<128x64xf32, #join_user_piece>
    // CHECK: %[[JOINED:.*]] = tt.join %[[JOIN_LO_REQ]], %[[JOIN_HI_REQ]]
    %joined = tt.join %lo, %hi : tensor<128x64xf32, #join_user_piece> -> tensor<128x64x2xf32, #join_packed>
    // CHECK: %[[JOIN_TRANS:.*]] = tt.trans %[[JOINED]]
    %transposed = tt.trans %joined {order = array<i32: 0, 2, 1>} : tensor<128x64x2xf32, #join_packed> -> tensor<128x2x64xf32, #join_trans>
    // CHECK: %[[JOIN_RESHAPED:.*]] = tt.reshape %[[JOIN_TRANS]]
    %reshaped = tt.reshape %transposed : tensor<128x2x64xf32, #join_trans> -> tensor<128x128xf32, #join_flat>
    tt.return %reshaped : tensor<128x128xf32, #join_flat>
  }
}

// -----

// Gather has source/result layout preferences and canonicalization patterns that
// may look through a convert_layout on the gathered source.  A user requirement
// on that source must remain the operand seen by tt.gather.

#gather_src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
#gather_user = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-DAG: #[[$GATHER_USER:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
  // CHECK-LABEL: tt.func @require_layout_blocks_gather_source_rewrite
  tt.func @require_layout_blocks_gather_source_rewrite(
      %src: tensor<64x64xf32, #gather_src>,
      %idx: tensor<64x64xi32, #gather_user>)
      -> tensor<64x64xf32, #gather_user> {
    // CHECK: %[[GATHER_TO_USER:.*]] = ttg.convert_layout %{{.*}} : tensor<64x64xf32, #{{.*}}> -> tensor<64x64xf32, #[[$GATHER_USER]]>
    // CHECK: %[[GATHER_REQ:.*]] = ttg.require_layout %[[GATHER_TO_USER]] : tensor<64x64xf32, #[[$GATHER_USER]]> -> tensor<64x64xf32, #[[$GATHER_USER]]>
    %required = ttg.require_layout %src : tensor<64x64xf32, #gather_src> -> tensor<64x64xf32, #gather_user>
    // CHECK: %[[GATHERED:.*]] = tt.gather %[[GATHER_REQ]][%{{.*}}]
    %gathered = tt.gather %required[%idx] {axis = 0 : i32} : (tensor<64x64xf32, #gather_user>, tensor<64x64xi32, #gather_user>) -> tensor<64x64xf32, #gather_user>
    tt.return %gathered : tensor<64x64xf32, #gather_user>
  }
}

// -----

// A direct split of a pinned rank-3 tensor exercises SplitOp's
// split(convert_layout(x)) canonicalization without the longer FA reshape chain.
// The split source must remain the user boundary.

#direct_split_src = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 1]], warp = [[32, 0, 0], [64, 0, 0], [16, 0, 0]], block = []}>
#direct_split_user = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
#direct_split_piece = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-DAG: #[[$DIRECT_SPLIT_USER:.*]] = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
  // CHECK-LABEL: tt.func @require_layout_blocks_direct_split_rewrite
  tt.func @require_layout_blocks_direct_split_rewrite(
      %src: tensor<128x64x2xf32, #direct_split_src>)
      -> (tensor<128x64xf32, #direct_split_piece>, tensor<128x64xf32, #direct_split_piece>) {
    // CHECK: %[[DS_TO_USER:.*]] = ttg.convert_layout %{{.*}} : tensor<128x64x2xf32, #{{.*}}> -> tensor<128x64x2xf32, #[[$DIRECT_SPLIT_USER]]>
    // CHECK: %[[DS_REQ:.*]] = ttg.require_layout %[[DS_TO_USER]] : tensor<128x64x2xf32, #[[$DIRECT_SPLIT_USER]]> -> tensor<128x64x2xf32, #[[$DIRECT_SPLIT_USER]]>
    %required = ttg.require_layout %src : tensor<128x64x2xf32, #direct_split_src> -> tensor<128x64x2xf32, #direct_split_user>
    // CHECK: %[[DS_LO:.*]], %[[DS_HI:.*]] = tt.split %[[DS_REQ]]
    %lo, %hi = tt.split %required : tensor<128x64x2xf32, #direct_split_user> -> tensor<128x64xf32, #direct_split_piece>
    tt.return %lo, %hi : tensor<128x64xf32, #direct_split_piece>, tensor<128x64xf32, #direct_split_piece>
  }
}

// -----

// allow_reorder reshape is the broadest shape-changing view form used by
// packing/unpacking helpers.  It may choose a cheaper physical route, but the
// reshape operand must be the user-required layout.

#reshape_any_src = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#reshape_any_user = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [0, 1]}>
#reshape_any_flat = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-DAG: #[[$RESHAPE_ANY_USER:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [0, 1]}>
  // CHECK-LABEL: tt.func @require_layout_blocks_allow_reorder_reshape
  tt.func @require_layout_blocks_allow_reorder_reshape(
      %src: tensor<64x64xf32, #reshape_any_src>)
      -> tensor<4096xf32, #reshape_any_flat> {
    // CHECK: %[[RA_TO_USER:.*]] = ttg.convert_layout %{{.*}} : tensor<64x64xf32, #{{.*}}> -> tensor<64x64xf32, #[[$RESHAPE_ANY_USER]]>
    // CHECK: %[[RA_REQ:.*]] = ttg.require_layout %[[RA_TO_USER]] : tensor<64x64xf32, #[[$RESHAPE_ANY_USER]]> -> tensor<64x64xf32, #[[$RESHAPE_ANY_USER]]>
    %required = ttg.require_layout %src : tensor<64x64xf32, #reshape_any_src> -> tensor<64x64xf32, #reshape_any_user>
    // CHECK: %[[RA_RESHAPE:.*]] = tt.reshape %[[RA_REQ]] allow_reorder
    %reshaped = tt.reshape %required allow_reorder : tensor<64x64xf32, #reshape_any_user> -> tensor<4096xf32, #reshape_any_flat>
    tt.return %reshaped : tensor<4096xf32, #reshape_any_flat>
  }
}

// -----

// Dot operands have very specific encodings and RLC aggressively propagates
// those layouts backward.  A source-level user boundary on an operand must
// still be represented explicitly before the dot consumes it.

#dot_src = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dot_mma, kWidth = 4}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dot_mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx942"} {
  // CHECK-DAG: #[[$DOT_MMA:.*]] = #ttg.amd_mfma
  // CHECK-LABEL: tt.func @require_layout_blocks_dot_operand_rewrite
  tt.func @require_layout_blocks_dot_operand_rewrite(
      %a: tensor<64x32xf16, #dot_src>,
      %b: tensor<32x64xf16, #dot_b>,
      %acc: tensor<64x64xf32, #dot_mma>)
      -> tensor<64x64xf32, #dot_mma> {
    // CHECK: %[[DOT_TO_A:.*]] = ttg.convert_layout %{{.*}} : tensor<64x32xf16, #{{.*}}> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_MMA]], kWidth = 4}>>
    // CHECK: %[[DOT_REQ:.*]] = ttg.require_layout %[[DOT_TO_A]] : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_MMA]], kWidth = 4}>> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_MMA]], kWidth = 4}>>
    %required = ttg.require_layout %a : tensor<64x32xf16, #dot_src> -> tensor<64x32xf16, #dot_a>
    // CHECK: %[[DOT:.*]] = tt.dot %[[DOT_REQ]], %{{.*}}, %{{.*}} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_MMA]], kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DOT_MMA]], kWidth = 4}>> -> tensor<64x64xf32, #[[$DOT_MMA]]>
    %dot = tt.dot %required, %b, %acc, inputPrecision = tf32 : tensor<64x32xf16, #dot_a> * tensor<32x64xf16, #dot_b> -> tensor<64x64xf32, #dot_mma>
    tt.return %dot : tensor<64x64xf32, #dot_mma>
  }
}

// -----

// AMD GEMM/addmm epilogues commonly upcast, add a bias, downcast, and
// materialize in LDS.  The initialized local_alloc is a flexible sink, but the
// cast chain it sees still has to be derived from the user-required layout.

#alloc_src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#alloc_user = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#alloc_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#alloc_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-DAG: #[[$ALLOC_USER:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-LABEL: tt.func @require_layout_blocks_trunc_alloc_epilogue
  tt.func @require_layout_blocks_trunc_alloc_epilogue(
      %src: tensor<64x64xf16, #alloc_src>,
      %bias: tensor<64x64xf32, #alloc_user>)
      -> !ttg.memdesc<64x64xf16, #alloc_shared, #alloc_smem, mutable> {
    // CHECK: %[[ALLOC_TO_USER:.*]] = ttg.convert_layout %{{.*}} : tensor<64x64xf16, #{{.*}}> -> tensor<64x64xf16, #[[$ALLOC_USER]]>
    // CHECK: %[[ALLOC_REQ:.*]] = ttg.require_layout %[[ALLOC_TO_USER]] : tensor<64x64xf16, #[[$ALLOC_USER]]> -> tensor<64x64xf16, #[[$ALLOC_USER]]>
    %required = ttg.require_layout %src : tensor<64x64xf16, #alloc_src> -> tensor<64x64xf16, #alloc_user>
    // CHECK: %[[ALLOC_EXT:.*]] = arith.extf %[[ALLOC_REQ]] : tensor<64x64xf16, #[[$ALLOC_USER]]> to tensor<64x64xf32, #[[$ALLOC_USER]]>
    %ext = arith.extf %required : tensor<64x64xf16, #alloc_user> to tensor<64x64xf32, #alloc_user>
    // CHECK: %[[ALLOC_ADD:.*]] = arith.addf %[[ALLOC_EXT]], %{{.*}} : tensor<64x64xf32, #[[$ALLOC_USER]]>
    %add = arith.addf %ext, %bias : tensor<64x64xf32, #alloc_user>
    // CHECK: %[[ALLOC_TRUNC:.*]] = arith.truncf %[[ALLOC_ADD]] : tensor<64x64xf32, #[[$ALLOC_USER]]> to tensor<64x64xf16, #[[$ALLOC_USER]]>
    %truncated = arith.truncf %add : tensor<64x64xf32, #alloc_user> to tensor<64x64xf16, #alloc_user>
    // CHECK: %[[ALLOC:.*]] = ttg.local_alloc %[[ALLOC_TRUNC]]
    %alloc = ttg.local_alloc %truncated : (tensor<64x64xf16, #alloc_user>) -> !ttg.memdesc<64x64xf16, #alloc_shared, #alloc_smem, mutable>
    tt.return %alloc : !ttg.memdesc<64x64xf16, #alloc_shared, #alloc_smem, mutable>
  }
}

// -----

// AMD FA kernels use long elementwise softmax chains after layout-sensitive
// inputs: where/select masking, maximum, exp2, and output casts.  Remat/hoist
// may clone cheap elementwise ops, but it must not clone through or bypass the
// user layout boundary feeding that chain.

#elt_src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#elt_user = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-DAG: #[[$ELT_USER:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-LABEL: tt.func @require_layout_blocks_elementwise_softmax_chain
  tt.func @require_layout_blocks_elementwise_softmax_chain(
      %src: tensor<64x32xf32, #elt_src>,
      %mask: tensor<64x32xi1, #elt_user>,
      %fallback: tensor<64x32xf32, #elt_user>)
      -> tensor<64x32xf16, #elt_user> {
    // CHECK: %[[ELT_TO_USER:.*]] = ttg.convert_layout %{{.*}} : tensor<64x32xf32, #{{.*}}> -> tensor<64x32xf32, #[[$ELT_USER]]>
    // CHECK: %[[ELT_REQ:.*]] = ttg.require_layout %[[ELT_TO_USER]] : tensor<64x32xf32, #[[$ELT_USER]]> -> tensor<64x32xf32, #[[$ELT_USER]]>
    %required = ttg.require_layout %src : tensor<64x32xf32, #elt_src> -> tensor<64x32xf32, #elt_user>
    // CHECK: %[[MASKED:.*]] = arith.select %{{.*}}, %[[ELT_REQ]], %{{.*}} : tensor<64x32xi1, #[[$ELT_USER]]>, tensor<64x32xf32, #[[$ELT_USER]]>
    %masked = arith.select %mask, %required, %fallback : tensor<64x32xi1, #elt_user>, tensor<64x32xf32, #elt_user>
    // CHECK: %[[MAXED:.*]] = arith.maximumf %[[MASKED]], %{{.*}} : tensor<64x32xf32, #[[$ELT_USER]]>
    %maxed = arith.maximumf %masked, %fallback : tensor<64x32xf32, #elt_user>
    // CHECK: %[[EXP:.*]] = math.exp2 %[[MAXED]] : tensor<64x32xf32, #[[$ELT_USER]]>
    %exp = math.exp2 %maxed : tensor<64x32xf32, #elt_user>
    // CHECK: %[[TRUNC:.*]] = arith.truncf %[[EXP]] : tensor<64x32xf32, #[[$ELT_USER]]> to tensor<64x32xf16, #[[$ELT_USER]]>
    %trunc = arith.truncf %exp : tensor<64x32xf32, #elt_user> to tensor<64x32xf16, #elt_user>
    tt.return %trunc : tensor<64x32xf16, #elt_user>
  }
}

// -----

// Rank-changing producer chains are another common source of convert sinking.
// The expand_dims/broadcast chain must start from the user-required slice
// layout, not from the original one-dimensional source layout.

#bcast_src_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#bcast_src = #ttg.slice<{dim = 1, parent = #bcast_src_parent}>
#bcast_user_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#bcast_user = #ttg.slice<{dim = 1, parent = #bcast_user_parent}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-DAG: #[[$BCAST_USER_PARENT:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
  // CHECK-LABEL: tt.func @require_layout_blocks_expand_broadcast_rewrite
  tt.func @require_layout_blocks_expand_broadcast_rewrite(
      %src: tensor<64xf32, #bcast_src>)
      -> tensor<64x64xf32, #bcast_user_parent> {
    // CHECK: %[[BCAST_TO_USER:.*]] = ttg.convert_layout %{{.*}} : tensor<64xf32, #{{.*}}> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[$BCAST_USER_PARENT]]}>>
    // CHECK: %[[BCAST_REQ:.*]] = ttg.require_layout %[[BCAST_TO_USER]] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[$BCAST_USER_PARENT]]}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[$BCAST_USER_PARENT]]}>>
    %required = ttg.require_layout %src : tensor<64xf32, #bcast_src> -> tensor<64xf32, #bcast_user>
    // CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[BCAST_REQ]]
    %expanded = tt.expand_dims %required {axis = 1 : i32} : tensor<64xf32, #bcast_user> -> tensor<64x1xf32, #bcast_user_parent>
    // CHECK: %[[BROADCASTED:.*]] = tt.broadcast %[[EXPANDED]]
    %broadcasted = tt.broadcast %expanded : tensor<64x1xf32, #bcast_user_parent> -> tensor<64x64xf32, #bcast_user_parent>
    tt.return %broadcasted : tensor<64x64xf32, #bcast_user_parent>
  }
}

// -----

// The local-store cleanup pass has a special
// local_store(reshape(convert_layout(x))) rewrite.  It may prune ordinary
// physical conversions, but it must not sink the store through a user
// requirement and reshape the original source directly.

#store_src = #ttg.linear<{register = [[1, 0, 0], [0, 0, 8], [8, 0, 0], [16, 0, 0], [0, 0, 16]], lane = [[2, 0, 0], [4, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 4]], warp = [[0, 1, 0], [0, 2, 0]], block = []}>
#store_user = #ttg.linear<{register = [[1, 0, 0], [8, 0, 0], [16, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16]], lane = [[2, 0, 0], [4, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], warp = [[0, 1, 0], [0, 2, 0]], block = []}>
#store_flat = #ttg.linear<{register = [[1, 0], [8, 0], [16, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[2, 0], [4, 0], [0, 0], [0, 0], [0, 0]], warp = [[0, 32], [0, 64]], block = []}>
#store_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#store_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @require_layout_blocks_store_reshape_cleanup
  tt.func @require_layout_blocks_store_reshape_cleanup(
      %src: tensor<32x4x32xf32, #store_src>,
      %dst: !ttg.memdesc<32x128xf32, #store_shared, #store_smem, mutable>) {
    // CHECK: %[[STORE_TO_USER:.*]] = ttg.convert_layout %{{.*}} : tensor<32x4x32xf32, #{{.*}}> -> tensor<32x4x32xf32, #{{.*}}>
    // CHECK: %[[STORE_REQ:.*]] = ttg.require_layout %[[STORE_TO_USER]] : tensor<32x4x32xf32, #{{.*}}> -> tensor<32x4x32xf32, #{{.*}}>
    %required = ttg.require_layout %src : tensor<32x4x32xf32, #store_src> -> tensor<32x4x32xf32, #store_user>
    // CHECK: %[[STORE_RESHAPE:.*]] = tt.reshape %[[STORE_REQ]]
    %reshaped = tt.reshape %required : tensor<32x4x32xf32, #store_user> -> tensor<32x128xf32, #store_flat>
    // CHECK: ttg.local_store %[[STORE_RESHAPE]], %{{.*}} : tensor<32x128xf32, #{{.*}}> -> !ttg.memdesc<32x128xf32, #{{.*}}, #{{.*}}, mutable>
    ttg.local_store %reshaped, %dst : tensor<32x128xf32, #store_flat> -> !ttg.memdesc<32x128xf32, #store_shared, #store_smem, mutable>
    tt.return
  }
}

// -----

// A reduction can strongly prefer a different operand/result layout.  The
// reduce rewrite may materialize a physical conversion before the requirement,
// but it must not bypass the user boundary and feed the original value directly
// to tt.reduce.

#red_src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>
#red_user = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#red_row = #ttg.slice<{dim = 1, parent = #red_user}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-DAG: #[[$RED_USER:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
  // CHECK-LABEL: tt.func @require_layout_blocks_reduce_operand_rewrite
  tt.func @require_layout_blocks_reduce_operand_rewrite(
      %arg0: tensor<64x32xf32, #red_src>)
      -> tensor<64xf32, #red_row> {
    // CHECK: %[[RED_TO_USER:.*]] = ttg.convert_layout %{{.*}} : tensor<64x32xf32, #{{.*}}> -> tensor<64x32xf32, #[[$RED_USER]]>
    // CHECK: %[[RED_REQ:.*]] = ttg.require_layout %[[RED_TO_USER]] : tensor<64x32xf32, #[[$RED_USER]]> -> tensor<64x32xf32, #[[$RED_USER]]>
    %required = ttg.require_layout %arg0 : tensor<64x32xf32, #red_src> -> tensor<64x32xf32, #red_user>
    // CHECK: %[[RED_SUM:.*]] = "tt.reduce"(%[[RED_REQ]])
    // CHECK: }) : (tensor<64x32xf32, #[[$RED_USER]]>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #[[$RED_USER]]}>>
    %sum = "tt.reduce"(%required) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %next = arith.addf %lhs, %rhs : f32
      tt.reduce.return %next : f32
    }) : (tensor<64x32xf32, #red_user>) -> tensor<64xf32, #red_row>
    tt.return %sum : tensor<64xf32, #red_row>
  }
}

// -----

// Reshape/trans/split/join are common in FA subtile helpers.  They rewrite
// encodings aggressively, but the first reshape must still start from the
// user-required layout rather than from the original source layout.

#split_src = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>
#split_user = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[32, 0], [64, 0], [16, 0]], block = []}>
#split_reshape = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 1, 0]], warp = [[32, 0, 0], [64, 0, 0], [16, 0, 0]], block = []}>
#split_trans = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 1]], warp = [[32, 0, 0], [64, 0, 0], [16, 0, 0]], block = []}>
#split_blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
#split_piece = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-DAG: #[[$SPLIT_USER:.*]] = #ttg.linear
  // CHECK-DAG: #[[$SPLIT_BLOCKED3D:.*]] = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
  // CHECK-LABEL: tt.func @require_layout_blocks_split_join_rewrite
  tt.func @require_layout_blocks_split_join_rewrite(
      %arg0: tensor<128x128xf32, #split_src>)
      -> tensor<128x64x2xf32, #split_blocked3d> {
    // CHECK: %[[SPLIT_TO_USER:.*]] = ttg.convert_layout %{{.*}} : tensor<128x128xf32, #{{.*}}> -> tensor<128x128xf32, #[[$SPLIT_USER]]>
    // CHECK: %[[SPLIT_REQ:.*]] = ttg.require_layout %[[SPLIT_TO_USER]] : tensor<128x128xf32, #[[$SPLIT_USER]]> -> tensor<128x128xf32, #[[$SPLIT_USER]]>
    %required = ttg.require_layout %arg0 : tensor<128x128xf32, #split_src> -> tensor<128x128xf32, #split_user>
    // CHECK: %[[RESHAPED:.*]] = tt.reshape %[[SPLIT_REQ]]
    %reshaped = tt.reshape %required : tensor<128x128xf32, #split_user> -> tensor<128x2x64xf32, #split_reshape>
    // CHECK: %[[TRANSPOSED:.*]] = tt.trans %[[RESHAPED]]
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #split_reshape> -> tensor<128x64x2xf32, #split_trans>
    // CHECK: %[[SPLIT_BLOCKED:.*]] = ttg.convert_layout %[[TRANSPOSED]] : tensor<128x64x2xf32, #{{.*}}> -> tensor<128x64x2xf32, #[[$SPLIT_BLOCKED3D]]>
    %blocked = ttg.convert_layout %transposed : tensor<128x64x2xf32, #split_trans> -> tensor<128x64x2xf32, #split_blocked3d>
    // CHECK: %[[LO:.*]], %[[HI:.*]] = tt.split %[[SPLIT_BLOCKED]]
    %lo, %hi = tt.split %blocked : tensor<128x64x2xf32, #split_blocked3d> -> tensor<128x64xf32, #split_piece>
    // CHECK: %[[JOINED:.*]] = tt.join %[[LO]], %[[HI]]
    %joined = tt.join %lo, %hi : tensor<128x64xf32, #split_piece> -> tensor<128x64x2xf32, #split_blocked3d>
    tt.return %joined : tensor<128x64x2xf32, #split_blocked3d>
  }
}

// -----

// tt.cat may rewrite both operands to match the result layout.  Each input
// boundary must still be visible before concatenation, not bypassed by a
// backward-propagated result encoding.

#cat_src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#cat_user = #ttg.linear<{register = [[1], [16]], lane = [[0], [0], [2], [4], [8]], warp = [[0], [0]], block = []}>
#cat_result = #ttg.linear<{register = [[1], [16]], lane = [[0], [0], [2], [4], [8]], warp = [[0], [0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-DAG: #[[$CAT_USER:.*]] = #ttg.linear
  // CHECK-LABEL: tt.func @require_layout_blocks_cat_operand_rewrite
  tt.func @require_layout_blocks_cat_operand_rewrite(
      %lhs_src: tensor<16xf32, #cat_src>,
      %rhs_src: tensor<16xf32, #cat_src>)
      -> tensor<32xf32, #cat_result> {
    // CHECK-DAG: %[[CAT_LHS_CVT:.*]] = ttg.convert_layout %arg0 : tensor<16xf32, #{{.*}}> -> tensor<16xf32, #[[$CAT_USER]]>
    // CHECK-DAG: %[[CAT_RHS_CVT:.*]] = ttg.convert_layout %arg1 : tensor<16xf32, #{{.*}}> -> tensor<16xf32, #[[$CAT_USER]]>
    // CHECK: %[[CAT_LHS_REQ:.*]] = ttg.require_layout %[[CAT_LHS_CVT]] : tensor<16xf32, #[[$CAT_USER]]> -> tensor<16xf32, #[[$CAT_USER]]>
    %lhs = ttg.require_layout %lhs_src : tensor<16xf32, #cat_src> -> tensor<16xf32, #cat_user>
    // CHECK: %[[CAT_RHS_REQ:.*]] = ttg.require_layout %[[CAT_RHS_CVT]] : tensor<16xf32, #[[$CAT_USER]]> -> tensor<16xf32, #[[$CAT_USER]]>
    %rhs = ttg.require_layout %rhs_src : tensor<16xf32, #cat_src> -> tensor<16xf32, #cat_user>
    // CHECK: %[[CAT:.*]] = tt.cat %[[CAT_LHS_REQ]], %[[CAT_RHS_REQ]]
    %cat = tt.cat %lhs, %rhs : tensor<16xf32, #cat_user> -> tensor<32xf32, #cat_result>
    tt.return %cat : tensor<32xf32, #cat_result>
  }
}

// -----

// tt.scan has a region and reduction-like layout preferences.  The scan input
// must remain anchored at the user-required layout even if the pass can choose
// a cheaper layout for surrounding arithmetic.

#scan_src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
#scan_user = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  // CHECK-DAG: #[[$SCAN_USER:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
  // CHECK-LABEL: tt.func @require_layout_blocks_scan_operand_rewrite
  tt.func @require_layout_blocks_scan_operand_rewrite(
      %src: tensor<64x32xf32, #scan_src>)
      -> tensor<64x32xf32, #scan_user> {
    // CHECK: %[[SCAN_TO_USER:.*]] = ttg.convert_layout %{{.*}} : tensor<64x32xf32, #{{.*}}> -> tensor<64x32xf32, #[[$SCAN_USER]]>
    // CHECK: %[[SCAN_REQ:.*]] = ttg.require_layout %[[SCAN_TO_USER]] : tensor<64x32xf32, #[[$SCAN_USER]]> -> tensor<64x32xf32, #[[$SCAN_USER]]>
    %required = ttg.require_layout %src : tensor<64x32xf32, #scan_src> -> tensor<64x32xf32, #scan_user>
    // CHECK: %[[SCAN:.*]] = "tt.scan"(%[[SCAN_REQ]])
    // CHECK: }) : (tensor<64x32xf32, #[[$SCAN_USER]]>) -> tensor<64x32xf32, #[[$SCAN_USER]]>
    %scan = "tt.scan"(%required) <{axis = 1 : i32, reverse = false}>({
    ^bb0(%lhs: f32, %rhs: f32):
      %next = arith.addf %lhs, %rhs : f32
      tt.scan.return %next : f32
    }) : (tensor<64x32xf32, #scan_user>) -> tensor<64x32xf32, #scan_user>
    tt.return %scan : tensor<64x32xf32, #scan_user>
  }
}
