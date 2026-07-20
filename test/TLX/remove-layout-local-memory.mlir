// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s

// Test that redundant layout conversion after local_load is removed

// CHECK: #[[$COALESCED:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: @local_load_coalesce
// CHECK: ttg.local_load %{{.*}} {ttg.amdg.syncedViaAsyncWait = true} : {{.*}} -> tensor<128x64xf16, #[[$COALESCED]]>
// CHECK-NOT: ttg.convert_layout
// CHECK: ttg.local_store

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func @local_load_coalesce(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem>, %arg1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>) {
  %0 = ttg.local_load %arg0 {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked1>
  %1 = ttg.convert_layout %0 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #blocked>
  ttg.local_store %1, %arg1 : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  tt.return
}
}

// -----

// Test that a store-only conversion through a non-reordering reshape is pruned:
//
//   local_store(reshape(convert_layout(x)), dst)
//
// becomes:
//
//   local_store(reshape(x), dst)
//
// CHECK-LABEL: @local_store_reshape_convert
// CHECK-SAME: %[[ARG:.*]]: tensor<32x4x32xf32, #[[$SRC:[^>]+]]>
// CHECK-NOT: ttg.convert_layout
// CHECK: %[[RESHAPE:.*]] = tt.reshape %[[ARG]] : tensor<32x4x32xf32, #[[$SRC]]> -> tensor<32x128xf32, #[[$DIRECT:.*]]>
// CHECK-NEXT: ttg.local_store %[[RESHAPE]], %{{.*}} {async_task_id = array<i32: 7>, loop.cluster = 2 : i32, loop.stage = 3 : i32} : tensor<32x128xf32, #[[$DIRECT]]> -> !ttg.memdesc<32x128xf32, #{{.*}}, #{{.*}}, mutable>
// CHECK-NEXT: tt.return

#linear_src = #ttg.linear<{register = [[1, 0, 0], [0, 0, 8], [8, 0, 0], [16, 0, 0], [0, 0, 16]], lane = [[2, 0, 0], [4, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 4]], warp = [[0, 1, 0], [0, 2, 0]], block = []}>
#linear_cvt = #ttg.linear<{register = [[1, 0, 0], [8, 0, 0], [16, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16]], lane = [[2, 0, 0], [4, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], warp = [[0, 1, 0], [0, 2, 0]], block = []}>
#linear_cvt_flat = #ttg.linear<{register = [[1, 0], [8, 0], [16, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[2, 0], [4, 0], [0, 0], [0, 0], [0, 0]], warp = [[0, 32], [0, 64]], block = []}>
#shared_store = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem_store = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  tt.func @local_store_reshape_convert(
      %arg0: tensor<32x4x32xf32, #linear_src>,
      %arg1: !ttg.memdesc<32x128xf32, #shared_store, #smem_store, mutable>) {
    %cvt = ttg.convert_layout %arg0 : tensor<32x4x32xf32, #linear_src> -> tensor<32x4x32xf32, #linear_cvt>
    %reshape = tt.reshape %cvt : tensor<32x4x32xf32, #linear_cvt> -> tensor<32x128xf32, #linear_cvt_flat>
    ttg.local_store %reshape, %arg1 {async_task_id = array<i32: 7>, loop.cluster = 2 : i32, loop.stage = 3 : i32} : tensor<32x128xf32, #linear_cvt_flat> -> !ttg.memdesc<32x128xf32, #shared_store, #smem_store, mutable>
    tt.return
  }
}

// -----

// Test layout conflict resolution when both tmem_load and local_load are in the
// same kernel with different layouts. The pass should prefer TMEM's layout with
// larger sizePerThread ([1, 128], score=128) for better memory access efficiency.
//
// After the pass, the larger layout ([1, 128]) should be selected for both loads,
// eliminating the need for intermediate convert_layout ops.

// CHECK: #[[$TMEM_LAYOUT:.*]] = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK-LABEL: @tmem_and_local_load_conflict_resolution
// Both loads should use the TMEM layout with higher score [1, 128]
// CHECK: ttng.tmem_load %{{.*}} -> tensor<128x128xf32, #[[$TMEM_LAYOUT]]>
// CHECK: ttg.local_load %{{.*}} -> tensor<128x128xbf16, #[[$TMEM_LAYOUT]]>
// The convert_layout to the original common layout should still exist at the end
// CHECK: ttg.convert_layout

#blocked_tmem = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_common = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_smem = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem1 = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
tt.func @tmem_and_local_load_conflict_resolution(
    %tmem_buf: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    %smem_buf: !ttg.memdesc<128x128xbf16, #shared1, #smem1>) -> tensor<128x128xf32, #blocked_common> {
  // TMEM load with large sizePerThread [1, 128], score = 128
  %result = ttng.tmem_load %tmem_buf : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_tmem>
  %result_cvt = ttg.convert_layout %result : tensor<128x128xf32, #blocked_tmem> -> tensor<128x128xf32, #blocked_common>
  // SMEM local_load with small sizePerThread [1, 8], score = 8
  %y = ttg.local_load %smem_buf : !ttg.memdesc<128x128xbf16, #shared1, #smem1> -> tensor<128x128xbf16, #blocked_smem>
  %y_cvt = ttg.convert_layout %y : tensor<128x128xbf16, #blocked_smem> -> tensor<128x128xbf16, #blocked_common>
  // Add them together (requires same layout)
  %y_ext = arith.extf %y_cvt : tensor<128x128xbf16, #blocked_common> to tensor<128x128xf32, #blocked_common>
  %z = arith.addf %result_cvt, %y_ext : tensor<128x128xf32, #blocked_common>
  tt.return %z : tensor<128x128xf32, #blocked_common>
}
}

// -----

// Test that tmem_load's linear layout takes priority over local_load's blocked
// layout. tmem_load produces a hardware-fixed linear layout that cannot be
// changed, while local_load can adapt to any layout. Preferring the linear
// layout avoids a convert_layout that would consume shared memory.

// CHECK: #[[$LINEAR:.*]] = #ttg.linear
// CHECK-LABEL: @tmem_linear_layout_priority
// CHECK: ttng.tmem_load {{.*}} -> tensor<64x128xf32, #[[$LINEAR]]>
// CHECK: ttg.local_load {{.*}} -> tensor<64x128xbf16, #[[$LINEAR]]>
// CHECK-NOT: ttg.convert_layout
// CHECK: arith.addf {{.*}} : tensor<64x128xf32, #[[$LINEAR]]>
// CHECK: ttg.local_store

#linear_tmem = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0], [0, 64]], block = []}>
#blocked_smem2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared_nv = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem2 = #ttg.shared_memory
#tmem2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_linear_layout_priority(%arg_o: !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable>, %arg_res: !ttg.memdesc<64x128xbf16, #shared_nv, #smem2, mutable>, %arg_out: !ttg.memdesc<64x128xbf16, #shared_nv, #smem2, mutable>) {
    %cst_eps = arith.constant dense<9.99999974E-6> : tensor<64x1xf32, #linear_tmem>
    %o = ttng.tmem_load %arg_o : !ttg.memdesc<64x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear_tmem>
    %sq = arith.mulf %o, %o : tensor<64x128xf32, #linear_tmem>
    %sum = "tt.reduce"(%sq) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<64x128xf32, #linear_tmem>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear_tmem}>>
    %sum_exp = tt.expand_dims %sum {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear_tmem}>> -> tensor<64x1xf32, #linear_tmem>
    %sum_eps = arith.addf %sum_exp, %cst_eps : tensor<64x1xf32, #linear_tmem>
    %rrms = tt.extern_elementwise %sum_eps {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<64x1xf32, #linear_tmem>) -> tensor<64x1xf32, #linear_tmem>
    %rrms_bcast = tt.broadcast %rrms : tensor<64x1xf32, #linear_tmem> -> tensor<64x128xf32, #linear_tmem>
    %result = arith.mulf %o, %rrms_bcast : tensor<64x128xf32, #linear_tmem>
    %result_cvt = ttg.convert_layout %result : tensor<64x128xf32, #linear_tmem> -> tensor<64x128xf32, #blocked_smem2>
    %res = ttg.local_load %arg_res : !ttg.memdesc<64x128xbf16, #shared_nv, #smem2, mutable> -> tensor<64x128xbf16, #blocked_smem2>
    %res_f32 = arith.extf %res : tensor<64x128xbf16, #blocked_smem2> to tensor<64x128xf32, #blocked_smem2>
    %add = arith.addf %result_cvt, %res_f32 : tensor<64x128xf32, #blocked_smem2>
    %out = arith.truncf %add : tensor<64x128xf32, #blocked_smem2> to tensor<64x128xbf16, #blocked_smem2>
    ttg.local_store %out, %arg_out : tensor<64x128xbf16, #blocked_smem2> -> !ttg.memdesc<64x128xbf16, #shared_nv, #smem2, mutable>
    tt.return
  }
}

// -----

// Explicit TLX local-memory reads are structural: a following linear layout
// requirement must be folded into local_load, including through rank-changing
// reshape and transpose chains that feed an MMA.  Anchoring the initial blocked
// loads here would leave shared-memory convert_layout traffic on every dot
// operand.

// CHECK-LABEL: @explicit_local_load_to_dot
// CHECK: %[[A_LOAD:.*]] = ttg.local_load %{{.*}} : {{.*}} -> tensor<1x1x2x1x16x32xf16, #[[$A_LOAD_LAYOUT:.*]]>
// CHECK-NEXT: %[[A:.*]] = tt.reshape %[[A_LOAD]] : tensor<1x1x2x1x16x32xf16, #[[$A_LOAD_LAYOUT]]> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$MMA:.*]], kWidth = 8}>>
// CHECK: %[[B_LOAD:.*]] = ttg.local_load %{{.*}} : {{.*}} -> tensor<1x1x4x1x32x16xf16, #[[$B_LOAD_LAYOUT:.*]]>
// CHECK-NEXT: %[[B_TRANS:.*]] = tt.trans %[[B_LOAD]] {order = array<i32: 0, 1, 4, 3, 2, 5>} : tensor<1x1x4x1x32x16xf16, #[[$B_LOAD_LAYOUT]]> -> tensor<1x1x32x1x4x16xf16, #[[$B_TRANS_LAYOUT:.*]]>
// CHECK-NEXT: %[[B:.*]] = tt.reshape %[[B_TRANS]] : tensor<1x1x32x1x4x16xf16, #[[$B_TRANS_LAYOUT]]> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$MMA]], kWidth = 8}>>
// CHECK-NOT: ttg.convert_layout
// CHECK: tt.dot %[[A]], %[[B]], %{{.*}}

#explicit_blocked_a = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 1, 2, 32], warpsPerCTA = [1, 1, 1, 1, 8, 1], order = [5, 4, 3, 2, 1, 0]}>
#explicit_blocked_b = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 1, 4, 16], warpsPerCTA = [1, 1, 1, 1, 8, 1], order = [5, 4, 3, 2, 1, 0]}>
#explicit_linear_a = #ttg.linear<{register = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 4]], lane = [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 0, 16]], warp = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], block = []}>
#explicit_linear_a_flat = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]], warp = [[0, 0], [0, 0], [16, 0]], block = []}>
#explicit_linear_b = #ttg.linear<{register = [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 4, 0]], lane = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 16, 0]], warp = [[0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0]], block = []}>
#explicit_linear_b_trans = #ttg.linear<{register = [[0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0], [0, 0, 4, 0, 0, 0]], lane = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 8], [0, 0, 8, 0, 0, 0], [0, 0, 16, 0, 0, 0]], warp = [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0]], block = []}>
#explicit_linear_b_flat = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [16, 0]], warp = [[0, 16], [0, 32], [0, 0]], block = []}>
#explicit_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}>
#explicit_dot_a = #ttg.dot_op<{opIdx = 0, parent = #explicit_mma, kWidth = 8}>
#explicit_dot_b = #ttg.dot_op<{opIdx = 1, parent = #explicit_mma, kWidth = 8}>
#explicit_shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [5, 4, 3, 2, 1, 0]}>
#explicit_shared_b = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [4, 5, 3, 2, 1, 0]}>
#explicit_smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @explicit_local_load_to_dot(
      %a_buf: !ttg.memdesc<1x1x2x1x16x32xf16, #explicit_shared_a, #explicit_smem, mutable, 2x2x2x8x16x32>,
      %b_buf: !ttg.memdesc<1x1x4x1x32x16xf16, #explicit_shared_b, #explicit_smem, mutable, 2x2x4x4x32x16>) -> tensor<32x64xf32, #explicit_mma> {
    %acc = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #explicit_mma>
    %a_load = ttg.local_load %a_buf : !ttg.memdesc<1x1x2x1x16x32xf16, #explicit_shared_a, #explicit_smem, mutable, 2x2x2x8x16x32> -> tensor<1x1x2x1x16x32xf16, #explicit_blocked_a>
    %a_linear = ttg.convert_layout %a_load : tensor<1x1x2x1x16x32xf16, #explicit_blocked_a> -> tensor<1x1x2x1x16x32xf16, #explicit_linear_a>
    %a_reshape = tt.reshape %a_linear : tensor<1x1x2x1x16x32xf16, #explicit_linear_a> -> tensor<32x32xf16, #explicit_linear_a_flat>
    %a = ttg.convert_layout %a_reshape : tensor<32x32xf16, #explicit_linear_a_flat> -> tensor<32x32xf16, #explicit_dot_a>
    %b_load = ttg.local_load %b_buf : !ttg.memdesc<1x1x4x1x32x16xf16, #explicit_shared_b, #explicit_smem, mutable, 2x2x4x4x32x16> -> tensor<1x1x4x1x32x16xf16, #explicit_blocked_b>
    %b_linear = ttg.convert_layout %b_load : tensor<1x1x4x1x32x16xf16, #explicit_blocked_b> -> tensor<1x1x4x1x32x16xf16, #explicit_linear_b>
    %b_trans = tt.trans %b_linear {order = array<i32: 0, 1, 4, 3, 2, 5>} : tensor<1x1x4x1x32x16xf16, #explicit_linear_b> -> tensor<1x1x32x1x4x16xf16, #explicit_linear_b_trans>
    %b_reshape = tt.reshape %b_trans : tensor<1x1x32x1x4x16xf16, #explicit_linear_b_trans> -> tensor<32x64xf16, #explicit_linear_b_flat>
    %b = ttg.convert_layout %b_reshape : tensor<32x64xf16, #explicit_linear_b_flat> -> tensor<32x64xf16, #explicit_dot_b>
    %result = tt.dot %a, %b, %acc : tensor<32x32xf16, #explicit_dot_a> * tensor<32x64xf16, #explicit_dot_b> -> tensor<32x64xf32, #explicit_mma>
    tt.return %result : tensor<32x64xf32, #explicit_mma>
  }
}
