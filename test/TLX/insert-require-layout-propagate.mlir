// RUN: triton-opt --split-input-file --tlx-insert-require-layout --tlx-propagate-layout %s | FileCheck %s
// RUN: triton-opt --split-input-file --tritongpu-remove-layout-conversions %s | FileCheck %s --check-prefix=UPSTREAM

// Test 1: direct local_load -> dot path.
// After both passes, the shared layout requirement should be propagated onto
// the memdesc producer chain and the local_load results should carry the final
// dot operand encodings directly.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
// CHECK-DAG: #{{.*}} = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
// CHECK-DAG: #{{.*}} = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @local_store_local_load_dot
  tt.func public @local_store_local_load_dot(%arg0: !tt.ptr<f16>, %arg1: tensor<64x32x!tt.ptr<f16>, #blocked>, %arg2: tensor<32x64x!tt.ptr<f16>, #blocked>) -> tensor<64x64xf32, #mma> {
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared1, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %buf_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    %buf_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    %a_val = tt.load %arg1 : tensor<64x32x!tt.ptr<f16>, #blocked>
    %b_val = tt.load %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked>
    ttg.local_store %a_val, %buf_a : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    ttg.local_store %b_val, %buf_b : tensor<32x64xf16, #blocked> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    // CHECK: %[[A_LOAD:.*]] = ttg.local_load %{{.*}} : !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %a = ttg.local_load %buf_a : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // CHECK: %[[B_LOAD:.*]] = ttg.local_load %{{.*}} : !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %b = ttg.local_load %buf_b : !ttg.memdesc<32x64xf16, #shared1, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %acc = ttg.convert_layout %cst : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #mma>
    %a_dot = ttg.convert_layout %a : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %b_dot = ttg.convert_layout %b : tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    // CHECK: tt.dot %[[A_LOAD]], %[[B_LOAD]], %{{.*}}
    %dot = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return %dot : tensor<64x64xf32, #mma>
  }
}

// -----
// Test 2: local_load feeds scf.for iter_args that eventually reach a dot.
// The TLX pipeline should rewrite the prologue and loop-body loads and make the
// loop carry the final dot encodings directly. For comparison, the upstream
// remove-layout-conversions pass still leaves explicit dot converts on the
// loop-carried values.

#blocked_1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared_1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// CHECK-DAG: #{{.*}} = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem_1 = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // UPSTREAM-LABEL: @local_load_through_iter_arg
  // UPSTREAM: scf.for {{.*}} iter_args({{.*}}, %[[ARG_A:.*]] = %{{.*}}, %[[ARG_B:.*]] = %{{.*}}) -> (tensor<64x64xf32, #mma>, tensor<64x32xf16, #blocked>, tensor<32x64xf16, #blocked>)
  // UPSTREAM: %[[A_CVT:.*]] = ttg.convert_layout %[[ARG_A]] : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
  // UPSTREAM: %[[B_CVT:.*]] = ttg.convert_layout %[[ARG_B]] : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
  // UPSTREAM: tt.dot %[[A_CVT]], %[[B_CVT]], %{{.*}}
  // CHECK-LABEL: @local_load_through_iter_arg
  tt.func public @local_load_through_iter_arg(%arg0: !tt.ptr<f16>) -> tensor<64x64xf32, #mma_1> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_1>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #shared_1, #smem_1, mutable>
    %buf0 = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<2x64x32xf16, #shared_1, #smem_1, mutable> -> !ttg.memdesc<64x32xf16, #shared_1, #smem_1, mutable>
    // CHECK: %[[PROLOGUE_A:.*]] = ttg.local_load {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %a_init = ttg.local_load %buf0 : !ttg.memdesc<64x32xf16, #shared_1, #smem_1, mutable> -> tensor<64x32xf16, #blocked_1>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x32x64xf16, #shared_1, #smem_1, mutable>
    %buf_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<2x32x64xf16, #shared_1, #smem_1, mutable> -> !ttg.memdesc<32x64xf16, #shared_1, #smem_1, mutable>
    // CHECK: %[[PROLOGUE_B:.*]] = ttg.local_load {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %b_init = ttg.local_load %buf_b : !ttg.memdesc<32x64xf16, #shared_1, #smem_1, mutable> -> tensor<32x64xf16, #blocked_1>
    // CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}, %[[ARG_A:.*]] = %[[PROLOGUE_A]], %[[ARG_B:.*]] = %[[PROLOGUE_B]])
    %result:3 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32
        iter_args(%acc = %cst, %a_reg = %a_init, %b_reg = %b_init)
        -> (tensor<64x64xf32, #mma_1>, tensor<64x32xf16, #blocked_1>, tensor<32x64xf16, #blocked_1>) : i32 {
      // CHECK: tt.dot %[[ARG_A]], %[[ARG_B]]
      %a_cvt = ttg.convert_layout %a_reg : tensor<64x32xf16, #blocked_1> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_1, kWidth = 4}>>
      %b_cvt = ttg.convert_layout %b_reg : tensor<32x64xf16, #blocked_1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_1, kWidth = 4}>>
      %dot = tt.dot %a_cvt, %b_cvt, %acc : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_1, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_1, kWidth = 4}>> -> tensor<64x64xf32, #mma_1>
      %buf_next = ttg.memdesc_index %alloc[%c1_i32] : !ttg.memdesc<2x64x32xf16, #shared_1, #smem_1, mutable> -> !ttg.memdesc<64x32xf16, #shared_1, #smem_1, mutable>
      // CHECK: ttg.local_load {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %a_next = ttg.local_load %buf_next : !ttg.memdesc<64x32xf16, #shared_1, #smem_1, mutable> -> tensor<64x32xf16, #blocked_1>
      %buf_b_next = ttg.memdesc_index %alloc_b[%c1_i32] : !ttg.memdesc<2x32x64xf16, #shared_1, #smem_1, mutable> -> !ttg.memdesc<32x64xf16, #shared_1, #smem_1, mutable>
      // CHECK: ttg.local_load {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %b_next = ttg.local_load %buf_b_next : !ttg.memdesc<32x64xf16, #shared_1, #smem_1, mutable> -> tensor<32x64xf16, #blocked_1>
      scf.yield %dot, %a_next, %b_next : tensor<64x64xf32, #mma_1>, tensor<64x32xf16, #blocked_1>, tensor<32x64xf16, #blocked_1>
    }
    tt.return %result#0 : tensor<64x64xf32, #mma_1>
  }
}

// -----
// Test 3: local_load inside scf.if branches.
// When all predecessors can agree, the TLX pipeline should rewrite the
// branch-carried values to the final dot encodings and remove the need for a
// downstream conversion on the scf.if results. For comparison, the upstream
// remove-layout-conversions pass still leaves both dot operand converts after
// the scf.if.

#blocked_2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_2 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared_2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// CHECK-DAG: #{{.*}} = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem_2 = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // UPSTREAM-LABEL: @local_load_through_scf_if
  // UPSTREAM: %[[IF_RESULT:.*]]:2 = scf.if {{.*}} -> (tensor<64x32xf16, #blocked>, tensor<32x64xf16, #blocked>)
  // UPSTREAM: %[[A_CVT:.*]] = ttg.convert_layout %[[IF_RESULT]]#0 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
  // UPSTREAM: %[[B_CVT:.*]] = ttg.convert_layout %[[IF_RESULT]]#1 : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
  // UPSTREAM: tt.dot %[[A_CVT]], %[[B_CVT]], %{{.*}}
  // CHECK-LABEL: @local_load_through_scf_if
  tt.func public @local_load_through_scf_if(%arg0: !tt.ptr<f16>, %cond: i1) -> tensor<64x64xf32, #mma_2> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_2>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #shared_2, #smem_2, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x32x64xf16, #shared_2, #smem_2, mutable>
    %buf_a0 = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<2x64x32xf16, #shared_2, #smem_2, mutable> -> !ttg.memdesc<64x32xf16, #shared_2, #smem_2, mutable>
    %buf_b0 = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<2x32x64xf16, #shared_2, #smem_2, mutable> -> !ttg.memdesc<32x64xf16, #shared_2, #smem_2, mutable>
    %buf_a1 = ttg.memdesc_index %alloc_a[%c1_i32] : !ttg.memdesc<2x64x32xf16, #shared_2, #smem_2, mutable> -> !ttg.memdesc<64x32xf16, #shared_2, #smem_2, mutable>
    %buf_b1 = ttg.memdesc_index %alloc_b[%c1_i32] : !ttg.memdesc<2x32x64xf16, #shared_2, #smem_2, mutable> -> !ttg.memdesc<32x64xf16, #shared_2, #smem_2, mutable>
    // CHECK: scf.if {{.*}} -> (tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)
    %if_result:2 = scf.if %cond -> (tensor<64x32xf16, #blocked_2>, tensor<32x64xf16, #blocked_2>) {
      // CHECK: ttg.local_load {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %a = ttg.local_load %buf_a0 : !ttg.memdesc<64x32xf16, #shared_2, #smem_2, mutable> -> tensor<64x32xf16, #blocked_2>
      // CHECK: ttg.local_load {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %b = ttg.local_load %buf_b0 : !ttg.memdesc<32x64xf16, #shared_2, #smem_2, mutable> -> tensor<32x64xf16, #blocked_2>
      // CHECK: scf.yield {{.*}} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      scf.yield %a, %b : tensor<64x32xf16, #blocked_2>, tensor<32x64xf16, #blocked_2>
    } else {
      // CHECK: ttg.local_load {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %a = ttg.local_load %buf_a1 : !ttg.memdesc<64x32xf16, #shared_2, #smem_2, mutable> -> tensor<64x32xf16, #blocked_2>
      // CHECK: ttg.local_load {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %b = ttg.local_load %buf_b1 : !ttg.memdesc<32x64xf16, #shared_2, #smem_2, mutable> -> tensor<32x64xf16, #blocked_2>
      // CHECK: scf.yield {{.*}} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      scf.yield %a, %b : tensor<64x32xf16, #blocked_2>, tensor<32x64xf16, #blocked_2>
    }
    // CHECK: tt.dot %{{.*}}#0, %{{.*}}#1, %{{.*}}
    %a_cvt = ttg.convert_layout %if_result#0 : tensor<64x32xf16, #blocked_2> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_2, kWidth = 4}>>
    %b_cvt = ttg.convert_layout %if_result#1 : tensor<32x64xf16, #blocked_2> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_2, kWidth = 4}>>
    %dot = tt.dot %a_cvt, %b_cvt, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_2, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_2, kWidth = 4}>> -> tensor<64x64xf32, #mma_2>
    tt.return %dot : tensor<64x64xf32, #mma_2>
  }
}

// -----
// Test 4: user-specified order=[0,1] on the source memdesc survives the full
// insert+propagate pipeline. The resulting local_loads should carry the final
// dot operand encodings directly, while the rewritten shared encodings preserve
// the source order contract.

#blocked_3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_3 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#shared_k_contig_3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared_default_3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// CHECK-DAG: #[[$SHARED_M:.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
// CHECK-DAG: #[[$SHARED_K:.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem_3 = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @user_specified_order_preserved_pipeline
  tt.func public @user_specified_order_preserved_pipeline() -> tensor<128x128xf32, #mma_3> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_3>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared_default_3, #smem_3, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared_k_contig_3, #smem_3, mutable>
    %buf_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared_default_3, #smem_3, mutable> -> !ttg.memdesc<128x64xf16, #shared_default_3, #smem_3, mutable>
    %buf_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared_k_contig_3, #smem_3, mutable> -> !ttg.memdesc<64x128xf16, #shared_k_contig_3, #smem_3, mutable>
    // CHECK: %[[A_LOAD:.*]] = ttg.local_load %{{.*}} : !ttg.memdesc<128x64xf16, #[[$SHARED_M]], #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %a = ttg.local_load %buf_a : !ttg.memdesc<128x64xf16, #shared_default_3, #smem_3, mutable> -> tensor<128x64xf16, #blocked_3>
    // CHECK: %[[B_LOAD:.*]] = ttg.local_load %{{.*}} : !ttg.memdesc<64x128xf16, #[[$SHARED_K]], #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %b = ttg.local_load %buf_b : !ttg.memdesc<64x128xf16, #shared_k_contig_3, #smem_3, mutable> -> tensor<64x128xf16, #blocked_3>
    %a_dot = ttg.convert_layout %a : tensor<128x64xf16, #blocked_3> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_3, kWidth = 8}>>
    %b_dot = ttg.convert_layout %b : tensor<64x128xf16, #blocked_3> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_3, kWidth = 8}>>
    // CHECK: tt.dot %[[A_LOAD]], %[[B_LOAD]], %{{.*}}
    %dot = tt.dot %a_dot, %b_dot, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_3, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_3, kWidth = 8}>> -> tensor<128x128xf32, #mma_3>
    tt.return %dot : tensor<128x128xf32, #mma_3>
  }
}

// -----
// Test 5: one local_load feeds both a dot path and a sibling non-dot path.
// The full pipeline should keep the mixed-use load in its original blocked
// layout, preserve the sibling conversion, and only propagate the clean dot-use
// load to the final dot encoding.

#blocked_4 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_4_alt = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma_4 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared_4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_4 = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @mixed_use_local_load_pipeline_fallback
  tt.func public @mixed_use_local_load_pipeline_fallback() -> tensor<64x64xf32, #mma_4> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked_4>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared_4, #smem_4, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared_4, #smem_4, mutable>
    %alloc_sink = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared_4, #smem_4, mutable>
    // CHECK: %[[DESC_A:.*]] = ttg.memdesc_index
    %buf_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared_4, #smem_4, mutable> -> !ttg.memdesc<64x32xf16, #shared_4, #smem_4, mutable>
    %buf_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared_4, #smem_4, mutable> -> !ttg.memdesc<32x64xf16, #shared_4, #smem_4, mutable>
    %buf_sink = ttg.memdesc_index %alloc_sink[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared_4, #smem_4, mutable> -> !ttg.memdesc<64x32xf16, #shared_4, #smem_4, mutable>
    // CHECK: %[[A_LOAD:.*]] = ttg.local_load %[[DESC_A]] : !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable> -> tensor<64x32xf16, #blocked>
    %a = ttg.local_load %buf_a : !ttg.memdesc<64x32xf16, #shared_4, #smem_4, mutable> -> tensor<64x32xf16, #blocked_4>
    // CHECK: %[[A_ALT:.*]] = ttg.convert_layout %[[A_LOAD]] : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #{{.*}}>
    %a_alt = ttg.convert_layout %a : tensor<64x32xf16, #blocked_4> -> tensor<64x32xf16, #blocked_4_alt>
    // CHECK: ttg.local_store %[[A_ALT]], %{{.*}}
    ttg.local_store %a_alt, %buf_sink : tensor<64x32xf16, #blocked_4_alt> -> !ttg.memdesc<64x32xf16, #shared_4, #smem_4, mutable>
    // CHECK: %[[B_LOAD:.*]] = ttg.local_load %{{.*}} : !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %b = ttg.local_load %buf_b : !ttg.memdesc<32x64xf16, #shared_4, #smem_4, mutable> -> tensor<32x64xf16, #blocked_4>
    %acc = ttg.convert_layout %cst : tensor<64x64xf32, #blocked_4> -> tensor<64x64xf32, #mma_4>
    // CHECK: %[[A_DOT:.*]] = ttg.convert_layout %[[A_LOAD]] : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %a_dot = ttg.convert_layout %a : tensor<64x32xf16, #blocked_4> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_4, kWidth = 4}>>
    %b_dot = ttg.convert_layout %b : tensor<32x64xf16, #blocked_4> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_4, kWidth = 4}>>
    // CHECK: tt.dot %[[A_DOT]], %[[B_LOAD]], %{{.*}}
    %dot = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_4, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_4, kWidth = 4}>> -> tensor<64x64xf32, #mma_4>
    tt.return %dot : tensor<64x64xf32, #mma_4>
  }
}

// -----
// Test 6: one local_load feeds two dot paths that demand different dot operand
// encodings. The full pipeline should keep the A-side local_load blocked and
// lower the residual tensor constraints to two explicit convert_layout bridges.

#blocked_5 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_5 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#mma_5_alt = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [32, 32, 8], isTransposed = true}>
#shared_5 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_5 = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @conflicting_dot_operands_pipeline_fallback
  tt.func public @conflicting_dot_operands_pipeline_fallback() -> tensor<64x64xf32, #mma_5> {
    %c0_i32 = arith.constant 0 : i32
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_5>
    %acc1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_5_alt>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared_5, #smem_5, mutable>
    %alloc_b0 = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared_5, #smem_5, mutable>
    %alloc_b1 = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared_5, #smem_5, mutable>
    %buf_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared_5, #smem_5, mutable> -> !ttg.memdesc<64x32xf16, #shared_5, #smem_5, mutable>
    %buf_b0 = ttg.memdesc_index %alloc_b0[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared_5, #smem_5, mutable> -> !ttg.memdesc<32x64xf16, #shared_5, #smem_5, mutable>
    %buf_b1 = ttg.memdesc_index %alloc_b1[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared_5, #smem_5, mutable> -> !ttg.memdesc<32x64xf16, #shared_5, #smem_5, mutable>
    // CHECK: %[[A_LOAD:.*]] = ttg.local_load %{{.*}} : !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable> -> tensor<64x32xf16, #blocked>
    %a = ttg.local_load %buf_a : !ttg.memdesc<64x32xf16, #shared_5, #smem_5, mutable> -> tensor<64x32xf16, #blocked_5>
    %b0 = ttg.local_load %buf_b0 : !ttg.memdesc<32x64xf16, #shared_5, #smem_5, mutable> -> tensor<32x64xf16, #blocked_5>
    %b1 = ttg.local_load %buf_b1 : !ttg.memdesc<32x64xf16, #shared_5, #smem_5, mutable> -> tensor<32x64xf16, #blocked_5>
    // CHECK: %[[A_DOT0:.*]] = ttg.convert_layout %[[A_LOAD]] : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %a_dot0 = ttg.convert_layout %a : tensor<64x32xf16, #blocked_5> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_5, kWidth = 4}>>
    %b_dot0 = ttg.convert_layout %b0 : tensor<32x64xf16, #blocked_5> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_5, kWidth = 4}>>
    // CHECK: %[[A_DOT1:.*]] = ttg.convert_layout %[[A_LOAD]] : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>>
    %a_dot1 = ttg.convert_layout %a : tensor<64x32xf16, #blocked_5> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_5_alt, kWidth = 4}>>
    %b_dot1 = ttg.convert_layout %b1 : tensor<32x64xf16, #blocked_5> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_5_alt, kWidth = 4}>>
    // CHECK: tt.dot %[[A_DOT0]], %{{.*}}, %{{.*}}
    %dot0 = tt.dot %a_dot0, %b_dot0, %acc0 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_5, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_5, kWidth = 4}>> -> tensor<64x64xf32, #mma_5>
    // CHECK: tt.dot %[[A_DOT1]], %{{.*}}, %{{.*}}
    %dot1 = tt.dot %a_dot1, %b_dot1, %acc1 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_5_alt, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_5_alt, kWidth = 4}>> -> tensor<64x64xf32, #mma_5_alt>
    %dot1_to_mma0 = ttg.convert_layout %dot1 : tensor<64x64xf32, #mma_5_alt> -> tensor<64x64xf32, #mma_5>
    %sum = arith.addf %dot0, %dot1_to_mma0 : tensor<64x64xf32, #mma_5>
    tt.return %sum : tensor<64x64xf32, #mma_5>
  }
}
