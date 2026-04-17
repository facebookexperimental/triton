// RUN: triton-opt -split-input-file --tlx-insert-require-layout %s | FileCheck %s

// Test 1: direct local_load -> convert_layout -> dot.
// InsertRequireLayout should synthesize explicit memdesc and tensor TLX
// constraints, but it should not retag the local_load result directly.

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
    // CHECK: %[[DESC_A:.*]] = ttg.memdesc_index
    %buf_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    // CHECK: %[[DESC_B:.*]] = ttg.memdesc_index
    %buf_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    %a_val = tt.load %arg1 : tensor<64x32x!tt.ptr<f16>, #blocked>
    %b_val = tt.load %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked>
    ttg.local_store %a_val, %buf_a : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    ttg.local_store %b_val, %buf_b : tensor<32x64xf16, #blocked> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    // CHECK: %[[REQ_A_MEM:.*]] = tlx.require_layout %[[DESC_A]] {{.*}} -> !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: %[[A_LOAD:.*]] = ttg.local_load %[[REQ_A_MEM]] : !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %a = ttg.local_load %buf_a : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // CHECK: %[[REQ_B_MEM:.*]] = tlx.require_layout %[[DESC_B]] {{.*}} -> !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: %[[B_LOAD:.*]] = ttg.local_load %[[REQ_B_MEM]] : !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %b = ttg.local_load %buf_b : !ttg.memdesc<32x64xf16, #shared1, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %acc = ttg.convert_layout %cst : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #mma>
    // CHECK: %[[A_REQ:.*]] = tlx.require_layout %[[A_LOAD]] : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %a_dot = ttg.convert_layout %a : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    // CHECK: %[[B_REQ:.*]] = tlx.require_layout %[[B_LOAD]] : tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %b_dot = ttg.convert_layout %b : tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    // CHECK: tt.dot %[[A_REQ]], %[[B_REQ]], %{{.*}}
    %dot = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return %dot : tensor<64x64xf32, #mma>
  }
}

// -----
// Test 2: local_load feeds scf.for iter_args that eventually reach a dot.
// InsertRequireLayout keeps the loop-carried values in their original type and
// materializes tensor constraints at the dot use.

#blocked_1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared_1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// CHECK-DAG: #{{.*}} = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem_1 = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @local_load_through_iter_arg
  tt.func public @local_load_through_iter_arg(%arg0: !tt.ptr<f16>) -> tensor<64x64xf32, #mma_1> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_1>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #shared_1, #smem_1, mutable>
    %buf_a0 = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<2x64x32xf16, #shared_1, #smem_1, mutable> -> !ttg.memdesc<64x32xf16, #shared_1, #smem_1, mutable>
    // CHECK: %[[REQ_A_MEM:.*]] = tlx.require_layout %{{.*}} {{.*}} -> !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: %[[A_INIT:.*]] = ttg.local_load %[[REQ_A_MEM]] : !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable> -> tensor<64x32xf16, #blocked_1>
    %a_init = ttg.local_load %buf_a0 : !ttg.memdesc<64x32xf16, #shared_1, #smem_1, mutable> -> tensor<64x32xf16, #blocked_1>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x32x64xf16, #shared_1, #smem_1, mutable>
    %buf_b0 = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<2x32x64xf16, #shared_1, #smem_1, mutable> -> !ttg.memdesc<32x64xf16, #shared_1, #smem_1, mutable>
    // CHECK: %[[REQ_B_MEM:.*]] = tlx.require_layout %{{.*}} {{.*}} -> !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: %[[B_INIT:.*]] = ttg.local_load %[[REQ_B_MEM]] : !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable> -> tensor<32x64xf16, #blocked_1>
    %b_init = ttg.local_load %buf_b0 : !ttg.memdesc<32x64xf16, #shared_1, #smem_1, mutable> -> tensor<32x64xf16, #blocked_1>
    // CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}, %[[A_ARG:.*]] = %[[A_INIT]], %[[B_ARG:.*]] = %[[B_INIT]])
    %result:3 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32
        iter_args(%acc = %cst, %a_reg = %a_init, %b_reg = %b_init)
        -> (tensor<64x64xf32, #mma_1>, tensor<64x32xf16, #blocked_1>, tensor<32x64xf16, #blocked_1>) : i32 {
      // CHECK: %[[A_REQ:.*]] = tlx.require_layout %[[A_ARG]] : tensor<64x32xf16, #blocked_1> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %a_dot = ttg.convert_layout %a_reg : tensor<64x32xf16, #blocked_1> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_1, kWidth = 4}>>
      // CHECK: %[[B_REQ:.*]] = tlx.require_layout %[[B_ARG]] : tensor<32x64xf16, #blocked_1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %b_dot = ttg.convert_layout %b_reg : tensor<32x64xf16, #blocked_1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_1, kWidth = 4}>>
      // CHECK: tt.dot %[[A_REQ]], %[[B_REQ]]
      %dot = tt.dot %a_dot, %b_dot, %acc : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_1, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_1, kWidth = 4}>> -> tensor<64x64xf32, #mma_1>
      %buf_a1 = ttg.memdesc_index %alloc_a[%c1_i32] : !ttg.memdesc<2x64x32xf16, #shared_1, #smem_1, mutable> -> !ttg.memdesc<64x32xf16, #shared_1, #smem_1, mutable>
      // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable>
      // CHECK-NEXT: ttg.local_load {{.*}} -> tensor<64x32xf16, #blocked_1>
      %a_next = ttg.local_load %buf_a1 : !ttg.memdesc<64x32xf16, #shared_1, #smem_1, mutable> -> tensor<64x32xf16, #blocked_1>
      %buf_b1 = ttg.memdesc_index %alloc_b[%c1_i32] : !ttg.memdesc<2x32x64xf16, #shared_1, #smem_1, mutable> -> !ttg.memdesc<32x64xf16, #shared_1, #smem_1, mutable>
      // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable>
      // CHECK-NEXT: ttg.local_load {{.*}} -> tensor<32x64xf16, #blocked_1>
      %b_next = ttg.local_load %buf_b1 : !ttg.memdesc<32x64xf16, #shared_1, #smem_1, mutable> -> tensor<32x64xf16, #blocked_1>
      scf.yield %dot, %a_next, %b_next : tensor<64x64xf32, #mma_1>, tensor<64x32xf16, #blocked_1>, tensor<32x64xf16, #blocked_1>
    }
    tt.return %result#0 : tensor<64x64xf32, #mma_1>
  }
}

// -----
// Test 3: user-specified order=[0,1] on the source memdesc is preserved.
// The pass should still synthesize explicit tensor constraints at the dot use.

#blocked_2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_2 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#shared_k_contig = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared_default = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// CHECK-DAG: #{{.*}} = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
// CHECK-DAG: #{{.*}} = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem_2 = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @user_specified_order_preserved
  tt.func public @user_specified_order_preserved(%arg0: !tt.ptr<f16>) -> tensor<128x128xf32, #mma_2> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_2>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared_default, #smem_2, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared_k_contig, #smem_2, mutable>
    %buf_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<1x128x64xf16, #shared_default, #smem_2, mutable> -> !ttg.memdesc<128x64xf16, #shared_default, #smem_2, mutable>
    %buf_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared_k_contig, #smem_2, mutable> -> !ttg.memdesc<64x128xf16, #shared_k_contig, #smem_2, mutable>
    // CHECK: %[[A_REQ_MEM:.*]] = tlx.require_layout %{{.*}} {{.*}} -> !ttg.memdesc<128x64xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: %[[A_LOAD:.*]] = ttg.local_load %[[A_REQ_MEM]] : !ttg.memdesc<128x64xf16, #{{.*}}, #smem, mutable> -> tensor<128x64xf16, #blocked_2>
    %a = ttg.local_load %buf_a : !ttg.memdesc<128x64xf16, #shared_default, #smem_2, mutable> -> tensor<128x64xf16, #blocked_2>
    // CHECK: %[[B_REQ_MEM:.*]] = tlx.require_layout %{{.*}} {{.*}} -> !ttg.memdesc<64x128xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: %[[B_LOAD:.*]] = ttg.local_load %[[B_REQ_MEM]] : !ttg.memdesc<64x128xf16, #{{.*}}, #smem, mutable> -> tensor<64x128xf16, #blocked_2>
    %b = ttg.local_load %buf_b : !ttg.memdesc<64x128xf16, #shared_k_contig, #smem_2, mutable> -> tensor<64x128xf16, #blocked_2>
    // CHECK: tlx.require_layout %[[A_LOAD]] : tensor<128x64xf16, #blocked_2> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %a_dot = ttg.convert_layout %a : tensor<128x64xf16, #blocked_2> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_2, kWidth = 8}>>
    // CHECK: tlx.require_layout %[[B_LOAD]] : tensor<64x128xf16, #blocked_2> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %b_dot = ttg.convert_layout %b : tensor<64x128xf16, #blocked_2> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_2, kWidth = 8}>>
    %dot = tt.dot %a_dot, %b_dot, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_2, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_2, kWidth = 8}>> -> tensor<128x128xf32, #mma_2>
    tt.return %dot : tensor<128x128xf32, #mma_2>
  }
}

// -----
// Test 4: local_load inside scf.if branches.
// InsertRequireLayout should leave the scf.if result types alone and rewrite
// the downstream dot-path conversions into explicit tensor TLX constraints.

#blocked_3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_3 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared_3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// CHECK-DAG: #{{.*}} = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem_3 = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @local_load_through_scf_if
  tt.func public @local_load_through_scf_if(%arg0: !tt.ptr<f16>, %cond: i1) -> tensor<64x64xf32, #mma_3> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_3>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #shared_3, #smem_3, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x32x64xf16, #shared_3, #smem_3, mutable>
    %buf_a0 = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<2x64x32xf16, #shared_3, #smem_3, mutable> -> !ttg.memdesc<64x32xf16, #shared_3, #smem_3, mutable>
    %buf_b0 = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<2x32x64xf16, #shared_3, #smem_3, mutable> -> !ttg.memdesc<32x64xf16, #shared_3, #smem_3, mutable>
    %buf_a1 = ttg.memdesc_index %alloc_a[%c1_i32] : !ttg.memdesc<2x64x32xf16, #shared_3, #smem_3, mutable> -> !ttg.memdesc<64x32xf16, #shared_3, #smem_3, mutable>
    %buf_b1 = ttg.memdesc_index %alloc_b[%c1_i32] : !ttg.memdesc<2x32x64xf16, #shared_3, #smem_3, mutable> -> !ttg.memdesc<32x64xf16, #shared_3, #smem_3, mutable>
    // CHECK: scf.if {{.*}} -> (tensor<64x32xf16, #blocked_3>, tensor<32x64xf16, #blocked_3>)
    %if_result:2 = scf.if %cond -> (tensor<64x32xf16, #blocked_3>, tensor<32x64xf16, #blocked_3>) {
      // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable>
      // CHECK-NEXT: ttg.local_load {{.*}} -> tensor<64x32xf16, #blocked_3>
      %a = ttg.local_load %buf_a0 : !ttg.memdesc<64x32xf16, #shared_3, #smem_3, mutable> -> tensor<64x32xf16, #blocked_3>
      // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable>
      // CHECK-NEXT: ttg.local_load {{.*}} -> tensor<32x64xf16, #blocked_3>
      %b = ttg.local_load %buf_b0 : !ttg.memdesc<32x64xf16, #shared_3, #smem_3, mutable> -> tensor<32x64xf16, #blocked_3>
      scf.yield %a, %b : tensor<64x32xf16, #blocked_3>, tensor<32x64xf16, #blocked_3>
    } else {
      // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable>
      // CHECK-NEXT: ttg.local_load {{.*}} -> tensor<64x32xf16, #blocked_3>
      %a = ttg.local_load %buf_a1 : !ttg.memdesc<64x32xf16, #shared_3, #smem_3, mutable> -> tensor<64x32xf16, #blocked_3>
      // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable>
      // CHECK-NEXT: ttg.local_load {{.*}} -> tensor<32x64xf16, #blocked_3>
      %b = ttg.local_load %buf_b1 : !ttg.memdesc<32x64xf16, #shared_3, #smem_3, mutable> -> tensor<32x64xf16, #blocked_3>
      scf.yield %a, %b : tensor<64x32xf16, #blocked_3>, tensor<32x64xf16, #blocked_3>
    }
    // CHECK: %[[IF_A_REQ:.*]] = tlx.require_layout %if_result#0 : tensor<64x32xf16, #blocked_3> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %a_dot = ttg.convert_layout %if_result#0 : tensor<64x32xf16, #blocked_3> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_3, kWidth = 4}>>
    // CHECK: %[[IF_B_REQ:.*]] = tlx.require_layout %if_result#1 : tensor<32x64xf16, #blocked_3> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %b_dot = ttg.convert_layout %if_result#1 : tensor<32x64xf16, #blocked_3> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_3, kWidth = 4}>>
    // CHECK: tt.dot %[[IF_A_REQ]], %[[IF_B_REQ]], %{{.*}}
    %dot = tt.dot %a_dot, %b_dot, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_3, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_3, kWidth = 4}>> -> tensor<64x64xf32, #mma_3>
    tt.return %dot : tensor<64x64xf32, #mma_3>
  }
}

// -----
// Test 5: one local_load feeds both a dot path and a sibling convert_layout
// path into a non-dot user. Mixed-use should block memdesc-side rewriting for
// A, but explicit tensor constraints can still model the dot uses.

#blocked_4 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_4_alt = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma_4 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared_4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_4 = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @mixed_use_local_load_stays_blocked
  tt.func public @mixed_use_local_load_stays_blocked(%arg0: !tt.ptr<f16>) -> tensor<64x64xf32, #mma_4> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked_4>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared_4, #smem_4, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared_4, #smem_4, mutable>
    %alloc_sink = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared_4, #smem_4, mutable>
    // CHECK: %[[DESC_A:.*]] = ttg.memdesc_index
    %buf_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared_4, #smem_4, mutable> -> !ttg.memdesc<64x32xf16, #shared_4, #smem_4, mutable>
    // CHECK: %[[DESC_B:.*]] = ttg.memdesc_index
    %buf_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared_4, #smem_4, mutable> -> !ttg.memdesc<32x64xf16, #shared_4, #smem_4, mutable>
    %buf_sink = ttg.memdesc_index %alloc_sink[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared_4, #smem_4, mutable> -> !ttg.memdesc<64x32xf16, #shared_4, #smem_4, mutable>
    // CHECK: %[[A_LOAD:.*]] = ttg.local_load %[[DESC_A]] : !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable> -> tensor<64x32xf16, #{{.*}}>
    %a = ttg.local_load %buf_a : !ttg.memdesc<64x32xf16, #shared_4, #smem_4, mutable> -> tensor<64x32xf16, #blocked_4>
    // CHECK: %[[A_ALT:.*]] = ttg.convert_layout %[[A_LOAD]] : tensor<64x32xf16, #{{.*}}> -> tensor<64x32xf16, #{{.*}}>
    %a_alt = ttg.convert_layout %a : tensor<64x32xf16, #blocked_4> -> tensor<64x32xf16, #blocked_4_alt>
    // CHECK: ttg.local_store %[[A_ALT]],
    ttg.local_store %a_alt, %buf_sink : tensor<64x32xf16, #blocked_4_alt> -> !ttg.memdesc<64x32xf16, #shared_4, #smem_4, mutable>
    // CHECK: %[[REQ_B_MEM:.*]] = tlx.require_layout %[[DESC_B]] {{.*}} -> !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: %[[B_LOAD:.*]] = ttg.local_load %[[REQ_B_MEM]] : !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable> -> tensor<32x64xf16, #{{.*}}>
    %b = ttg.local_load %buf_b : !ttg.memdesc<32x64xf16, #shared_4, #smem_4, mutable> -> tensor<32x64xf16, #blocked_4>
    %acc = ttg.convert_layout %cst : tensor<64x64xf32, #blocked_4> -> tensor<64x64xf32, #mma_4>
    // CHECK: %[[A_REQ:.*]] = tlx.require_layout %[[A_LOAD]] : tensor<64x32xf16, #{{.*}}> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %a_dot = ttg.convert_layout %a : tensor<64x32xf16, #blocked_4> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_4, kWidth = 4}>>
    // CHECK: %[[B_REQ:.*]] = tlx.require_layout %[[B_LOAD]] : tensor<32x64xf16, #{{.*}}> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %b_dot = ttg.convert_layout %b : tensor<32x64xf16, #blocked_4> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_4, kWidth = 4}>>
    // CHECK: tt.dot %[[A_REQ]], %[[B_REQ]], %{{.*}}
    %dot = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_4, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_4, kWidth = 4}>> -> tensor<64x64xf32, #mma_4>
    tt.return %dot : tensor<64x64xf32, #mma_4>
  }
}
