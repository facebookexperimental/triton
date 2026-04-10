// RUN: triton-opt -split-input-file --tlx-insert-require-layout %s| FileCheck %s

// Test 1: Basic case -- direct local_load -> convert_layout -> dot.
// Verify require_layout is inserted before each local_load with a swizzled_shared encoding.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#loc = loc("/home/kmanivannan/fb-triton/python/test/unit/language/test_tlx.py":158:0)
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
// CHECK-DAG: #{{.*}} = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
// CHECK-DAG: #{{.*}} = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @local_store_local_load_dot
  tt.func public @local_store_local_load_dot(%arg0: !tt.ptr<f16>, %arg1: tensor<64x32x!tt.ptr<f16>, #blocked>, %arg2: tensor<32x64x!tt.ptr<f16>, #blocked>) -> tensor<64x64xf32, #mma> {
    %24 = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable>
    %25 = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared1, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    // CHECK: %[[DESC_A:.*]] = ttg.memdesc_index
    %26 = ttg.memdesc_index %24[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    // CHECK: %[[DESC_B:.*]] = ttg.memdesc_index
    %27 = ttg.memdesc_index %25[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    %28 = tt.load %arg1 : tensor<64x32x!tt.ptr<f16>, #blocked>
    %29 = tt.load %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked>
    ttg.local_store %28, %26 : tensor<64x32xf16, #blocked> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    ttg.local_store %29, %27 : tensor<32x64xf16, #blocked> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
    // CHECK: %[[REQ_A:.*]] = tlx.require_layout %[[DESC_A]] {{.*}} -> !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: ttg.local_load %[[REQ_A]] : !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %30 = ttg.local_load %26 : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // CHECK: %[[REQ_B:.*]] = tlx.require_layout %[[DESC_B]] {{.*}} -> !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: ttg.local_load %[[REQ_B]] : !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %31 = ttg.local_load %27 : !ttg.memdesc<32x64xf16, #shared1, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %32 = ttg.convert_layout %cst : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #mma>
    %33 = ttg.convert_layout %30 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %34 = ttg.convert_layout %31 : tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %35 = tt.dot %33, %34, %32, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return %35 : tensor<64x64xf32, #mma>
  }
}

// -----
// Test 2: local_load feeds scf.for iter_arg that eventually reaches a dot.
// The dataflow analysis traces through yield -> body_arg -> init_value.
// Prologue loads and loop-body loads all get require_layout.

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
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #shared_1, #smem_1, mutable>
    %buf0 = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<2x64x32xf16, #shared_1, #smem_1, mutable> -> !ttg.memdesc<64x32xf16, #shared_1, #smem_1, mutable>
    // Prologue load for A: must get require_layout with swizzled encoding.
    // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: %[[PROLOGUE_A:.*]] = ttg.local_load {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %a_init = ttg.local_load %buf0 : !ttg.memdesc<64x32xf16, #shared_1, #smem_1, mutable> -> tensor<64x32xf16, #blocked_1>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x32x64xf16, #shared_1, #smem_1, mutable>
    %buf_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<2x32x64xf16, #shared_1, #smem_1, mutable> -> !ttg.memdesc<32x64xf16, #shared_1, #smem_1, mutable>
    // Prologue load for B: must also get require_layout.
    // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<32x64xf16, {{.*}}, #smem, mutable>
    // CHECK-NEXT: %[[PROLOGUE_B:.*]] = ttg.local_load {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %b_init = ttg.local_load %buf_b : !ttg.memdesc<32x64xf16, #shared_1, #smem_1, mutable> -> tensor<32x64xf16, #blocked_1>
    // Loop carries dot_op-encoded tensors from prologue loads.
    // CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}, %[[ARG_A:.*]] = %[[PROLOGUE_A]], %[[ARG_B:.*]] = %[[PROLOGUE_B]])
    %result:3 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32
        iter_args(%acc = %cst, %a_reg = %a_init, %b_reg = %b_init)
        -> (tensor<64x64xf32, #mma_1>, tensor<64x32xf16, #blocked_1>, tensor<32x64xf16, #blocked_1>) : i32 {
      // CHECK: tt.dot %[[ARG_A]], %[[ARG_B]]
      %a_cvt = ttg.convert_layout %a_reg : tensor<64x32xf16, #blocked_1> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_1, kWidth = 4}>>
      %b_cvt = ttg.convert_layout %b_reg : tensor<32x64xf16, #blocked_1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_1, kWidth = 4}>>
      %dot = tt.dot %a_cvt, %b_cvt, %acc : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_1, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_1, kWidth = 4}>> -> tensor<64x64xf32, #mma_1>
      %buf_next = ttg.memdesc_index %alloc[%c1_i32] : !ttg.memdesc<2x64x32xf16, #shared_1, #smem_1, mutable> -> !ttg.memdesc<64x32xf16, #shared_1, #smem_1, mutable>
      // Loop-body load for A: also gets require_layout.
      // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable>
      // CHECK-NEXT: ttg.local_load
      %a_next = ttg.local_load %buf_next : !ttg.memdesc<64x32xf16, #shared_1, #smem_1, mutable> -> tensor<64x32xf16, #blocked_1>
      %buf_b_next = ttg.memdesc_index %alloc_b[%c1_i32] : !ttg.memdesc<2x32x64xf16, #shared_1, #smem_1, mutable> -> !ttg.memdesc<32x64xf16, #shared_1, #smem_1, mutable>
      // Loop-body load for B: also gets require_layout.
      // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<32x64xf16, {{.*}}, #smem, mutable>
      // CHECK-NEXT: ttg.local_load
      %b_next = ttg.local_load %buf_b_next : !ttg.memdesc<32x64xf16, #shared_1, #smem_1, mutable> -> tensor<32x64xf16, #blocked_1>
      scf.yield %dot, %a_next, %b_next : tensor<64x64xf32, #mma_1>, tensor<64x32xf16, #blocked_1>, tensor<32x64xf16, #blocked_1>
    }
    tt.return %result#0 : tensor<64x64xf32, #mma_1>
  }
}

// -----
// Test 3: user-specified order=[0,1] on the source memdesc is preserved.
// A gets order=[1,0] (default), B gets order=[0,1] (K-contiguous).
// This is the pre-transposed B pattern for avoiding ds_read_tr.

#blocked_2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_2 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#shared_k_contig = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared_default = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
// A's output encoding has order=[1,0] (default).
// B's output encoding must preserve order=[0,1] (K-contiguous).
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
    // A: order=[1,0] in, order=[1,0] out.
    // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<128x64xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: ttg.local_load
    %a = ttg.local_load %buf_a : !ttg.memdesc<128x64xf16, #shared_default, #smem_2, mutable> -> tensor<128x64xf16, #blocked_2>
    // B: order=[0,1] in, order=[0,1] out -- preserved.
    // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<64x128xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: ttg.local_load
    %b = ttg.local_load %buf_b : !ttg.memdesc<64x128xf16, #shared_k_contig, #smem_2, mutable> -> tensor<64x128xf16, #blocked_2>
    %a_cvt = ttg.convert_layout %a : tensor<128x64xf16, #blocked_2> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_2, kWidth = 8}>>
    %b_cvt = ttg.convert_layout %b : tensor<64x128xf16, #blocked_2> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_2, kWidth = 8}>>
    %dot = tt.dot %a_cvt, %b_cvt, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_2, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_2, kWidth = 8}>> -> tensor<128x128xf32, #mma_2>
    tt.return %dot : tensor<128x128xf32, #mma_2>
  }
}

// -----
// Test 4: local_load inside scf.if branches -- dot encoding propagates
// backward through scf.if results and the pass fixes result types.

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
    // scf.if result types should be updated to dot_op.
    // CHECK: scf.if {{.*}} -> (tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)
    %if_result:2 = scf.if %cond -> (tensor<64x32xf16, #blocked_3>, tensor<32x64xf16, #blocked_3>) {
      // Then branch: A gets require_layout with swizzled encoding.
      // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable>
      // CHECK-NEXT: ttg.local_load {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %a = ttg.local_load %buf_a0 : !ttg.memdesc<64x32xf16, #shared_3, #smem_3, mutable> -> tensor<64x32xf16, #blocked_3>
      // Then branch: B gets require_layout.
      // CHECK: tlx.require_layout
      // CHECK-NEXT: ttg.local_load {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %b = ttg.local_load %buf_b0 : !ttg.memdesc<32x64xf16, #shared_3, #smem_3, mutable> -> tensor<32x64xf16, #blocked_3>
      // Yield types are updated to dot_op.
      // CHECK: scf.yield {{.*}} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      scf.yield %a, %b : tensor<64x32xf16, #blocked_3>, tensor<32x64xf16, #blocked_3>
    } else {
      // Else branch: same treatment.
      // CHECK: tlx.require_layout {{.*}} -> !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable>
      // CHECK-NEXT: ttg.local_load {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %a = ttg.local_load %buf_a1 : !ttg.memdesc<64x32xf16, #shared_3, #smem_3, mutable> -> tensor<64x32xf16, #blocked_3>
      // CHECK: tlx.require_layout
      // CHECK-NEXT: ttg.local_load {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %b = ttg.local_load %buf_b1 : !ttg.memdesc<32x64xf16, #shared_3, #smem_3, mutable> -> tensor<32x64xf16, #blocked_3>
      // CHECK: scf.yield {{.*}} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      scf.yield %a, %b : tensor<64x32xf16, #blocked_3>, tensor<32x64xf16, #blocked_3>
    }
    // Dot consumes scf.if results directly -- no convert_layout in between.
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot %{{.*}}#0, %{{.*}}#1, %{{.*}} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    %a_cvt = ttg.convert_layout %if_result#0 : tensor<64x32xf16, #blocked_3> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_3, kWidth = 4}>>
    %b_cvt = ttg.convert_layout %if_result#1 : tensor<32x64xf16, #blocked_3> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_3, kWidth = 4}>>
    %dot = tt.dot %a_cvt, %b_cvt, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_3, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_3, kWidth = 4}>> -> tensor<64x64xf32, #mma_3>
    tt.return %dot : tensor<64x64xf32, #mma_3>
  }
}

// -----
// Test 5: one local_load feeds both a dot path and a sibling convert_layout
// path into a non-dot user. The mixed use should make the rewrite illegal, so
// the load must stay in the original blocked layout.

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
    // A has a mixed use, so it must remain blocked and keep the explicit convert for dot.
    // CHECK: %[[A_LOAD:.*]] = ttg.local_load %[[DESC_A]] : !ttg.memdesc<64x32xf16, #{{.*}}, #smem, mutable> -> tensor<64x32xf16, #{{.*}}>
    %a = ttg.local_load %buf_a : !ttg.memdesc<64x32xf16, #shared_4, #smem_4, mutable> -> tensor<64x32xf16, #blocked_4>
    // CHECK: %[[A_ALT:.*]] = ttg.convert_layout %[[A_LOAD]] : tensor<64x32xf16, #{{.*}}> -> tensor<64x32xf16, #{{.*}}>
    %a_alt = ttg.convert_layout %a : tensor<64x32xf16, #blocked_4> -> tensor<64x32xf16, #blocked_4_alt>
    // CHECK: ttg.local_store %[[A_ALT]],
    ttg.local_store %a_alt, %buf_sink : tensor<64x32xf16, #blocked_4_alt> -> !ttg.memdesc<64x32xf16, #shared_4, #smem_4, mutable>
    // B is dot-only, so the pass can still rewrite it.
    // CHECK: %[[REQ_B:.*]] = tlx.require_layout %[[DESC_B]] {{.*}} -> !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable>
    // CHECK-NEXT: %[[B_LOAD:.*]] = ttg.local_load %[[REQ_B]] : !ttg.memdesc<32x64xf16, #{{.*}}, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %b = ttg.local_load %buf_b : !ttg.memdesc<32x64xf16, #shared_4, #smem_4, mutable> -> tensor<32x64xf16, #blocked_4>
    %acc = ttg.convert_layout %cst : tensor<64x64xf32, #blocked_4> -> tensor<64x64xf32, #mma_4>
    // CHECK: %[[A_DOT:.*]] = ttg.convert_layout %[[A_LOAD]] : tensor<64x32xf16, #{{.*}}> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %a_dot = ttg.convert_layout %a : tensor<64x32xf16, #blocked_4> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_4, kWidth = 4}>>
    // CHECK: tt.dot %[[A_DOT]], %[[B_LOAD]], %{{.*}} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    %b_dot = ttg.convert_layout %b : tensor<32x64xf16, #blocked_4> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_4, kWidth = 4}>>
    %dot = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_4, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_4, kWidth = 4}>> -> tensor<64x64xf32, #mma_4>
    tt.return %dot : tensor<64x64xf32, #mma_4>
  }
}
