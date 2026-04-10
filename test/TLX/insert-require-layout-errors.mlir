// RUN: triton-opt -split-input-file --tlx-insert-require-layout --verify-diagnostics %s

// Test that InsertRequireLayout fails with a clear diagnostic when a region
// branch successor would need conflicting result types after local_load
// retagging.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @conflicting_scf_if_result_types(%cond: i1) -> tensor<64x64xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %zero_a = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared, #smem, mutable>
    %buf_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    %buf_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    // expected-error @+1 {{conflicting region-branch successor input types after InsertRequireLayout rewrite}}
    %if_a = scf.if %cond -> (tensor<64x32xf16, #blocked>) {
      %a = ttg.local_load %buf_a : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #blocked>
      scf.yield %a : tensor<64x32xf16, #blocked>
    } else {
      scf.yield %zero_a : tensor<64x32xf16, #blocked>
    }
    %b = ttg.local_load %buf_b : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> tensor<32x64xf16, #blocked>
    %a_dot = ttg.convert_layout %if_a : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %b_dot = ttg.convert_layout %b : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %dot = tt.dot %a_dot, %b_dot, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return %dot : tensor<64x64xf32, #mma>
  }
}
