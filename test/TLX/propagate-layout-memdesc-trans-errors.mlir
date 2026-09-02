// RUN: triton-opt -split-input-file --tlx-propagate-layout --verify-diagnostics %s

// Test that malformed ttg.memdesc_trans permutation metadata is rejected by the
// op verifier before tlx-propagate-layout can run.

#shared_src_perm = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_trans_perm = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared_req_perm = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#blocked_perm = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#smem_perm = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reject_invalid_memdesc_trans_permutation() -> tensor<128x64xf16, #blocked_perm> {
    %c0_i32 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared_src_perm, #smem_perm, mutable>
    %slice = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared_src_perm, #smem_perm, mutable> -> !ttg.memdesc<64x128xf16, #shared_src_perm, #smem_perm, mutable>
    // expected-error @+1 {{order must be a permutation}}
    %trans = ttg.memdesc_trans %slice {order = array<i32: 0, 0>} : !ttg.memdesc<64x128xf16, #shared_src_perm, #smem_perm, mutable> -> !ttg.memdesc<128x64xf16, #shared_trans_perm, #smem_perm, mutable>
    %req = tlx.require_layout %trans : !ttg.memdesc<128x64xf16, #shared_trans_perm, #smem_perm, mutable> -> !ttg.memdesc<128x64xf16, #shared_req_perm, #smem_perm, mutable>
    %val = ttg.local_load %req : !ttg.memdesc<128x64xf16, #shared_req_perm, #smem_perm, mutable> -> tensor<128x64xf16, #blocked_perm>
    tt.return %val : tensor<128x64xf16, #blocked_perm>
  }
}

// -----

// One execution predicate cannot represent two incompatible native row
// ownerships after conversions are hoisted out of the predicated region.

#old_row = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#mma_a = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#mma_b = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#row_a = #ttg.slice<{dim = 1, parent = #mma_a}>
#row_b = #ttg.slice<{dim = 1, parent = #mma_b}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @reject_conflicting_predicate_layouts(
      %predicate: tensor<128xi1, #old_row>,
      %lhs: tensor<128xf32, #old_row>,
      %rhs: tensor<128xf32, #old_row>)
      -> (tensor<128xf32, #old_row>, tensor<128xf32, #old_row>) {
    // expected-error @+1 {{carried row values require conflicting predicate layouts}}
    %result:2 = ttg.warp_predicate %predicate (%lhs, %rhs) {
      %lhs_native = ttg.convert_layout %lhs : tensor<128xf32, #old_row> -> tensor<128xf32, #row_a>
      %lhs_old = ttg.convert_layout %lhs_native : tensor<128xf32, #row_a> -> tensor<128xf32, #old_row>
      %rhs_native = ttg.convert_layout %rhs : tensor<128xf32, #old_row> -> tensor<128xf32, #row_b>
      %rhs_old = ttg.convert_layout %rhs_native : tensor<128xf32, #row_b> -> tensor<128xf32, #old_row>
      ttg.predicate_yield %lhs_old, %rhs_old : tensor<128xf32, #old_row>, tensor<128xf32, #old_row>
    } : (tensor<128xi1, #old_row>, tensor<128xf32, #old_row>, tensor<128xf32, #old_row>) -> (tensor<128xf32, #old_row>, tensor<128xf32, #old_row>)
    tt.return %result#0, %result#1 : tensor<128xf32, #old_row>, tensor<128xf32, #old_row>
  }
}
