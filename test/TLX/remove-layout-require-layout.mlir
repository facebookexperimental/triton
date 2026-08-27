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
