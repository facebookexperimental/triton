// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -tritongpu-remove-layout-conversions | FileCheck %s

#blocked_acc = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked_row = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#mma_row = #ttg.slice<{dim = 1, parent = #mma}>

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-DAG: #[[$BLOCKED_ACC:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-DAG: #[[$BLOCKED_ROW:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
  // CHECK-DAG: #[[$MMA:.*]] = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
  // CHECK-LABEL: tt.func @warp_predicate_layout_island
  // CHECK-SAME: %[[PRED_ARG:.*]]: tensor<128xi1, #[[$BLOCKED_ROW]]>
  // CHECK-SAME: %[[ACC_ARG:.*]]: tensor<128x128xf32, #[[$BLOCKED_ACC]]>
  // CHECK-SAME: %[[ROW_ARG:.*]]: tensor<128xf32, #[[$BLOCKED_ROW]]>
  tt.func @warp_predicate_layout_island(
      %predicate: tensor<128xi1, #blocked_row>,
      %acc: tensor<128x128xf32, #blocked_acc>,
      %row: tensor<128xf32, #blocked_row>)
      -> (tensor<128x128xf32, #blocked_acc>, tensor<128xf32, #blocked_row>) {
    // CHECK-DAG: %[[PRED:.*]] = ttg.convert_layout %[[PRED_ARG]] : tensor<128xi1, #[[$BLOCKED_ROW]]> -> tensor<128xi1, #ttg.slice<{dim = 1, parent = #[[$MMA]]}>>
    // CHECK-DAG: %[[ACC_INIT:.*]] = ttg.convert_layout %[[ACC_ARG]] : tensor<128x128xf32, #[[$BLOCKED_ACC]]> -> tensor<128x128xf32, #[[$MMA]]>
    // CHECK-DAG: %[[ROW_INIT:.*]] = ttg.convert_layout %[[ROW_ARG]] : tensor<128xf32, #[[$BLOCKED_ROW]]> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[$MMA]]}>>
    // CHECK: %[[RESULT:.*]]:2 = ttg.warp_predicate %[[PRED]](%[[ACC_INIT]], %[[ROW_INIT]]) {
    %result:2 = ttg.warp_predicate %predicate (%acc, %row) {
      %acc_wave = ttg.convert_layout %acc : tensor<128x128xf32, #blocked_acc> -> tensor<128x128xf32, #mma>
      %row_wave = ttg.convert_layout %row : tensor<128xf32, #blocked_row> -> tensor<128xf32, #mma_row>
      %acc_next = arith.addf %acc_wave, %acc_wave : tensor<128x128xf32, #mma>
      %row_next = arith.addf %row_wave, %row_wave : tensor<128xf32, #mma_row>
      %acc_old = ttg.convert_layout %acc_next : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked_acc>
      %row_old = ttg.convert_layout %row_next : tensor<128xf32, #mma_row> -> tensor<128xf32, #blocked_row>
      // CHECK: %[[ACC_NEXT:.*]] = arith.addf {{.*}} : tensor<128x128xf32, #[[$MMA]]>
      // CHECK: %[[ROW_NEXT:.*]] = arith.addf {{.*}} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[$MMA]]}>>
      // CHECK: ttg.predicate_yield %[[ACC_NEXT]], %[[ROW_NEXT]]
      ttg.predicate_yield %acc_old, %row_old : tensor<128x128xf32, #blocked_acc>, tensor<128xf32, #blocked_row>
    } : (tensor<128xi1, #blocked_row>, tensor<128x128xf32, #blocked_acc>, tensor<128xf32, #blocked_row>) -> (tensor<128x128xf32, #blocked_acc>, tensor<128xf32, #blocked_row>)
    // CHECK: } : (tensor<128xi1, #ttg.slice<{dim = 1, parent = #[[$MMA]]}>>, tensor<128x128xf32, #[[$MMA]]>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[$MMA]]}>>) -> (tensor<128x128xf32, #[[$MMA]]>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[$MMA]]}>>)
    // CHECK-DAG: %[[ACC_RESULT:.*]] = ttg.convert_layout %[[RESULT]]#0 : tensor<128x128xf32, #[[$MMA]]> -> tensor<128x128xf32, #[[$BLOCKED_ACC]]>
    // CHECK-DAG: %[[ROW_RESULT:.*]] = ttg.convert_layout %[[RESULT]]#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[$MMA]]}>> -> tensor<128xf32, #[[$BLOCKED_ROW]]>
    // CHECK: tt.return %[[ACC_RESULT]], %[[ROW_RESULT]]
    tt.return %result#0, %result#1 : tensor<128x128xf32, #blocked_acc>, tensor<128xf32, #blocked_row>
  }
}

// -----

#predicate_only_blocked_row = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#predicate_only_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#predicate_only_mma_row = #ttg.slice<{dim = 1, parent = #predicate_only_mma}>

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @warp_predicate_rewrites_predicate_only
  // CHECK-SAME: %[[ONLY_PRED_ARG:[^,]+]]: tensor<128xi1, #[[$ONLY_BLOCKED_ROW:[A-Za-z0-9_]+]]>
  // CHECK-SAME: %[[ONLY_ACC_ARG:[^,]+]]: tensor<128x128xf32, #[[$ONLY_MMA:[A-Za-z0-9_]+]]>
  tt.func @warp_predicate_rewrites_predicate_only(
      %predicate: tensor<128xi1, #predicate_only_blocked_row>,
      %acc: tensor<128x128xf32, #predicate_only_mma>)
      -> tensor<128x128xf32, #predicate_only_mma> {
    // CHECK: %[[ONLY_PRED:.*]] = ttg.convert_layout %[[ONLY_PRED_ARG]] : tensor<128xi1, #[[$ONLY_BLOCKED_ROW]]> -> tensor<128xi1, #ttg.slice<{dim = 1, parent = #[[$ONLY_MMA]]}>>
    // CHECK: %[[ONLY_RESULT:.*]] = ttg.warp_predicate %[[ONLY_PRED]](%[[ONLY_ACC_ARG]]) {
    %result = ttg.warp_predicate %predicate (%acc) {
      %next = arith.addf %acc, %acc : tensor<128x128xf32, #predicate_only_mma>
      ttg.predicate_yield %next : tensor<128x128xf32, #predicate_only_mma>
    } : (tensor<128xi1, #predicate_only_blocked_row>, tensor<128x128xf32, #predicate_only_mma>) -> tensor<128x128xf32, #predicate_only_mma>
    // CHECK: tt.return %[[ONLY_RESULT]]
    tt.return %result : tensor<128x128xf32, #predicate_only_mma>
  }
}
