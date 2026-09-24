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

// -----

#register_order_old = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#register_order_a = #ttg.linear<{register = [[1], [2]], lane = [[4], [8], [16], [32], [64], [128]], warp = [], block = []}>
#register_order_b = #ttg.linear<{register = [[2], [1]], lane = [[4], [8], [16], [32], [64], [128]], warp = [], block = []}>
#register_order_a_pinned = #tlx.no_verify_layout<#tlx.user_layout<#register_order_a>>
#register_order_b_pinned = #tlx.no_verify_layout<#tlx.user_layout<#register_order_b>>

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-DAG: #[[$REGISTER_ORDER_A:.*]] = #ttg.linear<{register = {{\[\[}}1], [2{{\]\]}}
  // CHECK-DAG: #[[$REGISTER_ORDER_B:.*]] = #ttg.linear<{register = {{\[\[}}2], [1{{\]\]}}
  // CHECK-LABEL: tt.func @preserve_pinned_carried_register_order
  tt.func @preserve_pinned_carried_register_order(
      %predicate: tensor<256xi1, #register_order_old>,
      %lhs: tensor<256xf32, #register_order_old>,
      %rhs: tensor<256xf32, #register_order_old>)
      -> (tensor<256xf32, #register_order_old>, tensor<256xf32, #register_order_old>) {
    // CHECK: %[[RESULT:.*]]:2 = ttg.warp_predicate {{.*}} {
    %result:2 = ttg.warp_predicate %predicate (%lhs, %rhs) {
      %lhs_native = ttg.convert_layout %lhs : tensor<256xf32, #register_order_old> -> tensor<256xf32, #register_order_a_pinned>
      %lhs_next = arith.addf %lhs_native, %lhs_native : tensor<256xf32, #register_order_a_pinned>
      %lhs_old = ttg.convert_layout %lhs_next : tensor<256xf32, #register_order_a_pinned> -> tensor<256xf32, #register_order_old>
      %rhs_native = ttg.convert_layout %rhs : tensor<256xf32, #register_order_old> -> tensor<256xf32, #register_order_b_pinned>
      %rhs_next = arith.addf %rhs_native, %rhs_native : tensor<256xf32, #register_order_b_pinned>
      %rhs_old = ttg.convert_layout %rhs_next : tensor<256xf32, #register_order_b_pinned> -> tensor<256xf32, #register_order_old>
      // CHECK: ttg.predicate_yield %{{.*}}, %{{.*}} : tensor<256xf32, #[[$REGISTER_ORDER_A]]>, tensor<256xf32, #[[$REGISTER_ORDER_B]]>
      ttg.predicate_yield %lhs_old, %rhs_old : tensor<256xf32, #register_order_old>, tensor<256xf32, #register_order_old>
    } : (tensor<256xi1, #register_order_old>, tensor<256xf32, #register_order_old>, tensor<256xf32, #register_order_old>) -> (tensor<256xf32, #register_order_old>, tensor<256xf32, #register_order_old>)
    // CHECK: } : (tensor<256xi1, #[[$REGISTER_ORDER_A]]>, tensor<256xf32, #[[$REGISTER_ORDER_A]]>, tensor<256xf32, #[[$REGISTER_ORDER_B]]>) -> (tensor<256xf32, #[[$REGISTER_ORDER_A]]>, tensor<256xf32, #[[$REGISTER_ORDER_B]]>)
    tt.return %result#0, %result#1 : tensor<256xf32, #register_order_old>, tensor<256xf32, #register_order_old>
  }
}

// -----

#nested_old = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#nested_inner = #ttg.linear<{register = [[1], [2]], lane = [[4], [8], [16], [32], [64], [128]], warp = [], block = []}>
#nested_inner_pinned = #tlx.no_verify_layout<#tlx.user_layout<#nested_inner>>

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-DAG: #[[$NESTED_INNER:.*]] = #ttg.linear<{register = {{\[\[}}1], [2{{\]\]}}
  // CHECK-LABEL: tt.func @avoid_nested_predicate_conversion
  // CHECK-SAME: %[[OUTER_PRED_ARG:.*]]: tensor<256xi1, #{{.*}}>, %[[INNER_PRED_ARG:.*]]: tensor<256xi1, #{{.*}}>, %[[VALUE_ARG:.*]]: tensor<256xf32, #{{.*}}>, %[[PTRS_ARG:.*]]: tensor<256x!tt.ptr<f32>, #{{.*}}>
  tt.func @avoid_nested_predicate_conversion(
      %outer_predicate: tensor<256xi1, #nested_old>,
      %inner_predicate: tensor<256xi1, #nested_old>,
      %value: tensor<256xf32, #nested_inner_pinned>,
      %ptrs: tensor<256x!tt.ptr<f32>, #nested_inner_pinned>) {
    // CHECK-DAG: %[[OUTER_PRED_INPUT:.*]] = ttg.convert_layout %[[OUTER_PRED_ARG]] : tensor<256xi1, #{{.*}}> -> tensor<256xi1, #[[$NESTED_INNER]]>
    // CHECK-DAG: %[[INNER_PRED_INPUT:.*]] = ttg.convert_layout %[[INNER_PRED_ARG]] : tensor<256xi1, #{{.*}}> -> tensor<256xi1, #[[$NESTED_INNER]]>
    // CHECK: ttg.warp_predicate %[[OUTER_PRED_ARG]]() {
    ttg.warp_predicate %outer_predicate () {
      // CHECK: %[[INNER_PRED:.*]] = arith.ori %[[INNER_PRED_INPUT]], %[[OUTER_PRED_INPUT]] : tensor<256xi1, #[[$NESTED_INNER]]>
      // CHECK-NOT: ttg.convert_layout %[[INNER_PRED]]
      // CHECK-NEXT: %[[INNER:.*]] = ttg.warp_predicate %[[INNER_PRED]]
      %inner_pred = arith.ori %inner_predicate, %outer_predicate : tensor<256xi1, #nested_old>
      %inner = ttg.warp_predicate %inner_pred (%value) {
        %inner_next = arith.addf %value, %value : tensor<256xf32, #nested_inner_pinned>
        ttg.predicate_yield %inner_next : tensor<256xf32, #nested_inner_pinned>
      } : (tensor<256xi1, #nested_old>, tensor<256xf32, #nested_inner_pinned>) -> tensor<256xf32, #nested_inner_pinned>
      tt.store %ptrs, %inner : tensor<256x!tt.ptr<f32>, #nested_inner_pinned>
      ttg.predicate_yield
    } : (tensor<256xi1, #nested_old>) -> ()
    // CHECK: tt.return
    tt.return
  }
}
