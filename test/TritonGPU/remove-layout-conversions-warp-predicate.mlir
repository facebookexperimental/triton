// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -tlx-finalize-user-layouts | FileCheck %s --check-prefix=FINAL

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
    // CHECK: ttg.warp_predicate %[[OUTER_PRED_ARG]]() {
    ttg.warp_predicate %outer_predicate () {
      // CHECK: %[[INNER_PRED:.*]] = arith.ori %[[INNER_PRED_ARG]], %[[OUTER_PRED_ARG]] : tensor<256xi1, #{{.*}}>
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

// -----

#remat_old = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#remat_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#remat_slice = #ttg.slice<{dim = 1, parent = #remat_mma}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @rematerializable_carried_init
  tt.func @rematerializable_carried_init(%ptr: !tt.ptr<i32>) {
    // CHECK: %[[PID:.*]] = tt.get_program_id x : i32
    %pid = tt.get_program_id x : i32
    // CHECK: %[[INIT:.*]] = tt.splat %[[PID]] : i32 -> tensor<128xi32, #ttg.slice<{{.*}}>>
    %init = tt.splat %pid : i32 -> tensor<128xi32, #remat_old>
    %true = arith.constant true
    // CHECK: %[[RESULT:.*]] = ttg.warp_predicate %true(%[[INIT]]) {
    // CHECK-NOT: ttg.convert_layout
    %result = ttg.warp_predicate %true (%init) {
      %native = ttg.convert_layout %init : tensor<128xi32, #remat_old> -> tensor<128xi32, #remat_slice>
      %next = arith.addi %native, %native : tensor<128xi32, #remat_slice>
      %old = ttg.convert_layout %next : tensor<128xi32, #remat_slice> -> tensor<128xi32, #remat_old>
      tt.store %ptr, %pid : !tt.ptr<i32>
      // CHECK: ttg.predicate_yield %{{.*}} : tensor<128xi32, #ttg.slice<{{.*}}>>
      ttg.predicate_yield %old : tensor<128xi32, #remat_old>
    } : (i1, tensor<128xi32, #remat_old>) -> tensor<128xi32, #remat_old>
    // CHECK: } : (i1, tensor<128xi32, #ttg.slice<{{.*}}>>) -> tensor<128xi32, #ttg.slice<{{.*}}>>
    tt.return
  }
}

// -----

#equiv_blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#equiv_linear = #ttg.linear<{register = [[1], [2]], lane = [[4], [8], [16], [32], [64], [128]], warp = [], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @nested_equivalent_carrier_layouts
  // CHECK-SAME: %[[OUTER_PRED:.*]]: i1, %[[INNER_PRED:.*]]: i1, %[[INIT_ARG:.*]]: tensor<256xf32, #[[$EQUIV_BLOCKED:.*]]>, %[[PTR:.*]]: !tt.ptr<i32>
  tt.func @nested_equivalent_carrier_layouts(
      %outer_predicate: i1, %inner_predicate: i1,
      %init: tensor<256xf32, #equiv_blocked>, %ptr: !tt.ptr<i32>) {
    // CHECK: %[[LINEAR_INIT:.*]] = ttg.convert_layout %[[INIT_ARG]] : tensor<256xf32, #[[$EQUIV_BLOCKED]]> -> tensor<256xf32, #[[$EQUIV_LINEAR:.*]]>
    %pid = tt.get_program_id x : i32
    // CHECK: %[[OUTER:.*]] = ttg.warp_predicate %[[OUTER_PRED]](%[[LINEAR_INIT]]) {
    // CHECK-NOT: ttg.convert_layout
    %outer = ttg.warp_predicate %outer_predicate (%init) {
      // CHECK: %[[INNER:.*]] = ttg.warp_predicate %[[INNER_PRED]](%[[LINEAR_INIT]]) {
      // CHECK-NOT: ttg.convert_layout
      %inner = ttg.warp_predicate %inner_predicate (%init) {
        %native = ttg.convert_layout %init : tensor<256xf32, #equiv_blocked> -> tensor<256xf32, #equiv_linear>
        %next = arith.addf %native, %native : tensor<256xf32, #equiv_linear>
        %old = ttg.convert_layout %next : tensor<256xf32, #equiv_linear> -> tensor<256xf32, #equiv_blocked>
        // CHECK: ttg.predicate_yield %{{.*}} : tensor<256xf32, #[[$EQUIV_LINEAR]]>
        ttg.predicate_yield %old : tensor<256xf32, #equiv_blocked>
      } : (i1, tensor<256xf32, #equiv_blocked>) -> tensor<256xf32, #equiv_blocked>
      // CHECK: } : (i1, tensor<256xf32, #[[$EQUIV_LINEAR]]>) -> tensor<256xf32, #[[$EQUIV_LINEAR]]>
      tt.store %ptr, %pid : !tt.ptr<i32>
      // CHECK-NOT: ttg.convert_layout
      // CHECK: ttg.predicate_yield %[[INNER]] : tensor<256xf32, #[[$EQUIV_LINEAR]]>
      ttg.predicate_yield %inner : tensor<256xf32, #equiv_blocked>
    } : (i1, tensor<256xf32, #equiv_blocked>) -> tensor<256xf32, #equiv_blocked>
    // CHECK: } : (i1, tensor<256xf32, #[[$EQUIV_LINEAR]]>) -> tensor<256xf32, #[[$EQUIV_LINEAR]]>
    tt.return
  }
}

// -----

#nested_init_blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#nested_init_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#nested_init_slice = #ttg.slice<{dim = 1, parent = #nested_init_mma}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @force_nested_carried_init
  // CHECK-SAME: %[[OUTER_PRED:.*]]: i1, %[[INNER_PRED:.*]]: i1, %[[VALUE:.*]]: tensor<128xf32, #[[$INIT_BLOCKED:.*]]>, %[[NATIVE:.*]]: tensor<128xf32, #ttg.slice<{{.*}}>>, %[[PTR:.*]]: !tt.ptr<i32>
  tt.func @force_nested_carried_init(
      %outer_predicate: i1, %inner_predicate: i1,
      %value: tensor<128xf32, #nested_init_blocked>,
      %native: tensor<128xf32, #nested_init_slice>, %ptr: !tt.ptr<i32>) {
    // CHECK: %[[FORCED_VALUE:.*]] = ttg.convert_layout %[[VALUE]] : tensor<128xf32, #[[$INIT_BLOCKED]]> -> tensor<128xf32, #[[$INIT_NATIVE:.*]]>
    %pid = tt.get_program_id x : i32
    // CHECK: ttg.warp_predicate %[[OUTER_PRED]]() {
    // CHECK-NOT: ttg.convert_layout
    ttg.warp_predicate %outer_predicate () {
      // CHECK: %[[INNER_INIT:.*]] = arith.addf %[[FORCED_VALUE]], %[[FORCED_VALUE]] : tensor<128xf32, #[[$INIT_NATIVE]]>
      %inner_init = arith.addf %value, %value : tensor<128xf32, #nested_init_blocked>
      // CHECK-NEXT: %[[INNER:.*]] = ttg.warp_predicate %[[INNER_PRED]](%[[INNER_INIT]]) {
      // CHECK-NOT: ttg.convert_layout
      %inner = ttg.warp_predicate %inner_predicate (%inner_init) {
        %old = ttg.convert_layout %native : tensor<128xf32, #nested_init_slice> -> tensor<128xf32, #nested_init_blocked>
        tt.store %ptr, %pid : !tt.ptr<i32>
        // CHECK: ttg.predicate_yield %[[NATIVE]] : tensor<128xf32, #[[$INIT_NATIVE]]>
        ttg.predicate_yield %old : tensor<128xf32, #nested_init_blocked>
      } : (i1, tensor<128xf32, #nested_init_blocked>) -> tensor<128xf32, #nested_init_blocked>
      // CHECK: } : (i1, tensor<128xf32, #[[$INIT_NATIVE]]>) -> tensor<128xf32, #[[$INIT_NATIVE]]>
      // CHECK-NOT: ttg.convert_layout
      // CHECK: ttg.predicate_yield
      ttg.predicate_yield
    } : (i1) -> ()
    tt.return
  }
}

// -----

#uniform_blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#uniform_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#uniform_mma_row = #ttg.slice<{dim = 1, parent = #uniform_mma}>
#uniform_blocked_pinned = #tlx.no_verify_layout<#tlx.user_layout<#uniform_blocked>>
#uniform_mma_pinned = #tlx.no_verify_layout<#tlx.user_layout<#uniform_mma_row>>

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @wave_uniform_allows_cross_lane_conversion
  // CHECK-SAME: %[[PRED:.*]]: i1, %[[INIT:.*]]: tensor<128xf32, #{{.*}}>
  tt.func @wave_uniform_allows_cross_lane_conversion(
      %predicate: i1, %init: tensor<128xf32, #uniform_blocked_pinned>)
      -> tensor<128xf32, #uniform_blocked_pinned> {
    // CHECK: %[[RESULT:.*]] = ttg.warp_predicate %[[PRED]](%[[INIT]]) {
    // CHECK: %[[NATIVE:.*]] = ttg.convert_layout %[[INIT]] : tensor<128xf32, #{{.*}}> -> tensor<128xf32, #{{.*}}>
    // CHECK: %[[NEXT:.*]] = arith.addf %[[NATIVE]], %[[NATIVE]]
    // CHECK: %[[RESTORED:.*]] = ttg.convert_layout %[[NEXT]] : tensor<128xf32, #{{.*}}> -> tensor<128xf32, #{{.*}}>
    // CHECK: ttg.predicate_yield %[[RESTORED]]
    %result = ttg.warp_predicate %predicate (%init) {
      %native = ttg.convert_layout %init : tensor<128xf32, #uniform_blocked_pinned> -> tensor<128xf32, #uniform_mma_pinned>
      %next = arith.addf %native, %native : tensor<128xf32, #uniform_mma_pinned>
      %restored = ttg.convert_layout %next : tensor<128xf32, #uniform_mma_pinned> -> tensor<128xf32, #uniform_blocked_pinned>
      ttg.predicate_yield %restored : tensor<128xf32, #uniform_blocked_pinned>
    } {wave_uniform} : (i1, tensor<128xf32, #uniform_blocked_pinned>) -> tensor<128xf32, #uniform_blocked_pinned>
    // CHECK: } {wave_uniform} :
    // CHECK: tt.return %[[RESULT]]
    tt.return %result : tensor<128xf32, #uniform_blocked_pinned>
  }
}

// -----

#nested_uniform_blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#nested_uniform_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#nested_uniform_mma_row = #ttg.slice<{dim = 1, parent = #nested_uniform_mma}>

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @nested_wave_uniform_inherits_restricted_exec
  // CHECK-SAME: %[[OUTER_PRED:.*]]: i1, %[[INNER_PRED:.*]]: i1, %[[INIT:.*]]: tensor<128xf32, #[[$NESTED_BLOCKED:.*]]>, %[[NATIVE:.*]]: tensor<128xf32, #[[$NESTED_NATIVE:.*]]>, %[[PTR:.*]]: !tt.ptr<i32>
  tt.func @nested_wave_uniform_inherits_restricted_exec(
      %outer_predicate: i1, %inner_predicate: i1,
      %init: tensor<128xf32, #nested_uniform_blocked>,
      %native: tensor<128xf32, #nested_uniform_mma_row>, %ptr: !tt.ptr<i32>) {
    // CHECK: %[[FORCED_INIT:.*]] = ttg.convert_layout %[[INIT]] : tensor<128xf32, #[[$NESTED_BLOCKED]]> -> tensor<128xf32, #[[$NESTED_NATIVE]]>
    %pid = tt.get_program_id x : i32
    // CHECK: ttg.warp_predicate %[[OUTER_PRED]]() {
    // CHECK-NOT: ttg.convert_layout
    ttg.warp_predicate %outer_predicate () {
      %inner_init = arith.addf %init, %init : tensor<128xf32, #nested_uniform_blocked>
      // CHECK: %[[INNER:.*]] = ttg.warp_predicate %[[INNER_PRED]](%{{.*}}) {
      // CHECK-NOT: ttg.convert_layout
      %inner = ttg.warp_predicate %inner_predicate (%inner_init) {
        %restored = ttg.convert_layout %native : tensor<128xf32, #nested_uniform_mma_row> -> tensor<128xf32, #nested_uniform_blocked>
        tt.store %ptr, %pid : !tt.ptr<i32>
        // CHECK: ttg.predicate_yield %[[NATIVE]] : tensor<128xf32, #[[$NESTED_NATIVE]]>
        ttg.predicate_yield %restored : tensor<128xf32, #nested_uniform_blocked>
      } {wave_uniform} : (i1, tensor<128xf32, #nested_uniform_blocked>) -> tensor<128xf32, #nested_uniform_blocked>
      // CHECK: } {wave_uniform} : (i1, tensor<128xf32, #[[$NESTED_NATIVE]]>) -> tensor<128xf32, #[[$NESTED_NATIVE]]>
      // CHECK-NOT: ttg.convert_layout
      ttg.predicate_yield
    } : (i1) -> ()
    // CHECK: tt.return
    tt.return
  }
}

// -----

#release_blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#release_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#release_mma_row = #ttg.slice<{dim = 1, parent = #release_mma}>
#release_native = #tlx.user_layout<#release_mma_row>

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // FINAL-LABEL: tt.func @strict_release_finalizes_outside_restricted_exec
  // FINAL-SAME: %[[PRED:.*]]: i1, %[[INIT:.*]]: tensor<128xf32, #[[$RELEASE_BLOCKED:.*]]>, %[[NATIVE:.*]]: tensor<128xf32, #{{.*}}>
  tt.func @strict_release_finalizes_outside_restricted_exec(
      %predicate: i1, %init: tensor<128xf32, #release_blocked>,
      %native: tensor<128xf32, #release_native>)
      -> tensor<128xf32, #release_blocked> {
    // FINAL: %[[CONVERTED_INIT:.*]] = ttg.convert_layout %[[INIT]] : tensor<128xf32, #[[$RELEASE_BLOCKED]]> -> tensor<128xf32, #[[$RELEASE_NATIVE:.*]]>
    // FINAL: %[[RESULT:.*]] = ttg.warp_predicate %[[PRED]](%[[CONVERTED_INIT]]) {
    // FINAL-NOT: ttg.convert_layout
    // FINAL-NOT: ttg.release_layout
    %result = ttg.warp_predicate %predicate (%init) {
      %released = ttg.release_layout %native : tensor<128xf32, #release_native> -> tensor<128xf32, #release_blocked>
      // FINAL: ttg.predicate_yield %[[NATIVE]] : tensor<128xf32, #[[$RELEASE_NATIVE]]>
      ttg.predicate_yield %released : tensor<128xf32, #release_blocked>
    } : (i1, tensor<128xf32, #release_blocked>) -> tensor<128xf32, #release_blocked>
    // FINAL: %[[RESTORED:.*]] = ttg.convert_layout %[[RESULT]] : tensor<128xf32, #[[$RELEASE_NATIVE]]> -> tensor<128xf32, #[[$RELEASE_BLOCKED]]>
    // FINAL: tt.return %[[RESTORED]]
    tt.return %result : tensor<128xf32, #release_blocked>
  }

  // FINAL-LABEL: tt.func @strict_release_without_carried_result
  // FINAL-SAME: %[[SIDE_PRED:.*]]: i1, %[[SIDE_NATIVE:.*]]: tensor<128xf32, #{{.*}}>, %[[PTRS:.*]]: tensor<128x!tt.ptr<f32>, #[[$RELEASE_BLOCKED]]>
  tt.func @strict_release_without_carried_result(
      %predicate: i1, %native: tensor<128xf32, #release_native>,
      %ptrs: tensor<128x!tt.ptr<f32>, #release_blocked>) {
    // FINAL: %[[RELEASED:.*]] = ttg.convert_layout %[[SIDE_NATIVE]] : tensor<128xf32, #{{.*}}> -> tensor<128xf32, #[[$RELEASE_BLOCKED]]>
    // FINAL: ttg.warp_predicate %[[SIDE_PRED]]() {
    // FINAL-NOT: ttg.convert_layout
    // FINAL-NOT: ttg.release_layout
    ttg.warp_predicate %predicate () {
      %released = ttg.release_layout %native : tensor<128xf32, #release_native> -> tensor<128xf32, #release_blocked>
      // FINAL: tt.store %[[PTRS]], %[[RELEASED]]
      tt.store %ptrs, %released : tensor<128x!tt.ptr<f32>, #release_blocked>
      ttg.predicate_yield
    } : (i1) -> ()
    // FINAL: tt.return
    tt.return
  }

  // FINAL-LABEL: tt.func @strict_release_tensor_predicate_without_carried_result
  // FINAL-SAME: %[[TENSOR_PRED:.*]]: tensor<128xi1, #[[$RELEASE_BLOCKED]]>, %[[TENSOR_NATIVE:.*]]: tensor<128xf32, #{{.*}}>, %[[TENSOR_PTRS:.*]]: tensor<128x!tt.ptr<f32>, #[[$RELEASE_BLOCKED]]>
  tt.func @strict_release_tensor_predicate_without_carried_result(
      %predicate: tensor<128xi1, #release_blocked>,
      %native: tensor<128xf32, #release_native>,
      %ptrs: tensor<128x!tt.ptr<f32>, #release_blocked>) {
    // FINAL: %[[TENSOR_RELEASED:.*]] = ttg.convert_layout %[[TENSOR_NATIVE]] : tensor<128xf32, #{{.*}}> -> tensor<128xf32, #[[$RELEASE_BLOCKED]]>
    // FINAL: ttg.warp_predicate %[[TENSOR_PRED]]() {
    // FINAL-NOT: ttg.convert_layout
    // FINAL-NOT: ttg.release_layout
    ttg.warp_predicate %predicate () {
      %released = ttg.release_layout %native : tensor<128xf32, #release_native> -> tensor<128xf32, #release_blocked>
      // FINAL: tt.store %[[TENSOR_PTRS]], %[[TENSOR_RELEASED]]
      tt.store %ptrs, %released : tensor<128x!tt.ptr<f32>, #release_blocked>
      ttg.predicate_yield
    } : (tensor<128xi1, #release_blocked>) -> ()
    // FINAL: tt.return
    tt.return
  }

  // FINAL-LABEL: tt.func @require_finalizes_outside_restricted_exec
  // FINAL-SAME: %[[REQUIRE_PRED:.*]]: i1, %[[REQUIRE_SRC:.*]]: tensor<128xf32, #[[$RELEASE_BLOCKED]]>, %[[REQUIRE_PTRS:.*]]: tensor<128x!tt.ptr<f32>, #{{.*}}>
  tt.func @require_finalizes_outside_restricted_exec(
      %predicate: i1, %src: tensor<128xf32, #release_blocked>,
      %ptrs: tensor<128x!tt.ptr<f32>, #release_native>) {
    // FINAL: %[[REQUIRED:.*]] = ttg.convert_layout %[[REQUIRE_SRC]]
    // FINAL: ttg.warp_predicate %[[REQUIRE_PRED]]() {
    // FINAL-NOT: ttg.convert_layout
    // FINAL-NOT: ttg.require_layout
    ttg.warp_predicate %predicate () {
      %required = ttg.require_layout %src : tensor<128xf32, #release_blocked> -> tensor<128xf32, #release_native>
      // FINAL: tt.store %[[REQUIRE_PTRS]], %[[REQUIRED]]
      tt.store %ptrs, %required : tensor<128x!tt.ptr<f32>, #release_native>
      ttg.predicate_yield
    } : (i1) -> ()
    // FINAL: tt.return
    tt.return
  }
}

// -----

#reshape_src = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[32, 0], [64, 0], [16, 0]], block = []}>
#reshape_dst = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
#reshape_wrapped_dst = #tlx.no_verify_layout<#tlx.user_layout<#reshape_dst>>

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 32 : i32} {
  // FINAL-LABEL: tt.func @reshape_repair_stays_outside_restricted_exec
  // FINAL-SAME: %[[RESHAPE_PRED:.*]]: i1, %[[RESHAPE_SRC:.*]]: tensor<128x128xf32, #{{.*}}>, %[[RESHAPE_PTRS:.*]]: tensor<128x2x64x!tt.ptr<f32>, #[[$RESHAPE_DST:.*]]>
  tt.func @reshape_repair_stays_outside_restricted_exec(
      %predicate: i1, %src: tensor<128x128xf32, #reshape_src>,
      %ptrs: tensor<128x2x64x!tt.ptr<f32>, #reshape_wrapped_dst>) {
    // FINAL: %[[COMPATIBLE_SRC:.*]] = ttg.convert_layout %[[RESHAPE_SRC]]
    // FINAL: ttg.warp_predicate %[[RESHAPE_PRED]]() {
    // FINAL-NOT: ttg.convert_layout
    ttg.warp_predicate %predicate () {
      // FINAL: %[[RESHAPED:.*]] = tt.reshape %[[COMPATIBLE_SRC]] : {{.*}} -> tensor<128x2x64xf32, #[[$RESHAPE_DST]]>
      %reshaped = tt.reshape %src : tensor<128x128xf32, #reshape_src> -> tensor<128x2x64xf32, #reshape_wrapped_dst>
      // FINAL: tt.store %[[RESHAPE_PTRS]], %[[RESHAPED]]
      tt.store %ptrs, %reshaped : tensor<128x2x64x!tt.ptr<f32>, #reshape_wrapped_dst>
      ttg.predicate_yield
    } : (i1) -> ()
    // FINAL: tt.return
    tt.return
  }
}
