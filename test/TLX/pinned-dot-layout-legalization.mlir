// RUN: triton-opt -split-input-file --convert-triton-to-tritongpu="target=hip:gfx950 num-warps=8 threads-per-warp=64 num-ctas=1" %s | FileCheck %s
// RUN: triton-opt -split-input-file --convert-triton-to-tritongpu="target=hip:gfx950 num-warps=8 threads-per-warp=64 num-ctas=1" --tlx-resolve-placeholder-layouts %s | FileCheck %s --check-prefix=RESOLVE

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [8, 1], instrShape = [32, 32, 16], isTransposed = true}>
#result = #tlx.no_verify_layout<#tlx.user_layout<#mma>>
#operand_a = #tlx.no_verify_layout<#tlx.user_layout<#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>>
#operand_b = #tlx.no_verify_layout<#tlx.user_layout<#ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>>

// CHECK-DAG: #[[$PINNED_MMA:.*]] = #ttg.amd_mfma<{{.*}}>
// CHECK-DAG: #[[$PINNED_RESULT:.*]] = #tlx.user_layout<#[[$PINNED_MMA]]>
// CHECK-DAG: #[[$PINNED_A:.*]] = #tlx.user_layout<#ttg.dot_op<{opIdx = 0, parent = #[[$PINNED_MMA]], kWidth = 8}>>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // RESOLVE-DAG: #[[$MMA:.*]] = #ttg.amd_mfma<{{.*}}>
  // RESOLVE-NOT: #tlx.user_layout
  // RESOLVE-NOT: #tlx.no_verify_layout
  // CHECK-LABEL: tt.func @pinned_dot
  // CHECK-SAME: tensor<256x64xbf16, #tlx.no_verify_layout<#[[$PINNED_A]]>>
  // CHECK-SAME: -> tensor<256x64xf32, #tlx.no_verify_layout<#[[$PINNED_RESULT]]>>
  // RESOLVE-LABEL: tt.func @pinned_dot
  // RESOLVE: tt.dot {{.*}} tensor<256x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$MMA]], kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #[[$MMA]], kWidth = 8}>>
  // CHECK-NOT: #ttg.blocked
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.dot {{.*}} -> tensor<256x64xf32, #{{.*}}>
  tt.func @pinned_dot(%a: tensor<256x64xbf16, #operand_a>,
                      %b: tensor<64x64xbf16, #operand_b>,
                      %c: tensor<256x64xf32, #result>)
      -> tensor<256x64xf32, #result> {
    %dot = tt.dot %a, %b, %c : tensor<256x64xbf16, #operand_a> * tensor<64x64xbf16, #operand_b> -> tensor<256x64xf32, #result>
    tt.return %dot : tensor<256x64xf32, #result>
  }
}

// -----

// Slice/expand-dims encodings, including user pins, stay deferred through
// conversion and resolve together. Blocked layouts follow the same lifetime.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [8, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#slice = #ttg.slice<{dim = 1, parent = #dot}>
#deferred_slice = #tlx.no_verify_layout<#slice>
#deferred_user_slice = #tlx.no_verify_layout<#tlx.user_layout<#slice>>
#deferred_dot = #tlx.no_verify_layout<#dot>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
#deferred_blocked = #tlx.no_verify_layout<#tlx.user_layout<#blocked>>

// CHECK-DAG: #[[$SLICE_MMA:.*]] = #ttg.amd_mfma<{version = 4, warpsPerCTA = [8, 1], instrShape = [16, 16, 32], isTransposed = true}>
// CHECK-DAG: #[[$USER_SLICE:.*]] = #tlx.user_layout<#ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #[[$SLICE_MMA]], kWidth = 8}>}>>
// CHECK-DAG: #[[$OTHER_BLOCKED:.*]] = #ttg.blocked<{{.*}}>
// CHECK-DAG: #[[$OTHER_USER:.*]] = #tlx.user_layout<#[[$OTHER_BLOCKED]]>
// RESOLVE-DAG: #[[$SLICE_MMA:.*]] = #ttg.amd_mfma<{version = 4, warpsPerCTA = [8, 1], instrShape = [16, 16, 32], isTransposed = true}>
// RESOLVE-DAG: #[[$OTHER_BLOCKED:.*]] = #ttg.blocked<{{.*}}>
// RESOLVE-NOT: #tlx.no_verify_layout
// RESOLVE-NOT: #tlx.user_layout
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @deferred_mfma_slice_expand
  // CHECK-SAME: %[[RAW_SLICE:.*]]: tensor<256xi32, #tlx.no_verify_layout<#ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #[[$SLICE_MMA]], kWidth = 8}>}>>>
  // RESOLVE-LABEL: tt.func @deferred_mfma_slice_expand
  tt.func @deferred_mfma_slice_expand(%value: tensor<256xi32, #deferred_slice>)
      -> tensor<256x1xi32, #deferred_dot> {
    // CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[RAW_SLICE]]
    // CHECK-SAME: -> tensor<256x1xi32, #tlx.no_verify_layout<#ttg.dot_op<{opIdx = 0, parent = #[[$SLICE_MMA]], kWidth = 8}>>>
    // RESOLVE: tt.expand_dims
    // RESOLVE-SAME: tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #[[$SLICE_MMA]], kWidth = 8}>}>> -> tensor<256x1xi32, #ttg.dot_op<{opIdx = 0, parent = #[[$SLICE_MMA]], kWidth = 8}>>
    %expanded = tt.expand_dims %value {axis = 1 : i32}
        : tensor<256xi32, #deferred_slice> -> tensor<256x1xi32, #deferred_dot>
    tt.return %expanded : tensor<256x1xi32, #deferred_dot>
  }

  // CHECK-LABEL: tt.func @deferred_user_mfma_slice_expand
  // CHECK-SAME: %[[USER_VALUE:.*]]: tensor<256xi32, #tlx.no_verify_layout<#[[$USER_SLICE]]>>
  // RESOLVE-LABEL: tt.func @deferred_user_mfma_slice_expand
  tt.func @deferred_user_mfma_slice_expand(%value: tensor<256xi32, #deferred_user_slice>)
      -> tensor<256x1xi32, #deferred_dot> {
    // CHECK: tt.expand_dims %[[USER_VALUE]]
    // CHECK-SAME: tensor<256xi32, #tlx.no_verify_layout<#[[$USER_SLICE]]>> -> tensor<256x1xi32, #tlx.no_verify_layout<#ttg.dot_op<{opIdx = 0, parent = #[[$SLICE_MMA]], kWidth = 8}>>>
    // RESOLVE: tt.expand_dims
    // RESOLVE-SAME: tensor<256xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #[[$SLICE_MMA]], kWidth = 8}>}>> -> tensor<256x1xi32, #ttg.dot_op<{opIdx = 0, parent = #[[$SLICE_MMA]], kWidth = 8}>>
    %expanded = tt.expand_dims %value {axis = 1 : i32}
        : tensor<256xi32, #deferred_user_slice> -> tensor<256x1xi32, #deferred_dot>
    tt.return %expanded : tensor<256x1xi32, #deferred_dot>
  }

  // CHECK-LABEL: tt.func @keep_non_mfma_deferral
  // CHECK-SAME: tensor<128xi32, #tlx.no_verify_layout<#[[$OTHER_USER]]>>
  // CHECK-SAME: -> tensor<128xi32, #tlx.no_verify_layout<#[[$OTHER_USER]]>>
  // RESOLVE-LABEL: tt.func @keep_non_mfma_deferral
  // RESOLVE-SAME: tensor<128xi32, #[[$OTHER_BLOCKED]]>
  // RESOLVE-SAME: -> tensor<128xi32, #[[$OTHER_BLOCKED]]>
  tt.func @keep_non_mfma_deferral(%value: tensor<128xi32, #deferred_blocked>)
      -> tensor<128xi32, #deferred_blocked> {
    tt.return %value : tensor<128xi32, #deferred_blocked>
  }
}

// -----

// A user-pinned dot operand beside a bare dot operand and MFMA accumulator
// stays consistently deferred until resolution. Function types, entry
// arguments, and dense constants must agree after wrapper removal.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [8, 1], instrShape = [32, 32, 16], isTransposed = true}>
#result = #tlx.no_verify_layout<#mma>
#operand_a = #tlx.no_verify_layout<#tlx.user_layout<#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>>
#operand_b = #tlx.no_verify_layout<#ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>

// RESOLVE: #[[$RAW_MMA:.*]] = #ttg.amd_mfma<{{.*}}>
// RESOLVE-NOT: #tlx.no_verify_layout
// RESOLVE-NOT: #tlx.user_layout
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @deferred_dot
  // CHECK-SAME: tensor<256x64xbf16, #tlx.no_verify_layout<#{{[a-zA-Z0-9_]+}}>>
  // CHECK-SAME: tensor<64x64xbf16, #tlx.no_verify_layout<#ttg.dot_op<{{.*}}>>>
  // CHECK-SAME: -> tensor<256x64xf32, #tlx.no_verify_layout<#{{.*}}>>
  // RESOLVE-LABEL: tt.func @deferred_dot
  // RESOLVE-SAME: tensor<256x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$RAW_MMA]], kWidth = 8}>>
  // RESOLVE-SAME: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #[[$RAW_MMA]], kWidth = 8}>>
  // RESOLVE-SAME: -> tensor<256x64xf32, #[[$RAW_MMA]]>
  tt.func @deferred_dot(%a: tensor<256x64xbf16, #operand_a>,
                       %b: tensor<64x64xbf16, #operand_b>)
      -> tensor<256x64xf32, #result> {
    // RESOLVE: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #[[$RAW_MMA]]>
    %zero = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #result>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot {{.*}} -> tensor<256x64xf32, #tlx.no_verify_layout<#{{.*}}>>
    // RESOLVE: %[[DOT:.*]] = tt.dot {{.*}}, %[[ZERO]] : {{.*}} -> tensor<256x64xf32, #[[$RAW_MMA]]>
    %dot = tt.dot %a, %b, %zero : tensor<256x64xbf16, #operand_a> * tensor<64x64xbf16, #operand_b> -> tensor<256x64xf32, #result>
    // RESOLVE: tt.return %[[DOT]] : tensor<256x64xf32, #[[$RAW_MMA]]>
    tt.return %dot : tensor<256x64xf32, #result>
  }
}

// -----

// A partition's four-warp MFMA layout is valid even though the module uses
// eight warps. Resolution must unwrap nested slice parents and dense attributes.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1, 2], instrShape = [32, 32, 16], isTransposed = true}>
#deferred = #tlx.no_verify_layout<#mma>
#slice = #ttg.slice<{dim = 2, parent = #deferred}>

// RESOLVE: #[[$WS_MMA:.*]] = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1, 2], {{.*}}>
// RESOLVE-NOT: #tlx.no_verify_layout
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @deferred_partition_reduction
  // RESOLVE-LABEL: tt.func @deferred_partition_reduction
  tt.func @deferred_partition_reduction() {
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    // CHECK: partition0() num_warps(4)
    partition0() num_warps(4) {
      // CHECK: arith.constant dense<0.000000e+00> : tensor<2x32x64xf32, #tlx.no_verify_layout<#{{.*}}>>
      // RESOLVE: %[[VALUES:.*]] = arith.constant dense<0.000000e+00> : tensor<2x32x64xf32, #[[$WS_MMA]]>
      %values = arith.constant dense<0.000000e+00> : tensor<2x32x64xf32, #deferred>
      // RESOLVE: "tt.reduce"(%[[VALUES]])
      // RESOLVE: }) : (tensor<2x32x64xf32, #[[$WS_MMA]]>) -> tensor<2x32xf32, #ttg.slice<{dim = 2, parent = #[[$WS_MMA]]}>>
      %sum = "tt.reduce"(%values) <{axis = 2 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%lhs: f32, %rhs: f32):
        %next = arith.addf %lhs, %rhs : f32
        tt.reduce.return %next : f32
      }) : (tensor<2x32x64xf32, #deferred>) -> tensor<2x32xf32, #slice>
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
