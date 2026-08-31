// RUN: triton-opt --split-input-file %s --verify-diagnostics

// A pinned layout is still subject to the physical TDM layout constraints.
#tdm_bad_swizzle = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>
#tdm_bad_pinned = #tlx.user_layout<#tdm_bad_swizzle>
#tdm_bad_smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_pinned_layout_still_validated(
      %desc: !tt.tensordesc<32x32xf16>,
      %buf: !ttg.memdesc<32x32xf16, #tdm_bad_pinned, #tdm_bad_smem, mutable>) {
    // expected-error @+1 {{TDM does not support swizzling}}
    %token = amdg.async_tdm_copy_global_to_local %desc into %buf : !tt.tensordesc<32x32xf16> -> !ttg.memdesc<32x32xf16, #tdm_bad_pinned, #tdm_bad_smem, mutable>
    tt.return
  }
}

// -----

// expected-error @+1 {{WMMA version must be in the [1, 3] range}}
#wmma = #ttg.amd_wmma<{version = 0, isTranspose = false, ctaLayout = {warp = [[0, 1], [1, 0]]}}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<32x32x!tt.ptr<i32,1>, #wmma>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [1, 0], [2, 0]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_output_encoding(%arg0: tensor<32x32xf16, #blocked>) {
// expected-error-re @+15 {{Expect output layout to be transposed per thread:{{.*}}- register=1 -> (1, 0){{.*}}register=2 -> (2, 0){{.*}}register=4 -> (0, 1){{.*}}register=8 -> (0, 2)}}
// Full expected layout is following:
// - register=1 -> (1, 0)
//   register=2 -> (2, 0)
//   register=4 -> (0, 1)
//   register=8 -> (0, 2)}}
// - lane=1 -> (0, 4)
//   lane=2 -> (0, 8)
//   lane=4 -> (0, 16)
//   lane=8 -> (4, 0)
//   lane=16 -> (8, 0)
//   lane=32 -> (16, 0)
// - warp is a size 1 dimension
// - block is a size 1 dimension
// where out dims are: [dim0 (size 32), dim1 (size 32)]
    %0 = amdg.in_thread_transpose %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_input_encoding(%arg0: tensor<32x32xf16, #mfma>) {
// expected-error @+1 {{Expect input tensor in Blocked encoding}}
    %0 = amdg.in_thread_transpose %arg0 : tensor<32x32xf16, #mfma> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_shape(%arg0: tensor<64x64xf16, #blocked>) {
// expected-error @+1 {{Expect equal input and output shapes}}
    %0 = amdg.in_thread_transpose %arg0 : tensor<64x64xf16, #blocked> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_dtype(%arg0: tensor<32x32xf16, #blocked>) {
// expected-error @+1 {{Expect input and output tensor to have same dtype}}
    %0 = amdg.in_thread_transpose %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf32, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4, 4], threadsPerWarp = [1, 8, 8], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 0, 1], [0, 0, 2]], lane = [[0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_3d_shape(%arg0: tensor<2x32x32xf16, #blocked>) {
// expected-error @+1 {{Expect 2d tensor}}
    %0 = amdg.in_thread_transpose %arg0 : tensor<2x32x32xf16, #blocked> -> tensor<2x32x32xf16, #linear>
    tt.return
  }
}

// -----

#mma32 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @local_load_packed_tranposed_wrong_op_idx(%arg0: !ttg.memdesc<16x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x16xi8, #shared1, #smem, mutable>) {
// expected-error @+1 {{Order of dimensions don't match expected}}
    %1 = amdg.local_load_packed_tranposed %arg0 : !ttg.memdesc<16x64xi8, #shared, #smem, mutable> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  tt.func @local_load_packed_tranposed_wrong_op_idx2(%arg0: !ttg.memdesc<64x16xi8, #shared, #smem, mutable>) {
// expected-error @+1 {{Input and output dimensions don't match after packing changes}}
    %1 = amdg.local_load_packed_tranposed %arg0 : !ttg.memdesc<64x16xi8, #shared, #smem, mutable> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.return
  }
  //  CHECK-LABEL: ds_transpose_t_fp4_mfma16
  tt.func @local_load_packed_tranposed_wrong_shape(%arg0: !ttg.memdesc<8x128xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x8xi8, #shared1, #smem, mutable>) {
// expected-error @+1 {{only works with DotOperandEncodingAttr dst encoding}}
    %1 = amdg.local_load_packed_tranposed %arg0 : !ttg.memdesc<8x128xi8, #shared, #smem, mutable> -> tensor<256x128xi32, #blocked>
    tt.return
  }

}

// -----

#wmma_v3 = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#wmma_v2 = #ttg.amd_wmma<{version = 2, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#wmma_diff_warp = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [0, 0]]}, instrShape = [16, 16, 32]}>
#wmma_diff_shape = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 64]}>
#wmma_diff_transpose = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 64], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wmma_dot_incompatible_versions(
              %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>>,
              %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_v2, kWidth = 8}>>,
              %dst: tensor<16x16xf32, #wmma_v3>
  ) {
    // expected-error @+2 {{'tt.dot' op failed to infer returned types}}
    // expected-error @+1 {{Incompatible parent encoding}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_v2, kWidth = 8}>> -> tensor<16x16xf32, #wmma_v3>
    tt.return
  }

  tt.func @wmma_dot_incompatible_warp_layouts(
              %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>>,
              %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_warp, kWidth = 8}>>,
              %dst: tensor<16x16xf32, #wmma_v3>
  ) {
    // expected-error @+2 {{'tt.dot' op failed to infer returned types}}
    // expected-error @+1 {{Incompatible parent encoding}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_warp, kWidth = 8}>> -> tensor<16x16xf32, #wmma_v3>
    tt.return
  }

  tt.func @wmma_dot_incomptible_shapes(
              %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>>,
              %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_shape, kWidth = 8}>>,
              %dst: tensor<16x16xf32, #wmma_v3>
  ) {
    // expected-error @+2 {{'tt.dot' op failed to infer returned types}}
    // expected-error @+1 {{Incompatible parent encoding}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_shape, kWidth = 8}>> -> tensor<16x16xf32, #wmma_v3>
    tt.return
  }

  tt.func @wmma_dot_incomptible_transpose(
              %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>>,
              %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_transpose, kWidth = 8}>>,
              %dst: tensor<16x16xf32, #wmma_v3>
  ) {
    // expected-error @+2 {{'tt.dot' op failed to infer returned types}}
    // expected-error @+1 {{Incompatible parent encoding}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_transpose, kWidth = 8}>> -> tensor<16x16xf32, #wmma_v3>
    tt.return
  }
}

// -----

#wmma_acc = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, CGALayout = [[1, 0], [0, 1]], instrShape = [16, 16, 32]}>
#wmma_dim1 = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, CGALayout = [[1, 0], [0, 0]], instrShape = [16, 16, 32]}>
#wmma_dim2 = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, CGALayout = [[0, 0], [0, 1]], instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wmma_invalid_cga_split_operand_0(
              %arg0: tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim2, kWidth = 8}>>,
              %arg1: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim2, kWidth = 8}>>,
              %dst: tensor<32x32xf32, #wmma_acc>
  ) {
    // expected-error @+1 {{Incompatible CGA layout for operand 0}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim2, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim2, kWidth = 8}>> -> tensor<32x32xf32, #wmma_acc>
    tt.return
  }

  tt.func @wmma_invalid_cga_split_operand_1(
              %arg0: tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim1, kWidth = 8}>>,
              %arg1: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim1, kWidth = 8}>>,
              %dst: tensor<32x32xf32, #wmma_acc>
  ) {
    // expected-error @+1 {{Incompatible CGA layout for operand 1}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim1, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim1, kWidth = 8}>> -> tensor<32x32xf32, #wmma_acc>
    tt.return
  }

  tt.func @wmma_invalid_cga_split_accumulator(
              %arg0: tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim2, kWidth = 8}>>,
              %arg1: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim2, kWidth = 8}>>,
              %dst: tensor<32x32xf32, #wmma_dim1>
  ) {
    // expected-error @+1 {{Accumulator CGA layout should not broadcast or have repeated rows}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim2, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim2, kWidth = 8}>> -> tensor<32x32xf32, #wmma_dim1>
    tt.return
  }
}

// -----

#shared_32 = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [128, 64]}>
#shared_2_intervals = #ttg.padded_shared<[64:+4, 128:+4] {order = [1, 0], shape = [128, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @interval_not_matching_innermost_block_dimension(
    %tensorDesc: !tt.tensordesc<128x64xf16>,
    %memDesc: !ttg.memdesc<128x64xf16, #shared_32, #smem, mutable>
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{TDM store padding is only supported when padding interval equals the innermost block dimension}}
    amdg.async_tdm_copy_local_to_global %tensorDesc from %memDesc: !ttg.memdesc<128x64xf16, #shared_32, #smem, mutable> -> !tt.tensordesc<128x64xf16>
    tt.return
  }

  tt.func public @tdm_store_two_padding_intervals(
    %tensorDesc: !tt.tensordesc<128x64xf16>,
    %memDesc: !ttg.memdesc<128x64xf16, #shared_2_intervals, #smem, mutable>
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{TDM store only supports single interval paddings}}
    amdg.async_tdm_copy_local_to_global %tensorDesc from %memDesc: !ttg.memdesc<128x64xf16, #shared_2_intervals, #smem, mutable> -> !tt.tensordesc<128x64xf16>
    tt.return
  }

  tt.func public @tdm_load_two_padding_intervals(
    %tensorDesc: !tt.tensordesc<128x64xf16>,
    %memDesc: !ttg.memdesc<128x64xf16, #shared_2_intervals, #smem, mutable>
  ) {
    // expected-error @+1 {{TDM load only supports a single interval-padding pair}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc : !tt.tensordesc<128x64xf16> -> !ttg.memdesc<128x64xf16, #shared_2_intervals, #smem, mutable>
    tt.return
  }
}

// -----

// Fused TDM verifier coverage.
#fused = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @fused_tdm_overlapping_hints(
      %a: !tt.tensordesc<64x64xf16>, %b: !tt.tensordesc<64x64xf16>,
      %da: !ttg.memdesc<64x64xf16, #fused, #smem, mutable>,
      %db: !ttg.memdesc<64x64xf16, #fused, #smem, mutable>) {
    // expected-error @+1 {{requires pairwise-disjoint warp_used_hint values}}
    %0 = amdg.async_tdm_fused_copy_global_to_local %a, %b into %da, %db {warp_used_hints = array<i32: 3, 3>} : !tt.tensordesc<64x64xf16>, !tt.tensordesc<64x64xf16> -> !ttg.memdesc<64x64xf16, #fused, #smem, mutable>, !ttg.memdesc<64x64xf16, #fused, #smem, mutable>
    tt.return
  }
}

// -----

#fused = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @fused_tdm_rank_mismatch(
      %a: !tt.tensordesc<64x64xf16>, %b: !tt.tensordesc<64x64x1xf16>,
      %da: !ttg.memdesc<64x64xf16, #fused, #smem, mutable>,
      %db: !ttg.memdesc<64x64x1xf16, #fused, #smem, mutable>) {
    // expected-error @+1 {{requires all member descriptors to have the same rank}}
    %0 = amdg.async_tdm_fused_copy_global_to_local %a, %b into %da, %db {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x64xf16>, !tt.tensordesc<64x64x1xf16> -> !ttg.memdesc<64x64xf16, #fused, #smem, mutable>, !ttg.memdesc<64x64x1xf16, #fused, #smem, mutable>
    tt.return
  }
}

// -----

#fused = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @fused_tdm_element_width_mismatch(
      %a: !tt.tensordesc<64x64xf16>, %b: !tt.tensordesc<64x64xf16>,
      %da: !ttg.memdesc<64x64xf32, #fused, #smem, mutable>,
      %db: !ttg.memdesc<64x64xf16, #fused, #smem, mutable>) {
    // expected-error @+1 {{requires each descriptor and its destination to have the same element bitwidth}}
    %0 = amdg.async_tdm_fused_copy_global_to_local %a, %b into %da, %db {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x64xf16>, !tt.tensordesc<64x64xf16> -> !ttg.memdesc<64x64xf32, #fused, #smem, mutable>, !ttg.memdesc<64x64xf16, #fused, #smem, mutable>
    tt.return
  }
}

// -----

#fused = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @fused_tdm_one_member(
      %desc: !tt.tensordesc<64x64xf16>,
      %dst: !ttg.memdesc<64x64xf16, #fused, #smem, mutable>) {
    // expected-error @+1 {{requires 2 to 4 members}}
    %0 = amdg.async_tdm_fused_copy_global_to_local %desc into %dst {warp_used_hints = array<i32: 15>} : !tt.tensordesc<64x64xf16> -> !ttg.memdesc<64x64xf16, #fused, #smem, mutable>
    tt.return
  }
}

// -----

#multi_pad = #ttg.padded_shared<[32:+4, 64:+4] {order = [1, 0], shape = [64, 64]}>
#fused = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @fused_tdm_multi_padding_pairs(
      %a: !tt.tensordesc<64x64xf16>, %b: !tt.tensordesc<64x64xf16>,
      %da: !ttg.memdesc<64x64xf16, #multi_pad, #smem, mutable>,
      %db: !ttg.memdesc<64x64xf16, #fused, #smem, mutable>) {
    // expected-error @+1 {{TDM load only supports a single interval-padding pair}}
    %0 = amdg.async_tdm_fused_copy_global_to_local %a, %b into %da, %db {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x64xf16>, !tt.tensordesc<64x64xf16> -> !ttg.memdesc<64x64xf16, #multi_pad, #smem, mutable>, !ttg.memdesc<64x64xf16, #fused, #smem, mutable>
    tt.return
  }
}

// -----

#nondistributed = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @register_resident_requires_distributed_encoding(
      %arg0: tensor<16x16xbf16, #nondistributed>) {
    // expected-error @+1 {{requires a distributed tensor encoding}}
    %0 = amdg.register_resident %arg0 class "agpr" groups 1
        : tensor<16x16xbf16, #nondistributed>
    tt.return
  }
}

// -----

#distributed = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @register_resident_rejects_pointer_elements(
      %arg0: tensor<64x!tt.ptr<f32>, #distributed>) {
    // expected-error @+1 {{requires an integer or floating-point element type}}
    %0 = amdg.register_resident %arg0 class "agpr" groups 1
        : tensor<64x!tt.ptr<f32>, #distributed>
    tt.return
  }
}

// -----

#handoff_distributed = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @register_handoff_requires_complete_native_groups(
      %arg0: tensor<64xf16, #handoff_distributed>) {
    // expected-error @+1 {{requires 1 elements per thread to be divisible by the 2-element native tuple}}
    %0 = amdg.register_handoff %arg0 class "vgpr"
        : tensor<64xf16, #handoff_distributed>
    tt.return
  }
}

// -----

#distributed = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_atomic_rmw_rejects_i16(
      %ptr: !tt.ptr<i16>, %offsets: tensor<64xi32, #distributed>,
      %values: tensor<64xi16, #distributed>) {
    // expected-error @+1 {{supports only f16, bf16, f32, f64, i32, and i64 values}}
    %0 = amdg.buffer_atomic_rmw add, relaxed, gpu, %values, %ptr[%offsets]
        : tensor<64xi16, #distributed>
    tt.return
  }
}

// -----

#nondistributed = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @extract_slice_requires_distributed_encoding(
      %arg0: tensor<16x16xbf16, #nondistributed>) {
    // expected-error @+1 {{requires a distributed source layout}}
    %0 = amdg.extract_slice %arg0 [0, 0]
        : tensor<16x16xbf16, #nondistributed>
          to tensor<16x8xbf16, #nondistributed>
    tt.return
  }
}

// -----

#distributed = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#nondistributed = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @extract_slice_requires_distributed_result_encoding(
      %arg0: tensor<16x16xbf16, #distributed>) {
    // expected-error @+1 {{requires a distributed result layout}}
    %0 = amdg.extract_slice %arg0 [0, 0]
        : tensor<16x16xbf16, #distributed>
          to tensor<16x8xbf16, #nondistributed>
    tt.return
  }
}

// -----

#nondistributed = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @buffer_load_requires_distributed_encoding(
      %ptr: !tt.ptr<bf16>, %offsets: tensor<64xi32, #nondistributed>) {
    // expected-error @+1 {{requires a distributed tensor encoding}}
    %0 = amdg.buffer_load %ptr[%offsets]
        : tensor<64xbf16, #nondistributed>
    tt.return
  }
}

// -----

#scheduled_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#scheduled_lhs = #ttg.dot_op<{opIdx = 0, parent = #scheduled_mma, kWidth = 8}>
#scheduled_rhs = #ttg.dot_op<{opIdx = 1, parent = #scheduled_mma, kWidth = 8}>
#half_register_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = true}>
#half_register_lhs = #ttg.dot_op<{opIdx = 0, parent = #half_register_mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // The dependency's encoding is checked for a native instruction shape before
  // its per-lane fragment is sized. That ordering makes the "positive integral
  // number of 32-bit registers" diagnostic below unreachable for BF16/F16:
  // a native version-4 shape always yields 8 elements per lane (128 bits), and
  // the odd counts that would trip it need K in {4, 12, 20, ...}, which the
  // shape check rejects first. Pin the diagnostic that actually fires.
  tt.func @mfma_commit_rejects_non_native_dependency_shape(
      %a: tensor<16x32xbf16, #scheduled_lhs>,
      %b: tensor<32x16xbf16, #scheduled_rhs>,
      %dependency: tensor<16x4xbf16, #half_register_lhs>) {
    %acc = arith.constant dense<0.000000e+00> :
        tensor<16x16xf32, #scheduled_mma>
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #scheduled_lhs>,
          tensor<32x16xbf16, #scheduled_rhs>,
          tensor<16x16xf32, #scheduled_mma>
          -> tensor<16x16xf32, #scheduled_mma>
    // expected-error @+1 {{input 1 MFMA encoding version 4 supports only its native 32x32x16 and 16x16x32 shapes}}
    %committed, %preserved = amdg.mfma_commit %result, %dependency
        : tensor<16x16xf32, #scheduled_mma>,
          tensor<16x4xbf16, #half_register_lhs>
    tt.return
  }
}

// -----

#scheduled_mismatch_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#scheduled_mismatch_lhs = #ttg.dot_op<{opIdx = 0, parent = #scheduled_mismatch_mma, kWidth = 4}>
#scheduled_mismatch_rhs = #ttg.dot_op<{opIdx = 1, parent = #scheduled_mismatch_mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scheduled_mfma_rejects_mismatched_kwidth(
      %a: tensor<16x32xbf16, #scheduled_mismatch_lhs>,
      %b: tensor<32x16xbf16, #scheduled_mismatch_rhs>) {
    %acc = arith.constant dense<0.000000e+00> :
        tensor<16x16xf32, #scheduled_mismatch_mma>
    // expected-error @+1 {{operand dot layouts must use the same kWidth}}
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #scheduled_mismatch_lhs>,
          tensor<32x16xbf16, #scheduled_mismatch_rhs>,
          tensor<16x16xf32, #scheduled_mismatch_mma>
          -> tensor<16x16xf32, #scheduled_mismatch_mma>
    tt.return
  }
}

// -----

#scheduled_unsupported_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#scheduled_unsupported_lhs = #ttg.dot_op<{opIdx = 0, parent = #scheduled_unsupported_mma, kWidth = 16}>
#scheduled_unsupported_rhs = #ttg.dot_op<{opIdx = 1, parent = #scheduled_unsupported_mma, kWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scheduled_mfma_rejects_unsupported_kwidth(
      %a: tensor<16x64xbf16, #scheduled_unsupported_lhs>,
      %b: tensor<64x16xbf16, #scheduled_unsupported_rhs>) {
    %acc = arith.constant dense<0.000000e+00> :
        tensor<16x16xf32, #scheduled_unsupported_mma>
    // expected-error @+1 {{operand A must use the matching opIdx=0, kWidth=4/8 dot layout}}
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x64xbf16, #scheduled_unsupported_lhs>,
          tensor<64x16xbf16, #scheduled_unsupported_rhs>,
          tensor<16x16xf32, #scheduled_unsupported_mma>
          -> tensor<16x16xf32, #scheduled_unsupported_mma>
    tt.return
  }
}

// -----

#scheduled_partial_k_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#scheduled_partial_k_lhs = #ttg.dot_op<{opIdx = 0, parent = #scheduled_partial_k_mma, kWidth = 4}>
#scheduled_partial_k_rhs = #ttg.dot_op<{opIdx = 1, parent = #scheduled_partial_k_mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scheduled_mfma_rejects_partial_native_k_fragment(
      %a: tensor<16x16xbf16, #scheduled_partial_k_lhs>,
      %b: tensor<16x16xbf16, #scheduled_partial_k_rhs>) {
    %acc = arith.constant dense<0.000000e+00> :
        tensor<16x16xf32, #scheduled_partial_k_mma>
    // expected-error @+1 {{operand K ownership must contain complete native MFMA fragments}}
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x16xbf16, #scheduled_partial_k_lhs>,
          tensor<16x16xbf16, #scheduled_partial_k_rhs>,
          tensor<16x16xf32, #scheduled_partial_k_mma>
          -> tensor<16x16xf32, #scheduled_partial_k_mma>
    tt.return
  }
}

// -----

// Gather with an index layout that distributes values across lanes (invalid).
// parent blocked: threadsPerWarp = [32, 1] → lanes map to dim 0.
// slice dim 1 → 1D tensor where each lane holds a different value.
#blocked_lane_dist = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice_lane_dist = #ttg.slice<{dim = 1, parent = #blocked_lane_dist}>
#shared_gather = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_gather = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_gather_invalid_lane_distribution(
    %memDesc: !ttg.memdesc<32x128xf16, #shared_gather, #smem_gather, mutable>,
    %tensorDesc: !tt.tensordesc<32x128xf16, #shared_gather>,
    %row_indices: tensor<32xi32, #slice_lane_dist>,
    %pred: i32
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{index layout distributes values across lanes}}
    %token = amdg.async_tdm_gather %tensorDesc[%row_indices] to %memDesc : tensor<32xi32, #slice_lane_dist>, !ttg.memdesc<32x128xf16, #shared_gather, #smem_gather, mutable> -> !tt.tensordesc<32x128xf16, #shared_gather>
    tt.return
  }
}

// -----

#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], warp = [], block = [], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #linear1}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_gather_invalid_missing_index_block_basis(
    %memDesc: !ttg.memdesc<16x64xf16, #shared1, #smem, mutable>,
    %tensorDesc: !tt.tensordesc<16x64xf16>,
    %row_indices: tensor<16xi32, #slice1>,
    %pred: i32
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{TDM gather index and destination layout must both have a block basis or neither have a block basis}}
    %token = amdg.async_tdm_gather %tensorDesc[%row_indices] to %memDesc : tensor<16xi32, #slice1>, !ttg.memdesc<16x64xf16, #shared1, #smem, mutable> -> !tt.tensordesc<16x64xf16>
    tt.return
  }
}

// -----

#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], warp = [], block = [[16, 0]], order = [1, 0]}>
#slice1 = #ttg.slice<{dim = 1, parent = #linear1}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_gather_invalid_block_basis_count(
    %memDesc: !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>,
    %tensorDesc: !tt.tensordesc<32x64xf16>,
    %row_indices: tensor<32xi32, #slice1>,
    %pred: i32
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{TDM gather index and shared encoding must have the same block basis for the row dimension}}
    %token = amdg.async_tdm_gather %tensorDesc[%row_indices] to %memDesc : tensor<32xi32, #slice1>, !ttg.memdesc<32x64xf16, #shared1, #smem, mutable> -> !tt.tensordesc<32x64xf16>
    tt.return
  }
}

// -----

// Gather padding interval (128) does not divide the innermost block dimension
// (64), so the chunk-relative lds_addr padding would not distribute.
#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
#shared = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [16, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_gather_invalid_padding_interval(
    %tensorDesc: !tt.tensordesc<16x64xf16, #shared>,
    %memDesc: !ttg.memdesc<16x64xf16, #shared, #smem, mutable>,
    %row_indices: tensor<16xi32, #slice>
  ) {
    // expected-error @+1 {{TDM gather padding interval must divide the innermost block dimension}}
    %token = amdg.async_tdm_gather %tensorDesc[%row_indices] to %memDesc : tensor<16xi32, #slice>, !ttg.memdesc<16x64xf16, #shared, #smem, mutable> -> !tt.tensordesc<16x64xf16, #shared>
    tt.return
  }
}

// -----

// Gather of a sub-byte element type: the lds_addr byte-delta scaling truncates
// to zero for <8-bit elements.
#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_gather_invalid_subbyte_element(
    %tensorDesc: !tt.tensordesc<16x64xi4, #shared>,
    %memDesc: !ttg.memdesc<16x64xi4, #shared, #smem, mutable>,
    %row_indices: tensor<16xi32, #slice>
  ) {
    // expected-error @+1 {{TDM gather requires element types of at least 8 bits}}
    %token = amdg.async_tdm_gather %tensorDesc[%row_indices] to %memDesc : tensor<16xi32, #slice>, !ttg.memdesc<16x64xi4, #shared, #smem, mutable> -> !tt.tensordesc<16x64xi4, #shared>
    tt.return
  }
}

// -----

// Scatter of a sub-byte element type: same lds_addr byte-delta truncation.
#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_scatter_invalid_subbyte_element(
    %tensorDesc: !tt.tensordesc<16x64xi4, #shared>,
    %memDesc: !ttg.memdesc<16x64xi4, #shared, #smem, mutable>,
    %row_indices: tensor<16xi32, #slice>
  ) {
    // expected-error @+1 {{TDM scatter requires element types of at least 8 bits}}
    %token = amdg.async_tdm_scatter %tensorDesc[%row_indices] from %memDesc : tensor<16xi32, #slice>, !ttg.memdesc<16x64xi4, #shared, #smem, mutable> -> !tt.tensordesc<16x64xi4, #shared>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0], CGALayout = [[0, 0], [0, 0]]}>
#slice1 = #ttg.slice<{dim = 1, parent = #blocked1}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_gather_invalid_row_block_basis(
    %memDesc: !ttg.memdesc<16x64xf16, #shared1, #smem, mutable>,
    %tensorDesc: !tt.tensordesc<16x64xf16>,
    %row_indices: tensor<16xi32, #slice1>,
    %pred: i32
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{TDM gather index and shared encoding must have the same block basis for the row dimension}}
    %token = amdg.async_tdm_gather %tensorDesc[%row_indices] to %memDesc : tensor<16xi32, #slice1>, !ttg.memdesc<16x64xf16, #shared1, #smem, mutable> -> !tt.tensordesc<16x64xf16>
    tt.return
  }
}

// -----

// Scatter with padded shared layout where padding interval != innermost block dimension.
#shared_scatter_32 = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [8, 64]}>
// Scatter with two padding intervals (only single interval is supported).
#shared_scatter_2_intervals = #ttg.padded_shared<[64:+4, 128:+4] {order = [1, 0], shape = [8, 64]}>
#smem_scatter = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scatter_interval_not_matching_innermost_block_dimension(
    %tensorDesc: !tt.tensordesc<8x64xf16, #shared_scatter_32>,
    %memDesc: !ttg.memdesc<8x64xf16, #shared_scatter_32, #smem_scatter, mutable>,
    %row_indices: tensor<8xi32>
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{TDM scatter padding is only supported when padding interval equals the innermost block dimension}}
    amdg.async_tdm_scatter %tensorDesc[%row_indices] from %memDesc : tensor<8xi32>, !ttg.memdesc<8x64xf16, #shared_scatter_32, #smem_scatter, mutable> -> !tt.tensordesc<8x64xf16, #shared_scatter_32>
    tt.return
  }

  tt.func public @scatter_two_padding_intervals(
    %tensorDesc: !tt.tensordesc<8x64xf16, #shared_scatter_32>,
    %memDesc: !ttg.memdesc<8x64xf16, #shared_scatter_2_intervals, #smem_scatter, mutable>,
    %row_indices: tensor<8xi32>
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{TDM scatter only supports single interval paddings}}
    amdg.async_tdm_scatter %tensorDesc[%row_indices] from %memDesc : tensor<8xi32>, !ttg.memdesc<8x64xf16, #shared_scatter_2_intervals, #smem_scatter, mutable> -> !tt.tensordesc<8x64xf16, #shared_scatter_32>
    tt.return
  }
}

// -----

// warp_used_hint validation tests
#shared_wb = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_wb = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // hint == 0 has no active warps; rejected.
  tt.func @warp_used_hint_zero(
    %tensorDesc: !tt.tensordesc<256x64xf16>,
    %memDesc: !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>,
    %pred: i32
  ) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{warp_used_hint must have at least one bit set}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc {warp_used_hint = 0 : i32} : !tt.tensordesc<256x64xf16> -> !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>
    tt.return
  }

  // 0x69 (warps 0,3,5,6) is rejected: K=4 is a power of two but the
  // active set spans 3 warpId bit positions, not log2(K) = 2 -- a
  // non axis-aligned pattern is not supported.
  tt.func @warp_used_hint_non_axis_aligned(
    %tensorDesc: !tt.tensordesc<256x64xf16>,
    %memDesc: !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>,
    %pred: i32
  ) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{is not axis-aligned}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc {warp_used_hint = 105 : i32} : !tt.tensordesc<256x64xf16> -> !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>
    tt.return
  }

  // popcount must be a power of two.  0x07 has K=3 -- rejected even
  // though warps 0..2 are otherwise contiguous.
  tt.func @warp_used_hint_non_pow2_k(
    %tensorDesc: !tt.tensordesc<256x64xf16>,
    %memDesc: !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>,
    %pred: i32
  ) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{popcount(warp_used_hint) = 3 must be a power of two}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc {warp_used_hint = 7 : i32} : !tt.tensordesc<256x64xf16> -> !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>
    tt.return
  }

  // hint sets all 16 low bits but num_warps = 8 so bits 8..15 don't
  // correspond to any warp.  Reported by the bits-beyond check.
  tt.func @warp_used_hint_exceeds_num_warps(
    %tensorDesc: !tt.tensordesc<256x64xf16>,
    %memDesc: !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>,
    %pred: i32
  ) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{warp_used_hint = 0xffff sets bits beyond num_warps = 8}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc {warp_used_hint = 65535 : i32} : !tt.tensordesc<256x64xf16> -> !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>
    tt.return
  }

  // Bits outside [0, num_warps) must be zero.  K=2 is otherwise valid,
  // but warp index 9 is not in [0, 8).
  tt.func @warp_used_hint_bits_beyond_num_warps(
    %tensorDesc: !tt.tensordesc<256x64xf16>,
    %memDesc: !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>,
    %pred: i32
  ) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{sets bits beyond num_warps = 8}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc {warp_used_hint = 513 : i32} : !tt.tensordesc<256x64xf16> -> !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>
    tt.return
  }
}

// -----

#fp4_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#fp4_dst = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#fp4_scale_bad = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_upcast_fp4_incompatible_scale_encoding(%src: tensor<16x32xi8, #fp4_src>, %scale: tensor<16x64xbf16, #fp4_scale_bad>) {
    // expected-error @+1 {{scale and output encodings are not compatible}}
    %0 = amdg.scaled_upcast_fp4 %src scale %scale {axis = 1 : i32} : tensor<16x32xi8, #fp4_src>, tensor<16x64xbf16, #fp4_scale_bad> -> tensor<16x64xbf16, #fp4_dst>
    tt.return
  }
}

// -----

// Partitioned encoding requires K to be a multiple of numLogicalPieces
// (= numPartitions*numGroups = 4) so the hinted copy fits in a single
// TDM instruction.  Here K=2 < numLogicalPieces=4 is rejected.
#shared_inner_mi = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned_mi = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #shared_inner_mi}>
#smem_mi = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @warp_used_hint_partitioned_insufficient(
    %tensorDesc: !tt.tensordesc<128x16xf16>,
    %memDesc: !ttg.memdesc<128x16xf16, #partitioned_mi, #smem_mi, mutable>,
    %pred: i32
  ) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{warp_used_hint with a partitioned shared encoding must select K active warps}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc {warp_used_hint = 3 : i32} : !tt.tensordesc<128x16xf16> -> !ttg.memdesc<128x16xf16, #partitioned_mi, #smem_mi, mutable>
    tt.return
  }
}

// -----

#shared_inner_dynamic = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned_dynamic = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #shared_inner_dynamic}>
#smem_dynamic = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @dynamic_subslice_partitioned(
    %src: !ttg.memdesc<8x16xf16, #partitioned_dynamic, #smem_dynamic, mutable>,
    %row: i32
  ) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{dynamic subslices do not support partitioned shared encodings}}
    %view = ttg.memdesc_dynamic_subslice %src[%row, %zero] : !ttg.memdesc<8x16xf16, #partitioned_dynamic, #smem_dynamic, mutable> -> !ttg.memdesc<1x16xf16, #partitioned_dynamic, #smem_dynamic, mutable, 8x16>
    tt.return
  }
}

// -----

#fp4_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#fp4_dst = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#fp4_dst_bad = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_upcast_fp4_incompatible_src_encoding(%src: tensor<16x32xi8, #fp4_src>, %scale: tensor<16x64xbf16, #fp4_dst_bad>) {
    // expected-error @+1 {{Src and Dst encodings are not compatible}}
    %0 = amdg.scaled_upcast_fp4 %src scale %scale {axis = 1 : i32} : tensor<16x32xi8, #fp4_src>, tensor<16x64xbf16, #fp4_dst_bad> -> tensor<16x64xbf16, #fp4_dst_bad>
    tt.return
  }
}

// -----

#fp4_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#fp4_dst = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_upcast_fp4_invalid_result_type(%src: tensor<16x32xi8, #fp4_src>, %scale: tensor<16x64xbf16, #fp4_dst>) {
    // expected-error @+1 {{must be ranked tensor of 16-bit float or bfloat16 type values}}
    %0 = amdg.scaled_upcast_fp4 %src scale %scale {axis = 1 : i32} : tensor<16x32xi8, #fp4_src>, tensor<16x64xbf16, #fp4_dst> -> tensor<16x64xf32, #fp4_dst>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_upcast_fp8_invalid_result_type(%src: tensor<16x64xf8E4M3FN, #blocked>, %scale: tensor<16x64xbf16, #blocked>) {
    // expected-error @+1 {{must be ranked tensor of 16-bit float or bfloat16 type values}}
    %0 = amdg.scaled_upcast_fp8 %src scale %scale : tensor<16x64xf8E4M3FN, #blocked>, tensor<16x64xbf16, #blocked> -> tensor<16x64xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @update_tensor_descriptor_wrong_offset_count(
    %desc: !tt.tensordesc<64x64xf16, #shared>, %dx: i32
  ) -> !tt.tensordesc<64x64xf16, #shared> {
    // expected-error @+1 {{expected 2 add_offsets to match descriptor rank, got 1}}
    %result = amdg.update_tensor_descriptor %desc add_offsets = [%dx] : !tt.tensordesc<64x64xf16, #shared>
    tt.return %result : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @update_tensor_descriptor_wrong_bounds_count(
    %desc: !tt.tensordesc<64x64xf16, #shared>, %m: i32, %n: i32, %k: i32
  ) -> !tt.tensordesc<64x64xf16, #shared> {
    // expected-error @+1 {{expected 2 set_bounds to match descriptor rank, got 3}}
    %result = amdg.update_tensor_descriptor %desc set_bounds = [%m, %n, %k] : !tt.tensordesc<64x64xf16, #shared>
    tt.return %result : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @update_tensor_descriptor_no_kwargs(
    %desc: !tt.tensordesc<64x64xf16, #shared>
  ) -> !tt.tensordesc<64x64xf16, #shared> {
    // expected-error @+1 {{must provide at least one of add_offsets, set_bounds, or pred}}
    %result = amdg.update_tensor_descriptor %desc : !tt.tensordesc<64x64xf16, #shared>
    tt.return %result : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

// clamp_bounds requires add_offsets (it derives bounds from the advance).
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @clamp_bounds_requires_add_offsets(
    %desc: !tt.tensordesc<64x64xf16, #shared>, %p: i32
  ) -> !tt.tensordesc<64x64xf16, #shared> {
    // expected-error @+1 {{clamp_bounds requires add_offsets}}
    %result = amdg.update_tensor_descriptor %desc pred = %p {clamp_bounds} : !tt.tensordesc<64x64xf16, #shared>
    tt.return %result : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

// clamp_bounds and set_bounds are mutually exclusive (both write tensor_dim).
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @clamp_bounds_excludes_set_bounds(
    %desc: !tt.tensordesc<64x64xf16, #shared>, %i: i32, %j: i32, %m: i32, %k: i32
  ) -> !tt.tensordesc<64x64xf16, #shared> {
    // expected-error @+1 {{clamp_bounds and set_bounds are mutually exclusive}}
    %result = amdg.update_tensor_descriptor %desc add_offsets = [%i, %j] set_bounds = [%m, %k] {clamp_bounds} : !tt.tensordesc<64x64xf16, #shared>
    tt.return %result : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

// scatter: two padded layouts whose padding amount differs.
#scatter_desc = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [8, 64]}>
#scatter_alloc = #ttg.padded_shared<[64:+4] {order = [1, 0], shape = [8, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scatter_inconsistent_descriptor_layout(
    %tensorDesc: !tt.tensordesc<8x64xf16, #scatter_desc>,
    %memDesc: !ttg.memdesc<8x64xf16, #scatter_alloc, #smem, mutable>,
    %row_indices: tensor<8xi32>
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{is inconsistent with the shared memory allocation layout}}
    amdg.async_tdm_scatter %tensorDesc[%row_indices] from %memDesc : tensor<8xi32>, !ttg.memdesc<8x64xf16, #scatter_alloc, #smem, mutable> -> !tt.tensordesc<8x64xf16, #scatter_desc>
    tt.return
  }
}

// -----

// gather: descriptor and allocation use different encoding kinds (padded vs
// swizzled).
#gather_desc = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [8, 64]}>
#gather_alloc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gather_inconsistent_descriptor_layout(
    %tensorDesc: !tt.tensordesc<8x64xf16, #gather_desc>,
    %memDesc: !ttg.memdesc<8x64xf16, #gather_alloc, #smem, mutable>,
    %row_indices: tensor<8xi32>,
    %pred: i32
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{is inconsistent with the shared memory allocation layout}}
    %token = amdg.async_tdm_gather %tensorDesc[%row_indices] to %memDesc : tensor<8xi32>, !ttg.memdesc<8x64xf16, #gather_alloc, #smem, mutable> -> !tt.tensordesc<8x64xf16, #gather_desc>
    tt.return
  }
}

// -----

// load (global-to-local copy): two swizzled layouts whose order differs.
#load_desc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#load_alloc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_load_inconsistent_descriptor_layout(
    %tensorDesc: !tt.tensordesc<64x64xf16, #load_desc>,
    %memDesc: !ttg.memdesc<64x64xf16, #load_alloc, #smem, mutable>
  ) {
    // expected-error @+1 {{is inconsistent with the shared memory allocation layout}}
    %token = amdg.async_tdm_copy_global_to_local %tensorDesc into %memDesc : !tt.tensordesc<64x64xf16, #load_desc> -> !ttg.memdesc<64x64xf16, #load_alloc, #smem, mutable>
    tt.return
  }
}

// -----

#v4_on_gfx942_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#v4_on_gfx942_lhs = #ttg.dot_op<{opIdx = 0, parent = #v4_on_gfx942_mma, kWidth = 8}>
#v4_on_gfx942_rhs = #ttg.dot_op<{opIdx = 1, parent = #v4_on_gfx942_mma, kWidth = 8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @mfma_commit_rejects_cdna4_encoding_on_cdna3_target(
      %result: tensor<16x16xf32, #v4_on_gfx942_mma>,
      %dependency: tensor<32x16xbf16, #v4_on_gfx942_rhs>) {
    // expected-error @+1 {{input 0 uses MFMA version 4, but the target requires version 3}}
    %committed, %preserved = amdg.mfma_commit %result, %dependency
        : tensor<16x16xf32, #v4_on_gfx942_mma>,
          tensor<32x16xbf16, #v4_on_gfx942_rhs>
    tt.return
  }
}

// -----

// No ttg.target: the encoding version alone must still gate the native shape.
#untargeted_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#untargeted_lhs = #ttg.dot_op<{opIdx = 0, parent = #untargeted_mma, kWidth = 4}>
#untargeted_rhs = #ttg.dot_op<{opIdx = 1, parent = #untargeted_mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scheduled_mfma_rejects_cdna4_shape_on_v3_encoding(
      %a: tensor<16x32xbf16, #untargeted_lhs>,
      %b: tensor<32x16xbf16, #untargeted_rhs>) {
    %acc = arith.constant dense<0.000000e+00> :
        tensor<16x16xf32, #untargeted_mma>
    // expected-error @+1 {{MFMA encoding version 3 supports only its native 32x32x8 and 16x16x16 shapes}}
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x32xbf16, #untargeted_lhs>,
          tensor<32x16xbf16, #untargeted_rhs>,
          tensor<16x16xf32, #untargeted_mma>
          -> tensor<16x16xf32, #untargeted_mma>
    tt.return
  }
}

// -----

#v3_on_gfx950_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#v3_on_gfx950_lhs = #ttg.dot_op<{opIdx = 0, parent = #v3_on_gfx950_mma, kWidth = 4}>
#v3_on_gfx950_rhs = #ttg.dot_op<{opIdx = 1, parent = #v3_on_gfx950_mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scheduled_mfma_rejects_cdna3_encoding_on_cdna4_target(
      %a: tensor<16x16xbf16, #v3_on_gfx950_lhs>,
      %b: tensor<16x16xbf16, #v3_on_gfx950_rhs>) {
    %acc = arith.constant dense<0.000000e+00> :
        tensor<16x16xf32, #v3_on_gfx950_mma>
    // expected-error @+1 {{uses MFMA version 3, but the target requires version 4}}
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "transient"
        register_class "auto" initialize true
        : tensor<16x16xbf16, #v3_on_gfx950_lhs>,
          tensor<16x16xbf16, #v3_on_gfx950_rhs>,
          tensor<16x16xf32, #v3_on_gfx950_mma>
          -> tensor<16x16xf32, #v3_on_gfx950_mma>
    tt.return
  }
}

// -----

// The commit boundary derives its hazard wait from the shape triple, so a
// shape the encoding version does not have must be rejected rather than
// silently treated as the version's other native shape.
#nonnative_result_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @mfma_commit_rejects_cdna4_shape_on_v3_result(
      %result: tensor<32x32xf32, #nonnative_result_mma>) {
    // expected-error @+1 {{input 0 MFMA encoding version 3 supports only its native 32x32x8 and 16x16x16 shapes}}
    %committed = amdg.mfma_commit %result
        : tensor<32x32xf32, #nonnative_result_mma>
    tt.return
  }
}

// -----

#native_result_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#nonnative_dep_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#nonnative_dep_rhs = #ttg.dot_op<{opIdx = 1, parent = #nonnative_dep_mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @mfma_commit_rejects_cdna4_shape_on_v3_dependency(
      %result: tensor<16x16xf32, #native_result_mma>,
      %dependency: tensor<32x16xbf16, #nonnative_dep_rhs>) {
    // expected-error @+1 {{input 1 MFMA encoding version 3 supports only its native 32x32x8 and 16x16x16 shapes}}
    %committed, %preserved = amdg.mfma_commit %result, %dependency
        : tensor<16x16xf32, #native_result_mma>,
          tensor<32x16xbf16, #nonnative_dep_rhs>
    tt.return
  }
}

// -----

// An AGPR-resident accumulator is read back with a compiler-generated
// v_accvgpr_read that CDNA3's inline-assembly hazard padding cannot order
// against the MFMA drain, so the explicit class is rejected on that target.
#agpr_cdna3_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
#agpr_cdna3_lhs = #ttg.dot_op<{opIdx = 0, parent = #agpr_cdna3_mma, kWidth = 4}>
#agpr_cdna3_rhs = #ttg.dot_op<{opIdx = 1, parent = #agpr_cdna3_mma, kWidth = 4}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scheduled_mfma_rejects_explicit_agpr_on_cdna3(
      %a: tensor<16x16xbf16, #agpr_cdna3_lhs>,
      %b: tensor<16x16xbf16, #agpr_cdna3_rhs>) {
    %acc = arith.constant dense<0.000000e+00> :
        tensor<16x16xf32, #agpr_cdna3_mma>
    // expected-error @+1 {{accumulator_register_class "agpr" is not yet supported on CDNA3}}
    %result = amdg.scheduled_mfma %a, %b, %acc
        resident "none" accumulator "persistent"
        register_class "agpr" initialize true
        : tensor<16x16xbf16, #agpr_cdna3_lhs>,
          tensor<16x16xbf16, #agpr_cdna3_rhs>,
          tensor<16x16xf32, #agpr_cdna3_mma>
          -> tensor<16x16xf32, #agpr_cdna3_mma>
    tt.return
  }
}
