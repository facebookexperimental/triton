// RUN: triton-opt --split-input-file %s --verify-diagnostics

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
tt.func public @local_alloc_i1() {
    // expected-error @+1 {{element type bit width must be a multiple of 8}}
    %0 = ttg.local_alloc : () -> !ttg.memdesc<8xi1, #shared, #smem, mutable>
    tt.return
}

// -----

// expected-error @+1 {{LinearEncodingAttr requires a permutation matrix layout after removing broadcast bases}}
#linear_bad_perm = #ttg.linear<{register = [[1], [3]], lane = [], warp = [], block = []}>
module {
  tt.func public @invalid_linear_layout(%arg0: tensor<4xi32, #linear_bad_perm>) {
    tt.return
  }
}

// -----

#src = #ttg.linear<{register = [[16, 0], [1, 0], [2, 0], [4, 0], [8, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#dst = #ttg.linear<{register = [[32, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @fp4_reordered_result(%arg0: tensor<32x128xi8, #src>) {
    // expected-error @+1 {{failed to infer encoding}}
    %0 = ttg.fp4_to_fp %arg0 {axis = 0 : i32} : tensor<32x128xi8, #src> -> tensor<64x128xbf16, #dst>
    tt.return
  }
}

// -----

// expected-error @+1 {{After removing broadcast bases the CGA encoding must be a permutation matrix}}
#blocked_bad_cga = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1], CGALayout = [[1, 0], [1, 0]]}>
module {
  tt.func public @invalid_cga_layout(%arg0: tensor<1x1xf32, #blocked_bad_cga>) {
    tt.return
  }
}

// -----

// expected-error @+1 {{LinearEncodingAttr requires a permutation matrix layout after removing broadcast bases}}
#linear_bad_after_flatten = #ttg.linear<{register = [[1, 1], [1, 0]], lane = [], warp = [], block = []}>
module {
  tt.func public @invalid_linear_layout_after_flatten(%arg0: tensor<2x2xi32, #linear_bad_after_flatten>) {
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
module {
  // expected-error @+1 {{tensor descriptors must not wrap tensor types; use !tt.tensordesc<shape x element-type[, layout]> instead}}
  tt.func public @nested_tensordesc(%arg0: !tt.tensordesc<tensor<8x16xf32, #shared>>) {
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @miss_encoding(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{,}}
    %a = ttg.memdesc_subslice %arg0 [0, 0] : !ttg.memdesc<8x16xf32> -> !ttg.memdesc<8x16xf16>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @miss_memory_space(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{,}}
    %a = ttg.memdesc_subslice %arg0 [0, 0] : !ttg.memdesc<8x16xf32, #shared> -> !ttg.memdesc<8x16xf16>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @subview_element_ty(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{element type}}
    %a = ttg.memdesc_subslice %arg0 [0, 0] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<8x16xf16, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @too_many_offsets(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{offsets}}
    %a = ttg.memdesc_subslice %arg0 [0, 0, 0] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<8x16xf32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @too_few_offsets(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    // expected-error @+1 {{offsets}}
    %a = ttg.memdesc_subslice %arg0 [0] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<8x16xf32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @result_rank_too_large(%arg0: !ttg.memdesc<3x8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result rank}}
    %a = ttg.memdesc_index %arg0[%zero] : !ttg.memdesc<3x8x16xf32, #shared, #smem> -> !ttg.memdesc<3x8x16xf32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @memdesc_index_result_alloc_shape_mismatch(%arg0: !ttg.memdesc<3x8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{alloc shape must match shape for both result and src}}
    %a = ttg.memdesc_index %arg0[%zero] : !ttg.memdesc<3x8x16xf32, #shared, #smem> -> !ttg.memdesc<8x16xf32, #shared, #smem, 3x8x16>
    tt.return
}
// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0]}>
#smem = #ttg.shared_memory
tt.func public @result_1d_to_1d(%arg0: !ttg.memdesc<8xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result rank}}
    %a = ttg.memdesc_index %arg0[%zero] : !ttg.memdesc<8xf32, #shared, #smem> -> !ttg.memdesc<2xf32, #shared, #smem>
    tt.return
}


// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @subview_along_swizzling_pattern(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    // expected-error @+1 {{swizzling pattern}}
    %a = ttg.memdesc_subslice %arg0 [0, 0] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<8x4xf32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @subview_along_swizzling(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>, %index: i32) {
    // expected-error @+1 {{tile}}
    %a = ttg.memdesc_subslice %arg0 [2, 0] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<4x16xf32, #shared, #smem>
    tt.return
}

// -----

#linear = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
tt.func public @local_atomic_scatter_rmw_immutable_dst(%values: tensor<32xi32, #linear>,
                                                       %indices: tensor<32xi32, #linear>,
                                                       %dst: !ttg.memdesc<32xi32, #shared, #smem>) attributes {"ttg.num-warps" = 1 : i32} {
    // expected-error @+1 {{Cannot store into immutable memory}}
    %0 = ttg.local_atomic_scatter_rmw add, %dst[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<32xi32, #shared, #smem>, tensor<32xi32, #linear>, tensor<32xi32, #linear>) -> tensor<32xi32, #linear>
    tt.return
}

// -----

#linear = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
tt.func public @local_atomic_scatter_rmw_bad_value_kind(%values: tensor<32x!tt.ptr<i32>, #linear>,
                                                        %indices: tensor<32xi32, #linear>,
                                                        %dst: !ttg.memdesc<32x!tt.ptr<i32>, #shared, #smem, mutable>) attributes {"ttg.num-warps" = 1 : i32} {
    // expected-error @+1 {{values must have integer or floating element type}}
    %0 = ttg.local_atomic_scatter_rmw add, %dst[%indices], %values {axis = 0 : i32} : (!ttg.memdesc<32x!tt.ptr<i32>, #shared, #smem, mutable>, tensor<32x!tt.ptr<i32>, #linear>, tensor<32xi32, #linear>) -> tensor<32x!tt.ptr<i32>, #linear>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#shared1d = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0]}>
#smem = #ttg.shared_memory
tt.func public @result_dim_too_large(%arg0: !ttg.memdesc<8x16xf32, #shared1d, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result shape}}
    %a = ttg.memdesc_index %arg0[%zero] : !ttg.memdesc<8x16xf32, #shared1d, #smem> -> !ttg.memdesc<32xf32, #shared1d, #smem>
    tt.return
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf32, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{element types of operands A and B must have same bit width}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf32, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{mismatching encoding between A and B operands}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf16> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @memdesc_reinterpret_changed_storage_size(%arg0: !ttg.memdesc<8x16xf16, #shared, #smem>) {
    // expected-error @+1 {{result logical storage size must not exceed source logical storage size}}
    %a = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<8x16xf16, #shared, #smem> -> !ttg.memdesc<8x16xf32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
tt.func public @memdesc_reinterpret_changed_memory_space(%arg0: !ttg.memdesc<128x128xf16, #shared, #smem>) {
    // expected-error @+1 {{source and destination memory space must match}}
    %a = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @memdesc_reinterpret_changed_mutability(%arg0: !ttg.memdesc<8x16xf16, #shared, #smem>) {
    // expected-error @+1 {{source and result must have the same mutability}}
    %a = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<8x16xf16, #shared, #smem> -> !ttg.memdesc<8x16xf16, #shared, #smem, mutable>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @memdesc_reinterpret_subview(%arg0: !ttg.memdesc<8x16xf16, #shared, #smem, 16x16>) {
    // expected-error @+1 {{source and result must not be subviews}}
    %a = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<8x16xf16, #shared, #smem, 16x16> -> !ttg.memdesc<8x16xf16, #shared, #smem>
    tt.return
}

// -----

#shared_linear_src = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0]]}, alignment = 16>
#shared_linear_dst = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]]}, alignment = 16>
#smem = #ttg.shared_memory
tt.func public @memdesc_reinterpret_oversized_shared_linear(%arg0: !ttg.memdesc<16x16xf16, #shared_linear_src, #smem>) {
    // expected-error @+1 {{result logical storage size must not exceed source logical storage size}}
    %a = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<16x16xf16, #shared_linear_src, #smem> -> !ttg.memdesc<32x16xf16, #shared_linear_dst, #smem>
    tt.return
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32>) {
    // expected-error@+1 {{miss encoding of C operand}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32>
    tt.return
  }
}

// -----
#shared = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [16, 128]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 17376 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @memdesc_reinterpret_between_padded_nonpadded() {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable>
    // expected-error @+1 {{reinterpret between padded and non-padded}}
    %1 = ttg.memdesc_reinterpret %0 : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xbf16, #shared1, #smem, mutable>
    tt.return
  }
}

// -----
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.padded_shared<[128:+8,256:+4] {order = [1, 0], shape = [16, 128]}>
#shared2 = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [16, 128]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 17376 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @memdesc_reinterpret_different_padding() {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xbf16, #mma>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable>
    // expected-error @+1 {{cannot reinterpret with different padding pattern}}
    %1 = ttg.memdesc_reinterpret %0 : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<2x16x128xf16, #shared2, #smem, mutable>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{mismatching kWidth between A and B operands}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=4}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=4}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @dot_i8_invalid_operand_type(%A: tensor<16x32xi16, #dot_operand_a>, %B: tensor<32x8xi8, #dot_operand_b>, %C: tensor<16x8xi32, #mma0>) {
    // expected-error@+1 {{operand #0 must be ranked tensor of 8-bit signless integer values}}
    %D = "tti.dot_i8"(%A, %B, %C) {aSigned = true, bSigned = true} : (tensor<16x32xi16, #dot_operand_a>, tensor<32x8xi8, #dot_operand_b>, tensor<16x8xi32, #mma0>) -> tensor<16x8xi32, #mma0>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @dot_i8_non_mma_layout(%A: tensor<16x32xi8, #dot_operand_a>, %B: tensor<32x8xi8, #dot_operand_b>, %C: tensor<16x8xi32, #blocked>) {
    // expected-error@+1 {{requires NVIDIA MMAv2 operand and result layouts}}
    %D = tti.dot_i8 %A, %B, %C, aSigned = true, bSigned = true : tensor<16x32xi8, #dot_operand_a> * tensor<32x8xi8, #dot_operand_b> -> tensor<16x8xi32, #blocked>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1], instrShape = [16, 8]}>
#mma1 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1], instrShape = [32, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=4}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma1, kWidth=4}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @dot_i8_mismatched_layout(%A: tensor<16x32xi8, #dot_operand_a>, %B: tensor<32x8xi8, #dot_operand_b>, %C: tensor<16x8xi32, #mma0>) {
    // expected-error@+1 {{requires matching NVIDIA MMAv2 layouts}}
    %D = tti.dot_i8 %A, %B, %C, aSigned = true, bSigned = true : tensor<16x32xi8, #dot_operand_a> * tensor<32x8xi8, #dot_operand_b> -> tensor<16x8xi32, #mma0>
    tt.return
  }
}

// -----

tt.func @warp_specialize_no_holder() {
  // expected-error @below {{'ttg.warp_specialize' op expected to find only a `ttg.warp_specialize.partitions` op inside its second region}}
  "ttg.warp_specialize"() ({
    "ttg.warp_yield"() : () -> ()
  }, {
    "ttg.warp_yield"() : () -> ()
  }) {partitionNumWarps = array<i32>} : () -> ()
  tt.return
}

// -----

tt.func @warp_specialize_mismatch_partition_count() {
  // expected-error @below {{'ttg.warp_specialize' op has 0 partitions but `partitionNumWarps` has 1 elements}}
  "ttg.warp_specialize"() ({
    "ttg.warp_yield"() : () -> ()
  }, {
    "ttg.warp_specialize.partitions"() : () -> ()
  }) {partitionNumWarps = array<i32: 1>} : () -> ()
}

// -----

tt.func @not_power_of_2() {
  // expected-error @below {{'ttg.warp_specialize' op partition #0 number of warps (3) must be a power of 2}}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(3) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

// -----

tt.func @bad_argument_count() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  // expected-error @below {{'ttg.warp_specialize.partitions' op partition region #0 has 1 arguments but expected 0}}
  partition0(%arg0: i32) num_warps(4) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

// -----

tt.func @bad_default_yields(%arg0: i32) {
  ttg.warp_specialize()
  default {
    // expected-error @below {{'ttg.warp_yield' op has 0 operands but parent op expected 1}}
    ttg.warp_yield
  } : () -> i32
  tt.return
}

// -----

tt.func @bad_default_yields(%arg0: i32, %arg1: i64) {
  ttg.warp_specialize()
  default {
    // expected-error @below {{'ttg.warp_yield' op operand #0 has type 'i64' but parent op expected 'i32'}}
    ttg.warp_yield %arg1 : i64
  } : () -> i32
  tt.return
}

// -----

#blocked_4_warps = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @function_scope() attributes {"ttg.num-warps" = 8 : i32} {
  // expected-error @below {{Layout has 4 warps per CTA, but the context requires 8 warps per CTA}}
  tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_4_warps>
  tt.return
}

}

// -----

#blocked_1_warps = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @function_no_scope() {
  // expected-error @below {{Layout has 1 warps per CTA, but the context requires 4 warps per CTA}}
  tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_1_warps>
  tt.return
}

}

// -----

#blocked_8_warps = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @function_no_scope() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    // expected-error @below {{Layout has 8 warps per CTA, but the context requires 2 warps per CTA}}
    tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_8_warps>
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

#blocked_2_warps = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @function_no_scope() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    ttg.warp_return
  }
  partition1() num_warps(1) {
    // expected-error @below {{Layout has 2 warps per CTA, but the context requires 1 warps per CTA}}
    tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_2_warps>
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

tt.func @illegal_ws_nest() {
  ttg.warp_specialize()
  default {
    // expected-error @below {{'ttg.warp_specialize' op cannot be nested inside another `ttg.warp_specialize` op}}
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    } : () -> ()
    ttg.warp_yield
  } : () -> ()
  tt.return
}

// -----

tt.func @invalid_start_ids() {
  // expected-error @below {{'ttg.warp_specialize' op has 1 warp group start IDs but expected 2}}
  ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    ttg.warp_return
  }
  partition1() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

// -----

tt.func @partition_no_terminator() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  // expected-error @below {{region with at least 1 blocks}}
  partition0() num_warps(2) {
  } : () -> ()
  tt.return
}

// -----

tt.func @partition_no_terminator() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    // expected-error @below {{block with no terminator}}
    %c1_i32 = arith.constant 1 : i32
  } : () -> ()
  tt.return
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @async_copy_invalid_mask_type(%input: tensor<64x64x!tt.ptr<f16>, #blocked>,
    %view: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
    %invalid_mask: tensor<64x64xi32, #blocked> // expected-note {{prior use here}}
  ) {
    // expected-error @+1 {{expects different type than prior uses}}
    %token = ttg.async_copy_global_to_local %input, %view mask %invalid_mask
      : tensor<64x64x!tt.ptr<f16>, #blocked> -> <64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_copy_invalid_other_type(%input: tensor<64x64x!tt.ptr<f16>, #blocked>,
    %view: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
    %mask: tensor<64x64xi1, #blocked>,
    %invalid_other: tensor<64x64xf32, #blocked> // expected-note {{prior use here}}
  ) {
  // expected-error @+1 {{expects different type than prior uses}}
  %token = ttg.async_copy_global_to_local %input, %view mask %mask other %invalid_other : tensor<64x64x!tt.ptr<f16>, #blocked> -> <64x64xf16, #shared, #smem, mutable>
  tt.return
}
}

// -----

// expected-error @below {{parent layout must have at least rank >= 2}}
#slice = #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>}>

// -----

// expected-error @below {{slice dim=2 must be less than the parent rank=2}}
#slice = #ttg.slice<{dim = 2, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>

// -----

// expected-error @below {{rank 0 memdesc is not allowed}}
!memdesc = !ttg.memdesc<i64, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>

// -----

#shared = #ttg.padded_shared<[4:+4] {offset=[[1, 0], [2, 0], [0, 1], [0, 2]], block=[]}>
// expected-error @below {{rank must be equal to or one less than the shape size. Got 2 and 4}}
!rank_too_high = !ttg.memdesc<4x4x4x4xf32, #shared, #ttg.shared_memory>

// -----

#shared = #ttg.padded_shared<[4:+4] {offset=[[1, 0], [2, 0], [0, 1], [0, 2]], block=[]}>
// expected-error @below {{rank must be equal to or one less than the shape size. Got 2 and 1}}
!rank_too_small = !ttg.memdesc<4xf32, #shared, #ttg.shared_memory>

// -----

#shared = #ttg.padded_shared<[4:+4] {offset=[[1, 0], [2, 0], [0, 1], [0, 2]], block=[]}>
// expected-error @below {{Mismatch in expected shape for dimension 0. Expected: 2, got: 4}}
!out_dim_too_small = !ttg.memdesc<2x2xf32, #shared, #ttg.shared_memory>

// -----

#shared = #ttg.padded_shared<[4:+4] {offset=[[1, 0], [2, 0], [0, 1], [0, 2]], block=[]}>
// expected-error @below {{Mismatch in expected shape for dimension 0. Expected: 8, got: 4}}
!out_dim_too_large = !ttg.memdesc<8x8xf32, #shared, #ttg.shared_memory>

// -----

// expected-error @below {{Mismatch of shape and order ranks in padded layout}}
#shared = #ttg.padded_shared<[4:+4] {shape=[1, 2, 4], order=[1, 0]}>

// -----

#shared = #ttg.padded_shared<[4:+4] {shape=[32, 32], order=[1, 0]}>
#smem = #ttg.shared_memory
tt.func public @padded_subview_unsupported_size(%arg0: !ttg.memdesc<2x32x32xf32, #shared, #smem>) {
    // expected-error @+1 {{SubSlice of low rank PaddedSharedEncoding from higher rank tensors is not supported yet}}
    %a = ttg.memdesc_subslice %arg0 [0, 16, 0] : !ttg.memdesc<2x32x32xf32, #shared, #smem> -> !ttg.memdesc<2x16x32xf32, #shared, #smem, 2x32x32>
    tt.return
}

// -----

// expected-error @below {{alignment must be specified outside of the linear layout braces}}
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [1, 0], [2, 0]], block = [], alignment = 16}>
!alignment_in_layout = !ttg.memdesc<4x4xf32, #shared, #ttg.shared_memory>

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_predicate_yield_count(
      %predicate: tensor<64xi1, #blocked>,
      %init: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // expected-error @+1 {{expected equal numbers of inits, results, and yields, but got 1, 1, and 0}}
    %result = ttg.warp_predicate %predicate (%init) {
      ttg.predicate_yield
    } : (tensor<64xi1, #blocked>, tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %result : tensor<64xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_predicate_block_argument(
      %predicate: tensor<64xi1, #blocked>,
      %init: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // expected-error @+1 {{region block must not have arguments}}
    %result = "ttg.warp_predicate"(%predicate, %init) ({
    ^bb0(%arg: tensor<64xf32, #blocked>):
      "ttg.predicate_yield"(%arg) : (tensor<64xf32, #blocked>) -> ()
    }) : (tensor<64xi1, #blocked>, tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %result : tensor<64xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_predicate_cta_barrier(
      %predicate: tensor<64xi1, #blocked>,
      %init: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // expected-error @+1 {{region may not contain CTA barriers}}
    %result = ttg.warp_predicate %predicate (%init) {
      ttg.barrier none
      ttg.predicate_yield %init : tensor<64xf32, #blocked>
    } : (tensor<64xi1, #blocked>, tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %result : tensor<64xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_predicate_nested_region(
      %condition: i1,
      %predicate: tensor<64xi1, #blocked>,
      %init: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // expected-error @+1 {{region may not contain nested dynamic control flow}}
    %result = ttg.warp_predicate %predicate (%init) {
      scf.if %condition {
        scf.yield
      }
      ttg.predicate_yield %init : tensor<64xf32, #blocked>
    } : (tensor<64xi1, #blocked>, tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %result : tensor<64xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_predicate_reduce_region(
      %predicate: tensor<256xi1, #blocked>,
      %init: tensor<256xf32, #blocked>) -> tensor<256xf32, #blocked> {
    // expected-error @+1 {{region reduction axis must be warp-local}}
    %result = ttg.warp_predicate %predicate (%init) {
      %sum = "tt.reduce"(%init) <{axis = 0 : i32}> ({
      ^bb0(%lhs: f32, %rhs: f32):
        %next = arith.addf %lhs, %rhs : f32
        tt.reduce.return %next : f32
      }) : (tensor<256xf32, #blocked>) -> f32
      ttg.predicate_yield %init : tensor<256xf32, #blocked>
    } : (tensor<256xi1, #blocked>, tensor<256xf32, #blocked>) -> tensor<256xf32, #blocked>
    tt.return %result : tensor<256xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#row = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_predicate_requires_uniform_wave_for_local_reduce(
      %predicate: i1,
      %init: tensor<4x64xf32, #blocked>) -> tensor<4x64xf32, #blocked> {
    // expected-error @+1 {{cross-lane operation tt.reduce requires a wave-uniform predicate}}
    %result = ttg.warp_predicate %predicate (%init) {
      %sum = "tt.reduce"(%init) <{axis = 1 : i32}> ({
      ^bb0(%lhs: f32, %rhs: f32):
        %next = arith.addf %lhs, %rhs : f32
        tt.reduce.return %next : f32
      }) : (tensor<4x64xf32, #blocked>) -> tensor<4xf32, #row>
      ttg.predicate_yield %init : tensor<4x64xf32, #blocked>
    } : (i1, tensor<4x64xf32, #blocked>) -> tensor<4x64xf32, #blocked>
    tt.return %result : tensor<4x64xf32, #blocked>
  }
}

// -----

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx950"} {
  tt.func @warp_predicate_requires_uniform_wave_for_dot(
      %predicate: i1,
      %lhs: tensor<32x32xf16, #dot0>,
      %rhs: tensor<32x32xf16, #dot1>,
      %acc: tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #mma> {
    // expected-error @+1 {{cross-lane operation tt.dot requires a wave-uniform predicate}}
    %result = ttg.warp_predicate %predicate (%acc) {
      %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<32x32xf32, #mma>
      ttg.predicate_yield %dot : tensor<32x32xf32, #mma>
    } : (i1, tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #mma>
    tt.return %result : tensor<32x32xf32, #mma>
  }
}

// -----

#row = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#column = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_predicate_requires_uniform_wave_for_layout_shuffle(
      %predicate: i1,
      %init: tensor<64x64xf32, #row>) -> tensor<64x64xf32, #row> {
    // expected-error @+1 {{cross-lane operation ttg.convert_layout requires a wave-uniform predicate}}
    %result = ttg.warp_predicate %predicate (%init) {
      %local = arith.addf %init, %init : tensor<64x64xf32, #row>
      %shuffled = ttg.convert_layout %local : tensor<64x64xf32, #row> -> tensor<64x64xf32, #column>
      ttg.predicate_yield %init : tensor<64x64xf32, #row>
    } : (i1, tensor<64x64xf32, #row>) -> tensor<64x64xf32, #row>
    tt.return %result : tensor<64x64xf32, #row>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_predicate_reduce_axis_out_of_bounds(
      %predicate: tensor<64xi1, #blocked>,
      %init: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    %result = ttg.warp_predicate %predicate (%init) {
      // expected-error @+1 {{axis out of bounds for operand rank 1}}
      %sum = "tt.reduce"(%init) <{axis = 1 : i32}> ({
      ^bb0(%lhs: f32, %rhs: f32):
        %next = arith.addf %lhs, %rhs : f32
        tt.reduce.return %next : f32
      }) : (tensor<64xf32, #blocked>) -> f32
      ttg.predicate_yield %init : tensor<64xf32, #blocked>
    } : (tensor<64xi1, #blocked>, tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %result : tensor<64xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_predicate_shape_mismatch(
      %predicate: tensor<64xi1, #blocked>,
      %init: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    // expected-error @+1 {{predicate shape must be a leading shape of every carried tensor}}
    %result = ttg.warp_predicate %predicate (%init) {
      ttg.predicate_yield %init : tensor<128xf32, #blocked>
    } : (tensor<64xi1, #blocked>, tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    tt.return %result : tensor<128xf32, #blocked>
  }
}

// -----

#predicate = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
#carried = #ttg.linear<{register = [[1]], lane = [[2], [4], [8], [16], [32], [0]], warp = [], block = []}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @warp_predicate_lane_ownership_mismatch(
      %predicate_value: tensor<64xi1, #predicate>,
      %init: tensor<64xf32, #carried>) -> tensor<64xf32, #carried> {
    // expected-error @+1 {{predicate and carried tensors must have matching lane ownership}}
    %result = ttg.warp_predicate %predicate_value (%init) {
      ttg.predicate_yield %init : tensor<64xf32, #carried>
    } : (tensor<64xi1, #predicate>, tensor<64xf32, #carried>) -> tensor<64xf32, #carried>
    tt.return %result : tensor<64xf32, #carried>
  }
}
