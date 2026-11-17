// RUN: env TRITON_ALLOW_NPOT=1 triton-opt --split-input-file %s --verify-diagnostics

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[1, 0]]>
#tmem_space = #ttng.tensor_memory
module attributes {"ttg.num-ctas" = 2 : i32} {
  tt.func @multi_cta_pow2_alloc_view() {
    // expected-error @+1 {{NPOT memdesc pow2 alloc view with multiple CTAs is not yet supported}}
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<256x96xf32, #tmem, #tmem_space, mutable, 256x128>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#smem = #ttg.shared_memory
// A register encoding has no pow2 alloc view.
// expected-error @below {{shape must have power-of-2 and non-zero dimensions; got 8, 96}}
!not_a_pow2_alloc = !ttg.memdesc<8x96xf32, #blocked, #smem, mutable, 8x128>
module {
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module {
  // A view that passes the power-of-two shape test keeps the flag-off contract.
  tt.func @pot_view_is_not_a_pow2_alloc() {
    // expected-error @+1 {{result shape and its alloc shape must match}}
    %0 = ttg.local_alloc : () -> !ttg.memdesc<48x64xf16, #nvmma, #smem, mutable, 64x64>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module {
  tt.func @allocation_rank_mismatch() {
    // expected-error @+1 {{NPOT memdesc with different logical and physical allocation ranks is not yet supported}}
    %0 = ttg.local_alloc : () -> !ttg.memdesc<8x48xi8, #shared, #smem, mutable, 2x3x8x64>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory
module {
  tt.func @rank_three_pow2_alloc() {
    // expected-error @+1 {{NPOT memdesc pow2 alloc view with a layout rank other than 2 is not yet supported}}
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x8x48xi8, #shared, #smem, mutable, 2x8x64>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module {
  tt.func @indexed_pow2_alloc_suffix_is_valid(
      %arg0: !ttg.memdesc<2x8x48xi8, #shared, #smem, mutable, 2x8x64>) {
    %zero = arith.constant 0 : i32
    %view = ttg.memdesc_index %arg0[%zero]
        : !ttg.memdesc<2x8x48xi8, #shared, #smem, mutable, 2x8x64>
          -> !ttg.memdesc<8x48xi8, #shared, #smem, mutable, 8x64>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module {
  tt.func @indexed_pow2_alloc_keeps_its_allocation(
      %arg0: !ttg.memdesc<2x8x48xi8, #shared, #smem, mutable, 2x8x64>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{alloc shape must match shape for the result}}
    %view = ttg.memdesc_index %arg0[%zero]
        : !ttg.memdesc<2x8x48xi8, #shared, #smem, mutable, 2x8x64>
          -> !ttg.memdesc<8x48xi8, #shared, #smem, mutable, 3x8x64>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
// A view of fewer buffers than its allocation.
// expected-error @below {{NPOT memdesc allocation that does not match the encoding's pow2 alloc shape is not yet supported}}
!buffer_prefix_mismatch = !ttg.memdesc<2x8x48xi8, #shared, #smem, mutable, 3x8x64>
module {
}

// -----

#sw = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#part = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 1, partitionDim = 1, partitionLayout = #sw}>
#smem = #ttg.shared_memory
// Upstream getAllocationShapePerCTA rounds a swizzled_shared tile up to its
// 4x64 pow2 alloc shape but not a partitioned one, so this 3x64 allocation is
// too small.
// expected-error @below {{NPOT memdesc allocation that does not match the encoding's pow2 alloc shape is not yet supported}}
!leading_npot_unpadded = !ttg.memdesc<3x48xi8, #part, #smem, mutable, 3x64>
module {
}

// -----

#sw = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#part = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 1, partitionDim = 1, partitionLayout = #sw}>
#smem = #ttg.shared_memory
// The same holds for a compact allocation.
// expected-error @below {{NPOT memdesc allocation that does not match the encoding's pow2 alloc shape is not yet supported}}
!compact_unpadded = !ttg.memdesc<64x96xi8, #part, #smem, mutable>
module {
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_space = #ttng.tensor_memory
// Only a shared allocation may be compact.
// expected-error @below {{alloc shape must have power-of-2 and non-zero dimensions; got 128, 96}}
!compact_tmem = !ttg.memdesc<128x96xf32, #tmem, #tmem_space, mutable>
module {
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#smem = #ttg.shared_memory
// The 4x128 pow2 alloc shape is smaller than the 8x128 NVMMA core tile.
// expected-error @below {{NPOT memdesc allocation that does not match the encoding's pow2 alloc shape is not yet supported}}
!nvmma_pow2_alloc_below_core_tile = !ttg.memdesc<3x96xi8, #nvmma, #smem, mutable, 3x128>
module {
}

// -----

// A CTA split of an NPOT extent is also checked on a subslice's allocation.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @swizzled_subslice_of_extent6_split4(%arg0: !ttg.memdesc<6x64xf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{NPOT memdesc extent split across CTAs is not yet supported}}
    %0 = ttg.memdesc_subslice %arg0 [0, 0] : !ttg.memdesc<6x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<2x64xf16, #shared, #smem, mutable, 6x64>
    tt.return
  }
}
