// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --verify-diagnostics
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --allocate-shared-memory-nv='compute-capability=100' | FileCheck %s --check-prefix=ALLOC

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_n64 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_n32 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 32, colStride = 1>
#tmem_space = #ttng.tensor_memory
#smem = #ttg.shared_memory
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#nvmma_i8 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#sl = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0]]}, alignment = 16>

!tmem_npot = !ttg.memdesc<128x96xf32, #tmem, #tmem_space, mutable, 128x128>
!tmem_multibuffer = !ttg.memdesc<3x128x96xf32, #tmem, #tmem_space, mutable, 3x128x128>
!tmem_scales_cover = !ttg.memdesc<128x6xi8, #ttng.tensor_memory_scales_encoding<>, #tmem_space, mutable, 128x8>
// Power-of-two control: a compact TMEM allocation verifies.
!tmem_pot_compact = !ttg.memdesc<128x128xf32, #tmem_n32, #tmem_space, mutable>
!shared_cover = !ttg.memdesc<8x48xi8, #shared, #smem, mutable, 8x64>
// A larger allocation holds the cover.
!shared_overallocated = !ttg.memdesc<8x48xi8, #shared, #smem, mutable, 8x128>
// A leading-NPOT allocation reserves its power-of-two cover.
!shared_leading_npot = !ttg.memdesc<3x48xi8, #shared, #smem, mutable, 3x64>
!shared_leading_cover = !ttg.memdesc<3x48xi8, #shared, #smem, mutable, 4x64>
!nvmma_native_cover = !ttg.memdesc<128x48xi8, #nvmma_i8, #smem, mutable, 128x128>
// A shared_linear cover is the layout's own extent.
!shared_linear_cover = !ttg.memdesc<8x48xi8, #sl, #smem, mutable, 8x64>
!nvmma_runtime_view = !ttg.memdesc<128x48xi8, #nvmma_i8, #smem, mutable, 3x128x128>

module {
  tt.func @alloc_physical_covers() {
    %0 = ttng.tmem_alloc : () -> !tmem_npot
    %1 = ttng.tmem_alloc : () -> !tmem_multibuffer
    %7 = ttg.local_alloc : () -> !shared_cover
    %8 = ttg.local_alloc : () -> !shared_leading_npot
    %11 = ttg.local_alloc : () -> !nvmma_native_cover
    %12 = ttng.tmem_alloc : () -> !tmem_pot_compact
    %13 = ttg.local_alloc : () -> !shared_leading_cover
    %14 = ttg.local_alloc : () -> !shared_overallocated
    %15 = ttg.local_alloc : () -> !shared_linear_cover
    %16 = ttng.tmem_alloc : () -> !tmem_scales_cover
    tt.return
  }

  // Subviews that pass the power-of-two shape test are not NPOT physical
  // covers: with TRITON_ALLOW_NPOT set they keep the upstream view contract.
  tt.func @pot_subviews_keep_upstream_contract() {
    %tmem = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #tmem_space, mutable>
    %tmem_view = ttng.tmem_subslice %tmem {offset = 64 : i32}
        : !ttg.memdesc<128x128xf32, #tmem, #tmem_space, mutable>
        -> !ttg.memdesc<128x64xf32, #tmem_n64, #tmem_space, mutable, 128x128>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #nvmma, #smem, mutable>
    %smem_view = ttg.memdesc_subslice %smem[64, 0]
        : !ttg.memdesc<128x64xf16, #nvmma, #smem, mutable>
        -> !ttg.memdesc<64x64xf16, #nvmma, #smem, mutable, 128x64>
    tt.return
  }

  // ALLOC-LABEL: tt.func @runtime_enlarged_cover
  // ALLOC-SAME: !ttg.memdesc<128x48xi8, #{{.*}}, #smem, mutable, 3x128x128>
  tt.func @runtime_enlarged_cover(%view: !nvmma_runtime_view) {
    tt.return
  }
}
