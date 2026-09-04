// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx942 analyze-small-tensor-ofst=true" | FileCheck %s --check-prefix=GFX942
// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx950 analyze-small-tensor-ofst=true" | FileCheck %s --check-prefix=GFX950

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32,
                   "ttg.threads-per-warp" = 64 : i32} {
  // The first address is base + one f16, so axis analysis can only guarantee
  // 2-byte alignment. gfx950 may still use the four contiguous elements owned
  // by each thread; older targets retain scalar contiguity.
  // GFX942-LABEL: @plain_unaligned_load
  // GFX942: amdg.buffer_load
  // GFX942-NOT: contiguity
  // GFX942: tt.return
  // GFX950-LABEL: @plain_unaligned_load
  // GFX950: amdg.buffer_load {{.*}} {contiguity = 4 : i32}
  tt.func @plain_unaligned_load(
      %base: !tt.ptr<f16> {tt.divisibility = 16 : i32,
                           tt.pointer_range = 32 : i32})
      -> tensor<256xf16, #blocked> {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %one = arith.constant dense<1> : tensor<256xi32, #blocked>
    %indices = arith.addi %range, %one : tensor<256xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %bases, %indices : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    %value = tt.load %ptrs : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return %value : tensor<256xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32,
                   "ttg.threads-per-warp" = 64 : i32} {
  // An aligned mask permits one predicate for all four loaded elements.
  // GFX942-LABEL: @aligned_mask_unaligned_load
  // GFX942: amdg.buffer_load
  // GFX942-NOT: contiguity
  // GFX942: tt.return
  // GFX950-LABEL: @aligned_mask_unaligned_load
  // GFX950: amdg.buffer_load {{.*}} {contiguity = 4 : i32}
  tt.func @aligned_mask_unaligned_load(
      %base: !tt.ptr<f16> {tt.divisibility = 16 : i32,
                           tt.pointer_range = 32 : i32},
      %mask: tensor<256xi1, #blocked> {tt.constancy = 4 : i32})
      -> tensor<256xf16, #blocked> {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %one = arith.constant dense<1> : tensor<256xi32, #blocked>
    %indices = arith.addi %range, %one : tensor<256xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %bases, %indices : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    %other = arith.constant dense<0.0> : tensor<256xf16, #blocked>
    %value = tt.load %ptrs, %mask, %other : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return %value : tensor<256xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32,
                   "ttg.threads-per-warp" = 64 : i32} {
  // A mask keeps the normal alignment and mask-granularity checks even on
  // gfx950, so this access remains scalar.
  // GFX942-LABEL: @masked_unaligned_load
  // GFX942: amdg.buffer_load
  // GFX942-NOT: contiguity
  // GFX942: tt.return
  // GFX950-LABEL: @masked_unaligned_load
  // GFX950: amdg.buffer_load
  // GFX950-NOT: contiguity
  // GFX950: tt.return
  tt.func @masked_unaligned_load(
      %base: !tt.ptr<f16> {tt.divisibility = 16 : i32,
                           tt.pointer_range = 32 : i32},
      %mask: tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked> {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %one = arith.constant dense<1> : tensor<256xi32, #blocked>
    %indices = arith.addi %range, %one : tensor<256xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %bases, %indices : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    %other = arith.constant dense<0.0> : tensor<256xf16, #blocked>
    %value = tt.load %ptrs, %mask, %other : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return %value : tensor<256xf16, #blocked>
  }
}
