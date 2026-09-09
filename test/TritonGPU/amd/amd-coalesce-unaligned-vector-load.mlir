// RUN: triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "hip:gfx950",
                   "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: #[[$WIDE:.*]] = #ttg.blocked<{sizePerThread = [4]
  // CHECK-LABEL: @exact_gfx950
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f16>, #[[$WIDE]]>
  tt.func @exact_gfx950(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                        %unaligned: i32) -> tensor<1024xf16, #blocked> {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offset = tt.splat %unaligned : i32 -> tensor<1024xi32, #blocked>
    %indices = arith.addi %range, %offset : tensor<1024xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %bases, %indices : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %value = tt.load %ptrs : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return %value : tensor<1024xf16, #blocked>
  }

  // gfx950 supports byte-aligned vector accesses, so no residual element-size
  // alignment proof is required.
  // CHECK-LABEL: @byte_aligned_gfx950
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f16>, #[[$WIDE]]>
  tt.func @byte_aligned_gfx950(
      %base: !tt.ptr<f16> {tt.divisibility = 1 : i32})
      -> tensor<1024xf16, #blocked> {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %bases, %range : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %value = tt.load %ptrs : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return %value : tensor<1024xf16, #blocked>
  }

  // A masked access remains alignment-conservative. The compiler cannot
  // widen it merely because gfx950 supports unaligned vector instructions:
  // the mask still has to describe a legal vector transaction.
  // CHECK-LABEL: @masked_gfx950
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f16>, #blocked>
  tt.func @masked_gfx950(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                          %unaligned: i32,
                          %mask: tensor<1024xi1, #blocked>)
      -> tensor<1024xf16, #blocked> {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offset = tt.splat %unaligned : i32 -> tensor<1024xi32, #blocked>
    %indices = arith.addi %range, %offset : tensor<1024xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %bases, %indices : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %other = arith.constant dense<0.0> : tensor<1024xf16, #blocked>
    %value = tt.load %ptrs, %mask, %other : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return %value : tensor<1024xf16, #blocked>
  }

  // A mask whose predicate is constant across four adjacent elements permits
  // the same four-element vector width as the contiguous pointer.
  // CHECK-LABEL: @aligned_mask_gfx950
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f16>, #[[$WIDE]]>
  tt.func @aligned_mask_gfx950(
      %base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %unaligned: i32,
      %mask: tensor<1024xi1, #blocked> {tt.constancy = 4 : i32})
      -> tensor<1024xf16, #blocked> {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offset = tt.splat %unaligned : i32 -> tensor<1024xi32, #blocked>
    %indices = arith.addi %range, %offset : tensor<1024xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %bases, %indices : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %other = arith.constant dense<0.0> : tensor<1024xf16, #blocked>
    %value = tt.load %ptrs, %mask, %other : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return %value : tensor<1024xf16, #blocked>
  }

  // Volatile accesses also retain the generic alignment rule.
  // CHECK-LABEL: @volatile_gfx950
  // CHECK: tt.load {{.*}}isVolatile = true{{.*}} : tensor<1024x!tt.ptr<f16>, #blocked>
  tt.func @volatile_gfx950(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                            %unaligned: i32) -> tensor<1024xf16, #blocked> {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offset = tt.splat %unaligned : i32 -> tensor<1024xi32, #blocked>
    %indices = arith.addi %range, %offset : tensor<1024xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %bases, %indices : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %value = tt.load %ptrs {isVolatile = true} : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return %value : tensor<1024xf16, #blocked>
  }

  // A related relaxed load must not widen masked or volatile loads that share
  // its pointer slice.
  // CHECK-LABEL: @mixed_loads_gfx950
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f16>, #[[$WIDE]]>
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f16>, #blocked>
  // CHECK: tt.load {{.*}}isVolatile = true{{.*}} : tensor<1024x!tt.ptr<f16>, #blocked>
  tt.func @mixed_loads_gfx950(
      %base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %unaligned: i32,
      %mask: tensor<1024xi1, #blocked>) -> tensor<1024xf16, #blocked> {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offset = tt.splat %unaligned : i32 -> tensor<1024xi32, #blocked>
    %indices = arith.addi %range, %offset : tensor<1024xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %bases, %indices : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %other = arith.constant dense<0.0> : tensor<1024xf16, #blocked>
    %wide = tt.load %ptrs : tensor<1024x!tt.ptr<f16>, #blocked>
    %masked = tt.load %ptrs, %mask, %other : tensor<1024x!tt.ptr<f16>, #blocked>
    %volatile = tt.load %ptrs {isVolatile = true} : tensor<1024x!tt.ptr<f16>, #blocked>
    %sum0 = arith.addf %wide, %masked : tensor<1024xf16, #blocked>
    %sum1 = arith.addf %sum0, %volatile : tensor<1024xf16, #blocked>
    tt.return %sum1 : tensor<1024xf16, #blocked>
  }

  // A masked load must not count itself as a related relaxed load. It may
  // share the wider layout selected by an aligned store while retaining its
  // own scalar transaction width during lowering.
  // CHECK-LABEL: @masked_load_with_aligned_store
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f16>, #[[$WIDE]]>
  // CHECK: tt.store {{.*}} : tensor<1024x!tt.ptr<f16>, #[[$WIDE]]>
  tt.func @masked_load_with_aligned_store(
      %input: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %output: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %unaligned: i32,
      %mask: tensor<1024xi1, #blocked>) {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offset = tt.splat %unaligned : i32 -> tensor<1024xi32, #blocked>
    %inputIndices = arith.addi %range, %offset : tensor<1024xi32, #blocked>
    %inputBases = tt.splat %input : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %inputPtrs = tt.addptr %inputBases, %inputIndices : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %other = arith.constant dense<0.0> : tensor<1024xf16, #blocked>
    %loaded = tt.load %inputPtrs, %mask, %other : tensor<1024x!tt.ptr<f16>, #blocked>
    %outputBases = tt.splat %output : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %outputPtrs = tt.addptr %outputBases, %range : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %outputPtrs, %loaded : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // Stores are not covered by the load-only hardware capability.
  // CHECK-LABEL: @store_gfx950
  // CHECK: tt.store {{.*}} : tensor<1024x!tt.ptr<f16>, #blocked>
  tt.func @store_gfx950(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                         %unaligned: i32,
                         %value: tensor<1024xf16, #blocked>) {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offset = tt.splat %unaligned : i32 -> tensor<1024xi32, #blocked>
    %indices = arith.addi %range, %offset : tensor<1024xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %bases, %indices : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %ptrs, %value : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "hip:gfx942",
                   "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @gfx942_is_conservative
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f16>, #blocked>
  tt.func @gfx942_is_conservative(
      %base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %unaligned: i32) -> tensor<1024xf16, #blocked> {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offset = tt.splat %unaligned : i32 -> tensor<1024xi32, #blocked>
    %indices = arith.addi %range, %offset : tensor<1024xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %bases, %indices : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %value = tt.load %ptrs : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return %value : tensor<1024xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "hip:gfx950:sramecc+:xnack-",
                   "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: #[[$FEATURE_WIDE:.*]] = #ttg.blocked<{sizePerThread = [4]
  // CHECK-LABEL: @gfx950_with_features
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f16>, #[[$FEATURE_WIDE]]>
  tt.func @gfx950_with_features(
      %base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %unaligned: i32) -> tensor<1024xf16, #blocked> {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %offset = tt.splat %unaligned : i32 -> tensor<1024xi32, #blocked>
    %indices = arith.addi %range, %offset : tensor<1024xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %bases, %indices : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %value = tt.load %ptrs : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return %value : tensor<1024xf16, #blocked>
  }
}

// -----

#ptr = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "hip:gfx950",
                   "ttg.threads-per-warp" = 64 : i32} {
  // Axis analysis makes dimension 0 the pointer's vectorized axis even though
  // dimension 1 is the layout's fastest axis. The mask is constant only along
  // dimension 1, so it cannot permit a vector load along dimension 0.
  // CHECK: #[[$MASK_SCALAR:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4]
  // CHECK-LABEL: @mask_uses_pointer_axis
  // CHECK: tt.load {{.*}} : tensor<16x64x!tt.ptr<f16>, #[[$MASK_SCALAR]]>
  tt.func @mask_uses_pointer_axis(
      %ptr: tensor<16x64x!tt.ptr<f16>, #ptr>
          {tt.contiguity = dense<[4, 1]> : tensor<2xi32>,
           tt.divisibility = dense<[2, 2]> : tensor<2xi32>},
      %maskValue: tensor<16x64xi1, #ptr>
          {tt.constancy = dense<[1, 4]> : tensor<2xi32>})
      -> tensor<16x64xf16, #ptr> {
    %other = arith.constant dense<0.0> : tensor<16x64xf16, #ptr>
    %value = tt.load %ptr, %maskValue, %other : tensor<16x64x!tt.ptr<f16>, #ptr>
    tt.return %value : tensor<16x64xf16, #ptr>
  }
}
