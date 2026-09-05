// RUN: triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "hip:gfx950",
                   "ttg.threads-per-warp" = 64 : i32} {
  // CHECK: #[[WIDE:.*]] = #ttg.blocked<{sizePerThread = [4]
  // CHECK-LABEL: @exact_gfx950
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f16>, #[[WIDE]]>
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
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f16>, #[[WIDE]]>
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
