// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="smem-budget=232448" | FileCheck %s --check-prefix=OVER
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="smem-budget=300000" | FileCheck %s --check-prefix=UNDER

// Reduced from Tutorial 09's B200 AutoWS persistent-matmul configuration. Its
// live multibuffers and auxiliary workspace consume 229376 bytes of SMEM. The
// layout conversion needs 16384 bytes of scratch, so 229376 + 16384 exceeds
// the real 232448-byte device budget.

// OVER-LABEL: @tutorial09_autows_budget
// OVER-COUNT-6: ttg.local_alloc
// OVER: tt.return

#linear_tmem = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0], [0, 64]], block = []}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared_out = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_aux = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tutorial09_autows_budget(
      %arg0: tensor<64x128xf32, #linear_tmem>,
      %local: !ttg.memdesc<64x128xbf16, #shared_out, #smem>,
      %out0: !ttg.memdesc<64x128xf32, #shared_out, #smem, mutable>,
      %out1: !ttg.memdesc<64x128xf32, #shared_out, #smem, mutable>) ->
      (!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>,
       !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>,
       !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>,
       !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>,
       !ttg.memdesc<3x128x32xf16, #shared64, #smem, mutable>,
       !ttg.memdesc<8192xi8, #shared_aux, #smem, mutable>) {
    %a0 = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %a1 = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %epilogue0 = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %epilogue1 = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16, #shared64, #smem, mutable>
    %aux = ttg.local_alloc : () -> !ttg.memdesc<8192xi8, #shared_aux, #smem, mutable>
    %converted = ttg.convert_layout %arg0 : tensor<64x128xf32, #linear_tmem> -> tensor<64x128xf32, #blocked>
    %load = ttg.local_load %local : !ttg.memdesc<64x128xbf16, #shared_out, #smem> -> tensor<64x128xbf16, #blocked>
    %ext = arith.extf %load : tensor<64x128xbf16, #blocked> to tensor<64x128xf32, #blocked>
    %sub = arith.subf %converted, %ext : tensor<64x128xf32, #blocked>
    ttg.local_store %sub, %out0 : tensor<64x128xf32, #blocked> -> !ttg.memdesc<64x128xf32, #shared_out, #smem, mutable>
    %neg = arith.negf %ext : tensor<64x128xf32, #blocked>
    ttg.local_store %neg, %out1 : tensor<64x128xf32, #blocked> -> !ttg.memdesc<64x128xf32, #shared_out, #smem, mutable>
    tt.return %a0, %a1, %b, %epilogue0, %epilogue1, %aux : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x32xf16, #shared64, #smem, mutable>, !ttg.memdesc<8192xi8, #shared_aux, #smem, mutable>
  }
}

// -----

// A non-binding budget must retain the profitability gate and avoid cloning
// the layer-normalization reduction.

// UNDER-LABEL: @nonbinding_budget_keeps_cost_gate
// UNDER: "tt.reduce"(
// UNDER-NOT: "tt.reduce"(
// UNDER: tt.return

#src = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @nonbinding_budget_keeps_cost_gate(%arg0: tensor<16x512xf32, #src>) -> (tensor<16x512xf32, #src>, tensor<16x512xf32, #dst>) {
    %sum = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) : (tensor<16x512xf32, #src>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #src}>>
    %expanded = tt.expand_dims %sum {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #src}>> -> tensor<16x1xf32, #src>
    %broadcast = tt.broadcast %expanded : tensor<16x1xf32, #src> -> tensor<16x512xf32, #src>
    %centered = arith.subf %arg0, %broadcast : tensor<16x512xf32, #src>
    %converted = ttg.convert_layout %centered : tensor<16x512xf32, #src> -> tensor<16x512xf32, #dst>
    tt.return %centered, %converted : tensor<16x512xf32, #src>, tensor<16x512xf32, #dst>
  }
}
