// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -verify-diagnostics --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90

// A radix-3 register digit with a zero basis reaches no radix-2 lowering.
#gl3 = #ttg.generic_linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on tt.func}}
  tt.func @local_alloc(%arg0: tensor<128xf32, #gl3>) {
    %0 = ttg.local_alloc %arg0 : (tensor<128xf32, #gl3>) -> !ttg.memdesc<128xf32, #shared, #smem>
    tt.return
  }
}

// -----

// A slice takes its parent's digits.
#l2d = #ttg.linear<{register = [[0, 1], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128, 2]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on tt.func}}
  tt.func @slice(%arg0: tensor<128xf32, #ttg.slice<{dim = 1, parent = #l2d}>>) {
    tt.return
  }
}

// -----

// TLX wrappers reject non-radix-2 digits when parsed, before a memdesc
// verifier can query them.
#lin3 = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128]}>
// expected-error @+1 {{NPOT mixed-radix layout is not yet supported in a TLX layout wrapper}}
#user = #tlx.user_layout<#lin3>

// -----

#lin3 = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128]}>
// expected-error @+1 {{NPOT mixed-radix layout is not yet supported in a TLX layout wrapper}}
#no_verify = #tlx.no_verify_layout<#lin3>

// -----

// Memdesc encodings take the same check.
#lin3 = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128]}>
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on tt.func}}
  tt.func @memdesc(%arg0: !ttg.memdesc<128xf32, #lin3, #smem>) {
    %0 = ttg.local_load %arg0 : !ttg.memdesc<128xf32, #lin3, #smem> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

// Op results without op traits (arith) take the same check.
#lin3 = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @constant() {
    // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on arith.constant}}
    %0 = arith.constant dense<1.0> : tensor<128xf32, #lin3>
    %1 = arith.addf %0, %0 : tensor<128xf32, #lin3>
    tt.return
  }
}

// -----

// Not exact: a basis of 96 does not shrink to the pow2 ceiling of 5.
#p160 = #ttg.linear<{register = [[96]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [5], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [160]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_odd_coefficient_on_5() {
    // expected-error @+1 {{NPOT mixed-radix layout is not yet supported on tt.make_range}}
    %r = tt.make_range {end = 5 : i32, start = 0 : i32} : tensor<5xi32, #p160>
    tt.return
  }
}

// -----

// Control: a radix-2 register digit with explicit metadata lowers.
#lin2 = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = [], inputRadices = {register = [2], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [128]}>
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @convert_radix2(%arg0: tensor<128xf32, #lin2>) {
    %0 = ttg.convert_layout %arg0 : tensor<128xf32, #lin2> -> tensor<128xf32, #blocked>
    tt.return
  }
}
