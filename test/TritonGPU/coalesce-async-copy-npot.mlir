// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-coalesce-async-copy -mlir-print-local-scope | FileCheck %s

// A mixed-radix async copy predicates off duplicate threads, so the 80-wide
// tile, loaded with 4-element vectors, copies with 8. Every program copies
// the same tile, so the copy keeps the L1 caching (.ca) of its 8-byte form.
// CHECK-LABEL: @copy_64x80
// CHECK: ttg.async_copy_global_to_local {{.*}} {cachePolicy = #tt.cache_policy<cache_modifier = ca, {{.*}}>} : tensor<64x80x!tt.ptr<f16>, #ttg.linear<{register = {{\[\[}}0, 1], [0, 2], [0, 4], [0, 16]], {{.*}} warp = {{\[\[}}16, 0], [32, 0], [0, 0]]
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 16]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [32, 0]], block = [], inputRadices = {register = [2, 2, 5], lane = [2, 2, 2, 2, 2], warp = [2, 2, 2], block = []}, outDimSizes = [64, 80]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func @copy_64x80(%pb: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi32, #linear>
  %stride = arith.constant dense<80> : tensor<64x1xi32, #linear>
  %row = arith.muli %k1, %stride : tensor<64x1xi32, #linear>
  %rowb = tt.broadcast %row : tensor<64x1xi32, #linear> -> tensor<64x80xi32, #linear>
  %n = tt.make_range {end = 80 : i32, start = 0 : i32} : tensor<80xi32, #ttg.slice<{dim = 0, parent = #linear}>>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<80xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x80xi32, #linear>
  %nb = tt.broadcast %n1 : tensor<1x80xi32, #linear> -> tensor<64x80xi32, #linear>
  %off = arith.addi %rowb, %nb : tensor<64x80xi32, #linear>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x80x!tt.ptr<f16>, #linear>
  %ptr = tt.addptr %sb, %off : tensor<64x80x!tt.ptr<f16>, #linear>, tensor<64x80xi32, #linear>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<64x80xf16, #shared, #smem, mutable>
  %t = ttg.async_copy_global_to_local %ptr, %buf : tensor<64x80x!tt.ptr<f16>, #linear> -> <64x80xf16, #shared, #smem, mutable>
  tt.return
}
}

// -----

// Each program copies its own rows, so the 16-byte copy goes L2-only (.cg).
// CHECK-LABEL: @copy_64x80_pid
// CHECK: ttg.async_copy_global_to_local %{{[0-9]+}}, %{{[0-9]+}} : tensor<64x80x!tt.ptr<f16>, #ttg.linear<{register = {{\[\[}}0, 1], [0, 2], [0, 4], [0, 16]]
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 16]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [32, 0]], block = [], inputRadices = {register = [2, 2, 5], lane = [2, 2, 2, 2, 2], warp = [2, 2, 2], block = []}, outDimSizes = [64, 80]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func @copy_64x80_pid(%pb: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  %pid = tt.get_program_id x : i32
  %c5120 = arith.constant 5120 : i32
  %tile = arith.muli %pid, %c5120 : i32
  %pt = tt.addptr %pb, %tile : !tt.ptr<f16>, i32
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi32, #linear>
  %stride = arith.constant dense<80> : tensor<64x1xi32, #linear>
  %row = arith.muli %k1, %stride : tensor<64x1xi32, #linear>
  %rowb = tt.broadcast %row : tensor<64x1xi32, #linear> -> tensor<64x80xi32, #linear>
  %n = tt.make_range {end = 80 : i32, start = 0 : i32} : tensor<80xi32, #ttg.slice<{dim = 0, parent = #linear}>>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<80xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x80xi32, #linear>
  %nb = tt.broadcast %n1 : tensor<1x80xi32, #linear> -> tensor<64x80xi32, #linear>
  %off = arith.addi %rowb, %nb : tensor<64x80xi32, #linear>
  %sb = tt.splat %pt : !tt.ptr<f16> -> tensor<64x80x!tt.ptr<f16>, #linear>
  %ptr = tt.addptr %sb, %off : tensor<64x80x!tt.ptr<f16>, #linear>, tensor<64x80xi32, #linear>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<64x80xf16, #shared, #smem, mutable>
  %t = ttg.async_copy_global_to_local %ptr, %buf : tensor<64x80x!tt.ptr<f16>, #linear> -> <64x80xf16, #shared, #smem, mutable>
  tt.return
}
}

// -----

// Control: a power-of-two copy keeps its layout, though 8 would fit.
// CHECK-LABEL: @copy_64x64
// CHECK: ttg.async_copy_global_to_local {{.*}} : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 4]
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func @copy_64x64(%pb: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  %k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
  %k1 = tt.expand_dims %k {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
  %stride = arith.constant dense<64> : tensor<64x1xi32, #blocked>
  %row = arith.muli %k1, %stride : tensor<64x1xi32, #blocked>
  %rowb = tt.broadcast %row : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
  %n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
  %n1 = tt.expand_dims %n {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
  %nb = tt.broadcast %n1 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
  %off = arith.addi %rowb, %nb : tensor<64x64xi32, #blocked>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
  %ptr = tt.addptr %sb, %off : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %t = ttg.async_copy_global_to_local %ptr, %buf : tensor<64x64x!tt.ptr<f16>, #blocked> -> <64x64xf16, #shared, #smem, mutable>
  tt.return
}
}
