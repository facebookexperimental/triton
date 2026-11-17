// RUN: env TRITON_ALLOW_NPOT=1 MLIR_ENABLE_DIAGNOSTICS=remarks triton-opt --tlx-dump-layout -verify-diagnostics %s -o /dev/null
// RUN: env TRITON_ALLOW_NPOT=1 MLIR_ENABLE_DIAGNOSTICS=warnings,remarks triton-opt --tlx-dump-layout -verify-diagnostics %s -o /dev/null

// NPOT conversion diagnostics are not supported. Skip the conversion
// without emitting remarks or formatting its layouts.
#src = #ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16]], warp = [[32]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [192]}>
#dst = #ttg.linear<{register = [[64], [0]], lane = [[2], [4], [8], [16], [32]], warp = [[1]], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2], warp = [2], block = []}, outDimSizes = [192]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @npot_product_conversion(%p: !tt.ptr<i32>) {
    %r = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #src>
    %s = tt.splat %p : !tt.ptr<i32> -> tensor<192x!tt.ptr<i32>, #src>
    %a = tt.addptr %s, %r : tensor<192x!tt.ptr<i32>, #src>, tensor<192xi32, #src>
    %c = ttg.convert_layout %a : tensor<192x!tt.ptr<i32>, #src> -> tensor<192x!tt.ptr<i32>, #dst>
    %r2 = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #dst>
    tt.store %c, %r2 : tensor<192x!tt.ptr<i32>, #dst>
    tt.return
  }
}
