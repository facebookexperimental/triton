// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --triton-nvidia-gpu-cluster-barrier-mbar-allocator | FileCheck %s

// A Product convert stays within one CTA, so a partition needs no cluster
// barrier for it, whether it moves registers or crosses warps.
#src = #ttg.linear<{register = [[128]], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [384]}>
#dst = #ttg.linear<{register = [[128]], lane = [[1], [2], [4], [8], [16]], warp = [[64], [32]], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [384]}>
#rsrc = #ttg.linear<{register = [[1], [256]], lane = [[2], [4], [8], [16], [32]], warp = [[64], [128]], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [768]}>
#rdst = #ttg.linear<{register = [[256], [1]], lane = [[2], [4], [8], [16], [32]], warp = [[64], [128]], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [768]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: ttg.ws_cluster_barrier_count
  // CHECK-LABEL: @product_converts_in_partition
  tt.func @product_converts_in_partition(%p: !tt.ptr<i32>) {
    ttg.warp_specialize()
    default {
      %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32, #src>
      // CHECK: ttg.convert_layout
      // CHECK-NOT: ttg.mbar_offset
      %c = ttg.convert_layout %r : tensor<384xi32, #src> -> tensor<384xi32, #dst>
      %s = tt.splat %p : !tt.ptr<i32> -> tensor<384x!tt.ptr<i32>, #dst>
      tt.store %s, %c : tensor<384x!tt.ptr<i32>, #dst>
      %rr = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32, #rsrc>
      // CHECK: ttg.convert_layout
      // CHECK-NOT: ttg.mbar_offset
      %rc = ttg.convert_layout %rr : tensor<768xi32, #rsrc> -> tensor<768xi32, #rdst>
      %rs = tt.splat %p : !tt.ptr<i32> -> tensor<768x!tt.ptr<i32>, #rdst>
      tt.store %rs, %rc : tensor<768x!tt.ptr<i32>, #rdst>
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    // CHECK: tt.return
    tt.return
  }
}
