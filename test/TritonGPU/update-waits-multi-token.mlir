// RUN: triton-opt %s -tritongpu-pipeline | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.padded_shared<[4:+4] {order = [1, 0], shape=[16, 256]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: multi_token_wait_uses_min_over_all_token_histories
  tt.func public @multi_token_wait_uses_min_over_all_token_histories(%arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked>, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1>) {
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2

    // The second token has no interleaved commit after its definition, so the
    // recomputed multi-token wait count is min(1, 0) = 0, not the stale input.
    // CHECK: ttg.async_wait %{{[^,]+}}, %{{[^[:space:]]+}} {num = 0 : i32}
    %4 = ttg.async_wait %1, %3 {num = 7 : i32}
    tt.return
  }
}
