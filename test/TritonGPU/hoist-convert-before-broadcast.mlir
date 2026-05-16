// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -cse | FileCheck %s

// Test: hoistConvertOnTopOfExtOrBroadcast hoists convert before broadcast
// to reduce SMEM scratch (convert operates on smaller pre-broadcast tensor).
// e.g., convert on [1,256] = 1KB SMEM vs convert on [128,256] = 128KB SMEM.
// CHECK-LABEL: @test_hoist_convert_before_broadcast
// CHECK-SAME: (%[[ARG:.+]]: tensor<1x256xf32
//       CHECK:   %[[CVT:.+]] = ttg.convert_layout %[[ARG]]
//       CHECK:   %[[BC:.+]] = tt.broadcast %[[CVT]]
//       CHECK:   tt.return %[[BC]]
#blocked_src = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_dst = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:90"} {
tt.func @test_hoist_convert_before_broadcast(%arg0: tensor<1x256xf32, #blocked_src>) -> tensor<128x256xf32, #blocked_dst> {
    %bc = tt.broadcast %arg0 : tensor<1x256xf32, #blocked_src> -> tensor<128x256xf32, #blocked_src>
    %cvt = ttg.convert_layout %bc : tensor<128x256xf32, #blocked_src> -> tensor<128x256xf32, #blocked_dst>
    tt.return %cvt : tensor<128x256xf32, #blocked_dst>
}
}  // end module
