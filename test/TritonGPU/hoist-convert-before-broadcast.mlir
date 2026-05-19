// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -cse | FileCheck %s

// Test: hoistConvertOnTopOfExtOrBroadcast hoists convert before broadcast
// to reduce SMEM scratch (convert operates on smaller pre-broadcast tensor).
// e.g., convert on [1,256] = 1KB SMEM vs convert on [128,256] = 128KB SMEM.

// -----

// Blocked -> blocked: basic case.
// CHECK-LABEL: @test_blocked_to_blocked
// CHECK-SAME: (%[[ARG:.+]]: tensor<1x256xf32
//       CHECK:   %[[CVT:.+]] = ttg.convert_layout %[[ARG]] : tensor<1x256xf32, #{{.+}}> -> tensor<1x256xf32, #{{.+}}>
//       CHECK:   %[[BC:.+]] = tt.broadcast %[[CVT]] : tensor<1x256xf32, #{{.+}}> -> tensor<128x256xf32, #{{.+}}>
//       CHECK:   tt.return %[[BC]]
#blocked_src = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_dst = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:90"} {
tt.func @test_blocked_to_blocked(%arg0: tensor<1x256xf32, #blocked_src>) -> tensor<128x256xf32, #blocked_dst> {
    %bc = tt.broadcast %arg0 : tensor<1x256xf32, #blocked_src> -> tensor<128x256xf32, #blocked_src>
    %cvt = ttg.convert_layout %bc : tensor<128x256xf32, #blocked_src> -> tensor<128x256xf32, #blocked_dst>
    tt.return %cvt : tensor<128x256xf32, #blocked_dst>
}
}  // end module

// -----

// Blocked -> linear: the real-world addmm bias broadcast case from B200 persistent
// TMA matmul. The bias is loaded as [1,256] #blocked, broadcast to [128,256],
// then converted to #linear for the tmem accumulator add.
// CHECK-LABEL: @test_blocked_to_linear
// CHECK-SAME: (%[[ARG:.+]]: tensor<1x256xf32
//       CHECK:   %[[CVT:.+]] = ttg.convert_layout %[[ARG]] : tensor<1x256xf32, #{{.+}}> -> tensor<1x256xf32, #{{.+}}>
//       CHECK:   %[[BC:.+]] = tt.broadcast %[[CVT]] : tensor<1x256xf32, #{{.+}}> -> tensor<128x256xf32, #{{.+}}>
//       CHECK:   tt.return %[[BC]]
#blocked_bias = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear_acc = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:100"} {
tt.func @test_blocked_to_linear(%arg0: tensor<1x256xf32, #blocked_bias>) -> tensor<128x256xf32, #linear_acc> {
    %bc = tt.broadcast %arg0 : tensor<1x256xf32, #blocked_bias> -> tensor<128x256xf32, #blocked_bias>
    %cvt = ttg.convert_layout %bc : tensor<128x256xf32, #blocked_bias> -> tensor<128x256xf32, #linear_acc>
    tt.return %cvt : tensor<128x256xf32, #linear_acc>
}
}  // end module

// -----

// Ext + broadcast chain: extf(bf16->f32) then broadcast, matching the actual
// addmm epilogue pattern where bias is loaded as bf16, extended, then broadcast.
// CHECK-LABEL: @test_ext_then_broadcast_to_linear
// CHECK-SAME: (%[[ARG:.+]]: tensor<1x256xbf16
//       CHECK:   %[[EXT:.+]] = arith.extf %[[ARG]] : tensor<1x256xbf16, #{{.+}}> to tensor<1x256xf32, #{{.+}}>
//       CHECK:   %[[CVT:.+]] = ttg.convert_layout %[[EXT]] : tensor<1x256xf32, #{{.+}}> -> tensor<1x256xf32, #{{.+}}>
//       CHECK:   %[[BC:.+]] = tt.broadcast %[[CVT]] : tensor<1x256xf32, #{{.+}}> -> tensor<128x256xf32, #{{.+}}>
//       CHECK:   tt.return %[[BC]]
#blocked_bf16 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear_f32 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:100"} {
tt.func @test_ext_then_broadcast_to_linear(%arg0: tensor<1x256xbf16, #blocked_bf16>) -> tensor<128x256xf32, #linear_f32> {
    %ext = arith.extf %arg0 : tensor<1x256xbf16, #blocked_bf16> to tensor<1x256xf32, #blocked_bf16>
    %bc = tt.broadcast %ext : tensor<1x256xf32, #blocked_bf16> -> tensor<128x256xf32, #blocked_bf16>
    %cvt = ttg.convert_layout %bc : tensor<128x256xf32, #blocked_bf16> -> tensor<128x256xf32, #linear_f32>
    tt.return %cvt : tensor<128x256xf32, #linear_f32>
}
}  // end module

// -----

// Column broadcast: broadcast along dim1 instead of dim0.
// CHECK-LABEL: @test_col_broadcast_blocked
// CHECK-SAME: (%[[ARG:.+]]: tensor<128x1xf32
//       CHECK:   %[[CVT:.+]] = ttg.convert_layout %[[ARG]] : tensor<128x1xf32, #{{.+}}> -> tensor<128x1xf32, #{{.+}}>
//       CHECK:   %[[BC:.+]] = tt.broadcast %[[CVT]] : tensor<128x1xf32, #{{.+}}> -> tensor<128x256xf32, #{{.+}}>
//       CHECK:   tt.return %[[BC]]
#blocked_col_src = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_col_dst = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:90"} {
tt.func @test_col_broadcast_blocked(%arg0: tensor<128x1xf32, #blocked_col_src>) -> tensor<128x256xf32, #blocked_col_dst> {
    %bc = tt.broadcast %arg0 : tensor<128x1xf32, #blocked_col_src> -> tensor<128x256xf32, #blocked_col_src>
    %cvt = ttg.convert_layout %bc : tensor<128x256xf32, #blocked_col_src> -> tensor<128x256xf32, #blocked_col_dst>
    tt.return %cvt : tensor<128x256xf32, #blocked_col_dst>
}
}  // end module

// -----

// End-to-end addmm epilogue: local_load -> extf -> broadcast -> convert_layout.
// Mirrors the real B200 persistent TMA matmul bias path where bias is loaded
// from SMEM via local_load as bf16, extended to f32, broadcast, then converted
// to #linear for the tmem accumulator add.
// CHECK-LABEL: @test_local_load_ext_broadcast_to_linear
//       CHECK:   %[[LD:.+]] = ttg.local_load
//       CHECK:   %[[EXT:.+]] = arith.extf %[[LD]] : tensor<1x256xbf16, #{{.+}}> to tensor<1x256xf32, #{{.+}}>
//       CHECK:   %[[CVT:.+]] = ttg.convert_layout %[[EXT]] : tensor<1x256xf32, #{{.+}}> -> tensor<1x256xf32, #{{.+}}>
//       CHECK:   %[[BC:.+]] = tt.broadcast %[[CVT]] : tensor<1x256xf32, #{{.+}}> -> tensor<128x256xf32, #{{.+}}>
//       CHECK:   tt.return %[[BC]]
#blocked_ld = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear_tmem = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared_ld = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#smem_ld = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:100"} {
tt.func @test_local_load_ext_broadcast_to_linear(%alloc: !ttg.memdesc<1x256xbf16, #shared_ld, #smem_ld, mutable>) -> tensor<128x256xf32, #linear_tmem> {
    %ld = ttg.local_load %alloc : !ttg.memdesc<1x256xbf16, #shared_ld, #smem_ld, mutable> -> tensor<1x256xbf16, #blocked_ld>
    %ext = arith.extf %ld : tensor<1x256xbf16, #blocked_ld> to tensor<1x256xf32, #blocked_ld>
    %bc = tt.broadcast %ext : tensor<1x256xf32, #blocked_ld> -> tensor<128x256xf32, #blocked_ld>
    %cvt = ttg.convert_layout %bc : tensor<128x256xf32, #blocked_ld> -> tensor<128x256xf32, #linear_tmem>
    tt.return %cvt : tensor<128x256xf32, #linear_tmem>
}
}  // end module

// -----

// BLOCK_M=64, BLOCK_N=256, BLOCK_K=64, num_warps=4, num_stages=4. This config was
// verified to successfully hoist the convert before broadcast in end-to-end runs.
// CHECK-LABEL: @test_config3_block_m64_n256
//       CHECK:   %[[LD:.+]] = ttg.local_load
//       CHECK:   %[[EXT:.+]] = arith.extf %[[LD]] : tensor<1x256xbf16, #{{.+}}> to tensor<1x256xf32, #{{.+}}>
//       CHECK:   %[[CVT:.+]] = ttg.convert_layout %[[EXT]] : tensor<1x256xf32, #{{.+}}> -> tensor<1x256xf32, #{{.+}}>
//       CHECK:   %[[BC:.+]] = tt.broadcast %[[CVT]] : tensor<1x256xf32, #{{.+}}> -> tensor<64x256xf32, #{{.+}}>
//       CHECK:   tt.return %[[BC]]
#blocked_c3 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear_c3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 128]], warp = [[16, 0], [32, 0]], block = []}>
#shared_c3 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#smem_c3 = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:100"} {
tt.func @test_config3_block_m64_n256(%alloc: !ttg.memdesc<1x256xbf16, #shared_c3, #smem_c3, mutable>) -> tensor<64x256xf32, #linear_c3> {
    %ld = ttg.local_load %alloc : !ttg.memdesc<1x256xbf16, #shared_c3, #smem_c3, mutable> -> tensor<1x256xbf16, #blocked_c3>
    %ext = arith.extf %ld : tensor<1x256xbf16, #blocked_c3> to tensor<1x256xf32, #blocked_c3>
    %bc = tt.broadcast %ext : tensor<1x256xf32, #blocked_c3> -> tensor<64x256xf32, #blocked_c3>
    %cvt = ttg.convert_layout %bc : tensor<64x256xf32, #blocked_c3> -> tensor<64x256xf32, #linear_c3>
    tt.return %cvt : tensor<64x256xf32, #linear_c3>
}
}  // end module

// -----

// BLOCK_M=128, BLOCK_N=256, BLOCK_K=64,
// num_warps=4, num_stages=3. Uses the exact #linear encoding from the real TTGIR
// dump. This config previously failed to hoist because resolveConflicts picked
// #blocked over #linear based on sizePerThread score alone, without considering
// that the tmem_load anchor (128x256) would need an expensive 64KB convert.
// CHECK-LABEL: @test_config1_block_m128_n256
//       CHECK:   %[[LD:.+]] = ttg.local_load
//       CHECK:   %[[EXT:.+]] = arith.extf %[[LD]] : tensor<1x256xbf16, #{{.+}}> to tensor<1x256xf32, #{{.+}}>
//       CHECK:   %[[CVT:.+]] = ttg.convert_layout %[[EXT]] : tensor<1x256xf32, #{{.+}}> -> tensor<1x256xf32, #{{.+}}>
//       CHECK:   %[[BC:.+]] = tt.broadcast %[[CVT]] : tensor<1x256xf32, #{{.+}}> -> tensor<128x256xf32, #{{.+}}>
//       CHECK:   tt.return %[[BC]]
#blocked_c1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear_c1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared_c1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#smem_c1 = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:100"} {
tt.func @test_config1_block_m128_n256(%alloc: !ttg.memdesc<1x256xbf16, #shared_c1, #smem_c1, mutable>) -> tensor<128x256xf32, #linear_c1> {
    %ld = ttg.local_load %alloc : !ttg.memdesc<1x256xbf16, #shared_c1, #smem_c1, mutable> -> tensor<1x256xbf16, #blocked_c1>
    %ext = arith.extf %ld : tensor<1x256xbf16, #blocked_c1> to tensor<1x256xf32, #blocked_c1>
    %bc = tt.broadcast %ext : tensor<1x256xf32, #blocked_c1> -> tensor<128x256xf32, #blocked_c1>
    %cvt = ttg.convert_layout %bc : tensor<128x256xf32, #blocked_c1> -> tensor<128x256xf32, #linear_c1>
    tt.return %cvt : tensor<128x256xf32, #linear_c1>
}
}  // end module

// -----

// Conflict resolution test: tmem_load anchor (#linear, 128x256) vs descriptor_load
// anchor (#blocked, 1x256 bias). After AccelerateMatmul, both meet at an addf with
// an explicit #linear->#blocked convert on the accumulator. resolveConflicts must
// pick #linear (512B bias convert) over #blocked (128KB accumulator convert).
// This reproduces Config 1 (BLOCK_M=128, BLOCK_N=256) where the tiebreaker must
// walk backward through the layouts map to anchor boundaries to see the cost
// difference.
// CHECK-LABEL: @test_resolve_conflict_tmem_vs_broadcast
//       CHECK:   %[[ACC:.+]] = ttng.tmem_load
//       CHECK:   %[[BIAS_BF16:.+]] = tt.descriptor_load
//       CHECK:   %[[BIAS_F32:.+]] = arith.extf %[[BIAS_BF16]]
//  Convert must be on the small 1x256 bias, not the 128x256 accumulator:
//       CHECK:   %[[CVT:.+]] = ttg.convert_layout %[[BIAS_F32]] : tensor<1x256xf32, #{{.+}}> -> tensor<1x256xf32, #{{.+}}>
//       CHECK:   %[[BC:.+]] = tt.broadcast %[[CVT]] : tensor<1x256xf32, #{{.+}}> -> tensor<128x256xf32, #{{.+}}>
//       CHECK:   %[[ADD:.+]] = arith.addf %[[ACC]], %[[BC]]
#blocked_cr = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear_cr = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#tmem_cr = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:100"} {
tt.func @test_resolve_conflict_tmem_vs_broadcast(
    %tmem_desc: !ttg.memdesc<128x256xf32, #tmem_cr, #ttng.tensor_memory, mutable>,
    %bias_desc: !tt.tensordesc<tensor<1x256xbf16>>) -> tensor<128x256xbf16, #blocked_cr> {
    %acc_linear = ttng.tmem_load %tmem_desc : !ttg.memdesc<128x256xf32, #tmem_cr, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #linear_cr>
    %acc_blocked = ttg.convert_layout %acc_linear : tensor<128x256xf32, #linear_cr> -> tensor<128x256xf32, #blocked_cr>
    %c0 = arith.constant 0 : i32
    %bias_bf16 = tt.descriptor_load %bias_desc[%c0, %c0] : !tt.tensordesc<tensor<1x256xbf16>> -> tensor<1x256xbf16, #blocked_cr>
    %bias_f32 = arith.extf %bias_bf16 : tensor<1x256xbf16, #blocked_cr> to tensor<1x256xf32, #blocked_cr>
    %bias_bc = tt.broadcast %bias_f32 : tensor<1x256xf32, #blocked_cr> -> tensor<128x256xf32, #blocked_cr>
    %result = arith.addf %acc_blocked, %bias_bc : tensor<128x256xf32, #blocked_cr>
    %out = arith.truncf %result : tensor<128x256xf32, #blocked_cr> to tensor<128x256xbf16, #blocked_cr>
    tt.return %out : tensor<128x256xbf16, #blocked_cr>
}
}  // end module
