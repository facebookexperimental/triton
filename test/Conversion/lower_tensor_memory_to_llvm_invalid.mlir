// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -verify-diagnostics

// A register layout that cannot lower to a TMEM ld/st encoding must decline
// with a diagnostic, not crash lowering (no_verify_layout bypasses the
// verifier, so the conversion patterns must propagate the failure).

#b4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#v4 = #tlx.user_layout<#tlx.no_verify_layout<#b4>>
#tmem128 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @lower_noverify(%src: !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>) -> tensor<128x128xf32, #v4> {
    // expected-error@+2 {{warps=1,2 must map to rows=32,64}}
    // expected-error@+1 {{failed to legalize operation 'ttng.tmem_load' that was explicitly marked illegal}}
    %loaded = ttng.tmem_load %src : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #v4>
    tt.return %loaded : tensor<128x128xf32, #v4>
  }
}

// -----

// A task of 2 warps reaches only the TMEM lanes of its own warps. A register bit
// on row 64 of a whole 128-row buffer would put warp 0 on lanes 64-95.

#l2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0]], block = []}>
#v2 = #tlx.user_layout<#tlx.no_verify_layout<#l2>>
#tmem128x16 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 16, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @two_warps_past_their_lanes(%src: !ttg.memdesc<128x16xf32, #tmem128x16, #ttng.tensor_memory, mutable>) -> tensor<128x16xf32, #v2> {
    // expected-error@+2 {{A task of 2 warp(s) can only access the TMEM rows in the lanes of its own warps}}
    // expected-error@+1 {{failed to legalize operation 'ttng.tmem_load' that was explicitly marked illegal}}
    %loaded = ttng.tmem_load %src : !ttg.memdesc<128x16xf32, #tmem128x16, #ttng.tensor_memory, mutable> -> tensor<128x16xf32, #v2>
    tt.return %loaded : tensor<128x16xf32, #v2>
  }
}
