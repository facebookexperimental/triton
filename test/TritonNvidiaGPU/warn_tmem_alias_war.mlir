// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-warn-tmem-alias-war -verify-diagnostics

// An f32 TMEM accumulator (qk) read, then an aliased f16 store (P) into the same
// TMEM root via a different-element-type reinterpret, with no barrier between:
// an intra-task write-after-read hazard. The pass must warn (and NOT modify IR).

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  tt.func @aliased_f32_read_f16_store(%val: tensor<128x128xf16, #linear>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%c1] : !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // expected-note @+1 {{the aliased TMEM read is here}}
    %qk = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    // expected-warning @+1 {{aliased-TMEM write-after-read hazard}}
    ttng.tmem_store %val, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Same TMEM root, read and store the SAME element type (f32): no mismatch, so
// no warning. (-verify-diagnostics fails if any unexpected diagnostic appears.)

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  tt.func @matched_dtype_no_warning(%val: tensor<128x128xf32, #linear>, %pred: i1) {
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %qk = ttng.tmem_load %root : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttng.tmem_store %val, %root, %pred : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Already guarded by a barrier between the read and the aliased store: no warning.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  tt.func @guarded_no_warning(%val: tensor<128x128xf16, #linear>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%c1] : !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %qk = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttg.barrier all
    ttng.tmem_store %val, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
