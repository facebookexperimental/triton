// RUN: triton-opt %s -tritongpu-optimize-accumulator-init | FileCheck %s

// Two chained accumulating MMAv5 dots into one logical accumulator (the HSTU
// reduce_dq "compute fold" shape: dk = dot(a0,b0,dk); dk = dot(a1,b1,dk)).
// AccelerateMatmul lowers each dot into its own tmem_alloc + tc_gen5_mma +
// tmem_load, threading the accumulator through registers, so dot2's TMEM tile is
// initialized from dot1's TMEM read-out. ChainAccumulatorInPlace must coalesce
// them into a single in-place accumulation on ONE TMEM tile (no intermediate
// tile / read-out bridge), matching the TLX single-dk_tiles shape. Left alone,
// the two tiles become a TMEM<->TMEM load/store bridge that blocks TMEM slot
// reuse downstream (the WSCodePartition reuse-order check aborts).

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blockedT = #ttg.blocked<{sizePerThread = [128, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linearTMEM = #ttg.linear<{register = [[0, 1], [8, 0], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @chain_accumulator_in_place
// One accumulator tile; both MMAs accumulate into it; the second reads the
// first's completion token in place; a single read-out; no bridge tile/load.
// CHECK:       scf.for
// CHECK:         %[[ACC:.*]], %[[TK0:.*]] = ttng.tmem_alloc %arg
// CHECK:         %[[TK1:.*]] = ttng.tc_gen5_mma {{.*}} %[[ACC]][%[[TK0]]]
// CHECK-NOT:     ttng.tmem_alloc
// CHECK-NOT:     ttng.tmem_load
// CHECK:         %[[TK2:.*]] = ttng.tc_gen5_mma {{.*}} %[[ACC]][%[[TK1]]], %true
// CHECK:         ttng.tmem_load %[[ACC]][%[[TK2]]]
tt.func public @chain_accumulator_in_place(
    %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b0: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %a1: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b1: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %n: i32) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>) : i32 {
    // dot1 tile
    %t1, %tok1 = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m1 = ttng.tc_gen5_mma %a0, %b0, %t1[%tok1], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v1, %ld1 = ttng.tmem_load %t1[%m1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    // dot2 tile, initialized from dot1's read-out
    %t2, %tok2 = ttng.tmem_alloc %v1 : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m2 = ttng.tc_gen5_mma %a1, %b1, %t2[%tok2], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v2, %ld2 = ttng.tmem_load %t2[%m2] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    scf.yield %v2 : tensor<128x128xf32, #blocked>
  }
  tt.return
}

// CHECK-LABEL: @chain_accumulator_convert_layout_fuse
// A single-use convert_layout in the init chain is elementwise identity, so
// the chain still fuses onto one tile and the conversion is erased.
// CHECK:       scf.for
// CHECK:         %[[ACCC:.*]], %[[TKC0:.*]] = ttng.tmem_alloc %arg
// CHECK:         %[[TKC1:.*]] = ttng.tc_gen5_mma {{.*}} %[[ACCC]][%[[TKC0]]]
// CHECK-NOT:     ttng.tmem_alloc
// CHECK-NOT:     ttng.tmem_load
// CHECK-NOT:     ttg.convert_layout
// CHECK:         %[[TKC2:.*]] = ttng.tc_gen5_mma {{.*}} %[[ACCC]][%[[TKC1]]], %true
// CHECK:         ttng.tmem_load %[[ACCC]][%[[TKC2]]]
tt.func public @chain_accumulator_convert_layout_fuse(
    %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b0: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %a1: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b1: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %n: i32) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>) : i32 {
    %t1, %tok1 = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m1 = ttng.tc_gen5_mma %a0, %b0, %t1[%tok1], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v1, %ld1 = ttng.tmem_load %t1[%m1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %cv = ttg.convert_layout %v1 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #linearTMEM>
    %t2, %tok2 = ttng.tmem_alloc %cv : (tensor<128x128xf32, #linearTMEM>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m2 = ttng.tc_gen5_mma %a1, %b1, %t2[%tok2], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v2, %ld2 = ttng.tmem_load %t2[%m2] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    scf.yield %v2 : tensor<128x128xf32, #blocked>
  }
  tt.return
}

// CHECK-LABEL: @chain_accumulator_transpose_no_fuse
// A square-tile transpose in the init chain changes values (both tiles share
// the same tensor_memory_encoding), so the pattern must NOT fuse: both tiles,
// both read-outs, the transpose, and the conversion back to a TMEM-compatible
// layout all survive.
// CHECK:       scf.for
// CHECK:         ttng.tmem_alloc
// CHECK:         ttng.tc_gen5_mma
// CHECK:         ttng.tmem_load
// CHECK:         tt.trans
// CHECK:         ttg.convert_layout
// CHECK:         ttng.tmem_alloc
// CHECK:         ttng.tc_gen5_mma
// CHECK:         ttng.tmem_load
tt.func public @chain_accumulator_transpose_no_fuse(
    %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b0: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %a1: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b1: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %n: i32) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>) : i32 {
    %t1, %tok1 = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m1 = ttng.tc_gen5_mma %a0, %b0, %t1[%tok1], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v1, %ld1 = ttng.tmem_load %t1[%m1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %vt = tt.trans %v1 {order = array<i32: 1, 0>} : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blockedT>
    %cvt = ttg.convert_layout %vt : tensor<128x128xf32, #blockedT> -> tensor<128x128xf32, #blocked>
    %t2, %tok2 = ttng.tmem_alloc %cvt : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m2 = ttng.tc_gen5_mma %a1, %b1, %t2[%tok2], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v2, %ld2 = ttng.tmem_load %t2[%m2] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    scf.yield %v2 : tensor<128x128xf32, #blocked>
  }
  tt.return
}

// CHECK-LABEL: @chained_dead_src
// The tile init is dead even though it is nonzero: the first accessor in
// token order (mma1, useD=false, pred=true) overwrites the whole tile
// before mma2/load read it.
// CHECK: %[[ACC:.*]], %[[TK0:.*]] = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32
// CHECK: tc_gen5_mma {{.*}} %[[ACC]]
// CHECK: tc_gen5_mma {{.*}} %[[ACC]]
// CHECK: tmem_load %[[ACC]]
tt.func public @chained_dead_src(
    %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b0: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %n: i32) {
  %true = arith.constant true
  %false = arith.constant false
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %one = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %one) -> (tensor<128x128xf32, #blocked>) : i32 {
    %t1, %tok1 = ttng.tmem_alloc %one : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m1 = ttng.tc_gen5_mma %a0, %b0, %t1[%tok1], %false, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %m2 = ttng.tc_gen5_mma %a0, %b0, %t1[%m1], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v2, %ld2 = ttng.tmem_load %t1[%m2] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    scf.yield %v2 : tensor<128x128xf32, #blocked>
  }
  tt.return
}

// CHECK-LABEL: @read_before_overwrite
// The load observes the init before the overwriting MMA runs: src must stay.
// CHECK: ttng.tmem_alloc %{{.*}} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32
tt.func public @read_before_overwrite(
    %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b0: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %n: i32) {
  %true = arith.constant true
  %false = arith.constant false
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %one = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %one) -> (tensor<128x128xf32, #blocked>) : i32 {
    %t1, %tok1 = ttng.tmem_alloc %one : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %v0, %ld0 = ttng.tmem_load %t1[%tok1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %m1 = ttng.tc_gen5_mma %a0, %b0, %t1[%ld0], %false, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.yield %v0 : tensor<128x128xf32, #blocked>
  }
  tt.return
}

// CHECK-LABEL: @predicated_overwrite
// useD=false with pred=false writes nothing, so the load still observes the
// init: src must stay.
// CHECK: ttng.tmem_alloc %{{.*}} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32
tt.func public @predicated_overwrite(
    %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b0: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %n: i32) {
  %true = arith.constant true
  %false = arith.constant false
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %one = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %one) -> (tensor<128x128xf32, #blocked>) : i32 {
    %t1, %tok1 = ttng.tmem_alloc %one : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m1 = ttng.tc_gen5_mma %a0, %b0, %t1[%tok1], %false, %false : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v1, %ld1 = ttng.tmem_load %t1[%m1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    scf.yield %v1 : tensor<128x128xf32, #blocked>
  }
  tt.return
}

// CHECK-LABEL: @chain_fuse_clears_src
// End to end: two tiles fuse onto one, and the fused tile's zero init is
// dropped because the first MMA overwrites it before anyone reads it.
// CHECK: %[[ACC:.*]], %[[TK0:.*]] = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32
// CHECK: tc_gen5_mma {{.*}} %[[ACC]]
// CHECK: tc_gen5_mma {{.*}} %[[ACC]]
// CHECK: tmem_load %[[ACC]]
tt.func public @chain_fuse_clears_src(
    %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b0: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %n: i32) {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>) : i32 {
    %t1, %tok1 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m1 = ttng.tc_gen5_mma %a0, %b0, %t1[%tok1], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v1, %ld1 = ttng.tmem_load %t1[%m1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %t2, %tok2 = ttng.tmem_alloc %v1 : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m2 = ttng.tc_gen5_mma %a0, %b0, %t2[%tok2], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v2, %ld2 = ttng.tmem_load %t2[%m2] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    scf.yield %v2 : tensor<128x128xf32, #blocked>
  }
  tt.return
}

// CHECK-LABEL: @token_fanout_keeps_src
// Both the MMA and the load consume the alloc token directly, so execution
// order is ambiguous and the init must stay.
// CHECK: ttng.tmem_alloc %{{.*}} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32
tt.func public @token_fanout_keeps_src(
    %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b0: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %n: i32) {
  %true = arith.constant true
  %false = arith.constant false
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %one = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %one) -> (tensor<128x128xf32, #blocked>) : i32 {
    %t1, %tok1 = ttng.tmem_alloc %one : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m1 = ttng.tc_gen5_mma %a0, %b0, %t1[%tok1], %false, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v0, %ld0 = ttng.tmem_load %t1[%tok1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    scf.yield %v0 : tensor<128x128xf32, #blocked>
  }
  tt.return
}

// CHECK-LABEL: @store_user_keeps_src
// A tmem_store user is not a tracked accessor (a predicated-off store
// preserves the tile), so the init must stay even though an MMA overwrites
// the tile first in token order.
// CHECK: ttng.tmem_alloc %{{.*}} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32
tt.func public @store_user_keeps_src(
    %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b0: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %n: i32) {
  %true = arith.constant true
  %false = arith.constant false
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %one = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %one) -> (tensor<128x128xf32, #blocked>) : i32 {
    %t1, %tok1 = ttng.tmem_alloc %one : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %m1 = ttng.tc_gen5_mma %a0, %b0, %t1[%tok1], %false, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %st = ttng.tmem_store %one, %t1[%m1], %false : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %v1, %ld1 = ttng.tmem_load %t1[%st] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    scf.yield %v1 : tensor<128x128xf32, #blocked>
  }
  tt.return
}

// CHECK-LABEL: @branch_accessor_keeps_src
// Both arms overwrite the tile before the load reads it, so the init is dead
// in every execution order -- but the accessors are not in the alloc's block,
// so no linear token order can be proven and the init conservatively stays.
// CHECK: ttng.tmem_alloc %{{.*}} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32
tt.func public @branch_accessor_keeps_src(
    %a0: !ttg.memdesc<128x64xf16, #shared, #smem>,
    %b0: !ttg.memdesc<64x128xf16, #shared1, #smem>,
    %cond: i1) -> tensor<128x128xf32, #blocked> {
  %true = arith.constant true
  %false = arith.constant false
  %one = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
  %t1, %tok1 = ttng.tmem_alloc %one : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
  %mtok = scf.if %cond -> (!ttg.async.token) {
    %m1 = ttng.tc_gen5_mma %a0, %b0, %t1[%tok1], %false, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.yield %m1 : !ttg.async.token
  } else {
    %m2 = ttng.tc_gen5_mma %a0, %b0, %t1[%tok1], %false, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.yield %m2 : !ttg.async.token
  }
  %v0, %ld0 = ttng.tmem_load %t1[%mtok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
  tt.return %v0 : tensor<128x128xf32, #blocked>
}

}
