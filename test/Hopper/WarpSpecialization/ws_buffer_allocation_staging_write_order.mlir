// RUN: triton-opt %s --nvgpu-test-ws-buffer-allocation | FileCheck %s

// Regression test reduced from a persistent autoWS GEMM with a two-pass
// epilogue: pass 1 truncates every subtile and accumulates a row statistic,
// pass 2 TMA-stores the subtiles. CSE shares the pass-1 truncf with the
// stores, so each staging local_alloc's source is defined in pass 1.
//
// reorderEpilogOps (in doBufferAllocation) moves each op right after its last
// operand. A staging local_alloc has no memdesc operand, so it used to be
// hoisted next to its truncf, ahead of every TMA store: S S C C instead of
// S C S C. The memory planner later folds the staging allocs of one
// descriptor into a ring of K copies, so with more subtiles than copies a
// later subtile overwrote a slot an earlier TMA store was still reading
// (wrong output; with a separate epilogue-store partition, a hang).
//
// Check each staging write stays after the previous subtile's TMA store.

// CHECK-LABEL: @staging_write_order
// CHECK: ttg.local_store %{{[0-9]+}}, %[[S0:[0-9]+]]
// CHECK-NOT: ttg.local_store
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} %[[S0]]
// CHECK-NEXT: ttng.async_tma_store_token_wait
// CHECK: ttg.local_store %{{[0-9]+}}, %[[S1:[0-9]+]]
// CHECK: ttng.async_tma_copy_local_to_global {{.*}} %[[S1]]
// CHECK: tt.store

// Same for TMA reduce staging.
// CHECK-LABEL: @staging_write_order_reduce
// CHECK: ttg.local_store %{{[0-9]+}}, %[[R0:[0-9]+]]
// CHECK-NOT: ttg.local_store
// CHECK: ttng.async_tma_reduce add, {{.*}} %[[R0]]
// CHECK-NEXT: ttng.async_tma_store_token_wait
// CHECK: ttg.local_store %{{[0-9]+}}, %[[R1:[0-9]+]]
// CHECK: ttng.async_tma_reduce add, {{.*}} %[[R1]]
// CHECK: tt.store

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 1, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 1]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @staging_write_order(%a_desc: !tt.tensordesc<128x64xbf16, #shared>, %b_desc: !tt.tensordesc<64x64xbf16, #shared>, %o_desc: !tt.tensordesc<128x32xbf16, #shared1>, %r_ptr: !tt.ptr<f32>, %k_tiles: i32, %num_tiles: i32) attributes {noinline = false} {
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0>} 32 : i32
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 148 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %pid = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %rows = tt.make_range {async_task_id = array<i32: 0>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %rptr = tt.splat %r_ptr {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    scf.for %t = %pid to %num_tiles step %c148_i32  : i32 {
      %off_m = arith.muli %t, %c64_i32 {async_task_id = array<i32: 0, 2>} : i32
      %acc, %acc_tok = ttng.tmem_alloc {async_task_id = array<i32: 0, 1>} : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %loop:2 = scf.for %ki = %c0_i32 to %k_tiles step %c1_i32 iter_args(%use = %false, %tok = %acc_tok) -> (i1, !ttg.async.token)  : i32 {
        %off_k = arith.muli %ki, %c64_i32 {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        %a = ttg.local_alloc {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : () -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
        nvws.descriptor_load %a_desc[%off_m, %off_k] 16384 %a {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<128x64xbf16, #shared>, i32, i32, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
        %b = ttg.local_alloc {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : () -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
        nvws.descriptor_load %b_desc[%c0_i32, %off_k] 8192 %b {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<64x64xbf16, #shared>, i32, i32, !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
        %bt = ttg.memdesc_trans %b {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared2, #smem, mutable>
        %mma = ttng.tc_gen5_mma %a, %bt, %acc[%tok], %use, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1>} %true, %mma : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2>, tt.scheduled_max_stage = 1 : i32}
      %res, %ld_tok = ttng.tmem_load %acc[%loop#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear4>
      %r3 = tt.reshape %res {async_task_id = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x2x32xf32, #linear5>
      %tr = tt.trans %r3 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
      %s0, %s1 = tt.split %tr {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear>
      // Pass 1: truncate each subtile and accumulate its row sum of squares.
      %v0 = arith.truncf %s0 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear> to tensor<128x32xbf16, #linear>
      %v0c = ttg.convert_layout %v0 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear> -> tensor<128x32xbf16, #blocked1>
      %f0 = arith.extf %v0 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear> to tensor<128x32xf32, #linear>
      %q0 = arith.mulf %f0, %f0 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear>
      %ss0 = "tt.reduce"(%q0) <{axis = 1 : i32}> ({
      ^bb0(%x: f32, %y: f32):
        %z = arith.addf %x, %y {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %z {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<128x32xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %acc0 = arith.addf %ss0, %cst {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %v1 = arith.truncf %s1 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear> to tensor<128x32xbf16, #linear>
      %v1c = ttg.convert_layout %v1 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear> -> tensor<128x32xbf16, #blocked1>
      %f1 = arith.extf %v1 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear> to tensor<128x32xf32, #linear>
      %q1 = arith.mulf %f1, %f1 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear>
      %ss1 = "tt.reduce"(%q1) <{axis = 1 : i32}> ({
      ^bb0(%x: f32, %y: f32):
        %z = arith.addf %x, %y {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %z {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<128x32xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %acc1 = arith.addf %acc0, %ss1 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %accb = ttg.convert_layout %acc1 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      // Pass 2: TMA-store each subtile through its staging buffer.
      %st0 = ttg.local_alloc %v0c {async_task_id = array<i32: 0>} : (tensor<128x32xbf16, #blocked1>) -> !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable>
      %tk0 = ttng.async_tma_copy_local_to_global %o_desc[%off_m, %c0_i32] %st0 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x32xbf16, #shared1>, !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tk0 {async_task_id = array<i32: 0>} : !ttg.async.token
      %st1 = ttg.local_alloc %v1c {async_task_id = array<i32: 0>} : (tensor<128x32xbf16, #blocked1>) -> !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable>
      %tk1 = ttng.async_tma_copy_local_to_global %o_desc[%off_m, %c32_i32] %st1 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x32xbf16, #shared1>, !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tk1 {async_task_id = array<i32: 0>} : !ttg.async.token
      %rp = tt.addptr %rptr, %rows {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      tt.store %rp, %accb {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
    } {async_task_id = array<i32: 0, 1, 2>, tt.data_partition_factor = 1 : i32, tt.smem_alloc_algo = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
  tt.func public @staging_write_order_reduce(%a_desc: !tt.tensordesc<128x64xbf16, #shared>, %b_desc: !tt.tensordesc<64x64xbf16, #shared>, %o_desc: !tt.tensordesc<128x32xbf16, #shared1>, %r_ptr: !tt.ptr<f32>, %k_tiles: i32, %num_tiles: i32) attributes {noinline = false} {
    %false = arith.constant {async_task_id = array<i32: 1>} false
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0>} 32 : i32
    %c148_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 148 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %pid = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %rows = tt.make_range {async_task_id = array<i32: 0>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %rptr = tt.splat %r_ptr {async_task_id = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    scf.for %t = %pid to %num_tiles step %c148_i32  : i32 {
      %off_m = arith.muli %t, %c64_i32 {async_task_id = array<i32: 0, 2>} : i32
      %acc, %acc_tok = ttng.tmem_alloc {async_task_id = array<i32: 0, 1>} : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %loop:2 = scf.for %ki = %c0_i32 to %k_tiles step %c1_i32 iter_args(%use = %false, %tok = %acc_tok) -> (i1, !ttg.async.token)  : i32 {
        %off_k = arith.muli %ki, %c64_i32 {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        %a = ttg.local_alloc {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : () -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
        nvws.descriptor_load %a_desc[%off_m, %off_k] 16384 %a {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<128x64xbf16, #shared>, i32, i32, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
        %b = ttg.local_alloc {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : () -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
        nvws.descriptor_load %b_desc[%c0_i32, %off_k] 8192 %b {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32, multicast = false} : !tt.tensordesc<64x64xbf16, #shared>, i32, i32, !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
        %bt = ttg.memdesc_trans %b {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared2, #smem, mutable>
        %mma = ttng.tc_gen5_mma %a, %bt, %acc[%tok], %use, %true {async_task_id = array<i32: 1>, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xbf16, #shared2, #smem, mutable>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {async_task_id = array<i32: 0, 1>} %true, %mma : i1, !ttg.async.token
      } {async_task_id = array<i32: 0, 1, 2>, tt.scheduled_max_stage = 1 : i32}
      %res, %ld_tok = ttng.tmem_load %acc[%loop#1] {async_task_id = array<i32: 0>} : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear4>
      %r3 = tt.reshape %res {async_task_id = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x2x32xf32, #linear5>
      %tr = tt.trans %r3 {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x32xf32, #linear5> -> tensor<128x32x2xf32, #linear6>
      %s0, %s1 = tt.split %tr {async_task_id = array<i32: 0>} : tensor<128x32x2xf32, #linear6> -> tensor<128x32xf32, #linear>
      // Pass 1: truncate each subtile and accumulate its row sum of squares.
      %v0 = arith.truncf %s0 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear> to tensor<128x32xbf16, #linear>
      %v0c = ttg.convert_layout %v0 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear> -> tensor<128x32xbf16, #blocked1>
      %f0 = arith.extf %v0 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear> to tensor<128x32xf32, #linear>
      %q0 = arith.mulf %f0, %f0 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear>
      %ss0 = "tt.reduce"(%q0) <{axis = 1 : i32}> ({
      ^bb0(%x: f32, %y: f32):
        %z = arith.addf %x, %y {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %z {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<128x32xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %acc0 = arith.addf %ss0, %cst {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %v1 = arith.truncf %s1 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear> to tensor<128x32xbf16, #linear>
      %v1c = ttg.convert_layout %v1 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear> -> tensor<128x32xbf16, #blocked1>
      %f1 = arith.extf %v1 {async_task_id = array<i32: 0>} : tensor<128x32xbf16, #linear> to tensor<128x32xf32, #linear>
      %q1 = arith.mulf %f1, %f1 {async_task_id = array<i32: 0>} : tensor<128x32xf32, #linear>
      %ss1 = "tt.reduce"(%q1) <{axis = 1 : i32}> ({
      ^bb0(%x: f32, %y: f32):
        %z = arith.addf %x, %y {async_task_id = array<i32: 0>} : f32
        tt.reduce.return %z {async_task_id = array<i32: 0>} : f32
      }) {async_task_id = array<i32: 0>} : (tensor<128x32xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %acc1 = arith.addf %acc0, %ss1 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %accb = ttg.convert_layout %acc1 {async_task_id = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      // Pass 2: TMA-store each subtile through its staging buffer.
      %st0 = ttg.local_alloc %v0c {async_task_id = array<i32: 0>} : (tensor<128x32xbf16, #blocked1>) -> !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable>
      %tk0 = ttng.async_tma_reduce add, %o_desc[%off_m, %c0_i32] %st0 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x32xbf16, #shared1>, !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tk0 {async_task_id = array<i32: 0>} : !ttg.async.token
      %st1 = ttg.local_alloc %v1c {async_task_id = array<i32: 0>} : (tensor<128x32xbf16, #blocked1>) -> !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable>
      %tk1 = ttng.async_tma_reduce add, %o_desc[%off_m, %c32_i32] %st1 {async_task_id = array<i32: 0>} : !tt.tensordesc<128x32xbf16, #shared1>, !ttg.memdesc<128x32xbf16, #shared1, #smem, mutable> -> !ttg.async.token
      ttng.async_tma_store_token_wait %tk1 {async_task_id = array<i32: 0>} : !ttg.async.token
      %rp = tt.addptr %rptr, %rows {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      tt.store %rp, %accb {async_task_id = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
    } {async_task_id = array<i32: 0, 1, 2>, tt.data_partition_factor = 1 : i32, tt.smem_alloc_algo = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["computation", "gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
