// RUN: triton-opt %s --nvgpu-test-ws-buffer-allocation | FileCheck %s

// Regression test distilled from a persistent LCE+bias kernel (P2418636137).
//
// When the epilogue store lives inside the warp-specialized loop body, that
// body is the "epilog block" whose terminator is scf.yield. reorderEpilogOps
// (in doBufferAllocation) streamlines a channel consumer's forward slice to
// pack ops next to their operands. The loop-carried TMEM accumulator token
// makes scf.yield part of the tmem_load consumer's forward slice, so without
// an IsTerminator guard the yield was moved up next to tmem_load -- out of
// terminator position -- leaving the block with no terminator. This tripped
// the verifier ("'scf.yield' op must be the last operation in the parent
// block") and crashed doCodePartition downstream with a getTerminator()
// assertion (Block.cpp: mightHaveTerminator()).
//
// Check the epilogue chain stays in the body and scf.yield remains the
// terminator, immediately following the store.

// CHECK-LABEL: @reorder_epilog_inloop_store
// CHECK:      scf.for
// CHECK:        ttng.tmem_load
// CHECK:        arith.truncf
// CHECK:        ttg.convert_layout
// CHECK:        tt.store
// CHECK-NEXT:   scf.yield

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reorder_epilog_inloop_store(%x_desc: !tt.tensordesc<64x128xbf16, #shared>, %w_desc: !tt.tensordesc<64x128xbf16, #shared>, %out_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant {async_task_id = array<i32: 0, 2>} true
    %c0_i32 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
    %c152_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 152 : i32
    %c18432_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 18432 : i32
    %start_pid = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %optr = tt.splat %out_ptr {async_task_id = array<i32: 0>} : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
    %acc, %acc_tok = ttng.tmem_alloc {async_task_id = array<i32: 0, 2>} : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %r = scf.for %arg = %start_pid to %c18432_i32 step %c152_i32 iter_args(%tok = %acc_tok) -> (!ttg.async.token)  : i32 {
      %x = tt.descriptor_load %x_desc[%c0_i32, %c0_i32] {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x128xbf16, #shared> -> tensor<64x128xbf16, #blocked1>
      %xs = ttg.local_alloc %x {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
      %w = tt.descriptor_load %w_desc[%c0_i32, %c0_i32] {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<64x128xbf16, #shared> -> tensor<64x128xbf16, #blocked1>
      %ws = ttg.local_alloc %w {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<64x128xbf16, #blocked1>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
      %wt = ttg.memdesc_trans %ws {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared1, #smem>
      %mma_tok = ttng.tc_gen5_mma %xs, %wt, %acc[%tok], %true, %true {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.self_latency = 0 : i32} : !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<128x64xbf16, #shared1, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %res, %ld_tok = ttng.tmem_load %acc[%mma_tok] {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear>
      %t = arith.truncf %res {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x64xf32, #linear> to tensor<64x64xbf16, #linear>
      %c = ttg.convert_layout %t {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #linear> -> tensor<64x64xbf16, #blocked>
      tt.store %optr, %c {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x64x!tt.ptr<bf16>, #blocked>
      scf.yield {async_task_id = array<i32: 0>} %ld_tok : !ttg.async.token
    } {async_task_id = array<i32: 0, 1, 2>, tt.data_partition_factor = 1 : i32, tt.scheduled_max_stage = 1 : i32, tt.separate_epilogue_store = true, tt.smem_alloc_algo = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32], ttg.partition.types = ["computation", "load", "gemm"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
