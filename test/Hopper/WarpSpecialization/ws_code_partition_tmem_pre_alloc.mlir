// RUN: triton-opt %s --nvgpu-test-ws-code-partition=num-buffers=2 | FileCheck %s
// XFAIL: *

// Regression test for B-2-F1 / T273465371.
//
// A TMEM allocation with a source tensor is produced in task 0 and consumed by
// a Gen5 MMA in task 1. The pre-allocation channel discovery path classifies
// this as DataChannelKind::TMEM, but currently stores it in a plain Channel.
// TMEM memory lowering then casts it to TmemDataChannel and crashes/reads an
// invalid allocation contract. Once fixed, the pre-allocated TMEM channel
// should be multi-buffered and accessed through a memdesc_index subview.

// CHECK-LABEL: @tmem_alloc_src_cross_partition
// CHECK: ttg.warp_specialize
// CHECK: ttng.tmem_alloc
// CHECK: ttg.memdesc_index
// CHECK: ttng.tmem_store
// CHECK: ttng.tc_gen5_mma

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_alloc_src_cross_partition(%b_smem: !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>) attributes {noinline = false} {
    %true = arith.constant {ttg.partition = array<i32: 0, 1>} true
    %false = arith.constant {ttg.partition = array<i32: 0, 1>} false
    %cst = arith.constant {ttg.partition = array<i32: 0>} dense<1.000000e+00> : tensor<128x128xbf16, #blocked>
    %acc, %acc_token = ttng.tmem_alloc {ttg.partition = array<i32: 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %p_tmem = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xbf16, #blocked>) -> !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory>
    %mma_token = ttng.tc_gen5_mma %p_tmem, %b_smem, %acc[%acc_token], %false, %true {ttg.partition = array<i32: 1>, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
