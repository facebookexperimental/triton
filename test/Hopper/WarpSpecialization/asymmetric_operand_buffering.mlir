// RUN: triton-opt %s -tritongpu-assign-latencies='num-stages=3 use-meta-ws=true' -tritongpu-schedule-loops='num-stages=3 use-meta-ws=true' | FileCheck %s --check-prefix=SCHEDULE

#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SCHEDULE-LABEL: @asymmetric_operand_buffering
  // SCHEDULE: tt.descriptor_load {{.*}}loop.stage = 0 : i32{{.*}}tt.requested_buffer_depth = 3 : i32
  // SCHEDULE: tt.descriptor_load {{.*}}loop.stage = 1 : i32{{.*}}tt.requested_buffer_depth = 2 : i32
  // SCHEDULE: ttng.tc_gen5_mma {{.*}}loop.stage = 2 : i32

  tt.func @asymmetric_operand_buffering(
      %k_tiles: i32,
      %a_desc: !tt.tensordesc<128x64xf16, #shared>,
      %b_desc: !tt.tensordesc<64x128xf16, #shared>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
    %acc, %acc_token = ttng.tmem_alloc %zero : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result:2 = scf.for %k = %c0 to %k_tiles step %c1 iter_args(%token = %acc_token, %use_acc = %false) -> (!ttg.async.token, i1) : i32 {
      %a = tt.descriptor_load %a_desc[%c0, %k] {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #oper_layout>
      %b = tt.descriptor_load %b_desc[%k, %c0] {ttg.partition = array<i32: 1>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #oper_layout>
      %a_smem = ttg.local_alloc %a {ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_smem = ttg.local_alloc %b {ttg.partition = array<i32: 1>} : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %next = ttng.tc_gen5_mma %a_smem, %b_smem, %acc[%token], %use_acc, %true {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
      scf.yield %next, %true : !ttg.async.token, i1
    } {tt.disallow_acc_multi_buffer, tt.lhs_buffer_depth = 3 : i32, tt.num_stages = 3 : i32, tt.rhs_buffer_depth = 2 : i32, tt.warp_specialize, ttg.partition.stages = [1 : i32, 0 : i32], ttg.partition.types = ["gemm", "load"]}
    tt.return
  }
}
