// RUN: triton-opt %s -tritongpu-hoist-tmem-alloc -nvgpu-ws-data-partition=num-warp-groups=1 -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-automatic-warp-specialization="num-stages=2 use-meta-partitioner=true smem-budget=232448" | FileCheck %s

#indices_layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#b_layout = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @regular_load_alloc_mma_operand
  // CHECK: ttg.warp_specialize
  // CHECK: tt.load
  // CHECK: ttg.local_store
  // CHECK: ttng.tc_gen5_mma
  tt.func @regular_load_alloc_mma_operand(
      %a_desc: !tt.tensordesc<tensor<1x64xf16, #shared>>,
      %b_ptr_init: tensor<64x128x!tt.ptr<f16>, #b_layout>,
      %off_m: i32, %offs_n: tensor<64x128xi32, #b_layout>, %off_k: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
    %k_tiles = arith.constant 32 : i32
    %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init_tok = ttng.tmem_store %zero, %acc[%tok], %true : tensor<128x128xf32, #acc_layout> -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %loop:3 = scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(
        %flag = %true, %b_ptr = %b_ptr_init, %tok_arg = %init_tok)
        -> (i1, tensor<64x128x!tt.ptr<f16>, #b_layout>, !ttg.async.token) : i32 {
      %indices = tt.splat %off_m : i32 -> tensor<128xi32, #indices_layout>
      %a = tt.descriptor_gather %a_desc[%indices, %off_k] : (!tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #indices_layout>, i32) -> tensor<128x64xf16, #oper_layout>
      %b_ptrs = tt.addptr %b_ptr, %offs_n : tensor<64x128x!tt.ptr<f16>, #b_layout>, tensor<64x128xi32, #b_layout>
      %b = tt.load %b_ptrs : tensor<64x128x!tt.ptr<f16>, #b_layout>
      %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_smem = ttg.local_alloc %b : (tensor<64x128xf16, #b_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %mma = ttng.tc_gen5_mma %a_smem, %b_smem, %acc[%tok_arg], %flag, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
      %c, %load_tok = ttng.tmem_load %acc[%mma] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
      %next = math.exp2 %c : tensor<128x128xf32, #acc_layout>
      %store_tok = ttng.tmem_store %next, %acc[%load_tok], %true : tensor<128x128xf32, #acc_layout> -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
      scf.yield %true, %b_ptrs, %store_tok : i1, tensor<64x128x!tt.ptr<f16>, #b_layout>, !ttg.async.token
    } {tt.warp_specialize, tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32}
    tt.return
  }
}
