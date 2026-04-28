// RUN: triton-opt %s --nvws-ws-data-partition=num-warp-groups=3 -allow-unregistered-dialect | FileCheck %s

#indices_layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#b_layout = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @descriptor_gather_loop_unsliced_ptr_arg
  // CHECK-DAG: tt.descriptor_gather {{.*}}tensor<64xi32{{.*}} -> tensor<64x64xf16
  // CHECK-DAG: tt.descriptor_gather {{.*}}tensor<64xi32{{.*}} -> tensor<64x64xf16
  tt.func @descriptor_gather_loop_unsliced_ptr_arg(%a_desc: !tt.tensordesc<tensor<1x64xf16, #shared>>, %b_ptr_init: tensor<64x128x!tt.ptr<f16>, #b_layout> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
    %k_tiles = arith.constant 4 : i32
    %offs = arith.constant dense<0> : tensor<64x128xi32, #b_layout>
    scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero, %flag = %true, %b_ptr = %b_ptr_init) -> (tensor<128x128xf32, #acc_layout>, i1, tensor<64x128x!tt.ptr<f16>, #b_layout>) : i32 {
      %indices = tt.splat %k : i32 -> tensor<128xi32, #indices_layout>
      %a = tt.descriptor_gather %a_desc[%indices, %k] : (!tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #indices_layout>, i32) -> tensor<128x64xf16, #oper_layout>
      %b_ptrs = tt.addptr %b_ptr, %offs {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>} : tensor<64x128x!tt.ptr<f16>, #b_layout>, tensor<64x128xi32, #b_layout>
      %b = tt.load %b_ptrs : tensor<64x128x!tt.ptr<f16>, #b_layout>
      %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #b_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %flag, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
      %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
      %use_acc = arith.select %flag, %false, %true : i1
      scf.yield %c, %use_acc, %b_ptrs : tensor<128x128xf32, #acc_layout>, i1, tensor<64x128x!tt.ptr<f16>, #b_layout>
    } {tt.warp_specialize, tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32}
    tt.return
  }
}
