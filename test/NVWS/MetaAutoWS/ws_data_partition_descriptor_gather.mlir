// RUN: triton-opt %s --nvws-ws-data-partition=num-warp-groups=3 -allow-unregistered-dialect | FileCheck %s

#indices_layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @descriptor_gather_offsets
  // CHECK-DAG: tt.descriptor_gather {{.*}}tensor<64xi32{{.*}} -> tensor<64x64xf16
  // CHECK-DAG: tt.descriptor_gather {{.*}}tensor<64xi32{{.*}} -> tensor<64x64xf16
  tt.func @descriptor_gather_offsets(%a_desc: !tt.tensordesc<tensor<1x64xf16, #shared>>, %b_desc: !tt.tensordesc<tensor<64x64xf16, #shared>>) {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %false = arith.constant false
    %zero = arith.constant dense<0.0> : tensor<128x64xf32, #acc_layout>
    %indices = tt.splat %c0_i32 : i32 -> tensor<128xi32, #indices_layout>
    %a = tt.descriptor_gather %a_desc[%indices, %c0_i32] : (!tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #indices_layout>, i32) -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #oper_layout>
    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x64xf16, #oper_layout>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %zero : (tensor<128x64xf32, #acc_layout>) -> (!ttg.memdesc<128x64xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %false, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<128x64xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x64xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #acc_layout>
    "use"(%c) : (tensor<128x64xf32, #acc_layout>) -> ()
    tt.return
  }
}
