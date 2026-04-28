// RUN: triton-opt %s --nvws-ws-data-partition=num-warp-groups=3 -allow-unregistered-dialect | FileCheck %s

#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @smem_arg_subslice
  // CHECK-DAG: ttg.memdesc_subslice %arg0{{\[}}0, 0{{\]}} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared, #smem>
  // CHECK-DAG: ttg.memdesc_subslice %arg0{{\[}}64, 0{{\]}} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared, #smem>
  // CHECK-NOT: ttg.memdesc_subslice {{.*}}#ttng.tensor_memory
  tt.func @smem_arg_subslice(%a_shared: !ttg.memdesc<128x64xf16, #shared, #smem>, %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
    %b = tt.descriptor_load %b_desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %zero : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    "use"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
    tt.return
  }
}
