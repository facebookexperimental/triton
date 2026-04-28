// RUN: triton-opt %s --nvws-ws-data-partition=num-warp-groups=3 -allow-unregistered-dialect | FileCheck %s

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_alloc_init_layout_retains_metadp_layout
  // CHECK: ttg.convert_layout {{.*}} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #linear>
  // CHECK: ttng.tmem_alloc {{.*}} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
  tt.func @tmem_alloc_init_layout_retains_metadp_layout() {
    %true = arith.constant true
    %a = ub.poison : tensor<256x128xf16, #linear>
    %b = ub.poison : tensor<128x128xf16, #oper_layout>
    %b_shared = ttg.local_alloc %b : (tensor<128x128xf16, #oper_layout>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %d_tmem, %d_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %a_tmem = ttng.tmem_alloc %a : (tensor<256x128xf16, #linear>) -> !ttg.memdesc<256x128xf16, #tmem, #ttng.tensor_memory>
    %mma_tok = ttng.tc_gen5_mma %a_tmem, %b_shared, %d_tmem[%d_tok], %true, %true : !ttg.memdesc<256x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    "use"(%mma_tok) : (!ttg.async.token) -> ()
    tt.return
  }
}
