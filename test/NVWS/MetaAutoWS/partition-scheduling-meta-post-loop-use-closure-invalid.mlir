// RUN: not triton-opt %s --nvws-partition-scheduling-meta -allow-unregistered-dialect 2>&1 | FileCheck %s
// RUN: not triton-opt %s --nvws-partition-scheduling-meta -mlir-print-ir-after-failure -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=NOMUTATE

// CHECK: partitioned op outside tt.warp_specialize loop has a result use outside its WS tag closure
// NOMUTATE: partitioned op outside tt.warp_specialize loop has a result use outside its WS tag closure
// NOMUTATE-LABEL: @post_loop_bad_use_closure
// NOMUTATE: scf.for
// NOMUTATE: } {tt.warp_specialize}
// NOMUTATE-NOT: ttg.partition.types
// NOMUTATE: arith.addf

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#load_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
tt.func public @post_loop_bad_use_closure(
  %A_shared: !ttg.memdesc<128x64xf16, #shared, #smem>,
  %B_desc: !tt.tensordesc<tensor<64x64xf16, #shared>>,
  %n_tiles: i32
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %zero = arith.constant dense<0.0> : tensor<128x64xf32, #blocked>

  %loop_out = scf.for %i = %c0_i32 to %n_tiles step %c64_i32 iter_args(
    %acc = %zero
  ) -> (tensor<128x64xf32, #blocked>) : i32 {
    %B = tt.descriptor_load %B_desc[%i, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #load_blocked>
    %B_shared = ttg.local_alloc %B : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %B_trans = ttg.memdesc_trans %B_shared {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared_t, #smem>
    %C_tmem, %C_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %A_shared, %B_trans, %C_tmem[%C_tok], %false, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared_t, #smem>, !ttg.memdesc<128x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>
    %result, %result_tok = ttng.tmem_load %C_tmem[%mma_tok] : !ttg.memdesc<128x64xf32, #tmem_acc, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    %new_acc = arith.addf %acc, %result : tensor<128x64xf32, #blocked>
    scf.yield %new_acc : tensor<128x64xf32, #blocked>
  } {tt.warp_specialize}

  %post = arith.addf %loop_out, %zero {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked>
  "use"(%post) : (tensor<128x64xf32, #blocked>) -> ()
  tt.return
}
}
