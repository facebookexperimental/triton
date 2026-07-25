// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-partition-scheduling-meta="merge-reduction merge-epilogue-to-computation" | FileCheck %s
// XFAIL: *

// Regression test for B-3-F1 / T273467650.
// `merge-reduction` should route TMA reduction ops and their producers to the
// computation partition for the relevant dpId, without creating a dedicated
// reduction partition. The current pass tries to schedule them before the
// computation partition exists and dereferences a null fallback partition.

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#load_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#reduce_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_T = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared_reduce = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>

#smem = #ttg.shared_memory
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @merge_reduction_routes_descriptor_reduce
//
// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[COMP:[0-9]+]]>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: ttg.convert_layout {{.*}}ttg.partition = array<i32: [[COMP]]>
// CHECK: tt.descriptor_reduce {{.*}}ttg.partition = array<i32: [[COMP]]>
//
// CHECK: tt.warp_specialize
// CHECK-SAME: ttg.partition.types = ["computation", "load", "gemm"]
// CHECK-NOT: "reduction"
tt.func public @merge_reduction_routes_descriptor_reduce(
  %A_shared: !ttg.memdesc<128x64xf16, #shared, #smem>,
  %B_desc: !tt.tensordesc<64x64xf16, #shared>,
  %D_desc: !tt.tensordesc<128x64xf32, #shared_reduce>,
  %n_tiles: i32
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %zero = arith.constant dense<0.0> : tensor<128x64xf32, #blocked>
  %scale = arith.constant dense<1.0> : tensor<128x64xf32, #blocked>

  scf.for %i = %c0_i32 to %n_tiles step %c64_i32 iter_args(
    %acc = %zero
  ) -> (tensor<128x64xf32, #blocked>) : i32 {
    %B = tt.descriptor_load %B_desc[%i, %c0_i32] : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #load_blocked>
    %B_shared = ttg.local_alloc %B : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %B_trans = ttg.memdesc_trans %B_shared {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared_T, #smem>

    %C_tmem, %C_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %A_shared, %B_trans, %C_tmem[%C_tok], %false, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared_T, #smem>, !ttg.memdesc<128x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>

    %result, %result_tok = ttng.tmem_load %C_tmem[%mma_tok] : !ttg.memdesc<128x64xf32, #tmem_acc, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    %scaled = arith.mulf %result, %scale : tensor<128x64xf32, #blocked>
    %reduced = ttg.convert_layout %scaled : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #reduce_blocked>
    tt.descriptor_reduce add, %D_desc[%i, %c0_i32], %reduced : !tt.tensordesc<128x64xf32, #shared_reduce>, tensor<128x64xf32, #reduce_blocked>

    %new_acc = arith.addf %acc, %result : tensor<128x64xf32, #blocked>
    scf.yield %new_acc : tensor<128x64xf32, #blocked>
  } {tt.warp_specialize}

  tt.return
}

}
