// RUN: triton-opt %s --nvgpu-partition-scheduling-meta | FileCheck %s

// A non-persistent GEMM whose epilogue reads the accumulator after the
// warp-specialized K loop and writes it with a pointer store (no TMA store).
// The post-loop tmem_load needs a partition of its own: left unscheduled it
// lands in the gemm task and code partitioning asserts in handleOperandD
// ("Unexpected Producer Found").

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @post_loop_tmem_load_pointer_store
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[GEMM:[0-9]+]]>
// CHECK: tt.warp_specialize
// CHECK-SAME: ttg.partition.types = ["epilogue", "gemm", "load"]
// CHECK: ttng.tmem_load
// CHECK-SAME: ttg.partition = array<i32: [[EPIL:[0-9]+]]>
// CHECK: tt.store
// CHECK-SAME: ttg.partition = array<i32: [[EPIL]]>
tt.func public @post_loop_tmem_load_pointer_store(
  %A_desc: !tt.tensordesc<128x64xf16, #shared>,
  %B_desc: !tt.tensordesc<128x64xf16, #shared>,
  %C_ptrs: tensor<128x128x!tt.ptr<f32>, #blocked>,
  %k_tiles: i32
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i32 = arith.constant 1 : i32
  %acc_mem, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
  %loop_out:2 = scf.for %i = %c0_i32 to %k_tiles step %c1_i32
      iter_args(%use_acc = %false, %loop_tok = %acc_tok) -> (i1, !ttg.async.token) : i32 {
    %offs_k = arith.muli %i, %c64_i32 : i32
    %a = tt.descriptor_load %A_desc[%c0_i32, %offs_k] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
    %a_smem = ttg.local_alloc %a : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b = tt.descriptor_load %B_desc[%c0_i32, %offs_k] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
    %b_smem = ttg.local_alloc %b : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_trans = ttg.memdesc_trans %b_smem {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
    %mma_tok = ttng.tc_gen5_mma %a_smem, %b_trans, %acc_mem[%loop_tok], %use_acc, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.yield %true, %mma_tok : i1, !ttg.async.token
  } {tt.warp_specialize}
  %result, %result_tok = ttng.tmem_load %acc_mem[%loop_out#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
  tt.store %C_ptrs, %result : tensor<128x128x!tt.ptr<f32>, #blocked>
  tt.return
}

}
