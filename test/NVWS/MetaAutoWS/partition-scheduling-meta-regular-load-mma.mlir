// RUN: triton-opt %s --nvws-partition-scheduling-meta -allow-unregistered-dialect | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-hoist-tmem-alloc -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-automatic-warp-specialization="num-stages=2 use-meta-partitioner=true smem-budget=232448" | FileCheck %s --check-prefix=AUTO

#indices_layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#b_layout = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @regular_load_alloc_mma_operand
  // CHECK: %[[B:.*]] = tt.load {{.*}}ttg.partition = array<i32: 3>
  // CHECK: %[[B_SMEM:.*]] = ttg.local_alloc %[[B]] {{.*}}ttg.partition = array<i32: 3>
  // CHECK: ttng.tc_gen5_mma {{.*}}%[[B_SMEM]], {{.*}}ttg.partition = array<i32: 0>
  // CHECK: ttng.tc_gen5_mma {{.*}}%[[B_SMEM]], {{.*}}ttg.partition = array<i32: 0>
  // CHECK: ttg.partition.types = ["gemm", "load", "computation", "computation"]
  // AUTO-LABEL: @regular_load_alloc_mma_operand
  // AUTO: ttg.warp_specialize
  // AUTO: ttng.tc_gen5_mma
  // AUTO: tt.load
  // AUTO: ttg.local_store
  tt.func @regular_load_alloc_mma_operand(%a_desc: !tt.tensordesc<tensor<1x64xf16, #shared>>, %b_ptr_init: tensor<64x128x!tt.ptr<f16>, #b_layout> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %k_tiles = arith.constant 32 : i32
    %acc0, %tok0 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc1, %tok1 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%flag = %false, %b_ptr = %b_ptr_init, %tok_arg0 = %tok0, %tok_arg1 = %tok1) -> (i1, tensor<64x128x!tt.ptr<f16>, #b_layout>, !ttg.async.token, !ttg.async.token) : i32 {
      %off_m, %offs_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, tensor<64x128xi32, #b_layout>, i32)
      %indices0 = tt.splat %off_m : i32 -> tensor<64xi32, #indices_layout>
      %indices1 = tt.splat %off_m : i32 -> tensor<64xi32, #indices_layout>
      %a0 = tt.descriptor_gather %a_desc[%indices0, %off_k] : (!tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<64xi32, #indices_layout>, i32) -> tensor<64x64xf16, #oper_layout>
      %a1 = tt.descriptor_gather %a_desc[%indices1, %off_k] : (!tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<64xi32, #indices_layout>, i32) -> tensor<64x64xf16, #oper_layout>
      %b_ptrs = tt.addptr %b_ptr, %offs_n {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>} : tensor<64x128x!tt.ptr<f16>, #b_layout>, tensor<64x128xi32, #b_layout>
      %b = tt.load %b_ptrs : tensor<64x128x!tt.ptr<f16>, #b_layout>
      %a0_smem = ttg.local_alloc %a0 : (tensor<64x64xf16, #oper_layout>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %a1_smem = ttg.local_alloc %a1 : (tensor<64x64xf16, #oper_layout>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %b_smem = ttg.local_alloc %b : (tensor<64x128xf16, #b_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %mma0 = ttng.tc_gen5_mma %a0_smem, %b_smem, %acc0[%tok_arg0], %flag, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
      %mma1 = ttng.tc_gen5_mma %a1_smem, %b_smem, %acc1[%tok_arg1], %flag, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<64x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
      scf.yield %false, %b_ptrs, %mma0, %mma1 : i1, tensor<64x128x!tt.ptr<f16>, #b_layout>, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32}
    tt.return
  }
}
