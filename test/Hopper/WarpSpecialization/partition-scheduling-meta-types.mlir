// RUN: triton-opt %s --nvgpu-partition-scheduling-meta -allow-unregistered-dialect | FileCheck %s

// Tests that partition scheduling Meta pass serializes partition types as ttg.partition.types attribute.
// For bwd FA (hasReduction): reduction at index 0, then gemm, load, computation

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#load_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_T = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>

#smem = #ttg.shared_memory
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// Test: Verify partition types attribute is serialized and all tensor ops get partition IDs
// CHECK-LABEL: @simple_gemm_partition_types
//
// --- In-loop: descriptor_load and local_alloc → load partition ---
// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: [[LOAD:[0-9]+]]>
// CHECK: ttg.local_alloc {{.*}}ttg.partition = array<i32: [[LOAD]]>
// --- In-loop: memdesc_trans and MMA → gemm partition ---
// CHECK: ttg.memdesc_trans {{.*}}ttg.partition = array<i32: [[GEMM:[0-9]+]]>
// CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: [[GEMM]]>
// --- In-loop: tmem_load and addf → computation partition ---
// CHECK: ttng.tmem_load {{.*}}ttg.partition = array<i32: [[COMP:[0-9]+]]>
// CHECK: arith.addf {{.*}}ttg.partition = array<i32: [[COMP]]>
//
// --- Partition types ---
// CHECK: tt.warp_specialize
// CHECK-SAME: ttg.partition.types = ["gemm", "load", "computation"]
//
// --- Post-loop: use → no partition annotation (unregistered dialect op) ---
tt.func public @simple_gemm_partition_types(
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
    // Load B
    %B = tt.descriptor_load %B_desc[%i, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #load_blocked>
    %B_shared = ttg.local_alloc %B : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %B_trans = ttg.memdesc_trans %B_shared {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared_T, #smem>

    // MMA operation
    %C_tmem, %C_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %A_shared, %B_trans, %C_tmem[%C_tok], %false, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared_T, #smem>, !ttg.memdesc<128x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>

    %result, %result_tok = ttng.tmem_load %C_tmem[%mma_tok] : !ttg.memdesc<128x64xf32, #tmem_acc, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    %new_acc = arith.addf %acc, %result : tensor<128x64xf32, #blocked>

    scf.yield %new_acc : tensor<128x64xf32, #blocked>
  } {tt.warp_specialize}

  "use"(%loop_out) : (tensor<128x64xf32, #blocked>) -> ()
  tt.return
}

}
