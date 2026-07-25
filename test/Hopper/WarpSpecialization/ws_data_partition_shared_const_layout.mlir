// RUN: triton-opt %s --nvgpu-ws-data-partition=num-warp-groups=1 | FileCheck %s

// Regression test distilled from a persistent gate_mm_sigmoid2x kernel
// (P2418648199) with tt.data_partition_factor = 2 (M: 256 -> 128).
//
// The accumulator's distributed layout is a shape-dependent #linear layout.
// The 0.0 constant is shared between the accumulator-init tmem_store (which
// needs a TMEM-compatible layout, #linear1) and the downstream elementwise
// chain (arith.subf, which keeps the original #linear). When sliceOp converted
// the tmem_store source to #linear1 it remapped the *shared* constant value in
// the running IRMapping, so every later consumer -- including the subf --
// picked up the #linear1 value while the subf itself was typed #linear. That
// produced arith.subf with mismatched operand encodings:
//   'arith.subf' op requires the same encoding for all operands and results
//
// The fix keeps the tmem_store's layout conversion local: the shared constant
// must stay #linear for the elementwise chain, and only the store gets the
// #linear1 convert.

// CHECK-LABEL: @data_partition_shared_const
// The shared 0.0 constant stays in the original (sliced) #linear layout.
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #linear>
// Each partition's tmem_store gets its own *local* convert to the
// TMEM-compatible #linear1 layout, sourced from the original constant.
// CHECK: ttg.convert_layout %[[CST]] : tensor<128x64xf32, #linear> -> tensor<128x64xf32, #linear1>
// CHECK: ttng.tmem_store
// CHECK: ttg.convert_layout %[[CST]] : tensor<128x64xf32, #linear> -> tensor<128x64xf32, #linear1>
// CHECK: ttng.tmem_store
// Both partitioned subf ops consume the constant in the ORIGINAL #linear layout
// (not the tmem-compatible #linear1), so operand/result encodings agree.
// CHECK: arith.subf %[[CST]], %{{.*}} : tensor<128x64xf32, #linear>
// CHECK: arith.subf %[[CST]], %{{.*}} : tensor<128x64xf32, #linear>

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @data_partition_shared_const(%x_mm_desc: !tt.tensordesc<tensor<256x64xbf16, #shared>>, %w_desc: !tt.tensordesc<tensor<64x64xbf16, #shared>>, %out_desc: !tt.tensordesc<tensor<256x64xbf16, #shared>>, %M: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c4_i32 = arith.constant 4 : i32
    %c64_i32 = arith.constant 64 : i32
    %c152_i32 = arith.constant 152 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #linear>
    %start_pid = tt.get_program_id x : i32
    scf.for %tile_id = %start_pid to %M step %c152_i32  : i32 {
      %offs_k = arith.muli %tile_id, %c64_i32 : i32
      %accumulator, %accumulator_7 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %accumulator_8 = ttng.tmem_store %cst_0, %accumulator[%accumulator_7], %true : tensor<256x64xf32, #linear> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %x = tt.descriptor_load %x_mm_desc[%offs_k, %offs_k] : !tt.tensordesc<tensor<256x64xbf16, #shared>> -> tensor<256x64xbf16, #blocked>
      %x_20 = ttg.local_alloc %x : (tensor<256x64xbf16, #blocked>) -> !ttg.memdesc<256x64xbf16, #shared, #smem>
      %w_nk = tt.descriptor_load %w_desc[%offs_k, %offs_k] : !tt.tensordesc<tensor<64x64xbf16, #shared>> -> tensor<64x64xbf16, #blocked>
      %w_21 = ttg.local_alloc %w_nk : (tensor<64x64xbf16, #blocked>) -> !ttg.memdesc<64x64xbf16, #shared, #smem>
      %w_22 = ttg.memdesc_trans %w_21 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared, #smem> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
      %accumulator_23 = ttng.tc_gen5_mma %x_20, %w_22, %accumulator[%accumulator_8], %true, %true : !ttg.memdesc<256x64xbf16, #shared, #smem>, !ttg.memdesc<64x64xbf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %accumulator_10, %accumulator_11 = ttng.tmem_load %accumulator[%accumulator_23] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #linear>
      %gate = arith.subf %cst_0, %accumulator_10 : tensor<256x64xf32, #linear>
      %gate_16 = ttg.convert_layout %gate : tensor<256x64xf32, #linear> -> tensor<256x64xf32, #blocked>
      %0 = arith.truncf %gate_16 : tensor<256x64xf32, #blocked> to tensor<256x64xbf16, #blocked>
      tt.descriptor_store %out_desc[%offs_k, %offs_k], %0 : !tt.tensordesc<tensor<256x64xbf16, #shared>>, tensor<256x64xbf16, #blocked>
    } {tt.data_partition_factor = 2 : i32, tt.separate_epilogue_store = true, tt.smem_alloc_algo = 1 : i32, tt.warp_specialize}
    tt.return
  }
}
