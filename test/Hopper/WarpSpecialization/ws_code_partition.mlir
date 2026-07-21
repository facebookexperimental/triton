// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-buffer-allocation '--nvgpu-test-ws-memory-planner=num-buffers=1 smem-budget=200000' '--nvgpu-test-ws-code-partition=num-buffers=1' | FileCheck %s

// This exercises the production (post-allocation) code-partition path. Because
// that path consumes post-buffer-allocation
// IR, the raw kernels below are first run through the buffer-allocation and
// memory-planner stages, then code partition. The deleted legacy layernorm
// fixture's combined reduction-epilogue/descriptor-store case is covered by
// ws_code_partition_reduce_epilogue_store.mlir; full-pipeline descriptor-store
// lowering is covered by blackwell_ws_matmul_tma.mlir and
// autows_addmm_warp_specialization.mlir.

// CHECK-LABEL: @matmul_kernel_one_consumer
// CHECK: nvws.create_token
// CHECK: ttg.local_alloc
// CHECK: ttg.warp_specialize
// CHECK: default
// CHECK: scf.for
// CHECK: nvws.producer_acquire
// CHECK: ttg.local_store
// CHECK: nvws.producer_commit
// CHECK: nvws.producer_acquire
// CHECK: ttg.local_store
// CHECK: nvws.producer_commit
// CHECK: partition0
// CHECK: nvws.consumer_wait
// CHECK: ttg.local_load
// CHECK: nvws.consumer_release
// CHECK: nvws.consumer_wait
// CHECK: ttg.local_load
// CHECK: nvws.consumer_release
// CHECK: tt.dot

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_one_consumer(%ptrA: tensor<128x256x!tt.ptr<f16>, #blocked2>, %ptrB: tensor<256x128x!tt.ptr<f16>, #blocked1>, %row: tensor<1x256xi32, #blocked2>, %column: tensor<256x1xi32, #blocked1>, %inc: tensor<256x128xi32, #blocked1>, %store_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %arg5: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant {ttg.partition = array<i32: 1>} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c255_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 255 : i32
    %c127_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 127 : i32
    %c1_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
    %cst_0 = arith.constant {ttg.partition = array<i32: 0, 1>} dense<0.000000e+00> : tensor<256x128xf16, #blocked1>
    %cst_1 = arith.constant {ttg.partition = array<i32: 0, 1>} dense<0.000000e+00> : tensor<128x256xf16, #blocked2>
    %c8_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 8 : i32
    %c128_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 128 : i32
    %c256_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 256 : i32
    %cst_2 = arith.constant {ttg.partition = array<i32: 0, 1>} dense<256> : tensor<128x256xi32, #blocked2>
    %51 = arith.addi %arg5, %c255_i32 {ttg.partition = array<i32: 0, 1>} : i32
    %52 = arith.divsi %51, %c256_i32 {ttg.partition = array<i32: 0, 1>} : i32
    %55:3 = scf.for %arg9 = %c0_i32 to %52 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %ptrA, %arg12 = %ptrB) -> (tensor<128x128xf32, #blocked>, tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<256x128x!tt.ptr<f16>, #blocked1>)  : i32 {
      %74 = arith.muli %arg9, %c256_i32 {ttg.partition = array<i32: 0>} : i32
      %75 = arith.subi %arg5, %74 {ttg.partition = array<i32: 0>} : i32
      %76 = tt.splat %75 {ttg.partition = array<i32: 0>} : i32 -> tensor<1x256xi32, #blocked2>
      %77 = arith.cmpi slt, %row, %76 {ttg.partition = array<i32: 0>} : tensor<1x256xi32, #blocked2>
      %78 = tt.broadcast %77 {ttg.partition = array<i32: 0>} : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
      %79 = tt.load %arg11, %78, %cst_1 {ttg.partition = array<i32: 0>} : tensor<128x256x!tt.ptr<f16>, #blocked2>
      %80 = tt.splat %75 {ttg.partition = array<i32: 0>} : i32 -> tensor<256x1xi32, #blocked1>
      %81 = arith.cmpi slt, %column, %80 {ttg.partition = array<i32: 0>} : tensor<256x1xi32, #blocked1>
      %82 = tt.broadcast %81 {ttg.partition = array<i32: 0>} : tensor<256x1xi1, #blocked1> -> tensor<256x128xi1, #blocked1>
      %83 = tt.load %arg12, %82, %cst_0 {ttg.partition = array<i32: 0>} : tensor<256x128x!tt.ptr<f16>, #blocked1>
      // 2 loads in partition 0
      %84 = ttg.convert_layout %79 {ttg.partition = array<i32: 1>} : tensor<128x256xf16, #blocked2> -> tensor<128x256xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %85 = ttg.convert_layout %83 {ttg.partition = array<i32: 1>} : tensor<256x128xf16, #blocked1> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %86 = tt.dot %84, %85, %arg10, inputPrecision = tf32 {ttg.partition = array<i32: 1>} : tensor<128x256xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<256x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
      %87 = tt.addptr %arg11, %cst_2 {ttg.partition = array<i32: 0>} : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
      %88 = tt.addptr %arg12, %inc {ttg.partition = array<i32: 0>} : tensor<256x128x!tt.ptr<f16>, #blocked1>, tensor<256x128xi32, #blocked1>
      scf.yield {ttg.partition = array<i32: 0, 1>} %86, %87, %88 : tensor<128x128xf32, #blocked>, tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<256x128x!tt.ptr<f16>, #blocked1>
    } {ttg.partition = array<i32: 0, 1>}
    %56 = arith.truncf %55#0 {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %73 = ttg.convert_layout %56 {ttg.partition = array<i32: 1>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked1>
    tt.store %store_ptr, %73 {ttg.partition = array<i32: 1>} : tensor<128x128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: @matmul_kernel_two_consumers
// CHECK: ttg.warp_specialize
// CHECK: default
// CHECK: nvws.producer_acquire
// CHECK: ttg.local_store
// CHECK: nvws.producer_commit
// CHECK: partition0
// CHECK: nvws.consumer_wait
// CHECK: ttng.warp_group_dot
// CHECK: partition1
// CHECK: nvws.consumer_wait
// CHECK: ttng.warp_group_dot

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_two_consumers(%input_ptr1: tensor<64x64x!tt.ptr<f16>, #blocked>, %input_ptr2: tensor<64x128x!tt.ptr<f16>, #blocked1>, %input_ptr3: tensor<64x64x!tt.ptr<f16>, #blocked>, %row: tensor<1x64xi32, #blocked>, %column: tensor<64x1xi32, #blocked1>, %inc: tensor<64x128xi32, #blocked1>, %store_ptr1: tensor<64x128x!tt.ptr<f16>, #blocked1>, %store_ptr2: tensor<64x128x!tt.ptr<f16>, #blocked1>, %arg5: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant {ttg.partition = array<i32: 0>} dense<64> : tensor<64x64xi32, #blocked>
    %c64_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 64 : i32
    %c128_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 128 : i32
    %c8_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 8 : i32
    %cst_0 = arith.constant {ttg.partition = array<i32: 0>} dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    %cst_1 = arith.constant {ttg.partition = array<i32: 0>} dense<0.000000e+00> : tensor<64x128xf16, #blocked1>
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 1 : i32
    %c127_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 127 : i32
    %c63_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 63 : i32
    %cst_2 = arith.constant {ttg.partition = array<i32: 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %58 = arith.addi %arg5, %c63_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
    %59 = arith.divsi %58, %c64_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
    %64:5 = scf.for %arg9 = %c0_i32 to %59 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %cst_2, %arg12 = %input_ptr1, %arg13 = %input_ptr2, %arg14 = %input_ptr3) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x64x!tt.ptr<f16>, #blocked>)  : i32 {
      %93 = arith.muli %arg9, %c64_i32 {ttg.partition = array<i32: 0>} : i32
      %94 = arith.subi %arg5, %93 {ttg.partition = array<i32: 0>} : i32
      %95 = tt.splat %94 {ttg.partition = array<i32: 0>} : i32 -> tensor<1x64xi32, #blocked>
      %96 = arith.cmpi slt, %row, %95 {ttg.partition = array<i32: 0>} : tensor<1x64xi32, #blocked>
      %97 = tt.broadcast %96 {ttg.partition = array<i32: 0>} : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
      %98 = tt.load %arg12, %97, %cst_0 {ttg.partition = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %99 = ttg.local_alloc %98 {ttg.partition = array<i32: 1>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %100 = tt.splat %94 {ttg.partition = array<i32: 0>} : i32 -> tensor<64x1xi32, #blocked1>
      %101 = arith.cmpi slt, %column, %100 {ttg.partition = array<i32: 0>} : tensor<64x1xi32, #blocked1>
      %102 = tt.broadcast %101 {ttg.partition = array<i32: 0>} : tensor<64x1xi1, #blocked1> -> tensor<64x128xi1, #blocked1>
      %103 = tt.load %arg13, %102, %cst_1 {ttg.partition = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %104 = ttg.local_alloc %103 {ttg.partition = array<i32: 1, 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory>
      %105 = tt.load %arg14, %97, %cst_0 {ttg.partition = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %106 = ttg.local_alloc %105 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %107 = ttng.warp_group_dot %99, %104, %arg10 {ttg.partition = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      %108 = ttng.warp_group_dot %106, %104, %arg11 {ttg.partition = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      %109 = tt.addptr %arg12, %cst {ttg.partition = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %110 = tt.addptr %arg14, %cst {ttg.partition = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %111 = tt.addptr %arg13, %inc {ttg.partition = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %107, %108, %109, %111, %110 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x64x!tt.ptr<f16>, #blocked>
    } {ttg.partition = array<i32: 0, 1, 2>}
    %65 = arith.truncf %64#0 {ttg.partition = array<i32: 1>} : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
    %66 = arith.truncf %64#1 {ttg.partition = array<i32: 2>} : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
    %91 = ttg.convert_layout %65 {ttg.partition = array<i32: 1>} : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked1>
    tt.store %store_ptr1, %91 {ttg.partition = array<i32: 1>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
    %92 = ttg.convert_layout %66 {ttg.partition = array<i32: 2>} : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked1>
    tt.store %store_ptr2, %92 {ttg.partition = array<i32: 2>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-DAG: #[[$SHARED:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
// CHECK-DAG: #[[$SHARED1:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
// CHECK-LABEL: @_fbgemm_grouped_gemm_fp8_rowwise_ws
// CHECK-DAG: ttg.local_alloc {{.*}} : () -> !ttg.memdesc<1x64x64xf8E4M3FN, #[[$SHARED1]], #smem, mutable>
// CHECK-DAG: ttg.local_alloc {{.*}} : () -> !ttg.memdesc<1x128x64xf8E4M3FN, #[[$SHARED1]], #smem, mutable>
// CHECK-DAG: ttg.local_alloc {{.*}} : () -> !ttg.memdesc<1x128xf32, #[[$SHARED]], #smem, mutable>
// CHECK: ttg.warp_specialize
// CHECK: partition0
// CHECK: ttg.memdesc_trans
// CHECK: ttng.warp_group_dot
// CHECK: partition1
// CHECK: ttg.local_load

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_fbgemm_grouped_gemm_fp8_rowwise_ws(%arg0: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg1: i32, %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg3: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}) {
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 0 : i32
    %c2048_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 2048 : i32
    %c64_i32 = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 64 : i32
    %cst = arith.constant {ttg.partition = array<i32: 0, 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %0 = tt.get_program_id x {ttg.partition = array<i32: 0, 1, 2>} : i32
    %1 = ttng.reinterpret_tensor_descriptor %arg0 {ttg.partition = array<i32: 0>} : !tt.ptr<i8, 0> to !tt.tensordesc<64x64xf8E4M3FN, #shared>
    %2 = ttng.reinterpret_tensor_descriptor %arg2 {ttg.partition = array<i32: 0>} : !tt.ptr<i8, 0> to !tt.tensordesc<128x64xf8E4M3FN, #shared>
    %3 = ttng.reinterpret_tensor_descriptor %arg3 {ttg.partition = array<i32: 0>} : !tt.ptr<i8, 0> to !tt.tensordesc<128xf32, #shared1>
    scf.for %arg4 = %0 to %arg1 step %c64_i32  : i32 {
      %4 = arith.muli %arg4, %c2048_i32 {ttg.partition = array<i32: 0>} : i32
      %5 = scf.for %arg5 = %c0_i32 to %c2048_i32 step %c64_i32 iter_args(%arg6 = %cst) -> (tensor<64x128xf32, #mma>)  : i32 {
        %8 = tt.descriptor_load %1[%4, %arg5] {ttg.partition = array<i32: 0>} : !tt.tensordesc<64x64xf8E4M3FN, #shared> -> tensor<64x64xf8E4M3FN, #blocked>
        %9 = ttg.local_alloc %8 {ttg.partition = array<i32: 1>} : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>
        %10 = tt.descriptor_load %2[%4, %arg5] {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf8E4M3FN, #shared> -> tensor<128x64xf8E4M3FN, #blocked>
        %11 = ttg.local_alloc %10 {ttg.partition = array<i32: 1, 2>} : (tensor<128x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem>
        %12 = ttg.memdesc_trans %11 {ttg.partition = array<i32: 1, 2>, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #shared2, #smem>
        %13 = ttng.warp_group_dot %9, %12, %arg6 {ttg.partition = array<i32: 1>, inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem> * !ttg.memdesc<64x128xf8E4M3FN, #shared2, #smem> -> tensor<64x128xf32, #mma>
        scf.yield {ttg.partition = array<i32: 1, 2>} %13 : tensor<64x128xf32, #mma>
      } {ttg.partition = array<i32: 0, 1, 2>}
      %6 = tt.descriptor_load %3[%4] {ttg.partition = array<i32: 0>} : !tt.tensordesc<128xf32, #shared1> -> tensor<128xf32, #blocked1>
      %7 = ttg.convert_layout %6 {ttg.partition = array<i32: 1, 2>} : tensor<128xf32, #blocked1> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    } {ttg.partition = array<i32: 1, 2>}
    tt.return
  }
}
