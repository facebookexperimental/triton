// RUN: triton-opt -split-input-file --triton-nvidia-optimize-descriptor-encoding %s | FileCheck %s

// Test that encoding propagates from ReinterpretTensorDescOp back to MakeTensorDescOp
// when they share the same descPtr pointer.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-DAG: #[[SHARED:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
tt.func public @reinterpret_propagate_to_make_desc(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %true = arith.constant true

  // Allocate a pointer for the TMA descriptor
  %desc_ptr = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>

  // Create TMA descriptor and write to desc_ptr
  %0 = arith.extsi %arg2 : i32 to i64
  // CHECK: tt.make_tensor_descriptor {{.*}} descPtr = {{.*}} : !tt.ptr<i8> : !tt.ptr<i8>, !tt.tensordesc<tensor<128x64xi8, #[[SHARED]]>>
  %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64], descPtr = %desc_ptr : !tt.ptr<i8> : !tt.ptr<i8>, !tt.tensordesc<tensor<128x64xi8>>

  // Fence and reinterpret the pointer as a tensor descriptor
  ttng.tensormap_fenceproxy_acquire %desc_ptr : !tt.ptr<i8>
  // CHECK: ttng.reinterpret_tensor_descriptor {{.*}} : !tt.ptr<i8> to !tt.tensordesc<tensor<128x64xi8, #[[SHARED]]>>
  %2 = ttng.reinterpret_tensor_descriptor %desc_ptr : !tt.ptr<i8> to !tt.tensordesc<tensor<128x64xi8>>

  // Allocate shared memory buffer and barrier
  %buf = ttg.local_alloc : () -> !ttg.memdesc<128x64xi8, #shared, #smem, mutable>
  %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
  ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

  // Use ReinterpretTensorDescOp result with AsyncTMACopyGlobalToLocalOp
  // This should propagate the #shared encoding back to MakeTensorDescOp
  // CHECK: ttng.async_tma_copy_global_to_local {{.*}} : !tt.tensordesc<tensor<128x64xi8, #[[SHARED]]>>
  ttng.async_tma_copy_global_to_local %2[%c0_i32, %c0_i32] %buf, %bar, %true : !tt.tensordesc<tensor<128x64xi8>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x64xi8, #shared, #smem, mutable>

  tt.return
}
}
