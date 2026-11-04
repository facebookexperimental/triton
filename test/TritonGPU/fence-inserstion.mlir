// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-fence-insertion | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_like_fence
  tt.func public @matmul_like_fence(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked2>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = ttg.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %1 = ttg.local_alloc %arg1 : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    // CHECK: ttng.fence_async_shared
    %2 = ttng.warp_group_dot %0, %1, %cst : !ttg.memdesc<128x128xf16, #shared, #smem> * !ttg.memdesc<128x64xf16, #shared1, #smem> -> tensor<128x64xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_like_fence_local_store
  tt.func public @matmul_like_fence_local_store(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked2>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    ttg.local_store %arg0, %0 : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: ttng.fence_async_shared
    %2 = ttng.warp_group_dot %0, %1, %cst : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_like_fence_mma_v5
  tt.func public @matmul_like_fence_mma_v5(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked2>) {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked1>
    %0 = ttg.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %1 = ttg.local_alloc %arg1 : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    %acc_tm = ttng.tmem_alloc %cst : (tensor<128x64xf32, #blocked1>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>
    // CHECK: ttng.fence_async_shared
    ttng.tc_gen5_mma %0, %1, %acc_tm, %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x64xf16, #shared1, #ttg.shared_memory>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fence_outside_loop
  tt.func public @fence_outside_loop(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %1 = ttg.local_alloc %arg1 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    // CHECK: ttng.fence_async_shared
    // CHECK: scf.for
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK:   ttng.warp_group_dot
    scf.for %iv0 = %c0_i32 to %c64_i32 step %c32_i32 : i32 {
      scf.for %iv1 = %c0_i32 to %c64_i32 step %c32_i32 : i32 {
        %2 = ttng.warp_group_dot %0, %1, %cst : !ttg.memdesc<128x128xf16, #shared, #smem> * !ttg.memdesc<128x64xf16, #shared1, #smem> -> tensor<128x64xf32, #mma>
      }
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fence_store_in_loop
  tt.func public @fence_store_in_loop(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc %arg1 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: scf.for
    // CHECK: ttng.fence_async_shared
    // CHECK: ttng.warp_group_dot
    scf.for %iv0 = %c0_i32 to %c64_i32 step %c32_i32 : i32 {
      scf.for %iv1 = %c0_i32 to %c64_i32 step %c32_i32 : i32 {
        ttg.local_store %arg0, %0 : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %2 = ttng.warp_group_dot %0, %1, %cst : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x64xf16, #shared1, #smem> -> tensor<128x64xf32, #mma>
      }
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: reg_argument
  tt.func public @reg_argument(%arg0: tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg1: tensor<128x64xf16, #blocked>) {
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.warp_group_dot
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    %2 = ttng.warp_group_dot %arg0, %1, %cst : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf32, #mma>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @mma_inside_warp_specialize
tt.func @mma_inside_warp_specialize(%src: tensor<64x64xf16, #blocked>) {
  %A = ttg.local_alloc %src : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
  %B = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %D = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>

  ttg.warp_specialize(%A, %B, %D)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0
  partition0(%lhs: !ttg.memdesc<64x64xf16, #shared, #smem>, %rhs: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: ttng.fence_async_shared
    // CHECK-NEXT: scf.for
    scf.for %i = %c0_i32 to %c32_i32 step %c1_i32 : i32 {
      // CHECK-NEXT: ttng.tc_gen5_mma
      ttng.tc_gen5_mma %lhs, %rhs, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: ttng.tc_gen5_mma
      ttng.tc_gen5_mma %lhs, %rhs, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    }
    ttg.warp_return
  }
  // CHECK: partition1
  partition1(%lhs: !ttg.memdesc<64x64xf16, #shared, #smem>, %rhs: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
    // CHECK-NOT: ttng.fence_async_shared
    %true = arith.constant true
    // CHECK: ttng.tc_gen5_mma
    ttng.tc_gen5_mma %rhs, %rhs, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
  tt.return
}

}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Test that a fence inserted for a TMA store is elided when one already
  // exists earlier in the block, separated only by pure arithmetic.
  // CHECK-LABEL: no_duplicate_fence_tma_store
  tt.func public @no_duplicate_fence_tma_store(
      %desc: !tt.tensordesc<128x32xf16, #shared>,
      %data: tensor<128x32xf16, #linear>,
      %smem: !ttg.memdesc<128x32xf16, #shared, #smem, mutable>,
      %offs_am: i32, %offs_bn: i32) {
    %c32_i32 = arith.constant 32 : i32
    ttg.local_store %data, %smem : tensor<128x32xf16, #linear> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %offs_bn_1 = arith.addi %offs_bn, %c32_i32 : i32
    // CHECK: ttng.fence_async_shared
    // CHECK: arith.addi
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%offs_am, %offs_bn_1] %smem : !tt.tensordesc<128x32xf16, #shared>, !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 4 : i32} {

// The producer partition fences its generic-proxy write before the mbarrier
// arrive that publishes it, so the consumer partition needs no fence.
// CHECK-LABEL: @ws_producer_fenced_before_arrive
tt.func @ws_producer_fenced_before_arrive() {
  %A = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %D = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
  %full = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
  ttng.init_barrier %full, 1 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
  ttg.warp_specialize(%A, %D, %full)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0
  partition0(%a: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %bar: !ttg.memdesc<1xi64, #bar, #smem, mutable>) num_warps(4) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %x = arith.constant dense<1.000000e+00> : tensor<64x64xf16, #blocked>
    // CHECK: ttg.local_store
    // CHECK-NEXT: ttng.fence_async_shared
    // CHECK-NEXT: ttng.arrive_barrier
    scf.for %i = %c0_i32 to %c32_i32 step %c1_i32 : i32 {
      ttg.local_store %x, %a : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      ttng.fence_async_shared {bCluster = false}
      ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    }
    ttg.warp_return
  }
  // CHECK: partition1
  partition1(%a: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %bar: !ttg.memdesc<1xi64, #bar, #smem, mutable>) num_warps(4) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.wait_barrier
    // CHECK-NEXT: ttng.tc_gen5_mma
    scf.for %i = %c0_i32 to %c32_i32 step %c1_i32 : i32 {
      ttng.wait_barrier %bar, %c0_i32 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
      ttng.tc_gen5_mma %a, %a, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    }
    ttg.warp_return
  } : (!ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #bar, #smem, mutable>) -> ()
  tt.return
}

}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 4 : i32} {

// The producer fences only after the arrive, so the consumer keeps its fence.
// CHECK-LABEL: @ws_producer_fenced_after_arrive
tt.func @ws_producer_fenced_after_arrive() {
  %A = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %D = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
  %full = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
  ttng.init_barrier %full, 1 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
  ttg.warp_specialize(%A, %D, %full)
  default {
    ttg.warp_yield
  }
  partition0(%a: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %bar: !ttg.memdesc<1xi64, #bar, #smem, mutable>) num_warps(4) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %x = arith.constant dense<1.000000e+00> : tensor<64x64xf16, #blocked>
    scf.for %i = %c0_i32 to %c32_i32 step %c1_i32 : i32 {
      ttg.local_store %x, %a : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
      ttng.fence_async_shared {bCluster = false}
    }
    ttg.warp_return
  }
  // CHECK: partition1
  partition1(%a: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %bar: !ttg.memdesc<1xi64, #bar, #smem, mutable>) num_warps(4) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: ttng.wait_barrier
    // CHECK-NEXT: ttng.fence_async_shared
    // CHECK-NEXT: ttng.tc_gen5_mma
    scf.for %i = %c0_i32 to %c32_i32 step %c1_i32 : i32 {
      ttng.wait_barrier %bar, %c0_i32 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
      ttng.tc_gen5_mma %a, %a, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    }
    ttg.warp_return
  } : (!ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #bar, #smem, mutable>) -> ()
  tt.return
}

}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 4 : i32} {

// One slice of the MMA's allocation is fenced but another is not, so the
// consumer keeps its fence.
// CHECK-LABEL: @ws_same_alloc_other_slice_unfenced
tt.func @ws_same_alloc_other_slice_unfenced() {
  %A = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
  %D = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
  %full = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
  ttng.init_barrier %full, 1 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
  ttg.warp_specialize(%A, %D, %full)
  default {
    ttg.warp_yield
  }
  partition0(%a: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %bar: !ttg.memdesc<1xi64, #bar, #smem, mutable>) num_warps(4) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %x = arith.constant dense<1.000000e+00> : tensor<64x64xf16, #blocked>
    %a0 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttg.local_store %x, %a0 : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %a1 = ttg.memdesc_index %a[%c1_i32] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttg.local_store %x, %a1 : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttg.warp_return
  }
  // CHECK: partition1
  partition1(%a: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %bar: !ttg.memdesc<1xi64, #bar, #smem, mutable>) num_warps(4) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %a0 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: ttng.wait_barrier
    // CHECK-NEXT: ttng.fence_async_shared
    // CHECK-NEXT: ttng.tc_gen5_mma
    ttng.wait_barrier %bar, %c0_i32 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.tc_gen5_mma %a0, %a0, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #bar, #smem, mutable>) -> ()
  tt.return
}

}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 4 : i32} {

// The unfenced write goes to an allocation no async-proxy op reads, so it
// does not keep the consumer's fence.
// CHECK-LABEL: @ws_unfenced_write_to_generic_only_alloc
tt.func @ws_unfenced_write_to_generic_only_alloc() {
  %A = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %S = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %D = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
  %full = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
  ttng.init_barrier %full, 1 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
  ttg.warp_specialize(%A, %S, %D, %full)
  default {
    ttg.warp_yield
  }
  partition0(%a: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %s: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %bar: !ttg.memdesc<1xi64, #bar, #smem, mutable>) num_warps(4) {
    %x = arith.constant dense<1.000000e+00> : tensor<64x64xf16, #blocked>
    ttg.local_store %x, %a : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttg.local_store %x, %s : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %y = ttg.local_load %s : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #blocked>
    ttg.warp_return
  }
  // CHECK: partition1
  partition1(%a: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %s: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %acc: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %bar: !ttg.memdesc<1xi64, #bar, #smem, mutable>) num_warps(4) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.wait_barrier
    // CHECK-NEXT: ttng.tc_gen5_mma
    ttng.wait_barrier %bar, %c0_i32 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.tc_gen5_mma %a, %a, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #bar, #smem, mutable>) -> ()
  tt.return
}

}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 4 : i32} {

// TMA store and TMA reduce in the consumer partition read a buffer the
// producer fenced before its arrive, so neither gets a fence.
// CHECK-LABEL: @ws_producer_fenced_tma_store_reduce
tt.func @ws_producer_fenced_tma_store_reduce(%desc: !tt.tensordesc<64x64xf16, #shared>) {
  %A = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %full = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
  ttng.init_barrier %full, 1 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
  ttg.warp_specialize(%A, %full, %desc)
  default {
    ttg.warp_yield
  }
  partition0(%a: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #bar, #smem, mutable>, %d: !tt.tensordesc<64x64xf16, #shared>) num_warps(4) {
    %x = arith.constant dense<1.000000e+00> : tensor<64x64xf16, #blocked>
    ttg.local_store %x, %a : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttg.warp_return
  }
  // CHECK: partition1
  partition1(%a: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #bar, #smem, mutable>, %d: !tt.tensordesc<64x64xf16, #shared>) num_warps(4) {
    %c0_i32 = arith.constant 0 : i32
    // CHECK-NOT: ttng.fence_async_shared
    // CHECK: ttng.wait_barrier
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    // CHECK-NEXT: ttng.async_tma_reduce
    ttng.wait_barrier %bar, %c0_i32 : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    ttng.async_tma_copy_local_to_global %d[%c0_i32, %c0_i32] %a : !tt.tensordesc<64x64xf16, #shared>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.async_tma_reduce add, %d[%c0_i32, %c0_i32] %a : !tt.tensordesc<64x64xf16, #shared>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #bar, #smem, mutable>, !tt.tensordesc<64x64xf16, #shared>) -> ()
  tt.return
}

}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 4 : i32} {

// Written through one memdesc_reinterpret alias and fenced, read by the MMA
// through another alias of the same allocation: no compiler fence.
// CHECK-LABEL: @reinterpret_alias_fenced
tt.func @reinterpret_alias_fenced() {
  %true = arith.constant true
  %x = arith.constant dense<1.000000e+00> : tensor<64x64xf16, #blocked>
  %backing = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %w = ttg.memdesc_reinterpret %backing : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %r = ttg.memdesc_reinterpret %backing : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  // CHECK: ttg.local_store
  // CHECK-NEXT: ttng.fence_async_shared
  // CHECK-NEXT: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tc_gen5_mma
  ttg.local_store %x, %w : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  ttng.fence_async_shared {bCluster = false}
  %acc = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.tc_gen5_mma %r, %r, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}

// A second alias of the same allocation is written after the fence, so the
// MMA keeps its fence.
// CHECK-LABEL: @reinterpret_alias_unfenced
tt.func @reinterpret_alias_unfenced() {
  %true = arith.constant true
  %x = arith.constant dense<1.000000e+00> : tensor<64x64xf16, #blocked>
  %backing = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %w = ttg.memdesc_reinterpret %backing : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %v = ttg.memdesc_reinterpret %backing : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  %r = ttg.memdesc_reinterpret %backing : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  ttg.local_store %x, %w : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  ttng.fence_async_shared {bCluster = false}
  ttg.local_store %x, %v : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttng.fence_async_shared
  // CHECK-NEXT: ttng.tc_gen5_mma
  %acc = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.tc_gen5_mma %r, %r, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}

// A ring buffer carried through iter_args whose initial and yielded slots are
// views of the same allocation, all writes fenced: no compiler fence.
// CHECK-LABEL: @loop_carried_ring_fenced
tt.func @loop_carried_ring_fenced() {
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c8 = arith.constant 8 : i32
  %x = arith.constant dense<1.000000e+00> : tensor<64x64xf16, #blocked>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
  %s0 = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  ttg.local_store %x, %s0 : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  // CHECK: ttng.fence_async_shared
  // CHECK-NEXT: ttng.tmem_alloc
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: ttng.tc_gen5_mma
  ttng.fence_async_shared {bCluster = false}
  %acc = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%cur = %s0) -> (!ttg.memdesc<64x64xf16, #shared, #smem, mutable>) : i32 {
    ttng.tc_gen5_mma %cur, %cur, %acc, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %j = arith.remsi %i, %c2 : i32
    %nxt = ttg.memdesc_index %buf[%j] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    scf.yield %nxt : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
  }
  tt.return
}

}
