// RUN: not triton-opt %s -split-input-file -tritongpu-assign-latencies='num-stages=3 use-meta-ws=true' 2>&1 | FileCheck %s

#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK: error: lhs_buffer_depth and rhs_buffer_depth must be specified together
  tt.func @missing_rhs(%ub: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %k = %c0 to %ub step %c1 : i32 {
      scf.yield
    } {tt.lhs_buffer_depth = 3 : i32, tt.num_stages = 3 : i32, tt.warp_specialize}
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK: error: operand buffer depth 4 exceeds num_stages=3
  tt.func @depth_exceeds_stages(%ub: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %k = %c0 to %ub step %c1 : i32 {
      scf.yield
    } {tt.lhs_buffer_depth = 4 : i32, tt.num_stages = 3 : i32, tt.rhs_buffer_depth = 2 : i32, tt.warp_specialize}
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK: error: operand buffer depths require exactly one MMA in the annotated loop; found 0
  tt.func @no_mma(%ub: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %k = %c0 to %ub step %c1 : i32 {
      scf.yield
    } {tt.lhs_buffer_depth = 3 : i32, tt.num_stages = 3 : i32, tt.rhs_buffer_depth = 2 : i32, tt.warp_specialize}
    tt.return
  }
}
