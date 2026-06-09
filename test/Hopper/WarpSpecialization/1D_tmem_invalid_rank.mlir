// RUN: ! triton-opt %s -split-input-file --nvgpu-test-1D-tmem-alloc 2>&1 | FileCheck %s --check-prefix=ERR

// Regression test for B-8-F1 / T273479230.
// ERR: expected `tmem.start` producer to have rank 1

#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @invalid_2d_tmem_start(%arg0: tensor<128x1xf32, #blocked2>) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #blocked2>
    %producer = arith.addf %arg0, %cst {tmem.start = 0 : i32, async_task_id = array<i32: 0>} : tensor<128x1xf32, #blocked2>
    %consumer = arith.addf %producer, %cst {async_task_id = array<i32: 1>} : tensor<128x1xf32, #blocked2>
    tt.return
  }
}
