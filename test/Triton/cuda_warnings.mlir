// Test CudaWarningsPass with different compute capabilities
// Only SM103 (GB300) should emit FP64 math warnings

// RUN: triton-opt %s -split-input-file --test-cuda-warnings="compute-capability=103" 2>&1 | FileCheck %s --check-prefix=CHECK-SM103
// RUN: triton-opt %s -split-input-file --test-cuda-warnings="compute-capability=100" 2>&1 | FileCheck %s --check-prefix=CHECK-SM100 --allow-empty
// RUN: triton-opt %s -split-input-file --test-cuda-warnings="compute-capability=90" 2>&1 | FileCheck %s --check-prefix=CHECK-SM90 --allow-empty

// CHECK-SM103-DAG: warning: PERFORMANCE WARNING: fp64_add contains FP64 (double-precision) math operations on a GB300 GPU
// CHECK-SM103-DAG: warning: PERFORMANCE WARNING: fp64_mul contains FP64 (double-precision) math operations on a GB300 GPU
// CHECK-SM103-DAG: warning: PERFORMANCE WARNING: fp64_div contains FP64 (double-precision) math operations on a GB300 GPU
// CHECK-SM103-NOT: warning: PERFORMANCE WARNING: fp32_add
// CHECK-SM103-NOT: warning: PERFORMANCE WARNING: fp64_load_store
// CHECK-SM100-NOT: warning: PERFORMANCE WARNING
// CHECK-SM90-NOT: warning: PERFORMANCE WARNING

// -----

// Test: FP64 addition should warn on SM103 only

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:103"} {
  tt.func @fp64_add(%arg0: tensor<256xf64, #blocked>, %arg1: tensor<256xf64, #blocked>) -> tensor<256xf64, #blocked> {
    %0 = arith.addf %arg0, %arg1 : tensor<256xf64, #blocked>
    tt.return %0 : tensor<256xf64, #blocked>
  }
}

// -----

// Test: FP64 multiplication should warn on SM103 only

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:103"} {
  tt.func @fp64_mul(%arg0: tensor<256xf64, #blocked>, %arg1: tensor<256xf64, #blocked>) -> tensor<256xf64, #blocked> {
    %0 = arith.mulf %arg0, %arg1 : tensor<256xf64, #blocked>
    tt.return %0 : tensor<256xf64, #blocked>
  }
}

// -----

// Test: FP64 division should warn on SM103 only

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:103"} {
  tt.func @fp64_div(%arg0: tensor<256xf64, #blocked>, %arg1: tensor<256xf64, #blocked>) -> tensor<256xf64, #blocked> {
    %0 = arith.divf %arg0, %arg1 : tensor<256xf64, #blocked>
    tt.return %0 : tensor<256xf64, #blocked>
  }
}

// -----

// Test: FP32 operations should NEVER trigger a warning on any architecture

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:103"} {
  tt.func @fp32_add(%arg0: tensor<256xf32, #blocked>, %arg1: tensor<256xf32, #blocked>) -> tensor<256xf32, #blocked> {
    %0 = arith.addf %arg0, %arg1 : tensor<256xf32, #blocked>
    tt.return %0 : tensor<256xf32, #blocked>
  }
}

// -----

// Test: FP64 load/store should NEVER trigger a warning (only math ops should warn)

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:103"} {
  tt.func @fp64_load_store(%ptr: tensor<256x!tt.ptr<f64>, #blocked>, %val: tensor<256xf64, #blocked>) {
    %0 = tt.load %ptr : tensor<256x!tt.ptr<f64>, #blocked>
    tt.store %ptr, %val : tensor<256x!tt.ptr<f64>, #blocked>
    tt.return
  }
}
