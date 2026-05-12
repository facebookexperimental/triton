// RUN: triton-opt --split-input-file --triton-nvidia-gpu-plan-cta --mlir-print-local-scope %s | FileCheck %s

// Test PlanCTA with tt.store inside a warp_specialize partition with 1 warp.
// PlanCTA must use per-op numWarps (1 for partition0), not function-level
// numWarps (4). Without the fix, the store layout would get warpsPerCTA=[4],
// which is incorrect for a 1-warp partition.

#blocked2 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [2], CTASplitNum = [2], CTAOrder = [0]}>
#blocked_ws = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [2], CTASplitNum = [2], CTAOrder = [0]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: store_ws_plan_cta
  tt.func @store_ws_plan_cta(%ptr: !tt.ptr<f32>) {
    ttg.warp_specialize(%ptr)
    default {
      // Default partition (4 warps): store with warpsPerCTA=[4]
      %range = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #blocked2>
      %data = arith.sitofp %range : tensor<512xi32, #blocked2> to tensor<512xf32, #blocked2>
      %splatted = tt.splat %ptr : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked2>
      %ptrs = tt.addptr %splatted, %range : tensor<512x!tt.ptr<f32>, #blocked2>, tensor<512xi32, #blocked2>
      tt.store %ptrs, %data : tensor<512x!tt.ptr<f32>, #blocked2>
      ttg.warp_yield
    }
    partition0(%arg0: !tt.ptr<f32>) num_warps(1) {
      // Store partition (1 warp): store must keep warpsPerCTA=[1]
      %range = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #blocked_ws>
      %data = arith.sitofp %range : tensor<512xi32, #blocked_ws> to tensor<512xf32, #blocked_ws>
      %splatted = tt.splat %arg0 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked_ws>
      %ptrs = tt.addptr %splatted, %range : tensor<512x!tt.ptr<f32>, #blocked_ws>, tensor<512xi32, #blocked_ws>
      // CHECK: partition0
      // CHECK: tt.store {{.*}} warpsPerCTA = [1]
      tt.store %ptrs, %data : tensor<512x!tt.ptr<f32>, #blocked_ws>
      ttg.warp_return
    } : (!tt.ptr<f32>) -> ()
    tt.return
  }
}
