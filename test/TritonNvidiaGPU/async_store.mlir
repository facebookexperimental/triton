// RUN: triton-opt --split-input-file %s | FileCheck %s
// RUN: triton-opt --split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm=compute-capability=90 --convert-nv-gpu-to-llvm %s | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: triton-opt --split-input-file --triton-nvidia-gpu-plan-cta --mlir-print-local-scope %s | FileCheck %s --check-prefix=CHECK-CTA

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: async_store
  // CHECK-LLVM-LABEL: llvm.func @async_store
  tt.func @async_store(%dst: !tt.ptr<i8>, %size: i32) {
    %src = ttg.local_alloc : () -> !ttg.memdesc<1024xi8, #shared, #smem, mutable>
    // CHECK: ttng.async_store
    // CHECK-SAME: !ttg.memdesc<1024xi8, #shared, #smem, mutable>, !tt.ptr<i8>
    // CHECK-LLVM: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-LLVM-SAME: cp.async.bulk.global.shared::cta.bulk_group
    // CHECK-LLVM: nvvm.cp.async.bulk.commit.group
    ttng.async_store %src, %dst, %size : !ttg.memdesc<1024xi8, #shared, #smem, mutable>, !tt.ptr<i8>
    tt.return
  }
}

// -----

// Test async_store with data originating from a register layout (blocked).
// tl.arange creates a blocked layout in registers; local_alloc writes it to SMEM;
// async_store bulk-copies from SMEM to global memory.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem1 = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: async_store_from_registers
  // CHECK-LLVM-LABEL: llvm.func @async_store_from_registers
  tt.func @async_store_from_registers(%dst: !tt.ptr<f32>) {
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %data = arith.sitofp %range : tensor<128xi32, #blocked> to tensor<128xf32, #blocked>
    %smem = ttg.local_alloc %data : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared1, #smem1, mutable>
    %size = arith.constant 512 : i32
    // CHECK: ttng.async_store
    // CHECK-SAME: !ttg.memdesc<128xf32, #{{.*}}, #{{.*}}, mutable>, !tt.ptr<f32>
    // CHECK-LLVM: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-LLVM-SAME: cp.async.bulk.global.shared::cta.bulk_group
    // CHECK-LLVM: nvvm.cp.async.bulk.commit.group
    ttng.async_store %smem, %dst, %size : !ttg.memdesc<128xf32, #shared1, #smem1, mutable>, !tt.ptr<f32>
    tt.return
  }
}

// -----

// Test PlanCTA with tt.store inside a warp_specialize partition with 1 warp.
// PlanCTA must use per-op numWarps (1 for partition0), not function-level
// numWarps (4). Without the fix, the store layout would get warpsPerCTA=[4],
// which is incorrect for a 1-warp partition.

#blocked2 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>
#blocked_ws = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CGALayout = [[1]]}>


module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-CTA-LABEL: store_ws_plan_cta
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
      // CHECK-CTA: partition0
      // CHECK-CTA: tt.store {{.*}} warpsPerCTA = [1]
      tt.store %ptrs, %data : tensor<512x!tt.ptr<f32>, #blocked_ws>
      ttg.warp_return
    } : (!tt.ptr<f32>) -> ()
    tt.return
  }
}
