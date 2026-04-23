// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-test-generate-subtiled-region --triton-nvidia-optimize-tmem-layouts | FileCheck %s

// Test: multi-task chain — the split in the first SubtiledRegionOp's setup
// region is also converted to tmem_subslice + tmem_load.

#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#blocked3d2 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3d_perm2 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked_full2 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2d2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem2 = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @multi_task_setup_tmem_split_optimized
  // After optimize_tmem_layouts (which now also pushes setup to tile),
  // the setup has only tmem_subslice ops and the tile body has the
  // tmem_load + compute chain:
  // CHECK: ttng.subtiled_region
  // CHECK:   setup {
  // CHECK:     ttng.tmem_subslice
  // CHECK:     ttng.tmem_subslice
  // CHECK-NOT: tt.split
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   } tile{
  // CHECK:     ttng.tmem_load
  // CHECK:     ttg.convert_layout
  // CHECK:     arith.truncf
  // CHECK:     ttg.local_store
  // CHECK:     ttng.subtiled_region_yield
  // CHECK:   }
  tt.func @multi_task_setup_tmem_split_optimized(
      %tmem_buf: !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable>,
      %acc_tok: !ttg.async.token,
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared2>>,
      %off0: i32, %off1: i32, %off2: i32) {
    %loaded:2 = ttng.tmem_load %tmem_buf[%acc_tok] : !ttg.memdesc<128x128xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_full2>
    %reshaped = tt.reshape %loaded#0 : tensor<128x128xf32, #blocked_full2> -> tensor<128x2x64xf32, #blocked3d2>
    %transposed = tt.trans %reshaped {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3d2> -> tensor<128x64x2xf32, #blocked3d_perm2>
    %lhs, %rhs = tt.split %transposed : tensor<128x64x2xf32, #blocked3d_perm2> -> tensor<128x64xf32, #blocked2d2>

    %trunc0 = arith.truncf %lhs {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked2d2> to tensor<128x64xf16, #blocked2d2>
    %smem0 = ttg.local_alloc %trunc0 {async_task_id = array<i32: 3>} : (tensor<128x64xf16, #blocked2d2>) -> !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>
    ttng.async_tma_copy_local_to_global %desc[%off0, %off1] %smem0 {async_task_id = array<i32: 4>} : !tt.tensordesc<tensor<128x64xf16, #shared2>>, !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>

    %trunc1 = arith.truncf %rhs {async_task_id = array<i32: 3>} : tensor<128x64xf32, #blocked2d2> to tensor<128x64xf16, #blocked2d2>
    %smem1 = ttg.local_alloc %trunc1 {async_task_id = array<i32: 3>} : (tensor<128x64xf16, #blocked2d2>) -> !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>
    ttng.async_tma_copy_local_to_global %desc[%off0, %off2] %smem1 {async_task_id = array<i32: 4>} : !tt.tensordesc<tensor<128x64xf16, #shared2>>, !ttg.memdesc<128x64xf16, #shared2, #smem2, mutable>

    tt.return
  }
}
