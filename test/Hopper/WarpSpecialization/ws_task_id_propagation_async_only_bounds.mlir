// RUN: triton-opt %s --nvgpu-test-taskid-propagate=num-warp-groups=2 | FileCheck %s

// Regression test for B-6-F2 / T273474506.
//
// This is the Hopper async-only variant of `nested_for_constant_bounds` from
// `ws_task_id_propagation.mlir`: anchors already have `ttg.partition`, but no
// op has `ttg.partition`. Loop bound constants should still receive the union
// of all partition IDs.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @nested_for_async_only_constant_bounds
  // CHECK:       %[[C0:.*]] = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 0 : i32
  // CHECK-NEXT:  %[[C1:.*]] = arith.constant {ttg.partition = array<i32: 0, 1, 2>} 1 : i32
  // CHECK:       scf.for
  // CHECK:         scf.for %{{.*}} = %[[C0]] to %{{.*}} step %[[C1]]

  tt.func public @nested_for_async_only_constant_bounds(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %arg3: i32, %arg4: i32, %arg5: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %pid = tt.get_program_id x : i32
    %nprogs = tt.get_num_programs x : i32
    scf.for %tile = %pid to %arg3 step %nprogs : i32 {
      // Inner loop: only tasks 1 (loads) and 2 (dot/alloc) are present.
      // Bounds %c0 and %c1 are constants defined at function scope.
      %inner:2 = scf.for %k = %c0 to %arg5 step %c1 iter_args(%acc = %cst, %off = %c0) -> (tensor<128x256xf32, #mma>, i32) : i32 {
        %a = tt.descriptor_load %arg0[%tile, %off] {ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
        %a_alloc = ttg.local_alloc %a {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %b = tt.descriptor_load %arg1[%off, %tile] {ttg.partition = array<i32: 1>} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
        %b_alloc = ttg.local_alloc %b {ttg.partition = array<i32: 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        %dot = ttng.warp_group_dot %a_alloc, %b_alloc, %acc {ttg.partition = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        %new_off = arith.addi %off, %c64 {ttg.partition = array<i32: 1>} : i32
        scf.yield %dot, %new_off : tensor<128x256xf32, #mma>, i32
      }
      // Epilogue: only task 0 ops. This task has no ops inside the inner loop.
      %trunc = arith.truncf %inner#0 {ttg.partition = array<i32: 0>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %cvt = ttg.convert_layout %trunc {ttg.partition = array<i32: 0>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.descriptor_store %arg2[%tile, %tile], %cvt {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    }
    tt.return
  }
}
