// RUN: triton-opt %s --nvgpu-test-ws-buffer-allocation | FileCheck %s

// Regression test distilled from:
// python/test/unit/language/test_tutorial09_warp_specialization.py::test_tutorial09_matmul_tma_persistent_warp_specialize[True-True-1-2-False-False-False-2-False-4-2-64-256-256-128-128-128]
//
// The epilogue reordering pass may move producer ops before createBuffer
// inserts local_store/local_load pairs. For TMA stores, local_store order must
// match the descriptor_store order for the same descriptor; later TMA-store
// wait rotation reasons about that shared-buffer sequence.

// CHECK-LABEL: @tma_store_local_store_order
// CHECK: ttg.local_store {{.*}}, %[[C0_A:[0-9]+]]
// CHECK: ttg.local_store {{.*}}, %[[C0_B:[0-9]+]]
// CHECK: ttg.local_load %[[C0_A]]
// CHECK: tt.descriptor_store
// CHECK: ttg.local_load %[[C0_B]]
// CHECK: tt.descriptor_store
// CHECK: ttg.local_store {{.*}}, %[[C1_A:[0-9]+]]
// CHECK: ttg.local_store {{.*}}, %[[C1_B:[0-9]+]]
// CHECK: ttg.local_load %[[C1_A]]
// CHECK: tt.descriptor_store
// CHECK: ttg.local_load %[[C1_B]]
// CHECK: tt.descriptor_store

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 0, 64], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 64, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_store_local_store_order(%c_desc: !tt.tensordesc<tensor<128x128xf16, #shared>>) attributes {noinline = false} {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 2>} 0 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 2>} 128 : i32
    %acc0 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<128x256xf32, #linear1>
    %acc1 = arith.constant {async_task_id = array<i32: 0>} dense<1.000000e+00> : tensor<128x256xf32, #linear1>
    %row1 = arith.addi %c0_i32, %c128_i32 {async_task_id = array<i32: 2>} : i32
    %col1 = arith.addi %c0_i32, %c128_i32 {async_task_id = array<i32: 2>} : i32
    %acc0_reshape = tt.reshape %acc0 {async_task_id = array<i32: 0>} : tensor<128x256xf32, #linear1> -> tensor<128x2x128xf32, #linear2>
    %acc1_reshape = tt.reshape %acc1 {async_task_id = array<i32: 0>} : tensor<128x256xf32, #linear1> -> tensor<128x2x128xf32, #linear2>
    %acc0_trans = tt.trans %acc0_reshape {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #linear2> -> tensor<128x128x2xf32, #linear3>
    %acc1_trans = tt.trans %acc1_reshape {async_task_id = array<i32: 0>, order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #linear2> -> tensor<128x128x2xf32, #linear3>
    %out0_lhs, %out0_rhs = tt.split %acc0_trans {async_task_id = array<i32: 0>} : tensor<128x128x2xf32, #linear3> -> tensor<128x128xf32, #linear>
    %out1_lhs, %out1_rhs = tt.split %acc1_trans {async_task_id = array<i32: 0>} : tensor<128x128x2xf32, #linear3> -> tensor<128x128xf32, #linear>
    %c0_a = arith.truncf %out0_lhs {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
    %c0_b = arith.truncf %out1_lhs {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
    %c0_a_store = ttg.convert_layout %c0_a {async_task_id = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked>
    %c0_b_store = ttg.convert_layout %c0_b {async_task_id = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked>
    tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %c0_a_store {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked>
    tt.descriptor_store %c_desc[%row1, %c0_i32], %c0_b_store {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked>
    %c1_a = arith.truncf %out0_rhs {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
    %c1_b = arith.truncf %out1_rhs {async_task_id = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
    %c1_a_store = ttg.convert_layout %c1_a {async_task_id = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked>
    %c1_b_store = ttg.convert_layout %c1_b {async_task_id = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked>
    tt.descriptor_store %c_desc[%c0_i32, %col1], %c1_a_store {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked>
    tt.descriptor_store %c_desc[%row1, %col1], %c1_b_store {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked>
    tt.return
  }
}
