// RUN: triton-opt %s --nvgpu-test-ws-code-partition=num-buffers=2 | FileCheck %s
// RUN: env TRITON_USE_META_WS=1 triton-opt %s -mlir-disable-threading \
// RUN:   --nvgpu-warp-specialization='capability=90 num-stages=2 smem-budget=232448' \
// RUN:   --triton-nvidia-tma-lowering --tritongpu-allocate-warp-groups --convert-scf-to-cf \
// RUN:   --allocate-shared-memory-nv='compute-capability=90 ptx-version=80' \
// RUN:   --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=80' \
// RUN:   --initialize-ws-cluster-barriers='compute-capability=90 ptx-version=80' \
// RUN:   --canonicalize --cse --convert-nv-gpu-to-llvm --convert-warp-specialize-to-llvm \
// RUN:   -reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM

// A multicast TMA producer needs a full-cluster reuse rendezvous, and every
// consumer task needs an independent full-cluster ready rendezvous.

// CHECK-LABEL: @tma_multicast
// CHECK-NOT: ttng.cluster_barrier
// CHECK: ttng.arrive_barrier %[[REUSE:.*]], 1
// CHECK-SAME: ctaMask = 3 : i32
// CHECK: ttng.wait_barrier %[[REUSE]],
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK-SAME: tt.multicast_axes = array<i32: 1>
// CHECK: ttng.arrive_barrier %[[READY0:.*]], 1
// CHECK-SAME: ctaMask = 3 : i32
// CHECK: ttng.wait_barrier %[[READY0]],
// CHECK: ttng.arrive_barrier %[[READY1:.*]], 1
// CHECK-SAME: ctaMask = 3 : i32
// CHECK: ttng.wait_barrier %[[READY1]],

// LLVM-LABEL: @tma_multicast
// LLVM: %[[REUSE_PRED:.*]] = llvm.zext {{.*}} {async_task_id = array<i32: 1>
// LLVM: %[[REUSE_ID:.*]] = llvm.mlir.constant(2 : i32) : i32
// LLVM-NEXT: %[[REUSE_THREADS:.*]] = llvm.mlir.constant(128 : i32) : i32
// LLVM-NEXT: nvvm.barrier id = %[[REUSE_ID]] number_of_threads = %[[REUSE_THREADS]]
// LLVM: %[[READY0_PRED:.*]] = llvm.zext {{.*}} {async_task_id = array<i32: 2>
// LLVM: %[[READY0_ID:.*]] = llvm.mlir.constant(3 : i32) : i32
// LLVM-NEXT: %[[READY0_THREADS:.*]] = llvm.mlir.constant(128 : i32) : i32
// LLVM-NEXT: nvvm.barrier id = %[[READY0_ID]] number_of_threads = %[[READY0_THREADS]]
// LLVM: nvvm.cluster.wait {aligned}
// LLVM: %[[READY1_ID:.*]] = llvm.mlir.constant(0 : i32) : i32
// LLVM-NEXT: %[[READY1_THREADS:.*]] = llvm.mlir.constant(128 : i32) : i32
// LLVM-NEXT: nvvm.barrier id = %[[READY1_ID]] number_of_threads = %[[READY1_THREADS]]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {
  "ttg.cluster-dim-x" = 2 : i32,
  "ttg.cluster-dim-y" = 2 : i32,
  "ttg.cluster-dim-z" = 1 : i32,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  ttg.target = "cuda:90"
} {
  tt.func public @tma_multicast(
      %desc: !tt.tensordesc<128x64xf16, #shared>,
      %out0: !tt.ptr<f16>, %out1: !tt.ptr<f16>) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %ptrs0 = tt.splat %out0 {async_task_id = array<i32: 1>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %ptrs1 = tt.splat %out1 {async_task_id = array<i32: 2>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %i0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : index
    %i1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : index
    %i4 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 4 : index
    %buffer = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %iv = %i0 to %i4 step %i1 {
      %tile = tt.descriptor_load %desc[%pid, %c0] {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32, tt.multicast_axes = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      ttg.local_store %tile, %buffer {async_task_id = array<i32: 0>, loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %local0 = ttg.local_load %buffer {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %consumer0 = arith.addf %local0, %local0 {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked>
      %local1 = ttg.local_load %buffer {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %consumer1 = arith.subf %local1, %local1 {async_task_id = array<i32: 2>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x64xf16, #blocked>
      tt.store %ptrs0, %consumer0 {async_task_id = array<i32: 1>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      tt.store %ptrs1, %consumer1 {async_task_id = array<i32: 2>} : tensor<128x64x!tt.ptr<f16>, #blocked>
      scf.yield
    } {async_task_id = array<i32: 0, 1, 2>, tt.warp_specialize}
    tt.return
  }
}
