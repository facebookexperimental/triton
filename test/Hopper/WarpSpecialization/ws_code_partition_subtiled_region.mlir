// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=1 post-channel-creation=1" | FileCheck %s

// Test: Code partition with SubtiledRegionOps for epilogue subtiling.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @subtiled_smem_channel
  // CHECK: ttg.warp_specialize
  //
  // Partition 0 (epilogue): SubtiledRegionOp with inline producer ops
  // CHECK: partition0
  // CHECK:   ttng.subtiled_region
  // CHECK:     nvws.producer_acquire
  // CHECK:     ttg.local_store
  // CHECK:     nvws.producer_commit
  //
  // Partition 1 (store): SubtiledRegionOp with inline consumer ops
  // CHECK: partition1
  // CHECK:   ttng.subtiled_region
  // CHECK:     nvws.consumer_wait
  // CHECK:     ttng.async_tma_copy_local_to_global
  // CHECK:     nvws.consumer_release
  tt.func @subtiled_smem_channel(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %off0: i32, %off1: i32, %off2: i32,
      %lhs: tensor<128x64xf32, #linear>,
      %rhs: tensor<128x64xf32, #linear>) {
    %smem0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index

    scf.for %iv = %c0 to %c10 step %c1 {
      %dummy = arith.constant {async_task_id = array<i32: 0>} 0 : i32

      // Epilogue SubtiledRegionOp (task 1): truncf + local_store
      ttng.subtiled_region
          per_tile(%rhs, %lhs, %smem0, %smem1 :
                   tensor<128x64xf32, #linear>, tensor<128x64xf32, #linear>,
                   !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
                   !ttg.memdesc<128x64xf16, #shared, #smem, mutable>)
          {numTiles = 2 : i32, async_task_id = array<i32: 1>}
        tile(%t0: tensor<128x64xf32, #linear>,
             %t1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
             %tidx: i32) {
          %trunc = arith.truncf %t0 {async_task_id = array<i32: 1>}
            : tensor<128x64xf32, #linear> to tensor<128x64xf16, #linear>
          ttg.local_store %trunc, %t1 {async_task_id = array<i32: 1>}
            : tensor<128x64xf16, #linear> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.subtiled_region_yield
        }

      // TMA store SubtiledRegionOp (task 2): async_tma_copy
      ttng.subtiled_region
          per_tile(%smem0, %smem1, %off1, %off2 :
                   !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
                   !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
                   i32, i32)
          shared(%desc, %off0 :
                 !tt.tensordesc<tensor<128x64xf16, #shared>>, i32)
          {numTiles = 2 : i32, async_task_id = array<i32: 2>}
        tile(%t0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
             %t1: i32,
             %tdesc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
             %toff0: i32, %tidx: i32) {
          ttng.async_tma_copy_local_to_global %tdesc[%toff0, %t1] %t0
            {async_task_id = array<i32: 2>}
            : !tt.tensordesc<tensor<128x64xf16, #shared>>,
              !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.subtiled_region_yield
        }
    } {async_task_id = array<i32: 0, 1, 2>, tt.warp_specialize,
       tt.separate_epilogue_store = true,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["compute", "epilogue", "epilogue_store"]}

    tt.return
  }
}
