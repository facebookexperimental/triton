// RUN: triton-opt %s --nvgpu-test-ws-buffer-allocation | FileCheck %s

// A descriptor load with consumers in different software-pipeline stages is
// lowered through one shared local_load.  Give that load the earliest
// consumer's schedule so a forwarding consumer can publish the value before a
// later producer acquire blocks the load task.

// CHECK-LABEL: @tma_multi_consumer_schedule
// CHECK: nvws.descriptor_load
// CHECK: ttg.local_load {{.*}}loop.cluster = 1 : i32, loop.stage = 1 : i32
// CHECK: arith.addf {{.*}}loop.cluster = 1 : i32, loop.stage = 1 : i32
// CHECK: arith.subf {{.*}}loop.cluster = 0 : i32, loop.stage = 2 : i32

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_multi_consumer_schedule(%desc: !tt.tensordesc<128x64xf16, #shared>) attributes {noinline = false} {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 0>, loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
    %early = arith.addf %tile, %tile {async_task_id = array<i32: 1>, loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #blocked>
    %late = arith.subf %tile, %tile {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x64xf16, #blocked>
    tt.return
  }
}
