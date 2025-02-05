// RUN: triton-opt %s -split-input-file --tritonamdgpu-ws-lowering | FileCheck %s
// CHECK-NOT: triton_nvidia_gpu.get_async_task_id
// CHECK: foo
// CHECK: rocdl.workitem.id.x : i32
// CHECK: arith.divsi %0, %c64_i32 : i32

#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @foo() -> i32{
    %14 = triton_nvidia_gpu.get_async_task_id : i32
    tt.return %14 : i32
  }
}
