// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: clock64
  tt.func public @clock64(%out: !tt.ptr<i64>) {
    // CHECK: %[[CLOCK:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.memtime"() : () -> i64
    // CHECK: %[[CLOCK_VEC:.*]] = llvm.insertelement %[[CLOCK]], {{.*}} : vector<1xi64>
    // CHECK: llvm.store %[[CLOCK_VEC]],
    %clock = ttg.clock64
    tt.store %out, %clock : !tt.ptr<i64>
    tt.return
  }
}
