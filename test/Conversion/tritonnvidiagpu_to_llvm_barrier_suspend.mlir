// RUN: triton-opt %s --convert-triton-gpu-to-llvm=compute-capability=100 -reconcile-unrealized-casts | FileCheck %s

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @wait_barrier(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>, %phase: i32, %pred: i1) {
    // CHECK: mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], $1;
    // CHECK-NOT: nanosleep.u32
    // CHECK: @!complete bra.uni waitLoop
    // CHECK: %{{[0-9]+}}, %arg1 :
    ttng.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared0, #smem>

    // CHECK: @!$2 bra.uni skipWait
    // CHECK: mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], $1;
    // CHECK: %{{[0-9]+}}, %arg1, %arg2 :
    ttng.wait_barrier %alloc, %phase, %pred : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}
