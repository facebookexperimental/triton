// RUN: triton-opt %s --convert-triton-gpu-to-llvm=compute-capability=100 -reconcile-unrealized-casts | FileCheck %s --check-prefix=SUSPEND

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"tlx.mbarrier_try_wait_suspend_ns" = 50000 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @wait_barrier(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>, %phase: i32, %pred: i1) {
    // SUSPEND: mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], $1, $2;
    // SUSPEND-NOT: nanosleep.u32
    // SUSPEND: %{{[0-9]+}}, %arg1, %{{[0-9]+}} :
    ttng.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared0, #smem>

    // SUSPEND: @!$3 bra.uni skipWait
    // SUSPEND: mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], $1, $2;
    // SUSPEND-NOT: nanosleep.u32
    // SUSPEND: %{{[0-9]+}}, %arg1, %{{[0-9]+}}, %arg2 :
    ttng.wait_barrier %alloc, %phase, %pred : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}
