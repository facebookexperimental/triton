// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s -tritongpu-coalesce -verify-diagnostics

// A modular layout may map multiple physical slots to one logical element.
// Atomics must reject until lowering predicates one canonical representative;
// otherwise each alias applies the update again.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @npot_atomic(%arg0: tensor<48x!tt.ptr<i32>, #blocked>, %arg1: tensor<48xi32, #blocked>, %arg2: tensor<48xi1, #blocked>) {
    // expected-error@+1 {{NPOT atomic operations are not yet supported with modular tensor layouts}}
    %0 = tt.atomic_rmw add, relaxed, gpu, %arg0, %arg1, %arg2 : (tensor<48x!tt.ptr<i32>, #blocked>, tensor<48xi32, #blocked>, tensor<48xi1, #blocked>) -> tensor<48xi32, #blocked>
    tt.return
  }
}
