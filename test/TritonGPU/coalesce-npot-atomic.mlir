// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -tritongpu-coalesce -verify-diagnostics

// Consumed atomic results need the canonical owner's old value redistributed
// to every modular alias and remain unsupported.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @npot_atomic_result_used(%arg0: tensor<48x!tt.ptr<i32>, #blocked>, %arg1: tensor<48xi32, #blocked>, %arg2: tensor<48xi1, #blocked>) {
    // expected-error@+1 {{NPOT atomic results are not yet supported with modular tensor layouts}}
    %0 = tt.atomic_rmw add, relaxed, gpu, %arg0, %arg1, %arg2 : (tensor<48x!tt.ptr<i32>, #blocked>, tensor<48xi32, #blocked>, tensor<48xi1, #blocked>) -> tensor<48xi32, #blocked>
    tt.store %arg0, %0, %arg2 : tensor<48x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

// Consumed CAS results require the same alias redistribution as AtomicRMW.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @npot_atomic_cas_result_used(%ptr: tensor<48x!tt.ptr<i32>, #blocked>, %cmp: tensor<48xi32, #blocked>, %val: tensor<48xi32, #blocked>) {
    // expected-error@+1 {{NPOT atomic results are not yet supported with modular tensor layouts}}
    %0 = tt.atomic_cas relaxed, gpu, %ptr, %cmp, %val : (tensor<48x!tt.ptr<i32>, #blocked>, tensor<48xi32, #blocked>, tensor<48xi32, #blocked>) -> tensor<48xi32, #blocked>
    tt.store %ptr, %0 : tensor<48x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

// Multidimensional layouts use one pre-modulo bound per NPOT dimension.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @npot_atomic_result_unused_2d(%ptr: tensor<64x48x!tt.ptr<i32>, #blocked>, %val: tensor<64x48xi32, #blocked>) {
    %0 = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (tensor<64x48x!tt.ptr<i32>, #blocked>, tensor<64x48xi32, #blocked>) -> tensor<64x48xi32, #blocked>
    tt.return
  }
}

// -----

// Result-unused atomics are safe once lowering predicates one canonical
// register/lane/warp representative for every modular alias class.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @npot_atomic_result_unused(%arg0: tensor<48x!tt.ptr<i32>, #blocked>, %arg1: tensor<48xi32, #blocked>, %arg2: tensor<48xi1, #blocked>) {
    %0 = tt.atomic_rmw add, relaxed, gpu, %arg0, %arg1, %arg2 : (tensor<48x!tt.ptr<i32>, #blocked>, tensor<48xi32, #blocked>, tensor<48xi1, #blocked>) -> tensor<48xi32, #blocked>
    tt.return
  }
}

// -----

// Sub-32-bit atomics may require backend packing across an ownership boundary.
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @npot_atomic_f16(%arg0: tensor<48x!tt.ptr<f16>, #blocked>, %arg1: tensor<48xf16, #blocked>) {
    // expected-error@+1 {{NPOT atomic operations do not yet support element types narrower than 32 bits}}
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg1 : (tensor<48x!tt.ptr<f16>, #blocked>, tensor<48xf16, #blocked>) -> tensor<48xf16, #blocked>
    tt.return
  }
}

// -----

// Mixed-radix owner bounds convert correctly, but the normal pipeline remains
// gated until cluster-capable hardware validates cross-CTA ownership.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @npot_atomic_multicta(%arg0: tensor<48x!tt.ptr<i32>, #blocked>, %arg1: tensor<48xi32, #blocked>) {
    // expected-error@+1 {{NPOT atomic operations do not yet support multiple CTAs per cluster}}
    %0 = tt.atomic_rmw add, relaxed, gpu, %arg0, %arg1 : (tensor<48x!tt.ptr<i32>, #blocked>, tensor<48xi32, #blocked>) -> tensor<48xi32, #blocked>
    tt.return
  }
}
