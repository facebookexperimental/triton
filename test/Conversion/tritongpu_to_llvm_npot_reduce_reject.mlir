// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 -verify-diagnostics
//
// Fail-safe: an NPOT reduction axis rounds up to a power of two, creating
// phantom register/lane/warp slots that must be identity-filled before the
// reduce. When the combine region has no determinable reduction identity (here
// a single-op arith.divf, which is neither a known scalar neutral nor a
// directional cmp+select arg-reduce), predicateAccWithIdentity() returns false
// and the pass must LOUD-REJECT rather than silently miscompute. This proves
// unrecognized combines are rejected, not silently mislowered.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @npot_reduce_divf_no_identity(%arg0: tensor<1x48xf32, #blocked>) {
    // 48 is not a power of two, so the reduction axis wraps and needs
    // identity-fill; arith.divf has no reduction identity, so the pattern emits
    // a loud error and the conversion framework then reports the illegal op.
    // expected-error @+2 {{requires a known reduction identity}}
    // expected-error @+1 {{failed to legalize operation 'tt.reduce'}}
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.divf %lhs, %rhs : f32
      tt.reduce.return %1 : f32
    }) : (tensor<1x48xf32, #blocked>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return
  }
}
