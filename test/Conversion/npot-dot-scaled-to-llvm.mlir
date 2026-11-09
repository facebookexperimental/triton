// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 -verify-diagnostics

// The target rejects a dot_scaled whose only exact NPOT operand is B, so the
// op stays illegal and the conversion fails. A tt.dot can't get here: its A and
// B must be dot operands, and the verifier rejects NPOT dot operands.
#bl = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#p = #ttg.linear<{register = [[0, 1], [0, 2], [16, 0], [0, 32]], lane = [[0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = [], inputRadices = {register = [2, 2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [2, 2], block = []}, outDimSizes = [32, 96]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_radix_dot_scaled_npot_b() {
    %a = arith.constant dense<1.0> : tensor<32x32xf8E4M3FN, #bl>
    %b = arith.constant dense<1.0> : tensor<32x96xf8E4M3FN, #p>
    %c = arith.constant dense<0.0> : tensor<32x96xf32, #p>
    // expected-error @+2 {{NPOT dot is not yet supported on cuda:90}}
    // expected-error @+1 {{failed to legalize operation 'tt.dot_scaled' that was explicitly marked illegal}}
    %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x32xf8E4M3FN, #bl> * tensor<32x96xf8E4M3FN, #p> -> tensor<32x96xf32, #p>
    tt.return
  }
}
