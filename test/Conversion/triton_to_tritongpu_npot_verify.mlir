// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4 num-ctas=2' -verify-diagnostics

// NPOT TTIR verifies under TRITON_ALLOW_NPOT. No mixed-radix layout spans CTAs,
// so the conversion keeps the Blocked default and the verifier after it rejects
// it.

module {
  // expected-error @+1 {{Number of elements must be power-of-two, but <block argument> of type}}
  tt.func @print_96(%a: tensor<96xf32>) {
    tt.print ": " {hex = false, isSigned = array<i32: 0>} : %a : tensor<96xf32>
    tt.return
  }
}
