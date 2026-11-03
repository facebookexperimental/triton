// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' -verify-diagnostics

// NPOT TTIR verifies under TRITON_ALLOW_NPOT, but the conversion gives every
// tensor an encoding, so the verifier after the pass rejects it.

module {
  // expected-error @+1 {{Number of elements must be power-of-two, but <block argument> of type}}
  tt.func @print_96(%a: tensor<96xf32>) {
    tt.print ": " {hex = false, isSigned = array<i32: 0>} : %a : tensor<96xf32>
    tt.return
  }
}
