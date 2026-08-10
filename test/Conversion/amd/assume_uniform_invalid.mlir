// RUN: triton-opt %s -split-input-file --verify-diagnostics

// 8-bit and narrower types abort AMDGPU isel, so the verifier rejects them.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    tt.func @assume_uniform_rejects_i8(%i8: i8) {
        // expected-error @+1 {{must be 16/32/64-bit scalar or pointer}}
        %0 = amdg.assume_uniform %i8 : i8
        tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    tt.func @assume_uniform_rejects_fp8(%f8: f8E4M3FN) {
        // expected-error @+1 {{must be 16/32/64-bit scalar or pointer}}
        %0 = amdg.assume_uniform %f8 : f8E4M3FN
        tt.return
  }
}
