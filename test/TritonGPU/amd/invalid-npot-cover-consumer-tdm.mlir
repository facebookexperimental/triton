// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s --verify-diagnostics
// RUN: env -u TRITON_ALLOW_NPOT not triton-opt %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF --implicit-check-not="error: NPOT memdesc"

// A physical cover used as a TDM copy destination gets an error
// instead of reaching the TDM layout-consistency math.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module {
  // OFF: shape must have power-of-2 and non-zero dimensions; got 64, 40
  tt.func @tdm_copy_cover(%desc: !tt.tensordesc<64x40xf16, #shared>, %dst: !ttg.memdesc<64x40xf16, #shared, #smem, mutable, 64x64>) {
    // expected-error @+1 {{'amdg.async_tdm_copy_global_to_local' op NPOT memdesc physical cover view on this consumer is not yet supported}}
    %0 = amdg.async_tdm_copy_global_to_local %desc into %dst : !tt.tensordesc<64x40xf16, #shared> -> !ttg.memdesc<64x40xf16, #shared, #smem, mutable, 64x64>
    tt.return
  }
}
