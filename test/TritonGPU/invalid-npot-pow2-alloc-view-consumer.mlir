// RUN: env TRITON_ALLOW_NPOT=1 triton-opt --split-input-file %s --verify-diagnostics
// RUN: env -u TRITON_ALLOW_NPOT not triton-opt --split-input-file %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF --implicit-check-not="error: NPOT memdesc"

// A pow2 alloc view read through memdesc_reinterpret gets an error
// instead of reaching the reinterpret footprint math.

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module {
  // OFF: shape must have power-of-2 and non-zero dimensions; got 64, 40
  tt.func @reinterpret_shared_pow2_alloc(%a: !ttg.memdesc<64x40xf16, #shared, #smem, mutable, 64x64>) {
    // expected-error @+1 {{'ttg.memdesc_reinterpret' op NPOT memdesc pow2 alloc view on this consumer is not yet supported}}
    %0 = ttg.memdesc_reinterpret %a : !ttg.memdesc<64x40xf16, #shared, #smem, mutable, 64x64> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
