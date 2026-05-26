// RUN: triton-opt -split-input-file --tlx-rewrite-local-alias %s | FileCheck %s

#padded_a = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [256, 128]}>
#padded_c = #ttg.padded_shared<[256:+8] {order = [1, 0], shape = [256, 256]}>
#smem = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @padded_shared_gemm_epilogue_alias
  tt.func public @padded_shared_gemm_epilogue_alias() {
    // CHECK: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x256x128xf16, #[[$A:.*]], #smem, mutable>
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x256x128xf16, #padded_a, #smem, mutable>
    // CHECK-NOT: tlx.local_alias
    // CHECK: ttg.memdesc_reinterpret %[[ALLOC]] : !ttg.memdesc<2x256x128xf16, #[[$A]], #smem, mutable> -> !ttg.memdesc<1x256x256xbf16, #[[$C:.*]], #smem, mutable>
    %c = tlx.local_alias %a : !ttg.memdesc<2x256x128xf16, #padded_a, #smem, mutable> -> !ttg.memdesc<1x256x256xbf16, #padded_c, #smem, mutable>
    tt.return
  }
}

// -----

// Size-mismatched alias: the c view only covers 1/8th of the a allocation
// (2x32x128xf16 = 131072 bits, 1x32x32xbf16 = 16384 bits). Upstream PR
// #10243 rejects a single memdesc_reinterpret across different sizes, so
// the pass must emit reinterpret -> memdesc_index[0] -> reinterpret to
// produce a fresh alias descriptor.
#padded_a_small = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [32, 128]}>
#padded_c_small = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [32, 32]}>
#smem1 = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @shrink_alias
  tt.func public @shrink_alias() {
    // CHECK: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #[[$A:.*]], #smem, mutable>
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #padded_a_small, #smem1, mutable>
    // CHECK-NOT: tlx.local_alias
    // CHECK: %[[BIG:.*]] = ttg.memdesc_reinterpret %[[ALLOC]] : !ttg.memdesc<2x32x128xf16, #[[$A]], #smem, mutable> -> !ttg.memdesc<8x32x32xbf16, #[[$C:.*]], #smem, mutable>
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[SLOT:.*]] = ttg.memdesc_index %[[BIG]][%[[C0]]] : !ttg.memdesc<8x32x32xbf16, #[[$C]], #smem, mutable> -> !ttg.memdesc<32x32xbf16, #[[$C]], #smem, mutable>
    // CHECK: ttg.memdesc_reinterpret %[[SLOT]] : !ttg.memdesc<32x32xbf16, #[[$C]], #smem, mutable> -> !ttg.memdesc<1x32x32xbf16, #[[$C]], #smem, mutable>
    %c = tlx.local_alias %a : !ttg.memdesc<2x32x128xf16, #padded_a_small, #smem1, mutable> -> !ttg.memdesc<1x32x32xbf16, #padded_c_small, #smem1, mutable>
    tt.return
  }
}
