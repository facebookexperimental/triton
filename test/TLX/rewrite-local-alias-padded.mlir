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
