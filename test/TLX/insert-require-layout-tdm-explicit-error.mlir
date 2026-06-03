// RUN: triton-opt -split-input-file --tlx-insert-require-layout -verify-diagnostics %s
//
// TLXInsertRequireLayout must reject TDM operand buffers whose alloc carries
// a user-supplied non-padded shared encoding. The user-supplied case is
// signalled by the discardable `tlx.layout_is_explicit` attribute on the
// alloc op (set by the Python `tlx.local_alloc(layout=...)` builder). Without
// the marker the alloc is treated as auto-default and silently substituted —
// see insert-require-layout-tdm.mlir for those cases.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// =============================================================================
// Explicit swizzled on a TDM load buffer -> hard error.
// =============================================================================

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_load_explicit_swizzled(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc {tlx.layout_is_explicit} : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    // expected-error @+1 {{TDM operand requires a padded shared encoding}}
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// =============================================================================
// Explicit swizzled on a TDM store buffer -> hard error.
// =============================================================================

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_store_explicit_swizzled(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc {tlx.layout_is_explicit} : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    // expected-error @+1 {{TDM operand requires a padded shared encoding}}
    amdg.async_tdm_copy_local_to_global %desc[%m, %k] from %buf : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> !tt.tensordesc<128x32xf16>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// =============================================================================
// Same non-padded encoding without the marker -> silent override, no error.
// (Treated as auto-default; lit/raw-MLIR back-compat.)
// =============================================================================

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_no_marker_silent(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#padded = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>
#smem = #ttg.shared_memory

// =============================================================================
// Explicit padded encoding -> preserved, no error.
// =============================================================================

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_explicit_padded_ok(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc {tlx.layout_is_explicit} : () -> !ttg.memdesc<2x128x32xf16, #padded, #smem, mutable>
    %buf = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x128x32xf16, #padded, #smem, mutable> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable>
    %tok = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable>
    tt.return
  }
}
