// RUN: triton-opt %s --nvgpu-verify-ws-barriers="emit-coverage-table=true" -verify-diagnostics
//
// Phase-0 smoke test for the autoWS barrier verifier
// (--nvgpu-verify-ws-barriers). Goal 1: the barrier -> reuse group mapping is
// given, carried as a `buffer.id` attribute on each `ttng.init_barrier`. The
// verifier is verify-only (module unchanged): it groups WS-generated barriers by
// reuse group and reports the mapping, flagging barriers with no `buffer.id` as
// indeterminate. Later phases add the deadlock (BDG/SCC) and race (AOG) checks.
// See lib/Transforms/WarpSpecialization/docs/WSAliasingCoverage.proposal.md.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wsverify_smoke() {
    %c0_i32 = arith.constant 0 : i32
    %b0 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %v0 = ttg.memdesc_index %b0[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // expected-remark @below {{WS barrier guards reuse group buffer.id=6}}
    ttng.init_barrier %v0, 1 {buffer.id = 6 : i32} : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    %b1 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %v1 = ttg.memdesc_index %b1[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // expected-remark @below {{WS barrier guards reuse group buffer.id=6}}
    ttng.init_barrier %v1, 1 {buffer.id = 6 : i32} : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    // A barrier with no buffer.id -> reuse group unknown -> indeterminate.
    %b2 = ttg.local_alloc {ttg.ws_generated_barrier} : () -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %v2 = ttg.memdesc_index %b2[%c0_i32] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // expected-remark @below {{WS barrier has no buffer.id (reuse group unknown) -> indeterminate}}
    ttng.init_barrier %v2, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
