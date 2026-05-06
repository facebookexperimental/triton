// RUN: triton-opt -split-input-file --tritonamdgpu-optimize-descriptor-encoding --verify-diagnostics %s

// Test that `alignTDMDescriptorEncodings` rejects two TDM copies on the same
// descriptor that disagree on the destination memdesc encoding. There's no
// principled way to pick one encoding over the other, and silently keeping
// the default would re-introduce the OOB mismatch the pass is meant to
// prevent.

#shared_a = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 32]}>
#shared_b = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [128, 32]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_conflicting_destination_encodings(%desc: !tt.tensordesc<128x32xf16>, %m: i32, %k: i32, %p: i32) {
    %c0 = arith.constant 0 : i32
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #shared_a, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #shared_b, #smem, mutable>
    %buf_a = ttg.memdesc_index %alloc_a[%c0] : !ttg.memdesc<1x128x32xf16, #shared_a, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared_a, #smem, mutable>
    %buf_b = ttg.memdesc_index %alloc_b[%c0] : !ttg.memdesc<1x128x32xf16, #shared_b, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared_b, #smem, mutable>
    %tok_a = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf_a, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared_a, #smem, mutable>
    // expected-error @+1 {{TDM copies using the same descriptor require conflicting destination layouts}}
    %tok_b = amdg.async_tdm_copy_global_to_local %desc[%m, %k] into %buf_b, pred = %p : !tt.tensordesc<128x32xf16> -> !ttg.memdesc<128x32xf16, #shared_b, #smem, mutable>
    tt.return
  }
}
