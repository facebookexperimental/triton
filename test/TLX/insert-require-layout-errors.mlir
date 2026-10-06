// RUN: triton-opt --tlx-insert-require-layout --verify-diagnostics %s

// The destination and source distribute CTAs along different dimensions. The
// padded layout still has enough offset bases for register-layout inference,
// but the inferred layout cannot form a coalesced gfx950 direct-to-LDS write.
// Reject it before inserting any pinned offset layout.

#blocked_incompatible = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [0, 0]]}>
#padded_incompatible = #ttg.padded_shared<[128:+8] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], block = [[0, 0], [64, 0]]}>
#user_incompatible = #tlx.user_layout<#padded_incompatible>
#smem_incompatible = #ttg.shared_memory
module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @buffer_load_rejects_inferred_layout_incompatible_with_direct_to_lds(
      %ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %offsets: tensor<128x16xi32, #blocked_incompatible> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<[1, 16]> : tensor<2xi32>}) {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<128x16xbf16, #user_incompatible, #smem_incompatible, mutable>
    // expected-error @+1 {{the inferred offset layout cannot write directly to the destination}}
    %tok = amdg.buffer_load_to_local %ptr[%offsets] into %alloc {contiguity = 2 : i32} : <bf16>[tensor<128x16xi32, #blocked_incompatible>] -> <128x16xbf16, #user_incompatible, #smem_incompatible, mutable>
    tt.return
  }
}
