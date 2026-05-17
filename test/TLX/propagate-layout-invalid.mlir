// RUN: triton-opt -split-input-file --verify-diagnostics --tlx-propagate-layout %s

#shared_scales = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 2}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory
#dummy_tmem_layout = #tlx.dummy_tmem_layout<>
#scales_encoding = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @logical_rank2_scale_tmem_copy_after_propagate(
      %scale_smem: !ttg.memdesc<128x4xi8, #shared_scales, #smem, mutable>) {
    %scale_tmem = ttng.tmem_alloc : () -> !ttg.memdesc<128x4xi8, #dummy_tmem_layout, #tmem, mutable>

    // expected-error @+1 {{scale tmem_copy requires an explicit packed i8 SMEM shape}}
    ttng.tmem_copy %scale_smem, %scale_tmem : !ttg.memdesc<128x4xi8, #shared_scales, #smem, mutable>, !ttg.memdesc<128x4xi8, #dummy_tmem_layout, #tmem, mutable>

    %scale_req = tlx.require_layout %scale_tmem : !ttg.memdesc<128x4xi8, #dummy_tmem_layout, #tmem, mutable> -> !ttg.memdesc<128x4xi8, #scales_encoding, #tmem, mutable>
    tt.return
  }
}
