## GEMM kernels
[Pipelined GEMM on Hopper](third_party/tlx/tutorials/hopper_gemm_pipelined.py)

[Warp-specialized GEMM on Hopper](third_party/tlx/tutorials/hopper_gemm_ws.py)

[Warp-specialized GEMM on Blackwell](third_party/tlx/tutorials/blackwell_gemm_ws.py)

[Warp-specialized MXFP8 GEMM on Blackwell](third_party/tlx/tutorials/blackwell_gemm_ws_mxfp8.py)

[Grouped GEMM on Blackwell](third_party/tlx/tutorials/blackwell-grouped-gemm_test.py)

[Pipelined GEMM on Blackwell](third_party/tlx/tutorials/blackwell_gemm_pipelined.py)

[CLC GEMM on Blackwell](third_party/tlx/tutorials/blackwell_gemm_clc.py)

[2-CTA GEMM on Blackwell](third_party/tlx/tutorials/blackwell_gemm_2cta.py)

## Attention kernels

[Warp-specialized pipelined persistent FA fwd/bwd on Blackwell](third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py)

[Warp-Specialized computation-pipelined pingpong FA fwd on Hopper](third_party/tlx/tutorials/hopper_fa_ws_pipelined_pingpong.py)

## AMD kernels (gfx950 / CDNA4)

[LDS-pipelined GEMM](third_party/tlx/tutorials/amd_gemm_pipelined.py)

[Warp-pipelined GEMM](third_party/tlx/tutorials/amd_gemm_warp_pipeline.py)

[Async-DMA Flash Attention fwd — simple / prefetch](third_party/tlx/tutorials/amd_fa_pipelined.py)

[Persistent Flash Attention fwd — XCD zig-zag, cross-attention / decode](third_party/tlx/tutorials/amd_fa_persistent.py)

[Rotated 4-cluster Flash Attention fwd](third_party/tlx/tutorials/amd_fa_cluster.py)

[Fused addmm + GLU (Gated Linear Unit: out = x + x*y, x = A@B + bias)](third_party/tlx/tutorials/amd_addmm_glu.py)

[IKBO Flash Attention (In-Kernel Broadcast Optimization, candidate/user broadcast)](third_party/tlx/tutorials/ikbo/ikbo_fa_triton.py)

[IKBO LCE (logit cross-entropy over candidate/user embeddings — not attention)](third_party/tlx/tutorials/ikbo/ikbo_lce_triton.py)

## AMD kernels (gfx1250)

[TDM-pipelined GEMM](third_party/tlx/tutorials/amd_tdm_gemm_pipelined.py)

[MXFP TDM-pipelined GEMM](third_party/tlx/tutorials/amd_mxfp_gemm_tdm_pipelined.py)
