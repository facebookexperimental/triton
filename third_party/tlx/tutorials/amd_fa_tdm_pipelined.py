"""Compatibility entry points; use ``triton.tlx.ops.flash_attn`` for new callers."""

from triton.tlx.ops.kernels.flash_attn.gfx1250 import (
    attn_fwd_tdm_pipelined,
    attn_fwd_tdm_pipelined_kernel,
)

__all__ = ["attn_fwd_tdm_pipelined", "attn_fwd_tdm_pipelined_kernel"]
