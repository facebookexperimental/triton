"""Compatibility entry points for the promoted gfx950 Flash Attention kernel.

The implementation now lives in ``triton.tlx.ops.kernels.flash_attn.gfx950``.
New callers should use ``triton.tlx.ops.flash_attn``; these names remain for
the tutorial correctness and performance harnesses.
"""

from triton.tlx.ops.kernels.flash_attn.gfx950 import (
    CDNA_MFMA_ROWS_PER_WAVE,
    CDNA_WAVE_SIZE,
    CLUSTER_BUF_DEPTH,
    CLUSTER_PIPELINE_STAGES,
    DIAGONAL_LAZY_RESCALE_THRESHOLD,
    DIAGONAL_LAZY_RESCALE_THRESHOLD_FP16,
    LAZY_RESCALE_THRESHOLD,
    LazyProbabilityState,
    SoftmaxState,
    attention,
    flash_attn_cluster_persistent_pipeline,
    flash_attn_cluster_pipeline,
    persistent_attention,
)

__all__ = [
    "CDNA_MFMA_ROWS_PER_WAVE",
    "CDNA_WAVE_SIZE",
    "CLUSTER_BUF_DEPTH",
    "CLUSTER_PIPELINE_STAGES",
    "DIAGONAL_LAZY_RESCALE_THRESHOLD",
    "DIAGONAL_LAZY_RESCALE_THRESHOLD_FP16",
    "LAZY_RESCALE_THRESHOLD",
    "LazyProbabilityState",
    "SoftmaxState",
    "attention",
    "flash_attn_cluster_persistent_pipeline",
    "flash_attn_cluster_pipeline",
    "persistent_attention",
]
