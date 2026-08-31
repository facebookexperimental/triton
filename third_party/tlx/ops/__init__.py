"""TLX op library.

    from triton.tlx.ops import mm as tlx_mm
    c = tlx_mm(a, b)

This module is the API contract; everything under it is private -- reaching into
`triton.tlx.ops.kernels.*` is not supported. Exactly one implementation ships
per (op, arch), so there is no `variant=` argument, and architecture never
appears in caller code.

Two keyword-only overrides exist for testing and benchmarking, both defaulting
to the plain behaviour: `arch=` pins an entry instead of detecting one, and
`space=` selects the autotune search space ("full" for perf, "heuristic" or
"smoke" for correctness, where the kernel offers them).

An op with no implementation for the current GPU raises `UnsupportedOp` -- it
never falls back to torch.
"""

from __future__ import annotations

from ._catalog import InvalidInput, UnsupportedOp, check_inputs, impl_for

__all__ = ["mm", "flash_attn", "hstu_attn", "kimi_delta_attention", "UnsupportedOp", "InvalidInput"]


def mm(a, b, *, arch=None, space="full"):
    """`a @ b`, for `(M, K) @ (K, N)` fp16/bf16. Either operand may be column-major."""
    fn, spec = impl_for("mm", arch)
    # Mirror the kernel's operand prep: a non-contiguous operand is fed to its
    # descriptor transposed, so that is the stride TMA must find aligned.
    a_src = a if a.is_contiguous() else a.T
    b_src = b if b.is_contiguous() else b.T
    check_inputs(spec, dtype=a.dtype, row_strides=(a_src.stride(0), b_src.stride(0), b.shape[1]),
                 elem_bytes=a.element_size())
    return fn(a, b, space=space)


def flash_attn(q, k, v, causal=False, sm_scale=None, *, arch=None, space="full"):
    """Fused attention over `(Z, H, N_CTX, HEAD_DIM)` fp16/bf16. Differentiable.

    `sm_scale` defaults to `HEAD_DIM ** -0.5`.
    """
    fn, spec = impl_for("flash_attn", arch)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1])
    return fn(q, k, v, causal, sm_scale, space=space)


def hstu_attn(q, k, v, seq_offsets, max_seq_len, attn_scale, alpha=None, causal=True, num_targets=None, max_attn_len=0,
              contextual_seq_len=0, *, arch=None, space="full"):
    """HSTU ragged attention over `(total_tokens, H, HEAD_DIM)` fp16/bf16. Differentiable.

    Scores are SiLU-scaled rather than softmaxed, which is why this is its own
    op. `seq_offsets` is `(B + 1,)` prefix offsets; `alpha` defaults to
    `1 / HEAD_DIM`.
    """
    fn, spec = impl_for("hstu_attn", arch)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1])
    return fn(q, k, v, seq_offsets, max_seq_len, alpha if alpha is not None else 1.0 / q.shape[-1], causal=causal,
              attn_scale=attn_scale, num_targets=num_targets, max_attn_len=max_attn_len,
              contextual_seq_len=contextual_seq_len, space=space)


def kimi_delta_attention(q, k, v, g, beta, *, scale=1.0, cu_seqlens=None, cu_seqlens_cpu=None, arch=None, space="full"):
    """Kimi Delta Attention over packed `[1, T, H, 128]` fp16/bf16 inputs.

    Returns the TritonBench-compatible `(output, None)` pair.
    """
    fn, spec = impl_for("kimi_delta_attention", arch)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1])
    return fn(q, k, v, g, beta, scale=scale, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, space=space)
