"""TLX op library.

    from triton.tlx.ops import mm as tlx_mm
    c = tlx_mm(a, b)

This module is the API contract; everything under it is private -- reaching into
`triton.tlx.ops.kernels.*` is not supported. Exactly one implementation ships
per (op, arch), so there is no `variant=` argument, and architecture never
appears in caller code.

Two keyword-only overrides exist for testing and benchmarking: `arch=` pins an
entry instead of detecting one, and `space=` selects the autotune search space.

`space=` defaults to "heuristic" -- a single config chosen analytically -- for
any op that offers one, so that a first call stays interactive. Measured on
B200, `mm` at `space="full"` takes 221-285s on a cold Triton cache (348 configs
compiled and benchmarked for a 1024x1024x1024 product) and also accumulates
tens of GB of autotune workspaces; at "heuristic" the same call is under a
second. Pass `space="full"` explicitly to buy back the tuned configs, which are
worth up to ~4x on small shapes.

Ops with no heuristic yet -- flash_attn, hstu_attn, kimi_delta_attention --
still default to "full". Their remaining space is "smoke", which selects for
lowering-path coverage rather than speed, so defaulting to it would quietly
ship a bad config. Each needs its own `heuristic_config` before it can follow
`mm`.

An op with no implementation for the current GPU raises `UnsupportedOp` -- it
never falls back to torch.
"""

from __future__ import annotations

from ._catalog import InvalidInput, UnsupportedOp, check_inputs, impl_for

__all__ = ["mm", "flash_attn", "hstu_attn_dev", "kimi_delta_attention", "UnsupportedOp", "InvalidInput"]


def mm(a, b, *, arch=None, space="heuristic"):
    """`a @ b`, for `(M, K) @ (K, N)` fp16/bf16. Either operand may be column-major.

    Defaults to a single analytically chosen config so the first call stays
    interactive; pass `space="full"` to autotune. See the module docstring.
    """
    from .kernels.mm._layout import descriptor_layout

    fn, spec = impl_for("mm", arch)
    if a.ndim != 2 or b.ndim != 2:
        raise InvalidInput(f"{spec} expects two 2D tensors, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise InvalidInput(f"{spec} has incompatible dimensions {tuple(a.shape)} and {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise InvalidInput(f"{spec} expects matching dtypes, got {a.dtype} and {b.dtype}")
    if a.device != b.device:
        raise InvalidInput(f"{spec} expects tensors on the same device, got {a.device} and {b.device}")

    if spec.accepts is not None:
        try:
            a_layout = descriptor_layout(a, "a")
            b_layout = descriptor_layout(b, "b")
        except ValueError as exc:
            raise InvalidInput(f"{spec} does not support these inputs: {exc}") from exc
        check_inputs(spec, dtype=a.dtype, row_strides=(a_layout.row_stride, b_layout.row_stride, b.shape[1]),
                     elem_bytes=a.element_size())
    else:
        check_inputs(spec, dtype=a.dtype)
    return fn(a, b, space=space)


def flash_attn(q, k, v, causal=False, sm_scale=None, *, arch=None, space="full"):
    """Fused attention over `(Z, H, N_CTX, HEAD_DIM)` fp16/bf16. Differentiable.

    `sm_scale` defaults to `HEAD_DIM ** -0.5`.
    """
    fn, spec = impl_for("flash_attn", arch)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1])
    return fn(q, k, v, causal, sm_scale, space=space)


def hstu_attn_dev(q, k, v, seq_offsets, max_seq_len, attn_scale, alpha=None, causal=True, num_targets=None,
                  max_attn_len=0, contextual_seq_len=0, *, arch=None, space="full"):
    """HSTU ragged attention over `(total_tokens, H, HEAD_DIM)` fp16/bf16. Differentiable.

    Scores are SiLU-scaled rather than softmaxed, which is why this is its own
    op. `seq_offsets` is `(B + 1,)` prefix offsets; `alpha` defaults to
    `1 / HEAD_DIM`.

    Causal-only: `causal=False` raises `InvalidInput`. The argument is kept so
    the intent is stated at the call site rather than assumed.
    """
    fn, spec = impl_for("hstu_attn_dev", arch)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1], causal=causal)
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
