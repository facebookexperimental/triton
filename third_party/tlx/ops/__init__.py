"""TLX op library.

    from triton.tlx.ops import mm as tlx_mm
    c = tlx_mm(a, b)

This module is the API contract; everything under it is private -- reaching into
`triton.tlx.ops.kernels.*` is not supported. Exactly one implementation ships
per (op, arch), so there is no `variant=` argument, and architecture never
appears in caller code.

Two keyword-only overrides exist for testing and benchmarking: `arch=` pins an
entry instead of detecting one, and `space=` selects an implementation-defined
autotune search space. Passing a space that the selected implementation does
not provide raises `InvalidInput` rather than silently choosing another space.

`space=` defaults to "heuristic" -- a single config chosen analytically -- for
any op that offers one, so that a first call stays interactive. Measured on
B200, `mm` at `space="full"` takes 221-285s on a cold Triton cache (348 configs
compiled and benchmarked for a 1024x1024x1024 product) and also accumulates
tens of GB of autotune workspaces; at "heuristic" the same call is under a
second. On implementations that provide it, pass `space="full"` explicitly to
buy back the tuned configs, which are worth up to ~4x on small shapes. The
gfx950 LocalSplitU implementation currently provides only `"heuristic"`.

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

__all__ = ["mm", "flash_attn", "hstu_attn_dev", "kimi_delta_attention", "kda_paged_prefill", "kda_recurrent_decode",
           "UnsupportedOp", "InvalidInput"]


def mm(a, b, *, arch=None, space="heuristic"):
    """`a @ b`, for `(M, K) @ (K, N)` fp16/bf16. Either operand may be column-major.

    Defaults to a single analytically chosen config so the first call stays
    interactive. Pass `space="full"` to implementations that expose a full
    autotune space; unsupported spaces raise `InvalidInput`. See the module
    docstring.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise InvalidInput(
            "tlx.ops.mm expects two rank-2 tensors; "
            f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
        )
    if a.shape[1] != b.shape[0]:
        raise InvalidInput(
            "tlx.ops.mm reduction dimensions must match; "
            f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
        )
    if a.dtype != b.dtype or a.device != b.device:
        raise InvalidInput(
            "tlx.ops.mm operands must have the same dtype and device; "
            f"got a=({a.dtype}, {a.device}), b=({b.dtype}, {b.device})"
        )
    fn, spec = impl_for("mm", arch)
    # Mirror the kernel's operand prep: a non-contiguous operand is fed to its
    # descriptor transposed, so that is the stride TMA must find aligned.
    a_src = a if a.is_contiguous() else a.T
    b_src = b if b.is_contiguous() else b.T
    check_inputs(spec, dtype=a.dtype, M=a.shape[0], N=b.shape[1], K=a.shape[1],
                 row_strides=(a_src.stride(0), b_src.stride(0), b.shape[1]), elem_bytes=a.element_size())
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


def kda_paged_prefill(q, k, v, g, beta, *, scale=1.0, initial_state, cu_seqlens, arch=None):
    """Prepared-input chunked KDA prefill for packed `[1, T, H, 128]` tensors.

    `g` contains per-channel log decays and `beta` is sigmoid-applied.
    State is FP32 and V-major: `[N, H, 128, 128]`.
    """
    fn, spec = impl_for("kda_paged_prefill", arch)
    check_inputs(spec, dtype=q.dtype, KEY_DIM=q.shape[-1], VALUE_DIM=v.shape[-1])
    return fn(q, k, v, g, beta, scale=scale, initial_state=initial_state, cu_seqlens=cu_seqlens)


def kda_recurrent_decode(q, k, v, g, beta, *, scale=1.0, state_pool, read_indices, write_indices, cu_seqlens,
                         arch=None):
    """Prepared-input indexed KDA recurrence over an FP32 V-major state pool."""
    fn, spec = impl_for("kda_recurrent_decode", arch)
    check_inputs(spec, dtype=q.dtype, KEY_DIM=q.shape[-1], VALUE_DIM=v.shape[-1])
    return fn(q, k, v, g, beta, scale=scale, state_pool=state_pool, read_indices=read_indices,
              write_indices=write_indices, cu_seqlens=cu_seqlens)
