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

`space=` also accepts a `Space`, which bundles the search-space name with a
`BitwiseInvarianceSpec` naming the logical dims that may vary across calls
while overlapping outputs stay bitwise identical. Unknown dim names raise
`InvalidInput` before dispatch; a non-empty spec on an op that does not
enforce it yet raises `UnsupportedOp` rather than being silently ignored.
"""

from __future__ import annotations

import functools

from ._catalog import (
    InvalidInput,
    UnsupportedOp,
    any_enforces_invariance,
    check_inputs,
    impl_for,
    invariant_vocab_for,
)
from ._invariance import BitwiseInvarianceSpec, Space, normalize_space

__all__ = [
    "mm", "flash_attn", "hstu_attn", "UnsupportedOp", "InvalidInput", "BitwiseInvarianceSpec", "Space"
]


def _normalize_space_arg(op, space):
    """Split `space=` into (search-space name, resolved invariance spec).

    Unknown dim names raise `InvalidInput` here, before dispatch. A non-empty
    spec raises `UnsupportedOp` here when no arch entry enforces it yet; ops
    with partial enforcement re-check the dispatched entry afterwards.
    """
    name, spec = normalize_space(space)
    resolved = spec.resolve(invariant_vocab_for(op))
    if resolved.needs_bitwise() and not any_enforces_invariance(op):
        raise UnsupportedOp(f"tlx.ops.{op} does not enforce bitwise_invariance_spec yet")
    return name, resolved


def _check_enforced(op, entry, invariance):
    """Refuse an invariance request the dispatched entry does not enforce."""
    if invariance.needs_bitwise() and not entry.enforces_invariance:
        raise UnsupportedOp(f"tlx.ops.{op} does not enforce bitwise_invariance_spec yet on {entry.arch}")


@functools.lru_cache(maxsize=None)
def _mm_impl_for_device(device):
    """Cache the stable device-to-implementation lookup on the MM hot path."""
    return impl_for("mm", device=device)


def mm(a, b, *, out=None, arch=None, space=None):
    """`a @ b`, for `(M, K) @ (K, N)` fp16/bf16. Either operand may be column-major.

    `space=None` uses the implementation's default: a single analytically
    chosen config so the first call stays interactive. `space=` also accepts
    a `Space` carrying a `BitwiseInvarianceSpec` over the M/N/K vocabulary.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise InvalidInput("tlx.ops.mm expects two rank-2 tensors; "
                           f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise InvalidInput("tlx.ops.mm reduction dimensions must match; "
                           f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    if a.dtype != b.dtype or a.device != b.device:
        raise InvalidInput("tlx.ops.mm operands must have the same dtype and device; "
                           f"got a=({a.dtype}, {a.device}), b=({b.dtype}, {b.device})")
    space_name, invariance = _normalize_space_arg("mm", space)
    if arch is None:
        fn, spec = _mm_impl_for_device(a.device)
    else:
        fn, spec = impl_for("mm", arch)
    # Mirror the kernel's operand prep: a non-contiguous operand is fed to its
    # descriptor transposed, so that is the stride TMA must find aligned.
    a_src = a if a.is_contiguous() else a.T
    b_src = b if b.is_contiguous() else b.T
    check_inputs(spec, dtype=a.dtype, row_strides=(a_src.stride(0), b_src.stride(0), b.shape[1]),
                 elem_bytes=a.element_size())
    _check_enforced("mm", spec, invariance)
    if space_name is None:
        space_name = spec.default_space
    if out is None:
        return fn(a, b, space=space_name)
    return fn(a, b, out=out, space=space_name)


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
