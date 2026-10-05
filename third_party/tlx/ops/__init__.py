"""TLX op library.

    from triton.tlx.ops import mm as tlx_mm
    c = tlx_mm(a, b)

    from triton.tlx.ops import addmm as tlx_addmm
    c = tlx_addmm(bias, a, b)

This module is the API contract; everything under it is private -- reaching into
`triton.tlx.ops.kernels.*` is not supported. Exactly one implementation ships
per (op, arch), so there is no `variant=` argument, and architecture never
appears in caller code.

Dispatch derives the architecture from the input tensor's device. If that
architecture cannot be determined, or the op has no implementation for it, the
call raises `UnsupportedOp`. The `space=` keyword selects an
implementation-defined autotune search space. Passing a space that the selected
implementation does not provide raises `InvalidInput` rather than silently
choosing another space. `mm` and `addmm` also accept `out=` for implementations
that support a preallocated output.

`space=` defaults to "heuristic" -- a single config chosen analytically -- for
any op that offers one, so that a first call stays interactive. Measured on
B200, `mm` at `space="full"` takes 221-285s on a cold Triton cache (348 configs
compiled and benchmarked for a 1024x1024x1024 product) and also accumulates
tens of GB of autotune workspaces; at "heuristic" the same call is under a
second. On implementations that provide it, pass `space="full"` explicitly to
buy back the tuned configs, which are worth up to ~4x on small shapes. The
gfx950 implementation currently provides only `"heuristic"`.

Ops with no heuristic yet -- flash_attn, flash_attn_mxfp8, hstu_attn,
kimi_delta_attention -- still default to "full". Their remaining space is "smoke", which selects for
lowering-path coverage rather than speed, so defaulting to it would quietly
ship a bad config. Each needs its own `heuristic_config` before it can follow
`mm`.

An op with no implementation for the current GPU raises `UnsupportedOp` -- it
never falls back to torch. A forward-only implementation raises
`UnsupportedBackward` before launch when autograd is enabled and any supported
tensor input requires gradients; inference under `torch.no_grad()` is allowed.
"""

from __future__ import annotations

from ._catalog import InvalidInput, UnsupportedBackward, UnsupportedOp, check_backward, check_inputs, impl_for

__all__ = [
    "mm", "grouped_gemm", "addmm", "flash_attn", "flash_attn_mxfp8", "hstu_attn_dev", "kimi_delta_attention",
    "kda_paged_prefill", "kda_recurrent_decode", "UnsupportedOp", "UnsupportedBackward", "InvalidInput"
]


def mm(a, b, *, out=None, space="heuristic"):
    """`a @ b`, for `(M, K) @ (K, N)` fp16/bf16. Either operand may be column-major.

    This op is currently forward-only.

    Defaults to a single analytically chosen config so the first call stays
    interactive. Pass `space="full"` to implementations that expose a full
    autotune space; unsupported spaces raise `InvalidInput`. See the module
    docstring.
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
    fn, spec = impl_for("mm", device=a.device)
    if spec.accepts is None:
        check_inputs(spec, dtype=a.dtype)
    else:
        # Mirror the kernel's operand prep: a non-contiguous operand is fed to
        # its descriptor transposed, so that is the stride TMA must find aligned.
        a_src = a if a.is_contiguous() else a.T
        b_src = b if b.is_contiguous() else b.T
        check_inputs(spec, dtype=a.dtype, M=a.shape[0], N=b.shape[1], K=a.shape[1],
                     row_strides=(a_src.stride(0), b_src.stride(0), b.shape[1]), elem_bytes=a.element_size())
    check_backward(spec, a, b)
    if out is None:
        return fn(a, b, space=space)
    return fn(a, b, out=out, space=space)


def grouped_gemm(group_a, group_b):
    """Run a ragged group of FP16 ``A[i] @ B[i]`` products.

    ``A[i]`` must be rank-2 row-major. ``B[i]`` must be a rank-2 column-major
    view with K contiguous. The operation is currently forward-only;
    architecture-specific alignment restrictions may also apply.
    """
    import torch

    if not isinstance(group_a, (list, tuple)) or not isinstance(group_b, (list, tuple)):
        raise InvalidInput("tlx.ops.grouped_gemm expects group_a and group_b to be lists or tuples")
    group_a = tuple(group_a)
    group_b = tuple(group_b)
    if len(group_a) != len(group_b):
        raise InvalidInput("tlx.ops.grouped_gemm operand groups must have the same length; "
                           f"got len(group_a)={len(group_a)}, len(group_b)={len(group_b)}")
    if not group_a:
        raise InvalidInput("tlx.ops.grouped_gemm requires at least one matrix pair")

    for index, (a, b) in enumerate(zip(group_a, group_b)):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise InvalidInput("tlx.ops.grouped_gemm expects tensor elements; "
                               f"pair {index} has a={type(a).__name__}, b={type(b).__name__}")
        if a.ndim != 2 or b.ndim != 2:
            raise InvalidInput("tlx.ops.grouped_gemm expects rank-2 tensors; "
                               f"pair {index} has a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
        if any(dim <= 0 for dim in (*a.shape, *b.shape)):
            raise InvalidInput("tlx.ops.grouped_gemm dimensions must be positive; "
                               f"pair {index} has a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
        if a.shape[1] != b.shape[0]:
            raise InvalidInput("tlx.ops.grouped_gemm reduction dimensions must match; "
                               f"pair {index} has a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
        if a.stride() != (a.shape[1], 1):
            raise InvalidInput("tlx.ops.grouped_gemm requires row-major A tensors; "
                               f"pair {index} has a.stride={a.stride()}")
        if b.stride() != (1, b.shape[0]):
            raise InvalidInput("tlx.ops.grouped_gemm requires column-major B tensors; "
                               f"pair {index} has b.stride={b.stride()}")

    dtype = group_a[0].dtype
    device = group_a[0].device
    for index, tensor in enumerate((*group_a, *group_b)):
        if tensor.dtype != dtype or tensor.device != device:
            raise InvalidInput("tlx.ops.grouped_gemm operands must share a dtype and device; "
                               f"tensor {index} is ({tensor.dtype}, {tensor.device}), expected ({dtype}, {device})")

    fn, spec = impl_for("grouped_gemm", device=device)
    # TMA implementations describe A, the zero-copy B.T view, and C as
    # row-major tensors. Direct-load implementations ignore these facts.
    row_strides = tuple(stride for a, b in zip(group_a, group_b) for stride in (a.stride(0), b.stride(1), b.shape[1]))
    base_ptrs = tuple(tensor.data_ptr() for pair in zip(group_a, group_b) for tensor in pair)
    check_inputs(
        spec,
        dtype=dtype,
        row_strides=row_strides,
        base_ptrs=base_ptrs,
        elem_bytes=group_a[0].element_size(),
    )
    check_backward(spec, *group_a, *group_b)
    return fn(group_a, group_b)


def addmm(input, a, b, *, out=None, space="heuristic"):
    """Fused ``input + a @ b`` for two-dimensional fp16/bf16 matrices.

    ``input`` may be ``(N,)`` or two-dimensional and broadcastable to the
    ``(M, N)`` result. Matrix and input scale factors are both one. This op is
    currently forward-only.
    """
    fn, spec = impl_for("addmm", device=a.device)
    check_inputs(spec, dtype=a.dtype)
    check_backward(spec, input, a, b)
    return fn(input, a, b, out=out, space=space)


def flash_attn(q, k, v, causal=False, sm_scale=None, *, space="full"):
    """Fused attention over `(Z, H, N_CTX, HEAD_DIM)` fp16/bf16. Differentiable.

    `sm_scale` defaults to `HEAD_DIM ** -0.5`.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise InvalidInput("tlx.ops.flash_attn expects rank-4 Q/K/V tensors")
    if space not in ("full", "smoke"):
        raise InvalidInput(f"tlx.ops.flash_attn does not provide space={space!r}")
    fn, spec = impl_for("flash_attn", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1])
    check_backward(spec, q, k, v)
    # Triton's launcher uses the process' current device rather than deriving it
    # from pointer arguments. Keep compilation and launch on the input device;
    # the context manager restores the caller's current device afterwards.
    import torch
    with torch.cuda.device(q.device):
        return fn(q, k, v, causal, sm_scale, space=space)


def flash_attn_mxfp8(q, k, v, causal=False, sm_scale=None, *, space="full"):
    """MXFP8 attention over contiguous BF16 ``(Z, H, N_CTX, HEAD_DIM)`` tensors.

    Q, K, V and the softmax probabilities are quantized internally to E4M3
    data with E8M0 scales: per 32x32 block for Q and K, per 32 keys for V and
    the probabilities. gfx950 quantizes as Blackwell does, but its non-causal
    kernel computes the probabilities with an approximate exp2 (see
    kernels/flash_attn_mxfp8/gfx950.py), and the two agree to within the
    quantization error rather than bitwise. Head dim 128 and sequence lengths
    divisible by 256. Blackwell returns BF16 gradients; gfx950 is forward only.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects rank-4 Q/K/V tensors")
    if q.shape != k.shape or q.shape != v.shape:
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects Q/K/V to have identical shapes; "
                           f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.device != k.device or q.device != v.device:
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects Q/K/V to have the same dtype and device")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects contiguous Q/K/V tensors")
    if space not in ("full", "smoke"):
        raise InvalidInput(f"tlx.ops.flash_attn_mxfp8 does not provide space={space!r}")
    fn, spec = impl_for("flash_attn_mxfp8", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1], N_CTX=q.shape[-2])
    check_backward(spec, q, k, v)
    return fn(q, k, v, causal, sm_scale, space=space)


def hstu_attn_dev(q, k, v, seq_offsets, max_seq_len, attn_scale, alpha=None, causal=True, num_targets=None,
                  max_attn_len=0, contextual_seq_len=0, *, space="full"):
    """HSTU ragged attention over `(total_tokens, H, HEAD_DIM)` fp16/bf16. Differentiable.

    Scores are SiLU-scaled rather than softmaxed, which is why this is its own
    op. `seq_offsets` is `(B + 1,)` prefix offsets; `alpha` defaults to
    `1 / HEAD_DIM`.

    Causal-only: `causal=False` raises `InvalidInput`. The argument is kept so
    the intent is stated at the call site rather than assumed.
    """
    fn, spec = impl_for("hstu_attn_dev", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1], causal=causal)
    check_backward(spec, q, k, v)
    return fn(q, k, v, seq_offsets, max_seq_len, alpha if alpha is not None else 1.0 / q.shape[-1], causal=causal,
              attn_scale=attn_scale, num_targets=num_targets, max_attn_len=max_attn_len,
              contextual_seq_len=contextual_seq_len, space=space)


def kimi_delta_attention(q, k, v, g, beta, *, scale=1.0, cu_seqlens=None, cu_seqlens_cpu=None, space="full"):
    """Kimi Delta Attention over packed `[1, T, H, 128]` fp16/bf16 inputs.

    Returns the TritonBench-compatible `(output, None)` pair.
    """
    fn, spec = impl_for("kimi_delta_attention", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1])
    check_backward(spec, q, k, v, g, beta)
    return fn(q, k, v, g, beta, scale=scale, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, space=space)


def kda_paged_prefill(q, k, v, g, beta, *, scale=1.0, initial_state, cu_seqlens):
    """Prepared-input chunked KDA prefill for packed `[1, T, H, 128]` tensors.

    `g` contains per-channel log decays and `beta` is sigmoid-applied.
    State is FP32 and V-major: `[N, H, 128, 128]`. This op is currently
    forward-only.
    """
    fn, spec = impl_for("kda_paged_prefill", device=q.device)
    check_inputs(spec, dtype=q.dtype, KEY_DIM=q.shape[-1], VALUE_DIM=v.shape[-1])
    check_backward(spec, q, k, v, g, beta, initial_state)
    return fn(q, k, v, g, beta, scale=scale, initial_state=initial_state, cu_seqlens=cu_seqlens)


def kda_recurrent_decode(q, k, v, g, beta, *, scale=1.0, state_pool, read_indices, write_indices, cu_seqlens):
    """Forward-only indexed KDA recurrence over an FP32 V-major state pool."""
    fn, spec = impl_for("kda_recurrent_decode", device=q.device)
    check_inputs(spec, dtype=q.dtype, KEY_DIM=q.shape[-1], VALUE_DIM=v.shape[-1])
    check_backward(spec, q, k, v, g, beta, state_pool)
    return fn(q, k, v, g, beta, scale=scale, state_pool=state_pool, read_indices=read_indices,
              write_indices=write_indices, cu_seqlens=cu_seqlens)
