"""TLX op library.

    from triton.tlx.ops import mm as tlx_mm
    c = tlx_mm(a, b)

    from triton.tlx.ops import addmm as tlx_addmm
    c = tlx_addmm(bias, a, b)

    from triton.tlx.ops import addmm_mul_add
    out = addmm_mul_add(bias, a, b, x0, res)

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
most ops that offer one, so that a first call stays interactive. `mm` on sm90
and sm100 defaults to "full" instead: sm90's full space prunes to 8-28 configs per shape, and
on sm100 the tuned configs are worth up to ~4x on small shapes and beat the
heuristic on skinny, long-K shapes. The cost on sm100 is the first call per
shape: on B200, `mm` at `space="full"` takes 221-285s on a cold Triton cache
(348 configs compiled and benchmarked for a 1024x1024x1024 product) and also
accumulates tens of GB of autotune workspaces; at "heuristic" the same call is
under a second. Pass `space="heuristic"` explicitly for a fast first call. The
gfx950 implementation currently provides only `"heuristic"`.

Ops with no heuristic yet -- flash_attn on sm90/sm100/gfx950, flash_attn_mxfp8, hstu_attn,
kimi_delta_attention -- still default to "full". Their remaining space is "smoke", which selects for
lowering-path coverage rather than speed, so defaulting to it would quietly
ship a bad config. Each needs its own `heuristic_config` before it can offer a
fast first call. `mm_mxfp` defaults to "full" on sm100 because its tuned configs are measurably
faster; gfx950 uses "heuristic". Pass `space="heuristic"` for a fast first call.
`mm_mxfp6` also defaults to "full" because its tuned configs are measurably faster.
`hstu_cross_attn` on gfx950 defaults to "curated": at head dims 64 and 128 it
tunes eight forward configs picked from its 144-config "full" grid, which come
within a few percent of it; other head dims tune the whole grid.

An op with no implementation for the current GPU raises `UnsupportedOp` -- it
never falls back to torch. A forward-only implementation raises
`UnsupportedBackward` before launch when autograd is enabled and any supported
tensor input requires gradients; inference under `torch.no_grad()` is allowed.

`prepare_flash_attn_varlen_backward` and `flash_attn_varlen_backward` expose
explicit first-order attention gradients from an existing forward result.
Prepare the immutable sequence plan once, then reuse it for backward calls;
this explicit API accepts inputs requiring gradients but does not build an
autograd graph for higher-order derivatives.

`space=` also accepts a `Space`, which bundles the search-space name with a
`BitwiseInvarianceSpec` naming the logical dims that may vary across calls
while overlapping outputs stay bitwise identical. Unknown dim names raise
`InvalidInput` before dispatch; a non-empty spec on an op that does not
enforce it yet raises `UnsupportedOp` rather than being silently ignored.
"""

from __future__ import annotations

import contextlib
import functools
import math

from torch.compiler import assume_constant_result

from ._catalog import (
    InvalidInput,
    UnsupportedBackward,
    UnsupportedOp,
    any_enforces_invariance,
    check_backward,
    check_inputs,
    impl_for,
    invariant_vocab_for,
    _target,
)
from ._invariance import BitwiseInvarianceSpec, Space, normalize_space
from ._normalization import (
    compute_norm_nvfp4_global_scale,
    layernorm,
    layernorm_add,
    layernorm_backward,
    layernorm_backward_from_x_norm,
    layernorm_mxfp4,
    layernorm_mxfp8_gdpa_precomputed_no_normalized,
    layernorm_mxfp8_gdpa_with_backward,
    layernorm_mxfp8_quantize,
    layernorm_mxfp8_quantize_numerics_preserving,
    layernorm_mxfp8_with_backward,
    layernorm_nvfp4_backward_from_x_norm_padded_dy,
    layernorm_nvfp4_quantize,
    layernorm_nvfp4_quantize_inplace,
    layernorm_nvfp4_quantize_q0_jagged_transpose,
    layernorm_nvfp4_quantize_q0_jagged_transpose_inplace,
    norm_backward_from_normalized,
    norm_bwd,
    norm_fwd,
    norm_quant_mxfp8_fwd,
    nvfp4_quantize_normalized,
    rmsnorm,
    rmsnorm_backward,
    rmsnorm_backward_from_x_norm,
    rmsnorm_backward_quantize_dx_mxfp8,
    rmsnorm_backward_quantize_dx_mxfp8_bf16_precomputed,
    rmsnorm_backward_quantize_dx_mxfp8_from_output_recompute_q_precomputed,
    rmsnorm_backward_quantize_dx_mxfp8_precomputed,
    rmsnorm_mxfp4,
    rmsnorm_mxfp8_quantize,
    rmsnorm_mxfp8_quantize_3232_2pass,
    rmsnorm_nvfp4_quantize,
    softmax,
    softmax_bwd,
    softmax_fwd,
)

__all__ = [
    "mm", "mm_bias_grad", "int4_embedding", "mm_mxfp", "mm_mxfp6", "grouped_gemm", "grouped_gemm_mxfp8", "moe",
    "moe_routing", "moe_finalize", "moe_router_topk_align", "moe_mxfp4_sort_quant", "norm_residual_norm", "all_reduce",
    "all_reduce_workspace", "all_reduce_norm_residual_norm", "rmsnorm_gate", "router_gemm", "qk_norm_rope_kv_cache",
    "addmm", "addmm_mul_add", "addmm_mul_add_launch", "flash_attn", "flash_attn_mxfp8", "flash_attn_varlen",
    "linear_rowscale", "linear_rowscale_launch", "window_attn_varlen", "rmsnorm_cat", "swiglu_ffn", "mla_window_attn",
    "hstu_attn_dev", "hstu_cross_attn", "kimi_delta_attention", "kda_paged_prefill", "kda_recurrent_decode",
    "prepare_flash_attn_varlen_backward", "flash_attn_varlen_backward", "paged_attention_decode",
    "paged_attention_prefill", "paged_attention_decode_workspace", "allocate_paged_kv_cache_5d", "reshape_and_cache_5d",
    "UnsupportedOp", "UnsupportedBackward", "InvalidInput", "BitwiseInvarianceSpec", "Space"
]
__all__ += [
    'paged_attention_decode_fp8', 'paged_attention_decode_fp8_workspace', 'compute_norm_nvfp4_global_scale',
    'layernorm', 'layernorm_add', 'layernorm_backward', 'layernorm_backward_from_x_norm', 'layernorm_mxfp4',
    'layernorm_mxfp8_gdpa_precomputed_no_normalized', 'layernorm_mxfp8_gdpa_with_backward', 'layernorm_mxfp8_quantize',
    'layernorm_mxfp8_quantize_numerics_preserving', 'layernorm_mxfp8_with_backward',
    'layernorm_nvfp4_backward_from_x_norm_padded_dy', 'layernorm_nvfp4_quantize', 'layernorm_nvfp4_quantize_inplace',
    'layernorm_nvfp4_quantize_q0_jagged_transpose', 'layernorm_nvfp4_quantize_q0_jagged_transpose_inplace',
    'norm_backward_from_normalized', 'norm_bwd', 'norm_fwd', 'norm_quant_mxfp8_fwd', 'nvfp4_quantize_normalized',
    'rmsnorm', 'rmsnorm_backward', 'rmsnorm_backward_from_x_norm', 'rmsnorm_backward_quantize_dx_mxfp8',
    'rmsnorm_backward_quantize_dx_mxfp8_bf16_precomputed',
    'rmsnorm_backward_quantize_dx_mxfp8_from_output_recompute_q_precomputed',
    'rmsnorm_backward_quantize_dx_mxfp8_precomputed', 'rmsnorm_mxfp4', 'rmsnorm_mxfp8_quantize',
    'rmsnorm_mxfp8_quantize_3232_2pass', 'rmsnorm_nvfp4_quantize', 'softmax', 'softmax_bwd', 'softmax_fwd'
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


def _check_single_config_space(op, space_name):
    """Reject a search-space name on an op with a single fixed config.

    Such ops accept only `space=None` (implementation default) or the
    canonical single-config scope `"heuristic"` (also the `Space` carrier
    name for an invariance request). Anything else names a tuning space
    the op does not provide and must not be silently ignored.
    """
    if space_name is not None and space_name != "heuristic":
        raise InvalidInput(f"tlx.ops.{op} does not provide space={space_name!r}")


@functools.lru_cache(maxsize=None)
def _mm_impl_for_device(device):
    """Cache the stable device-to-implementation lookup on the MM hot path."""
    return impl_for("mm", device=device)


@functools.lru_cache(maxsize=None)
def _flash_attn_varlen_impl_for_device(device):
    """Cache the device-to-implementation lookup, which otherwise re-queries device properties on every call."""
    return impl_for("flash_attn_varlen", device=device)


def mm(a, b, *, out=None, space=None, b_static=False):
    """`a @ b`, for `(M, K) @ (K, N)` fp16/bf16. Either operand may be column-major.

    This op is currently forward-only.

    `space=None` uses the implementation's default: "full" on sm90 and sm100,
    otherwise a single analytically chosen config so the first call stays
    interactive. Pass `space="heuristic"` for a fast first call on sm100.
    Pass `space="full"` to implementations that expose a full autotune space;
    unsupported spaces raise `InvalidInput`. See the module docstring.
    `space=` also accepts a `Space` carrying a `BitwiseInvarianceSpec` over
    the M/N/K vocabulary.

    `b_static=True` promises that no kernel that may still be running when this
    op launches writes `b` (a model weight, for instance). Implementations may
    then start reading `b` before waiting on earlier kernels; without it they wait
    first. It never changes the result of a call that keeps the promise.
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
    fn, spec = _mm_impl_for_device(a.device)
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
    _check_enforced("mm", spec, invariance)
    if space_name is None:
        space_name = spec.default_space
    promise = {"b_static": True} if b_static else {}
    if out is None:
        return fn(a, b, space=space_name, invariance=invariance, **promise)
    return fn(a, b, out=out, space=space_name, invariance=invariance, **promise)


def int4_embedding(weights, weights_offsets, indices, offsets, D, *, out=None, space=None):
    """Return the FP16 rows ``indices`` select from INT4 tables, without pooling.

    ``weights`` is the contiguous uint8 buffer of all tables, ``weights_offsets``
    the int64 byte offset of each table, each row ``[fp16 scale, fp16 shift]``
    and ``D / 2`` bytes of 4-bit values padded to a multiple of 16 bytes; within
    each 32-bit word, nibble ``j`` holds column ``8 * w + [0, 4, 2, 6, 1, 5, 3, 7][j]``.
    ``indices`` and ``offsets`` are int32, with
    ``T * B + 1`` offsets for ``T`` tables of ``B`` bags. Returns a
    ``[len(indices), D]`` FP16 tensor; row ``p`` is
    ``fma(q, scale, shift)`` of row ``indices[p]`` of its table, rounded once,
    and zeros where the index is negative.
    ``out`` may supply that tensor. Only ``D = 64`` and Hopper are supported.
    Forward only.
    """
    tensors = (weights, weights_offsets, indices, offsets)
    if any(t.ndim != 1 for t in tensors):
        raise InvalidInput("tlx.ops.int4_embedding expects rank-1 tensors")
    if any(t.device != weights.device for t in tensors):
        raise InvalidInput("tlx.ops.int4_embedding inputs must be on the same device")
    if (str(weights_offsets.dtype), str(indices.dtype), str(offsets.dtype)) != ("torch.int64", "torch.int32",
                                                                                "torch.int32"):
        raise InvalidInput("tlx.ops.int4_embedding needs int64 weights_offsets and int32 indices and offsets")
    if not all(t.is_contiguous() for t in tensors):
        raise InvalidInput("tlx.ops.int4_embedding requires contiguous inputs")
    _check_single_config_space("int4_embedding", space)
    fn, spec = impl_for("int4_embedding", device=weights.device)
    check_inputs(spec, dtype=weights.dtype, D=D, base_ptrs=tuple(t.data_ptr() for t in tensors))
    return fn(weights, weights_offsets, indices, offsets, D, out=out, space=spec.default_space)


def mm_bias_grad(dout, input_act, *, out=None, space=None):
    """Return ``(dout.T @ input_act, dout.sum(0))`` for a linear layer.

    Inputs are contiguous BF16 ``[batch, out_features]`` and
    ``[batch, in_features]`` tensors on the same CUDA device. Both feature
    dimensions must be multiples of eight; batch may have a partial K tile.
    Outputs have the input dtype and shapes ``[out_features, in_features]``
    and ``[out_features]``. ``out`` may supply that pair of output tensors.

    These are explicit first-order gradients. Inputs may require gradients,
    but the returned tensors have no autograd graph for higher derivatives.
    ``space=None`` selects "heuristic"; "full" autotunes the fused operation,
    including the split-K reduction. Only Blackwell is currently supported.
    """
    if dout.ndim != 2 or input_act.ndim != 2:
        raise InvalidInput("tlx.ops.mm_bias_grad expects two rank-2 tensors")
    if dout.shape[0] != input_act.shape[0] or min(*dout.shape, *input_act.shape) <= 0:
        raise InvalidInput("tlx.ops.mm_bias_grad requires matching, positive batch dimensions")
    if dout.dtype != input_act.dtype or dout.device != input_act.device:
        raise InvalidInput("tlx.ops.mm_bias_grad inputs must have the same dtype and device")
    if not dout.is_contiguous() or not input_act.is_contiguous():
        raise InvalidInput("tlx.ops.mm_bias_grad requires contiguous inputs")
    fn, spec = impl_for("mm_bias_grad", device=dout.device)
    check_inputs(spec, dtype=dout.dtype, row_strides=(dout.stride(0), input_act.stride(0)),
                 elem_bytes=dout.element_size(), base_ptrs=(dout.data_ptr(), input_act.data_ptr()))
    return fn(dout, input_act, out=out, space=spec.default_space if space is None else space)


def _check_mm_mxfp_scales(a_scale, b_scale, *, M, N, K, sf_layout):
    if sf_layout == "natural":
        expected_a, expected_b = (M, K // 32), (N, K // 32)
        if a_scale.shape != expected_a or b_scale.shape != expected_b:
            raise InvalidInput("tlx.ops.mm_mxfp natural scales must have exact shapes "
                               f"a_scale={expected_a}, b_scale={expected_b}; got "
                               f"a_scale={tuple(a_scale.shape)}, b_scale={tuple(b_scale.shape)}")
        return
    if sf_layout == "mfma_preshuffled":
        expected_a = ((M + 127) // 128) * 128 * (K // 32)
        expected_b = (2 * N * (K // 32) if M <= 128 and N % 32 == 0 and K % 128 == 0 else
                      ((N + 127) // 128) * 128 * (K // 32) if (M, N, K) in {
                          (1024, 1024, 24576),
                          (1024, 8192, 3072),
                          (2048, 1024, 24576),
                          (2048, 32768, 6144),
                      } else ((N + 95) // 96) * 128 * (K // 32))
        if a_scale.numel() != expected_a or b_scale.numel() != expected_b:
            raise InvalidInput("tlx.ops.mm_mxfp mfma_preshuffled scales must contain exactly "
                               f"{expected_a} a-scale and {expected_b} b-scale bytes; "
                               f"got {a_scale.numel()} and {b_scale.numel()}")
        return
    if sf_layout != "cublas_blocked":
        raise InvalidInput("tlx.ops.mm_mxfp sf_layout must be 'natural', 'cublas_blocked', "
                           "or 'mfma_preshuffled'; "
                           f"got {sf_layout!r}")
    # Each 128x4 atom includes padded rows when M or N is ragged.
    expected_a = ((M + 127) // 128) * 128 * (K // 32)
    expected_b = ((N + 127) // 128) * 128 * (K // 32)
    if a_scale.numel() != expected_a or b_scale.numel() != expected_b:
        raise InvalidInput("tlx.ops.mm_mxfp cublas_blocked scales must contain exactly "
                           f"{expected_a} a-scale and {expected_b} b-scale bytes; "
                           f"got {a_scale.numel()} and {b_scale.numel()}")


def mm_mxfp(a, a_scale, b, b_scale, *, a_dtype="e4m3", b_dtype="e4m3", out=None, sf_layout="natural", space=None):
    """``a @ b.T`` over pre-quantized MX inputs, returning BF16 ``[M, N]``.

    ``a_dtype`` and ``b_dtype`` name the MX element formats; every backend supports only ``"e4m3"`` today.
    ``a`` is contiguous E4M3 ``[M, K]`` and ``b`` is contiguous E4M3 ``[N, K]``
    (K-major, as for a linear weight); K must be a multiple of 128.
    M and N may be ragged on gfx950; sm100 still requires 128-aligned rows,
    and gfx1250 requires 256-aligned M and N with K of at least 256.
    Each run of 32 values along K has one E8M0 scale. ``sf_layout="natural"``
    takes ``[M, K // 32]`` and ``[N, K // 32]`` scales; ``"cublas_blocked"``
    takes the swizzled byte layout returned by torchao's
    ``MXTensor.to_mx(..., is_swizzled_scales=True)`` and pads its final M atom
    to 128 rows. ``"mfma_preshuffled"`` accepts the packed XDL32 scale layout
    used by supported shape-selected FlyDSL configs: repeat-2 A / repeat-3 B
    for 128x192 tiles, repeat-4 A / repeat-3 B for 256x192 tiles, and repeat-2
    A / repeat-4 B for 128x256 tiles. Each 96-row B chunk in a 192-wide tile
    occupies 128 physical rows. A supplied ``out`` is returned by identity.
    Forward-only.

    ``space=None`` uses "full" on sm100 (autotuning) and "heuristic" on gfx950
    and gfx1250. "heuristic" launches one shape-picked config without
    autotuning; gfx950 and gfx1250 provide only this space. A ``Space`` is also
    accepted; its invariance request is validated but not enforced yet.
    """
    import torch

    if (a_dtype, b_dtype) != ("e4m3", "e4m3"):
        raise UnsupportedOp("tlx.ops.mm_mxfp supports only a_dtype='e4m3' and b_dtype='e4m3'; "
                            f"got a_dtype={a_dtype!r}, b_dtype={b_dtype!r}")
    tensors = (a, a_scale, b, b_scale)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise InvalidInput("tlx.ops.mm_mxfp expects tensor inputs")
    if a.ndim != 2 or b.ndim != 2:
        raise InvalidInput("tlx.ops.mm_mxfp expects rank-2 a and b; "
                           f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    M, K = a.shape
    N, b_k = b.shape
    if b_k != K:
        raise InvalidInput("tlx.ops.mm_mxfp reduction dimensions must match; "
                           f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    if min(M, N, K) <= 0 or K % 128:
        raise InvalidInput("tlx.ops.mm_mxfp requires positive M and N and K divisible by 128; "
                           f"got M={M}, N={N}, K={K}")
    if (M % 128 or N % 128) and (a.device.type != "cuda" or _target(a.device).key != "gfx950"):
        raise InvalidInput("tlx.ops.mm_mxfp requires M and N divisible by 128 outside gfx950; "
                           f"got M={M}, N={N}")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise InvalidInput("tlx.ops.mm_mxfp expects E4M3 a and b; "
                           f"got a.dtype={a.dtype}, b.dtype={b.dtype}")
    valid_scale_dtypes = ((torch.uint8, torch.float8_e8m0fnu) if sf_layout == "mfma_preshuffled" else
                          (torch.float8_e8m0fnu, ))
    if a_scale.dtype not in valid_scale_dtypes or b_scale.dtype not in valid_scale_dtypes:
        raise InvalidInput("tlx.ops.mm_mxfp expects E8M0 a_scale and b_scale; "
                           f"got a_scale.dtype={a_scale.dtype}, b_scale.dtype={b_scale.dtype}")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise InvalidInput("tlx.ops.mm_mxfp expects all inputs to be contiguous")
    if a.device.type != "cuda" or any(tensor.device != a.device for tensor in tensors[1:]):
        raise InvalidInput("tlx.ops.mm_mxfp expects all inputs on the same CUDA device; "
                           f"got {[tensor.device for tensor in tensors]}")
    _check_mm_mxfp_scales(a_scale, b_scale, M=M, N=N, K=K, sf_layout=sf_layout)
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise InvalidInput("tlx.ops.mm_mxfp expects out to be a tensor or None")
        if out.shape != (M, N) or out.dtype != torch.bfloat16 or out.device != a.device or not out.is_contiguous():
            raise InvalidInput("tlx.ops.mm_mxfp out must be contiguous BF16 [M, N] on a's device; "
                               f"got shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device}")
        if any(torch._C._overlaps(out, tensor) for tensor in tensors):
            raise InvalidInput("tlx.ops.mm_mxfp out must not overlap any input")
    space_name, invariance = _normalize_space_arg("mm_mxfp", space)
    if space_name is not None and space_name not in ("heuristic", "full"):
        raise InvalidInput(f"tlx.ops.mm_mxfp does not provide space={space_name!r}")

    fn, spec = impl_for("mm_mxfp", device=a.device)
    if space_name is None:
        space_name = spec.default_space
    tma_tensors = tensors if out is None else (*tensors, out)
    check_inputs(spec, dtype=a.dtype, base_ptrs=tuple(tensor.data_ptr() for tensor in tma_tensors))
    check_backward(spec, *tensors, out)
    _check_enforced("mm_mxfp", spec, invariance)
    return fn(a, a_scale, b, b_scale, out=out, sf_layout=sf_layout, space=space_name)


def _check_mm_mxfp6_scales(a_scale, b_scale, *, M, N, K, sf_layout):
    if sf_layout == "natural":
        expected_a, expected_b = (M, K // 32), (N, K // 32)
        if a_scale.shape != expected_a or b_scale.shape != expected_b:
            raise InvalidInput("tlx.ops.mm_mxfp6 natural scales must have exact shapes "
                               f"a_scale={expected_a}, b_scale={expected_b}; got "
                               f"a_scale={tuple(a_scale.shape)}, b_scale={tuple(b_scale.shape)}")
        return
    if sf_layout != "cublas_blocked":
        raise InvalidInput("tlx.ops.mm_mxfp6 sf_layout must be 'natural' or 'cublas_blocked'; "
                           f"got {sf_layout!r}")
    # M, N and K are 128-aligned, so the 128x4 atoms need no padding.
    if a_scale.numel() != M * K // 32 or b_scale.numel() != N * K // 32:
        raise InvalidInput("tlx.ops.mm_mxfp6 cublas_blocked scales must contain exactly "
                           f"{M * K // 32} a-scale and {N * K // 32} b-scale bytes; "
                           f"got {a_scale.numel()} and {b_scale.numel()}")


def mm_mxfp6(a, a_scale, b, b_scale, *, out=None, sf_layout="natural", space="full", a_format="e2m3", b_format="e3m2"):
    """``a @ b.T`` over pre-quantized MXFP6 inputs, returning BF16 ``[M, N]``.

    ``a`` is contiguous dense-packed ``[M, K]`` and ``b`` is contiguous
    dense-packed ``[N, K]`` (K-major, as for a linear weight), both uint8
    with groups of 16 codes per 12 bytes LSB-first and the K dim padded to
    the unpacked width; M, N and K must be multiples of 128. The format
    arguments ``a_format`` and ``b_format`` independently select "e2m3"
    or "e3m2", defaulting to E2M3 A x E3M2 B. They specialize the kernel
    as constexpr arguments. Each run of 32 values along K
    has one E8M0 scale. ``sf_layout="natural"`` takes ``[M, K // 32]`` and
    ``[N, K // 32]`` scales; ``"cublas_blocked"`` takes the swizzled byte
    layout returned by torchao's ``MXTensor.to_mx(..., is_swizzled_scales=True)``.
    Quantize with torchao (``elem_dtype="fp6_e2m3"`` / ``"fp6_e3m2"``), shift
    its unpacked codes down (``qdata >> 2``), and dense-pack. A supplied
    ``out`` is returned by identity. Forward-only.

    ``space`` defaults to "full" (autotune; the first call per shape compiles
    and benchmarks the pruned space). "heuristic" launches one shape-picked
    config without autotuning.
    """
    import torch

    if a_format not in ("e2m3", "e3m2") or b_format not in ("e2m3", "e3m2"):
        raise InvalidInput("tlx.ops.mm_mxfp6 a_format and b_format must each be 'e2m3' or 'e3m2'")
    tensors = (a, a_scale, b, b_scale)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise InvalidInput("tlx.ops.mm_mxfp6 expects tensor inputs")
    if a.ndim != 2 or b.ndim != 2:
        raise InvalidInput("tlx.ops.mm_mxfp6 expects rank-2 a and b; "
                           f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    M, K = a.shape
    N, b_k = b.shape
    if b_k != K:
        raise InvalidInput("tlx.ops.mm_mxfp6 reduction dimensions must match; "
                           f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    if min(M, N, K) <= 0 or M % 128 or N % 128 or K % 128:
        raise InvalidInput("tlx.ops.mm_mxfp6 requires positive M, N and K divisible by 128; "
                           f"got M={M}, N={N}, K={K}")
    if a.dtype != torch.uint8 or b.dtype != torch.uint8:
        raise InvalidInput("tlx.ops.mm_mxfp6 expects dense-packed uint8 a and b; "
                           f"got a.dtype={a.dtype}, b.dtype={b.dtype}")
    if a_scale.dtype != torch.float8_e8m0fnu or b_scale.dtype != torch.float8_e8m0fnu:
        raise InvalidInput("tlx.ops.mm_mxfp6 expects E8M0 a_scale and b_scale; "
                           f"got a_scale.dtype={a_scale.dtype}, b_scale.dtype={b_scale.dtype}")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise InvalidInput("tlx.ops.mm_mxfp6 expects all inputs to be contiguous")
    if a.device.type != "cuda" or any(tensor.device != a.device for tensor in tensors[1:]):
        raise InvalidInput("tlx.ops.mm_mxfp6 expects all inputs on the same CUDA device; "
                           f"got {[tensor.device for tensor in tensors]}")
    _check_mm_mxfp6_scales(a_scale, b_scale, M=M, N=N, K=K, sf_layout=sf_layout)
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise InvalidInput("tlx.ops.mm_mxfp6 expects out to be a tensor or None")
        if out.shape != (M, N) or out.dtype != torch.bfloat16 or out.device != a.device or not out.is_contiguous():
            raise InvalidInput("tlx.ops.mm_mxfp6 out must be contiguous BF16 [M, N] on a's device; "
                               f"got shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device}")
        if any(torch._C._overlaps(out, tensor) for tensor in tensors):
            raise InvalidInput("tlx.ops.mm_mxfp6 out must not overlap any input")
    if space not in ("heuristic", "full"):
        raise InvalidInput(f"tlx.ops.mm_mxfp6 does not provide space={space!r}")

    fn, spec = impl_for("mm_mxfp6", device=a.device)
    tma_tensors = tensors if out is None else (*tensors, out)
    check_inputs(spec, dtype=a.dtype, base_ptrs=tuple(tensor.data_ptr() for tensor in tma_tensors))
    check_backward(spec, *tensors, out)
    return fn(a, a_scale, b, b_scale, out=out, sf_layout=sf_layout, space=space, a_format=a_format, b_format=b_format)


def grouped_gemm(group_a, group_b, *, space=None):
    """Run a ragged group of FP16 ``A[i] @ B[i]`` products.

    ``A[i]`` must be rank-2 row-major. ``B[i]`` must be a rank-2 column-major
    view with K contiguous. The operation is currently forward-only;
    architecture-specific alignment restrictions may also apply.
    ``space=`` takes no search space today; a ``Space`` is accepted only to
    carry a ``BitwiseInvarianceSpec``, which is validated but not enforced yet.
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

    space_name, invariance = _normalize_space_arg("grouped_gemm", space)
    _check_single_config_space("grouped_gemm", space_name)
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
    _check_enforced("grouped_gemm", spec, invariance)
    return fn(group_a, group_b)


def _grouped_gemm_mxfp8_dims(x, w, split_sizes):
    GM, K = x.shape
    G = split_sizes.shape[0]
    if w.ndim == 3:
        w_groups, N, w_k = w.shape
        if w_groups != G:
            raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects w.shape[0] == G; "
                               f"got w.shape={tuple(w.shape)}, G={G}")
    else:
        grouped_n, w_k = w.shape
        if G == 0 or grouped_n % G != 0:
            raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects packed w.shape[0] divisible by G; "
                               f"got w.shape={tuple(w.shape)}, G={G}")
        N = grouped_n // G
    return GM, G, N, K, w_k


def _check_grouped_gemm_mxfp8_scales(x_scale, w_scale, w, *, GM, G, N, K, sf_layout):
    scale_k = K // 32
    if sf_layout == "natural":
        expected_x = (GM, scale_k)
        expected_w = (G, N, scale_k) if w.ndim == 3 else (G * N, scale_k)
        if x_scale.shape != expected_x or w_scale.shape != expected_w:
            raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 natural scales must have exact shapes "
                               f"x_scale={expected_x}, w_scale={expected_w}; got "
                               f"x_scale={tuple(x_scale.shape)}, w_scale={tuple(w_scale.shape)}")
        return

    if sf_layout != "cublas_blocked":
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 sf_layout must be 'natural' or "
                           f"'cublas_blocked'; got {sf_layout!r}")
    if x_scale.ndim != 2 or w_scale.ndim != 2:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 cublas_blocked scales must be rank-2 byte tensors; "
                           f"got x_scale.ndim={x_scale.ndim}, w_scale.ndim={w_scale.ndim}")
    padded_scale_k = ((scale_k + 3) // 4) * 4
    expected_x_bytes = ((GM + 127) // 128) * 128 * padded_scale_k
    expected_w_bytes = G * ((N + 127) // 128) * 128 * padded_scale_k
    x_bytes = x_scale.numel() * x_scale.element_size()
    w_bytes = w_scale.numel() * w_scale.element_size()
    if x_bytes != expected_x_bytes or w_bytes != expected_w_bytes:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 cublas_blocked scales must contain exactly "
                           f"{expected_x_bytes} x-scale bytes and {expected_w_bytes} w-scale bytes; "
                           f"got {x_bytes} and {w_bytes}")


def _check_grouped_gemm_mxfp8_out(out, tensors, *, GM, N, device):
    import torch

    if out is None:
        return
    if not isinstance(out, torch.Tensor):
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects out to be a tensor or None")
    if out.shape != (GM, N) or out.dtype != torch.bfloat16 or out.device != device or not out.is_contiguous():
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 out must be contiguous BF16 [GM, N] on x's device; "
                           f"got shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device}")
    if any(torch._C._overlaps(out, tensor) for tensor in tensors):
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 out must not overlap any input")


def grouped_gemm_mxfp8(x, x_scale, w, w_scale, split_sizes, *, out=None, num_sms=None, sf_layout="natural", space=None):
    """Run a forward-only SM100 or gfx950 grouped GEMM over pre-quantized MXFP8 inputs.

    ``x`` is contiguous ``[GM, K]`` and ``w`` is contiguous ``[G, N, K]``
    or packed ``[G * N, K]`` E4M3 data. Natural E8M0 scales match those
    logical data shapes with a final ``K // 32`` dimension. CuBLAS-blocked
    scales are rank-2 opaque byte tensors with exact 128x4-atom storage.

    ``split_sizes`` stays on device: the kernel validates that values are
    nonnegative, every prefix is 128-aligned, and the final sum is ``GM``.
    The BF16 result has shape ``[GM, N]``; a supplied ``out`` is returned by
    identity. ``num_sms`` defaults to all SMs (CUs on AMD) on ``x.device``.
    ``space=`` takes no search space today; a ``Space`` is accepted only to
    carry a ``BitwiseInvarianceSpec``, which is validated but not enforced yet.
    """
    import torch

    tensors = (x, x_scale, w, w_scale, split_sizes)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects tensor inputs")
    if x.ndim != 2 or w.ndim not in (2, 3) or split_sizes.ndim != 1:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects x/w/split_sizes ranks 2/(2 or 3)/1; "
                           f"got {x.ndim}/{w.ndim}/{split_sizes.ndim}")

    GM, G, N, K, w_k = _grouped_gemm_mxfp8_dims(x, w, split_sizes)
    if min(GM, G, N, K) <= 0:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 requires positive GM, G, N, and K; "
                           f"got GM={GM}, G={G}, N={N}, K={K}")
    if w_k != K:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 reduction dimensions must match; "
                           f"got x.shape={tuple(x.shape)}, w.shape={tuple(w.shape)}")
    if GM % 128 != 0 or K % 128 != 0:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 requires GM and K divisible by 128; "
                           f"got GM={GM}, K={K}")

    if x.dtype != torch.float8_e4m3fn or w.dtype != torch.float8_e4m3fn:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects E4M3 x and w; "
                           f"got x.dtype={x.dtype}, w.dtype={w.dtype}")
    if x_scale.dtype != torch.float8_e8m0fnu or w_scale.dtype != torch.float8_e8m0fnu:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects E8M0 x_scale and w_scale; "
                           f"got x_scale.dtype={x_scale.dtype}, w_scale.dtype={w_scale.dtype}")
    if split_sizes.dtype != torch.int32:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects split_sizes.dtype == torch.int32; "
                           f"got {split_sizes.dtype}")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects all inputs to be contiguous")
    if x.device.type != "cuda" or any(tensor.device != x.device for tensor in tensors[1:]):
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects all inputs on the same CUDA device; "
                           f"got {[tensor.device for tensor in tensors]}")

    _check_grouped_gemm_mxfp8_scales(x_scale, w_scale, w, GM=GM, G=G, N=N, K=K, sf_layout=sf_layout)
    if num_sms is not None:
        device_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        if type(num_sms) is not int or not 1 <= num_sms <= device_sms:
            raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 num_sms must be an int in "
                               f"[1, {device_sms}] for {x.device}; got {num_sms!r}")
    _check_grouped_gemm_mxfp8_out(out, tensors, GM=GM, N=N, device=x.device)

    space_name, invariance = _normalize_space_arg("grouped_gemm_mxfp8", space)
    _check_single_config_space("grouped_gemm_mxfp8", space_name)
    fn, spec = impl_for("grouped_gemm_mxfp8", device=x.device)
    tma_tensors = (x, x_scale, w, w_scale) if out is None else (x, x_scale, w, w_scale, out)
    check_inputs(
        spec,
        dtype=x.dtype,
        row_bytes=(
            x.stride(0) * x.element_size(),
            w.stride(-2) * w.element_size(),
            N * torch.bfloat16.itemsize,
        ),
        base_ptrs=tuple(tensor.data_ptr() for tensor in tma_tensors),
    )
    check_backward(spec, x, x_scale, w, w_scale, split_sizes, out)
    _check_enforced("grouped_gemm_mxfp8", spec, invariance)
    return fn(
        x,
        x_scale,
        w,
        w_scale,
        split_sizes,
        out=out,
        num_sms=num_sms,
        sf_layout=sf_layout,
    )


# The local-expert histogram and the [next_pow2(topk), next_pow2(H)] fp32 finalize tile live in registers. 2**16
# elements fit 4 warps without spilling; 2**17 is a whole gfx950 CU register file and spills at any warp count.
_MOE_MAX_LOCAL_EXPERTS = 512
_MOE_MAX_FINALIZE_TILE = 1 << 16


def moe(x, topk_weights, topk_ids, w13, w2, *, expert_start=0, post_norm_weight=None, post_norm_eps=1e-6,
        post_norm_gain=0.0, out=None, space=None):
    """Routed SiLU-gated mixture of experts over precomputed top-k routing. Forward only, bf16.

    For each token ``t`` returns ``sum_k topk_weights[t, k] * f(y_k)`` where, for local expert
    ``e = topk_ids[t, k] - expert_start``, ``y_k = (silu(x[t] @ w13[e, :I].T) * (x[t] @ w13[e, I:].T)) @ w2[e].T``.
    ``f`` is the identity, or with ``post_norm_weight`` an RMSNorm without affine of
    ``bf16((post_norm_weight + post_norm_gain) * y_k)``. Routes whose id falls outside
    ``[expert_start, expert_start + E)`` belong to other expert-parallel ranks and contribute zero.

    Shapes: ``x (T, K)``, ``topk_weights``/``topk_ids (T, topk)``, ``w13 (E, 2I, K)``, ``w2 (E, H, I)``,
    ``post_norm_weight (H,)``; returns or fills ``out (T, H)``. ``K``, ``I`` and ``H`` must be positive multiples
    of 128, ``E <= 512`` and ``next_pow2(topk) * next_pow2(H) <= 2**16``. Ids are never read on the host, so
    the call is graph-capturable.
    Only ``space="heuristic"`` is provided.
    """
    import torch

    tensors = (x, topk_weights, topk_ids, w13, w2)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise InvalidInput("tlx.ops.moe expects tensor x, topk_weights, topk_ids, w13 and w2")
    if x.ndim != 2 or topk_ids.ndim != 2 or w13.ndim != 3 or w2.ndim != 3:
        raise InvalidInput("tlx.ops.moe expects x (T, K), topk_ids (T, topk), w13 (E, 2I, K) and w2 (E, H, I); got "
                           f"{[tuple(t.shape) for t in (x, topk_ids, w13, w2)]}")
    T, K = x.shape
    E, two_i, w13_k = w13.shape
    I = two_i // 2
    H = w2.shape[1]
    if (topk_weights.shape != topk_ids.shape or topk_ids.shape[0] != T or two_i % 2 or w13_k != K
            or w2.shape != (E, H, I)):
        raise InvalidInput("tlx.ops.moe shape mismatch: x=%s topk_weights=%s topk_ids=%s w13=%s w2=%s" %
                           tuple(tuple(t.shape) for t in tensors))
    if any(t.dtype != x.dtype for t in (w13, w2)):
        raise InvalidInput("tlx.ops.moe expects x, w13 and w2 to share a dtype")
    if topk_ids.dtype not in (torch.int32, torch.int64) or not topk_weights.dtype.is_floating_point:
        raise InvalidInput("tlx.ops.moe expects integer topk_ids and floating-point topk_weights; got "
                           f"{topk_ids.dtype} and {topk_weights.dtype}")
    topk = topk_ids.shape[1]
    tile = (1 << (topk - 1).bit_length()) * (1 << (H - 1).bit_length()) if topk >= 1 and H >= 1 else 0
    if not 1 <= tile <= _MOE_MAX_FINALIZE_TILE or not 1 <= E <= _MOE_MAX_LOCAL_EXPERTS:
        raise InvalidInput(f"tlx.ops.moe supports next_pow2(topk) * next_pow2(H) <= {_MOE_MAX_FINALIZE_TILE} and "
                           f"1 <= E <= {_MOE_MAX_LOCAL_EXPERTS} local experts; got topk={topk}, H={H}, E={E}")
    if not isinstance(expert_start, int) or expert_start < 0:
        raise InvalidInput(f"tlx.ops.moe expects a non-negative int expert_start; got {expert_start!r}")
    extra = []
    if post_norm_weight is not None:
        if (not isinstance(post_norm_weight, torch.Tensor) or post_norm_weight.shape != (H, )
                or not post_norm_weight.dtype.is_floating_point):
            raise InvalidInput(f"tlx.ops.moe expects a floating-point post_norm_weight of shape ({H},)")
        extra.append(post_norm_weight)
    if out is not None:
        if (not isinstance(out, torch.Tensor) or out.shape != (T, H) or out.dtype != torch.bfloat16
                or not out.is_contiguous()):
            raise InvalidInput(f"tlx.ops.moe expects a contiguous bf16 out of shape ({T}, {H})")
        if any(torch._C._overlaps(out, t) for t in (*tensors, *extra)):
            raise InvalidInput("tlx.ops.moe out must not alias an input")
        extra.append(out)
    if any(t.device != x.device for t in (*tensors, *extra)):
        raise InvalidInput("tlx.ops.moe expects all tensors on one device")

    space_name, invariance = _normalize_space_arg("moe", space)
    _check_single_config_space("moe", space_name)
    fn, spec = impl_for("moe", device=x.device)
    check_inputs(spec, dtype=x.dtype, K=K, I=I, H=H)
    check_backward(spec, *tensors, *extra)
    _check_enforced("moe", spec, invariance)
    with torch.cuda.device(x.device):
        return fn(x, topk_weights, topk_ids, w13, w2, expert_start, post_norm_weight, post_norm_eps, post_norm_gain,
                  out)


_MOE_ROUTING_MAX_EXPERTS = 1024
_MOE_ROUTING_MAX_TOPK = 32


def moe_routing(router_logits, correction_bias, topk=8, *, router_multiplier=1.0, route_scale=1.0, eps=0.0,
                num_valid_tokens=None, out=None, space=None):
    """Sigmoid top-k MoE routing with a selection-only correction bias. Forward only, fp32.

    Per token ``t``, with ``s = sigmoid(router_logits[t] * router_multiplier)``: picks the ``topk`` experts with the
    largest ``s + correction_bias`` (equal values pick the lower expert id), in that order, and weights expert ``k``
    by ``s[id_k] * (route_scale / (s[id_0] + s[id_1] + ... + s[id_{topk-1}] + eps))``, summed left to right. The
    bias steers selection only; the weights use the unbiased scores. NaN scores are never picked: a token with fewer
    than ``topk`` non-NaN experts gets id ``-1`` and weight ``0`` in its remaining slots, and its weights are
    normalized over the experts it got.

    Shapes: ``router_logits (T, E)`` fp32, ``correction_bias (E,)``; returns or fills
    ``out = (topk_weights, topk_ids)``, fp32 and int32 ``(T, topk)`` with unit column stride and a row stride of at
    least ``topk``. ``E <= 1024``, ``1 <= topk <= min(E, 32)``. The ids are global expert ids, as ``tlx.ops.moe`` takes
    them. ``num_valid_tokens`` is an optional one-element int32/int64 device tensor holding the real token count: rows
    at or past it get id ``-1`` and weight ``0``, which ``tlx.ops.moe`` treats as routed to no local expert. Nothing is
    read on the host, so the call is graph-capturable. Batch invariant: the launch layout depends on ``T``, but every
    layout gives the same bits.
    Only ``space="heuristic"`` is provided.
    """
    import torch

    if not isinstance(router_logits, torch.Tensor) or not isinstance(correction_bias, torch.Tensor):
        raise InvalidInput("tlx.ops.moe_routing expects tensor router_logits and correction_bias")
    if router_logits.ndim != 2 or correction_bias.shape != router_logits.shape[1:]:
        raise InvalidInput("tlx.ops.moe_routing expects router_logits (T, E) and correction_bias (E,); got "
                           f"{tuple(router_logits.shape)} and {tuple(correction_bias.shape)}")
    T, E = router_logits.shape
    if not correction_bias.dtype.is_floating_point:
        raise InvalidInput(f"tlx.ops.moe_routing expects a floating-point correction_bias; got {correction_bias.dtype}")
    if not 1 <= E <= _MOE_ROUTING_MAX_EXPERTS:
        raise InvalidInput(f"tlx.ops.moe_routing supports 1 <= E <= {_MOE_ROUTING_MAX_EXPERTS}; got E={E}")
    if not isinstance(topk, int) or isinstance(topk, bool) or not 1 <= topk <= min(E, _MOE_ROUTING_MAX_TOPK):
        raise InvalidInput(f"tlx.ops.moe_routing supports 1 <= topk <= min(E, {_MOE_ROUTING_MAX_TOPK}); got "
                           f"topk={topk!r}, E={E}")
    scalars = dict(router_multiplier=router_multiplier, route_scale=route_scale, eps=eps)
    for name, value in scalars.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidInput(f"tlx.ops.moe_routing expects a Python number for {name}; got {value!r}")
    tensors = [router_logits, correction_bias]
    if num_valid_tokens is not None:
        if (not isinstance(num_valid_tokens, torch.Tensor) or num_valid_tokens.numel() != 1
                or num_valid_tokens.dtype not in (torch.int32, torch.int64)):
            raise InvalidInput("tlx.ops.moe_routing expects num_valid_tokens as a one-element int32/int64 tensor")
        tensors.append(num_valid_tokens)
    if out is not None:
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise InvalidInput("tlx.ops.moe_routing expects out=(topk_weights, topk_ids)")
        weights, ids = out
        for t, dtype in ((weights, torch.float32), (ids, torch.int32)):
            if (not isinstance(t, torch.Tensor) or t.shape != (T, topk) or t.dtype != dtype
                    or (topk > 1 and t.stride(1) != 1) or (T > 1 and t.stride(0) < topk)):
                raise InvalidInput(f"tlx.ops.moe_routing expects out as fp32 and int32 ({T}, {topk}) tensors with "
                                   "unit column stride and rows that do not overlap")
        if any(torch._C._overlaps(o, t) for o in out for t in tensors) or torch._C._overlaps(weights, ids):
            raise InvalidInput("tlx.ops.moe_routing out must not alias an input or each other")
        tensors.extend(out)
    if any(t.device != router_logits.device for t in tensors):
        raise InvalidInput("tlx.ops.moe_routing expects all tensors on one device")

    space_name, invariance = _normalize_space_arg("moe_routing", space)
    _check_single_config_space("moe_routing", space_name)
    fn, spec = impl_for("moe_routing", device=router_logits.device)
    check_inputs(spec, dtype=router_logits.dtype, E=E, topk=topk)
    check_backward(spec, router_logits, correction_bias, *(out or ()))
    _check_enforced("moe_routing", spec, invariance)
    with torch.cuda.device(router_logits.device):
        return fn(router_logits, correction_bias, topk, router_multiplier, route_scale, eps, num_valid_tokens, out)


_MOE_FINALIZE_MAX_H = 8192
_MOE_FINALIZE_MAX_TOPK = 16


def moe_finalize(expert_out, topk_weights, *, row_index=None, topk_ids=None, expert_mask=None, norm_weight=None,
                 input_scale=True, eps=1e-6, gain_center=0.0, out=None, space=None):
    """Combine routed expert outputs per token: RMSNorm each slot's expert row, scale it by its routing weight, sum
    over the top-k slots. BF16. Differentiable in ``expert_out``, ``topk_weights`` and ``norm_weight``.

    Layouts, with ``topk_weights`` fp32 ``(T, K)``:

    - ``row_index`` given: ``expert_out`` is ``(R, H)``, the expert outputs in any row order (for example grouped by
      expert, as an expert GEMM writes them), and ``row_index`` int32/int64 ``(T, K)`` holds the row of each slot. A
      slot whose row is outside ``[0, R)``, for example -1 for an expert on another rank, has no row.
    - otherwise ``expert_out`` is ``(T, K, H)``. With ``topk_ids`` int32/int64 ``(T, K)`` and ``expert_mask``
      bool/int32 ``(E,)``, a slot contributes only when ``0 <= topk_ids[t, k] < E`` and its mask entry is nonzero;
      without them every slot does.

    A slot that does not contribute is not read. For a contributing slot with row ``x`` (upcast), weight
    ``w = bf16(norm_weight + gain_center)`` (``norm_weight`` BF16 or FP32 ``(H,)``; ``w = 1`` without one) and
    ``s = topk_weights[t, k]``, in FP32:
    ``n = bf16(x * w)`` and ``y = bf16(n * rsqrt(mean(n * n) + eps))`` with ``input_scale``, else
    ``y = bf16(x * rsqrt(mean(x * x) + eps) * w)``; then ``out[t] = bf16(sum_k bf16(y * s))``, summed in slot order.
    A token without contributing slots gets zeros. Returns ``out`` (BF16 ``(T, H)``), filling it when passed.
    ``H`` is at most 8192 and ``K`` at most 16. Batch invariant. Only ``space="heuristic"`` is provided.

    The backward recomputes the norm. The gradient of ``expert_out`` has its shape and is zero for rows that no
    contributing slot names; with ``row_index`` each row must be named by at most one slot, which the backward
    checks on the device (a row named twice fails a device-side assertion). The gradient of ``topk_weights`` is
    ``sum(g * y)`` per slot (zero for slots that do not contribute) and that of ``norm_weight`` is accumulated in
    FP32 and returned in its dtype; ``gain_center`` passes it through unchanged. ``g * s`` and the
    gradient of the norm input are rounded to BF16, as in the eager chain.

    ``gain_center`` folds the offset of a norm whose weight is stored centered at zero (for example a learnable
    input scale applied as ``weight + 1``) into the kernel. It needs ``norm_weight``; the default 0 leaves a BF16
    weight unchanged.
    ``out`` is forward only: it cannot be combined with inputs that require grad.
    """
    import torch

    name = "tlx.ops.moe_finalize"
    if not isinstance(expert_out, torch.Tensor) or not isinstance(topk_weights, torch.Tensor):
        raise InvalidInput(f"{name} expects tensor expert_out and topk_weights")
    if (topk_weights.ndim != 2 or topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous()):
        raise InvalidInput(f"{name} expects contiguous fp32 topk_weights (T, K); got {tuple(topk_weights.shape)} "
                           f"{topk_weights.dtype}")
    T, K = topk_weights.shape
    if row_index is not None and (topk_ids is not None or expert_mask is not None):
        raise InvalidInput(f"{name} takes row_index or topk_ids/expert_mask, not both")
    if (topk_ids is None) != (expert_mask is None):
        raise InvalidInput(f"{name} expects topk_ids and expert_mask together")
    want_dims = 2 if row_index is not None else 3
    if (expert_out.ndim != want_dims or expert_out.dtype != torch.bfloat16 or not expert_out.is_contiguous()
            or (want_dims == 3 and tuple(expert_out.shape[:2]) != (T, K))):
        layout = "(R, H) with row_index" if want_dims == 2 else f"({T}, {K}, H)"
        raise InvalidInput(f"{name} expects a contiguous BF16 expert_out of shape {layout}; got "
                           f"{tuple(expert_out.shape)} {expert_out.dtype}")
    H = expert_out.shape[-1]
    if not 1 <= H <= _MOE_FINALIZE_MAX_H or not 1 <= K <= _MOE_FINALIZE_MAX_TOPK:
        raise InvalidInput(f"{name} supports 1 <= H <= {_MOE_FINALIZE_MAX_H} and 1 <= K <= "
                           f"{_MOE_FINALIZE_MAX_TOPK}; got H={H}, K={K}")
    for arg, t in (("row_index", row_index), ("topk_ids", topk_ids)):
        if t is not None and (not isinstance(t, torch.Tensor) or t.shape != (T, K)
                              or t.dtype not in (torch.int32, torch.int64) or not t.is_contiguous()):
            raise InvalidInput(f"{name} expects a contiguous int32/int64 {arg} of shape ({T}, {K})")
    if expert_mask is not None and (not isinstance(expert_mask, torch.Tensor) or expert_mask.ndim != 1
                                    or expert_mask.numel() == 0 or expert_mask.dtype not in (torch.bool, torch.int32)
                                    or not expert_mask.is_contiguous()):
        raise InvalidInput(f"{name} expects a contiguous non-empty bool/int32 expert_mask (E,)")
    if norm_weight is not None and (not isinstance(norm_weight, torch.Tensor) or norm_weight.shape != (H, )
                                    or norm_weight.dtype not in (torch.bfloat16, torch.float32)
                                    or not norm_weight.is_contiguous()):
        raise InvalidInput(f"{name} expects a contiguous BF16 or FP32 norm_weight of shape ({H},)")
    if (isinstance(gain_center, bool) or not isinstance(gain_center, (int, float)) or not math.isfinite(gain_center)):
        raise InvalidInput(f"{name} expects a finite Python number gain_center; got {gain_center!r}")
    if gain_center != 0 and norm_weight is None:
        raise InvalidInput(f"{name} applies gain_center to norm_weight; pass norm_weight with gain_center")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not 0 <= eps < float("inf"):
        raise InvalidInput(f"{name} expects a finite non-negative Python number eps; got {eps!r}")
    tensors = [t for t in (expert_out, topk_weights, row_index, topk_ids, expert_mask, norm_weight) if t is not None]
    if out is not None:
        if (not isinstance(out, torch.Tensor) or out.shape != (T, H) or out.dtype != torch.bfloat16
                or not out.is_contiguous()):
            raise InvalidInput(f"{name} expects a contiguous BF16 out of shape ({T}, {H})")
        if any(torch._C._overlaps(out, t) for t in tensors):
            raise InvalidInput(f"{name} out must not alias an input")
        tensors.append(out)
    if any(t.device != expert_out.device for t in tensors):
        raise InvalidInput(f"{name} expects all tensors on one device")

    space_name, invariance = _normalize_space_arg("moe_finalize", space)
    _check_single_config_space("moe_finalize", space_name)
    fn, spec = impl_for("moe_finalize", device=expert_out.device)
    check_inputs(spec, dtype=expert_out.dtype, H=H, K=K)
    check_backward(spec, *tensors)
    if out is not None and torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise UnsupportedBackward(f"{name} fills out only without autograd; pass out=None to differentiate")
    _check_enforced("moe_finalize", spec, invariance)
    with torch.cuda.device(expert_out.device):
        return fn(expert_out, topk_weights, row_index, topk_ids, expert_mask, norm_weight, bool(input_scale), eps,
                  float(gain_center), out)


def _check_boundary_inputs(op, stream, branch, post_weight, pre_weight, alpha, beta, out, new_stream):
    """Validate a residual-stream boundary call; returns its tensors (inputs, then given outputs) and ``H``."""
    import torch

    if not isinstance(stream, torch.Tensor) or not isinstance(branch, torch.Tensor):
        raise InvalidInput(f"tlx.ops.{op} expects tensor stream and branch")
    if (stream.ndim != 2 or stream.shape != branch.shape or stream.dtype != torch.float32
            or branch.dtype != torch.bfloat16 or not stream.is_contiguous() or not branch.is_contiguous()):
        raise InvalidInput(f"tlx.ops.{op} expects contiguous FP32 stream and BF16 branch of one "
                           f"(T, H) shape; got {tuple(stream.shape)} {stream.dtype} and {tuple(branch.shape)} "
                           f"{branch.dtype}")
    T, H = branch.shape
    for name, w in (("post_weight", post_weight), ("pre_weight", pre_weight)):
        if w is not None and (not isinstance(w, torch.Tensor) or w.shape != (H, ) or w.dtype != torch.bfloat16
                              or not w.is_contiguous()):
            raise InvalidInput(f"tlx.ops.{op} expects a contiguous BF16 {name} of shape ({H},)")
    if (alpha is None) != (beta is None):
        raise InvalidInput(f"tlx.ops.{op} expects alpha and beta together")
    if alpha is not None and any(not isinstance(g, torch.Tensor) or g.shape not in (
        (H, ), (1, )) or g.dtype != torch.float32 or not g.is_contiguous() or g.shape != alpha.shape
                                 for g in (alpha, beta)):
        raise InvalidInput(f"tlx.ops.{op} expects contiguous FP32 alpha and beta of shape ({H},) "
                           "or (1,)")
    inputs = tuple(t for t in (stream, branch, post_weight, pre_weight, alpha, beta) if t is not None)
    outputs = []
    for name, buf, dtype in (("out", out, torch.bfloat16), ("new_stream", new_stream, torch.float32)):
        if buf is None:
            continue
        if not isinstance(buf, torch.Tensor) or buf.shape != (T, H) or buf.dtype != dtype or not buf.is_contiguous():
            raise InvalidInput(f"tlx.ops.{op} expects a contiguous {dtype} {name} of shape "
                               f"({T}, {H})")
        if any(torch._C._overlaps(buf, t) for t in (*inputs, *outputs)):
            raise InvalidInput(f"tlx.ops.{op} {name} must not alias an input or output")
        outputs.append(buf)
    if any(t.device != branch.device for t in (*inputs, *outputs)):
        raise InvalidInput(f"tlx.ops.{op} expects all tensors on one device")

    return (*inputs, *outputs), H


def norm_residual_norm(stream, branch, *, post_weight=None, post_norm_input_scale=False, pre_weight=None, alpha=None,
                       beta=None, post_eps=1e-6, pre_eps=1e-6, pre_norm_bf16=False, out=None, new_stream=None,
                       space=None):
    """One residual-stream boundary: RMSNorm the branch, add it to the stream, RMSNorm the new stream. Forward only.

    With ``p = branch`` (BF16, upcast) and ``w_post = post_weight``:
    ``y = bf16(rmsnorm(p) * w_post)``, or with ``post_norm_input_scale`` ``y = bf16(rmsnorm(bf16(p * w_post)))``;
    ``new_stream = alpha * stream + beta * y`` (``stream + y`` without a gate), in FP32;
    ``out = bf16(rmsnorm(q) * pre_weight)`` with ``q = new_stream``, or ``bf16(new_stream)`` with ``pre_norm_bf16``.
    Each RMSNorm is ``x * rsqrt(mean(x * x) + eps)`` in FP32; a missing weight is the identity.

    Shapes: ``stream`` FP32 and ``branch`` BF16 ``(T, H)``, contiguous; ``post_weight``/``pre_weight`` BF16 ``(H,)``;
    ``alpha``/``beta`` FP32 ``(H,)`` or ``(1,)``, both or neither. Returns ``(out, new_stream)``, BF16 and FP32
    ``(T, H)``, filling the given buffers when passed. ``H`` must be 4096, 8192 or 16384.
    Only ``space="heuristic"`` is provided.
    """
    import torch

    tensors, H = _check_boundary_inputs("norm_residual_norm", stream, branch, post_weight, pre_weight, alpha, beta, out,
                                        new_stream)
    space_name, invariance = _normalize_space_arg("norm_residual_norm", space)
    _check_single_config_space("norm_residual_norm", space_name)
    fn, spec = impl_for("norm_residual_norm", device=branch.device)
    check_inputs(spec, dtype=branch.dtype, H=H)
    check_backward(spec, *tensors)
    _check_enforced("norm_residual_norm", spec, invariance)
    with torch.cuda.device(branch.device):
        return fn(stream, branch, post_weight, bool(post_norm_input_scale), pre_weight, alpha, beta, post_eps, pre_eps,
                  bool(pre_norm_bf16), out, new_stream)


def all_reduce_workspace(max_numel, *, group=None):
    """Peer-mapped buffers for ``all_reduce`` and ``all_reduce_norm_residual_norm`` on the current device.

    Collective over ``group`` (default: the world group of an initialized ``torch.distributed`` with one GPU per
    rank on one node). ``workspace.input(*shape)`` returns a view of the symmetric BF16 input buffer, which holds
    ``max_numel`` elements; write the local partial into it and pass that view to the op. Calls on one workspace
    must not overlap: issue them from one stream or order the streams, and give each concurrent stream its own
    workspace. A failure during setup raises on every rank.
    ``workspace.close()``, called on every rank of the group, releases the peer mappings; later calls raise.
    """
    import torch

    device = torch.device("cuda", torch.cuda.current_device())
    fn, spec = impl_for("all_reduce_workspace", device=device)
    with torch.cuda.device(device):
        return fn(int(max_numel), group)


def _check_all_reduce_input(op, x, workspace):
    import torch

    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise InvalidInput(f"tlx.ops.{op} expects a contiguous BF16 tensor")
    if getattr(workspace, "closed", False):
        raise InvalidInput(f"tlx.ops.{op} got a closed workspace")
    if not hasattr(workspace, "holds") or not workspace.holds(x):
        raise InvalidInput(f"tlx.ops.{op} expects the input to be workspace.input(...) of its workspace")


def all_reduce(x, workspace, *, out=None, space=None):
    """Sum of ``x`` over the ranks of ``workspace``'s group, returned on every rank. Forward only.

    ``x`` is BF16 and must be ``workspace.input(*x.shape)``. Each element is the FP32 sum of the ranks' values in rank
    order, rounded to BF16, so every rank gets the same bits for every token count. Returns ``out`` (BF16, the
    shape of ``x``), filling it when passed. Only ``space="heuristic"`` is provided.
    """
    import torch

    _check_all_reduce_input("all_reduce", x, workspace)
    if out is not None and (not isinstance(out, torch.Tensor) or out.shape != x.shape or out.dtype != x.dtype
                            or not out.is_contiguous() or out.device != x.device or out.data_ptr() % 16
                            or torch._C._overlaps(out, x)):
        raise InvalidInput(f"tlx.ops.all_reduce expects a contiguous, 16-byte aligned BF16 out of shape "
                           f"{tuple(x.shape)} that does not alias the input")
    space_name, invariance = _normalize_space_arg("all_reduce", space)
    _check_single_config_space("all_reduce", space_name)
    fn, spec = impl_for("all_reduce", device=x.device)
    check_inputs(spec, dtype=x.dtype, world=workspace.world)
    check_backward(spec, x, out)
    _check_enforced("all_reduce", spec, invariance)
    with torch.cuda.device(x.device):
        return fn(x, workspace, out)


def all_reduce_norm_residual_norm(stream, branch, workspace, *, post_weight=None, post_norm_input_scale=False,
                                  pre_weight=None, alpha=None, beta=None, post_eps=1e-6, pre_eps=1e-6,
                                  pre_norm_bf16=False, out=None, new_stream=None, space=None):
    """``norm_residual_norm`` of ``stream`` and ``all_reduce(branch)``. Forward only.

    ``branch`` is this rank's BF16 partial and must be ``workspace.input(T, H)``; the other arguments and the
    returns are those of ``norm_residual_norm``. The result is bitwise equal to ``all_reduce`` followed by
    ``norm_residual_norm``, on every rank. Decode-sized calls run as one kernel. Only
    ``space="heuristic"`` is provided.
    """
    import torch

    _check_all_reduce_input("all_reduce_norm_residual_norm", branch, workspace)
    tensors, H = _check_boundary_inputs("all_reduce_norm_residual_norm", stream, branch, post_weight, pre_weight, alpha,
                                        beta, out, new_stream)
    space_name, invariance = _normalize_space_arg("all_reduce_norm_residual_norm", space)
    _check_single_config_space("all_reduce_norm_residual_norm", space_name)
    fn, spec = impl_for("all_reduce_norm_residual_norm", device=branch.device)
    check_inputs(spec, dtype=branch.dtype, H=H, world=workspace.world)
    check_backward(spec, *tensors)
    _check_enforced("all_reduce_norm_residual_norm", spec, invariance)
    with torch.cuda.device(branch.device):
        return fn(stream, branch, workspace, post_weight, bool(post_norm_input_scale), pre_weight, alpha, beta,
                  post_eps, pre_eps, bool(pre_norm_bf16), out, new_stream)


def router_gemm(x, weight, *, out=None, space=None):
    """MoE router logits in fp32: ``x.float() @ weight.T`` for a bf16 ``x`` and an fp32 ``weight``. Forward only.

    Every product is exact (each fp32 weight element is the exact sum of three bf16 values, multiplied by the bf16
    ``x`` on bf16 matrix cores) and accumulated in fp32, so the logits match an fp32 GEMM on the upcast ``x`` up to
    the order of the fp32 additions, which depends on the token count's bucket. Pass the bf16 hidden states, not
    their fp32 upcast; the weight stays fp32 and may be updated in place.

    Shapes: ``x (T, K)`` bf16 with unit column stride, ``weight (E, K)`` fp32 contiguous; returns or fills
    ``out (T, E)`` fp32 contiguous. ``E`` must be a multiple of 16 and ``K`` a multiple of 1024.
    Only ``space="heuristic"`` is provided.
    """
    import torch

    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise InvalidInput("tlx.ops.router_gemm expects tensor x and weight")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise InvalidInput("tlx.ops.router_gemm expects x (T, K) and weight (E, K); got "
                           f"{tuple(x.shape)} and {tuple(weight.shape)}")
    T, K = x.shape
    E = weight.shape[0]
    if weight.dtype != torch.float32:
        raise InvalidInput(f"tlx.ops.router_gemm expects an fp32 weight; got {weight.dtype}")
    if not weight.is_contiguous() or (T > 1 and x.stride(0) < K) or (K > 1 and x.stride(1) != 1):
        raise InvalidInput("tlx.ops.router_gemm expects a contiguous weight and x with unit column stride and rows "
                           "that do not overlap")
    tensors = [x, weight]
    if out is not None:
        if (not isinstance(out, torch.Tensor) or out.shape != (T, E) or out.dtype != torch.float32
                or not out.is_contiguous()):
            raise InvalidInput(f"tlx.ops.router_gemm expects out as a contiguous fp32 ({T}, {E}) tensor")
        if torch._C._overlaps(out, x) or torch._C._overlaps(out, weight):
            raise InvalidInput("tlx.ops.router_gemm out must not alias x or weight")
        tensors.append(out)
    if any(t.device != x.device for t in tensors):
        raise InvalidInput("tlx.ops.router_gemm expects all tensors on one device")

    space_name, invariance = _normalize_space_arg("router_gemm", space)
    _check_single_config_space("router_gemm", space_name)
    fn, spec = impl_for("router_gemm", device=x.device)
    check_inputs(spec, dtype=x.dtype, E=E, K=K)
    check_backward(spec, *tensors)
    _check_enforced("router_gemm", spec, invariance)
    with torch.cuda.device(x.device):
        return fn(x, weight, out)


def moe_router_topk_align(x, weight, correction_bias, topk=8, *, num_local_experts, local_expert_start=0, block_m,
                          router_multiplier=1.0, route_scale=1.0, eps=0.0, space=None):
    """The MoE routing front end: router logits, sigmoid top-k, and the expert-aligned slot order of the local experts.
    Forward only.

    Returns what ``router_gemm``, then ``moe_routing``, then the slot sort inside ``tlx.ops.moe`` return, bit for bit:
    ``logits = router_gemm(x, weight)`` (fp32, exact products), ``(topk_weights, topk_ids) = moe_routing(logits,
    correction_bias, topk, ...)``, and the routed slots of the experts ``[local_expert_start, local_expert_start +
    num_local_experts)`` sorted by local expert into ``block_m``-padded blocks. It runs as three launches: the top-k
    kernel adds the router GEMM's split-K partials itself, so the GEMM's reduction launch and the logits round trip
    are gone. The logits are not returned.

    Returns ``(topk_weights, topk_ids, sorted_ids, expert_ids, num_tokens_post_padded)``: fp32 and int32 ``(T, topk)``;
    ``sorted_ids`` int32 of length ``cap = ceil((T * topk + num_local_experts * (block_m - 1)) / block_m) * block_m``,
    holding flat slot indices ``token * topk + slot``, in slot order within an expert, and ``T * topk`` in padding
    positions; ``expert_ids`` int32 ``(cap / block_m,)``, the local expert of each block, ``-1`` past the last; and
    ``num_tokens_post_padded`` int32 ``(1,)``, the number of positions in used blocks. Slots routed to other ranks'
    experts are left out. Shapes and dtypes otherwise as ``router_gemm`` and ``moe_routing``; ``block_m`` must be a
    power of two of at least 16. Graph-capturable: nothing is read on the host.
    Only ``space="heuristic"`` is provided.
    """
    import torch

    if not all(isinstance(t, torch.Tensor) for t in (x, weight, correction_bias)):
        raise InvalidInput("tlx.ops.moe_router_topk_align expects tensors x, weight and correction_bias")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1] or correction_bias.shape != weight.shape[:1]:
        raise InvalidInput("tlx.ops.moe_router_topk_align expects x (T, K), weight (E, K) and correction_bias (E,); "
                           f"got {tuple(x.shape)}, {tuple(weight.shape)} and {tuple(correction_bias.shape)}")
    T, K = x.shape
    E = weight.shape[0]
    if (x.dtype != torch.bfloat16 or weight.dtype != torch.float32 or correction_bias.dtype != torch.float32
            or not weight.is_contiguous() or not correction_bias.is_contiguous() or (T > 1 and x.stride(0) < K)
            or (K > 1 and x.stride(1) != 1)):
        raise InvalidInput("tlx.ops.moe_router_topk_align expects bf16 x with unit column stride and rows that do not "
                           "overlap, a contiguous fp32 weight and a contiguous fp32 correction_bias")
    for name, value in (("topk", topk), ("num_local_experts", num_local_experts),
                        ("local_expert_start", local_expert_start), ("block_m", block_m)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidInput(f"tlx.ops.moe_router_topk_align expects an int {name}; got {value!r}")
    if not 1 <= topk <= min(E, _MOE_ROUTING_MAX_TOPK):
        raise InvalidInput(f"tlx.ops.moe_router_topk_align supports 1 <= topk <= min(E, {_MOE_ROUTING_MAX_TOPK}); got "
                           f"topk={topk}, E={E}")
    if not 1 <= num_local_experts <= E or not 0 <= local_expert_start <= E - num_local_experts:
        raise InvalidInput(f"tlx.ops.moe_router_topk_align expects local experts within [0, {E}); got start "
                           f"{local_expert_start} and count {num_local_experts}")
    if block_m < 16 or block_m & (block_m - 1):
        raise InvalidInput(f"tlx.ops.moe_router_topk_align expects a power-of-two block_m >= 16; got {block_m}")
    for name, value in (("router_multiplier", router_multiplier), ("route_scale", route_scale), ("eps", eps)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidInput(f"tlx.ops.moe_router_topk_align expects a Python number for {name}; got {value!r}")
    if any(t.device != x.device for t in (weight, correction_bias)):
        raise InvalidInput("tlx.ops.moe_router_topk_align expects all tensors on one device")

    space_name, invariance = _normalize_space_arg("moe_router_topk_align", space)
    _check_single_config_space("moe_router_topk_align", space_name)
    fn, spec = impl_for("moe_router_topk_align", device=x.device)
    check_inputs(spec, dtype=x.dtype, E=E, K=K, topk=topk)
    check_backward(spec, x, weight, correction_bias)
    _check_enforced("moe_router_topk_align", spec, invariance)
    with torch.cuda.device(x.device):
        return fn(x, weight, correction_bias, topk, num_local_experts, local_expert_start, block_m, router_multiplier,
                  route_scale, eps)


def moe_mxfp4_sort_quant(x, topk_ids, *, num_local_experts, local_expert_start=0, num_experts, block_m=32, space=None):
    """The decode-sized prologue of an MXFP4 (A4W4) routed MoE in one launch: the expert-aligned slot sort, the
    per-row MXFP4 quantization of the activations and the scale sort that AITER's A4W4 stage kernels read. Forward
    only.

    Bit-exact with AITER's ``moe_sorting`` (with the expert mask of the local experts), ``dynamic_mxfp4_quant``
    (round-up amax, scale ``2^(floor(log2(amax)) - 2)``, RNE E2M1, unshuffled) and ``moe_mxfp4_sort`` on every byte the
    stage kernels read. Returns ``(x_fp4, sorted_ids, sorted_expert_ids, num_valid_ids, sorted_scales)``:

    - ``x_fp4`` uint8 ``(T, K / 2)``, two E2M1 values per byte (even element in the low nibble), in token order;
    - ``sorted_ids`` int32 of length ``cap = T * topk + num_experts * block_m - topk`` (AITER's allocation), the routed
      slots of experts ``[local_expert_start, local_expert_start + num_local_experts)`` grouped by local expert
      (ascending) into ``block_m``-row blocks, tokens ascending within an expert, each entry ``token | slot << 24`` and
      padding ``T | topk << 24``; entries past ``num_valid_ids[0]`` are not written;
    - ``sorted_expert_ids`` int32 ``(ceil(cap / block_m),)``, the local expert of each used block;
    - ``num_valid_ids`` int32 ``(2,)``: the number of positions in used blocks, and ``T``;
    - ``sorted_scales`` uint8 ``(ceil(cap / 32) * 32, K / 32)``, the E8M0 scale of each sorted row in
      ``moe_mxfp4_sort``'s 32 x 8 tiles; rows of padding slots are not written (the GEMMs never use them for a real
      token's output).

    ``x`` is bf16 ``(T, K)`` with unit column stride, ``1 <= T <= 64`` and ``K % 256 == 0``; ``topk_ids`` int32
    ``(T, topk)`` with ``1 <= topk <= 8``, contiguous global expert ids, a token's ids distinct, ids outside the local
    range skipped; ``1 <= num_local_experts <= 64``; ``block_m`` is 32. Graph-capturable: launch shapes depend only on
    ``T``, nothing is read on the host. Rows are quantized independently, so a row's bytes do not depend on the batch.
    Only ``space="heuristic"`` is provided.
    """
    import torch

    if not isinstance(x, torch.Tensor) or not isinstance(topk_ids, torch.Tensor):
        raise InvalidInput("tlx.ops.moe_mxfp4_sort_quant expects tensors x and topk_ids")
    if x.ndim != 2 or topk_ids.ndim != 2 or topk_ids.shape[0] != x.shape[0]:
        raise InvalidInput("tlx.ops.moe_mxfp4_sort_quant expects x (T, K) and topk_ids (T, topk); got "
                           f"{tuple(x.shape)} and {tuple(topk_ids.shape)}")
    T, K = x.shape
    topk = topk_ids.shape[1]
    if (x.dtype != torch.bfloat16 or (K > 1 and x.stride(1) != 1) or (T > 1 and x.stride(0) < K)
            or topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous()):
        raise InvalidInput("tlx.ops.moe_mxfp4_sort_quant expects bf16 x with unit column stride and rows that do not "
                           "overlap, and contiguous int32 topk_ids")
    for name, value in (("num_local_experts", num_local_experts), ("local_expert_start", local_expert_start),
                        ("num_experts", num_experts), ("block_m", block_m)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidInput(f"tlx.ops.moe_mxfp4_sort_quant expects an int {name}; got {value!r}")
    if not 1 <= num_local_experts <= num_experts or not 0 <= local_expert_start <= num_experts - num_local_experts:
        raise InvalidInput(f"tlx.ops.moe_mxfp4_sort_quant expects local experts within [0, {num_experts}); got start "
                           f"{local_expert_start} and count {num_local_experts}")
    if topk_ids.device != x.device:
        raise InvalidInput("tlx.ops.moe_mxfp4_sort_quant expects all tensors on one device")

    space_name, invariance = _normalize_space_arg("moe_mxfp4_sort_quant", space)
    _check_single_config_space("moe_mxfp4_sort_quant", space_name)
    fn, spec = impl_for("moe_mxfp4_sort_quant", device=x.device)
    check_inputs(spec, dtype=x.dtype, T=T, K=K, topk=topk, E_local=num_local_experts, block_m=block_m)
    check_backward(spec, x)
    _check_enforced("moe_mxfp4_sort_quant", spec, invariance)
    with torch.cuda.device(x.device):
        return fn(x, topk_ids, local_expert_start, num_local_experts, num_experts, block_m)


def qk_norm_rope_kv_cache(q, k, v, k_cache, v_cache, slot_mapping, *, eps, q_scale=1.0, positions=None,
                          cos_sin_cache=None, rope_bf16=True, slot_remap=None, k_scale=None, v_scale=None,
                          q_out_scale=None, q_out=None, space=None):
    """Attention input preparation in one launch: q/k head RMSNorm, RoPE, the paged FP8 KV cache write and the q cast.
    Forward only.

    Per token ``t``, with ``D`` the head dim:

    - every head of ``q`` and ``k`` gets a weightless RMSNorm in FP32 (``x * rsqrt(mean(x * x) + eps)``), rounded to
      BF16; ``q`` is then multiplied by ``q_scale`` and rounded to BF16 again;
    - with ``positions``, RoPE on interleaved pairs ``(x[2i], x[2i + 1])`` of every head, ``cos``/``sin`` taken from
      row ``positions[t]`` of the FP32 table ``cos_sin_cache (max_position, D)`` (``cos`` in the first ``D / 2``
      columns). ``rope_bf16=True`` rounds ``cos``, ``sin`` and every product and sum to BF16, as rotary-embedding
      kernels that compute in the activation dtype do; ``rope_bf16=False`` computes each output in FP32 and rounds
      once;
    - ``k`` (normed and rotated) and ``v`` are divided by ``k_scale``/``v_scale`` (if given) in FP32, converted to
      ``float8_e4m3fn`` (round to nearest even, saturating at +-448) and written to ``cache[s // page, s % page]``
      with ``s = slot_mapping[t]``, or ``s = slot_remap[slot_mapping[t]]`` with ``slot_remap``. A negative slot, before
      or after the remap, writes nothing;
    - returns ``q`` in BF16, or with ``q_out_scale`` in ``float8_e4m3fn`` as ``clamp(q * rcp(q_out_scale), -448,
      448)`` rounded to nearest even, ``rcp`` being the hardware reciprocal.

    Shapes: ``q (T, HQ * D)``, ``k`` and ``v (T, HK * D)`` BF16 with unit column stride and any row stride (column
    slices of a fused QKV projection work); ``k_cache`` and ``v_cache (num_blocks, page, HK, D)`` ``float8_e4m3fn``,
    contiguous within a block and with one block stride; ``slot_mapping (T,)``, ``positions (T,)`` and ``slot_remap``
    int32 or int64; ``k_scale``, ``v_scale`` and ``q_out_scale`` one-element FP32 tensors. ``HQ``, ``HK`` and ``D`` are
    powers of two with ``16 <= D <= 256`` and ``HQ * D``, ``HK * D`` at most 4096. Slots are not range checked; nothing
    is read on the host, so the call is graph-capturable. Every token is computed alike, so a token's results do not
    depend on the batch. Only ``space="heuristic"`` is provided.
    """
    import torch

    tensors = dict(q=q, k=k, v=v, k_cache=k_cache, v_cache=v_cache, slot_mapping=slot_mapping)
    if not all(isinstance(t, torch.Tensor) for t in tensors.values()):
        raise InvalidInput("tlx.ops.qk_norm_rope_kv_cache expects tensors q, k, v, k_cache, v_cache and slot_mapping")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise InvalidInput("tlx.ops.qk_norm_rope_kv_cache expects k_cache and v_cache of one shape (num_blocks, page, "
                           f"HK, D); got {tuple(k_cache.shape)} and {tuple(v_cache.shape)}")
    _, page, HK, D = k_cache.shape
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or D == 0 or q.shape[1] % D or k.shape != (q.shape[0], HK * D) or (
            v.shape != k.shape) or slot_mapping.shape != (q.shape[0], ):
        raise InvalidInput(
            "tlx.ops.qk_norm_rope_kv_cache expects q (T, HQ * D), k and v (T, HK * D) and "
            f"slot_mapping (T,) for caches of {HK} heads of {D}; got {tuple(q.shape)}, {tuple(k.shape)}, "
            f"{tuple(v.shape)} and {tuple(slot_mapping.shape)}")
    T, HQ = q.shape[0], q.shape[1] // D
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype != torch.bfloat16 or (t.shape[1] > 1 and t.stride(1) != 1) or (T > 1 and t.stride(0) < t.shape[1]):
            raise InvalidInput(f"tlx.ops.qk_norm_rope_kv_cache expects {name} in bfloat16 with unit column stride and "
                               "rows that do not overlap")
    for name, t in (("k_cache", k_cache), ("v_cache", v_cache)):
        if t.dtype != torch.float8_e4m3fn or t.stride()[1:] != (HK * D, D, 1) or t.stride(0) < page * HK * D:
            raise InvalidInput(f"tlx.ops.qk_norm_rope_kv_cache expects {name} in float8_e4m3fn, contiguous within a "
                               "block")
    if v_cache.stride(0) != k_cache.stride(0):
        raise InvalidInput("tlx.ops.qk_norm_rope_kv_cache expects k_cache and v_cache with one block stride")
    index_dtypes = (torch.int32, torch.int64)
    if slot_mapping.dtype not in index_dtypes or (T > 1 and slot_mapping.stride(0) != 1):
        raise InvalidInput("tlx.ops.qk_norm_rope_kv_cache expects a contiguous int32/int64 slot_mapping")
    if slot_remap is not None:
        if (not isinstance(slot_remap, torch.Tensor) or slot_remap.ndim != 1 or slot_remap.dtype not in index_dtypes
                or (slot_remap.numel() > 1 and slot_remap.stride(0) != 1)):
            raise InvalidInput("tlx.ops.qk_norm_rope_kv_cache expects slot_remap as a contiguous 1-D int32/int64 "
                               "tensor")
        tensors["slot_remap"] = slot_remap
    if (positions is None) != (cos_sin_cache is None):
        raise InvalidInput("tlx.ops.qk_norm_rope_kv_cache expects positions and cos_sin_cache together")
    if positions is not None:
        if (not isinstance(positions, torch.Tensor) or positions.shape != (T, ) or positions.dtype not in index_dtypes
                or (T > 1 and positions.stride(0) != 1)):
            raise InvalidInput(f"tlx.ops.qk_norm_rope_kv_cache expects positions as a contiguous int32/int64 ({T},) "
                               "tensor")
        if (not isinstance(cos_sin_cache, torch.Tensor) or cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != D
                or cos_sin_cache.dtype != torch.float32 or cos_sin_cache.stride(1) != 1):
            raise InvalidInput(f"tlx.ops.qk_norm_rope_kv_cache expects cos_sin_cache as an fp32 (max_position, {D}) "
                               "tensor with unit column stride")
        tensors.update(positions=positions, cos_sin_cache=cos_sin_cache)
    if (k_scale is None) != (v_scale is None):
        raise InvalidInput("tlx.ops.qk_norm_rope_kv_cache expects k_scale and v_scale together")
    for name, t in (("k_scale", k_scale), ("v_scale", v_scale), ("q_out_scale", q_out_scale)):
        if t is not None:
            if not isinstance(t, torch.Tensor) or t.numel() != 1 or t.dtype != torch.float32:
                raise InvalidInput(f"tlx.ops.qk_norm_rope_kv_cache expects {name} as a one-element fp32 tensor")
            tensors[name] = t
    for name, value in (("eps", eps), ("q_scale", q_scale)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidInput(f"tlx.ops.qk_norm_rope_kv_cache expects a Python number for {name}; got {value!r}")
    if q_out is not None:
        dtype = torch.float8_e4m3fn if q_out_scale is not None else torch.bfloat16
        if (not isinstance(q_out, torch.Tensor) or q_out.shape != (T, HQ * D) or q_out.dtype != dtype
                or not q_out.is_contiguous()):
            raise InvalidInput(f"tlx.ops.qk_norm_rope_kv_cache expects q_out as a contiguous {dtype} ({T}, {HQ * D}) "
                               "tensor")
        tensors["q_out"] = q_out
    if not torch.compiler.is_compiling():  # overlap checks do not trace
        for out_name in ("k_cache", "v_cache", "q_out"):
            for name, t in tensors.items():
                if out_name in tensors and name != out_name and torch._C._overlaps(tensors[out_name], t):
                    raise InvalidInput(f"tlx.ops.qk_norm_rope_kv_cache {out_name} must not alias {name}")
    if any(t.device != q.device for t in tensors.values()):
        raise InvalidInput("tlx.ops.qk_norm_rope_kv_cache expects all tensors on one device")

    space_name, invariance = _normalize_space_arg("qk_norm_rope_kv_cache", space)
    _check_single_config_space("qk_norm_rope_kv_cache", space_name)
    fn, spec = impl_for("qk_norm_rope_kv_cache", device=q.device)
    check_inputs(spec, dtype=q.dtype, HQ=HQ, HK=HK, D=D)
    check_backward(spec, *tensors.values())
    _check_enforced("qk_norm_rope_kv_cache", spec, invariance)
    with torch.cuda.device(q.device):
        return fn(q, k, v, k_cache, v_cache, slot_mapping, eps, q_scale, positions, cos_sin_cache, rope_bf16,
                  slot_remap, k_scale, v_scale, q_out_scale, q_out)


def rmsnorm_gate(x, gate, *, head_dim=64, eps=1e-6, out=None, space=None):
    """Per-head RMSNorm without a weight, times a sigmoid gate. Forward only.

    For each token and each ``head_dim``-element head of ``x``, in FP32:
    ``out = bf16(x * rstd * sigmoid(gate))`` with ``rstd = 1 / sqrt(sum(x * x) / head_dim + eps)``. The sum adds
    adjacent element pairs and then folds the pair sums in halves (pair ``i`` plus pair ``i + 16``, then ``+ 8``,
    ``+ 4``, ``+ 2``, ``+ 1``), the order of a plain Triton kernel with 32 two-element lanes per head, so the result
    is bitwise that kernel's. A row's result does not depend on the batch.

    Shapes: ``x`` and ``gate`` BF16 ``(T, H * head_dim)``, each with a contiguous last dimension and any row stride
    (``gate`` can be a column slice of a wider projection output). ``head_dim`` must be 64 and ``H`` a multiple of 8.
    Returns a BF16 ``(T, H * head_dim)`` tensor; ``out`` (contiguous last dimension, any row stride, not overlapping
    the inputs) is filled and returned when passed. Only ``space="heuristic"`` is provided.
    """
    import torch

    if not isinstance(x, torch.Tensor) or not isinstance(gate, torch.Tensor):
        raise InvalidInput("tlx.ops.rmsnorm_gate expects tensor x and gate")
    if (x.ndim != 2 or gate.shape != x.shape or x.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16
            or x.stride(1) != 1 or gate.stride(1) != 1):
        raise InvalidInput("tlx.ops.rmsnorm_gate expects BF16 x and gate of one (T, N) shape with contiguous rows; "
                           f"got {tuple(x.shape)} {x.dtype} and {tuple(gate.shape)} {gate.dtype}")
    T, N = x.shape
    if not isinstance(head_dim, int) or head_dim <= 0 or N % head_dim:
        raise InvalidInput(f"tlx.ops.rmsnorm_gate expects head_dim to divide the row width {N}; got {head_dim!r}")
    if not 0 < eps < math.inf:
        raise InvalidInput(f"tlx.ops.rmsnorm_gate expects a positive finite eps; got {eps!r}")
    outputs = []
    if out is not None:
        if (not isinstance(out, torch.Tensor) or out.shape != (T, N) or out.dtype != torch.bfloat16
                or out.stride(1) != 1):
            raise InvalidInput(f"tlx.ops.rmsnorm_gate expects a BF16 out of shape ({T}, {N}) with contiguous rows")
        if any(torch._C._overlaps(out, t) for t in (x, gate)):
            raise InvalidInput("tlx.ops.rmsnorm_gate out must not alias x or gate")
        outputs.append(out)
    if any(t.device != x.device for t in (gate, *outputs)):
        raise InvalidInput("tlx.ops.rmsnorm_gate expects all tensors on one device")

    space_name, invariance = _normalize_space_arg("rmsnorm_gate", space)
    _check_single_config_space("rmsnorm_gate", space_name)
    fn, spec = impl_for("rmsnorm_gate", device=x.device)
    check_inputs(spec, dtype=x.dtype, head_dim=head_dim, H=N // head_dim)
    check_backward(spec, x, gate, *outputs)
    _check_enforced("rmsnorm_gate", spec, invariance)
    with torch.cuda.device(x.device):
        return fn(x, gate, float(eps), out)


def addmm(input, a, b, *, out=None, space=None):
    """Fused ``input + a @ b`` for two-dimensional fp16/bf16 matrices.

    ``input`` may be ``(N,)`` or two-dimensional and broadcastable to the
    ``(M, N)`` result. Matrix and input scale factors are both one. This op is
    currently forward-only.

    `space=None` uses the implementation's default: "full" on sm90, otherwise
    "heuristic". ``space=`` also accepts a ``Space``; its invariance request is
    validated but not enforced yet.
    """
    space_name, invariance = _normalize_space_arg("addmm", space)
    fn, spec = impl_for("addmm", device=a.device)
    if space_name is None:
        space_name = spec.default_space
    if space_name not in ("heuristic", "full", "smoke", "heuristic_2cta"):
        raise InvalidInput(f"tlx.ops.addmm does not provide space={space_name!r}")
    check_inputs(spec, dtype=a.dtype)
    check_backward(spec, input, a, b)
    _check_enforced("addmm", spec, invariance)
    return fn(input, a, b, out=out, space=space_name)


def addmm_mul_add(bias, a, b, x0, res=None, *, out=None, space="heuristic"):
    """Fused ``x0 * (bias + a @ b) + res`` (``res`` defaults to ``x0``).

    The cross layer of a low-rank DCN: ``a`` is ``(M, K)``, ``b`` is
    ``(K, N)`` (``linear.weight.t()`` for an ``nn.Linear``), ``bias`` is
    ``(N,)`` or ``None``, and ``x0``, ``res`` and ``out`` are ``(M, N)``.
    ``out`` may alias ``x0`` or ``res``. The linear output is rounded to the
    input dtype before the mul/add, as in the unfused sequence. fp16/bf16,
    forward-only.
    """
    space_name, invariance = _normalize_space_arg("addmm_mul_add", space)
    if space_name not in ("heuristic", "full"):
        raise InvalidInput(f"tlx.ops.addmm_mul_add does not provide space={space_name!r}")
    fn, spec = impl_for("addmm_mul_add", device=a.device)
    check_inputs(spec, dtype=a.dtype)
    check_backward(spec, bias, a, b, x0, res)
    _check_enforced("addmm_mul_add", spec, invariance)
    return fn(bias, a, b, x0, res, out=out, space=space_name)


def addmm_mul_add_launch(bias, a, b, x0, res=None, *, out):
    """The heuristic launch of ``addmm_mul_add`` as ``(kernel, grid, args, kwargs)``.

    For callers that must launch the Triton kernel themselves, e.g. a
    ``torch.library.triton_op`` that runs ``wrap_triton(kernel)[grid](*args,
    **kwargs)`` so that AOTInductor compiles it into the exported artifact.
    Same operands and result as ``addmm_mul_add(..., out=out)``; ``out`` is
    required. Works on fake tensors with a symbolic M.
    """
    fn, spec = impl_for("addmm_mul_add_launch", device=a.device)
    check_inputs(spec, dtype=a.dtype)
    check_backward(spec, bias, a, b, x0, res)
    return fn(bias, a, b, x0, res, out=out)


def flash_attn(q, k, v, causal=False, sm_scale=None, *, space=None):
    """Fused BHSD attention with architecture-specific dtype and shape support.

    gfx1250 supports noncausal D128 FP16/BF16 inference, including cross attention.
    Its default space is "heuristic"; "smoke" selects the same configuration.
    sm90, sm100, and gfx950 support backward and retain the "full" default.
    `sm_scale` defaults to `HEAD_DIM ** -0.5`.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise InvalidInput("tlx.ops.flash_attn expects rank-4 Q/K/V tensors")
    space_name, invariance = _normalize_space_arg("flash_attn", space)
    fn, spec = impl_for("flash_attn", device=q.device)
    if space_name is None:
        space_name = spec.default_space
    if space_name not in ("full", "smoke", spec.default_space):
        raise InvalidInput(f"tlx.ops.flash_attn does not provide space={space_name!r}")
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1])
    check_backward(spec, q, k, v)
    _check_enforced("flash_attn", spec, invariance)
    # Triton's launcher uses the process' current device rather than deriving it
    # from pointer arguments. Keep compilation and launch on the input device;
    # the context manager restores the caller's current device afterwards.
    import torch
    with torch.cuda.device(q.device):
        return fn(q, k, v, causal, sm_scale, space=space_name)


def prepare_flash_attn_varlen_backward(cu_seqlens_q, cu_seqlens_k, *, space=None):
    """Prepare reusable packed-attention backward metadata from int32 offsets.

    The two rank-1 CUDA tensors must describe the same nonempty batch. Their
    offsets must start at zero and be strictly increasing; empty sequences
    are unsupported. Preparation copies offsets into plan-owned storage and
    synchronizes to validate them and determine compact workspace sizes.
    Call this once outside CUDA graph capture, then reuse the returned plan.

    Prepare and consume the plan on the same CUDA stream. Treat every tensor
    owned by the plan as immutable; the caller's original offsets may be
    modified after preparation. The plan also enables eligible compact dS
    backward implementations without repeating host synchronization.
    ``space=`` takes no search space today; a ``Space`` is accepted only to
    carry a ``BitwiseInvarianceSpec``, which is validated but not enforced yet.
    """
    import torch

    for name, offsets in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if not isinstance(offsets, torch.Tensor) or offsets.ndim != 1 or offsets.numel() < 2:
            raise InvalidInput(f"{name} must be a rank-1 tensor with at least two elements")
        if offsets.dtype is not torch.int32:
            raise InvalidInput(f"{name} must have dtype torch.int32")
    if cu_seqlens_q.device != cu_seqlens_k.device:
        raise InvalidInput("cu_seqlens_q and cu_seqlens_k must be on the same device")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise InvalidInput("cu_seqlens_q and cu_seqlens_k must describe the same batch")
    space_name, invariance = _normalize_space_arg("prepare_flash_attn_varlen_backward", space)
    _check_single_config_space("prepare_flash_attn_varlen_backward", space_name)
    fn, spec = impl_for("prepare_flash_attn_varlen_backward", device=cu_seqlens_q.device)
    check_inputs(spec, dtype=cu_seqlens_q.dtype)
    _check_enforced("prepare_flash_attn_varlen_backward", spec, invariance)
    with torch.cuda.device(cu_seqlens_q.device):
        if torch.cuda.is_current_stream_capturing():
            raise InvalidInput("prepare_flash_attn_varlen_backward must run outside CUDA graph capture")
        try:
            return fn(cu_seqlens_q, cu_seqlens_k)
        except (TypeError, ValueError) as error:
            raise InvalidInput(f"tlx.ops.prepare_flash_attn_varlen_backward: {error}") from error


def flash_attn_varlen_backward(q, k, v, o, do, lse, plan, sm_scale=None, causal=False, *, dq_atomic_fp32=True,
                               space=None):
    """Return explicit first-order dQ/dK/dV for packed BF16 D128 attention.

    Q/K/V use ``(tokens, heads, head_dim)`` order. Supply the existing forward
    output ``o``, its gradient ``do``, and contiguous natural-log softmax LSE
    with shape ``(query_heads, query_tokens)`` and dtype FP32. The forward
    must use the same scale and causal flag, with no dropout or local window.
    ``sm_scale`` defaults to ``head_dim ** -0.5``.

    Use a plan returned by :func:`prepare_flash_attn_varlen_backward` on the
    same device and CUDA stream, and do not modify its tensors. Noncausal
    MHA/GQA allows independent Q/KV lengths. Causal attention requires MHA
    with identical Q/KV sequence offsets. Q/K/O/dO must be contiguous. V must
    be contiguous in noncausal mode; causal V may have a larger token stride
    when its head and head-dimension axes remain dense.

    Inputs may require gradients. This call builds no autograd graph and its
    BF16 gradient outputs do not support higher-order differentiation.
    Deterministic algorithms are unsupported. dQ accumulation uses FP32 by
    default; ``dq_atomic_fp32=False`` selects the BF16 accumulation path.
    Eligible eager calls may allocate an optional materialized-dS workspace;
    CUDA graph capture or its allocation OOM uses the existing atomic route.
    """
    import torch

    tensors = (q, k, v, o, do, lse)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise InvalidInput("flash_attn_varlen_backward expects tensor Q/K/V/O/dO/LSE inputs")
    if any(tensor.ndim != 3 for tensor in (q, k, v, o, do)):
        raise InvalidInput("flash_attn_varlen_backward expects rank-3 packed THD Q/K/V/O/dO tensors")
    if any(tensor.device != q.device for tensor in tensors):
        raise InvalidInput("flash_attn_varlen_backward inputs must be on the same device")
    if torch.are_deterministic_algorithms_enabled():
        raise UnsupportedOp("flash_attn_varlen_backward does not provide deterministic gradients")
    space_name, invariance = _normalize_space_arg("flash_attn_varlen_backward", space)
    _check_single_config_space("flash_attn_varlen_backward", space_name)
    fn, spec = impl_for("flash_attn_varlen_backward", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1])
    _check_enforced("flash_attn_varlen_backward", spec, invariance)
    scale = q.shape[-1]**-0.5 if sm_scale is None else sm_scale
    with torch.cuda.device(q.device), torch.no_grad():
        try:
            return fn(q, k, v, o, do, lse, plan, scale, causal, dq_atomic_fp32=dq_atomic_fp32)
        except (TypeError, ValueError) as error:
            raise InvalidInput(f"tlx.ops.flash_attn_varlen_backward: {error}") from error


def flash_attn_mxfp8(q, k, v, causal=False, sm_scale=None, *, space="full", p_expcast=False):
    """Differentiable MXFP8 attention over contiguous BF16 Q/K/V.

    Inputs have identical [batch, heads, sequence, 128] shapes with sequence
    lengths divisible by 256. Blackwell and gfx950 return BF16 gradients;
    gfx1250 is forward only. Backward computes gradients only for inputs with
    ``requires_grad=True``. It skips unrequested gradient outputs and operations
    used only by those gradients, including quantization used only by them.
    All three use 32x32 Q/K blocks. On gfx950, calls
    where autograd records Q, K or V use a training forward with hardware exp2
    that saves separate row maxima and log2 normalization terms, the statistics
    backward recomputes P from. Other calls use the faster inference ping-pong
    kernel (non-causal, and causal at sequence lengths divisible by 512), whose
    approximate exp2 makes its output differ from the training forward by
    about 0.04 relative RMS, less than either differs from BF16 attention.
    gfx1250 quantizes as Blackwell does and, like gfx950's ping-pong kernel,
    computes P with an approximate exp2. See the kernel module for the recipes.
    ``p_expcast=True`` (Blackwell only) encodes the forward probabilities with
    ExpCast-FP8 instead of exp2 (see kernels/flash_attn_mxfp8/sm100.py).
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects rank-4 Q/K/V tensors")
    if q.shape != k.shape or q.shape != v.shape:
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects Q/K/V to have identical shapes; "
                           f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
    if any(size == 0 for size in q.shape):
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects nonempty Q/K/V dimensions")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.device != k.device or q.device != v.device:
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects Q/K/V to have the same dtype and device")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects contiguous Q/K/V tensors")
    space_name, invariance = _normalize_space_arg("flash_attn_mxfp8", space)
    if space_name not in ("full", "smoke"):
        raise InvalidInput(f"tlx.ops.flash_attn_mxfp8 does not provide space={space_name!r}")
    fn, spec = impl_for("flash_attn_mxfp8", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1], N_CTX=q.shape[-2])
    check_backward(spec, q, k, v)
    _check_enforced("flash_attn_mxfp8", spec, invariance)
    import torch
    with torch.cuda.device(q.device):
        return fn(q, k, v, causal, sm_scale, space=space_name, p_expcast=p_expcast)


def flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale=None, causal=False, *,
                      window_size=(-1, -1), space=None):
    """Attention over packed ``(tokens, heads, HEAD_DIM)`` fp16/bf16 sequences.

    Sequence ``b`` owns query rows ``cu_seqlens_q[b]:cu_seqlens_q[b + 1]`` and
    key/value rows ``cu_seqlens_k[b]:cu_seqlens_k[b + 1]``; both offset tensors
    are int32 ``(batch + 1,)``. ``max_seqlen_q`` must bound every query length
    (it sizes the grid); ``max_seqlen_k`` must bound every key length on sm100
    (it sizes the backward grid) and is otherwise accepted for flash-attn API
    parity. K/V may have fewer heads than Q (GQA) if they divide its head count.
    Causal masking is bottom-right aligned: query ``i`` sees keys
    ``j <= i + k_len - q_len``, and a query row that sees no key returns zeros.
    ``window_size=(left, right)`` (sm100 only) further limits query ``i`` to keys
    ``i + k_len - q_len - left <= j <= i + k_len - q_len + right``, as in
    flash-attn; ``-1`` leaves a side unbounded and ``causal`` sets ``right`` to 0.
    sm100 is differentiable; sm90 and gfx950 are forward-only.
    `sm_scale` defaults to `HEAD_DIM ** -0.5` and must be positive and finite.
    Returns the output only (no log-sum-exp). Output rows not covered by ``cu_seqlens_q`` are left
    uninitialized.
    """
    import torch
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise InvalidInput("tlx.ops.flash_attn_varlen expects rank-3 (tokens, heads, head_dim) Q/K/V tensors")
    if k.shape != v.shape or q.shape[2] != k.shape[2] or q.shape[1] % k.shape[1] != 0:
        raise InvalidInput("tlx.ops.flash_attn_varlen expects K/V of identical shape, Q's head dim, and a head count "
                           f"dividing Q's; got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise InvalidInput("tlx.ops.flash_attn_varlen expects Q/K/V to have the same dtype")
    if q.stride(2) != 1 or k.stride(2) != 1 or v.stride(2) != 1:
        raise InvalidInput("tlx.ops.flash_attn_varlen expects unit-stride head dims")
    if (cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape != cu_seqlens_k.shape or cu_seqlens_q.shape[0] < 1
            or cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32):
        raise InvalidInput("tlx.ops.flash_attn_varlen expects int32 (batch + 1,) cu_seqlens_q and cu_seqlens_k")
    if any(t.device != q.device for t in (k, v, cu_seqlens_q, cu_seqlens_k)):
        raise InvalidInput("tlx.ops.flash_attn_varlen expects all tensors on one device")
    space_name, invariance = _normalize_space_arg("flash_attn_varlen", space)
    _check_single_config_space("flash_attn_varlen", space_name)
    if sm_scale is not None and not 0 < sm_scale < math.inf:
        raise InvalidInput(f"tlx.ops.flash_attn_varlen expects a positive finite sm_scale; got {sm_scale!r}")
    window_size = tuple(int(w) for w in window_size)
    if len(window_size) != 2 or min(window_size) < -1:
        raise InvalidInput(f"tlx.ops.flash_attn_varlen expects window_size=(left, right) with -1 for unbounded; "
                           f"got {window_size!r}")
    fn, spec = _flash_attn_varlen_impl_for_device(q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[2])
    check_backward(spec, q, k, v)
    _check_enforced("flash_attn_varlen", spec, invariance)
    # Entering torch.cuda.device costs microseconds per call; skip it when q is already on the current device.
    on_current = q.device.index == torch.cuda.current_device()
    with contextlib.nullcontext() if on_current else torch.cuda.device(q.device):
        if spec.supports_window:
            return fn(q, k, v, cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q), sm_scale, bool(causal),
                      max_seqlen_k=int(max_seqlen_k), window_size=window_size)
        if window_size != (-1, -1):
            raise UnsupportedOp(f"tlx.ops.flash_attn_varlen does not support window_size on {spec.arch}")
        return fn(q, k, v, cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q), sm_scale, bool(causal))


def window_attn_varlen(q, k, v, cu_seqlens, window_size, sm_scale=None, *, space=None):
    """Forward-only sliding-window self-attention over packed ``(tokens, heads, HEAD_DIM)`` fp16/bf16 sequences.

    Sequence ``b`` owns rows ``cu_seqlens[b]:cu_seqlens[b + 1]`` of Q, K and V (int32, ``(batch + 1,)``, starting at 0
    and ending at ``tokens``; empty sequences are allowed). With ``window_size=(left, right)``, row ``i`` attends the
    keys ``i - left .. i + right`` of its own sequence, both ends inclusive: FlashAttention's ``window_size`` for
    self-attention, without causal masking. K/V may have fewer heads than Q (GQA) if they divide its head count.
    ``sm_scale`` defaults to ``HEAD_DIM ** -0.5`` and must be positive and finite, with ``sm_scale * log2(e)`` finite
    in float32 (the kernel works in base 2); a scale whose ``sm_scale * log2(e)`` is below the smallest normal float32
    is treated as 0, which gives the visible keys equal weight.
    Supported: ``HEAD_DIM`` in (64, 128), ``0 <= left``, ``0 <= right``, ``left + right <= 256``. Returns the output
    only (no log-sum-exp).
    """
    import torch
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise InvalidInput("tlx.ops.window_attn_varlen expects rank-3 (tokens, heads, head_dim) Q/K/V tensors")
    if (k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or k.shape[1] < 1
            or q.shape[1] % k.shape[1] != 0):
        raise InvalidInput("tlx.ops.window_attn_varlen expects K/V of identical shape, Q's tokens and head dim, and a "
                           f"head count dividing Q's; got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise InvalidInput("tlx.ops.window_attn_varlen expects Q/K/V to have the same dtype")
    if q.stride(2) != 1 or k.stride(2) != 1 or v.stride(2) != 1:
        raise InvalidInput("tlx.ops.window_attn_varlen expects unit-stride head dims")
    if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2 or cu_seqlens.dtype != torch.int32:
        raise InvalidInput("tlx.ops.window_attn_varlen expects int32 (batch + 1,) cu_seqlens with batch >= 1")
    if any(t.device != q.device for t in (k, v, cu_seqlens)):
        raise InvalidInput("tlx.ops.window_attn_varlen expects all tensors on one device")
    if (not isinstance(window_size, (tuple, list)) or len(window_size) != 2
            or any(type(w) is not int or w < 0 for w in window_size) or sum(window_size) > 256):
        raise InvalidInput("tlx.ops.window_attn_varlen expects window_size=(left, right) with non-negative ints and "
                           f"left + right <= 256; got {window_size!r}")
    if sm_scale is not None and not 0 < sm_scale < math.inf:
        raise InvalidInput(f"tlx.ops.window_attn_varlen expects a positive finite sm_scale; got {sm_scale!r}")
    if sm_scale is not None and sm_scale * math.log2(math.e) > 3.4028234663852886e38:
        raise InvalidInput(
            f"tlx.ops.window_attn_varlen: sm_scale * log2(e) overflows float32; got sm_scale={sm_scale!r}")
    space_name, invariance = _normalize_space_arg("window_attn_varlen", space)
    _check_single_config_space("window_attn_varlen", space_name)
    fn, spec = impl_for("window_attn_varlen", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[2])
    check_backward(spec, q, k, v)
    _check_enforced("window_attn_varlen", spec, invariance)
    with torch.cuda.device(q.device):
        return fn(q, k, v, cu_seqlens, tuple(window_size), sm_scale)


def _check_lora(name, proj, d_model, out_features):
    """Check a low-rank projection ``(down, norm, up)``: ``(rank, d_model)``, ``(rank,)``, ``(out_features, rank)``."""
    if not isinstance(proj, (tuple, list)) or len(proj) != 3:
        raise InvalidInput(f"tlx.ops.mla_window_attn expects {name}=(down, norm, up)")
    down, norm, up = proj
    rank = down.shape[0] if down.ndim == 2 else -1
    if (down.shape != (rank, d_model) or norm.shape != (rank, ) or up.shape != (out_features, rank)
            or rank not in (16, 32, 64, 128)):
        raise InvalidInput(f"tlx.ops.mla_window_attn expects {name} weights of shapes (rank, {d_model}), (rank,) "
                           f"and ({out_features}, rank) with rank in (16, 32, 64, 128); got "
                           f"{[tuple(t.shape) for t in proj]}")


def mla_window_attn(x, cu_seqlens, norm_weight, q_proj, kv_proj, gate_proj, *, num_heads, window_size, norm_eps=1e-6,
                    lora_eps=1e-6, space=None):
    """Gated sliding-window self-attention with low-rank (MLA-style) projections, forward only, fp16.

    ``x`` is ``(tokens, d_model)`` packed sequences; sequence ``b`` owns rows ``cu_seqlens[b]:cu_seqlens[b + 1]``
    (int32, ``(batch + 1,)``, starting at 0 and ending at ``tokens``). Each projection is a tuple
    ``(down, norm, up)`` of ``nn.Linear``/RMS-norm weights, and with ``h = rms_norm(x, norm_weight, norm_eps)``,
    ``lora(p) = rms_norm(h @ p.down.T, p.norm, lora_eps) @ p.up.T``:

    - ``q = lora(q_proj)`` and ``gate = sigmoid(lora(gate_proj))`` have ``d_model`` columns;
    - ``lora(kv_proj)`` has ``2 * d_model`` columns, which hold, for each head in turn, that head's k then v.

    Returns ``x + attention(q, k, v) * gate`` with ``num_heads`` heads of ``d_model // num_heads`` columns, scale
    ``head_dim ** -0.5``, and ``window_size=(left, right)``: query ``i`` sees keys ``i - left .. i + right`` of its own
    sequence. Supported: ``d_model`` in (128, 256), ``head_dim`` in (64, 128), ranks in (16, 32, 64, 128),
    ``0 <= left``, ``0 <= right``, ``left + right <= 32``. The projection weights are read by TMA, so their data must
    start on a 16-byte boundary (fresh allocations and parameters do; a view with an odd element offset does not).
    """
    import torch
    if x.ndim != 2 or norm_weight.shape != (x.shape[1], ):
        raise InvalidInput("tlx.ops.mla_window_attn expects x of shape (tokens, d_model) and norm_weight of "
                           f"shape (d_model,); got {tuple(x.shape)} and {tuple(norm_weight.shape)}")
    d_model = x.shape[1]
    if type(num_heads) is not int or num_heads < 1 or d_model % num_heads != 0:
        raise InvalidInput(f"tlx.ops.mla_window_attn expects num_heads dividing d_model; got {num_heads!r}")
    _check_lora("q_proj", q_proj, d_model, d_model)
    _check_lora("kv_proj", kv_proj, d_model, 2 * d_model)
    _check_lora("gate_proj", gate_proj, d_model, d_model)
    weights = (norm_weight, *q_proj, *kv_proj, *gate_proj)
    if any(t.dtype != x.dtype for t in weights):
        raise InvalidInput("tlx.ops.mla_window_attn expects every weight in x's dtype")
    if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2 or cu_seqlens.dtype != torch.int32:
        raise InvalidInput("tlx.ops.mla_window_attn expects int32 (batch + 1,) cu_seqlens with batch >= 1")
    if any(t.device != x.device for t in (cu_seqlens, *weights)):
        raise InvalidInput("tlx.ops.mla_window_attn expects all tensors on one device")
    if (not isinstance(window_size, (tuple, list)) or len(window_size) != 2
            or any(type(w) is not int or w < 0 for w in window_size) or sum(window_size) > 32):
        raise InvalidInput("tlx.ops.mla_window_attn expects window_size=(left, right) with non-negative ints and "
                           f"left + right <= 32; got {window_size!r}")
    space_name, invariance = _normalize_space_arg("mla_window_attn", space)
    _check_single_config_space("mla_window_attn", space_name)
    fn, spec = impl_for("mla_window_attn", device=x.device)
    check_inputs(spec, dtype=x.dtype, D=d_model, HEAD_DIM=d_model // num_heads,
                 base_ptrs=tuple(p[i].data_ptr() for p in (q_proj, kv_proj, gate_proj) for i in (0, 2)))
    check_backward(spec, x, *weights)
    _check_enforced("mla_window_attn", spec, invariance)
    with torch.cuda.device(x.device):
        return fn(x, cu_seqlens, norm_weight, tuple(q_proj), tuple(kv_proj), tuple(gate_proj), num_heads,
                  tuple(window_size), float(norm_eps), float(lora_eps))


def linear_rowscale(x, weight, bias=None, row_scale=None, *, space=None):
    """Linear layer with an optional per-row output scale, forward only.

    Returns ``((x @ weight.T + bias) * row_scale[:, None]).to(x.dtype)`` for ``x`` of shape ``(M, K)``, ``weight`` of
    shape ``(N, K)`` (``nn.Linear`` layout), an optional ``bias`` of ``N`` elements in ``x``'s dtype and an optional
    FP32 ``row_scale`` of ``M`` elements, accumulated in FP32 and rounded once. FP16 or BF16, ``K <= 512`` and ``K``,
    ``N`` multiples of 8. The kernel reads ``x`` and ``weight`` through TMA; one that TMA cannot address in place
    (column stride not 1, row stride or base not 16-byte aligned) is copied first. A row's result does not depend on
    ``M`` or on the other rows. The op has one launch configuration, so ``space`` only selects bitwise-invariance
    checking. For ``torch.compile`` and AOTInductor, launch the kernel through :func:`linear_rowscale_launch` from a
    ``torch.library.triton_op``.
    """
    import torch
    if x.ndim != 2 or weight.ndim != 2 or weight.shape[1] != x.shape[1]:
        raise InvalidInput("tlx.ops.linear_rowscale expects x of shape (M, K) and weight of shape (N, K); got "
                           f"{tuple(x.shape)} and {tuple(weight.shape)}")
    M, K = x.shape
    N = weight.shape[0]
    if weight.dtype != x.dtype or (bias is not None and bias.dtype != x.dtype):
        raise InvalidInput("tlx.ops.linear_rowscale expects weight and bias in x's dtype")
    if bias is not None and bias.numel() != N:
        raise InvalidInput(f"tlx.ops.linear_rowscale expects a bias of {N} elements; got {tuple(bias.shape)}")
    if row_scale is not None and (row_scale.numel() != M or row_scale.dtype != torch.float32):
        raise InvalidInput(f"tlx.ops.linear_rowscale expects an FP32 row_scale of {M} elements; got "
                           f"{tuple(row_scale.shape)} {row_scale.dtype}")
    if any(t is not None and t.device != x.device for t in (weight, bias, row_scale)):
        raise InvalidInput("tlx.ops.linear_rowscale expects all tensors on one device")
    space_name, invariance = _normalize_space_arg("linear_rowscale", space)
    _check_single_config_space("linear_rowscale", space_name)
    fn, spec = impl_for("linear_rowscale", device=x.device)
    check_inputs(spec, dtype=x.dtype, M=M, N=N, K=K)
    check_backward(spec, x, weight, bias, row_scale)
    _check_enforced("linear_rowscale", spec, invariance)
    return fn(x, weight, None if bias is None else bias.reshape(N), None if row_scale is None else row_scale.reshape(M))


def linear_rowscale_launch(x, weight, bias=None, row_scale=None, *, out):
    """The launch of ``linear_rowscale`` as ``(kernel, grid, args, kwargs)``.

    For callers that must launch the Triton kernel themselves, e.g. a ``torch.library.triton_op`` that runs
    ``wrap_triton(kernel)[grid](*args, **kwargs)`` so that AOTInductor compiles it into the exported artifact. Same
    operands and result as ``linear_rowscale``, written to the ``(M, N)`` ``out``; ``bias`` is ``(N,)`` and
    ``row_scale`` ``(M,)``, both contiguous. Nothing is copied: ``x``, ``weight`` and ``out`` must have a unit column
    stride and a 16-byte-aligned base and row stride, or ``InvalidInput`` is raised. Works on fake tensors with a
    symbolic M.
    """
    if x.ndim != 2 or weight.ndim != 2:
        raise InvalidInput("tlx.ops.linear_rowscale_launch expects two-dimensional x and weight")
    fn, spec = impl_for("linear_rowscale_launch", device=x.device)
    check_inputs(spec, dtype=x.dtype, N=weight.shape[0], K=x.shape[1])
    check_backward(spec, x, weight, bias, row_scale)
    return fn(x, weight, bias, row_scale, out=out)


def swiglu_ffn(x, norm_weight, w1, w2, w3, *, eps=1e-5, space=None):
    """RMS-normalized SwiGLU feed-forward with residual, forward only, fp16.

    Returns ``x + (silu(n @ w1.T) * (n @ w3.T)) @ w2.T`` with ``n = rms_norm(x, norm_weight, eps)``, for ``x`` of shape
    ``(tokens, d_model)``, ``norm_weight`` of ``d_model`` elements and ``(d_model, d_model)`` weights in ``nn.Linear``
    layout. Supported: ``d_model`` in (128, 256). The weights are read by TMA, so their data must start on a 16-byte
    boundary (fresh allocations and parameters do; a view with an odd element offset does not).
    """
    import torch
    if x.ndim != 2 or norm_weight.numel() != x.shape[1]:
        raise InvalidInput("tlx.ops.swiglu_ffn expects x of shape (tokens, d_model) and a norm_weight of d_model "
                           f"elements; got {tuple(x.shape)} and {tuple(norm_weight.shape)}")
    d_model = x.shape[1]
    if any(w.shape != (d_model, d_model) for w in (w1, w2, w3)):
        raise InvalidInput(f"tlx.ops.swiglu_ffn expects ({d_model}, {d_model}) weights; got "
                           f"{[tuple(w.shape) for w in (w1, w2, w3)]}")
    if any(t.dtype != x.dtype for t in (norm_weight, w1, w2, w3)):
        raise InvalidInput("tlx.ops.swiglu_ffn expects every weight in x's dtype")
    if any(t.device != x.device for t in (norm_weight, w1, w2, w3)):
        raise InvalidInput("tlx.ops.swiglu_ffn expects all tensors on one device")
    space_name, invariance = _normalize_space_arg("swiglu_ffn", space)
    _check_single_config_space("swiglu_ffn", space_name)
    fn, spec = impl_for("swiglu_ffn", device=x.device)
    check_inputs(spec, dtype=x.dtype, D=d_model, base_ptrs=tuple(w.data_ptr() for w in (w1, w2, w3)))
    check_backward(spec, x, norm_weight, w1, w2, w3)
    _check_enforced("swiglu_ffn", spec, invariance)
    with torch.cuda.device(x.device):
        return fn(x, norm_weight.reshape(d_model), w1, w2, w3, float(eps))


def rmsnorm_cat(parts, weights, *, part_weights=None, eps=1e-5, space=None):
    """RMSNorm over the row-wise concatenation of ``parts``, each optionally RMS-normalized first. Forward only.

    ``parts`` are fp16/bf16 ``[B, n_i]`` or ``[B, R_i, C_i]`` tensors on one device, with any strides; a 3-D part is
    flattened row-major and its ``C_i`` must be a power of two. ``part_weights[i]`` (``R_i * C_i`` elements) normalizes
    part ``i`` with its own RMSNorm first; ``None`` (the default for every part) leaves it as is. For each of the one
    to three ``weights`` (``N = sum R_i * C_i`` elements each) the result is a contiguous ``[B, N]`` tensor in the
    parts' dtype::

        cat = concat_i(part_i / sqrt(mean(part_i^2) + eps) * part_weights[i])
        out = cat / sqrt(mean(cat^2) + eps) * weights[k]

    The math is fp32; the normalized parts are rounded to the I/O dtype before the final norm, whose mean is taken
    over the unrounded values. A row may hold at most 32768 elements, at most three parts may be normalized, and the
    parts' row counts must split into at most six power-of-two pieces in all. Returns a list, one tensor per weight.

    A row's outputs do not depend on the batch size or on the other rows. The same values in another memory layout,
    or misaligned, can run another kernel that sums in another order, so its outputs can differ by an ulp.
    """
    import torch
    if not isinstance(parts, (list, tuple)) or not parts:
        raise InvalidInput("tlx.ops.rmsnorm_cat expects a non-empty list of parts")
    if not isinstance(weights, (list, tuple)) or not 1 <= len(weights) <= 3:
        raise InvalidInput("tlx.ops.rmsnorm_cat expects one to three output weights")
    part_weights = [None] * len(parts) if part_weights is None else list(part_weights)
    if len(part_weights) != len(parts):
        raise InvalidInput("tlx.ops.rmsnorm_cat expects one part weight (or None) per part")
    if any(not isinstance(p, torch.Tensor) or p.ndim not in (2, 3) for p in parts):
        raise InvalidInput("tlx.ops.rmsnorm_cat expects rank-2 [B, n] or rank-3 [B, R, C] parts")
    lead = parts[0]
    if any(p.shape[0] != lead.shape[0] or p.dtype != lead.dtype or p.device != lead.device for p in parts):
        raise InvalidInput("tlx.ops.rmsnorm_cat expects parts with one batch size, dtype and device")
    if any(0 in p.shape[1:] for p in parts):
        raise InvalidInput("tlx.ops.rmsnorm_cat expects every part to have at least one element per row")
    if not 0 < eps < math.inf:
        raise InvalidInput(f"tlx.ops.rmsnorm_cat expects a positive finite eps; got {eps!r}")
    # A 2-D part is one row of n columns, or n rows of one column when n is not a power of two.
    parts = [p if p.ndim == 3 else p.unsqueeze(1 if p.shape[1] & (p.shape[1] - 1) == 0 else 2) for p in parts]
    n = sum(p.shape[1] * p.shape[2] for p in parts)
    for p, w in zip(parts, part_weights):
        if w is not None and (w.numel() != p.shape[1] * p.shape[2] or w.device != lead.device):
            raise InvalidInput("tlx.ops.rmsnorm_cat expects each part weight to match its part's row size and device")
    if any(w.numel() != n or w.device != lead.device for w in weights):
        raise InvalidInput(f"tlx.ops.rmsnorm_cat expects every output weight to hold {n} elements on the parts' device")
    space_name, invariance = _normalize_space_arg("rmsnorm_cat", space)
    _check_single_config_space("rmsnorm_cat", space_name)
    fn, spec = impl_for("rmsnorm_cat", device=lead.device)
    check_inputs(spec, dtype=lead.dtype)
    check_backward(spec, *parts, *weights, *(w for w in part_weights if w is not None))
    _check_enforced("rmsnorm_cat", spec, invariance)
    with torch.cuda.device(lead.device):
        try:
            return fn(parts, [w.reshape(-1).contiguous() for w in weights],
                      [w if w is None else w.reshape(-1).contiguous() for w in part_weights], float(eps))
        except ValueError as error:
            raise InvalidInput(f"tlx.ops.rmsnorm_cat: {error}") from error


def hstu_attn_dev(q, k, v, seq_offsets, max_seq_len, attn_scale, alpha=None, causal=True, num_targets=None,
                  max_attn_len=0, contextual_seq_len=0, *, space="full"):
    """HSTU ragged attention over `(total_tokens, H, HEAD_DIM)` fp16/bf16. Differentiable.

    Scores are SiLU-scaled rather than softmaxed, which is why this is its own
    op. `seq_offsets` is `(B + 1,)` prefix offsets; `alpha` defaults to
    `1 / HEAD_DIM`.

    Causal-only: `causal=False` raises `InvalidInput`. The argument is kept so
    the intent is stated at the call site rather than assumed.
    """
    space_name, invariance = _normalize_space_arg("hstu_attn_dev", space)
    fn, spec = impl_for("hstu_attn_dev", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1], causal=causal)
    check_backward(spec, q, k, v)
    _check_enforced("hstu_attn_dev", spec, invariance)
    return fn(q, k, v, seq_offsets, max_seq_len, alpha if alpha is not None else 1.0 / q.shape[-1], causal=causal,
              attn_scale=attn_scale, num_targets=num_targets, max_attn_len=max_attn_len,
              contextual_seq_len=contextual_seq_len, space=space_name)


_HSTU_CROSS_ATTN_IMPLS = {}


@assume_constant_result
def _hstu_cross_attn_impl_key(device):
    # Tensor devices have explicit indices. The catalog and device mapping stay fixed during graph reuse.
    if device not in _HSTU_CROSS_ATTN_IMPLS:
        _HSTU_CROSS_ATTN_IMPLS[device] = impl_for("hstu_cross_attn", device=device)
    return device


def hstu_cross_attn(q, k, v, seq_offsets, max_seq_len, attn_scale, alpha=None, seq_offsets_q=None, max_q_len=None,
                    num_softmax_heads=0, num_targets=None, causal=False, shared_kv=False, *, space="curated"):
    """HSTU cross attention over jagged `q` `(total_q, H, D)` and `k`/`v` `(total_kv, H_kv, D/V)`. Differentiable.

    `seq_offsets` are the `(B + 1,)` KV prefix offsets and `seq_offsets_q` the
    query ones; both default to self-attention style (queries are the KV
    sequence). The first `num_softmax_heads` heads use softmax, the rest SiLU
    scaled by the scalar `attn_scale`. `alpha` defaults to `HEAD_DIM ** -0.5`.
    `shared_kv=True` requires `v is k`.
    """
    if (seq_offsets_q is None) != (max_q_len is None):
        raise InvalidInput("tlx.ops.hstu_cross_attn takes seq_offsets_q and max_q_len together")
    if seq_offsets_q is None:
        seq_offsets_q, max_q_len = seq_offsets, max_seq_len
    space_name, invariance = _normalize_space_arg("hstu_cross_attn", space)
    # Read from a global cache so Dynamo can guard the callable and catalog specification.
    fn, spec = _HSTU_CROSS_ATTN_IMPLS[_hstu_cross_attn_impl_key(q.device)]
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1])
    check_backward(spec, q, k, v)
    _check_enforced("hstu_cross_attn", spec, invariance)
    import torch
    with torch.cuda.device(q.device):
        return fn(q, k, v, seq_offsets, seq_offsets_q, max_seq_len, max_q_len, attn_scale,
                  alpha if alpha is not None else q.shape[-1]**-0.5, num_softmax_heads=num_softmax_heads,
                  num_targets=num_targets, causal=causal, shared_kv=shared_kv, space=space_name)


def kimi_delta_attention(q, k, v, g, beta, *, scale=1.0, cu_seqlens=None, cu_seqlens_cpu=None, space="full"):
    """Kimi Delta Attention over packed `[1, T, H, 128]` fp16/bf16 inputs.

    Returns the TritonBench-compatible `(output, None)` pair.
    """
    space_name, invariance = _normalize_space_arg("kimi_delta_attention", space)
    fn, spec = impl_for("kimi_delta_attention", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1])
    check_backward(spec, q, k, v, g, beta)
    _check_enforced("kimi_delta_attention", spec, invariance)
    return fn(q, k, v, g, beta, scale=scale, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, space=space_name)


def kda_paged_prefill(q, k, v, g, beta, *, scale=1.0, initial_state=None, cu_seqlens, A_log=None, dt_bias=None,
                      lower_bound=None, beta_is_raw=False, state_pool=None, state_indices=None, space=None):
    """Chunked KDA prefill for packed `[1, T, H, 128]` tensors.

    By default `g` contains per-channel log decays, `q`/`k` are l2-normalized
    and `beta` is sigmoid-applied. Passing `A_log` and `dt_bias` takes raw
    inputs instead: `q`/`k` are normalized in the kernel and `g` is the gate
    pre-activation, mapped to `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`,
    or to `-exp(A_log) * softplus(g + dt_bias)` without a lower bound.
    `beta_is_raw` applies the sigmoid to `beta`.

    State is FP32 and V-major: a dense `initial_state` `[N, H, 128, 128]`, or a
    `state_pool` `[slots, H, 128, 128]` with `state_indices` `[N]` naming each
    sequence's slot, updated in place. This op is currently forward-only.
    ``space=`` takes no search space today; a ``Space`` is accepted only to
    carry a ``BitwiseInvarianceSpec``, which is validated but not enforced yet.
    """
    space_name, invariance = _normalize_space_arg("kda_paged_prefill", space)
    _check_single_config_space("kda_paged_prefill", space_name)
    fn, spec = impl_for("kda_paged_prefill", device=q.device)
    check_inputs(spec, dtype=q.dtype, KEY_DIM=q.shape[-1], VALUE_DIM=v.shape[-1])
    check_backward(spec, q, k, v, g, beta, initial_state, state_pool, A_log, dt_bias)
    _check_enforced("kda_paged_prefill", spec, invariance)
    return fn(q, k, v, g, beta, scale=scale, initial_state=initial_state, cu_seqlens=cu_seqlens, A_log=A_log,
              dt_bias=dt_bias, lower_bound=lower_bound, beta_is_raw=beta_is_raw, state_pool=state_pool,
              state_indices=state_indices)


def kda_recurrent_decode(q, k, v, g, beta, *, scale=1.0, state_pool, read_indices, write_indices, cu_seqlens,
                         A_log=None, dt_bias=None, lower_bound=None, beta_is_raw=False, space=None):
    """Forward-only indexed KDA recurrence over an FP32 V-major state pool.

    Inputs are prepared (l2-normalized q/k, log decay g, activated beta) unless
    `A_log` and `dt_bias` are passed, in which case they are raw, as for
    `kda_paged_prefill`. ``space=`` takes no search space today; a ``Space`` is
    accepted only to carry a ``BitwiseInvarianceSpec``, which is validated but
    not enforced yet.
    """
    space_name, invariance = _normalize_space_arg("kda_recurrent_decode", space)
    _check_single_config_space("kda_recurrent_decode", space_name)
    fn, spec = impl_for("kda_recurrent_decode", device=q.device)
    check_inputs(spec, dtype=q.dtype, KEY_DIM=q.shape[-1], VALUE_DIM=v.shape[-1])
    check_backward(spec, q, k, v, g, beta, state_pool, A_log, dt_bias)
    _check_enforced("kda_recurrent_decode", spec, invariance)
    return fn(q, k, v, g, beta, scale=scale, state_pool=state_pool, read_indices=read_indices,
              write_indices=write_indices, cu_seqlens=cu_seqlens, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
              beta_is_raw=beta_is_raw)


def _check_sliding_window(op, sliding_window):
    if isinstance(sliding_window, bool) or not isinstance(sliding_window, int) or sliding_window < 0:
        raise InvalidInput(f"tlx.ops.{op} expects a non-negative int sliding_window; got {sliding_window!r}")


def _check_kv_descales(name, kv_fp8, k_descale, v_descale, device):
    import torch
    if not kv_fp8:
        if k_descale is not None or v_descale is not None:
            raise InvalidInput(f"{name}: k_descale / v_descale apply only to a float8_e4m3fn K/V cache")
        return
    for arg, t in (("k_descale", k_descale), ("v_descale", v_descale)):
        if t is None or t.numel() != 1 or t.dtype != torch.float32 or t.device != device:
            raise InvalidInput(f"{name} expects a one-element float32 {arg} tensor on the query's device with an "
                               f"FP8 K/V cache")


def _check_paged_kv(query, key_cache, value_cache, context_lens, block_tables, query_length, kv_layout):
    import torch
    if query.ndim != 3 or key_cache.ndim != value_cache.ndim or key_cache.ndim not in (4, 5):
        raise InvalidInput("tlx.ops.paged_attention_decode expects (tokens, q_heads, head_dim) query and 4-D or "
                           f"packed 5-D K/V caches; got q={tuple(query.shape)}, k={tuple(key_cache.shape)}, "
                           f"v={tuple(value_cache.shape)}")
    if kv_layout not in ("HND", "NHD") or (kv_layout == "NHD" and key_cache.ndim != 4):
        raise InvalidInput(f"tlx.ops.paged_attention_decode expects kv_layout 'HND', or 'NHD' for a 4-D cache; got "
                           f"{kv_layout!r} with a {key_cache.ndim}-D cache")
    fp8 = torch.float8_e4m3fn
    kv_fp8 = key_cache.dtype == fp8
    if value_cache.dtype != key_cache.dtype or not (kv_fp8 or key_cache.dtype == query.dtype):
        raise InvalidInput("tlx.ops.paged_attention_decode expects K/V caches of the query's dtype, or "
                           "float8_e4m3fn K/V caches")
    if (query.dtype == fp8 and not kv_fp8) or (kv_fp8 and key_cache.ndim != 4):
        raise InvalidInput("tlx.ops.paged_attention_decode supports a float8_e4m3fn query only with float8_e4m3fn "
                           "K/V, and FP8 K/V only on a 4-D cache")
    if context_lens.dtype not in (torch.int32, torch.int64):
        raise InvalidInput("tlx.ops.paged_attention_decode expects int32 or int64 context_lens")
    if block_tables.dtype != torch.int32 or block_tables.ndim != 2:
        raise InvalidInput("tlx.ops.paged_attention_decode expects 2-D int32 block_tables")
    if any(t.device != query.device for t in (key_cache, value_cache, context_lens, block_tables)):
        raise InvalidInput("tlx.ops.paged_attention_decode expects all tensors on one device")
    num_kv_heads = key_cache.shape[2 if kv_layout == "NHD" else 1]
    if num_kv_heads == 0 or query.shape[1] % num_kv_heads:
        raise InvalidInput(f"tlx.ops.paged_attention_decode: {num_kv_heads} KV heads must divide "
                           f"{query.shape[1]} query heads")
    if query_length < 1 or query.shape[0] % query_length or query.shape[0] // query_length != context_lens.shape[0]:
        raise InvalidInput(f"tlx.ops.paged_attention_decode: {query.shape[0]} query tokens must be "
                           f"{context_lens.shape[0]} sequences x query_length={query_length}")
    return query.shape[1] // num_kv_heads


def _decode_compute_dtype(query, out):
    """Output (and MFMA) dtype: the query's, or for an FP8 query ``out``'s dtype, BF16 by default."""
    import torch
    if query.dtype != torch.float8_e4m3fn:
        return query.dtype
    return torch.bfloat16 if out is None else out.dtype


def paged_attention_decode(query, key_cache, value_cache, context_lens, block_tables, *, sm_scale=None, query_length=1,
                           sliding_window=0, max_context_len=None, out=None, workspace=None, kv_layout="HND",
                           k_descale=None, v_descale=None, q_descale=None, out_gate=None, out_norm_eps=1e-6,
                           space=None):
    """Forward-only GQA decode attention over a paged K/V cache.

    ``query`` is ``(num_seqs * query_length, q_heads, head_dim)``; sequence ``s``
    owns ``query_length`` consecutive rows, causal among themselves and ending at
    key ``context_lens[s] - 1``, so speculative verification passes its draft
    rows here. ``block_tables[s, p]`` is the physical block of the sequence's page
    ``p``. K/V caches are 4-D, either head-major ``kv_layout="HND"`` ``(blocks,
    kv_heads, page, head_dim)`` or page-major ``kv_layout="NHD"`` ``(blocks, page,
    kv_heads, head_dim)``, or the AITER-packed 5-D layout from
    ``allocate_paged_kv_cache_5d``. A positive ``sliding_window`` limits each
    query row to its last ``sliding_window`` keys, itself included; pages before
    the earliest window are never read. ``context_lens`` may be int32 or int64.
    ``max_context_len`` bounds every context and
    picks the split-K plan; pass it, and a buffer from
    ``paged_attention_decode_workspace``, to keep launches graph-capturable.

    FP8: 4-D ``float8_e4m3fn`` K/V caches take one-element float32 ``k_descale``
    / ``v_descale`` tensors (K/V = stored value * descale). The query may then be
    BF16/FP16, or ``float8_e4m3fn`` with ``q_descale``; an FP8 query writes a
    BF16 output unless ``out`` is FP16.

    Output norm and gate: with a BF16 ``out_gate`` of shape ``(tokens, q_heads * 64)`` (contiguous last dimension,
    any row stride, so a column slice of a projection output) and head_dim 64, the result is
    ``rmsnorm_gate(attention, out_gate, eps=out_norm_eps)`` of the BF16 attention output, bitwise, applied in the
    split-K reduce (the plan then has at least two splits; build the workspace with ``norm_gate=True``).
    """
    import torch
    group = _check_paged_kv(query, key_cache, value_cache, context_lens, block_tables, query_length, kv_layout)
    name = "tlx.ops.paged_attention_decode"
    _check_kv_descales(name, key_cache.dtype == torch.float8_e4m3fn, k_descale, v_descale, query.device)
    q_fp8 = query.dtype == torch.float8_e4m3fn
    if q_fp8 != (q_descale is not None) or (q_fp8 and (q_descale.numel() != 1 or q_descale.dtype != torch.float32
                                                       or q_descale.device != query.device)):
        raise InvalidInput(f"{name} expects a one-element float32 q_descale exactly when the query is float8_e4m3fn")
    out_dtype = _decode_compute_dtype(query, out)
    space_name, invariance = _normalize_space_arg("paged_attention_decode", space)
    _check_single_config_space("paged_attention_decode", space_name)
    if sm_scale is None:
        sm_scale = query.shape[2]**-0.5
    if not 0 < sm_scale < math.inf:
        raise InvalidInput(f"tlx.ops.paged_attention_decode expects a positive finite sm_scale; got {sm_scale!r}")
    _check_sliding_window("paged_attention_decode", sliding_window)
    if out is not None and (out.shape != query.shape or out.dtype != out_dtype or out.device != query.device
                            or out.dtype not in (torch.bfloat16, torch.float16)):
        raise InvalidInput("tlx.ops.paged_attention_decode expects out to match query's shape, dtype and device "
                           "(BF16/FP16 for an FP8 query)")
    if out_gate is not None:
        rows, q_heads, head_dim = query.shape
        if (not isinstance(out_gate, torch.Tensor) or head_dim != 64 or out_dtype != torch.bfloat16
                or out_gate.dtype != torch.bfloat16 or out_gate.shape != (rows, q_heads * 64) or out_gate.stride(1) != 1
                or out_gate.device != query.device):
            raise InvalidInput(f"{name} expects head_dim 64, a BF16 output and a BF16 out_gate of shape "
                               f"({rows}, {q_heads * 64}) with a contiguous last dimension")
        if not 0 < out_norm_eps < math.inf:
            raise InvalidInput(f"{name} expects a positive finite out_norm_eps; got {out_norm_eps!r}")
        if out is not None and torch._C._overlaps(out, out_gate):
            raise InvalidInput(f"{name} out must not alias out_gate")
    fn, spec = impl_for("paged_attention_decode", device=query.device)
    check_inputs(spec, dtype=out_dtype, HEAD_DIM=query.shape[2], QLEN=query_length, GROUP=group)
    check_backward(spec, query, key_cache, value_cache, out, out_gate, k_descale, v_descale, q_descale)
    _check_enforced("paged_attention_decode", spec, invariance)
    gate_kwargs = {} if out_gate is None else dict(out_gate=out_gate, out_norm_eps=float(out_norm_eps))
    with torch.cuda.device(query.device):
        return fn(query, key_cache, value_cache, context_lens, block_tables, float(sm_scale), int(query_length),
                  int(sliding_window), max_context_len, out, workspace, kv_layout, k_descale=k_descale,
                  v_descale=v_descale, q_descale=q_descale, **gate_kwargs)


def paged_attention_prefill(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal,
                            window_size, block_table, softcap, q_descale, k_descale, v_descale, q_scales=None,
                            alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None, shuffled_kv_cache=False,
                            skip_reduce=False, *, space=None):
    """Forward-only causal GQA prefill over a paged K/V cache; a drop-in for AITER's
    ``aiter.ops.triton.attention.unified_attention`` with the same arguments.

    ``q`` is packed ``(tokens, q_heads, head_dim)``; sequence ``b`` owns rows
    ``cu_seqlens_q[b]:cu_seqlens_q[b + 1]``, which are the last of its
    ``seqused_k[b]`` keys, so any earlier keys are a cached prefix. ``k`` and ``v``
    are ``(blocks, page, kv_heads, head_dim)`` and ``block_table[b, p]`` is the
    physical block of sequence ``b``'s page ``p``. ``window_size=(W - 1, 0)`` limits
    each row to its last ``W`` keys, itself included; ``(-1, -1)`` disables it.
    ``max_seqlen_q`` and ``max_seqlen_k`` bound every sequence's query rows and keys.
    The result is written into ``out``, which is returned.

    Supported subset: causal, BF16/FP16 Q with K/V of Q's dtype or FP8 (e4m3fn) K/V with one-element float32
    ``k_descale`` / ``v_descale`` (K/V = stored value * descale), no softcap, no Q scales or FP4, no ALiBi,
    ``qq_bias`` or sinks, the linear (unshuffled) cache, and ``skip_reduce=False``; anything else raises
    ``InvalidInput``.
    """
    import torch
    name = "tlx.ops.paged_attention_prefill"
    unsupported = {
        "causal=False": not causal, "softcap": bool(softcap), "q_descale": q_descale is not None, "q_scales": q_scales
        is not None, "alibi_slopes": alibi_slopes is not None, "output_scale": output_scale is not None, "qq_bias":
        qq_bias is not None, "sinks": sinks is not None, "shuffled_kv_cache": bool(shuffled_kv_cache), "skip_reduce":
        bool(skip_reduce)
    }
    if any(unsupported.values()):
        raise InvalidInput(f"{name} does not support {[key for key, bad in unsupported.items() if bad]}")
    if q.ndim != 3 or k.ndim != 4 or v.shape != k.shape:
        raise InvalidInput(f"{name} expects (tokens, q_heads, head_dim) q and 4-D (blocks, page, kv_heads, head_dim) "
                           f"k/v; got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    if v.dtype != k.dtype or not (kv_fp8 or k.dtype == q.dtype):
        raise InvalidInput(f"{name} expects k/v of q's dtype, or float8_e4m3fn k/v; got q {q.dtype}, k {k.dtype}, "
                           f"v {v.dtype}")
    _check_kv_descales(name, kv_fp8, k_descale, v_descale, q.device)

    # FP8 tiles load through int32 buffer offsets, so bound the largest offset the strides reach, not numel().
    def fp8_loadable(t):
        last = sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
        return t.data_ptr() % 16 == 0 and all(s % 16 == 0 for s in t.stride()[:3]) and last < 2**31

    if kv_fp8 and not (fp8_loadable(k) and fp8_loadable(v)):
        raise InvalidInput(f"{name} expects FP8 k/v with 16-byte aligned base pointers and strides, each spanning "
                           "under 2**31 elements")
    if k.shape[3] != q.shape[2] or q.stride(2) != 1 or k.stride(3) != 1 or v.stride(3) != 1:
        raise InvalidInput(f"{name} expects one head_dim, contiguous in q and k/v")
    if (cu_seqlens_q.dtype != torch.int32 or block_table.dtype != torch.int32 or block_table.ndim != 2
            or seqused_k.dtype not in (torch.int32, torch.int64)):
        raise InvalidInput(f"{name} expects int32 cu_seqlens_q, 2-D int32 block_table and int32/int64 seqused_k")
    if cu_seqlens_q.shape != (seqused_k.shape[0] + 1, ) or block_table.shape[0] != seqused_k.shape[0]:
        raise InvalidInput(f"{name} expects cu_seqlens_q of batch + 1 entries and one seqused_k entry and "
                           "block_table row per sequence")
    if any(t.device != q.device for t in (k, v, out, cu_seqlens_q, seqused_k, block_table)):
        raise InvalidInput(f"{name} expects all tensors on one device")
    if out.shape != q.shape or out.dtype != q.dtype or out.stride(2) != 1:
        raise InvalidInput(f"{name} expects out to match q's shape and dtype")
    num_kv_heads = k.shape[2]
    if num_kv_heads == 0 or q.shape[1] % num_kv_heads:
        raise InvalidInput(f"{name}: {num_kv_heads} KV heads must divide {q.shape[1]} query heads")
    for arg, value in (("max_seqlen_q", max_seqlen_q), ("max_seqlen_k", max_seqlen_k)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**31:
            raise InvalidInput(f"{name} expects an int {arg} in [0, 2**31); got {value!r}")
    if not 0 < softmax_scale < math.inf:
        raise InvalidInput(f"{name} expects a positive finite softmax_scale; got {softmax_scale!r}")
    if len(window_size) != 2 or not (tuple(window_size) == (-1, -1) or (window_size[0] >= 0 and window_size[1] == 0)):
        raise InvalidInput(f"{name} expects window_size (W - 1, 0) or (-1, -1); got {window_size!r}")
    space_name, invariance = _normalize_space_arg("paged_attention_prefill", space)
    _check_single_config_space("paged_attention_prefill", space_name)
    fn, spec = impl_for("paged_attention_prefill", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[2], GROUP=q.shape[1] // num_kv_heads)
    check_backward(spec, q, k, v, out, k_descale, v_descale)
    _check_enforced("paged_attention_prefill", spec, invariance)
    with torch.cuda.device(q.device):
        fn(q, k, v, cu_seqlens_q, seqused_k.to(torch.int32), block_table, max_seqlen_q, sm_scale=float(softmax_scale),
           sliding_window=int(window_size[0]) + 1, out=out, max_context_len=max_seqlen_k, k_descale=k_descale,
           v_descale=v_descale)
    return out


def paged_attention_decode_workspace(query, key_cache, value_cache, block_tables, *, query_length=1, sliding_window=0,
                                     max_context_len=None, kv_layout="HND", norm_gate=False):
    """Reusable FP32 split-K workspace for ``paged_attention_decode`` with the same
    arguments (``norm_gate=True`` for calls with ``out_gate``), or None when the plan needs none."""
    _check_sliding_window("paged_attention_decode_workspace", sliding_window)
    fn, spec = impl_for("paged_attention_decode_workspace", device=query.device)
    check_inputs(spec, dtype=_decode_compute_dtype(query, None))
    import torch
    with torch.cuda.device(query.device):
        return fn(query, key_cache, value_cache, block_tables, int(query_length), int(sliding_window), max_context_len,
                  kv_layout, bool(norm_gate))


def allocate_paged_kv_cache_5d(num_blocks, num_kv_heads, page_size, head_dim, *, dtype, device):
    """Uninitialized packed K/V caches: K ``(blocks, kv_heads, head_dim / X, page, X)`` and V
    ``(blocks, kv_heads, page / X, head_dim, X)``, with X elements per 16 bytes."""
    import torch
    fn, spec = impl_for("allocate_paged_kv_cache_5d", device=torch.device(device))
    check_inputs(spec, dtype=dtype)
    return fn(num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)


def reshape_and_cache_5d(key, value, key_cache, value_cache, slot_mapping):
    """Scatter ``(tokens, kv_heads, head_dim)`` K/V into packed 5-D caches at
    ``slot_mapping = block * page_size + offset``; negative slots are skipped."""
    fn, spec = impl_for("reshape_and_cache_5d", device=key.device)
    check_inputs(spec, dtype=key.dtype)
    return fn(key, value, key_cache, value_cache, slot_mapping)


def _check_fp8_paged_kv(query, key_cache, value_cache, block_tables, *, max_context_len, query_length, sliding_window,
                        num_splits):
    import torch
    if query.ndim != 3 or query.shape[0] == 0 or query.shape[1] == 0 or query.shape[2] not in (64, 128):
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8 expects nonempty Q [tokens, heads, D64/D128]")
    if query.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8 expects BF16 or E4M3FN Q")
    if key_cache.ndim != 5 or value_cache.ndim != 5:
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8 expects packed 5-D K/V caches")
    blocks, kv_heads, groups, page, x = key_cache.shape
    if (blocks == 0 or kv_heads == 0 or query.shape[1] % kv_heads or x != 16 or groups * x != query.shape[2]
            or page < 16 or page % 16):
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8: invalid packed K layout or head grouping")
    if value_cache.shape != (blocks, kv_heads, page // x, query.shape[2], x):
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8: invalid packed V layout")
    if key_cache.dtype != torch.float8_e4m3fn or value_cache.dtype != torch.float8_e4m3fn:
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8 requires plain E4M3FN K/V")
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8 requires contiguous packed K/V")
    if isinstance(query_length,
                  bool) or not isinstance(query_length, int) or query_length < 1 or query.shape[0] % query_length:
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8: query_length must divide the query token count")
    if isinstance(max_context_len, bool) or not isinstance(max_context_len, int) or max_context_len < 1:
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8 expects a positive integer max_context_len")
    batch = query.shape[0] // query_length
    if (block_tables.ndim != 2 or block_tables.shape[0] != batch or block_tables.stride(1) != 1
            or block_tables.dtype not in (torch.int32, torch.int64)
            or block_tables.shape[1] < (max_context_len + page - 1) // page):
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8: block table capacity/layout must cover max_context_len")
    if any(t.device != query.device for t in (key_cache, value_cache, block_tables)):
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8 expects all inputs on the query device")
    _check_sliding_window("paged_attention_decode_fp8", sliding_window)
    if num_splits is not None and (isinstance(num_splits, bool) or not isinstance(num_splits, int)
                                   or not 1 <= num_splits <= 1024):
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8 expects num_splits in [1, 1024]")
    return batch, kv_heads


def paged_attention_decode_fp8(query, key_cache, value_cache, context_lens, block_tables, k_scale, v_scale, *,
                               max_context_len, q_scale=None, query_length=1, causal=True, sliding_window=0,
                               sm_scale=None, num_splits=None, out=None, workspace=None, space=None):
    """Forward-only attention over packed E4M3FN KV with per-head multipliers.

    Q is ``[batch * query_length, query_heads, D]``, BF16 or E4M3FN; D is 64/128.
    K is ``[blocks, kv_heads, D/16, page_size, 16]`` and V is
    ``[blocks, kv_heads, page_size/16, D, 16]``, contiguous E4M3FN.
    ``page_size`` counts cached tokens per physical KV block and is a positive
    multiple of 16; the inner 16 is a separate packing factor. ``block_tables``
    maps each sequence's logical page to a physical block id: token t uses
    entry ``t // page_size`` and within-page offset ``t % page_size``.
    Separate FP32 ``k_scale``/``v_scale`` vectors have one multiplier per KV
    head; FP8 Q requires an FP32 ``q_scale`` per query head. Scale values must be positive
    and finite and inputs finite. This is not E8M0 block-scaled MXFP8.

    ``context_lens`` includes the queries' K/V. Causal queries align with the
    context's final positions; noncausal query sets share the whole cache.
    A positive ``sliding_window`` includes the current key and requires causal
    attention. Empty/fully masked rows produce zero. Context lengths must be
    in ``[0, max_context_len]`` and referenced page ids valid; these device-side
    value preconditions are not copied to the host for inspection.

    Returns contiguous BF16 with Q's shape, before output projection. ``out``
    can supply that allocation. Reuse ``paged_attention_decode_fp8_workspace``
    with the same planning arguments for graph capture. Allocate both before
    capture; workspace planning uses the same query/cache shapes, context
    bound, query length, sliding window and optional split override as attention.
    Automatic batched single-token decode may choose active splits per request
    from device context lengths within a fixed graph/workspace capacity. Short
    requests can bypass split scratch and merge while long requests retain
    partition parallelism. Explicit split counts override this policy. Only the
    ``heuristic`` search space is supported; implementation choices remain private.
    """
    import torch
    batch, kv_heads = _check_fp8_paged_kv(query, key_cache, value_cache, block_tables, max_context_len=max_context_len,
                                          query_length=query_length, sliding_window=sliding_window,
                                          num_splits=num_splits)
    if (context_lens.shape != (batch, ) or context_lens.dtype not in (torch.int32, torch.int64)
            or not context_lens.is_contiguous() or context_lens.device != query.device):
        raise InvalidInput(
            "tlx.ops.paged_attention_decode_fp8 expects integer context_lens [batch] on the query device")
    if not isinstance(causal, bool) or (sliding_window and not causal):
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8: sliding windows require causal=True")
    if query.dtype == torch.float8_e4m3fn and q_scale is None:
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8: FP8 Q requires per-query-head q_scale")
    if query.dtype == torch.bfloat16 and q_scale is not None:
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8: q_scale applies only to FP8 Q")
    for name, scale, heads in (("k_scale", k_scale, kv_heads), ("v_scale", v_scale, kv_heads), ("q_scale", q_scale,
                                                                                                query.shape[1])):
        if name == "q_scale" and scale is None:
            continue
        if (not isinstance(scale, torch.Tensor) or scale.shape != (heads, ) or scale.dtype != torch.float32
                or scale.device != query.device or not scale.is_contiguous()):
            raise InvalidInput(
                f"tlx.ops.paged_attention_decode_fp8: {name} must be contiguous FP32 [{heads}] on the query device")
    sm_scale = query.shape[2]**-.5 if sm_scale is None else sm_scale
    if not isinstance(sm_scale, (int, float)) or isinstance(sm_scale, bool) or not 0 < sm_scale < math.inf:
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8 expects positive finite sm_scale")
    if out is not None and (out.shape != query.shape or out.dtype != torch.bfloat16 or out.device != query.device
                            or not out.is_contiguous()):
        raise InvalidInput("tlx.ops.paged_attention_decode_fp8 expects contiguous BF16 out with Q's shape/device")
    if workspace is not None:
        if not isinstance(workspace, (tuple, list)) or len(workspace) != 2:
            raise InvalidInput("tlx.ops.paged_attention_decode_fp8 workspace must be a partial/LSE pair")
        partial, lse = workspace
        if (not isinstance(partial, torch.Tensor) or not isinstance(lse, torch.Tensor) or partial.ndim != 4
                or lse.ndim != 3 or partial.shape[:2] != query.shape[:2] or partial.shape[-1] != query.shape[-1]
                or partial.shape[:-1] != lse.shape
                or any(t.dtype != torch.float32 or t.device != query.device or not t.is_contiguous()
                       for t in workspace)):
            raise InvalidInput("tlx.ops.paged_attention_decode_fp8 expects contiguous FP32 partial/LSE workspace for Q")
    space_name, invariance = _normalize_space_arg("paged_attention_decode_fp8", space)
    _check_single_config_space("paged_attention_decode_fp8", space_name)
    fn, spec = impl_for("paged_attention_decode_fp8", device=query.device)
    check_inputs(spec, dtype=query.dtype, HEAD_DIM=query.shape[2])
    check_backward(spec, query, key_cache, value_cache, k_scale, v_scale, q_scale)
    _check_enforced("paged_attention_decode_fp8", spec, invariance)
    with torch.cuda.device(query.device):
        try:
            return fn(query, key_cache, value_cache, context_lens, block_tables, k_scale, v_scale,
                      max_context_len=max_context_len, q_scale=q_scale, query_length=query_length, causal=causal,
                      sliding_window=sliding_window, sm_scale=float(sm_scale), num_splits=num_splits, out=out,
                      workspace=workspace)
        except ValueError as error:
            raise InvalidInput(f"tlx.ops.paged_attention_decode_fp8: {error}") from error


def paged_attention_decode_fp8_workspace(query, key_cache, value_cache, block_tables, *, max_context_len,
                                         query_length=1, sliding_window=0, num_splits=None, space=None):
    """Allocate reusable opaque FP32 split buffers, or None for a single split.

    Planning arguments must match ``paged_attention_decode_fp8``. Returned
    capacity shapes are ``[tokens, query_heads, splits, D]`` for partial values
    and ``[tokens, query_heads, splits]`` for log-sum-exp. Both partial values
    and log-sum-exp retain FP32 precision; treat their contents as opaque.
    No device metadata values are read back and no quantization or attention
    is run. Allocate this workspace and output before graph capture.
    """
    import torch
    _check_fp8_paged_kv(query, key_cache, value_cache, block_tables, max_context_len=max_context_len,
                        query_length=query_length, sliding_window=sliding_window, num_splits=num_splits)
    name, invariance = _normalize_space_arg("paged_attention_decode_fp8_workspace", space)
    _check_single_config_space("paged_attention_decode_fp8_workspace", name)
    fn, spec = impl_for("paged_attention_decode_fp8_workspace", device=query.device)
    check_inputs(spec, dtype=query.dtype)
    _check_enforced("paged_attention_decode_fp8_workspace", spec, invariance)
    with torch.cuda.device(query.device):
        return fn(query, key_cache, value_cache, block_tables, max_context_len=max_context_len,
                  query_length=query_length, sliding_window=sliding_window, num_splits=num_splits)
