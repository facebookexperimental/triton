"""Opt-in PyTorch dispatcher integration for TLX FlashAttention kernels.

Importing this module registers the ``TLX_GFX950_BWD`` provider with
``torch.nn.attention``. It does not activate the provider. Activation replaces
only dense FlashAttention backward; PyTorch continues to own forward and every
unsupported backward call is sent to the CUDA kernel captured at activation.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch

from triton.tlx.ops.kernels.flash_attn import gfx950_bwd

PROVIDER_NAME = "TLX_GFX950_BWD"
_D64_ROUTE = "d64"
_D128_SHORT_ROUTE = "d128_short"
_D128_INTERLEAVED_ROUTE = "d128_interleaved"
_D256_ROUTE = "d256"
_D128_SPLIT_ENTRY = "split"
_D128_COMBINED_ENTRY = "combined"
_D128_INTERLEAVED_ENTRY = "interleaved"
_D128_NATIVE_CONVERT_ENTRY = "native_convert"
_D256_PRODUCER_ENTRY = "producer"
# (entry, block_m, block_n, num_warps, pipelined, rectangular, exact)
_D128_NONCAUSAL_SPLIT_CONFIG = (_D128_SPLIT_ENTRY, 32, 64, 4, True, True, False)
_D128_CAUSAL_SPLIT_CONFIG = (_D128_SPLIT_ENTRY, 32, 32, 2, False, False, False)
_D128_EXACT_CONFIG = (_D128_COMBINED_ENTRY, 16, 256, 4, False, False, True)
_D128_INTERLEAVED_CONFIG = (
    _D128_INTERLEAVED_ENTRY,
    16,
    256,
    4,
    1,
    16,
    _D128_NATIVE_CONVERT_ENTRY,
    128,
    4,
    False,
    False,
    False,
)
_D256_NONCAUSAL_CONFIG = (_D256_PRODUCER_ENTRY, 4, False, True)
_D256_CAUSAL_CONFIG = (_D256_PRODUCER_ENTRY, 4, False, False)
_REQUIRED_BASE_ALIGNMENT_BYTES = 16


def _d64_noncausal_config(kv_splits):
    return (
        "noncausal_fused_n256",
        32,
        256,
        kv_splits,
        False,
        0,
        64,
        False,
        (),
        None,
        False,
        None,
        None,
    )


def _d64_causal_config(
    family,
    owner_rows,
    key_rows,
    stat_mode,
    dq_use_xcd,
    launch_tiles,
    owner_fragments,
    *,
    gqa_grid_mode=None,
    dkdv_lifetime=None,
    q3_register_class=None,
):
    return (
        family,
        owner_rows,
        key_rows,
        4 if family == "causal_scheduled_gqa8" else 1,
        True,
        stat_mode,
        32,
        dq_use_xcd,
        ((launch_tiles, True, 0, 0, owner_fragments, 0), ),
        gqa_grid_mode,
        False,
        dkdv_lifetime,
        q3_register_class,
    )


_D64_NONCAUSAL_GQA8_CONFIG = _d64_noncausal_config(8)
_D64_NONCAUSAL_MHA_CONFIG = _d64_noncausal_config(1)
_D64_CAUSAL_MHA_M192_CONFIG = _d64_causal_config(
    "causal_scheduled_mha",
    192,
    64,
    0,
    False,
    22,
    3,
)
_D64_CAUSAL_GQA8_M192_CONFIG = _d64_causal_config(
    "causal_scheduled_gqa8",
    192,
    128,
    1,
    True,
    6,
    3,
    gqa_grid_mode="xcd",
    dkdv_lifetime="direct_d64",
    q3_register_class="agpr",
)
# Exact (Q shape, K/V shape, causal) signatures measured through the activated
# provider on MI350X. Each entry records the dispatcher-selected tuning
# configuration so runtime experiments cannot silently select an unmeasured
# route. Static launch constants remain part of the kernel implementation.
_PERFORMANCE_VALIDATED_SIGNATURES = {
    ((1, 16, 4096, 64), (1, 2, 4096, 64), False): (_D64_ROUTE, _D64_NONCAUSAL_GQA8_CONFIG),
    ((1, 16, 4096, 64), (1, 16, 4096, 64), False): (_D64_ROUTE, _D64_NONCAUSAL_MHA_CONFIG),
    ((1, 24, 4096, 64), (1, 24, 4096, 64), True): (_D64_ROUTE, _D64_CAUSAL_MHA_M192_CONFIG),
    ((4, 48, 1024, 64), (4, 6, 1024, 64), True): (_D64_ROUTE, _D64_CAUSAL_GQA8_M192_CONFIG),
    ((16, 27, 200, 128), (16, 27, 200, 128), False):
    (_D128_SHORT_ROUTE, frozenset({_D128_NONCAUSAL_SPLIT_CONFIG, _D128_EXACT_CONFIG})),
    ((16, 27, 200, 128), (16, 27, 200, 128), True):
    (_D128_SHORT_ROUTE, frozenset({_D128_CAUSAL_SPLIT_CONFIG, _D128_EXACT_CONFIG})),
    ((16, 64, 1024, 128), (16, 8, 1024, 128), False): (_D128_INTERLEAVED_ROUTE, _D128_INTERLEAVED_CONFIG),
    ((16, 64, 2048, 128), (16, 8, 2048, 128), False): (_D128_INTERLEAVED_ROUTE, _D128_INTERLEAVED_CONFIG),
    ((16, 64, 4096, 128), (16, 8, 4096, 128), False): (_D128_INTERLEAVED_ROUTE, _D128_INTERLEAVED_CONFIG),
    ((32, 1, 2600, 256), (32, 1, 2600, 256), False): (_D256_ROUTE, _D256_NONCAUSAL_CONFIG),
    ((32, 1, 2600, 256), (32, 1, 2600, 256), True): (_D256_ROUTE, _D256_CAUSAL_CONFIG),
}


@dataclass
class _TLXFlashAttentionHandle:
    library: torch.library.Library | None

    def remove(self) -> None:
        # torch.library.Library deregisters its implementations on destruction.
        self.library = None


def _d128_dispatch_config(dispatch):
    if dispatch.entry is gfx950_bwd._attn_bwd_dkdv_d128_split_kernel:
        entry = _D128_SPLIT_ENTRY
    elif dispatch.entry is gfx950_bwd._attn_bwd_dkdv_dq_d128_combined_kernel:
        entry = _D128_COMBINED_ENTRY
    else:
        return None
    return (
        entry,
        dispatch.block_m,
        dispatch.block_n,
        dispatch.num_warps,
        dispatch.pipelined,
        dispatch.rectangular,
        dispatch.exact,
    )


def _d128_interleaved_dispatch_config(dispatch):
    if dispatch.main_entry is not gfx950_bwd._attn_bwd_dkdv_dq_d128_gqa_kernel:
        return None
    if dispatch.convert_entry is not gfx950_bwd._attn_bwd_dq_native_convert_kernel:
        return None
    return (
        _D128_INTERLEAVED_ENTRY,
        dispatch.block_m,
        dispatch.block_n,
        dispatch.num_warps,
        dispatch.num_stages,
        dispatch.matrix_instr_nonkdim,
        _D128_NATIVE_CONVERT_ENTRY,
        dispatch.convert_block_m,
        dispatch.convert_num_warps,
        dispatch.sink_insts_to_avoid_spills,
        dispatch.regclass_priority_trumps_globalness,
        dispatch.reverse_local_assignment,
    )


def _d256_dispatch_config(dispatch):
    if dispatch.entry is not gfx950_bwd._attn_bwd_dkdv_d256_producer_kernel:
        return None
    return (_D256_PRODUCER_ENTRY, dispatch.num_warps, dispatch.staged, dispatch.pipelined)


def _d64_dispatch_config(dispatch):
    launches = tuple((launch.launch_tiles, launch.skip_owner_tail, launch.owner_pid_base, launch.launch_q_tiles,
                      launch.owner_fragments, launch.grid_owner_m) for launch in dispatch.dq_launches)
    return (
        dispatch.family,
        dispatch.owner_rows,
        dispatch.key_rows,
        dispatch.kv_splits,
        dispatch.selected_causal,
        dispatch.stat_mode,
        dispatch.dq_logical_n,
        dispatch.dq_use_xcd,
        launches,
        dispatch.gqa_grid_mode,
        dispatch.cyclic_query_split,
        dispatch.dkdv_lifetime,
        gfx950_bwd._d64_q3_register_class(dispatch),
    )


def _native_fallback(
    native_kernel,
    dispatch_keys,
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    *,
    scale,
):
    return native_kernel.call_boxed(
        dispatch_keys,
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )


def _is_performance_validated(query, key, value, out, grad_out, logsumexp, scale, is_causal):
    signature = (tuple(query.shape), tuple(key.shape), bool(is_causal))
    route = _PERFORMANCE_VALIDATED_SIGNATURES.get(signature)
    if route is None:
        return False
    try:
        if any(tensor.data_ptr() % _REQUIRED_BASE_ALIGNMENT_BYTES
               for tensor in (query, key, value, out, grad_out, logsumexp)):
            return False
    except (AttributeError, RuntimeError, TypeError):
        return False

    route_kind, expected_config = route
    if route_kind == _D128_SHORT_ROUTE:
        if any(gfx950_bwd._d128_regalloc_options().values()):
            return False
        try:
            dispatch = gfx950_bwd._select_d128_dispatch(tuple(query.shape), is_causal)
            selected_config = _d128_dispatch_config(dispatch)
        except (AssertionError, AttributeError, RuntimeError, TypeError, ValueError):
            return False
        return selected_config in expected_config
    if route_kind == _D128_INTERLEAVED_ROUTE:
        try:
            dispatch = gfx950_bwd._select_d128_interleaved_dispatch()
            selected_config = _d128_interleaved_dispatch_config(dispatch)
        except (AssertionError, AttributeError, RuntimeError, TypeError, ValueError):
            return False
        return selected_config == expected_config
    if route_kind == _D256_ROUTE:
        try:
            dispatch = gfx950_bwd._select_d256_dispatch(is_causal)
            selected_config = _d256_dispatch_config(dispatch)
        except (AssertionError, AttributeError, RuntimeError, TypeError, ValueError):
            return False
        return selected_config == expected_config

    if route_kind != _D64_ROUTE:
        return False
    try:
        dispatch = gfx950_bwd._select_d64_dispatch_for_device(
            query,
            key,
            value,
            out,
            grad_out,
            logsumexp,
            scale,
            is_causal,
        )
        selected_config = _d64_dispatch_config(dispatch)
    except (AssertionError, AttributeError, RuntimeError, TypeError, ValueError):
        return False
    return selected_config == expected_config


def _tlx_support_error(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    *,
    scale,
):
    if dropout_p != 0.0:
        return "dropout_p must be zero"
    if cum_seq_q is not None or cum_seq_k is not None:
        return "only dense attention is supported"
    if query.layout is not torch.strided:
        return "query must use strided layout"
    if torch.are_deterministic_algorithms_enabled():
        return "deterministic algorithms are enabled"
    if query.ndim != 4 or key.ndim != 4:
        return "query and key must be rank-4 BHSD tensors"
    if query.shape[-1] <= 0:
        return "query head dimension must be positive"
    if max_q != query.shape[2] or max_k != key.shape[2]:
        return "max_q and max_k must match the dense sequence lengths"

    resolved_scale = query.shape[-1]**-0.5 if scale is None else scale
    error = gfx950_bwd.fa_backward_support_error(
        query,
        key,
        value,
        out,
        grad_out,
        logsumexp,
        resolved_scale,
        is_causal,
    )
    if error is not None:
        return error
    if not _is_performance_validated(query, key, value, out, grad_out, logsumexp, resolved_scale, is_causal):
        return "shape is supported but not performance-validated for dispatcher use"
    return None


def _tlx_scaled_dot_product_flash_attention_backward(
    native_kernel,
    dispatch_keys,
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    *,
    scale=None,
):
    error = _tlx_support_error(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        scale=scale,
    )
    if error is not None:
        return _native_fallback(
            native_kernel,
            dispatch_keys,
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            philox_seed,
            philox_offset,
            scale=scale,
        )

    resolved_scale = query.shape[-1]**-0.5 if scale is None else scale
    return gfx950_bwd.fa_backward(
        query,
        key,
        value,
        out,
        grad_out,
        logsumexp,
        resolved_scale,
        is_causal,
    )


def register_tlx_gfx950_flash_attention_backward() -> _TLXFlashAttentionHandle:
    """Install the conditional gfx950 dense-backward dispatcher override."""
    get_kernel = getattr(torch.library, "get_kernel", None)
    if get_kernel is None:
        raise RuntimeError("TLX FlashAttention integration requires torch.library.get_kernel")

    op = torch.ops.aten._scaled_dot_product_flash_attention_backward.default
    native_kernel: Any = get_kernel(op, "CUDA")
    library = torch.library.Library("aten", "IMPL", "CUDA")
    library.impl(
        "_scaled_dot_product_flash_attention_backward",
        functools.partial(_tlx_scaled_dot_product_flash_attention_backward, native_kernel),
        "CUDA",
        with_keyset=True,
    )
    return _TLXFlashAttentionHandle(library)


try:
    from torch.nn.attention import register_flash_attention_impl
except ImportError as error:
    raise ImportError(
        "TLX FlashAttention integration requires torch.nn.attention.register_flash_attention_impl") from error

register_flash_attention_impl(
    PROVIDER_NAME,
    register_fn=register_tlx_gfx950_flash_attention_backward,
)

__all__ = ["PROVIDER_NAME", "register_tlx_gfx950_flash_attention_backward"]
