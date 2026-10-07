"""Opt-in PyTorch dispatcher integration for TLX FlashAttention kernels.

Importing this module registers the ``TLX_GFX950_BWD`` provider with
``torch.nn.attention``. It does not activate the provider. Activation replaces
only selected FlashAttention backward calls; PyTorch continues to own forward and every
unsupported backward call is sent to the CUDA kernel captured at activation.
"""

from __future__ import annotations

import functools
import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any

import torch

from triton.tlx.ops.kernels.flash_attn import gfx950_bwd, gfx950_varlen_bwd

PROVIDER_NAME = "TLX_GFX950_BWD"
_D64_ROUTE = "d64"
_D128_SHORT_ROUTE = "d128_short"
_D128_INTERLEAVED_ROUTE = "d128_interleaved"
_D256_ROUTE = "d256"
_D128_SPLIT_ENTRY = "split"
_D128_COMBINED_ENTRY = "combined"
_D128_INTERLEAVED_ENTRY = "interleaved"
_D128_BM32_ENTRY = "dense_bm32"
_D128_NATIVE_CONVERT_ENTRY = "native_convert"
_D256_PRODUCER_ENTRY = "producer"
# (entry, block_m, block_n, num_warps, pipelined, rectangular, exact)
_D128_NONCAUSAL_SPLIT_CONFIG = (_D128_SPLIT_ENTRY, 32, 64, 4, True, True, False)
_D128_CAUSAL_SPLIT_CONFIG = (_D128_SPLIT_ENTRY, 32, 32, 2, False, False, False)
_D128_EXACT_CONFIG = (_D128_COMBINED_ENTRY, 16, 256, 4, False, False, True)
_D128_INTERLEAVED_CONFIG = (
    _D128_INTERLEAVED_ENTRY, 16, 256, 4, 1, 16, _D128_NATIVE_CONVERT_ENTRY, 128, 4, False, False, False,
    1,  # query-head owner splits
    -1,  # no phase scheduling hint
    False,  # do not peel the causal diagonal
)
_D128_IGLP3_CONFIG = (*_D128_INTERLEAVED_CONFIG[:-2], 3, False)
_D128_CAUSAL_PEEL_CONFIG = (*_D128_INTERLEAVED_CONFIG[:-2], 3, True)
_D128_HEAD_SPLIT_CONFIG = (*_D128_INTERLEAVED_CONFIG[:-3], 4, -1, False)
_D128_BM32_CONFIG = (_D128_BM32_ENTRY, 32, *_D128_INTERLEAVED_CONFIG[2:])
_D256_NONCAUSAL_CONFIG = (_D256_PRODUCER_ENTRY, 4, False, True)
_D256_CAUSAL_CONFIG = (_D256_PRODUCER_ENTRY, 4, False, False)
_REQUIRED_BASE_ALIGNMENT_BYTES = 16

# (Q shape, K/V shape, batch, max_q, max_k, causal). Admission measurements
# include offset validation, plan construction, and O/dO normalization. The
# explicit prepared-plan API also exposes shapes outside this dispatch table.
# Values fingerprint the measured Q/KV prefix sums (little-endian int32).
# Equal totals/maxima do not imply equal work for a ragged batch. These are
# the five balanced MHA, five varied GQA, and two seed-42 Prefix cases from
# #3775, measured here through ATen with per-call plan preparation included.
_PERFORMANCE_VALIDATED_VARLEN_SIGNATURES = {
    ((62780, 4, 128), (946279, 4, 128), 768, 300, 3200, False):
    "7422d07f3a298fbc3a0c1fcc029608dbba2b37569bc015c63dccb011a3603ce6",
    ((80900, 4, 128), (940737, 4, 128), 768, 300, 3200, False):
    "2537eb7b783f4e5c3459137364aac8dc169fbc4f58cf17ef57c9a81becf80487",
    ((88660, 4, 128), (944840, 4, 128), 768, 300, 3200, False):
    "7ce7155052b350f7359ce26605818a4864b5f72e7b176bc5f915d9132f0edad2",
    ((101900, 4, 128), (946104, 4, 128), 768, 300, 3200, False):
    "41aa5a701bdfbec0569480fe74a0f3b6e794246c2ccd7379d20b5d57568caf33",
    ((84880, 4, 128), (945242, 4, 128), 768, 400, 3200, False):
    "423d7406b2e8bf31cb5027073d53aa2249746d65982fce1edf6a1f5769b2982b",
    ((62780, 12, 128), (946279, 4, 128), 768, 300, 3200, False):
    "903d4e2d98a02c3dc49f6ce017bd23a2df0fb394a5a5637a401fde9a518bf8ca",
    ((80900, 12, 128), (940737, 4, 128), 768, 300, 3200, False):
    "18a8a9fa206376ff21de9e9b9da1115db761b2fde8f4c8b81bd7c67195dfeb44",
    ((88660, 12, 128), (944840, 4, 128), 768, 300, 3200, False):
    "2684285e955c4a506b055acf2c1184e42172dcb4ff407f681aaddd91083746a7",
    ((101900, 12, 128), (946104, 4, 128), 768, 300, 3200, False):
    "da8403c99e37e1a7046c9f1a62cdfee7aca63dc08bcb5f6e8438df69db2b2217",
    ((84880, 12, 128), (945242, 4, 128), 768, 400, 3200, False):
    "5d42d6075aa9f63112519866b2e996ae1cda203dafeb8da89698cd0a2b27e1f6",
    ((50754, 12, 128), (100696, 4, 128), 19, 5662, 10414, False):
    "10678ac419b52fc0e7c8441ed106adc4115cf90a40c84810ce2938cdbd322c2e",
    ((50754, 64, 128), (100696, 8, 128), 19, 5662, 10414, False):
    "10678ac419b52fc0e7c8441ed106adc4115cf90a40c84810ce2938cdbd322c2e",
}


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
_D64_CAUSAL_MHA_M256_XCD_64_CONFIG = _d64_causal_config(
    "causal_scheduled_mha",
    256,
    64,
    0,
    True,
    64,
    4,
)
_D64_CAUSAL_MHA_M256_XCD_65_CONFIG = _d64_causal_config(
    "causal_scheduled_mha",
    256,
    64,
    0,
    True,
    65,
    4,
)
_D64_CAUSAL_MHA_M256_LINEAR_64_CONFIG = _d64_causal_config(
    "causal_scheduled_mha",
    256,
    64,
    0,
    False,
    64,
    4,
)
_D64_CAUSAL_MHA_M256_LINEAR_128_CONFIG = _d64_causal_config(
    "causal_scheduled_mha",
    256,
    64,
    0,
    False,
    128,
    4,
)
_D64_CAUSAL_GQA8_M256_16_CONFIG = _d64_causal_config(
    "causal_scheduled_gqa8",
    256,
    128,
    1,
    True,
    16,
    4,
    gqa_grid_mode="xcd",
    dkdv_lifetime="direct_d64",
    q3_register_class="agpr",
)
_D64_CAUSAL_GQA8_M256_64_CONFIG = _d64_causal_config(
    "causal_scheduled_gqa8",
    256,
    128,
    1,
    True,
    64,
    4,
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
    ((2, 32, 16384, 64), (2, 32, 16384, 64), False): (_D64_ROUTE, _D64_NONCAUSAL_MHA_CONFIG),
    ((2, 32, 16384, 64), (2, 4, 16384, 64), False): (_D64_ROUTE, _D64_NONCAUSAL_GQA8_CONFIG),
    ((2, 32, 16384, 64), (2, 32, 16384, 64), True): (_D64_ROUTE, _D64_CAUSAL_MHA_M256_XCD_64_CONFIG),
    ((2, 32, 16384, 64), (2, 4, 16384, 64), True): (_D64_ROUTE, _D64_CAUSAL_GQA8_M256_64_CONFIG),
    ((4, 48, 4096, 64), (4, 6, 4096, 64), True): (_D64_ROUTE, _D64_CAUSAL_GQA8_M256_16_CONFIG),
    ((4, 48, 4096, 64), (4, 6, 8192, 64), True): (_D64_ROUTE, _D64_CAUSAL_GQA8_M256_16_CONFIG),
    ((4, 48, 4096, 64), (4, 6, 12288, 64), True): (_D64_ROUTE, _D64_CAUSAL_GQA8_M256_16_CONFIG),
    ((4, 48, 4096, 64), (4, 6, 16384, 64), True): (_D64_ROUTE, _D64_CAUSAL_GQA8_M256_16_CONFIG),
    ((1, 8, 16384, 64), (1, 8, 16384, 64), True): (_D64_ROUTE, _D64_CAUSAL_MHA_M256_XCD_64_CONFIG),
    ((1, 8, 16640, 64), (1, 8, 16640, 64), True): (_D64_ROUTE, _D64_CAUSAL_MHA_M256_XCD_65_CONFIG),
    ((1, 4, 32768, 64), (1, 4, 32768, 64), True): (_D64_ROUTE, _D64_CAUSAL_MHA_M256_LINEAR_128_CONFIG),
    ((3, 3, 16384, 64), (3, 3, 16384, 64), True): (_D64_ROUTE, _D64_CAUSAL_MHA_M256_LINEAR_64_CONFIG),
    ((2, 8, 16384, 64), (2, 8, 16384, 64), True): (_D64_ROUTE, _D64_CAUSAL_MHA_M256_XCD_64_CONFIG),
    ((16, 27, 200, 128), (16, 27, 200, 128), False):
    (_D128_SHORT_ROUTE, frozenset({_D128_NONCAUSAL_SPLIT_CONFIG, _D128_EXACT_CONFIG})),
    ((16, 27, 200, 128), (16, 27, 200, 128), True):
    (_D128_SHORT_ROUTE, frozenset({_D128_CAUSAL_SPLIT_CONFIG, _D128_EXACT_CONFIG})),
    ((16, 64, 1024, 128), (16, 8, 1024, 128), False): (_D128_INTERLEAVED_ROUTE, _D128_INTERLEAVED_CONFIG),
    ((16, 64, 2048, 128), (16, 8, 2048, 128), False): (_D128_INTERLEAVED_ROUTE, _D128_INTERLEAVED_CONFIG),
    ((16, 64, 4096, 128), (16, 8, 4096, 128), False): (_D128_INTERLEAVED_ROUTE, _D128_INTERLEAVED_CONFIG),
    ((16, 64, 8192, 128), (16, 8, 8192, 128), False): (_D128_INTERLEAVED_ROUTE, _D128_BM32_CONFIG),
    ((16, 16, 1024, 128), (16, 16, 1024, 128), False): (_D128_INTERLEAVED_ROUTE, _D128_INTERLEAVED_CONFIG),
    ((16, 16, 2048, 128), (16, 16, 2048, 128), False): (_D128_INTERLEAVED_ROUTE, _D128_INTERLEAVED_CONFIG),
    ((16, 16, 4096, 128), (16, 16, 4096, 128), False): (_D128_INTERLEAVED_ROUTE, _D128_IGLP3_CONFIG),
    ((16, 16, 4096, 128), (16, 16, 4096, 128), True): (_D128_INTERLEAVED_ROUTE, _D128_CAUSAL_PEEL_CONFIG),
    ((16, 16, 8192, 128), (16, 16, 8192, 128), True): (_D128_INTERLEAVED_ROUTE, _D128_CAUSAL_PEEL_CONFIG),
    ((16, 16, 16384, 128), (16, 16, 16384, 128), True): (_D128_INTERLEAVED_ROUTE, _D128_CAUSAL_PEEL_CONFIG),
    ((16, 64, 1024, 128), (16, 8, 1024, 128), True): (_D128_INTERLEAVED_ROUTE, _D128_INTERLEAVED_CONFIG),
    ((16, 64, 2048, 128), (16, 8, 2048, 128), True): (_D128_INTERLEAVED_ROUTE, _D128_HEAD_SPLIT_CONFIG),
    ((16, 64, 4096, 128), (16, 8, 4096, 128), True): (_D128_INTERLEAVED_ROUTE, _D128_HEAD_SPLIT_CONFIG),
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
    if dispatch.main_entry is gfx950_bwd._attn_bwd_dkdv_dq_d128_gqa_kernel:
        main_entry = _D128_INTERLEAVED_ENTRY
    elif dispatch.main_entry is gfx950_bwd._dense_bwd_dkdv_dq_bm32_kernel:
        main_entry = _D128_BM32_ENTRY
    else:
        return None
    if dispatch.convert_entry is not gfx950_bwd._attn_bwd_dq_native_convert_kernel:
        return None
    return (
        main_entry,
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
        dispatch.kv_splits,
        dispatch.phase_iglp,
        dispatch.peel_causal,
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
            dispatch = gfx950_bwd._select_d128_interleaved_dispatch(tuple(query.shape), tuple(key.shape), is_causal)
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
    with torch.cuda.device(query.device):
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


def _is_varlen_performance_validated(query, key, cum_seq_q, max_q, max_k, is_causal, q_offsets=None, k_offsets=None,
                                     tensors=()):
    signature = (tuple(query.shape), tuple(key.shape), cum_seq_q.numel() - 1, max_q, max_k, bool(is_causal))
    expected = _PERFORMANCE_VALIDATED_VARLEN_SIGNATURES.get(signature)
    if expected is None:
        return False
    if any(tensor.data_ptr() % _REQUIRED_BASE_ALIGNMENT_BYTES for tensor in (query, key, *tensors)):
        return False
    if q_offsets is None:
        # Cheap rejection before synchronization or plan allocation.
        return True
    packed = struct.pack(f"<{len(q_offsets) + len(k_offsets)}i", *q_offsets, *k_offsets)
    return hashlib.sha256(packed).hexdigest() == expected


def _varlen_support_error(grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p,
                          is_causal, scale, window_size_left, window_size_right):
    if dropout_p != 0.0:
        return "dropout_p must be zero"
    if torch.are_deterministic_algorithms_enabled():
        return "deterministic algorithms are enabled"
    left = -1 if window_size_left is None else window_size_left
    right = -1 if window_size_right is None else window_size_right
    if left != -1 or right not in ((-1, 0) if is_causal else (-1, )):
        return "windowed attention is unsupported"
    if query.device.type != "cuda" or query.layout is not torch.strided:
        return "query must be a strided CUDA tensor"
    if query.ndim != 3 or key.ndim != 3:
        return "query and key must be packed THD tensors"
    if query.shape[0] <= 0 or key.shape[0] <= 0 or max_q <= 0 or max_k <= 0:
        return "token totals and maxima must be positive"
    if query.shape[-1] != 128 or key.shape[-1] != 128:
        return "packed backward requires D128"
    if query.shape[1] <= 0 or key.shape[1] <= 0 or query.shape[1] % key.shape[1]:
        return "Q heads must be divisible by KV heads"
    if is_causal and query.shape[1] != key.shape[1]:
        return "causal packed backward requires MHA"
    if not math.isfinite(float(scale)):
        return "scale must be finite"
    for tensor, shape in ((query, query.shape), (key, key.shape), (value, key.shape), (out, query.shape),
                          (grad_out, query.shape)):
        if tensor.layout is not torch.strided or tensor.shape != shape or tensor.device != query.device:
            return "Q/K/V/O/dO shapes and devices must match"
        if tensor.dtype is not torch.bfloat16:
            return "packed backward requires BF16 inputs"
    if not query.is_contiguous() or not key.is_contiguous():
        return "Q and K must be contiguous"
    if is_causal:
        if value.stride(-1) != 1 or value.stride(-2) != 128 or value.stride(0) < key.shape[1] * 128:
            return "V must have dense head and D axes"
    elif not value.is_contiguous():
        return "V must be contiguous"
    if (logsumexp.shape != (query.shape[1], query.shape[0]) or logsumexp.dtype is not torch.float32
            or logsumexp.device != query.device or not logsumexp.is_contiguous()):
        return "LSE must be contiguous FP32 (heads, total_q)"
    for offsets in (cum_seq_q, cum_seq_k):
        if (offsets is None or offsets.ndim != 1 or offsets.numel() < 2 or offsets.dtype is not torch.int32
                or offsets.device != query.device or not offsets.is_contiguous()):
            return "offsets must be contiguous CUDA int32 prefix sums"
    if cum_seq_q.shape != cum_seq_k.shape:
        return "Q and KV offsets must describe the same batch"
    arch = getattr(torch.cuda.get_device_properties(query.device), "gcnArchName", "")
    if not arch.startswith("gfx950"):
        return "gfx950 is required"
    return None


def _tlx_flash_attention_backward(native_kernel, dispatch_keys, grad_out, query, key, value, out, logsumexp, cum_seq_q,
                                  cum_seq_k, max_q, max_k, dropout_p, is_causal, rng_state, unused, *, scale=None,
                                  window_size_left=None, window_size_right=None):

    def native():
        return native_kernel.call_boxed(dispatch_keys, grad_out, query, key, value, out, logsumexp, cum_seq_q,
                                        cum_seq_k, max_q, max_k, dropout_p, is_causal, rng_state, unused, scale=scale,
                                        window_size_left=window_size_left, window_size_right=window_size_right)

    resolved_scale = 128**-0.5 if scale is None else scale
    error = _varlen_support_error(grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
                                  dropout_p, is_causal, resolved_scale, window_size_left, window_size_right)
    if error is not None or not _is_varlen_performance_validated(query, key, cum_seq_q, max_q, max_k, is_causal):
        return native()
    with torch.cuda.device(query.device):
        # Offset values determine correctness as well as compact workspace
        # size. Do not infer them from totals or cache mutable tensor pointers.
        # Capture cannot synchronize for validation; explicit prepared plans
        # remain available for capture through tlx.ops.
        if torch.cuda.is_current_stream_capturing():
            return native()
        try:
            q_offsets, q_lengths = gfx950_varlen_bwd._read_cu_seqlens("cum_seq_q", cum_seq_q)
            k_offsets, k_lengths = ((q_offsets, q_lengths) if cum_seq_k is cum_seq_q else
                                    gfx950_varlen_bwd._read_cu_seqlens("cum_seq_k", cum_seq_k))
            if (q_offsets[-1] != query.shape[0] or k_offsets[-1] != key.shape[0] or max(q_lengths) > max_q
                    or max(k_lengths) > max_k or (is_causal and q_offsets != k_offsets)):
                return native()
        except (TypeError, ValueError):
            return native()
        # Match ATen's normalization for expanded upstream gradients.
        tlx_out, tlx_grad_out = out.contiguous(), grad_out.contiguous()
        if not _is_varlen_performance_validated(query, key, cum_seq_q, max_q, max_k, is_causal, q_offsets, k_offsets,
                                                (value, tlx_out, tlx_grad_out, logsumexp)):
            return native()
        # Preparation compiles/launches schedule kernels. Their errors must
        # propagate, just like errors from the backward kernels below.
        plan = gfx950_varlen_bwd.prepare_varlen_backward(cum_seq_q, cum_seq_k)
        try:
            gfx950_varlen_bwd._validate_backward_inputs(query, key, value, tlx_out, tlx_grad_out, logsumexp, plan,
                                                        resolved_scale, is_causal, dq_atomic_fp32=True)
        except (TypeError, ValueError):
            return native()
        # Kernel/runtime failures propagate; never retry native after launch.
        return gfx950_varlen_bwd.fa_varlen_backward(query, key, value, tlx_out, tlx_grad_out, logsumexp, plan,
                                                    resolved_scale, is_causal, dq_atomic_fp32=True)


def register_tlx_gfx950_flash_attention_backward() -> _TLXFlashAttentionHandle:
    """Install conditional gfx950 dense and packed-backward overrides."""
    get_kernel = getattr(torch.library, "get_kernel", None)
    if get_kernel is None:
        raise RuntimeError("TLX FlashAttention integration requires torch.library.get_kernel")

    op = torch.ops.aten._scaled_dot_product_flash_attention_backward.default
    native_kernel: Any = get_kernel(op, "CUDA")
    varlen_kernel: Any = get_kernel(torch.ops.aten._flash_attention_backward.default, "CUDA")
    library = torch.library.Library("aten", "IMPL", "CUDA")
    library.impl(
        "_scaled_dot_product_flash_attention_backward",
        functools.partial(_tlx_scaled_dot_product_flash_attention_backward, native_kernel),
        "CUDA",
        with_keyset=True,
    )
    library.impl(
        "_flash_attention_backward",
        functools.partial(_tlx_flash_attention_backward, varlen_kernel),
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
