# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""
Cross-attention kernels with optional TMA (Tensor Memory Accelerator) support.

Forward kernels support TMA and non-TMA paths:
    - TMA path (ENABLE_TMA=True): Uses tl.make_tensor_descriptor for efficient memory access.
      Requires Triton 3.3.2+ with on-device TMA descriptor API.
    - Non-TMA path (ENABLE_TMA=False): Uses standard tl.load/tl.store with pointer arithmetic.
      Works on all hardware and Triton versions.
"""

import os
from enum import Enum
from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from stubs import switch_to_contiguous_if_needed
from stubs import (
    is_sm90,
    is_sm90_plus,
    maybe_register_custom_op,
)
from stubs import (
    autotune_max_seq_len,
    next_power_of_2,
    triton_autotune,
)
from stubs import get_full_autotune
from triton_attention_utils import (
    backward_softmax_activation_scaled_alpha,
    fast_silu,
    forward_softmax_activation_scaled_alpha,
    forward_softmax_activation_trans_scaled_alpha,
)
from triton_hstu_cross_attention import (
    _attn_bwd_preprocess,
    backward_common_preprocess,
    backward_d_silu_activation,
    backward_d_softmax_activation,
    backward_silu_activation,
    backward_valid_mask,
    forward_custom_vars,
    forward_epilogue,
    forward_softmax_common_preprocess,
    forward_valid_mask,
    target_common_preprocess,
    uih_common_preprocess,
)

# Check for on-device TMA API availability (requires Triton 3.5+)
HAS_TENSOR_DESCRIPTOR = hasattr(tl, "make_tensor_descriptor")


# Frozen (hashable) wrapper for the autoWS per-dot schedule attrs, usable in a
# triton.Config kwarg. Supports .get(key) like a dict but is hashable for Triton's
# JIT cache key. Mirrors the pattern in fused_attention_ws_device_tma.py.
class FrozenDotAttrs:
    def __init__(self, d):
        import json as _json

        self._data = d
        self._hash = hash(_json.dumps(d, sort_keys=True)) if d else hash(None)

    def get(self, key, default=None):
        return self._data.get(key, default) if self._data else default

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, FrozenDotAttrs):
            return self._data == other._data
        return NotImplemented

    def __repr__(self):
        return f"FrozenDotAttrs({self._data})"

    def __bool__(self):
        return bool(self._data)


# autoWS per-dot schedule for `_hstu_attn_bwd_inner`. Keys map to the 5 dots:
#   qkT = S = k@q^T ; dv = p@do ; dpT = v@do^T ; dk = ds^T@q ; dq = ds^T@k -> dq^T
# Channel format: "role,space,copy,id" (role: opndA/opndB inputs, opndD accumulator;
# space: tmem/smem; copy: buffer.copy depth; id: buffer.id).
#
# DEFAULT: the schedule this kernel shipped with (FA-derived; dq^T on its own
# single-copy TMEM buffer id 11, dP on id 5).
_HSTU_BWD_DOT_ATTRS_DEFAULT = FrozenDotAttrs(
    {
        "qkT": {"stage": "0", "order": "0", "channels": ["opndD,tmem,1,2"]},
        "dv": {"stage": "0", "order": "2", "channels": ["opndA,tmem,1,2", "opndD,tmem,1,7"]},
        "dpT": {"stage": "0", "order": "2", "channels": ["opndD,tmem,1,5"]},
        "dk": {"stage": "1", "order": "1", "channels": ["opndA,smem,1,8", "opndD,tmem,1,10"]},
        "dk_shared": {"stage": "1", "order": "1", "channels": ["opndA,smem,1,8", "opndD,tmem,1,7"]},
        "dq": {"stage": "1", "order": "1", "channels": ["opndB,smem,1,8", "opndD,tmem,1,11"]},
    }
)

# TLX-aligned: mirror the hand-written TLX kernel (attn_bwd_ws) TMEM buffer scheme.
# TLX reuses the dP tile in dq's TMEM slot (`dp_tiles reuse=dq_tiles`): dP is fully
# consumed (folded into dqk_trans) before dq^T is produced, so they can share one
# single-copy slot. Here that means dP's accumulator points at dq's buffer id 11
# (the larger 128x64 tile, which subsumes dP's 64x64) instead of a separate id 5.
# Stages/order match DEFAULT (== TLX: qkT/dpT/dv "current", dq/dk one iteration
# behind). Everything single-copy, as in TLX (NUM_BUFFERS_TMEM=1).
_HSTU_BWD_DOT_ATTRS_TLX = FrozenDotAttrs(
    {
        "qkT": {"stage": "0", "order": "0", "channels": ["opndD,tmem,1,2"]},
        "dv": {"stage": "0", "order": "2", "channels": ["opndA,tmem,1,2", "opndD,tmem,1,7"]},
        "dpT": {"stage": "0", "order": "2", "channels": ["opndD,tmem,1,11"]},
        "dk": {"stage": "1", "order": "1", "channels": ["opndA,smem,1,8", "opndD,tmem,1,10"]},
        "dk_shared": {"stage": "1", "order": "1", "channels": ["opndA,smem,1,8", "opndD,tmem,1,7"]},
        "dq": {"stage": "1", "order": "1", "channels": ["opndB,smem,1,8", "opndD,tmem,1,11"]},
    }
)

# for SHARED_KV + DP and _HSTU_COMPUTE_FOLD
# --> we can do dk += dot(...) dk += dot(...)
# --> we use "dk_shared" instead of "dk"
# Block b0 schedule. dq (opndD id 11) is a SINGLE accumulator SHARED with block
# b1: both blocks MMA-accumulate dq^T into the same TMEM tile (dq is a reduction
# over KV blocks -- like the hand-written attn_bwd_ws_2kv), then ONE store. So
# dpT must NOT time-share id 11 with dq (as the single-KV _TLX schedule does); it
# gets its own tile (id 5). Everything else (qkT/S+P id 2, dv/dk id 7) is b0's.
_HSTU_BWD_DOT_ATTRS_2KV = FrozenDotAttrs(
    {
        "qkT": {"stage": "0", "order": "0", "channels": ["opndD,tmem,1,2"]},
        "dv": {"stage": "0", "order": "2", "channels": ["opndA,tmem,1,2", "opndD,tmem,1,7"]},
        "dpT": {"stage": "0", "order": "2", "channels": ["opndD,tmem,1,5"]},
        "dk": {"stage": "1", "order": "1", "channels": ["opndA,smem,1,8", "opndD,tmem,1,10"]},
        "dk_shared": {"stage": "1", "order": "1", "channels": ["opndA,smem,1,8", "opndD,tmem,1,7"]},
        "dq": {"stage": "1", "order": "1", "channels": ["opndB,smem,1,8", "opndD,tmem,1,11"]},
    }
)


# Triton's @jit frontend accepts `attrs=` only as a dict literal or a bare
# module-global name bound to a dict -- it rejects passing a FrozenDotAttrs (or
# calling .get) as a kernel constexpr param or inside the kernel. So we expose the
# ACTIVE schedule as per-dot module-global dicts that the dots reference by name,
# and `set_bwd_dot_attrs()` swaps them (recompile picks up the new globals).
#
# ACTIVE = the TLX-aligned schedule+reuse: same SWP stages/order as DEFAULT
# (qk/dP/dv stage-0 ahead, dq/dk stage-1 behind; order qk -> dq/dk -> dP/dv) AND
# the hand-TLX TMEM reuse groups -- S+P share id2, and dP time-shares dqᵀ's slot
# id11 (TLX `dp_tiles reuse=dq_tiles`, freeing id5). dk/dv/dsᵀ standalone. Switch
# to _HSTU_BWD_DOT_ATTRS_DEFAULT (dP on its own id5) via set_bwd_dot_attrs().
_HSTU_ATTRS_QKT = tl.constexpr(_HSTU_BWD_DOT_ATTRS_TLX.get("qkT"))
_HSTU_ATTRS_DV = tl.constexpr(_HSTU_BWD_DOT_ATTRS_TLX.get("dv"))
_HSTU_ATTRS_DPT = tl.constexpr(_HSTU_BWD_DOT_ATTRS_TLX.get("dpT"))
_HSTU_ATTRS_DK = tl.constexpr(_HSTU_BWD_DOT_ATTRS_TLX.get("dk"))
_HSTU_ATTRS_DK_SHARED = tl.constexpr(_HSTU_BWD_DOT_ATTRS_TLX.get("dk_shared"))
_HSTU_ATTRS_DQ = tl.constexpr(_HSTU_BWD_DOT_ATTRS_TLX.get("dq"))

# Per-dot 2KV variants, selected in-kernel on the SHARED_KV + compute-fold path
# (SHARED_KV and _HSTU_COMPUTE_FOLD). The 2KV schedule is fixed to the TLX
# 2-KV-block layout and is NOT swapped by set_bwd_dot_attrs(); the active globals
# above still drive the non-fold paths.
_HSTU_ATTRS_QKT_2KV = tl.constexpr(_HSTU_BWD_DOT_ATTRS_2KV.get("qkT"))
_HSTU_ATTRS_DV_2KV = tl.constexpr(_HSTU_BWD_DOT_ATTRS_2KV.get("dv"))
_HSTU_ATTRS_DPT_2KV = tl.constexpr(_HSTU_BWD_DOT_ATTRS_2KV.get("dpT"))
_HSTU_ATTRS_DK_2KV = tl.constexpr(_HSTU_BWD_DOT_ATTRS_2KV.get("dk"))
_HSTU_ATTRS_DK_SHARED_2KV = tl.constexpr(_HSTU_BWD_DOT_ATTRS_2KV.get("dk_shared"))
_HSTU_ATTRS_DQ_2KV = tl.constexpr(_HSTU_BWD_DOT_ATTRS_2KV.get("dq"))

# MANUAL 2-KV DP (milestone 2): with warp specialization ON, the two KV blocks'
# per-block MMAs run concurrently and their operand/accumulator tiles are
# simultaneously live, so block b1 uses DISJOINT buffer.ids from b0 (which keeps
# _2KV above) for qkT/S+P (2->12), dv/dk (7->17), dpT (5->15), and the smem
# operands (8->9). This is the hand-written equivalent of the per-half id-remap
# the compiler DP pass is supposed to synthesize. The ONE exception is dq's
# accumulator (opndD id 11): it is INTENTIONALLY shared -- dq is a reduction over
# both KV blocks, accumulated into one TMEM tile and stored once.
_HSTU_BWD_DOT_ATTRS_2KV_B1 = FrozenDotAttrs(
    {
        "qkT": {"stage": "0", "order": "0", "channels": ["opndD,tmem,1,12"]},
        "dv": {"stage": "0", "order": "2", "channels": ["opndA,tmem,1,12", "opndD,tmem,1,17"]},
        "dpT": {"stage": "0", "order": "2", "channels": ["opndD,tmem,1,15"]},
        "dk": {"stage": "1", "order": "1", "channels": ["opndA,smem,1,9", "opndD,tmem,1,20"]},
        "dk_shared": {"stage": "1", "order": "1", "channels": ["opndA,smem,1,9", "opndD,tmem,1,17"]},
        # dq opndD id 11 is SHARED with b0 (single dq accumulator over both KV
        # blocks); only its input operand (opndB smem) is b1's own (id 9).
        "dq": {"stage": "1", "order": "1", "channels": ["opndB,smem,1,9", "opndD,tmem,1,11"]},
    }
)
_HSTU_ATTRS_QKT_2KV_B1 = tl.constexpr(_HSTU_BWD_DOT_ATTRS_2KV_B1.get("qkT"))
_HSTU_ATTRS_DV_2KV_B1 = tl.constexpr(_HSTU_BWD_DOT_ATTRS_2KV_B1.get("dv"))
_HSTU_ATTRS_DPT_2KV_B1 = tl.constexpr(_HSTU_BWD_DOT_ATTRS_2KV_B1.get("dpT"))
_HSTU_ATTRS_DK_2KV_B1 = tl.constexpr(_HSTU_BWD_DOT_ATTRS_2KV_B1.get("dk"))
_HSTU_ATTRS_DK_SHARED_2KV_B1 = tl.constexpr(_HSTU_BWD_DOT_ATTRS_2KV_B1.get("dk_shared"))
_HSTU_ATTRS_DQ_2KV_B1 = tl.constexpr(_HSTU_BWD_DOT_ATTRS_2KV_B1.get("dq"))

# "compute fold": fold dk_attn (dqk @ q) into the dv/dk accumulator (TLX 2kv
# parity) instead of a separate dk_attn tile + register add. Kept behind a
# constexpr gate (env HSTU_COMPUTE_FOLD=1) while the partition scheduler
# interaction (spurious correction partition) is under investigation.
_HSTU_COMPUTE_FOLD = tl.constexpr(os.environ.get("HSTU_COMPUTE_FOLD", "0") == "1")


def set_bwd_dot_attrs(cfg: "FrozenDotAttrs") -> None:
    """Select the autoWS per-dot schedule for `_hstu_attn_bwd_inner`.

    Pass _HSTU_BWD_DOT_ATTRS_DEFAULT or _HSTU_BWD_DOT_ATTRS_TLX. Call before the
    kernel compiles (use TRITON_ALWAYS_COMPILE=1 to force a recompile).
    """
    global _HSTU_ATTRS_QKT, _HSTU_ATTRS_DV, _HSTU_ATTRS_DPT, _HSTU_ATTRS_DK, _HSTU_ATTRS_DK_SHARED, _HSTU_ATTRS_DQ
    _HSTU_ATTRS_QKT = tl.constexpr(cfg.get("qkT"))
    _HSTU_ATTRS_DV = tl.constexpr(cfg.get("dv"))
    _HSTU_ATTRS_DPT = tl.constexpr(cfg.get("dpT"))
    _HSTU_ATTRS_DK = tl.constexpr(cfg.get("dk"))
    _HSTU_ATTRS_DK_SHARED = tl.constexpr(cfg.get("dk_shared"))
    _HSTU_ATTRS_DQ = tl.constexpr(cfg.get("dq"))




class FwdVariant(Enum):
    TRITON = "triton"
    TRITON_DYN_SPEC = "triton_dyn_spec"


class BwdVariant(Enum):
    AUTO = "auto"
    TRITON_REDKV = "triton"
    TRITON_REDQ = "triton_redq"
    # Experimental: reduce_dq (per-kv-head, atomic-free) layout with the inner Q
    # loop annotated for automatic warp specialization (autoWS / beta triton).
    TRITON_AUTOWS = "triton_autows"
    # Hand-written TLX warp-specialized reduce_dq (attn_bwd_ws). Dispatched here
    # so the existing autograd path + tritonbench --bwd contract drive it.
    TLX = "tlx"
    # TLX 2-KV-block data-partitioned reduce_dq (attn_bwd_ws_2kv): two independent
    # MMA groups per program. Shared-KV only (V aliases K).
    TLX_2KV = "tlx_2kv"
    # MANUAL 2-KV-block data-partitioned reduce_dq written explicitly in Triton
    # (_hstu_attn_bwd_redq_2kv): processes two KV blocks per step to get 2-way
    # parallelism WITHOUT triggering the compiler's automatic DP pass. Shared-KV +
    # compute-fold only. Milestone 1: WS off.
    TRITON_AUTOWS_2KV = "triton_autows_2kv"


# Global variables for variant selection, can be overridden in unit tests
FWD_VARIANT: FwdVariant = FwdVariant.TRITON_DYN_SPEC
BWD_VARIANT: BwdVariant = BwdVariant.AUTO


def set_fwd_variant(variant: FwdVariant) -> None:
    """Set the forward variant globally. Useful for unit test coverage."""
    global FWD_VARIANT
    FWD_VARIANT = variant


def set_bwd_variant(variant: BwdVariant) -> None:
    """Set the backward variant globally. Useful for unit test coverage."""
    global BWD_VARIANT
    BWD_VARIANT = variant


def get_fwd_variant() -> FwdVariant:
    """Get the current forward variant."""
    return FWD_VARIANT


def get_bwd_variant() -> BwdVariant:
    """Get the current backward variant."""
    env = os.environ.get("HSTU_BWD_VARIANT")
    if env:
        return BwdVariant(env)
    return BWD_VARIANT


@triton.jit
def _compute_offsets(
    H,
    G,
    BLOCK_M: tl.constexpr,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    TRUNCATE_METHOD: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h // G
    start_m = tl.program_id(0) * BLOCK_M
    seq_start_kv = tl.load(seq_offsets + off_z)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    if TRUNCATE_METHOD != "none":
        truncated_len = tl.minimum(seq_len_kv, max_seq_len.to(tl.int32))
        # KEEP_LAST: skip early tokens; KEEP_FIRST: keep start unchanged
        if TRUNCATE_METHOD == "keep_last":
            seq_start_kv = seq_start_kv + (seq_len_kv - truncated_len)
        seq_len_kv = truncated_len
    seq_start_q = tl.load(seq_offsets_q + off_z)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    return (
        start_m,
        off_h,
        off_h_kv,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    )




def get_fwd_triton_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "TMA_STORE": tma_trans[0],
                "TRANS": tma_trans[1],
                "NUM_STAGES": ns,
            },
            num_stages=0,
            num_warps=nw,
        )
        for bm in [32, 64, 128]
        for bn in [32, 64, 128]
        for nw in [2, 4, 8]
        for ns in [1, 2, 4]
        for off in [False]
        for mask in [True]
        for tma_trans in [(True, False), (False, True), (False, False)]
        # trans doesn't work with TMA
    ]
    return configs


@triton.jit
def forward_valid_mask_trans(
    offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL: tl.constexpr
):
    valid_mask = (offs_m[None, :] < seq_len_q) & (offs_n[:, None] < seq_len_kv)
    if HAS_CAUSAL:
        offs_m = offs_m + seq_len_kv - uih_len_q
        causal_mask = offs_m[None, :] >= offs_n[:, None]
        valid_mask = valid_mask & causal_mask
    return valid_mask


@triton.jit
def forward_epilogue_trans(
    acc,
    l_i,
    m_i,
    offs_m,
    seq_start_q,
    stride_mm,
    seq_len_q,
    M,
    off_h,
    num_softmax_heads: tl.constexpr,
):
    if off_h + 1 < num_softmax_heads + 1:  # For AOTInductor lowering walkaround.
        acc = acc / l_i[None, :]
        m_i += tl.math.log2(l_i)
        m_ptrs = M + (offs_m + seq_start_q) * stride_mm + off_h
        tl.store(m_ptrs, m_i, mask=offs_m < seq_len_q)
    return acc


@triton.jit
def _attn_fwd_compute(
    acc,
    alpha,
    desc_k,
    desc_v,
    q,
    K,
    V,
    offs_m,
    offs_n_0,
    off_h,
    off_h_kv,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    attn_scale,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_mm,
    m_i,
    l_i,
    M,
    start_m,
    uih_len_q,
    num_softmax_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TRANS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    SHARED_KV: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    IS_SOFTMAX: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
):
    if HAS_CAUSAL:
        high = start_m + BLOCK_M + seq_len_kv - uih_len_q
        high = tl.cdiv(min(high, seq_len_kv), BLOCK_N) * BLOCK_N
        last_block_start_n = tl.maximum(0, (tl.cdiv(high, BLOCK_N) - 1) * BLOCK_N)
    else:
        last_block_start_n = tl.maximum(0, (tl.cdiv(seq_len_kv, BLOCK_N) - 1) * BLOCK_N)
    if IS_SOFTMAX:
        alpha = alpha * 1.44269504
    # Main loop - no masking needed for non-last kv blocks
    for start_n in tl.range(0, last_block_start_n, BLOCK_N):
        if ENABLE_TMA:
            k = desc_k.load(
                [
                    (seq_start_kv + start_n).to(tl.int32),
                    off_h_kv * stride_kh,
                ]
            )
        else:
            offs_n_load = start_n + tl.arange(0, BLOCK_N)
            offs_qk_d = tl.arange(0, HEAD_DIM)
            k_ptrs = (
                K
                + (seq_start_kv + offs_n_load).to(tl.int64)[:, None] * stride_kn
                + (off_h_kv * stride_kh + offs_qk_d)[None, :]
            )
            k = tl.load(k_ptrs)

        if TRANS:
            qk = tl.dot(k, tl.trans(q))  # BN by BM
        else:
            qk = tl.dot(q, tl.trans(k))  # BM by BN
        if HAS_CAUSAL:
            offs_n = offs_n_0 + start_n
            if TRANS:
                causal_mask = (offs_m[None, :] + seq_len_kv - uih_len_q) >= offs_n[
                    :, None
                ]
            else:
                causal_mask = (offs_m[:, None] + seq_len_kv - uih_len_q) >= offs_n[
                    None, :
                ]
            if IS_SOFTMAX:
                qk = tl.where(causal_mask, qk, float("-inf"))
            else:
                qk = tl.where(causal_mask, qk, 0.0)
        if IS_SOFTMAX:
            qk = qk * alpha
            if TRANS:
                m_ij = tl.maximum(m_i, tl.max(qk, 0))
                qk -= m_ij[None, :]
                act_qk = tl.math.exp2(qk)
                corr = tl.math.exp2(m_i - m_ij)
                l_ij = tl.sum(act_qk, 0)
                acc = acc * corr[None, :]
            else:
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
                act_qk = tl.math.exp2(qk)
                corr = tl.math.exp2(m_i - m_ij)
                l_ij = tl.sum(act_qk, 1)
                acc = acc * corr[:, None]
            l_i = l_i * corr + l_ij
            m_i = m_ij
        else:
            qk = qk * alpha
            act_qk = fast_silu(qk, MULT_BY_X=True)
        if SHARED_KV:
            v = k
        else:
            if ENABLE_TMA:
                v = desc_v.load(
                    [
                        (seq_start_kv + start_n).to(tl.int32),
                        off_h_kv * stride_vh,
                    ]
                )
            else:
                offs_n_load = start_n + tl.arange(0, BLOCK_N)
                offs_v_d = tl.arange(0, BLOCK_D_V)
                v_ptrs = (
                    V
                    + (seq_start_kv + offs_n_load).to(tl.int64)[:, None] * stride_vn
                    + (off_h_kv * stride_vh + offs_v_d)[None, :]
                )
                v = tl.load(v_ptrs)
        act_qk = act_qk.to(v.dtype)
        if TRANS:
            acc += tl.dot(tl.trans(v), act_qk)
        else:
            acc += tl.dot(act_qk, v)

    # Last kv block - needs masking for out-of-bounds elements;
    # q masking is not needed here because we mask when storing results.
    start_n = last_block_start_n
    if ENABLE_TMA:
        k = desc_k.load(
            [
                (seq_start_kv + start_n).to(tl.int32),
                off_h_kv * stride_kh,
            ]
        )
    else:
        offs_n_load = start_n + tl.arange(0, BLOCK_N)
        offs_qk_d = tl.arange(0, HEAD_DIM)
        k_ptrs = (
            K
            + (seq_start_kv + offs_n_load).to(tl.int64)[:, None] * stride_kn
            + (off_h_kv * stride_kh + offs_qk_d)[None, :]
        )
        k = tl.load(
            k_ptrs,
            mask=(offs_n_load < seq_len_kv)[:, None],
            other=0.0,
        )

    offs_n = offs_n_0 + start_n
    if TRANS:
        qk = tl.dot(k, tl.trans(q))  # BN by BM
    else:
        qk = tl.dot(q, tl.trans(k))  # BM by BN
    if TRANS:
        valid_mask = forward_valid_mask_trans(
            offs_m,
            offs_n,
            uih_len_q,
            seq_len_q,
            seq_len_kv,
            HAS_CAUSAL,
        )
    else:
        valid_mask = forward_valid_mask(
            offs_m,
            offs_n,
            uih_len_q,
            seq_len_q,
            seq_len_kv,
            HAS_CAUSAL,
        )
    if IS_SOFTMAX:
        if TRANS:
            act_qk, acc, l_i, m_i = forward_softmax_activation_trans_scaled_alpha(
                qk, alpha, valid_mask, m_i, acc, l_i
            )
        else:
            act_qk, acc, l_i, m_i = forward_softmax_activation_scaled_alpha(
                qk, alpha, valid_mask, m_i, acc, l_i
            )
    else:
        masked_alpha = tl.where(valid_mask, alpha, 0.0)
        qk = qk * masked_alpha
        act_qk = fast_silu(qk, MULT_BY_X=True)
    if SHARED_KV:
        v = k
    else:
        if ENABLE_TMA:
            v = desc_v.load(
                [
                    (seq_start_kv + start_n).to(tl.int32),
                    off_h_kv * stride_vh,
                ]
            )
        else:
            offs_n_load = start_n + tl.arange(0, BLOCK_N)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            v_ptrs = (
                V
                + (seq_start_kv + offs_n_load).to(tl.int64)[:, None] * stride_vn
                + (off_h_kv * stride_vh + offs_v_d)[None, :]
            )
            v = tl.load(
                v_ptrs,
                mask=(offs_n_load < seq_len_kv)[:, None],
                other=0.0,
            )
    act_qk = act_qk.to(v.dtype)
    if TRANS:
        acc += tl.dot(tl.trans(v), act_qk)
    else:
        acc += tl.dot(act_qk, v)

    # epilogue
    if IS_SOFTMAX:
        if TRANS:
            acc = forward_epilogue_trans(
                acc,
                l_i,
                m_i,
                offs_m,
                seq_start_q,
                stride_mm,
                seq_len_q,
                M,
                off_h,
                num_softmax_heads,
            )
        else:
            acc = forward_epilogue(
                acc,
                l_i,
                m_i,
                offs_m,
                seq_start_q,
                stride_mm,
                seq_len_q,
                M,
                off_h,
                num_softmax_heads,
            )
    else:
        scale = tl.load(attn_scale).to(tl.float32)
        acc = acc * scale
    return acc


@triton.jit
def _attn_fwd_triton_inner(
    alpha,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    Q,
    K,
    V,
    Out,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    attn_scale,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    start_m,
    off_h,
    off_h_kv,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    M,
    stride_mm,
    uih_len_q,
    num_softmax_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    TRANS: tl.constexpr,
    TMA_STORE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    SHARED_KV: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
):
    # initialize offsets
    if start_m < seq_len_q:
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n_0 = tl.arange(0, BLOCK_N)
        qo_offset_y_split = seq_start_q + start_m
        if ENABLE_TMA:
            q = desc_q.load([qo_offset_y_split.to(tl.int32), off_h * stride_qh])
        else:
            offs_qk_d = tl.arange(0, HEAD_DIM)
            q_ptrs = (
                Q
                + (qo_offset_y_split + tl.arange(0, BLOCK_M)).to(tl.int64)[:, None]
                * stride_qm
                + (off_h * stride_qh + offs_qk_d)[None, :]
            )
            q = tl.load(
                q_ptrs,
                mask=(offs_m < seq_len_q)[:, None],
                other=0.0,
            )
        m_i, l_i = forward_softmax_common_preprocess(off_h, num_softmax_heads, BLOCK_M)
        if TRANS:
            acc = tl.zeros([BLOCK_D_V, BLOCK_M], dtype=tl.float32)
        else:
            acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        # when seq_len_kv is 0, we can skip computation but still needs to write acc to out
        # to avoid NaN.
        if seq_len_kv > 0:
            if off_h < num_softmax_heads:
                acc = _attn_fwd_compute(
                    acc,
                    alpha,
                    desc_k,
                    desc_v,
                    q,
                    K,
                    V,
                    offs_m,
                    offs_n_0,
                    off_h,
                    off_h_kv,
                    seq_start_kv,
                    seq_len_kv,
                    seq_start_q,
                    seq_len_q,
                    attn_scale,
                    stride_kn,
                    stride_kh,
                    stride_vn,
                    stride_vh,
                    stride_mm,
                    m_i,
                    l_i,
                    M,
                    start_m,
                    uih_len_q,
                    num_softmax_heads,
                    HEAD_DIM,
                    BLOCK_D_V,
                    BLOCK_M,
                    BLOCK_N,
                    TRANS,
                    NUM_STAGES,
                    SHARED_KV,
                    ENABLE_TMA,
                    IS_SOFTMAX=True,
                    HAS_CAUSAL=HAS_CAUSAL,
                )
            else:
                acc = _attn_fwd_compute(
                    acc,
                    alpha,
                    desc_k,
                    desc_v,
                    q,
                    K,
                    V,
                    offs_m,
                    offs_n_0,
                    off_h,
                    off_h_kv,
                    seq_start_kv,
                    seq_len_kv,
                    seq_start_q,
                    seq_len_q,
                    attn_scale,
                    stride_kn,
                    stride_kh,
                    stride_vn,
                    stride_vh,
                    stride_mm,
                    m_i,
                    l_i,
                    M,
                    start_m,
                    uih_len_q,
                    num_softmax_heads,
                    HEAD_DIM,
                    BLOCK_D_V,
                    BLOCK_M,
                    BLOCK_N,
                    TRANS,
                    NUM_STAGES,
                    SHARED_KV,
                    ENABLE_TMA,
                    IS_SOFTMAX=False,
                    HAS_CAUSAL=HAS_CAUSAL,
                )
        out_offset = off_h.to(tl.int64) * stride_oh
        end_o = seq_start_q + seq_len_q
        # we are writing out Out.T which is hDim x BM
        if TMA_STORE and ENABLE_TMA:
            if TRANS:  # This does not work
                o_desc = tl.make_tensor_descriptor(
                    Out,
                    shape=[HEAD_DIM * H, end_o.to(tl.int32)],
                    # pyrefly: ignore [bad-argument-type]
                    strides=[1, HEAD_DIM * H],
                    block_shape=[BLOCK_D_V, BLOCK_M],
                )
                o_desc.store(
                    [
                        (out_offset).to(tl.int32),
                        (seq_start_q + start_m).to(tl.int32),
                    ],
                    acc.to(Out.dtype.element_ty),
                )
            else:
                o_desc = tl.make_tensor_descriptor(
                    Out,
                    shape=[end_o.to(tl.int32), HEAD_DIM * H],
                    # pyrefly: ignore [bad-argument-type]
                    strides=[HEAD_DIM * H, 1],
                    block_shape=[BLOCK_M, BLOCK_D_V],
                )
                o_desc.store(
                    [
                        (seq_start_q + start_m).to(tl.int32),
                        (out_offset).to(tl.int32),
                    ],
                    acc.to(Out.dtype.element_ty),
                )
        else:
            if TRANS:
                off_o = Out + seq_start_q * stride_om + off_h * stride_oh
                offs_m = start_m + tl.arange(0, BLOCK_M)
                offs_v_d = tl.arange(0, BLOCK_D_V)
                out_ptrs = off_o + offs_m[None, :] * stride_om + offs_v_d[:, None]
                acc = acc.to(Out.dtype.element_ty)
                tl.store(out_ptrs, acc, mask=(offs_m < seq_len_q)[None, :])
            else:
                off_o = Out + seq_start_q * stride_om + off_h * stride_oh
                offs_m = start_m + tl.arange(0, BLOCK_M)
                offs_v_d = tl.arange(0, BLOCK_D_V)
                out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
                acc = acc.to(Out.dtype.element_ty)
                tl.store(out_ptrs, acc, mask=(offs_m < seq_len_q)[:, None])


@triton.autotune(
    configs=get_fwd_triton_configs(),
    key=[
        "Z",
        "HEAD_DIM",
        "AUTOTUNE_MAX_Q_LEN",
        "AUTOTUNE_MAX_SEQ_LEN",
        "num_softmax_heads",
    ],
)
@triton.jit
def _attn_fwd_triton(
    alpha,
    Z,
    H,
    G,
    Q,
    K,
    V,
    total_seq_len_q,
    total_seq_len_kv,
    Out,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    attn_scale,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    M,
    stride_mm,
    num_targets,
    AUTOTUNE_MAX_Q_LEN,
    AUTOTUNE_MAX_SEQ_LEN,
    num_softmax_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    TRANS: tl.constexpr,
    TMA_STORE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TRUNCATE_METHOD: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
):
    # Create TMA descriptors on device
    H_kv = H // G
    desc_q = None
    desc_k = None
    desc_v = None
    if ENABLE_TMA:
        desc_q = tl.make_tensor_descriptor(
            Q,
            shape=[total_seq_len_q, H * HEAD_DIM],
            # pyrefly: ignore [bad-argument-type]
            strides=[H * HEAD_DIM, 1],
            block_shape=[BLOCK_M, HEAD_DIM],
        )
        desc_k = tl.make_tensor_descriptor(
            K,
            shape=[total_seq_len_kv, H_kv * HEAD_DIM],
            # pyrefly: ignore [bad-argument-type]
            strides=[H_kv * HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
        desc_v = tl.make_tensor_descriptor(
            V,
            shape=[total_seq_len_kv, H_kv * BLOCK_D_V],
            # pyrefly: ignore [bad-argument-type]
            strides=[H_kv * BLOCK_D_V, 1],
            block_shape=[BLOCK_N, BLOCK_D_V],
        )

    (
        start_m,
        off_h,
        off_h_kv,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    ) = _compute_offsets(
        H, G, BLOCK_M, seq_offsets_q, seq_offsets, max_seq_len, TRUNCATE_METHOD
    )
    if HAS_CAUSAL:
        n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
        uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    else:
        uih_len_q = seq_len_q
    _attn_fwd_triton_inner(
        alpha,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        Q,
        K,
        V,
        Out,
        seq_offsets_q,
        seq_offsets,
        max_seq_len,
        attn_scale,
        stride_qm,
        stride_qh,
        stride_kn,
        stride_kh,
        stride_vn,
        stride_vh,
        stride_om,
        stride_oh,
        start_m,
        off_h,
        off_h_kv,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        M,
        stride_mm,
        uih_len_q,
        num_softmax_heads,
        HEAD_DIM,
        BLOCK_D_V,
        BLOCK_M,
        BLOCK_N,
        TRANS,
        TMA_STORE,
        NUM_STAGES,
        SHARED_KV=False,
        ENABLE_TMA=ENABLE_TMA,
        HAS_CAUSAL=HAS_CAUSAL,
    )




def get_fwd_triton_spec_configs() -> List[triton.Config]:
    if torch.version.hip:
        configs = []
        for BLOCK_M in [32, 64, 128]:
            for BLOCK_N in [32, 64]:
                for num_stages in [1, 2]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            for waves_per_eu in [1, 2, 3]:
                                configs.append(
                                    triton.Config(
                                        {
                                            "BLOCK_M": BLOCK_M,
                                            "BLOCK_N": BLOCK_N,
                                            "BLOCK_M1": BLOCK_M,
                                            "BLOCK_N1": BLOCK_N,
                                            "TMA_STORE": False,
                                            "TRANS": False,
                                            "NUM_STAGES": num_stages,
                                            "NUM_STAGES1": num_stages,
                                            "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                            "waves_per_eu": waves_per_eu,
                                            "kpack": 2,
                                            "IS_DYNAMIC": False,
                                        },
                                        num_stages=num_stages,
                                        num_warps=num_warps,
                                    )
                                )
        return configs
    if get_full_autotune():
        configs = [
            triton.Config(
                {
                    "BLOCK_M": bm,
                    "BLOCK_N": bn,
                    "BLOCK_M1": bm1,
                    "BLOCK_N1": bn1,
                    "TMA_STORE": tma_trans[0],
                    "TRANS": tma_trans[1],
                    "NUM_STAGES": ns,
                    "NUM_STAGES1": ns1,
                    "IS_DYNAMIC": is_dynamic,
                },
                num_stages=0,
                num_warps=nw,
            )
            for bm in [64, 128]
            for bn in [64, 128]
            for bm1 in [32, 64]
            for bn1 in [32, 64]
            for nw in [2, 4, 8]
            for ns in [2, 4]
            for ns1 in [2, 4]
            for tma_trans in [(True, False), (False, True), (False, False)]
            for is_dynamic in [True, False]
        ]
    else:
        configs = [
            triton.Config(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 64,
                    "TMA_STORE": False,
                    "TRANS": True,
                    "NUM_STAGES": 2,
                    "NUM_STAGES1": 2,
                    "IS_DYNAMIC": True,
                },
                num_stages=0,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 64,
                    "TMA_STORE": False,
                    "TRANS": True,
                    "NUM_STAGES": 2,
                    "NUM_STAGES1": 2,
                    "IS_DYNAMIC": True,
                },
                num_stages=0,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 64,
                    "BLOCK_N1": 64,
                    "TMA_STORE": False,
                    "TRANS": True,
                    "NUM_STAGES": 2,
                    "NUM_STAGES1": 2,
                    "IS_DYNAMIC": True,
                },
                num_stages=0,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 32,
                    "TMA_STORE": False,
                    "TRANS": True,
                    "NUM_STAGES": 2,
                    "NUM_STAGES1": 4,
                    "IS_DYNAMIC": True,
                },
                num_stages=0,
                num_warps=2,
            ),
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 64,
                    "TMA_STORE": False,
                    "TRANS": False,
                    "NUM_STAGES": 2,
                    "NUM_STAGES1": 2,
                    "IS_DYNAMIC": True,
                },
                num_stages=0,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 64,
                    "BLOCK_N1": 64,
                    "TMA_STORE": False,
                    "TRANS": False,
                    "NUM_STAGES": 2,
                    "NUM_STAGES1": 2,
                    "IS_DYNAMIC": True,
                },
                num_stages=0,
                num_warps=4,
            ),
        ]
        if is_sm90():
            # for H100 inference
            configs += [
                triton.Config(
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 64,
                        "BLOCK_M1": 32,
                        "BLOCK_N1": 64,
                        "TMA_STORE": False,
                        "TRANS": False,
                        "NUM_STAGES": 4,
                        "NUM_STAGES1": 4,
                        "IS_DYNAMIC": True,
                    },
                    num_stages=0,
                    num_warps=4,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 64,
                        "BLOCK_M1": 64,
                        "BLOCK_N1": 64,
                        "TMA_STORE": False,
                        "TRANS": False,
                        "NUM_STAGES": 4,
                        "NUM_STAGES1": 4,
                        "IS_DYNAMIC": False,
                    },
                    num_stages=0,
                    num_warps=4,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 64,
                        "BLOCK_M1": 64,
                        "BLOCK_N1": 64,
                        "TMA_STORE": False,
                        "TRANS": False,
                        "NUM_STAGES": 2,
                        "NUM_STAGES1": 2,
                        "IS_DYNAMIC": False,
                    },
                    num_stages=0,
                    num_warps=4,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 64,
                        "BLOCK_M1": 128,
                        "BLOCK_N1": 64,
                        "TMA_STORE": False,
                        "TRANS": False,
                        "NUM_STAGES": 4,
                        "NUM_STAGES1": 4,
                        "IS_DYNAMIC": False,
                    },
                    num_stages=0,
                    num_warps=8,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 64,
                        "BLOCK_M1": 128,
                        "BLOCK_N1": 64,
                        "TMA_STORE": False,
                        "TRANS": False,
                        "NUM_STAGES": 4,
                        "NUM_STAGES1": 4,
                        "IS_DYNAMIC": False,
                    },
                    num_stages=0,
                    num_warps=4,
                ),
            ]
    return configs


@triton_autotune(
    configs=get_fwd_triton_spec_configs(),
    key=[
        "AUTOTUNE_Z",
        "HEAD_DIM",
        "AUTOTUNE_MAX_Q_LEN",
        "AUTOTUNE_MAX_SEQ_LEN",
        "num_softmax_heads",
    ],
)
@triton.jit
def _attn_fwd_triton_spec(
    alpha,
    Z,
    H,
    G,
    Q,
    K,
    V,
    total_seq_len_q,
    total_seq_len_kv,
    Out,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    attn_scale,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    M,
    stride_mm,
    num_targets,
    AUTOTUNE_Z,
    AUTOTUNE_MAX_Q_LEN,
    AUTOTUNE_MAX_SEQ_LEN,
    num_softmax_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    TRANS: tl.constexpr,
    TMA_STORE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_STAGES1: tl.constexpr,
    SHARED_KV: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    IS_DYNAMIC: tl.constexpr,
    TRUNCATE_METHOD: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
):
    # Create TMA descriptors on device
    H_kv = H // G
    if ENABLE_TMA:
        desc_q = tl.make_tensor_descriptor(
            Q,
            shape=[total_seq_len_q, H * HEAD_DIM],
            # pyrefly: ignore [bad-argument-type]
            strides=[H * HEAD_DIM, 1],
            block_shape=[BLOCK_M, HEAD_DIM],
        )
        desc_k = tl.make_tensor_descriptor(
            K,
            shape=[total_seq_len_kv, H_kv * HEAD_DIM],
            # pyrefly: ignore [bad-argument-type]
            strides=[H_kv * HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
        desc_v = tl.make_tensor_descriptor(
            V,
            shape=[total_seq_len_kv, H_kv * BLOCK_D_V],
            # pyrefly: ignore [bad-argument-type]
            strides=[H_kv * BLOCK_D_V, 1],
            block_shape=[BLOCK_N, BLOCK_D_V],
        )
        desc_q1 = tl.make_tensor_descriptor(
            Q,
            shape=[total_seq_len_q, H * HEAD_DIM],
            # pyrefly: ignore [bad-argument-type]
            strides=[H * HEAD_DIM, 1],
            block_shape=[BLOCK_M1, HEAD_DIM],
        )
        desc_k1 = tl.make_tensor_descriptor(
            K,
            shape=[total_seq_len_kv, H_kv * HEAD_DIM],
            # pyrefly: ignore [bad-argument-type]
            strides=[H_kv * HEAD_DIM, 1],
            block_shape=[BLOCK_N1, HEAD_DIM],
        )
        desc_v1 = tl.make_tensor_descriptor(
            V,
            shape=[total_seq_len_kv, H_kv * BLOCK_D_V],
            # pyrefly: ignore [bad-argument-type]
            strides=[H_kv * BLOCK_D_V, 1],
            block_shape=[BLOCK_N1, BLOCK_D_V],
        )
    else:
        desc_q = None
        desc_k = None
        desc_v = None
        desc_q1 = None
        desc_k1 = None
        desc_v1 = None

    (
        start_m,
        off_h,
        off_h_kv,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    ) = _compute_offsets(
        H, G, BLOCK_M, seq_offsets_q, seq_offsets, max_seq_len, TRUNCATE_METHOD
    )
    if HAS_CAUSAL:
        n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
        uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    else:
        uih_len_q = seq_len_q
    # DO NOT merge IS_DYNAMIC into a compound boolean expression
    # (e.g. "if IS_DYNAMIC and condition:"). The Triton CC compiler
    # passes constexprs as i32 and cannot bitcast i32 to i1 for the
    # AND operator. Keeping IS_DYNAMIC as a standalone constexpr
    # branch allows dead-code elimination without the bitcast.
    if IS_DYNAMIC:
        if start_m + BLOCK_M1 >= seq_len_q:
            # Use smaller descriptors (BLOCK_M1, BLOCK_N1) for tail handling
            _attn_fwd_triton_inner(
                alpha,
                Z,
                H,
                desc_q1,
                desc_k1,
                desc_v1,
                Q,
                K,
                V,
                Out,
                seq_offsets_q,
                seq_offsets,
                max_seq_len,
                attn_scale,
                stride_qm,
                stride_qh,
                stride_kn,
                stride_kh,
                stride_vn,
                stride_vh,
                stride_om,
                stride_oh,
                start_m,
                off_h,
                off_h_kv,
                seq_start_kv,
                seq_len_kv,
                seq_start_q,
                seq_len_q,
                M,
                stride_mm,
                uih_len_q,
                num_softmax_heads,
                HEAD_DIM,
                BLOCK_D_V,
                BLOCK_M1,
                BLOCK_N1,
                TRANS,
                TMA_STORE,
                NUM_STAGES1,
                SHARED_KV=SHARED_KV,
                ENABLE_TMA=ENABLE_TMA,
                HAS_CAUSAL=HAS_CAUSAL,
            )
            return
    _attn_fwd_triton_inner(
        alpha,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        Q,
        K,
        V,
        Out,
        seq_offsets_q,
        seq_offsets,
        max_seq_len,
        attn_scale,
        stride_qm,
        stride_qh,
        stride_kn,
        stride_kh,
        stride_vn,
        stride_vh,
        stride_om,
        stride_oh,
        start_m,
        off_h,
        off_h_kv,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        M,
        stride_mm,
        uih_len_q,
        num_softmax_heads,
        HEAD_DIM,
        BLOCK_D_V,
        BLOCK_M,
        BLOCK_N,
        TRANS,
        TMA_STORE,
        NUM_STAGES,
        SHARED_KV=SHARED_KV,
        ENABLE_TMA=ENABLE_TMA,
        HAS_CAUSAL=HAS_CAUSAL,
    )


def _bwd_pre_hook_v3(nargs):
    """Pre-hook for v3 TMA API backward kernel.

    Zeros output gradient tensors (DK, DV) before each kernel launch.
    """
    if "DK" not in nargs:
        return
    if nargs.get("PER_KV_HEAD", False) and nargs["TRUNCATE_METHOD"] == "none":
        return
    nargs["DK"].zero_()
    if not nargs["SHARED_KV"]:
        nargs["DV"].zero_()


def _get_triton_bw_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                "BLOCK_M": M,
                "BLOCK_N": N,
                "BLOCK_M1": M1,
                "BLOCK_N1": N1,
            },
            num_stages=ns,
            num_warps=nw,
            pre_hook=_bwd_pre_hook_v3,
        )
        for (M, N, M1, N1) in [
            (128, 64, 32, 128),
            (128, 64, 64, 64),
            (64, 128, 32, 128),
            (64, 64, 32, 64),
            (64, 64, 32, 128),
            (32, 64, 32, 64),
            (32, 128, 32, 128),
        ]
        for ns in [2, 3]
        for nw in [4, 8]
    ]
    return configs


@triton.jit
def _compute_bwd_offsets(
    H,
    G,
    BLOCK_M,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    TRUNCATE_METHOD: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h // G
    start_m = tl.program_id(0) * BLOCK_M
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int32)
    seq_end_kv = tl.load(seq_offsets + off_z + 1).to(tl.int32)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    # Truncate to last max_seq_len KV tokens if actual seq len exceeds max_seq_len
    if TRUNCATE_METHOD != "none":
        truncated_len = tl.minimum(seq_len_kv, max_seq_len.to(tl.int32))
        if TRUNCATE_METHOD == "keep_last":
            seq_start_kv = seq_start_kv + (seq_len_kv - truncated_len)
        seq_len_kv = truncated_len
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int32)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int32)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    return (
        off_h,
        off_h_kv,
        start_m,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    )


@triton.jit
def _compute_bwd_reduce_dq_offsets(
    H,
    G,
    BLOCK_N,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    TRUNCATE_METHOD: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h // G
    start_n = tl.program_id(1) * BLOCK_N
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int32)
    seq_end_kv = tl.load(seq_offsets + off_z + 1).to(tl.int32)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    # Truncate to last max_seq_len KV tokens if actual seq len exceeds max_seq_len
    if TRUNCATE_METHOD != "none":
        truncated_len = tl.minimum(seq_len_kv, max_seq_len.to(tl.int32))
        if TRUNCATE_METHOD == "keep_last":
            seq_start_kv = seq_start_kv + (seq_len_kv - truncated_len)
        seq_len_kv = truncated_len
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int32)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int32)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    return (
        off_h,
        off_h_kv,
        start_n,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    )


def _bwd_pre_hook_redq_v3(nargs):
    """
    In the redq variant, the kernel parallelizes over K/V blocks:
    - DQ: Multiple K/V blocks accumulate into the same Q locations via atomic_add (needs zeroing)
    - DK, DV: Each kernel instance writes to distinct K/V locations via tl.store (no zeroing)
    """
    if "DQ" in nargs:
        nargs["DQ"].zero_()
        if nargs["TRUNCATE_METHOD"] != "none":
            # When truncating KV, we need to zero out the entire dk/dv tensor
            nargs["DK"].zero_()
            if not nargs["SHARED_KV"]:
                nargs["DV"].zero_()
    # When G > 1 (GQA), multiple Q heads write to the same K/V head,
    # so dk/dv must be zero-initialized for atomic accumulation
    if nargs["G"] > 1 and not nargs.get("PER_KV_HEAD", False) and "DK" in nargs:
        nargs["DK"].zero_()
        if not nargs["SHARED_KV"]:
            nargs["DV"].zero_()


def _get_triton_bw_redq_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {
                "BLOCK_M": M,
                "BLOCK_N": N,
            },
            num_stages=ns,
            num_warps=nw,
            pre_hook=_bwd_pre_hook_redq_v3,
        )
        for M in [64]#, 32]
        for N in [128]#, 128]
        for ns in [2]#, 3]
        # EXPERIMENT (autoWS): force nw=4 only to test whether the PSM warp-budget
        # guard (skip WS if estimate > 16) is what suppresses warp specialization.
        for nw in [4]
        if not (N == 128 and ns == 3)  # pruned: OOM on shared memory
    ]
    return configs


@triton_autotune(
    configs=_get_triton_bw_redq_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "max_q_len",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "num_softmax_heads",
    ],
)
@triton.jit
def _hstu_attn_bwd_redq(  # noqa C901
    Q,
    K,
    V,
    DO,
    total_seq_len_q,
    total_seq_len_kv,
    seq_offsets,
    seq_offsets_q,
    DQ,
    DK,
    DV,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    max_seq_len,
    attn_scale,
    M,
    Delta,
    stride_mm,
    num_targets,
    Z,
    AUTOTUNE_Z,
    H,
    G: tl.constexpr,
    num_softmax_heads: tl.constexpr,
    max_q_len: tl.constexpr,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    SHARED_KV: tl.constexpr,
    TRUNCATE_METHOD: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    # pyrefly: ignore [bad-function-definition]
    PER_KV_HEAD: tl.constexpr = False,
    # pyrefly: ignore [bad-function-definition]
    AUTOWS: tl.constexpr = False,
    # gates warp-specialization behavior only (see _hstu_attn_bwd_inner); normally
    # == AUTOWS, but AUTOWS=True/WS_ON=False = autoWS structure with WS disabled.
    WS_ON: tl.constexpr = False,
):
    if PER_KV_HEAD:
        # Per-KV-head (atomic-free GQA) variant
        H_kv = H // G
        off_hz = tl.program_id(0)
        off_z = off_hz // H_kv
        off_h_kv = off_hz % H_kv
        seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int32)
        seq_end_kv = tl.load(seq_offsets + off_z + 1).to(tl.int32)
        seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
        if TRUNCATE_METHOD != "none":
            truncated_len = tl.minimum(seq_len_kv, max_seq_len.to(tl.int32))
            if TRUNCATE_METHOD == "keep_last":
                seq_start_kv = seq_start_kv + (seq_len_kv - truncated_len)
            seq_len_kv = truncated_len
        seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int32)
        seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int32)
        seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)
        if HAS_CAUSAL:
            n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
            uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
        else:
            uih_len_q = seq_len_q
        if seq_len_kv == 0 or seq_len_q == 0:
            return

        desc_q = tl.make_tensor_descriptor(
            Q,
            shape=[total_seq_len_q, H * DimQ],
            # pyrefly: ignore [bad-argument-type]
            strides=[H * DimQ, 1],
            block_shape=[BLOCK_M, DimQ],
        )
        desc_do = tl.make_tensor_descriptor(
            DO,
            shape=[total_seq_len_q, H * DimV],
            # pyrefly: ignore [bad-argument-type]
            strides=[H * DimV, 1],
            block_shape=[BLOCK_M, DimV],
        )
        desc_k = tl.make_tensor_descriptor(
            K,
            shape=[total_seq_len_kv, H_kv * DimQ],
            # pyrefly: ignore [bad-argument-type]
            strides=[H_kv * DimQ, 1],
            block_shape=[BLOCK_N, DimQ],
        )
        desc_v = tl.make_tensor_descriptor(
            V,
            shape=[total_seq_len_kv, H_kv * DimV],
            # pyrefly: ignore [bad-argument-type]
            strides=[H_kv * DimV, 1],
            block_shape=[BLOCK_N, DimV],
        )
        end_dq = seq_start_q + seq_len_q
        desc_dq = tl.make_tensor_descriptor(
            DQ,
            shape=[end_dq.to(tl.int32), DimQ * H],
            # pyrefly: ignore [bad-argument-type]
            strides=[DimQ * H, 1],
            block_shape=[BLOCK_M, DimQ],
        )

        offs_qk_d = tl.arange(0, DimQ)
        offs_v_d = tl.arange(0, DimV)
        tl.static_assert(ATTN_SCALE_TYPE == "scalar")
        scale = tl.load(attn_scale).to(tl.float32)

        for start_n in tl.range(0, seq_len_kv, BLOCK_N, num_stages=1):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len_kv
            k_offset = (seq_start_kv + start_n).to(tl.int32)
            k = desc_k.load([k_offset, off_h_kv * stride_kh])
            if SHARED_KV:
                v = k
            else:
                v = desc_v.load([k_offset, off_h_kv * stride_vh])
            dk = tl.zeros([BLOCK_N, DimQ], dtype=tl.float32)
            if not SHARED_KV:
                dv = tl.zeros([BLOCK_N, DimV], dtype=tl.float32)
            for g in range(G):
                off_h = off_h_kv * G + g
                if off_h < num_softmax_heads:
                    scaled_alpha = alpha * 1.44269504
                else:
                    scaled_alpha = alpha
                M_off, Delta_off = backward_common_preprocess(
                    M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm
                )
                # EXPERIMENT (autoWS): warp_specialize the inner Q loop. The
                # `warp_specialize` kwarg only exists in beta triton, so this
                # variant must be built with -m ovr_config//triton:beta. Guard
                # before landing (constexpr-branch the tl.range on AUTOWS).
                for start_m in tl.range(
                    0, seq_len_q, BLOCK_M, num_stages=1, warp_specialize=WS_ON
                ):
                    offs_m = start_m + tl.arange(0, BLOCK_M)
                    mask_m = offs_m < seq_len_q
                    valid_mask_trans = backward_valid_mask(
                        offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL
                    )
                    q_offset = (seq_start_q + start_m).to(tl.int32)
                    q = desc_q.load([q_offset, off_h * stride_qh])
                    do = desc_do.load([q_offset, off_h * stride_doh])
                    qk_trans = tl.dot(k, tl.trans(q), allow_tf32=ALLOW_TF32)
                    if off_h < num_softmax_heads:
                        qk_trans, act_qk_trans, pT = (
                            backward_softmax_activation_scaled_alpha(
                                qk_trans,
                                scaled_alpha,
                                valid_mask_trans,
                                M_off,
                                offs_m,
                                stride_mm,
                                mask_m,
                                k,
                            )
                        )
                    else:
                        qk_trans, act_qk_trans, pT = backward_silu_activation(
                            qk_trans, alpha, valid_mask_trans, k.dtype, scale
                        )
                    if SHARED_KV:
                        dk += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
                    else:
                        # pyrefly: ignore [unbound-name]
                        dv += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
                    dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
                    if off_h < num_softmax_heads:
                        dqk_trans = backward_d_softmax_activation(
                            dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT
                        )
                    else:
                        dqk_trans = backward_d_silu_activation(
                            dact_qk_trans, pT, qk_trans, scale, valid_mask_trans
                        )
                    dqk_trans = dqk_trans.to(k.dtype)
                    dk += tl.dot(dqk_trans, q, allow_tf32=ALLOW_TF32) * alpha
                    dq_trans = (
                        tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32) * alpha
                    )
                    dq = tl.trans(dq_trans)
                    desc_dq.store(
                        [q_offset, DimQ * off_h],
                        dq.to(desc_dq.dtype),
                        store_reduce="add",
                    )
            DK_off = (
                DK
                + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dkn, tl.int64)
                + off_h_kv * tl.cast(stride_dkh, tl.int64)
            )
            dk_ptrs = DK_off + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
            tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
            if not SHARED_KV:
                DV_off = (
                    DV
                    + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dvn, tl.int64)
                    + off_h_kv * tl.cast(stride_dvh, tl.int64)
                )
                dv_ptrs = DV_off + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
                # pyrefly: ignore [unbound-name]
                tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
        return
    (
        off_h,
        off_h_kv,
        start_n,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    ) = _compute_bwd_reduce_dq_offsets(
        H,
        G,
        BLOCK_N,
        seq_offsets_q,
        seq_offsets,
        max_seq_len,
        TRUNCATE_METHOD,
    )
    n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
    uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    # It is possible that uih_len is 0, need to handle this case
    # to avoid illegal instruction error using on-device tma store
    # for dk.
    if seq_len_kv == 0:
        return
    # Create TMA descriptors on device
    H_kv = H // G
    desc_q = tl.make_tensor_descriptor(
        Q,
        shape=[total_seq_len_q, H * DimQ],
        # pyrefly: ignore [bad-argument-type]
        strides=[H * DimQ, 1],
        block_shape=[BLOCK_M, DimQ],
    )
    desc_k = tl.make_tensor_descriptor(
        K,
        shape=[total_seq_len_kv, H_kv * DimQ],
        # pyrefly: ignore [bad-argument-type]
        strides=[H_kv * DimQ, 1],
        block_shape=[BLOCK_N, DimQ],
    )
    desc_v = tl.make_tensor_descriptor(
        V,
        shape=[total_seq_len_kv, H_kv * DimV],
        # pyrefly: ignore [bad-argument-type]
        strides=[H_kv * DimV, 1],
        block_shape=[BLOCK_N, DimV],
    )
    desc_do = tl.make_tensor_descriptor(
        DO,
        shape=[total_seq_len_q, H * DimV],
        # pyrefly: ignore [bad-argument-type]
        strides=[H * DimV, 1],
        block_shape=[BLOCK_M, DimV],
    )
    end_kv = seq_start_kv + seq_len_kv
    desc_dk = tl.make_tensor_descriptor(
        DK,
        shape=[end_kv.to(tl.int32), DimQ * H_kv],
        # pyrefly: ignore [bad-argument-type]
        strides=[DimQ * H_kv, 1],
        block_shape=[BLOCK_N, DimQ],
    )
    end_dq = seq_start_q + seq_len_q
    desc_dq = tl.make_tensor_descriptor(
        DQ,
        shape=[end_dq.to(tl.int32), DimQ * H],
        # pyrefly: ignore [bad-argument-type]
        strides=[DimQ * H, 1],
        block_shape=[BLOCK_M, DimQ],
    )
    last_block_start_n = tl.maximum(0, (tl.cdiv(seq_len_kv, BLOCK_N) - 1) * BLOCK_N)
    if AUTOWS:
        # AutoWS requires a SINGLE inner-loop structure. The peeled main-loop +
        # separate masked-last-block call below expands (via inlining) into TWO
        # inner Q-loops, so the memory planner merges the two loops' per-iteration
        # TMEM slots into one cross-loop reuse group that the reuse-barrier pass
        # cannot order ("TMEM reuse group has no unique dependency-chain order").
        # Match the TLX kernel (attn_bwd_ws): iterate every KV block in one masked
        # loop — masking is a no-op for full blocks. Kept autoWS-only (constexpr)
        # so the non-WS peeled fast path is unchanged.
        for start_n in tl.range(
            0,
            seq_len_kv,
            BLOCK_N,
            warp_specialize=WS_ON,
            merge_epilogue_to_computation=WS_ON,
            # With the compute fold, the dk accumulator's cross-iteration
            # tmem_load is categorized as a correction op; merge it into the
            # computation partition instead of spawning a separate correction
            # partition (which would exceed the 16-warp budget).
            merge_correction=WS_ON,
            data_partition_factor=(2 if BLOCK_N >= 256 else 1),
        ):
            _hstu_attn_bwd_inner(
                start_n,
                seq_start_kv,
                seq_len_kv,
                seq_start_q,
                seq_len_q,
                desc_q,
                desc_k,
                desc_v,
                desc_do,
                desc_dk,
                desc_dq,
                DV,
                stride_qh,
                stride_kh,
                stride_vh,
                stride_doh,
                stride_dvn,
                stride_dvh,
                alpha,
                attn_scale,
                M,
                Delta,
                stride_mm,
                off_h,
                off_h_kv,
                H_kv,
                uih_len_q,
                num_softmax_heads,
                DimQ,
                DimV,
                ALLOW_TF32,
                BLOCK_M,
                BLOCK_N,
                ATTN_SCALE_TYPE,
                G > 1,  # GQA_ATOMIC_ADD
                SHARED_KV,
                True,  # MASK_KV
                HAS_CAUSAL,
                AUTOWS,
                WS_ON,
            )
        return
    for start_n in tl.range(0, last_block_start_n, BLOCK_N):
        _hstu_attn_bwd_inner(
            start_n,
            seq_start_kv,
            seq_len_kv,
            seq_start_q,
            seq_len_q,
            desc_q,
            desc_k,
            desc_v,
            desc_do,
            desc_dk,
            desc_dq,
            DV,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_doh,
            stride_dvn,
            stride_dvh,
            alpha,
            attn_scale,
            M,
            Delta,
            stride_mm,
            off_h,
            off_h_kv,
            H_kv,
            uih_len_q,
            num_softmax_heads,
            DimQ,
            DimV,
            ALLOW_TF32,
            BLOCK_M,
            BLOCK_N,
            ATTN_SCALE_TYPE,
            GQA_ATOMIC_ADD=G > 1,
            SHARED_KV=SHARED_KV,
            MASK_KV=False,
            HAS_CAUSAL=HAS_CAUSAL,
            AUTOWS=AUTOWS,
        )
    # Handle last block with masking
    _hstu_attn_bwd_inner(
        last_block_start_n,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        desc_q,
        desc_k,
        desc_v,
        desc_do,
        desc_dk,
        desc_dq,
        DV,
        stride_qh,
        stride_kh,
        stride_vh,
        stride_doh,
        stride_dvn,
        stride_dvh,
        alpha,
        attn_scale,
        M,
        Delta,
        stride_mm,
        off_h,
        off_h_kv,
        H_kv,
        uih_len_q,
        num_softmax_heads,
        DimQ,
        DimV,
        ALLOW_TF32,
        BLOCK_M,
        BLOCK_N,
        ATTN_SCALE_TYPE,
        GQA_ATOMIC_ADD=G > 1,
        SHARED_KV=SHARED_KV,
        MASK_KV=True,
        HAS_CAUSAL=HAS_CAUSAL,
        AUTOWS=AUTOWS,
    )


def _get_triton_bw_redq_2kv_configs() -> List[triton.Config]:
    # The manual 2-KV-block loop supplies the 2-way parallelism, so BLOCK_N stays
    # well below the compiler-DP trigger (BLOCK_N >= 256) and the pass never runs.
    #
    # List-schedule autotuning: when TRITON_USE_LIST_SCHEDULE=1, sweep the inner
    # (start_m) loop's schedule rank INNER_PICK over 0..K-1 (K from
    # HSTU_INNER_SCHED_K, default 4). Each INNER_PICK is a distinct tl.constexpr
    # -> a distinct compile key, so the autotuner compiles+times each inner
    # schedule and caches the winner. Deliberately NOT keyed on
    # TRITON_LIST_SCHEDULE_TOPK: that env is the pass's *global* generation count
    # and would also make the OUTER loop generate variants. We leave it unset so
    # the outer loop keeps a single schedule (only the inner, which carries the
    # per-loop pick attr, is generated ranked). With list scheduling off,
    # INNER_PICK stays [0] (no extra configs; the attr is ignored downstream).
    if os.environ.get("TRITON_USE_LIST_SCHEDULE"):
        inner_k = int(os.environ.get("HSTU_INNER_SCHED_K", "4"))
        inner_picks = list(range(max(1, inner_k)))
    else:
        inner_picks = [0]
    configs = [
        triton.Config(
            {
                "BLOCK_M": M,
                "BLOCK_N": N,
                "INNER_PICK": pick,
            },
            num_stages=ns,
            num_warps=nw,
            pre_hook=_bwd_pre_hook_redq_v3,
        )
        # BLOCK_M=64, BLOCK_N=64. The compute fold MMA-accumulates BOTH KV blocks'
        # dk in TMEM (each [BLOCK_N, DimQ]); at BLOCK_N=64 the two dk tiles stack
        # in TMEM's two 64-row groups at the SAME columns (128 cols for both, not
        # 256), which -- with dq as a single shared accumulator -- keeps the peak
        # at ~480 < 512 cols and lets BLOCK_M reach 64. BLOCK_N=128 would need 256
        # dk cols and overflows at BLOCK_M=64 (608 > 512).
        for M in [64]
        for N in [64]
        for ns in [1]
        for nw in [4]
        for pick in inner_picks
    ]
    return configs


@triton_autotune(
    configs=_get_triton_bw_redq_2kv_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "max_q_len",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "num_softmax_heads",
    ],
)
@triton.jit
def _hstu_attn_bwd_redq_2kv(  # noqa C901
    Q,
    K,
    V,
    DO,
    total_seq_len_q,
    total_seq_len_kv,
    seq_offsets,
    seq_offsets_q,
    DQ,
    DK,
    DV,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    max_seq_len,
    attn_scale,
    M,
    Delta,
    stride_mm,
    num_targets,
    Z,
    AUTOTUNE_Z,
    H,
    G: tl.constexpr,
    num_softmax_heads: tl.constexpr,
    max_q_len: tl.constexpr,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    SHARED_KV: tl.constexpr,
    TRUNCATE_METHOD: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    # pyrefly: ignore [bad-function-definition]
    PER_KV_HEAD: tl.constexpr = False,
    # pyrefly: ignore [bad-function-definition]
    AUTOWS: tl.constexpr = False,
    WS_ON: tl.constexpr = False,
    # Inner (start_m) loop list-schedule rank; swept by autotune under
    # TRITON_USE_LIST_SCHEDULE=1. See _get_triton_bw_redq_2kv_configs.
    # pyrefly: ignore [bad-function-definition]
    INNER_PICK: tl.constexpr = 0,
):
    """MANUAL 2-KV-block data-partition fork of `_hstu_attn_bwd_redq`.

    Steps the KV loop by 2*BLOCK_N and dispatches `_hstu_attn_bwd_inner_2kv`
    (two KV blocks per call), rounding the block count up so an odd leftover
    block is handled by the same 2-KV inner with its second block fully masked
    (no separate single-KV tail, whose buffer.ids would clash under WS).
    SHARED_KV + compute-fold only.
    """
    tl.static_assert(SHARED_KV, "_hstu_attn_bwd_redq_2kv is SHARED_KV only")
    tl.static_assert(not PER_KV_HEAD, "_hstu_attn_bwd_redq_2kv requires G == 1")
    (
        off_h,
        off_h_kv,
        start_n,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    ) = _compute_bwd_reduce_dq_offsets(
        H,
        G,
        BLOCK_N,
        seq_offsets_q,
        seq_offsets,
        max_seq_len,
        TRUNCATE_METHOD,
    )
    n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
    uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    if seq_len_kv == 0:
        return
    H_kv = H // G
    desc_q = tl.make_tensor_descriptor(
        Q,
        shape=[total_seq_len_q, H * DimQ],
        # pyrefly: ignore [bad-argument-type]
        strides=[H * DimQ, 1],
        block_shape=[BLOCK_M, DimQ],
    )
    desc_k = tl.make_tensor_descriptor(
        K,
        shape=[total_seq_len_kv, H_kv * DimQ],
        # pyrefly: ignore [bad-argument-type]
        strides=[H_kv * DimQ, 1],
        block_shape=[BLOCK_N, DimQ],
    )
    desc_v = tl.make_tensor_descriptor(
        V,
        shape=[total_seq_len_kv, H_kv * DimV],
        # pyrefly: ignore [bad-argument-type]
        strides=[H_kv * DimV, 1],
        block_shape=[BLOCK_N, DimV],
    )
    desc_do = tl.make_tensor_descriptor(
        DO,
        shape=[total_seq_len_q, H * DimV],
        # pyrefly: ignore [bad-argument-type]
        strides=[H * DimV, 1],
        block_shape=[BLOCK_M, DimV],
    )
    end_kv = seq_start_kv + seq_len_kv
    desc_dk = tl.make_tensor_descriptor(
        DK,
        shape=[end_kv.to(tl.int32), DimQ * H_kv],
        # pyrefly: ignore [bad-argument-type]
        strides=[DimQ * H_kv, 1],
        block_shape=[BLOCK_N, DimQ],
    )
    end_dq = seq_start_q + seq_len_q
    desc_dq = tl.make_tensor_descriptor(
        DQ,
        shape=[end_dq.to(tl.int32), DimQ * H],
        # pyrefly: ignore [bad-argument-type]
        strides=[DimQ * H, 1],
        block_shape=[BLOCK_M, DimQ],
    )

    # Pair up BLOCK_N KV blocks: process two blocks per _hstu_attn_bwd_inner_2kv
    # call, rounding the block count UP so an odd leftover block is handled by
    # the SAME 2-KV inner with its second block (b1) fully masked -- b1's dq
    # contribution is zero and its dk1 store lands past end_kv (dropped by the
    # desc_dk TMA bounds). This avoids a separate single-KV tail inner, whose
    # distinct TLX buffer.ids collide with the 2-KV ids and crash the WS pass.
    n_blocks = tl.cdiv(seq_len_kv, BLOCK_N)
    n_pairs = tl.cdiv(n_blocks, 2)
    pair_end_n = n_pairs * 2 * BLOCK_N
    for start_n in tl.range(0, pair_end_n, 2 * BLOCK_N):
        _hstu_attn_bwd_inner_2kv(
            start_n,
            seq_start_kv,
            seq_len_kv,
            seq_start_q,
            seq_len_q,
            desc_q,
            desc_k,
            desc_v,
            desc_do,
            desc_dk,
            desc_dq,
            DV,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_doh,
            stride_dvn,
            stride_dvh,
            alpha,
            attn_scale,
            M,
            Delta,
            stride_mm,
            off_h,
            off_h_kv,
            H_kv,
            uih_len_q,
            num_softmax_heads,
            DimQ,
            DimV,
            ALLOW_TF32,
            BLOCK_M,
            BLOCK_N,
            ATTN_SCALE_TYPE,
            G > 1,  # GQA_ATOMIC_ADD
            SHARED_KV,
            True,  # MASK_KV
            HAS_CAUSAL,
            AUTOWS,
            WS_ON,
            INNER_PICK,
        )


@triton_autotune(
    configs=_get_triton_bw_redq_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_Q_LEN",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "num_softmax_heads",
    ],
)
@triton.jit
def _hstu_attn_bwd(  # noqa C901
    Q,
    K,
    V,
    DO,
    total_seq_len_q,
    total_seq_len_kv,
    seq_offsets,
    seq_offsets_q,
    DQ,
    DK,
    DV,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    max_seq_len,
    attn_scale,
    M,
    Delta,
    stride_mm,
    num_targets,
    Z,
    AUTOTUNE_Z,
    H,
    G,
    num_softmax_heads: tl.constexpr,
    AUTOTUNE_MAX_Q_LEN,  # Quantized MAX_Q_LEN used as an autotuning key
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    SHARED_KV: tl.constexpr,
    TRUNCATE_METHOD: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
):
    (
        off_h,
        off_h_kv,
        start_n,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    ) = _compute_bwd_reduce_dq_offsets(
        H,
        G,
        BLOCK_N,
        seq_offsets_q,
        seq_offsets,
        max_seq_len,
        TRUNCATE_METHOD,
    )
    if HAS_CAUSAL:
        n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
        uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    else:
        uih_len_q = seq_len_q
    if seq_len_kv == 0:
        return
    # Create TMA descriptors on device
    H_kv = H // G
    desc_q = tl.make_tensor_descriptor(
        Q,
        shape=[total_seq_len_q, H * DimQ],
        # pyrefly: ignore [bad-argument-type]
        strides=[H * DimQ, 1],
        block_shape=[BLOCK_M, DimQ],
    )
    desc_k = tl.make_tensor_descriptor(
        K,
        shape=[total_seq_len_kv, H_kv * DimQ],
        # pyrefly: ignore [bad-argument-type]
        strides=[H_kv * DimQ, 1],
        block_shape=[BLOCK_N, DimQ],
    )
    desc_v = tl.make_tensor_descriptor(
        V,
        shape=[total_seq_len_kv, H_kv * DimV],
        # pyrefly: ignore [bad-argument-type]
        strides=[H_kv * DimV, 1],
        block_shape=[BLOCK_N, DimV],
    )
    desc_do = tl.make_tensor_descriptor(
        DO,
        shape=[total_seq_len_q, H * DimV],
        # pyrefly: ignore [bad-argument-type]
        strides=[H * DimV, 1],
        block_shape=[BLOCK_M, DimV],
    )
    end_kv = seq_start_kv + seq_len_kv
    desc_dk = tl.make_tensor_descriptor(
        DK,
        shape=[end_kv.to(tl.int32), DimQ * H_kv],
        # pyrefly: ignore [bad-argument-type]
        strides=[DimQ * H_kv, 1],
        block_shape=[BLOCK_N, DimQ],
    )
    end_dq = seq_start_q + seq_len_q
    desc_dq = tl.make_tensor_descriptor(
        DQ,
        shape=[end_dq.to(tl.int32), DimQ * H],
        # pyrefly: ignore [bad-argument-type]
        strides=[DimQ * H, 1],
        block_shape=[BLOCK_M, DimQ],
    )
    last_block_start_n = tl.maximum(0, (tl.cdiv(seq_len_kv, BLOCK_N) - 1) * BLOCK_N)
    short_q = seq_len_q <= BLOCK_M
    if short_q:
        q_offset = seq_start_q
        offs_m = tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len_q
        tl.static_assert(ATTN_SCALE_TYPE == "scalar")
        scale = tl.load(attn_scale).to(tl.float32)
        M_off, Delta_off = backward_common_preprocess(
            M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm
        )
        DK_off = (
            DK
            + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dkn, tl.int64)
            + off_h_kv * tl.cast(stride_dkh, tl.int64)
        )
        DV_off = DV
        if not SHARED_KV:
            DV_off = (
                DV
                + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dvn, tl.int64)
                + off_h_kv * tl.cast(stride_dvh, tl.int64)
            )
        dq_acc = tl.zeros([DimQ, BLOCK_M], dtype=tl.float32)
        if off_h < num_softmax_heads:
            scaled_alpha = alpha * 1.44269504
        else:
            scaled_alpha = alpha
        q = desc_q.load([q_offset.to(tl.int32), off_h * stride_qh])
        do = desc_do.load([q_offset.to(tl.int32), off_h * stride_doh])
        q_trans = tl.trans(q)
        for start_n in tl.range(0, seq_len_kv, BLOCK_N):
            k_offset = (seq_start_kv + start_n).to(tl.int32)
            k = desc_k.load([k_offset, off_h_kv * stride_kh])
            if SHARED_KV:
                v = k
            else:
                v = desc_v.load([k_offset, off_h_kv * stride_vh])
            qk_trans = tl.dot(k, q_trans)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_mask_trans = (offs_n[:, None] < seq_len_kv) & (
                offs_m[None, :] < seq_len_q
            )
            if off_h < num_softmax_heads:
                qk_trans, act_qk_trans, pT = backward_softmax_activation_scaled_alpha(
                    qk_trans,
                    scaled_alpha,
                    valid_mask_trans,
                    M_off,
                    offs_m,
                    stride_mm,
                    mask_m,
                    k,
                )
            else:
                qk_trans, act_qk_trans, pT = backward_silu_activation(
                    qk_trans, alpha, valid_mask_trans, k.dtype, scale
                )
            if SHARED_KV:
                dk = tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
            else:
                dv = tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
            dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
            if off_h < num_softmax_heads:
                dqk_trans = backward_d_softmax_activation(
                    dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT
                )
            else:
                dqk_trans = backward_d_silu_activation(
                    dact_qk_trans, pT, qk_trans, scale, valid_mask_trans
                )
            dqk_trans = dqk_trans.to(k.dtype)
            if SHARED_KV:
                dk_attn = tl.dot(dqk_trans, tl.trans(q_trans), allow_tf32=ALLOW_TF32)
                # pyrefly: ignore [unbound-name]
                dk = dk + dk_attn * alpha
            else:
                dk = tl.dot(dqk_trans, tl.trans(q_trans), allow_tf32=ALLOW_TF32) * alpha
            offs_qk_d = tl.arange(0, DimQ)
            dk_ptrs = DK_off + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
            mask_n = offs_n < seq_len_kv
            if G > 1:
                tl.atomic_add(
                    dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None], sem="relaxed"
                )
            else:
                tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
            if not SHARED_KV:
                offs_v_d = tl.arange(0, DimV)
                dv_ptrs = DV_off + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
                if G > 1:
                    tl.atomic_add(
                        # pyrefly: ignore [unbound-name]
                        dv_ptrs,
                        # pyrefly: ignore [unbound-name]
                        dv.to(k.dtype),
                        mask=mask_n[:, None],
                        sem="relaxed",
                    )
                else:
                    # pyrefly: ignore [unbound-name]
                    tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
            dq_trans = tl.dot(tl.trans(k), dqk_trans)
            dq_trans = dq_trans * alpha
            dq_acc += dq_trans
        dq = tl.trans(dq_acc)
        dq_ptrs = (
            DQ
            + (seq_start_q + offs_m[:, None]) * stride_dqm
            + off_h * stride_dqh
            + tl.arange(0, DimQ)[None, :]
        )
        tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])
    else:
        for start_n in tl.range(0, last_block_start_n, BLOCK_N):
            _hstu_attn_bwd_inner(
                start_n,
                seq_start_kv,
                seq_len_kv,
                seq_start_q,
                seq_len_q,
                desc_q,
                desc_k,
                desc_v,
                desc_do,
                desc_dk,
                desc_dq,
                DV,
                stride_qh,
                stride_kh,
                stride_vh,
                stride_doh,
                stride_dvn,
                stride_dvh,
                alpha,
                attn_scale,
                M,
                Delta,
                stride_mm,
                off_h,
                off_h_kv,
                H_kv,
                uih_len_q,
                num_softmax_heads,
                DimQ,
                DimV,
                ALLOW_TF32,
                BLOCK_M,
                BLOCK_N,
                ATTN_SCALE_TYPE,
                GQA_ATOMIC_ADD=G > 1,
                SHARED_KV=SHARED_KV,
                MASK_KV=False,
                HAS_CAUSAL=HAS_CAUSAL,
            )
        _hstu_attn_bwd_inner(
            last_block_start_n,
            seq_start_kv,
            seq_len_kv,
            seq_start_q,
            seq_len_q,
            desc_q,
            desc_k,
            desc_v,
            desc_do,
            desc_dk,
            desc_dq,
            DV,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_doh,
            stride_dvn,
            stride_dvh,
            alpha,
            attn_scale,
            M,
            Delta,
            stride_mm,
            off_h,
            off_h_kv,
            H_kv,
            uih_len_q,
            num_softmax_heads,
            DimQ,
            DimV,
            ALLOW_TF32,
            BLOCK_M,
            BLOCK_N,
            ATTN_SCALE_TYPE,
            GQA_ATOMIC_ADD=G > 1,
            SHARED_KV=SHARED_KV,
            MASK_KV=True,
            HAS_CAUSAL=HAS_CAUSAL,
        )


@triton_autotune(
    configs=_get_triton_bw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "max_q_len",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "num_softmax_heads",
        "PER_KV_HEAD",
    ],
)
@triton.jit
def _hstu_attn_bwd_redkv(
    Q,
    K,
    V,
    DO,
    total_seq_len_q,
    total_seq_len_kv,
    seq_offsets,
    seq_offsets_q,
    DQ,
    DK,
    DV,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    max_seq_len,
    attn_scale,
    M,
    Delta,
    stride_mm,
    num_targets,
    Z,
    AUTOTUNE_Z,
    H,
    G: tl.constexpr,
    num_softmax_heads: tl.constexpr,
    max_q_len: tl.constexpr,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    SHARED_KV: tl.constexpr,
    TRUNCATE_METHOD: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    # pyrefly: ignore [bad-function-definition]
    PER_KV_HEAD: tl.constexpr = False,
):
    # Per-KV-head atomic-free GQA mode
    if PER_KV_HEAD:
        H_kv = H // G
        off_hz = tl.program_id(0)
        off_z = off_hz // H_kv
        off_h_kv = off_hz % H_kv
        seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int32)
        seq_end_kv = tl.load(seq_offsets + off_z + 1).to(tl.int32)
        seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
        if TRUNCATE_METHOD != "none":
            truncated_len = tl.minimum(seq_len_kv, max_seq_len.to(tl.int32))
            if TRUNCATE_METHOD == "keep_last":
                seq_start_kv = seq_start_kv + (seq_len_kv - truncated_len)
            seq_len_kv = truncated_len
        seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int32)
        seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int32)
        seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)
        if HAS_CAUSAL:
            n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
            uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
        else:
            uih_len_q = seq_len_q
        if seq_len_q == 0:
            return

        desc_q = tl.make_tensor_descriptor(
            Q,
            shape=[total_seq_len_q, H * DimQ],
            # pyrefly: ignore [bad-argument-type]
            strides=[H * DimQ, 1],
            block_shape=[BLOCK_M, DimQ],
        )
        desc_do = tl.make_tensor_descriptor(
            DO,
            shape=[total_seq_len_q, H * DimV],
            # pyrefly: ignore [bad-argument-type]
            strides=[H * DimV, 1],
            block_shape=[BLOCK_M, DimV],
        )
        desc_k = tl.make_tensor_descriptor(
            K,
            shape=[total_seq_len_kv, H_kv * DimQ],
            # pyrefly: ignore [bad-argument-type]
            strides=[H_kv * DimQ, 1],
            block_shape=[BLOCK_N, DimQ],
        )
        desc_v = tl.make_tensor_descriptor(
            V,
            shape=[total_seq_len_kv, H_kv * DimV],
            # pyrefly: ignore [bad-argument-type]
            strides=[H_kv * DimV, 1],
            block_shape=[BLOCK_N, DimV],
        )

        offs_qk_d = tl.arange(0, DimQ)
        offs_v_d = tl.arange(0, DimV)
        tl.static_assert(ATTN_SCALE_TYPE == "scalar")
        scale = tl.load(attn_scale).to(tl.float32)

        if max_q_len <= BLOCK_M and G * BLOCK_M * DimQ <= 32768:
            # One KV sweep: dk/dv folded over G, dq carried per head
            offs_m = tl.arange(0, BLOCK_M)
            mask_m = offs_m < seq_len_q
            offs_g = tl.arange(0, G)
            dq_acc = tl.zeros([DimQ, G, BLOCK_M], dtype=tl.float32)
            for start_n in tl.range(0, seq_len_kv, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < seq_len_kv
                valid_mask_trans = backward_valid_mask(
                    offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL
                )
                k_offset = (seq_start_kv + start_n).to(tl.int32)
                k = desc_k.load([k_offset, off_h_kv * stride_kh])
                if SHARED_KV:
                    v = k
                else:
                    v = desc_v.load([k_offset, off_h_kv * stride_vh])
                dk = tl.zeros([BLOCK_N, DimQ], dtype=tl.float32)
                dv = tl.zeros([BLOCK_N, DimV], dtype=tl.float32)
                for g in range(G):
                    off_h = off_h_kv * G + g
                    if off_h < num_softmax_heads:
                        scaled_alpha = alpha * 1.44269504
                    else:
                        scaled_alpha = alpha
                    M_off, Delta_off = backward_common_preprocess(
                        M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm
                    )
                    q = desc_q.load([seq_start_q, off_h * stride_qh])
                    do = desc_do.load([seq_start_q, off_h * stride_doh])
                    qk_trans = tl.dot(k, tl.trans(q), allow_tf32=ALLOW_TF32)
                    if off_h < num_softmax_heads:
                        qk_trans, act_qk_trans, pT = (
                            backward_softmax_activation_scaled_alpha(
                                qk_trans,
                                scaled_alpha,
                                valid_mask_trans,
                                M_off,
                                offs_m,
                                stride_mm,
                                mask_m,
                                k,
                            )
                        )
                    else:
                        qk_trans, act_qk_trans, pT = backward_silu_activation(
                            qk_trans, alpha, valid_mask_trans, k.dtype, scale
                        )
                    if SHARED_KV:
                        dk += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
                    else:
                        dv += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
                    dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
                    if off_h < num_softmax_heads:
                        dqk_trans = backward_d_softmax_activation(
                            dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT
                        )
                    else:
                        dqk_trans = backward_d_silu_activation(
                            dact_qk_trans, pT, qk_trans, scale, valid_mask_trans
                        )
                    dqk_trans = dqk_trans.to(k.dtype)
                    dk += tl.dot(dqk_trans, q, allow_tf32=ALLOW_TF32) * alpha
                    # dq for head g
                    dq_g = tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32)
                    dq_acc += tl.where(
                        (offs_g == g)[None, :, None], dq_g[:, None, :], 0.0
                    )
                DK_off = (
                    DK
                    + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dkn, tl.int64)
                    + off_h_kv * tl.cast(stride_dkh, tl.int64)
                )
                dk_ptrs = DK_off + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
                tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
                if not SHARED_KV:
                    DV_off = (
                        DV
                        + tl.cast(seq_start_kv, tl.int64)
                        * tl.cast(stride_dvn, tl.int64)
                        + off_h_kv * tl.cast(stride_dvh, tl.int64)
                    )
                    dv_ptrs = DV_off + (
                        offs_n[:, None] * stride_dvn + offs_v_d[None, :]
                    )
                    tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
            # Write dq per head from the accumulator.
            for g in range(G):
                off_h = off_h_kv * G + g
                dq_trans = tl.sum(
                    tl.where((offs_g == g)[None, :, None], dq_acc, 0.0), axis=1
                )
                dq = tl.trans(dq_trans) * alpha
                dq_ptrs = (
                    DQ
                    + (seq_start_q + offs_m[:, None]) * stride_dqm
                    + off_h * stride_dqh
                    + offs_qk_d[None, :]
                )
                tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])
            return

        # Phase 1: dk/dv (fold the G heads, tile over Q).
        for start_n in tl.range(0, seq_len_kv, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len_kv
            k_offset = (seq_start_kv + start_n).to(tl.int32)
            k = desc_k.load([k_offset, off_h_kv * stride_kh])
            if SHARED_KV:
                v = k
            else:
                v = desc_v.load([k_offset, off_h_kv * stride_vh])
            dk = tl.zeros([BLOCK_N, DimQ], dtype=tl.float32)
            dv = tl.zeros([BLOCK_N, DimV], dtype=tl.float32)
            for g in range(G):
                off_h = off_h_kv * G + g
                if off_h < num_softmax_heads:
                    scaled_alpha = alpha * 1.44269504
                else:
                    scaled_alpha = alpha
                M_off, Delta_off = backward_common_preprocess(
                    M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm
                )
                for start_m in tl.range(0, seq_len_q, BLOCK_M):
                    offs_m = start_m + tl.arange(0, BLOCK_M)
                    mask_m = offs_m < seq_len_q
                    valid_mask_trans = backward_valid_mask(
                        offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL
                    )
                    q_offset = (seq_start_q + start_m).to(tl.int32)
                    q = desc_q.load([q_offset, off_h * stride_qh])
                    do = desc_do.load([q_offset, off_h * stride_doh])
                    qk_trans = tl.dot(k, tl.trans(q), allow_tf32=ALLOW_TF32)
                    if off_h < num_softmax_heads:
                        qk_trans, act_qk_trans, pT = (
                            backward_softmax_activation_scaled_alpha(
                                qk_trans,
                                scaled_alpha,
                                valid_mask_trans,
                                M_off,
                                offs_m,
                                stride_mm,
                                mask_m,
                                k,
                            )
                        )
                    else:
                        qk_trans, act_qk_trans, pT = backward_silu_activation(
                            qk_trans, alpha, valid_mask_trans, k.dtype, scale
                        )
                    if SHARED_KV:
                        dk += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
                    else:
                        dv += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
                    dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
                    if off_h < num_softmax_heads:
                        dqk_trans = backward_d_softmax_activation(
                            dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT
                        )
                    else:
                        dqk_trans = backward_d_silu_activation(
                            dact_qk_trans, pT, qk_trans, scale, valid_mask_trans
                        )
                    dqk_trans = dqk_trans.to(k.dtype)
                    dk += tl.dot(dqk_trans, q, allow_tf32=ALLOW_TF32) * alpha
            DK_off = (
                DK
                + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dkn, tl.int64)
                + off_h_kv * tl.cast(stride_dkh, tl.int64)
            )
            dk_ptrs = DK_off + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
            tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
            if not SHARED_KV:
                DV_off = (
                    DV
                    + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dvn, tl.int64)
                    + off_h_kv * tl.cast(stride_dvh, tl.int64)
                )
                dv_ptrs = DV_off + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
                tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
        # Phase 2: dq (per head, tile over Q).
        for g in range(G):
            off_h = off_h_kv * G + g
            if off_h < num_softmax_heads:
                scaled_alpha = alpha * 1.44269504
            else:
                scaled_alpha = alpha
            M_off, Delta_off = backward_common_preprocess(
                M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm
            )
            for start_m in tl.range(0, seq_len_q, BLOCK_M):
                offs_m = start_m + tl.arange(0, BLOCK_M)
                mask_m = offs_m < seq_len_q
                q_offset = (seq_start_q + start_m).to(tl.int32)
                q = desc_q.load([q_offset, off_h * stride_qh])
                do = desc_do.load([q_offset, off_h * stride_doh])
                q_trans = tl.trans(q)
                dq_trans = tl.zeros([DimQ, BLOCK_M], dtype=tl.float32)
                for start_n in tl.range(0, seq_len_kv, BLOCK_N):
                    offs_n = start_n + tl.arange(0, BLOCK_N)
                    valid_mask_trans = backward_valid_mask(
                        offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL
                    )
                    k_offset = (seq_start_kv + start_n).to(tl.int32)
                    k = desc_k.load([k_offset, off_h_kv * stride_kh])
                    if SHARED_KV:
                        v = k
                    else:
                        v = desc_v.load([k_offset, off_h_kv * stride_vh])
                    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32)
                    if off_h < num_softmax_heads:
                        qk_trans, act_qk_trans, pT = (
                            backward_softmax_activation_scaled_alpha(
                                qk_trans,
                                scaled_alpha,
                                valid_mask_trans,
                                M_off,
                                offs_m,
                                stride_mm,
                                mask_m,
                                k,
                            )
                        )
                    else:
                        qk_trans, act_qk_trans, pT = backward_silu_activation(
                            qk_trans, alpha, valid_mask_trans, k.dtype, scale
                        )
                    dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
                    if off_h < num_softmax_heads:
                        dqk_trans = backward_d_softmax_activation(
                            dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT
                        )
                    else:
                        dqk_trans = backward_d_silu_activation(
                            dact_qk_trans, pT, qk_trans, scale, valid_mask_trans
                        )
                    dqk_trans = dqk_trans.to(k.dtype)
                    dq_trans += tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32)
                dq = tl.trans(dq_trans) * alpha
                dq_ptrs = (
                    DQ
                    + (seq_start_q + offs_m[:, None]) * stride_dqm
                    + off_h * stride_dqh
                    + offs_qk_d[None, :]
                )
                tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])
        return

    # Create TMA descriptors on device with BLOCK_M/BLOCK_N sizes
    H_kv = H // G
    desc_q = tl.make_tensor_descriptor(
        Q,
        shape=[total_seq_len_q, H * DimQ],
        # pyrefly: ignore [bad-argument-type]
        strides=[H * DimQ, 1],
        block_shape=[BLOCK_M, BLOCK_D_Q],
    )
    desc_q1 = tl.make_tensor_descriptor(
        Q,
        shape=[total_seq_len_q, H * DimQ],
        # pyrefly: ignore [bad-argument-type]
        strides=[H * DimQ, 1],
        block_shape=[BLOCK_M1, BLOCK_D_Q],
    )
    desc_k = tl.make_tensor_descriptor(
        K,
        shape=[total_seq_len_kv, H_kv * DimQ],
        # pyrefly: ignore [bad-argument-type]
        strides=[H_kv * DimQ, 1],
        block_shape=[BLOCK_N, BLOCK_D_Q],
    )
    desc_v = tl.make_tensor_descriptor(
        V,
        shape=[total_seq_len_kv, H_kv * DimV],
        # pyrefly: ignore [bad-argument-type]
        strides=[H_kv * DimV, 1],
        block_shape=[BLOCK_N, BLOCK_D_V],
    )
    desc_do = tl.make_tensor_descriptor(
        DO,
        shape=[total_seq_len_q, H * DimV],
        # pyrefly: ignore [bad-argument-type]
        strides=[H * DimV, 1],
        block_shape=[BLOCK_M, BLOCK_D_V],
    )
    desc_do1 = tl.make_tensor_descriptor(
        DO,
        shape=[total_seq_len_q, H * DimV],
        # pyrefly: ignore [bad-argument-type]
        strides=[H * DimV, 1],
        block_shape=[BLOCK_M1, BLOCK_D_V],
    )

    (
        off_h,
        off_h_kv,
        start_m,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    ) = _compute_bwd_offsets(
        H,
        G,
        BLOCK_M,
        seq_offsets_q,
        seq_offsets,
        max_seq_len,
        TRUNCATE_METHOD,
    )
    if HAS_CAUSAL:
        n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
        uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    else:
        uih_len_q = seq_len_q
    if start_m + BLOCK_M1 >= seq_len_q:
        _hstu_attn_bwd_redkv_inner(
            off_h,
            off_h_kv,
            start_m,
            seq_start_kv,
            seq_len_kv,
            seq_start_q,
            seq_len_q,
            desc_q1,
            desc_k,
            desc_v,
            desc_do1,
            DQ,
            DK,
            DV,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_doh,
            stride_dqm,
            stride_dqh,
            stride_dkn,
            stride_dkh,
            stride_dvn,
            stride_dvh,
            alpha,
            attn_scale,
            M,
            Delta,
            stride_mm,
            uih_len_q,
            num_softmax_heads,
            max_q_len,
            ALLOW_TF32,
            BLOCK_D_Q,
            BLOCK_D_V,
            BLOCK_M1,
            BLOCK_N,
            ATTN_SCALE_TYPE,
            GQA_ATOMIC_ADD=G > 1,
            SHARED_KV=SHARED_KV,
            HAS_CAUSAL=HAS_CAUSAL,
        )
    else:
        _hstu_attn_bwd_redkv_inner(
            off_h,
            off_h_kv,
            start_m,
            seq_start_kv,
            seq_len_kv,
            seq_start_q,
            seq_len_q,
            desc_q,
            desc_k,
            desc_v,
            desc_do,
            DQ,
            DK,
            DV,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_doh,
            stride_dqm,
            stride_dqh,
            stride_dkn,
            stride_dkh,
            stride_dvn,
            stride_dvh,
            alpha,
            attn_scale,
            M,
            Delta,
            stride_mm,
            uih_len_q,
            num_softmax_heads,
            max_q_len,
            ALLOW_TF32,
            BLOCK_D_Q,
            BLOCK_D_V,
            BLOCK_M,
            BLOCK_N,
            ATTN_SCALE_TYPE,
            GQA_ATOMIC_ADD=G > 1,
            SHARED_KV=SHARED_KV,
            HAS_CAUSAL=HAS_CAUSAL,
        )


@triton.jit
def _hstu_attn_bwd_redkv_inner(  # noqa C901
    off_h,
    off_h_kv,
    start_m,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    desc_q,
    desc_k,
    desc_v,
    desc_do,
    DQ,
    DK,
    DV,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    attn_scale,
    M,
    Delta,
    stride_mm,
    uih_len_q,
    num_softmax_heads: tl.constexpr,
    max_q_len: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    GQA_ATOMIC_ADD: tl.constexpr,
    SHARED_KV: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
):
    q_offset = seq_start_q + start_m
    q = desc_q.load([q_offset.to(tl.int32), off_h * stride_qh])
    do = desc_do.load([q_offset.to(tl.int32), off_h * stride_doh])
    q_trans = tl.trans(q)
    dq_trans = tl.zeros([BLOCK_D_Q, BLOCK_M], dtype=tl.float32)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # Offset DQ, DK, DV pointers by sequence start and head offset
    DQ = DQ + seq_start_q * stride_dqm + off_h * stride_dqh
    DK = (
        DK
        + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dkn, tl.int64)
        + off_h_kv * tl.cast(stride_dkh, tl.int64)
    )
    if not SHARED_KV:
        DV = (
            DV
            + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dvn, tl.int64)
            + off_h_kv * tl.cast(stride_dvh, tl.int64)
        )
    mask_m = offs_m < seq_len_q
    if ATTN_SCALE_TYPE == "scalar":
        scale = tl.load(attn_scale).to(tl.float32)
    else:
        tl.static_assert(ATTN_SCALE_TYPE == "dynamic")
        scale = tl.load(attn_scale + offs_m, mask=mask_m).to(tl.float32)

    # Preprocess M and Delta offsets for softmax
    M_off, Delta_off = backward_common_preprocess(
        M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm
    )

    if HAS_CAUSAL:
        high_kv = start_m + BLOCK_M + seq_len_kv - uih_len_q
        high_kv = tl.cdiv(min(high_kv, seq_len_kv), BLOCK_N) * BLOCK_N
    else:
        high_kv = seq_len_kv
    last_block_start_n = tl.maximum(0, (tl.cdiv(high_kv, BLOCK_N) - 1) * BLOCK_N)
    # Skip Q masking when entire Q block is within bounds
    q_needs_mask = start_m + BLOCK_M > seq_len_q
    # Main loop - no KV masking needed for non-last blocks
    if off_h < num_softmax_heads:
        scaled_alpha = alpha * 1.44269504
    else:
        scaled_alpha = alpha
    for start_n in tl.range(0, last_block_start_n, BLOCK_N):
        # load K
        k_offset = seq_start_kv + start_n
        k = desc_k.load([k_offset.to(tl.int32), off_h_kv * stride_kh])
        qk_trans = tl.dot(
            k,
            q_trans,
        )
        offs_n_loop = start_n + tl.arange(0, BLOCK_N)
        if off_h < num_softmax_heads:
            if q_needs_mask or HAS_CAUSAL:
                valid_mask_trans = backward_valid_mask(
                    offs_m,
                    offs_n_loop,
                    uih_len_q,
                    seq_len_q,
                    seq_len_kv,
                    HAS_CAUSAL,
                )
                qk_trans, act_qk_trans, pT = backward_softmax_activation_scaled_alpha(
                    qk_trans,
                    scaled_alpha,
                    valid_mask_trans,
                    M_off,
                    offs_m,
                    stride_mm,
                    mask_m,
                    k,
                )
            else:
                qk_trans = qk_trans * scaled_alpha
                m = tl.load(M_off + offs_m * stride_mm)
                pT = tl.math.exp2(qk_trans - m[None, :])
                act_qk_trans = pT.to(k.dtype)
        else:
            if q_needs_mask or HAS_CAUSAL:
                valid_mask_trans = backward_valid_mask(
                    offs_m,
                    offs_n_loop,
                    uih_len_q,
                    seq_len_q,
                    seq_len_kv,
                    HAS_CAUSAL,
                )
                qk_trans, act_qk_trans, pT = backward_silu_activation(
                    qk_trans, alpha, valid_mask_trans, k.dtype, scale
                )
            else:
                qk_trans = qk_trans * alpha
                pT = fast_silu(qk_trans, MULT_BY_X=False)
                silu_trans = qk_trans * pT * scale[None, :]
                act_qk_trans = silu_trans.to(k.dtype)
        if SHARED_KV:
            dk = tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
            v = k
        else:
            dv = tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
            # acc dv
            offs_v_d = tl.arange(0, BLOCK_D_V)
            dv_ptrs = DV + (offs_n_loop[:, None] * stride_dvn + offs_v_d[None, :])
            if BLOCK_M < max_q_len or GQA_ATOMIC_ADD:
                tl.atomic_add(
                    dv_ptrs,
                    dv.to(q.dtype),
                    sem="relaxed",
                )
            else:
                tl.store(
                    dv_ptrs,
                    dv.to(q.dtype),
                )

            # load V
            v_offset = seq_start_kv + start_n
            v = desc_v.load([v_offset.to(tl.int32), off_h_kv * stride_vh])

        dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
        if off_h < num_softmax_heads:
            dqk_trans = backward_d_softmax_activation(
                dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT
            )
        else:
            if q_needs_mask or HAS_CAUSAL:
                valid_mask_trans = backward_valid_mask(
                    offs_m,
                    offs_n_loop,
                    uih_len_q,
                    seq_len_q,
                    seq_len_kv,
                    HAS_CAUSAL,
                )
                dqk_trans = backward_d_silu_activation(
                    dact_qk_trans,
                    pT,
                    qk_trans,
                    scale,
                    valid_mask_trans,
                )
            else:
                dqk_trans = (
                    dact_qk_trans * pT * (1 + qk_trans * (1 - pT)) * scale[None, :]
                )
        dqk_trans = dqk_trans.to(k.dtype)
        if SHARED_KV:
            dk_attn = tl.dot(dqk_trans, q, allow_tf32=ALLOW_TF32)
            # pyrefly: ignore [unbound-name]
            dk = dk + dk_attn * alpha
        else:
            dk = tl.dot(dqk_trans, q, allow_tf32=ALLOW_TF32) * alpha
        offs_qk_d = tl.arange(0, BLOCK_D_Q)
        dk_ptrs = DK + (offs_n_loop[:, None] * stride_dkn + offs_qk_d[None, :])
        if BLOCK_M < max_q_len or GQA_ATOMIC_ADD:
            tl.atomic_add(
                dk_ptrs,
                dk.to(q.dtype),
                sem="relaxed",
            )
        else:
            tl.store(
                dk_ptrs,
                dk.to(q.dtype),
            )
        # acc dq
        dq_trans += tl.dot(tl.trans(k), dqk_trans)

    # Last KV block - needs masking for out-of-bounds elements;
    # Q/Do masking is kept since Q seq len is short.
    if seq_len_kv > 0:
        start_n = last_block_start_n
        k_offset = seq_start_kv + start_n
        k = desc_k.load([k_offset.to(tl.int32), off_h_kv * stride_kh])
        qk_trans = tl.dot(
            k,
            q_trans,
        )
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_mask_trans = backward_valid_mask(
            offs_m,
            offs_n,
            uih_len_q,
            seq_len_q,
            seq_len_kv,
            HAS_CAUSAL,
        )
        if off_h < num_softmax_heads:
            qk_trans, act_qk_trans, pT = backward_softmax_activation_scaled_alpha(
                qk_trans,
                scaled_alpha,
                valid_mask_trans,
                M_off,
                offs_m,
                stride_mm,
                mask_m,
                k,
            )
        else:
            qk_trans, act_qk_trans, pT = backward_silu_activation(
                qk_trans, alpha, valid_mask_trans, k.dtype, scale
            )
        if SHARED_KV:
            dk = tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
            v = k
        else:
            dv = tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
            mask_n = offs_n < seq_len_kv
            offs_v_d = tl.arange(0, BLOCK_D_V)
            dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
            if BLOCK_M < max_q_len or GQA_ATOMIC_ADD:
                tl.atomic_add(
                    dv_ptrs,
                    dv.to(q.dtype),
                    mask=mask_n[:, None],
                    sem="relaxed",
                )
            else:
                tl.store(dv_ptrs, dv.to(q.dtype), mask=mask_n[:, None])
            v_offset = seq_start_kv + start_n
            v = desc_v.load([v_offset.to(tl.int32), off_h_kv * stride_vh])

        dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
        if off_h < num_softmax_heads:
            dqk_trans = backward_d_softmax_activation(
                dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT
            )
        else:
            dqk_trans = backward_d_silu_activation(
                dact_qk_trans, pT, qk_trans, scale, valid_mask_trans
            )
        dqk_trans = dqk_trans.to(k.dtype)
        if SHARED_KV:
            dk_attn = tl.dot(dqk_trans, q, allow_tf32=ALLOW_TF32)
            # pyrefly: ignore [unbound-name]
            dk = dk + dk_attn * alpha
        else:
            dk = tl.dot(dqk_trans, q, allow_tf32=ALLOW_TF32) * alpha
        offs_qk_d = tl.arange(0, BLOCK_D_Q)
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
        mask_n = offs_n < seq_len_kv
        if BLOCK_M < max_q_len or GQA_ATOMIC_ADD:
            tl.atomic_add(
                dk_ptrs,
                dk.to(q.dtype),
                mask=mask_n[:, None],
                sem="relaxed",
            )
        else:
            tl.store(dk_ptrs, dk.to(q.dtype), mask=mask_n[:, None])
        dq_trans += tl.dot(tl.trans(k), dqk_trans)
    dq_trans = dq_trans * alpha
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len_q
    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_qk_d[None, :])
    dq = tl.trans(dq_trans)
    tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m[:, None])


# AutoWS reduce_dq per-dot schedule (stage/order/channels) lives in the module-level
# configs _HSTU_BWD_DOT_ATTRS_{DEFAULT,TLX}; the active one is exposed as the per-dot
# constexpr globals _HSTU_ATTRS_{QKT,DV,DPT,DK,DQ} (Triton's @jit frontend only
# accepts `attrs=` as a dict literal or a bare tl.constexpr global, not a config
# object passed as a param). Switch configs with set_bwd_dot_attrs(). Channel fmt:
# "operand,memType,copies,bufferId"; a shared bufferId forms a reuse group. DEFAULT:
#   id2: S(qkᵀ) owner, reused by P (dv's operand A)      -> P shares S's slot
#   id5: dP(dact) owner (DEFAULT) ; id11: dqᵀ owner. TLX config points dP at id11
#        (dP consumed before dqᵀ is produced -> dp reuse=dq, freeing id5).
#   id8: dsᵀ (dqk_trans) in SMEM, operand A of dk and operand B of dq
#   id7: dv (or dk's P·do part) standalone ; id10: dk (dsᵀ·q) standalone


@triton.jit
def _hstu_attn_bwd_inner(  # noqa C901
    start_n,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    desc_q,
    desc_k,
    desc_v,
    desc_do,
    desc_dk,
    desc_dq,
    DV,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_doh,
    stride_dvn,
    stride_dvh,
    alpha,
    attn_scale,
    M,
    Delta,
    stride_mm,
    off_h,
    off_h_kv,
    H_kv,
    uih_len_q,
    num_softmax_heads: tl.constexpr,
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    GQA_ATOMIC_ADD: tl.constexpr,
    SHARED_KV: tl.constexpr,
    MASK_KV: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    # pyrefly: ignore [bad-function-definition]
    AUTOWS: tl.constexpr = False,
    # WS_ON gates only the warp-specialization *behavior* (warp_specialize +
    # merge_epilogue + the tt.autows dot annotations), independent of AUTOWS which
    # gates the single-masked-KV-loop *structure*. Normally WS_ON==AUTOWS; setting
    # AUTOWS=True, WS_ON=False runs the autoWS loop structure with NO warp
    # specialization -- a byte-reference that differs from autoWS only in WS.
    WS_ON: tl.constexpr = False,
):
    kv_offset = (seq_start_kv + start_n).to(tl.int32)
    k = desc_k.load([kv_offset.to(tl.int32), off_h_kv * stride_kh])
    if SHARED_KV:
        v = k
    else:
        v = desc_v.load([kv_offset.to(tl.int32), off_h_kv * stride_vh])

    dk = tl.zeros([BLOCK_N, DimQ], dtype=tl.float32)
    if not SHARED_KV:
        dv = tl.zeros([BLOCK_N, DimV], dtype=tl.float32)
    # Offset DV pointers by sequence start and head offset
    if not SHARED_KV:
        DV = (
            DV
            + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dvn, tl.int64)
            + off_h_kv * tl.cast(stride_dvh, tl.int64)
        )
    tl.static_assert(ATTN_SCALE_TYPE == "scalar")
    scale = tl.load(attn_scale).to(tl.float32)

    M_off, Delta_off = backward_common_preprocess(
        M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm
    )

    offs_n = start_n + tl.arange(0, BLOCK_N)
    # EXPERIMENT (autoWS): for the all-or-nothing softmax shape (num_softmax_heads
    # is 0 or H), off_h < num_softmax_heads == the constexpr (num_softmax_heads > 0).
    # The constexpr form folds at compile time (no runtime scf.if/else), so the
    # autoWS hasElse guard does not drop warp specialization. NOT valid for partial
    # num_softmax_heads -- gate behind a SOFTMAX constexpr before landing.
    if num_softmax_heads > 0:
        scaled_alpha = alpha * 1.44269504
    else:
        scaled_alpha = alpha
    if HAS_CAUSAL:
        low_q = max(0, start_n + uih_len_q - seq_len_kv)
    else:
        low_q = 0
    # autoWS depth-2 per-head reduce_dq: warp_specialize the inner Q loop.
    # warp_specialize requires beta triton (-m ovr_config//triton:beta).
    # merge_epilogue=WS_ON: fold the dk/dv epilogue stores into the reduction
    # partition (Meta partition-scheduler merge-epilogue knob) so autoWS has the
    # same 4-partition layout as the hand-TLX kernel (which stores dk/dv in the
    # reduction/default task) instead of a separate epilogue partition.
    for start_m in tl.range(
        low_q, seq_len_q, BLOCK_M, warp_specialize=WS_ON, merge_epilogue=WS_ON
    ):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len_q
        q_offset = seq_start_q + start_m
        q = desc_q.load([q_offset.to(tl.int32), off_h * stride_qh])
        qk_trans = tl.dot(
            k,
            tl.trans(q),
            attrs=(_HSTU_ATTRS_QKT_2KV if (SHARED_KV and _HSTU_COMPUTE_FOLD)
                   else _HSTU_ATTRS_QKT) if WS_ON else None,
        )
        if MASK_KV or HAS_CAUSAL:
            valid_mask_trans = backward_valid_mask(
                offs_m,
                offs_n,
                uih_len_q,
                seq_len_q,
                seq_len_kv,
                HAS_CAUSAL,
            )
        else:
            valid_mask_trans = offs_m[None, :] < seq_len_q
        if num_softmax_heads > 0:  # autoWS: constexpr (see note above)
            qk_trans, act_qk_trans, pT = backward_softmax_activation_scaled_alpha(
                qk_trans,
                scaled_alpha,
                valid_mask_trans,
                M_off,
                offs_m,
                stride_mm,
                mask_m,
                k,
            )
        else:
            qk_trans, act_qk_trans, pT = backward_silu_activation(
                qk_trans, alpha, valid_mask_trans, k.dtype, scale
            )
        do = desc_do.load([q_offset.to(tl.int32), off_h * stride_doh])

        dact_qk_trans = tl.dot(
            v, tl.trans(do), allow_tf32=ALLOW_TF32,
            attrs=(_HSTU_ATTRS_DPT_2KV if (SHARED_KV and _HSTU_COMPUTE_FOLD)
                   else _HSTU_ATTRS_DPT) if WS_ON else None,
        )
        if SHARED_KV:
            dk = tl.dot(
                act_qk_trans, do, dk, allow_tf32=ALLOW_TF32,
                attrs=(_HSTU_ATTRS_DV_2KV if _HSTU_COMPUTE_FOLD
                       else _HSTU_ATTRS_DV) if WS_ON else None,
            )
        else:
            # pyrefly: ignore [unbound-name]
            dv = tl.dot(
                act_qk_trans, do, dv, allow_tf32=ALLOW_TF32,
                attrs=_HSTU_ATTRS_DV if WS_ON else None,
            )

        if num_softmax_heads > 0:  # autoWS: constexpr (see note above)
            dqk_trans = backward_d_softmax_activation(
                dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT
            )
        else:
            dqk_trans = backward_d_silu_activation(
                dact_qk_trans, pT, qk_trans, scale, valid_mask_trans
            )
        if SHARED_KV and _HSTU_COMPUTE_FOLD:
            # Fold dk_attn into dk: pre-scale alpha into dqk_trans (fp32) so both
            # the dq and dk_attn dots carry alpha, then accumulate dk_attn
            # directly into the dv/dk accumulator (dk_shared -> buffer id 7).
            dqk_trans = (dqk_trans * alpha).to(k.dtype)
            dq_trans = tl.dot(
                tl.trans(k), dqk_trans,
                attrs=_HSTU_ATTRS_DQ_2KV if WS_ON else None,
            )
            dk = tl.dot(
                dqk_trans, q, dk, allow_tf32=ALLOW_TF32,
                attrs=_HSTU_ATTRS_DK_SHARED_2KV if WS_ON else None,
            )
        else:
            dqk_trans = dqk_trans.to(k.dtype)
            dq_trans = tl.dot(
                tl.trans(k), dqk_trans,
                attrs=_HSTU_ATTRS_DQ if WS_ON else None,
            )
            if SHARED_KV:
                dk_attn = tl.dot(
                    dqk_trans, q, allow_tf32=ALLOW_TF32,
                    attrs=_HSTU_ATTRS_DK if WS_ON else None,
                )
                dk = dk + dk_attn * alpha
            else:
                dk = tl.dot(
                    dqk_trans, q, dk, allow_tf32=ALLOW_TF32,
                    attrs=_HSTU_ATTRS_DK if WS_ON else None,
                )
            dq_trans = dq_trans * alpha
        dq = tl.trans(dq_trans)
        desc_dq.store(
            [q_offset.to(tl.int32), DimQ * off_h],
            dq.to(desc_dq.dtype),
            store_reduce="add",
        )

    offs_n = start_n + tl.arange(0, BLOCK_N)
    if not SHARED_KV:
        offs_v_d = tl.arange(0, DimV)
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
        if MASK_KV:
            mask_n = offs_n < seq_len_kv
            if GQA_ATOMIC_ADD:
                tl.atomic_add(
                    dv_ptrs,
                    # pyrefly: ignore [unbound-name]
                    dv.to(k.dtype),
                    mask=mask_n[:, None],
                    sem="relaxed",
                )
            else:
                tl.store(
                    dv_ptrs,
                    # pyrefly: ignore [unbound-name]
                    dv.to(k.dtype),
                    mask=mask_n[:, None],
                )
        else:
            if GQA_ATOMIC_ADD:
                tl.atomic_add(
                    dv_ptrs,
                    # pyrefly: ignore [unbound-name]
                    dv.to(k.dtype),
                    sem="relaxed",
                )
            else:
                tl.store(
                    dv_ptrs,
                    # pyrefly: ignore [unbound-name]
                    dv.to(k.dtype),
                )

    if not SHARED_KV:
        dk = dk * alpha

    if GQA_ATOMIC_ADD:
        desc_dk.atomic_add([kv_offset, DimQ * off_h_kv], dk.to(desc_dk.dtype))
    else:
        desc_dk.store([kv_offset, DimQ * off_h_kv], dk.to(desc_dk.dtype))


@triton.jit
def _hstu_attn_bwd_inner_2kv(  # noqa C901
    start_n,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    desc_q,
    desc_k,
    desc_v,
    desc_do,
    desc_dk,
    desc_dq,
    DV,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_doh,
    stride_dvn,
    stride_dvh,
    alpha,
    attn_scale,
    M,
    Delta,
    stride_mm,
    off_h,
    off_h_kv,
    H_kv,
    uih_len_q,
    num_softmax_heads: tl.constexpr,
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ATTN_SCALE_TYPE: tl.constexpr,
    GQA_ATOMIC_ADD: tl.constexpr,
    SHARED_KV: tl.constexpr,
    MASK_KV: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    # pyrefly: ignore [bad-function-definition]
    AUTOWS: tl.constexpr = False,
    WS_ON: tl.constexpr = False,
    # Rank of the list-schedule variant to apply to the inner (start_m) loop.
    # Swept by @triton.autotune when TRITON_USE_LIST_SCHEDULE=1; ignored
    # otherwise. Only the inner loop is autotuned (outer keeps one schedule).
    # pyrefly: ignore [bad-function-definition]
    INNER_PICK: tl.constexpr = 0,
):
    """MANUAL 2-KV-block fork of `_hstu_attn_bwd_inner`.

    Processes TWO KV blocks per call -- block b0 at `start_n` and block b1 at
    `start_n + BLOCK_N` -- so a single program does the 2-way KV parallelism the
    compiler DP pass would otherwise synthesize, but written out explicitly in
    Triton so that pass never runs. SHARED_KV + compute-fold ONLY.
    """
    tl.static_assert(SHARED_KV, "_hstu_attn_bwd_inner_2kv is SHARED_KV only")
    tl.static_assert(ATTN_SCALE_TYPE == "scalar")

    kv_offset0 = (seq_start_kv + start_n).to(tl.int32)
    kv_offset1 = (seq_start_kv + start_n + BLOCK_N).to(tl.int32)
    k0 = desc_k.load([kv_offset0, off_h_kv * stride_kh])
    k1 = desc_k.load([kv_offset1, off_h_kv * stride_kh])
    # shared-KV: v aliases k.
    v0 = k0
    v1 = k1

    dk0 = tl.zeros([BLOCK_N, DimQ], dtype=tl.float32)
    dk1 = tl.zeros([BLOCK_N, DimQ], dtype=tl.float32)

    scale = tl.load(attn_scale).to(tl.float32)

    M_off, Delta_off = backward_common_preprocess(
        M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm
    )

    offs_n0 = start_n + tl.arange(0, BLOCK_N)
    offs_n1 = start_n + BLOCK_N + tl.arange(0, BLOCK_N)
    if num_softmax_heads > 0:
        scaled_alpha = alpha * 1.44269504
    else:
        scaled_alpha = alpha
    if HAS_CAUSAL:
        low_q = max(0, start_n + uih_len_q - seq_len_kv)
    else:
        low_q = 0

    # merge_epilogue is intentionally OFF (unlike the single-KV inner). Folding
    # the dq epilogue store into the computation partition duplicates/misplaces
    # the store for the two-block shared dq accumulator, over-counting dq (norm
    # ratio ~1.37) while dk stays correct. A separate epilogue partition stores
    # dq once, correctly. warp_specialize stays ON.
    #
    # num_stages=1: warp specialization already supplies the parallelism; software
    # pipelining the two-block loop (num_stages>=2) trips the pipeliner
    # ("local_alloc can't predicate") on the doubled operand allocs, so the loop
    # pipeline depth is pinned to 1 here regardless of the kernel num_stages.
    for start_m in tl.range(
        low_q, seq_len_q, BLOCK_M, num_stages=1,
        warp_specialize=WS_ON, merge_epilogue=False,
        list_schedule_pick=INNER_PICK,
    ):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len_q
        q_offset = seq_start_q + start_m
        # q/do are loaded ONCE and shared by both KV blocks.
        q = desc_q.load([q_offset.to(tl.int32), off_h * stride_qh])
        do = desc_do.load([q_offset.to(tl.int32), off_h * stride_doh])

        # ---- KV block b0 (SHARED_KV + compute fold) ----
        qk_trans0 = tl.dot(
            k0, tl.trans(q),
            attrs=_HSTU_ATTRS_QKT_2KV if WS_ON else None,
        )

        ######### computation/activation for P0
        if MASK_KV or HAS_CAUSAL:
            valid_mask_trans0 = backward_valid_mask(
                offs_m, offs_n0, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL
            )
        else:
            valid_mask_trans0 = offs_m[None, :] < seq_len_q
        if num_softmax_heads > 0:
            qk_trans0, act_qk_trans0, pT0 = (
                backward_softmax_activation_scaled_alpha(
                    qk_trans0, scaled_alpha, valid_mask_trans0, M_off, offs_m,
                    stride_mm, mask_m, k0,
                )
            )
        else:
            qk_trans0, act_qk_trans0, pT0 = backward_silu_activation(
                qk_trans0, alpha, valid_mask_trans0, k0.dtype, scale
            )
        dact_qk_trans0 = tl.dot(
            v0, tl.trans(do), allow_tf32=ALLOW_TF32,
            attrs=_HSTU_ATTRS_DPT_2KV if WS_ON else None,
        )
        if num_softmax_heads > 0:
            dqk_trans0 = backward_d_softmax_activation(
                dact_qk_trans0, Delta_off, offs_m, stride_mm, mask_m, pT0
            )
        else:
            dqk_trans0 = backward_d_silu_activation(
                dact_qk_trans0, pT0, qk_trans0, scale, valid_mask_trans0
            )
        ######### computation/activation for P1
        # ---- KV block b1 (SHARED_KV + compute fold) ----
        # WS on: b1 uses DISJOINT buffer.ids (_B1) so its concurrently-live
        # tiles never alias b0's (which stays on _2KV).
        qk_trans1 = tl.dot(
            k1, tl.trans(q),
            attrs=_HSTU_ATTRS_QKT_2KV_B1 if WS_ON else None,
        )
        if MASK_KV or HAS_CAUSAL:
            valid_mask_trans1 = backward_valid_mask(
                offs_m, offs_n1, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL
            )
        else:
            valid_mask_trans1 = offs_m[None, :] < seq_len_q
        # calculate pT
        if num_softmax_heads > 0:
            qk_trans1, act_qk_trans1, pT1 = (
                backward_softmax_activation_scaled_alpha(
                    qk_trans1, scaled_alpha, valid_mask_trans1, M_off, offs_m,
                    stride_mm, mask_m, k1,
                )
            )
        else:
            qk_trans1, act_qk_trans1, pT1 = backward_silu_activation(
                qk_trans1, alpha, valid_mask_trans1, k1.dtype, scale
            )
        dact_qk_trans1 = tl.dot(
            v1, tl.trans(do), allow_tf32=ALLOW_TF32,
            attrs=_HSTU_ATTRS_DPT_2KV_B1 if WS_ON else None,
        )
        # calculate dqk_trans
        if num_softmax_heads > 0:
            dqk_trans1 = backward_d_softmax_activation(
                dact_qk_trans1, Delta_off, offs_m, stride_mm, mask_m, pT1
            )
        else:
            dqk_trans1 = backward_d_silu_activation(
                dact_qk_trans1, pT1, qk_trans1, scale, valid_mask_trans1
            )

        # compute fold: pre-scale alpha into dqk so both dq and dk_attn carry it.
        dqk_trans0 = (dqk_trans0 * alpha).to(k0.dtype)
        # dq is a SINGLE accumulator over both KV blocks: b0 fresh-writes it, b1
        # accumulates into the same TMEM tile (opndD id 11) below. This avoids a
        # cross-partition register sum (dq0 and dq1 land in different warp
        # partitions under WS) and a two-store race, and needs one TMEM tile.
        dq_trans = tl.dot(
            tl.trans(k0), dqk_trans0,
            attrs=_HSTU_ATTRS_DQ_2KV if WS_ON else None,
        )
        # dv fold: accumulate dv into the shared dk accumulator.
        dk0 = tl.dot(
            act_qk_trans0, do, dk0, allow_tf32=ALLOW_TF32,
            attrs=_HSTU_ATTRS_DV_2KV if WS_ON else None,
        )
        dk0 = tl.dot(
            dqk_trans0, q, dk0, allow_tf32=ALLOW_TF32,
            attrs=_HSTU_ATTRS_DK_SHARED_2KV if WS_ON else None,
        )

        # ---- KV block b1 (SHARED_KV + compute fold) ----
        # WS on: b1 uses DISJOINT buffer.ids (_B1) so its concurrently-live
        # tiles never alias b0's (which stays on _2KV).
        dqk_trans1 = (dqk_trans1 * alpha).to(k1.dtype)
        # Accumulate b1's dq into the SAME dq_trans TMEM tile (opndD id 11 via
        # _B1), reducing both KV blocks' contributions in TMEM.
        dq_trans = tl.dot(
            tl.trans(k1), dqk_trans1, dq_trans,
            attrs=_HSTU_ATTRS_DQ_2KV_B1 if WS_ON else None,
        )
        dk1 = tl.dot(
            act_qk_trans1, do, dk1, allow_tf32=ALLOW_TF32,
            attrs=_HSTU_ATTRS_DV_2KV_B1 if WS_ON else None,
        )
        dk1 = tl.dot(
            dqk_trans1, q, dk1, allow_tf32=ALLOW_TF32,
            attrs=_HSTU_ATTRS_DK_SHARED_2KV_B1 if WS_ON else None,
        )

        # ONE store per Q: dq_trans already holds b0+b1.
        dq = tl.trans(dq_trans)
        desc_dq.store(
            [q_offset.to(tl.int32), DimQ * off_h],
            dq.to(desc_dq.dtype),
            store_reduce="add",
        )

    # No dv store (shared-KV: dv folded into dk). alpha already folded into dk.
    if GQA_ATOMIC_ADD:
        desc_dk.atomic_add([kv_offset0, DimQ * off_h_kv], dk0.to(desc_dk.dtype))
        desc_dk.atomic_add([kv_offset1, DimQ * off_h_kv], dk1.to(desc_dk.dtype))
    else:
        desc_dk.store([kv_offset0, DimQ * off_h_kv], dk0.to(desc_dk.dtype))
        desc_dk.store([kv_offset1, DimQ * off_h_kv], dk1.to(desc_dk.dtype))


@maybe_register_custom_op("hammer::triton_hstu_cross_attn_v3_fwd", mutates_args=())
def triton_hstu_cross_attn_v3_fwd(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    max_q_len: int,
    attn_scale: torch.Tensor,
    G: int,
    shared_kv: bool = False,
    num_softmax_heads: int = 0,
    enable_tma: bool = True,
    truncate_method: str = "none",
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    enable_tma = (
        enable_tma
        and HAS_TENSOR_DESCRIPTOR
        and is_sm90_plus()
        and not torch.version.hip
    )
    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)

    # Validate dtype consistency: q, k, v must have matching dtypes for
    # correct type casting in kernels (e.g., dv.to(q.dtype))
    assert q.dtype == k.dtype, f"q.dtype ({q.dtype}) != k.dtype ({k.dtype})"
    assert q.dtype == v.dtype, f"q.dtype ({q.dtype}) != v.dtype ({v.dtype})"

    Z = seq_offsets.numel() - 1
    total_seq_len_q, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty(total_seq_len_q, H, DimV, device=q.device, dtype=q.dtype)
    if total_seq_len_q == 0:
        M = torch.empty(0, device=q.device, dtype=torch.float32)
        return out, M, 0

    total_seq_len_kv, _, _ = k.shape

    if enable_tma:
        # TMA descriptors require a global memory allocation.
        # Note: stream parameter is unused; torch.empty uses current CUDA stream.
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    grid = lambda meta: (  # noqa E731
        triton.cdiv(max_q_len, meta["BLOCK_M"]),
        Z * H,
    )

    # Create M tensor for softmax heads if needed
    if num_softmax_heads > 0:
        M, stride_mm = forward_custom_vars(q, num_softmax_heads, [])
    else:
        M = torch.empty(0, device=q.device, dtype=torch.float32)
        stride_mm = 0

    HAS_NUM_TARGETS = num_targets is not None
    HAS_CAUSAL = causal

    variant = get_fwd_variant()
    if variant == FwdVariant.TRITON:
        _attn_fwd_triton[grid](
            alpha=alpha,
            Z=Z,
            H=H,
            G=G,
            Q=q,
            K=k,
            V=v,
            total_seq_len_q=total_seq_len_q,
            total_seq_len_kv=total_seq_len_kv,
            Out=out,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_om=out.stride(0),
            stride_oh=out.stride(1),
            seq_offsets_q=seq_offsets_q,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            attn_scale=attn_scale,
            M=M,
            stride_mm=stride_mm,
            num_targets=num_targets,
            AUTOTUNE_MAX_Q_LEN=autotune_max_seq_len(max_q_len),
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            num_softmax_heads=num_softmax_heads,
            HEAD_DIM=DimQ,
            BLOCK_D_V=DimV,
            ENABLE_TMA=enable_tma,
            TRUNCATE_METHOD=truncate_method,
            HAS_CAUSAL=HAS_CAUSAL,
            HAS_NUM_TARGETS=HAS_NUM_TARGETS,
        )
    elif variant == FwdVariant.TRITON_DYN_SPEC:
        _attn_fwd_triton_spec[grid](
            alpha=alpha,
            Z=Z,
            H=H,
            G=G,
            Q=q,
            K=k,
            V=v,
            total_seq_len_q=total_seq_len_q,
            total_seq_len_kv=total_seq_len_kv,
            Out=out,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_om=out.stride(0),
            stride_oh=out.stride(1),
            seq_offsets_q=seq_offsets_q,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            attn_scale=attn_scale,
            M=M,
            stride_mm=stride_mm,
            num_targets=num_targets,
            AUTOTUNE_Z=next_power_of_2(Z),
            AUTOTUNE_MAX_Q_LEN=autotune_max_seq_len(max_q_len),
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            num_softmax_heads=num_softmax_heads,
            HEAD_DIM=DimQ,
            BLOCK_D_V=DimV,
            SHARED_KV=shared_kv,
            ENABLE_TMA=enable_tma,
            TRUNCATE_METHOD=truncate_method,
            HAS_CAUSAL=HAS_CAUSAL,
            HAS_NUM_TARGETS=HAS_NUM_TARGETS,
        )
    return out, M, stride_mm


@torch.library.custom_op("hammer::triton_hstu_cross_attn_v3_bwd", mutates_args=())
def triton_hstu_cross_attn_v3_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    max_q_len: Optional[int],
    seq_offsets_q: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    causal: bool,
    shared_kv: bool,
    G: int,
    num_softmax_heads: int = 0,
    M: Optional[torch.Tensor] = None,
    Delta: Optional[torch.Tensor] = None,
    stride_mm: int = 0,
    truncate_method: str = "none",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_q_len is None:
        max_q_len = max_seq_len
        assert seq_offsets_q is None
        seq_offsets_q = seq_offsets
    bwd_variant = get_bwd_variant()
    if (bwd_variant == BwdVariant.AUTO) and (max_q_len < 128):
        bwd_variant = BwdVariant.TRITON_REDKV
    use_per_kv_head = (
        bwd_variant in (BwdVariant.AUTO, BwdVariant.TRITON_REDKV)
        and G > 1
        and (max_q_len < 192 or (num_softmax_heads == 0 and max_q_len <= 256))
    )
    use_redq_per_kv_head = (
        bwd_variant in (BwdVariant.TRITON_REDQ, BwdVariant.TRITON_AUTOWS) and G > 1
    )
    if use_per_kv_head:
        dq_dtype = q.dtype
        dkdv_dtype = k.dtype
    elif use_redq_per_kv_head:
        dq_dtype = torch.float32
        dkdv_dtype = k.dtype
    elif bwd_variant == BwdVariant.TRITON_REDKV:
        dq_dtype = q.dtype
        dkdv_dtype = k.dtype if max_q_len < 128 else torch.float32
    else:
        dq_dtype = torch.float32
        dkdv_dtype = torch.float32 if G > 1 else k.dtype
    dq = torch.empty_like(q, dtype=dq_dtype)
    dk = torch.empty_like(k, dtype=dkdv_dtype)
    if shared_kv:
        dv = dk
    else:
        dv = torch.empty_like(v, dtype=dkdv_dtype)

    dout = switch_to_contiguous_if_needed(dout)
    dq = switch_to_contiguous_if_needed(dq)
    dk = switch_to_contiguous_if_needed(dk)
    if not shared_kv:
        dv = switch_to_contiguous_if_needed(dv)

    # Validate dtype consistency: q, k, v must have matching dtypes for
    # correct type casting in kernels (e.g., dk.to(q.dtype))
    assert q.dtype == k.dtype, f"q.dtype ({q.dtype}) != k.dtype ({k.dtype})"
    assert q.dtype == v.dtype, f"q.dtype ({q.dtype}) != v.dtype ({v.dtype})"

    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    _, H, DimQ = q.shape
    _, _, DimV = v.shape

    total_seq_len_q, H, DimQ = q.shape
    total_seq_len_kv, _, _ = k.shape

    if attn_scale.ndim == 0:
        attn_scale_type = "scalar"
    else:
        attn_scale_type = "dynamic"

    # TMA descriptors require a global memory allocation.
    # Note: stream parameter is unused; torch.empty uses current CUDA stream.
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    HAS_NUM_TARGETS = num_targets is not None
    HAS_CAUSAL = causal

    AUTOTUNE_Z = next_power_of_2(Z)
    if use_per_kv_head or bwd_variant == BwdVariant.TRITON_REDKV:
        if use_per_kv_head:
            H_kv = H // G
            grid = lambda meta: (Z * H_kv,)  # noqa E731
        else:
            grid = lambda meta: (  # noqa E731
                triton.cdiv(max_q_len, meta["BLOCK_M"]),
                Z * H,
            )
        _hstu_attn_bwd_redkv[grid](
            Q=q,
            K=k,
            V=v,
            DO=dout,
            total_seq_len_q=total_seq_len_q,
            total_seq_len_kv=total_seq_len_kv,
            seq_offsets=seq_offsets,
            seq_offsets_q=seq_offsets_q,
            DQ=dq,
            DK=dk,
            DV=dv,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            stride_dqm=dq.stride(0),
            stride_dqh=dq.stride(1),
            stride_dkn=dk.stride(0),
            stride_dkh=dk.stride(1),
            stride_dvn=dv.stride(0),
            stride_dvh=dv.stride(1),
            alpha=alpha,
            max_seq_len=max_seq_len,
            attn_scale=attn_scale,
            M=M,
            Delta=Delta,
            stride_mm=stride_mm,
            num_targets=num_targets,
            Z=Z,
            AUTOTUNE_Z=AUTOTUNE_Z,
            H=H,
            G=G,
            num_softmax_heads=num_softmax_heads,
            max_q_len=next_power_of_2(max_q_len),
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            DimQ=DimQ,
            DimV=DimV,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            BLOCK_D_Q=DimQ,
            BLOCK_D_V=DimV,
            ATTN_SCALE_TYPE=attn_scale_type,
            SHARED_KV=shared_kv,
            TRUNCATE_METHOD=truncate_method,
            HAS_CAUSAL=HAS_CAUSAL,
            HAS_NUM_TARGETS=HAS_NUM_TARGETS,
            PER_KV_HEAD=use_per_kv_head,
        )
    elif bwd_variant in (BwdVariant.TRITON_REDQ, BwdVariant.TRITON_AUTOWS):
        if use_redq_per_kv_head:
            H_kv = H // G
            grid = lambda meta: (Z * H_kv,)  # noqa E731
        else:
            grid = lambda meta: (Z * H, 1)  # noqa E731
        _hstu_attn_bwd_redq[grid](
            Q=q,
            K=k,
            V=v,
            DO=dout,
            total_seq_len_q=total_seq_len_q,
            total_seq_len_kv=total_seq_len_kv,
            seq_offsets=seq_offsets,
            seq_offsets_q=seq_offsets_q,
            DQ=dq,
            DK=dk,
            DV=dv,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            stride_dqm=dq.stride(0),
            stride_dqh=dq.stride(1),
            stride_dkn=dk.stride(0),
            stride_dkh=dk.stride(1),
            stride_dvn=dv.stride(0),
            stride_dvh=dv.stride(1),
            alpha=alpha,
            max_seq_len=max_seq_len,
            attn_scale=attn_scale,
            M=M,
            Delta=Delta,
            stride_mm=stride_mm,
            num_targets=num_targets,
            Z=Z,
            AUTOTUNE_Z=AUTOTUNE_Z,
            H=H,
            G=G,
            num_softmax_heads=num_softmax_heads,
            max_q_len=next_power_of_2(max_q_len),
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            DimQ=DimQ,
            DimV=DimV,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            ATTN_SCALE_TYPE=attn_scale_type,
            SHARED_KV=shared_kv,
            TRUNCATE_METHOD=truncate_method,
            HAS_CAUSAL=HAS_CAUSAL,
            HAS_NUM_TARGETS=HAS_NUM_TARGETS,
            PER_KV_HEAD=use_redq_per_kv_head,
            AUTOWS=(bwd_variant == BwdVariant.TRITON_AUTOWS),
            # WS_ON == AUTOWS normally; set HSTU_AUTOWS_WS_OFF=1 to run the autoWS
            # loop STRUCTURE with warp specialization DISABLED (isolation reference).
            WS_ON=(
                (bwd_variant == BwdVariant.TRITON_AUTOWS)
                and os.environ.get("HSTU_AUTOWS_WS_OFF", "0") != "1"
            ),
        )
    elif bwd_variant == BwdVariant.TRITON_AUTOWS_2KV:
        # MANUAL 2-KV-block data-partition variant. Shared-KV only; G == 1
        # (self-attn) so no per-kv-head atomic path is needed.
        assert shared_kv, "BwdVariant.TRITON_AUTOWS_2KV requires shared_kv"
        assert G == 1, "BwdVariant.TRITON_AUTOWS_2KV requires G == 1"
        grid = lambda meta: (Z * H, 1)  # noqa E731
        _hstu_attn_bwd_redq_2kv[grid](
            Q=q,
            K=k,
            V=v,
            DO=dout,
            total_seq_len_q=total_seq_len_q,
            total_seq_len_kv=total_seq_len_kv,
            seq_offsets=seq_offsets,
            seq_offsets_q=seq_offsets_q,
            DQ=dq,
            DK=dk,
            DV=dv,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            stride_dqm=dq.stride(0),
            stride_dqh=dq.stride(1),
            stride_dkn=dk.stride(0),
            stride_dkh=dk.stride(1),
            stride_dvn=dv.stride(0),
            stride_dvh=dv.stride(1),
            alpha=alpha,
            max_seq_len=max_seq_len,
            attn_scale=attn_scale,
            M=M,
            Delta=Delta,
            stride_mm=stride_mm,
            num_targets=num_targets,
            Z=Z,
            AUTOTUNE_Z=AUTOTUNE_Z,
            H=H,
            G=G,
            num_softmax_heads=num_softmax_heads,
            max_q_len=next_power_of_2(max_q_len),
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            DimQ=DimQ,
            DimV=DimV,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            ATTN_SCALE_TYPE=attn_scale_type,
            SHARED_KV=shared_kv,
            TRUNCATE_METHOD=truncate_method,
            HAS_CAUSAL=HAS_CAUSAL,
            HAS_NUM_TARGETS=HAS_NUM_TARGETS,
            PER_KV_HEAD=False,
            # AUTOWS=False keeps the compiler DP pass from ever running; the
            # manual 2-KV loop + disjoint b0/b1 buffer.ids do the partitioning.
            AUTOWS=False,
            # Milestone 2: warp specialization ON.
            WS_ON=True,
        )
    else:
        grid = lambda meta: (Z * H, 1)  # noqa E731
        _hstu_attn_bwd[grid](
            Q=q,
            K=k,
            V=v,
            DO=dout,
            total_seq_len_q=total_seq_len_q,
            total_seq_len_kv=total_seq_len_kv,
            seq_offsets=seq_offsets,
            seq_offsets_q=seq_offsets_q,
            DQ=dq,
            DK=dk,
            DV=dv,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            stride_dqm=dq.stride(0),
            stride_dqh=dq.stride(1),
            stride_dkn=dk.stride(0),
            stride_dkh=dk.stride(1),
            stride_dvn=dv.stride(0),
            stride_dvh=dv.stride(1),
            alpha=alpha,
            max_seq_len=max_seq_len,
            attn_scale=attn_scale,
            M=M,
            Delta=Delta,
            stride_mm=stride_mm,
            num_targets=num_targets,
            Z=Z,
            AUTOTUNE_Z=AUTOTUNE_Z,
            H=H,
            G=G,
            num_softmax_heads=num_softmax_heads,
            AUTOTUNE_MAX_Q_LEN=autotune_max_seq_len(max_q_len),
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            DimQ=DimQ,
            DimV=DimV,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            ATTN_SCALE_TYPE=attn_scale_type,
            SHARED_KV=shared_kv,
            TRUNCATE_METHOD=truncate_method,
            HAS_CAUSAL=HAS_CAUSAL,
            HAS_NUM_TARGETS=HAS_NUM_TARGETS,
        )

    # When shared_kv=True, dv aliases dk. Return a placeholder to avoid aliasing in return
    # which is required by torch.library.custom_op, otherwise need v.clone()
    if shared_kv:
        dv = torch.empty(0)
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class AttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        max_seq_len: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: Optional[torch.Tensor],
        seq_offsets: torch.Tensor,
        attn_scale: torch.Tensor,
        seq_offsets_q: torch.Tensor,
        max_q_len: int,
        shared_kv: bool,
        num_softmax_heads: int,
        enable_tma: bool,
        truncate_method: str,
        num_targets: Optional[torch.Tensor] = None,
        causal: bool = False,
    ):
        q = switch_to_contiguous_if_needed(q)
        k = switch_to_contiguous_if_needed(k)
        if not shared_kv:
            # pyre-ignore[6]
            v = switch_to_contiguous_if_needed(v)
        else:
            v = k
        Z = seq_offsets.numel() - 1
        total_seq_len_q, H, DimQ = q.shape
        _, H_kv, DimV = v.shape
        assert H % H_kv == 0, f"H ({H}) must be divisible by H_kv ({H_kv})"
        G = H // H_kv  # GQA group size (number of Q heads per KV head)
        if total_seq_len_q == 0:
            out = torch.zeros(total_seq_len_q, H, DimV, device=q.device, dtype=q.dtype)
            return out

        if (
            G > 1
            and H_kv == 1
            and (num_softmax_heads == 0 or num_softmax_heads == H)
            and not causal
        ):
            # GQA can not enabled with causal masking
            seq_offsets_q = seq_offsets_q * G
            max_q_len = max_q_len * G
            num_softmax_heads = num_softmax_heads // G
            q = q.view(total_seq_len_q * G, H_kv, DimQ)
            G = 1

        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert HEAD_DIM_V in {16, 32, 64, 128, 256}

        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        out, M, stride_mm = triton_hstu_cross_attn_v3_fwd(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            seq_offsets_q=seq_offsets_q,
            max_q_len=max_q_len,
            attn_scale=attn_scale,
            shared_kv=shared_kv,
            num_softmax_heads=num_softmax_heads,
            G=G,
            enable_tma=enable_tma,
            truncate_method=truncate_method,
            num_targets=num_targets,
            causal=causal,
        )

        saved_tensors = [q, k, v, seq_offsets, attn_scale, seq_offsets_q]
        if num_softmax_heads > 0:
            saved_tensors.extend([M, out])
        if num_targets is not None:
            saved_tensors.append(num_targets)
        ctx.has_num_targets = num_targets is not None
        ctx.causal = causal
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.max_seq_len = max_seq_len
        ctx.max_q_len = max_q_len
        ctx.truncate_method = truncate_method
        ctx.shared_kv = shared_kv
        ctx.num_softmax_heads = num_softmax_heads
        ctx.stride_mm = stride_mm
        ctx.total_seq_len_q = total_seq_len_q
        ctx.G = G
        ctx.H = H
        ctx.H_kv = H_kv
        return out.view(total_seq_len_q, H, DimV)

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dout: torch.Tensor
    ) -> Tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        saved_tensors = ctx.saved_tensors
        q, k, v, seq_offsets, attn_scale, seq_offsets_q = saved_tensors[:6]
        idx = 6
        num_softmax_heads = ctx.num_softmax_heads
        # Reshape dout from [T, H, DimV] to folded [T*G, H_kv, DimV]
        if (
            ctx.H > 1
            and ctx.H_kv == 1
            and (num_softmax_heads == 0 or num_softmax_heads == ctx.H)
            and not ctx.causal
        ):
            dout = dout.view(ctx.total_seq_len_q * ctx.H, -1, dout.shape[-1])
        if num_softmax_heads > 0:
            M = saved_tensors[idx]
            idx += 1
            out = saved_tensors[idx]
            idx += 1
            Delta = torch.empty_like(M)
            pre_grid = (triton.cdiv(out.shape[0], 128), num_softmax_heads)
            _attn_bwd_preprocess[pre_grid](
                out,
                dout,
                Delta,
                out.shape[0],
                H=out.shape[1],
                softmax_heads=num_softmax_heads,
                BLOCK_M=128,  # pyre-ignore [6]
                HEAD_DIM=out.shape[2],  # pyre-ignore [6]
            )
        else:
            M = torch.empty(0, device=q.device, dtype=torch.float32)
            Delta = torch.empty(0, device=q.device, dtype=torch.float32)
        causal = ctx.causal
        if ctx.has_num_targets:
            num_targets = saved_tensors[idx]
            idx += 1
        else:
            num_targets = None
        if get_bwd_variant() in (BwdVariant.TLX, BwdVariant.TLX_2KV):
            # Hand-written TLX warp-specialized reduce_dq (attn_bwd_ws). Computes
            # its own Delta internally; pass out/M for the softmax path. TLX_2KV
            # dispatches the 2-KV-block data-partitioned variant (attn_bwd_ws_2kv),
            # which is shared-KV only.
            from tlx_bw_cross_attention import tlx_bw_reduce_dq

            use_2kv = get_bwd_variant() == BwdVariant.TLX_2KV
            assert not use_2kv or ctx.shared_kv, (
                "BwdVariant.TLX_2KV (attn_bwd_ws_2kv) requires shared_kv"
            )
            out_t = out if num_softmax_heads > 0 else None
            M_t = M if num_softmax_heads > 0 else None
            dq, dk, dv = tlx_bw_reduce_dq(
                q,
                k,
                v,
                dout,
                seq_offsets,
                attn_scale,
                ctx.max_seq_len,
                ctx.alpha,
                max_q_len=ctx.max_q_len,
                seq_offsets_q=seq_offsets_q,
                shared_kv=ctx.shared_kv,
                num_softmax_heads=num_softmax_heads,
                out=out_t,
                M=M_t,
                use_2kv=use_2kv,
            )
        else:
            dq, dk, dv = torch.ops.hammer.triton_hstu_cross_attn_v3_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                attn_scale=attn_scale,
                max_seq_len=ctx.max_seq_len,
                alpha=ctx.alpha,
                max_q_len=ctx.max_q_len,
                seq_offsets_q=seq_offsets_q,
                num_targets=num_targets,
                causal=causal,
                shared_kv=ctx.shared_kv,
                num_softmax_heads=num_softmax_heads,
                M=M,
                Delta=Delta,
                stride_mm=ctx.stride_mm,
                G=ctx.G,
                truncate_method=ctx.truncate_method,
            )
        dq = dq.view(ctx.total_seq_len_q, ctx.H, -1)
        return (
            None,
            None,
            dq,
            dk,
            dv if not ctx.shared_kv else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.jit.unused
@torch.fx.wrap
def triton_bw_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int] = None,
    seq_offsets_q: Optional[torch.Tensor] = None,
    sort_by_length: bool = False,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
    shared_kv: bool = False,
    enable_tma: bool = True,
    actual_max_seq_len: int = 0,
    truncate_method: str = "none",
) -> torch.Tensor:
    return AttentionFunction.apply(
        max_seq_len,
        alpha,
        q,
        k,
        v if not shared_kv else None,
        seq_offsets,
        attn_scale,
        seq_offsets_q,
        max_q_len,
        shared_kv,
        num_softmax_heads,
        enable_tma,
        truncate_method,
        num_targets,
        causal,
    )


def triton_bw_hstu_mha_wrapper(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int] = None,
    seq_offsets_q: Optional[torch.Tensor] = None,
    sort_by_length: bool = False,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
    shared_kv: bool = False,
    enable_tma: bool = True,
    actual_max_seq_len: int = 0,
    truncate_method: str = "none",
) -> torch.Tensor:
    assert not sort_by_length, "sort_by_length not supported yet in v3 triton kernel"
    return triton_bw_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        attn_scale=attn_scale,
        max_q_len=max_q_len,
        seq_offsets_q=seq_offsets_q,
        sort_by_length=sort_by_length,
        num_softmax_heads=num_softmax_heads,
        num_targets=num_targets,
        causal=causal,
        shared_kv=shared_kv,
        enable_tma=enable_tma,
        actual_max_seq_len=actual_max_seq_len,
        truncate_method=truncate_method,
    )




