# pyre-unsafe
"""
AMD IKBO Flash Attention — TLX kernel benchmark / correctness harness
=====================================================================

In-batch broadcast (IKBO) flash attention: many ad "seeds" (queries) attend over
a longer shared user/request K/V history. `ad_to_user_mapping[ad]` gives the user
batch that owns the K/V for that ad, so the KV sequence length is independent of
the query length and the kernels broadcast K/V across ads internally.

Two calling conventions:
  * base layout  (tlx_base, tlx_tricks):
        query [B*n_seed, H, D], key/value [Bu*kv, H, D], mapping, n_seed, kv
  * ref  layout  (async_simple/async_prefetch/cluster_pipeline):
        q [B, H, n_seed, D], k/v [Bu, H, kv, D], mapping   (SDPA-permuted)

Reference is PyTorch SDPA over the broadcast (index_select'd) K/V.

Usage:
    python amd-ikbo-fa_pipelined_test.py
    python amd-ikbo-fa_pipelined_test.py --kernel tlx_base -b 32 -nseed 256 -skv 512
    python amd-ikbo-fa_pipelined_test.py --kernel tlx_base tlx_tricks async_simple async_prefetch cluster_pipeline -b 1024 -nseed 256 -skv 2560 -d 128 --low 60 --high 70
    python amd-ikbo-fa_pipelined_test.py --mode correctness
"""

import argparse
import math
import os
import random
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
import triton  # @manual
import triton.language as tl  # @manual
import triton.language.extra.tlx as tlx  # @manual

DEVICE = triton.runtime.driver.active.get_active_torch_device()

TRITON_AUTOTUNE = os.environ.get("TRITON_AUTOTUNE", "0")
if TRITON_AUTOTUNE == "1":
    print("Autotuning is enabled, run on exhuastive triton configs!")

_is_hip = torch.version.hip is not None


# ═══════════════════════════════════════════════════════════════════════════
# Base IKBO kernels (from tlx_amd_ikbo_fa_base.py): tlx_flash_attn_ikbo[_base],
# tlx_flash_attn_ikbo_tricks. 3D layout: query [B*n_seed, H, D], K/V [Bu*kv, H, D].
# ═══════════════════════════════════════════════════════════════════════════

_BASE_AMD_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "matrix_instr_nonkdim": 16, "NUM_BUFFERS": 2},
        num_stages=2,
        num_warps=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "matrix_instr_nonkdim": 16, "NUM_BUFFERS": 2},
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "matrix_instr_nonkdim": 16, "NUM_BUFFERS": 2},
        num_stages=1,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "matrix_instr_nonkdim": 32, "NUM_BUFFERS": 2},
        num_stages=1,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "matrix_instr_nonkdim": 32, "NUM_BUFFERS": 2},
        num_stages=1,
        num_warps=2,
    ),
]


def _get_base_amd_autotune_configs():
    if TRITON_AUTOTUNE == "1":
        configs = []
        for block_m in [32, 64, 128]:
            for block_n in [32, 64, 128]:
                for matrix_instr_nonkdim in [16, 32]:
                    for num_stage in [1, 2]:
                        for num_warp in [2, 4, 8]:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_M": block_m,
                                        "BLOCK_N": block_n,
                                        "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                        "NUM_BUFFERS": 2,
                                    },
                                    num_stages=num_stage,
                                    num_warps=num_warp,
                                )
                            )
        return configs
    return _BASE_AMD_CONFIGS


@triton.autotune(
    configs=_get_base_amd_autotune_configs(),
    key=["q_seq_len", "kv_seq_len", "d_model"],
)
@triton.jit  # pragma: no cover
def _attn_fwd_tlx(
    query,
    key,
    value,
    output,
    ad_to_request_offset,
    q_stride0,
    q_stride1,
    q_stride2,
    k_stride0,
    k_stride1,
    k_stride2,
    v_stride0,
    v_stride1,
    v_stride2,
    o_stride0,
    o_stride1,
    o_stride2,
    qk_scale,
    q_seq_len,
    kv_seq_len,
    d_model,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    pid_q_seq = tl.program_id(axis=0)
    pid_ads_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    pid_usr_batch = tl.load(ad_to_request_offset + pid_ads_batch)
    seq_start_kv = pid_usr_batch * kv_seq_len

    q_offset = pid_ads_batch * q_seq_len * q_stride0 + pid_head * q_stride1
    k_offset = seq_start_kv * k_stride0 + pid_head * k_stride1
    v_offset = seq_start_kv * v_stride0 + pid_head * v_stride1

    offs_m = pid_q_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (
        query + q_offset + offs_m[:, None] * q_stride0 + offs_d[None, :] * q_stride2
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len, other=0.0)

    k_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), key.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), value.dtype.element_ty, NUM_BUFFERS)

    k_ptrs = key + k_offset + offs_n[:, None] * k_stride0 + offs_d[None, :] * k_stride2
    v_ptrs = (
        value + v_offset + offs_n[:, None] * v_stride0 + offs_d[None, :] * v_stride2
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    n_blocks = tl.cdiv(kv_seq_len, BLOCK_N)
    n_main = tl.maximum(0, n_blocks - 1)

    buffer_id = 0

    # ---- Prologue: prefetch block 0 ----
    k_buf_cur = tlx.local_view(k_buf, 0)
    k0_token = tlx.async_load(k_ptrs, k_buf_cur, mask=offs_n[:, None] < kv_seq_len)
    v_buf_cur = tlx.local_view(v_buf, 0)
    v0_token = tlx.async_load(v_ptrs, v_buf_cur, mask=offs_n[:, None] < kv_seq_len)
    tlx.async_load_commit_group([k0_token, v0_token])

    # ---- pipeline stage ----
    for block_id in tl.range(0, n_main * BLOCK_N, BLOCK_N, num_stages=0):
        next_off = block_id + BLOCK_N
        # Assuming NUM_BUFFERS=2 in the current implementation
        buffer_id_next = buffer_id ^ 1

        tlx.async_load_wait_group(tl.constexpr(0))
        k_buf_cur = tlx.local_view(k_buf, buffer_id)
        kt_view = tlx.local_trans(k_buf_cur)
        kt_cur = tlx.local_load(kt_view)
        v_buf_cur = tlx.local_view(v_buf, buffer_id)
        v_cur = tlx.local_load(v_buf_cur)

        next_mask = (next_off + offs_n[:, None]) < kv_seq_len
        k_buf_next = tlx.local_view(k_buf, buffer_id_next)
        k_token = tlx.async_load(
            k_ptrs + next_off * k_stride0, k_buf_next, mask=next_mask
        )
        v_buf_next = tlx.local_view(v_buf, buffer_id_next)
        v_token = tlx.async_load(
            v_ptrs + next_off * v_stride0, v_buf_next, mask=next_mask
        )
        tlx.async_load_commit_group([k_token, v_token])

        qk = tl.dot(q, kt_cur)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

        buffer_id = buffer_id_next

    # ---- Epilogue ----
    tlx.async_load_wait_group(tl.constexpr(0))
    k_buf_cur = tlx.local_view(k_buf, buffer_id)
    kt_view = tlx.local_trans(k_buf_cur)
    kt_cur = tlx.local_load(kt_view)
    v_buf_cur = tlx.local_view(v_buf, buffer_id)
    v_cur = tlx.local_load(v_buf_cur)

    kn_last = n_main * BLOCK_N + offs_n
    qk = tl.dot(q, kt_cur)
    qk = tl.where(kn_last[None, :] < kv_seq_len, qk, -1.0e10)
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
    qk = qk * qk_scale - m_ij[:, None]
    p = tl.math.exp2(qk)
    alpha = tl.math.exp2(m_i - m_ij)
    l_ij = tl.sum(p, 1)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)
    acc = acc / l_i[:, None]

    o_off = pid_head * o_stride1 + pid_ads_batch * q_seq_len * o_stride0
    o_ptrs = output + o_off + offs_m[:, None] * o_stride0 + offs_d[None, :] * o_stride2
    tl.store(
        o_ptrs,
        acc.to(output.dtype.element_ty),
        mask=(offs_m[:, None] < q_seq_len) & (offs_d[None, :] < d_model),
    )


def tlx_flash_attn_ikbo_base(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ad_to_request_mapping: torch.Tensor,
    q_seq_len: int,
    kv_seq_len: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """AMD-optimized flash attention for IKBO (base, peeled-epilogue prefetch).

    query: [Ba * n_seeds, H, D]; key/value: [Bu * kv_seq_len, H, D];
    ad_to_request_mapping: [Ba] ad batch id -> user batch id.
    """
    d_model = query.shape[-1]
    BLOCK_D = triton.next_power_of_2(d_model)

    sm_scale = scale
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d_model)
    qk_scale = sm_scale / math.log(2.0)

    output = torch.empty_like(query)

    def grid(META: dict[str, int]) -> tuple[int, int, int]:
        return (
            triton.cdiv(q_seq_len, META["BLOCK_M"]),
            query.shape[0] // q_seq_len,
            query.shape[1],
        )

    _attn_fwd_tlx[grid](
        query,
        key,
        value,
        output,
        ad_to_request_mapping,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        qk_scale,
        q_seq_len,
        kv_seq_len,
        d_model=d_model,
        BLOCK_D=BLOCK_D,
    )
    return output


@triton.autotune(
    configs=_get_base_amd_autotune_configs(),
    key=["q_seq_len", "kv_seq_len", "d_model"],
)
@triton.jit  # pragma: no cover
def _attn_fwd_tlx_tricks(
    query,
    key,
    value,
    output,
    ad_to_request_offset,
    q_stride0,
    q_stride1,
    q_stride2,
    k_stride0,
    k_stride1,
    k_stride2,
    v_stride0,
    v_stride1,
    v_stride2,
    o_stride0,
    o_stride1,
    o_stride2,
    qk_scale,
    q_seq_len,
    kv_seq_len,
    d_model,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    pid_q_seq = tl.program_id(axis=0)
    pid_ads_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    pid_usr_batch = tl.load(ad_to_request_offset + pid_ads_batch)
    seq_start_kv = pid_usr_batch * kv_seq_len

    q_offset = pid_ads_batch * q_seq_len * q_stride0 + pid_head * q_stride1
    k_offset = seq_start_kv * k_stride0 + pid_head * k_stride1
    v_offset = seq_start_kv * v_stride0 + pid_head * v_stride1

    offs_m = pid_q_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (
        query + q_offset + offs_m[:, None] * q_stride0 + offs_d[None, :] * q_stride2
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len, other=0.0)

    k_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), key.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), value.dtype.element_ty, NUM_BUFFERS)

    k_ptrs = key + k_offset + offs_n[:, None] * k_stride0 + offs_d[None, :] * k_stride2
    v_ptrs = (
        value + v_offset + offs_n[:, None] * v_stride0 + offs_d[None, :] * v_stride2
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    buffer_id = 0

    # ---- Prologue: prefetch block 0 ----
    k_buf_cur = tlx.local_view(k_buf, 0)
    k0_token = tlx.async_load(k_ptrs, k_buf_cur, mask=offs_n[:, None] < kv_seq_len)
    v_buf_cur = tlx.local_view(v_buf, 0)
    v0_token = tlx.async_load(v_ptrs, v_buf_cur, mask=offs_n[:, None] < kv_seq_len)
    tlx.async_load_commit_group([k0_token, v0_token])

    # ---- pipeline stage ----
    n_iter = tl.cdiv(kv_seq_len, BLOCK_N)
    n_main = tl.maximum(0, n_iter - 1)
    for i_iter in tl.range(0, n_main, num_stages=0):
        next_off = (i_iter + 1) * BLOCK_N
        next_off = tl.multiple_of(next_off, BLOCK_N)
        # Assuming NUM_BUFFERS=2 in the current implementation
        buffer_id_next = buffer_id ^ 1

        tlx.async_load_wait_group(tl.constexpr(0))
        k_buf_cur = tlx.local_view(k_buf, buffer_id)
        kt_view = tlx.local_trans(k_buf_cur)
        kt_cur = tlx.local_load(kt_view)
        v_buf_cur = tlx.local_view(v_buf, buffer_id)
        v_cur = tlx.local_load(v_buf_cur)

        next_mask = None
        if (i_iter == n_main - 1) & (kv_seq_len % BLOCK_N != 0):
            next_mask = (next_off + offs_n[:, None]) < kv_seq_len

        k_buf_next = tlx.local_view(k_buf, buffer_id_next)
        k_token = tlx.async_load(
            k_ptrs + next_off * k_stride0, k_buf_next, mask=next_mask
        )
        v_buf_next = tlx.local_view(v_buf, buffer_id_next)
        v_token = tlx.async_load(
            v_ptrs + next_off * v_stride0, v_buf_next, mask=next_mask
        )
        tlx.async_load_commit_group([k_token, v_token])

        qk = tl.dot(q, kt_cur)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

        buffer_id = buffer_id_next

    # ---- Epilogue ----
    tlx.async_load_wait_group(tl.constexpr(0))
    k_buf_cur = tlx.local_view(k_buf, buffer_id)
    kt_view = tlx.local_trans(k_buf_cur)
    kt_cur = tlx.local_load(kt_view)
    v_buf_cur = tlx.local_view(v_buf, buffer_id)
    v_cur = tlx.local_load(v_buf_cur)

    kn_last = n_main * BLOCK_N + offs_n
    qk = tl.dot(q, kt_cur)
    qk = tl.where(kn_last[None, :] < kv_seq_len, qk, -1.0e10)
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
    qk = qk * qk_scale - m_ij[:, None]
    p = tl.math.exp2(qk)
    alpha = tl.math.exp2(m_i - m_ij)
    l_ij = tl.sum(p, 1)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)
    inv_li = 1.0 / l_i[:, None]
    acc *= inv_li

    o_off = pid_head * o_stride1 + pid_ads_batch * q_seq_len * o_stride0
    o_ptrs = output + o_off + offs_m[:, None] * o_stride0 + offs_d[None, :] * o_stride2
    tl.store(
        o_ptrs,
        acc.to(output.dtype.element_ty),
        mask=(offs_m[:, None] < q_seq_len) & (offs_d[None, :] < d_model),
    )


def tlx_flash_attn_ikbo_tricks(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ad_to_request_mapping: torch.Tensor,
    q_seq_len: int,
    kv_seq_len: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """AMD-optimized flash attention for IKBO with small tricks (peeled tail mask)."""
    d_model = query.shape[-1]
    BLOCK_D = triton.next_power_of_2(d_model)

    sm_scale = scale
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d_model)
    qk_scale = sm_scale / math.log(2.0)

    output = torch.empty_like(query)

    def grid(META: dict[str, int]) -> tuple[int, int, int]:
        return (
            triton.cdiv(q_seq_len, META["BLOCK_M"]),
            query.shape[0] // q_seq_len,
            query.shape[1],
        )

    _attn_fwd_tlx_tricks[grid](
        query,
        key,
        value,
        output,
        ad_to_request_mapping,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        qk_scale,
        q_seq_len,
        kv_seq_len,
        d_model=d_model,
        BLOCK_D=BLOCK_D,
    )
    return output


# ═══════════════════════════════════════════════════════════════════════════
# Ref IKBO kernels (from tlx_amd_ikbo_fa_ref.py): async_simple, async_prefetch,
# cluster_pipeline. 4D SDPA layout: q [B, H, n_seed, D], K/V [Bu, H, kv, D].
# ═══════════════════════════════════════════════════════════════════════════

_REF_AMD_CONFIGS = [
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_stages=1, num_warps=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=1, num_warps=8),
]


def _get_ref_amd_autotune_configs():
    if TRITON_AUTOTUNE == "1":
        configs = []
        for block_m in [32, 64, 128, 256]:
            for block_n in [32, 64]:
                for num_stage in [1, 2, 3]:
                    for num_warp in [4, 8]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n},
                                num_stages=num_stage,
                                num_warps=num_warp,
                            )
                        )
        return configs
    return _REF_AMD_CONFIGS


@triton.jit
def _assume_strides(
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
):
    tl.assume(stride_qz >= 0)
    tl.assume(stride_qh >= 0)
    tl.assume(stride_qm > 0)
    tl.assume(stride_qk >= 0)
    tl.assume(stride_kz >= 0)
    tl.assume(stride_kh >= 0)
    tl.assume(stride_kn > 0)
    tl.assume(stride_kk >= 0)
    tl.assume(stride_vz >= 0)
    tl.assume(stride_vh >= 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_vk >= 0)
    tl.assume(stride_oz >= 0)
    tl.assume(stride_oh >= 0)
    tl.assume(stride_om > 0)
    tl.assume(stride_ok >= 0)


@triton.autotune(
    configs=_get_ref_amd_autotune_configs(), key=["N_CTX", "KV_N_CTX", "HEAD_DIM"]
)
@triton.jit
def _attn_fwd_async_simple(
    query,
    key,
    value,
    output,
    ad_to_request_offset,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    KV_N_CTX,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    _assume_strides(
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
    )

    pid_m = tl.program_id(axis=0)
    pid_hz = tl.program_id(axis=1)
    off_z = pid_hz // H
    pid_usr = tl.load(ad_to_request_offset + off_z)
    off_h = pid_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = pid_usr * stride_kz + off_h * stride_kh
    v_off = pid_usr * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        query + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if IS_CAUSAL:
        hi = min(KV_N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = KV_N_CTX

    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), key.dtype.element_ty, tl.constexpr(1))
    v_buf = tlx.local_alloc(
        (BLOCK_N, HEAD_DIM), value.dtype.element_ty, tl.constexpr(1)
    )

    k_base = key + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_base = value + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    for start_n in tl.range(0, hi, BLOCK_N, num_stages=0):
        kn = start_n + offs_n
        k_mask = kn[:, None] < KV_N_CTX
        v_mask = kn[:, None] < KV_N_CTX

        tok_k = tlx.async_load(
            k_base + start_n * stride_kn, tlx.local_view(k_buf, 0), mask=k_mask
        )
        tok_v = tlx.async_load(
            v_base + start_n * stride_vn, tlx.local_view(v_buf, 0), mask=v_mask
        )
        tlx.async_load_commit_group([tok_k, tok_v])

        wait_tok = tlx.async_load_wait_group(tl.constexpr(0))
        kt_view = tlx.local_trans(tlx.local_view(k_buf, 0))
        kt_cur = tlx.local_load(kt_view, token=wait_tok)
        v_cur = tlx.local_load(tlx.local_view(v_buf, 0), token=wait_tok)

        qk = tl.dot(q, kt_cur)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        qk = tl.where(kn[None, :] < KV_N_CTX, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * sm_scale)
        qk = qk * sm_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

    acc = acc / l_i[:, None]
    o_ptrs = output + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(
        o_ptrs,
        acc.to(output.dtype.element_ty),
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )


def flash_attn_ikbo_async_simple(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ad_to_request_mapping: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Single-buffer async-DMA FA. q [B,H,N,D]; k/v [Bu,H,KV,D] (BHND)."""
    Ba, H, N_CTX, D = q.shape
    KV_N_CTX = k.shape[2]
    o = torch.empty_like(q)

    if scale is None:
        scale = 1.0 / math.sqrt(D)
    sm_scale = scale / math.log(2.0)

    def grid(META: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(N_CTX, META["BLOCK_M"]), Ba * H)

    _attn_fwd_async_simple[grid](
        q,
        k,
        v,
        o,
        ad_to_request_mapping,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        Ba,
        H,
        N_CTX,
        KV_N_CTX,
        sm_scale,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
    )
    return o


@triton.autotune(
    configs=_get_ref_amd_autotune_configs(), key=["N_CTX", "KV_N_CTX", "HEAD_DIM"]
)
@triton.jit
def _attn_fwd_async_prefetch(
    Q,
    K,
    V,
    Out,
    ad_to_request_offset,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    KV_N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Double-buffered prefetch FA with modulo-scheduled prologue/hot-loop/epilogue."""
    _assume_strides(
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
    )

    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    off_z = pid_hz // H
    pid_usr = tl.load(ad_to_request_offset + off_z)
    off_h = pid_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = pid_usr * stride_kz + off_h * stride_kh
    v_off = pid_usr * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )

    if IS_CAUSAL:
        hi = min(KV_N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = KV_N_CTX

    NUM_BUFFERS: tl.constexpr = tl.constexpr(2)
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_BUFFERS)

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = (hi + BLOCK_N - 1) // BLOCK_N
    n_main = tl.maximum(n_blocks - 1, 0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # ---- Prologue: prefetch block 0 ----
    tok_k0 = tlx.async_load(
        k_ptrs, tlx.local_view(k_buf, 0), mask=offs_n[:, None] < KV_N_CTX
    )
    tok_v0 = tlx.async_load(
        v_ptrs, tlx.local_view(v_buf, 0), mask=offs_n[:, None] < KV_N_CTX
    )
    tlx.async_load_commit_group([tok_k0, tok_v0])

    # ---- Steady-state hot loop ----
    for block_id in tl.range(0, n_main * BLOCK_N, BLOCK_N, num_stages=0):
        next_off = block_id + BLOCK_N
        kn = block_id + offs_n
        next_mask = (next_off + offs_n[:, None]) < KV_N_CTX

        i = block_id // BLOCK_N
        slot_cur = i % 2
        slot_nxt = (i + 1) % 2

        wait_tok = tlx.async_load_wait_group(tl.constexpr(0))
        kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_cur))
        kt_cur = tlx.local_load(kt_view, token=wait_tok)
        v_cur = tlx.local_load(tlx.local_view(v_buf, slot_cur), token=wait_tok)

        tok_k = tlx.async_load(
            k_ptrs + next_off * stride_kn,
            tlx.local_view(k_buf, slot_nxt),
            mask=next_mask,
        )
        tok_v = tlx.async_load(
            v_ptrs + next_off * stride_vn,
            tlx.local_view(v_buf, slot_nxt),
            mask=next_mask,
        )
        tlx.async_load_commit_group([tok_k, tok_v])

        qk = tl.dot(q, kt_cur)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * sm_scale)
        p = tl.math.exp2(qk * sm_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

    # ---- Epilogue: consume the last tile with boundary + causal masking ----
    wait_tok = tlx.async_load_wait_group(tl.constexpr(0))
    slot_last = n_main % 2
    kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_last))
    kt_cur = tlx.local_load(kt_view, token=wait_tok)
    v_cur = tlx.local_load(tlx.local_view(v_buf, slot_last), token=wait_tok)

    kn_last = n_main * BLOCK_N + offs_n
    qk = tl.dot(q, kt_cur)
    qk = tl.where(kn_last[None, :] < KV_N_CTX, qk, float("-inf"))
    if IS_CAUSAL:
        qk = tl.where(offs_m[:, None] >= kn_last[None, :], qk, float("-inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1) * sm_scale)
    p = tl.math.exp2(qk * sm_scale - m_ij[:, None])
    l_ij = tl.sum(p, 1)

    alpha = tl.math.exp2(m_i - m_ij)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    m_i = m_ij

    acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(
        o_ptrs,
        acc.to(Out.dtype.element_ty),
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )


def flash_attn_ikbo_async_prefetch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ad_to_request_mapping: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Double-buffered prefetch FA. q [B,H,N,D]; k/v [Bu,H,KV,D] (BHND)."""
    Ba, H, N_CTX, D = q.shape
    KV_N_CTX = k.shape[2]
    o = torch.empty_like(q)

    if scale is None:
        scale = 1.0 / math.sqrt(D)
    sm_scale = scale / math.log(2.0)

    def grid(META: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(N_CTX, META["BLOCK_M"]), Ba * H)

    _attn_fwd_async_prefetch[grid](
        q,
        k,
        v,
        o,
        ad_to_request_mapping,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        Ba,
        H,
        N_CTX,
        KV_N_CTX,
        sm_scale,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
    )
    return o


# ── Rotated 4-cluster warp pipeline (gfx950) ───────────────────────────────


@triton.aggregate
class SoftmaxState:
    """Running softmax accumulator (acc), denominator (l_i), row-max (m_i)."""

    acc: tl.tensor
    l_i: tl.tensor
    m_i: tl.tensor

    @triton.jit
    def create(BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
        return SoftmaxState(
            tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32),
            tl.full([BLOCK_M], 1.0, dtype=tl.float32),
            tl.full([BLOCK_M], float("-inf"), dtype=tl.float32),
        )

    @triton.jit
    def _nan_max_combine(a, b):
        return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

    @triton.jit
    def _row_max(qk):
        """Row-max reduction with IEEE-754 NaN propagation."""
        return tl.reduce(qk, 1, SoftmaxState._nan_max_combine)

    @triton.jit
    def vec1(
        self,
        qk,
        start_n,
        offs_m,
        offs_n,
        QK_SCALE: tl.constexpr,
        DIAG_OFFSET: tl.constexpr,
        MASK_STEPS: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """VEC1: softmax numerator -- new row-max + exp2 burst -> p, alpha."""
        if MASK_STEPS:
            qk_sm = qk * QK_SCALE
            if IS_CAUSAL:
                kn = start_n + offs_n
                qk_sm = tl.where(
                    offs_m[:, None] + DIAG_OFFSET >= kn[None, :], qk_sm, float("-inf")
                )
            m_ij = SoftmaxState._row_max(qk_sm)
            m_new = tl.maximum(self.m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
            p = tl.math.exp2(qk_sm - m_new[:, None])
            alpha = tl.math.exp2(self.m_i - m_new)
        else:
            m_ij = SoftmaxState._row_max(qk) * QK_SCALE
            m_new = tl.maximum(self.m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
            p = tl.math.exp2(qk * QK_SCALE - m_new[:, None])
            alpha = tl.math.exp2(self.m_i - m_new)
        return SoftmaxState(self.acc, self.l_i, m_new), p, alpha

    @triton.jit
    def vec2(self, p, alpha, out_dtype: tl.constexpr):
        """VEC2: softmax denominator + accumulator correction."""
        l_ij = tl.sum(p, 1)
        acc = self.acc * alpha[:, None]
        l_i = self.l_i * alpha + l_ij
        p_cast = p.to(out_dtype)
        return SoftmaxState(acc, l_i, self.m_i), p_cast


@triton.jit
def _attn_inner_pipelined(
    state,
    q,
    k_ptrs,
    v_ptrs,
    offs_m,
    offs_n,
    block_start,
    block_end,
    k_buf,
    v_buf,
    stride_kn,
    stride_vn,
    QK_SCALE: tl.constexpr,
    DIAG_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BUF_DEPTH: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Rotated 4-cluster pipeline over output tiles [block_start, block_end).
    Requires (block_end - block_start) >= 4."""
    # -- Prologue: prime the pipeline for output tile block_start ----------
    b0 = block_start
    tok_k0 = tlx.async_load(k_ptrs + b0 * BLOCK_N * stride_kn, tlx.local_view(k_buf, 0))
    tok_v0 = tlx.async_load(v_ptrs + b0 * BLOCK_N * stride_vn, tlx.local_view(v_buf, 0))
    tlx.async_load_commit_group([tok_k0])  # ACK[0]
    tlx.async_load_commit_group([tok_v0])  # ACV[0]
    tok_k1 = tlx.async_load(
        k_ptrs + (b0 + 1) * BLOCK_N * stride_kn, tlx.local_view(k_buf, 1)
    )
    tlx.async_load_commit_group([tok_k1])  # ACK[1]

    wait0 = tlx.async_load_wait_group(2)  # K[0] complete
    kt0 = tlx.local_load(
        tlx.local_trans(tlx.local_view(k_buf, 0)), token=wait0, relaxed=True
    )  # LRK[0]
    qk = tl.dot(q, kt0)  # dot_qk[0]
    state, p_c, alpha_c = state.vec1(
        qk, b0 * BLOCK_N, offs_m, offs_n, QK_SCALE, DIAG_OFFSET, MASK_STEPS, IS_CAUSAL
    )  # VEC1[block_start]

    tl.debug_barrier()  # WAR: LRK[0] ds_read vs K[2] write into slot 0
    tok_k2 = tlx.async_load(
        k_ptrs + (b0 + 2) * BLOCK_N * stride_kn, tlx.local_view(k_buf, 0)
    )
    tlx.async_load_commit_group([tok_k2])  # ACK[2] (slot 0 reuse)
    wait1 = tlx.async_load_wait_group(1)  # K[1] complete
    kt_dot = tlx.local_load(
        tlx.local_trans(tlx.local_view(k_buf, 1)), token=wait1, relaxed=True
    )  # LRK[1]
    tok_v1 = tlx.async_load(
        v_ptrs + (b0 + 1) * BLOCK_N * stride_vn, tlx.local_view(v_buf, 1)
    )
    tlx.async_load_commit_group([tok_v1])  # ACV[1]

    # -- Main loop: output tiles [block_start, block_end-3) ----------------
    for block_n in tl.range(block_start, block_end - 3, num_stages=0):
        cur_slot = (block_n - block_start) % BUF_DEPTH
        nxt_slot = (block_n + 1 - block_start) % BUF_DEPTH
        ack_n = (block_n + 3) * BLOCK_N
        acv_n = (block_n + 2) * BLOCK_N
        ahead_n = (block_n + 1) * BLOCK_N

        with tlx.warp_pipeline_stage("dot1", priority=0):
            qk = tl.dot(q, kt_dot)  # dot_qk s1 -> qk[i+1]
            state, p_dot = state.vec2(p_c, alpha_c, q.dtype)  # VEC2 s0

        tlx.async_load_wait_group(1)  # V[i] complete (for LRV[i])

        with tlx.warp_pipeline_stage("mem1", priority=1):
            v_dot = tlx.local_load(
                tlx.local_view(v_buf, cur_slot), relaxed=True
            )  # LRV s0
            tok_k = tlx.async_load(
                k_ptrs + ack_n * stride_kn, tlx.local_view(k_buf, nxt_slot)
            )
            tlx.async_load_commit_group([tok_k])  # ACK s3

        with tlx.warp_pipeline_stage("dot2", priority=0):
            acc = tl.dot(p_dot, v_dot, state.acc)  # dot_pv s0
            state = SoftmaxState(acc, state.l_i, state.m_i)
            state, p_c, alpha_c = state.vec1(
                qk,
                ahead_n,
                offs_m,
                offs_n,
                QK_SCALE,
                DIAG_OFFSET,
                MASK_STEPS,
                IS_CAUSAL,
            )  # VEC1 s1 -> p[i+1]

        tlx.async_load_wait_group(1)  # K[i+2] complete (for LRK[i+2])

        with tlx.warp_pipeline_stage("mem2", priority=1):
            kt_dot = tlx.local_load(
                tlx.local_trans(tlx.local_view(k_buf, cur_slot)), relaxed=True
            )  # LRK s2
            tok_v = tlx.async_load(
                v_ptrs + acv_n * stride_vn, tlx.local_view(v_buf, cur_slot)
            )
            tlx.async_load_commit_group([tok_v])  # ACV s2

    # -- Drain: last 3 output tiles, no OOB global prefetch ----------------
    nm3 = block_end - 3
    nm2 = block_end - 2
    nm1 = block_end - 1
    s_nm3 = (nm3 - block_start) % BUF_DEPTH
    s_nm2 = (nm2 - block_start) % BUF_DEPTH
    s_nm1 = (nm1 - block_start) % BUF_DEPTH

    qk = tl.dot(q, kt_dot)  # dot_qk[n-2]
    tlx.async_load_wait_group(2)  # V[n-3] complete
    v_dot = tlx.local_load(tlx.local_view(v_buf, s_nm3), relaxed=True)  # LRV[n-3]
    state, p_dot = state.vec2(p_c, alpha_c, q.dtype)  # VEC2[n-3]
    acc = tl.dot(p_dot, v_dot, state.acc)  # dot_pv[n-3]
    state = SoftmaxState(acc, state.l_i, state.m_i)
    state, p_c, alpha_c = state.vec1(
        qk, nm2 * BLOCK_N, offs_m, offs_n, QK_SCALE, DIAG_OFFSET, MASK_STEPS, IS_CAUSAL
    )  # VEC1[n-2]
    tl.debug_barrier()  # WAR: LRV[n-3] vs V[n-1] write
    tok_vlast = tlx.async_load(
        v_ptrs + nm1 * BLOCK_N * stride_vn, tlx.local_view(v_buf, s_nm1)
    )
    tlx.async_load_commit_group([tok_vlast])  # ACV[n-1]
    tlx.async_load_wait_group(2)  # K[n-1] complete
    kt_dot = tlx.local_load(
        tlx.local_trans(tlx.local_view(k_buf, s_nm1)), relaxed=True
    )  # LRK[n-1]

    qk = tl.dot(q, kt_dot)  # dot_qk[n-1]
    tlx.async_load_wait_group(1)  # V[n-2] complete
    v_dot = tlx.local_load(tlx.local_view(v_buf, s_nm2), relaxed=True)  # LRV[n-2]
    state, p_dot = state.vec2(p_c, alpha_c, q.dtype)  # VEC2[n-2]
    acc = tl.dot(p_dot, v_dot, state.acc)  # dot_pv[n-2]
    state = SoftmaxState(acc, state.l_i, state.m_i)
    state, p_c, alpha_c = state.vec1(
        qk, nm1 * BLOCK_N, offs_m, offs_n, QK_SCALE, DIAG_OFFSET, MASK_STEPS, IS_CAUSAL
    )  # VEC1[n-1]

    tlx.async_load_wait_group(0)  # V[n-1] complete
    v_dot = tlx.local_load(tlx.local_view(v_buf, s_nm1), relaxed=True)  # LRV[n-1]
    state, p_dot = state.vec2(p_c, alpha_c, q.dtype)  # VEC2[n-1]
    acc = tl.dot(p_dot, v_dot, state.acc)  # dot_pv[n-1]
    state = SoftmaxState(acc, state.l_i, state.m_i)

    return state


@triton.jit
def _attn_fwd_cluster_pipeline(
    Q,
    K,
    V,
    Out,
    ad_to_request_offset,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    N_CTX,
    KV_N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Rotated 4-cluster FA forward. Square anchor: N_CTX % BLOCK_M == 0,
    KV_N_CTX % BLOCK_N == 0, and each block range used is >= 4 tiles."""
    _assume_strides(
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
    )

    if IS_CAUSAL:
        off_h = tl.program_id(0)
        pid_m = tl.program_id(1)
    else:
        pid_m = tl.program_id(0)
        off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    # IKBO: ad batch (off_z) maps to the shared user/request batch owning K/V.
    pid_usr = tl.load(ad_to_request_offset + off_z)

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = pid_usr * stride_kz + off_h * stride_kh
    v_off = pid_usr * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )

    # sm_scale already carries 1/ln(2) (host pre-divides): exp2(qk*sm_scale)==exp(qk*scale).
    QK_SCALE: tl.constexpr = sm_scale
    DIAG_OFFSET: tl.constexpr = 0  # square anchor: N_CTX_K == N_CTX_Q

    state = SoftmaxState.create(BLOCK_M, HEAD_DIM)

    BUF_DEPTH: tl.constexpr = 2
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, BUF_DEPTH)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, BUF_DEPTH)

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = KV_N_CTX // BLOCK_N

    if IS_CAUSAL:
        masked_blocks: tl.constexpr = BLOCK_M // BLOCK_N
        causal_end = (pid_m + 1) * masked_blocks
        n_full = causal_end - masked_blocks
        if n_full > 0:
            state = _attn_inner_pipelined(
                state,
                q,
                k_ptrs,
                v_ptrs,
                offs_m,
                offs_n,
                0,
                n_full,
                k_buf,
                v_buf,
                stride_kn,
                stride_vn,
                QK_SCALE,
                DIAG_OFFSET,
                BLOCK_N,
                BUF_DEPTH,
                False,
                True,
            )
        state = _attn_inner_pipelined(
            state,
            q,
            k_ptrs,
            v_ptrs,
            offs_m,
            offs_n,
            n_full,
            causal_end,
            k_buf,
            v_buf,
            stride_kn,
            stride_vn,
            QK_SCALE,
            DIAG_OFFSET,
            BLOCK_N,
            BUF_DEPTH,
            True,
            True,
        )
    else:
        state = _attn_inner_pipelined(
            state,
            q,
            k_ptrs,
            v_ptrs,
            offs_m,
            offs_n,
            0,
            n_blocks,
            k_buf,
            v_buf,
            stride_kn,
            stride_vn,
            QK_SCALE,
            DIAG_OFFSET,
            BLOCK_N,
            BUF_DEPTH,
            False,
            False,
        )

    acc = state.acc / state.l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(
        o_ptrs,
        acc.to(Out.dtype.element_ty),
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )


def _cluster_default_block_n(causal):
    """Shape-based BLOCK_N (BLOCK_M fixed at 256). Causal=32, non-causal=64."""
    if causal:
        return 32
    return 64


def flash_attn_ikbo_cluster_pipeline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ad_to_request_mapping: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Rotated 4-cluster warp-pipeline FA, IKBO variant. q [B,H,N,D]; k/v [Bu,H,KV,D].

    Mask-free rotated pipeline: requires N_CTX % BLOCK_M(256) == 0,
    KV_N_CTX % BLOCK_N == 0, KV_N_CTX // BLOCK_N >= 4, D == 128.
    """
    Ba, H, N_CTX, D = q.shape
    KV_N_CTX = k.shape[2]
    assert k.shape[1] == H and k.shape[3] == D, (
        f"cluster_pipeline: K must be [Bu, {H}, KV_N_CTX, {D}], got {tuple(k.shape)}"
    )
    assert v.shape == k.shape, (
        f"cluster_pipeline: V {tuple(v.shape)} must match K {tuple(k.shape)}"
    )

    if scale is None:
        scale = 1.0 / math.sqrt(D)
    sm_scale = scale / math.log(2.0)

    MFMA_M = 32
    BLOCK_M = 256
    BLOCK_N = _cluster_default_block_n(causal)
    num_warps = min(8, max(1, BLOCK_M // MFMA_M))
    waves_per_eu = 0 if causal else 2

    assert N_CTX % BLOCK_M == 0, (
        f"cluster_pipeline: N_CTX ({N_CTX}) must be a multiple of BLOCK_M ({BLOCK_M})"
    )
    assert KV_N_CTX % BLOCK_N == 0, (
        f"cluster_pipeline: KV_N_CTX ({KV_N_CTX}) must be a multiple of BLOCK_N ({BLOCK_N}); "
        "the mask-free rotated pipeline has no ragged-tail handling"
    )
    assert BLOCK_M >= num_warps * MFMA_M, (
        f"cluster_pipeline: BLOCK_M ({BLOCK_M}) must be >= num_warps ({num_warps}) * MFMA_M ({MFMA_M})"
    )
    if causal:
        assert BLOCK_M % BLOCK_N == 0, (
            f"cluster_pipeline causal: BLOCK_M ({BLOCK_M}) must be a multiple of BLOCK_N ({BLOCK_N})"
        )
        assert BLOCK_M // BLOCK_N >= 4, (
            f"cluster_pipeline causal: diagonal band BLOCK_M/BLOCK_N ({BLOCK_M // BLOCK_N}) must be >= 4 tiles"
        )
    else:
        assert KV_N_CTX // BLOCK_N >= 4, (
            "cluster_pipeline: need at least 4 K/V blocks for the rotated pipeline"
        )

    o = torch.empty_like(q)
    m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    grid = (H, m_blocks, Ba) if causal else (m_blocks, H, Ba)
    _attn_fwd_cluster_pipeline[grid](
        q,
        k,
        v,
        o,
        ad_to_request_mapping,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        N_CTX,
        KV_N_CTX,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    return o


# ═══════════════════════════════════════════════════════════════════════════
# Input generation + reference (adapted from ikbo_flash_attention_bench.py)
# ═══════════════════════════════════════════════════════════════════════════


def _generate_num_ads_per_user(low: int, high: int, max_threshold: int) -> list[int]:
    """List of #ads per request, summing to `max_threshold` (~uniform in [low, high])."""
    res = []
    cum_sum = 0
    while True:
        cur = random.randint(low, high)
        if cum_sum + cur >= max_threshold:
            res.append(max_threshold - cum_sum)
            break
        cum_sum += cur
        res.append(cur)
    return res


def generate_ikbo_flash_attention_inputs(
    low_num_ads_per_req: int,
    high_num_ads_per_req: int,
    B: int,  # ads batch size
    num_heads: int,
    n_seed: int,  # query seq length (# seeds)
    d_model: int,
    max_seq_len: int,  # key/value seq length
    dtype: torch.dtype = torch.float16,
    device=DEVICE,
):
    num_ads_per_user = torch.tensor(
        _generate_num_ads_per_user(
            low=low_num_ads_per_req, high=high_num_ads_per_req, max_threshold=B
        )
    )
    Bu = num_ads_per_user.size(0)
    ad_to_user_mapping = (
        torch.repeat_interleave(torch.arange(Bu), num_ads_per_user).int().to(device)
    )
    query = torch.randn((B * n_seed, num_heads, d_model), device=device, dtype=dtype)
    key = torch.randn(
        (Bu * max_seq_len, num_heads, d_model), device=device, dtype=dtype
    )
    value = torch.randn(
        (Bu * max_seq_len, num_heads, d_model), device=device, dtype=dtype
    )
    return query, key, value, ad_to_user_mapping


def pytorch_sdpa(query, key, value, sm_scale):
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=sm_scale,
    )


def broadcast_sdpa(
    query,
    key,
    value,
    ad_to_user_mapping,
    B,
    n_seed,
    num_heads,
    d_model,
    max_seq_len,
    sm_scale,
):
    """Eager reference: SDPA over the per-ad broadcast K/V. Returns [B, H, n_seed, D]."""
    query_sdpa = query.view(B, n_seed, num_heads, d_model).permute(0, 2, 1, 3)
    key_sdpa = key.view(-1, max_seq_len, num_heads, d_model)
    key_b = torch.index_select(key_sdpa, dim=0, index=ad_to_user_mapping).permute(
        0, 2, 1, 3
    )
    value_sdpa = value.view(-1, max_seq_len, num_heads, d_model)
    value_b = torch.index_select(value_sdpa, dim=0, index=ad_to_user_mapping).permute(
        0, 2, 1, 3
    )
    return pytorch_sdpa(query_sdpa, key_b, value_b, sm_scale)


def measure_tflops(ms, B, num_heads, q_seq_len, kv_seq_len, d_model):
    # 2 matmuls (QK, PV), 2 flops each: 2 * 2 * B * H * (q*kv) * D
    total_flops = 2 * 2.0 * B * num_heads * (q_seq_len * kv_seq_len) * d_model
    return total_flops / ms * 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# Kernel registry — (callable, layout). "base" takes 3D + (n_seed, kv); "ref"
# takes the 4D SDPA-permuted tensors and broadcasts K/V internally via mapping.
# ═══════════════════════════════════════════════════════════════════════════

KERNEL_REGISTRY = {
    "tlx_base": (tlx_flash_attn_ikbo_base, "base"),
    "tlx_tricks": (tlx_flash_attn_ikbo_tricks, "base"),
    "async_simple": (flash_attn_ikbo_async_simple, "ref"),
    "async_prefetch": (flash_attn_ikbo_async_prefetch, "ref"),
    "cluster_pipeline": (flash_attn_ikbo_cluster_pipeline, "ref"),
}


def get_kernel(name):
    if name not in KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown kernel: {name!r}. Available: {list(KERNEL_REGISTRY.keys())}"
        )
    return KERNEL_REGISTRY[name]


def make_kernel_call(
    name, query, key, value, mapping, B, H, n_seed, d_model, max_seq_len
):
    """Return (call_fn, canonicalize_fn). `call_fn()` runs only the kernel (timed);
    `canonicalize_fn(out)` maps the raw output to the [B, H, n_seed, D] reference layout."""
    fn, layout = get_kernel(name)
    if layout == "base":

        def call():
            return fn(query, key, value, mapping, n_seed, max_seq_len)

        def canon(out):
            return out.view(B, n_seed, H, d_model).permute(0, 2, 1, 3)

    else:  # "ref" — SDPA-permuted 4D inputs (K/V per-user, broadcast in-kernel)
        q4 = query.view(B, n_seed, H, d_model).permute(0, 2, 1, 3)
        k4 = key.view(-1, max_seq_len, H, d_model).permute(0, 2, 1, 3)
        v4 = value.view(-1, max_seq_len, H, d_model).permute(0, 2, 1, 3)

        def call():
            return fn(q4, k4, v4, mapping)

        def canon(out):
            return out  # already [B, H, n_seed, D]

    return call, canon


# ═══════════════════════════════════════════════════════════════════════════
# Verification + summary table
# ═══════════════════════════════════════════════════════════════════════════


def verify(name, got, ref, atol=2e-2, rtol=2e-2, log=True):
    diff = (got.float() - ref.float()).abs()
    ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
    if log:
        status = "PASS" if ok else "FAIL"
        print(
            f"  {name:<30} {status}  max={diff.max().item():.6f}  mean={diff.mean().item():.6f}"
        )
    return ok


def print_summary_table(results, kernel_names):
    providers = ["Torch SDPA", "Broadcast SDPA"] + list(kernel_names)
    rows = []
    for key in sorted(results.keys()):
        B, H, D, n_seed, kv = key
        rows.append((f"B={B}, H={H}, D={D}, n_seed={n_seed}, kv={kv}", results[key]))

    cfg_w = (
        max([len("Config")] + [len(lbl) for lbl, _ in rows]) if rows else len("Config")
    )
    col_w = max([14] + [len(p) for p in providers])

    hdr = f"| {'Config':<{cfg_w}} |" + "".join(f" {p:>{col_w}} |" for p in providers)
    sep = f"|{'-' * (cfg_w + 2)}|" + "".join(f"{'-' * (col_w + 2)}|" for _ in providers)

    print(f"\n{'=' * len(sep)}")
    print("Summary (TFLOPS)")
    print(f"{'=' * len(sep)}")
    print(hdr)
    print(sep)
    for label, prov in rows:
        vals = (
            f"{prov[p]['tflops']:>{col_w}.1f}" if p in prov else f"{'—':>{col_w}}"
            for p in providers
        )
        print(f"| {label:<{cfg_w}} |" + "".join(f" {v} |" for v in vals))
    print(f"{'=' * len(sep)}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════


def run_benchmark(args):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    results = {}

    for B in args.b:
        for H in args.hq:
            for D in args.d:
                for n_seed in args.nseed:
                    for kv in args.skv:
                        random.seed(42)
                        torch.manual_seed(42)
                        query, key, value, mapping = (
                            generate_ikbo_flash_attention_inputs(
                                low_num_ads_per_req=args.low,
                                high_num_ads_per_req=args.high,
                                B=B,
                                num_heads=H,
                                n_seed=n_seed,
                                d_model=D,
                                max_seq_len=kv,
                                dtype=dtype,
                            )
                        )
                        sm_scale = 1.0 / math.sqrt(D)
                        ref = broadcast_sdpa(
                            query, key, value, mapping, B, n_seed, H, D, kv, sm_scale
                        )

                        key_id = (B, H, D, n_seed, kv)
                        results.setdefault(key_id, {})

                        if "Torch SDPA" not in results[key_id]:
                            q4 = query.view(B, n_seed, H, D).permute(0, 2, 1, 3)
                            k_sd = key.view(-1, kv, H, D)
                            kb = torch.index_select(k_sd, 0, mapping).permute(
                                0, 2, 1, 3
                            )
                            v_sd = value.view(-1, kv, H, D)
                            vb = torch.index_select(v_sd, 0, mapping).permute(
                                0, 2, 1, 3
                            )
                            sdpa_fn = lambda: pytorch_sdpa(q4, kb, vb, sm_scale)  # noqa: E731
                            ms = triton.testing.do_bench(sdpa_fn, warmup=25, rep=100)
                            # ms = torch._inductor.utils.do_bench_using_profiling(
                            #     sdpa_fn, warmup=25, rep=100
                            # )
                            results[key_id]["Torch SDPA"] = {
                                "ms": ms,
                                "tflops": measure_tflops(ms, B, H, n_seed, kv, D),
                            }

                        # Broadcast SDPA: the full eager path the IKBO kernels
                        # replace -- the per-ad K/V broadcast (index_select) IS
                        # timed too, not just the attention.
                        if "Broadcast SDPA" not in results[key_id]:
                            bcast_fn = lambda: broadcast_sdpa(  # noqa: E731
                                query,
                                key,
                                value,
                                mapping,
                                B,
                                n_seed,
                                H,
                                D,
                                kv,
                                sm_scale,
                            )
                            ms = triton.testing.do_bench(bcast_fn, warmup=25, rep=100)
                            results[key_id]["Broadcast SDPA"] = {
                                "ms": ms,
                                "tflops": measure_tflops(ms, B, H, n_seed, kv, D),
                            }

                        for kernel_name in args.kernel:
                            tag = f"{kernel_name} B={B} H={H} D={D} n_seed={n_seed} kv={kv}"
                            try:
                                call, canon = make_kernel_call(
                                    kernel_name,
                                    query,
                                    key,
                                    value,
                                    mapping,
                                    B,
                                    H,
                                    n_seed,
                                    D,
                                    kv,
                                )
                                out = call()
                                if not verify("", canon(out), ref, log=False):
                                    print(f"  {tag:55s} -> SKIPPED (correctness)")
                                    continue
                                ms = triton.testing.do_bench(call, warmup=25, rep=100)
                                # ms = torch._inductor.utils.do_bench_using_profiling(
                                #     sdpa_fn, warmup=25, rep=100
                                # )
                            except Exception as e:
                                print(f"  {tag:55s} -> SKIPPED ({e})")
                                continue
                            results[key_id][kernel_name] = {
                                "ms": ms,
                                "tflops": measure_tflops(ms, B, H, n_seed, kv, D),
                            }

    print_summary_table(results, args.kernel)


def run_correctness(args):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    all_ok = True
    for B in args.b:
        for H in args.hq:
            for D in args.d:
                for n_seed in args.nseed:
                    for kv in args.skv:
                        random.seed(42)
                        torch.manual_seed(42)
                        query, key, value, mapping = (
                            generate_ikbo_flash_attention_inputs(
                                args.low, args.high, B, H, n_seed, D, kv, dtype=dtype
                            )
                        )
                        sm_scale = 1.0 / math.sqrt(D)
                        ref = broadcast_sdpa(
                            query, key, value, mapping, B, n_seed, H, D, kv, sm_scale
                        )
                        print(f"B={B} H={H} D={D} n_seed={n_seed} kv={kv}")
                        for kernel_name in args.kernel:
                            try:
                                call, canon = make_kernel_call(
                                    kernel_name,
                                    query,
                                    key,
                                    value,
                                    mapping,
                                    B,
                                    H,
                                    n_seed,
                                    D,
                                    kv,
                                )
                                ok = verify(kernel_name, canon(call()), ref)
                            except Exception as e:
                                ok = False
                                print(f"  {kernel_name:<30} SKIPPED ({e})")
                            all_ok &= ok
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# Pytest correctness (vs broadcast SDPA), modeled on ikbo_flash_attention_test.py
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("kernel_name", list(KERNEL_REGISTRY))
def test_ikbo_fa_correctness(kernel_name, B=1024, H=1, n_seed=256, D=128, kv=2560):
    """Each IKBO kernel vs PyTorch broadcast SDPA. Shape satisfies the cluster
    kernel's square-anchor constraints (n_seed % 256 == 0, kv % 64 == 0, >= 4 kv
    blocks, D == 128) so every kernel is exercised."""
    random.seed(2)
    torch.manual_seed(2)
    query, key, value, mapping = generate_ikbo_flash_attention_inputs(
        low_num_ads_per_req=1,
        high_num_ads_per_req=100,
        B=B,
        num_heads=H,
        n_seed=n_seed,
        d_model=D,
        max_seq_len=kv,
    )
    sm_scale = 1.0 / math.sqrt(D)
    ref = broadcast_sdpa(query, key, value, mapping, B, n_seed, H, D, kv, sm_scale)
    call, canon = make_kernel_call(
        kernel_name, query, key, value, mapping, B, H, n_seed, D, kv
    )
    assert verify(kernel_name, canon(call()), ref), f"correctness failed: {kernel_name}"


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(prog="AMD TLX IKBO FA Pipelined")
    p.add_argument(
        "-b", type=int, nargs="+", default=[1024, 2048], help="ads batch sizes"
    )
    p.add_argument("-hq", type=int, nargs="+", default=[1], help="num heads")
    p.add_argument(
        "-nseed", type=int, nargs="+", default=[256], help="query seq length (# seeds)"
    )
    p.add_argument(
        "-skv", type=int, nargs="+", default=[2560], help="kv (max) seq length"
    )
    p.add_argument("-d", type=int, nargs="+", default=[128], help="d_model / head dim")
    p.add_argument("--low", type=int, default=60, help="min ads per user")
    p.add_argument("--high", type=int, default=70, help="max ads per user")
    p.add_argument("--dtype", type=str, default="fp16", choices=["bf16", "fp16"])
    p.add_argument(
        "--kernel",
        type=str,
        nargs="+",
        default=list(KERNEL_REGISTRY),
        choices=list(KERNEL_REGISTRY),
        help="IKBO kernel variants",
    )
    p.add_argument("--mode", choices=["benchmark", "correctness"], default="benchmark")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "correctness":
        run_correctness(args)
    else:
        run_benchmark(args)
