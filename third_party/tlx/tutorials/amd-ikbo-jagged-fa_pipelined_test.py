# pyre-unsafe
"""
AMD IKBO *Jagged* Flash Attention — TLX kernel benchmark / correctness harness
==============================================================================

In-batch broadcast (IKBO) flash attention with 2D jaggedness: many ad "seeds"
(queries) attend over a longer, per-user shared K/V history whose length varies
per request. `ad_to_user_mapping[ad]` gives the user batch that owns the K/V for
that ad; `query_offset` / `key_offset` are the packed jagged offsets.

Kernels (all share one calling convention — packed jagged tensors):
    jagged                    -- double-buffered async-DMA baseline
    jagged_persistent         -- XCD-pinned persistent scheduler
    jagged_cluster_pipeline   -- rotated 4-cluster warp pipeline (gfx950)

Reference is the per-ad PyTorch SDPA over each ad's real (unpadded) user K/V range.

Usage (TLX + the cluster pipeline need MI350x / gfx950 and the triton beta):
    # all three kernels, default sweep (B in {1024, 2048}, q=256, kv in [400, 2000])
    python amd-ikbo-jagged-fa_pipelined_test.py

    # correctness only, vs the per-ad PyTorch jagged SDPA reference
    python amd-ikbo-jagged-fa_pipelined_test.py --mode correctness

    # a single kernel at one batch size
    python amd-ikbo-jagged-fa_pipelined_test.py --kernel jagged_cluster_pipeline -b 1024

    # explicit full sweep across the three kernels
    python amd-ikbo-jagged-fa_pipelined_test.py --kernel jagged jagged_persistent jagged_cluster_pipeline -b 1024 2048 -nseed 256 --min-kv 400 --max-kv 2000 -d 128 --low 30 --high 40

    # larger batch / longer, wider-varying jagged kv histories
    python amd-ikbo-jagged-fa_pipelined_test.py --kernel jagged_persistent -b 4096 -nseed 256 --min-kv 800 --max-kv 4000

    # fixed (non-jagged) kv length, and also time the eager reference for a baseline row
    python amd-ikbo-jagged-fa_pipelined_test.py --kernel jagged_cluster_pipeline --min-kv 512 --max-kv 512 --time-ref

    # sweep several (min, max) kv ranges in one run (paired element-wise)
    python amd-ikbo-jagged-fa_pipelined_test.py --min-kv 200 400 --max-kv 1000 2000 -b 1024

    # exhaustive autotuning of the base kernel's triton configs
    TRITON_AUTOTUNE=1 python amd-ikbo-jagged-fa_pipelined_test.py --kernel jagged
"""

import argparse
import math
import os
import random
from functools import lru_cache
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

IS_HIP = tl.constexpr(torch.version.hip is not None)
_is_hip = torch.version.hip is not None


# ═══════════════════════════════════════════════════════════════════════════
# Jagged IKBO kernels — inlined from tlx_amd_ikbo_jagged_fa_base.py, converted to
# plain host wrappers (no custom_triton_op / capture_triton) for a standalone run.
# ═══════════════════════════════════════════════════════════════════════════


_AMD_CONFIGS = [
    # Top performers with matrix_instr_nonkdim=16 (16x16 MFMA)
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "matrix_instr_nonkdim": 16, "NUM_BUFFERS_KV": 2},
        num_stages=2,
        num_warps=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "matrix_instr_nonkdim": 16, "NUM_BUFFERS_KV": 2},
        num_stages=2,
        num_warps=4,
    ),
    # AMD MI350x perf for num_head=1, q_seq_len=32, kv_seq_length=[200, 2000], d_model=256
    triton.Config(
        {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "matrix_instr_nonkdim": 32,
            "NUM_BUFFERS_KV": 2,
        },
        num_stages=2,
        num_warps=2,
    ),
    # AMD MI350x perf for num_head=1, q_seq_len=32, kv_seq_length=[200, 2000], d_model=128
    triton.Config(
        {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "matrix_instr_nonkdim": 16,
            "NUM_BUFFERS_KV": 2,
        },
        num_stages=2,
        num_warps=2,
    ),
    # AMD MI350x perf for num_head=1, q_seq_len=256, kv_seq_length=[200, 2000], d_model=256
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "matrix_instr_nonkdim": 32,
            "NUM_BUFFERS_KV": 2,
        },
        num_stages=2,
        num_warps=4,
    ),
    # AMD MI350x perf for num_head=1, q_seq_len=256, kv_seq_length=[200, 2000], d_model=128
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "matrix_instr_nonkdim": 32,
            "NUM_BUFFERS_KV": 2,
        },
        num_stages=1,
        num_warps=2,
    ),
]


def _get_amd_autotune_configs():
    if TRITON_AUTOTUNE == "1":
        configs = []
        block_m_list = [32, 64, 128]
        block_n_list = [32, 64, 128]
        matrix_instr_nonkdim_list = [16, 32]
        num_stage_list = [1, 2]
        num_warp_list = [2, 4, 8]
        for block_m in block_m_list:
            for block_n in block_n_list:
                for matrix_instr_nonkdim in matrix_instr_nonkdim_list:
                    for num_stage in num_stage_list:
                        for num_warp in num_warp_list:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_M": block_m,
                                        "BLOCK_N": block_n,
                                        "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                        "NUM_BUFFERS_KV": 2,
                                    },
                                    num_stages=num_stage,
                                    num_warps=num_warp,
                                )
                            )
    else:
        configs = _AMD_CONFIGS
    return configs


@lru_cache
def get_num_sms() -> int | None:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties("cuda").multi_processor_count


def expect_contiguous(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is not None and not x.is_contiguous():
        return x.contiguous()
    return x


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


@triton.jit  # pragma: no cover
def pid_swizzle(pid: int, off_hz: int, n_tile_num: int, HZ: int) -> tuple[int, int]:
    if IS_HIP:
        off_hz, pid = tl.swizzle2d(off_hz, pid, HZ, n_tile_num, 16)
    # pyrefly: ignore [missing-attribute]
    return pid.to(tl.int32), off_hz.to(tl.int32)


"""
===============================================================================
TLX IKBO jagged FA base version (Assume Q is jagged, K/V are jagged and shared the same offset)
===============================================================================
"""


@triton.autotune(
    configs=_get_amd_autotune_configs(),
    key=["q_seq_len", "kv_seq_len", "d_head"],
)
@triton.jit  # pragma: no cover
def _attn_fwd_jagged_tlx(
    query,
    q_offsets,
    key,
    k_offsets,
    value,
    out,
    ad_to_request_offset,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    qk_scale,
    d_head,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
):
    start_m = tl.program_id(axis=0)
    off_z = tl.program_id(axis=1)
    off_h = tl.program_id(axis=2)
    q_offset = off_h.to(tl.int64) * stride_qh
    kv_offset = off_h.to(tl.int64) * stride_kh

    # Get q sequence length
    begin_q = tl.load(q_offsets + off_z)
    end_q = tl.load(q_offsets + off_z + 1)
    q_seq_len = end_q - begin_q

    # Get k/v sequence length
    off_zkv = tl.load(ad_to_request_offset + off_z)
    begin_k = tl.load(k_offsets + off_zkv)
    end_k = tl.load(k_offsets + off_zkv + 1)
    kv_seq_len = end_k - begin_k

    if start_m * BLOCK_M >= q_seq_len:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (
        query
        + q_offset
        + (begin_q + offs_m[:, None]) * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len, other=0.0)

    k_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), key.dtype.element_ty, NUM_BUFFERS_KV)
    v_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), value.dtype.element_ty, NUM_BUFFERS_KV)

    k_ptrs = (
        key
        + kv_offset
        + (begin_k + offs_n[:, None]) * stride_kn
        + offs_d[None, :] * stride_kd
    )
    v_ptrs = (
        value
        + kv_offset
        + (begin_k + offs_n[:, None]) * stride_vn
        + offs_d[None, :] * stride_vd
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

        # Mask every prefetch unconditionally (matches the proven padded TLX
        # kernel). A data-dependent `if` that leaves next_mask=None on the
        # non-boundary path does not lower reliably in @triton.jit (None vs
        # tensor across control flow) -> the boundary block can load unmasked
        # -> OOB / stale LDS -> intermittent NaN.
        next_mask = (next_off + offs_n[:, None]) < kv_seq_len

        k_buf_next = tlx.local_view(k_buf, buffer_id_next)
        k_token = tlx.async_load(
            k_ptrs + next_off * stride_kn, k_buf_next, mask=next_mask
        )
        v_buf_next = tlx.local_view(v_buf, buffer_id_next)
        v_token = tlx.async_load(
            v_ptrs + next_off * stride_vn, v_buf_next, mask=next_mask
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
    # # async_load(mask=...) does NOT zero-fill masked lanes, so the partial last
    # # block's out-of-range kt_cur/v_cur lanes are uninitialized LDS. Zero them
    # # before the dots: a NaN/inf in any MFMA input lane can contaminate the
    # # whole output tile, and `p (=0) * v(inf/nan)` in the P·V dot yields NaN.
    # kt_cur = tl.where(kn_last[None, :] < kv_seq_len, kt_cur, 0.0)
    # v_cur = tl.where(kn_last[:, None] < kv_seq_len, v_cur, 0.0)
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

    o_ptrs = (
        out
        + off_h.to(tl.int64) * stride_oh
        + (begin_q + offs_m[:, None]) * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(
        o_ptrs,
        acc.to(out.dtype.element_ty),
        mask=(offs_m[:, None] < q_seq_len) & (offs_d[None, :] < d_head),
    )


def tlx_jagged_flash_attn_ikbo(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    ad_to_request_mapping: torch.Tensor,
    max_seq_len: int,  # Maximum sequence length of queries
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    AMD-optimized flash attention for IKBO with 2D jaggedness base version.

    Same interface as triton_flash_attn_ikbo. Requires AMD GPU (HIP).
    Set ADS_MKL_IKBO_USE_TLX=1 (default) for TLX async DMA path (needs
    triton beta), or ADS_MKL_IKBO_USE_TLX=0 for standard path.

    query: [Ba * n_seeds, H, D] Dense tensor
    key: [[k0_seq_len, k1_seq_len, ...], H, D] Dense tensor (jagged-like)
    value: [[v0_seq_len, v1_seq_len, ...], H, D] Dense tensor
    ad_to_request_mapping: [Ba] tensor mapping ad batch id -> user batch id
    """
    d_head = query.shape[-1]
    BLOCK_D = triton.next_power_of_2(d_head)

    sm_scale = scale
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d_head)
    qk_scale = sm_scale / math.log(2.0)

    output = torch.empty_like(query)

    nheads = query.shape[1]
    BATCH = query_offset.size(0) - 1

    def grid(META: dict[str, int]) -> tuple[int, int, int]:
        return (
            triton.cdiv(max_seq_len, META["BLOCK_M"]),
            BATCH,
            nheads,
        )

    _attn_fwd_jagged_tlx[grid](
        query,
        query_offset,
        key,
        key_offset,
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
        d_head=d_head,
        BLOCK_D=BLOCK_D,
    )
    return output


"""
===============================================================================
Persistent variant: XCD-pinned, flattened work units, async-prefetch per tile.

Learned from `flash_attn_persistent` in tlx_amd_fa_ref.py / amd_fa_persistent.py:
- Launch a 1D grid of NUM_SMS programs (one persistent workgroup per CU).
- Flatten the (batch, head) work dim into a single HZ pile and pin each HZ id
  to a fixed XCD (`hz % NUM_XCDS == xcd`) so all m-tiles of a given batch-head
  reuse the same K/V in that XCD's L2 slice.
- Each program round-robins over its XCD's units. IKBO jagged FA is non-causal,
  so every m-tile is equal cost => TILES_PER_UNIT=1 (no zig-zag folding needed).
- The per-tile compute is the proven double-buffered async-prefetch body from
  `_attn_fwd_jagged_tlx`, factored into `_jagged_attn_tile` so the K/V LDS
  buffers are allocated ONCE per program and reused across all tiles it runs.

Jagged specifics: q_seq_len varies per batch, so the number of live m-tiles per
HZ varies. We use `num_m_blocks = cdiv(max_seq_len, BLOCK_M)` as a static upper
bound (matching the baseline 3D grid's axis-0 extent) and cheaply skip tiles
whose `start_m * BLOCK_M >= q_seq_len`.
===============================================================================
"""


@triton.jit  # pragma: no cover
def _jagged_attn_tile(
    start_m,
    query,
    key,
    value,
    out,
    q_head_off,
    kv_head_off,
    o_head_off,
    begin_q,
    q_seq_len,
    begin_k,
    kv_seq_len,
    stride_qm,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_om,
    stride_od,
    k_buf,
    v_buf,
    qk_scale,
    d_head,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
):
    """Compute one output m-tile of one (batch, head) using the double-buffered
    async-prefetch softmax loop (numerically identical to _attn_fwd_jagged_tlx).

    The K/V LDS buffers are owned by the persistent parent kernel and passed in
    so they are reused across every tile a program runs. The caller guards the
    jagged skip (m-tile entirely past q_seq_len) before invoking this helper.
    """
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (
        query
        + q_head_off
        + (begin_q + offs_m[:, None]) * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len, other=0.0)

    k_ptrs = (
        key
        + kv_head_off
        + (begin_k + offs_n[:, None]) * stride_kn
        + offs_d[None, :] * stride_kd
    )
    v_ptrs = (
        value
        + kv_head_off
        + (begin_k + offs_n[:, None]) * stride_vn
        + offs_d[None, :] * stride_vd
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
            k_ptrs + next_off * stride_kn, k_buf_next, mask=next_mask
        )
        v_buf_next = tlx.local_view(v_buf, buffer_id_next)
        v_token = tlx.async_load(
            v_ptrs + next_off * stride_vn, v_buf_next, mask=next_mask
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

    # ---- Epilogue (masked last block) ----
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

    o_ptrs = (
        out
        + o_head_off
        + (begin_q + offs_m[:, None]) * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(
        o_ptrs,
        acc.to(out.dtype.element_ty),
        mask=(offs_m[:, None] < q_seq_len) & (offs_d[None, :] < d_head),
    )


@triton.autotune(
    configs=_get_amd_autotune_configs(),
    key=["q_seq_len", "kv_seq_len", "d_head"],
)
@triton.jit  # pragma: no cover
def _attn_fwd_jagged_tlx_persistent(
    query,
    q_offsets,
    key,
    k_offsets,
    value,
    out,
    ad_to_request_offset,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    qk_scale,
    d_head,
    max_seq_len,
    HZ,
    H,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """Persistent, XCD-pinned scheduler over the flattened (batch, head) x m-tile
    work space. Non-causal => every m-tile is equal cost (TILES_PER_UNIT=1).

    Work units (one per (head, m-tile)) are handed out XCD-strided so that all
    m-tiles of a given head land on the same XCD (shared L2 for its K/V), while
    individual m-tiles are spread across that XCD's workgroups -- this keeps the
    fine-grained parallelism the baseline flat-grid relies on rather than
    serializing a head's tiles into one program.
    """
    pid = tl.program_id(0)
    xcd = pid % NUM_XCDS
    local = pid // NUM_XCDS
    NUM_LOCAL: tl.constexpr = NUM_SMS // NUM_XCDS

    # K/V LDS double buffers, allocated once and reused across every tile this
    # persistent program runs.
    k_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), key.dtype.element_ty, NUM_BUFFERS_KV)
    v_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), value.dtype.element_ty, NUM_BUFFERS_KV)

    # Upper-bound m-tiles per head (jagged tiles past q_seq_len are skipped).
    units_per_hz = tl.cdiv(max_seq_len, BLOCK_M)
    hz_per_xcd = (HZ + NUM_XCDS - 1) // NUM_XCDS
    units = hz_per_xcd * units_per_hz

    for unit in tl.range(local, units, NUM_LOCAL, num_stages=0):
        local_hz = unit // units_per_hz
        start_m = unit % units_per_hz
        pid_hz = xcd + local_hz * NUM_XCDS  # global (batch, head) id.
        if pid_hz < HZ:
            off_z = pid_hz // H
            off_h = pid_hz % H

            q_head_off = off_h.to(tl.int64) * stride_qh
            kv_head_off = off_h.to(tl.int64) * stride_kh
            o_head_off = off_h.to(tl.int64) * stride_oh

            begin_q = tl.load(q_offsets + off_z)
            end_q = tl.load(q_offsets + off_z + 1)
            q_seq_len = end_q - begin_q

            off_zkv = tl.load(ad_to_request_offset + off_z)
            begin_k = tl.load(k_offsets + off_zkv)
            end_k = tl.load(k_offsets + off_zkv + 1)
            kv_seq_len = end_k - begin_k

            # Jagged skip: this m-tile is entirely past the query length.
            if start_m * BLOCK_M < q_seq_len:
                _jagged_attn_tile(
                    start_m,
                    query,
                    key,
                    value,
                    out,
                    q_head_off,
                    kv_head_off,
                    o_head_off,
                    begin_q,
                    q_seq_len,
                    begin_k,
                    kv_seq_len,
                    stride_qm,
                    stride_qd,
                    stride_kn,
                    stride_kd,
                    stride_vn,
                    stride_vd,
                    stride_om,
                    stride_od,
                    k_buf,
                    v_buf,
                    qk_scale,
                    d_head,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_D=BLOCK_D,
                    NUM_BUFFERS_KV=NUM_BUFFERS_KV,
                )


def tlx_jagged_flash_attn_ikbo_persistent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    ad_to_request_mapping: torch.Tensor,
    max_seq_len: int,  # Maximum sequence length of queries
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Persistent AMD-optimized flash attention for IKBO with 2D jaggedness.

    Same interface/semantics as tlx_jagged_flash_attn_ikbo, but uses a 1D
    persistent grid with an XCD-pinned, flattened work scheduler for better L2
    K/V reuse and lower launch/scheduling overhead. Requires AMD GPU (HIP).
    """
    d_head = query.shape[-1]
    BLOCK_D = triton.next_power_of_2(d_head)

    sm_scale = scale
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d_head)
    qk_scale = sm_scale / math.log(2.0)

    output = torch.empty_like(query)

    nheads = query.shape[1]
    BATCH = query_offset.size(0) - 1
    HZ = BATCH * nheads

    num_xcds = 8
    cu_count = get_num_sms()
    assert cu_count is not None, "persistent IKBO jagged FA requires a CUDA/HIP device"
    # Oversubscription: launch OVERSUB workgroups per CU. At 1 WG/CU (LDS-bound)
    # a single wave of persistent programs cannot hide memory latency for this
    # abundantly-parallel workload; more resident-eligible workgroups let the
    # hardware overlap turnover. Env-tunable for sweeping.
    oversub = int(os.environ.get("IKBO_PERSISTENT_OVERSUB", "1"))
    num_sms = (cu_count // num_xcds) * num_xcds * oversub

    grid = (num_sms,)

    _attn_fwd_jagged_tlx_persistent[grid](
        query,
        query_offset,
        key,
        key_offset,
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
        d_head=d_head,
        max_seq_len=max_seq_len,
        HZ=HZ,
        H=nheads,
        BLOCK_D=BLOCK_D,
        NUM_SMS=num_sms,
        NUM_XCDS=num_xcds,
    )
    return output


"""
===============================================================================
Cluster-pipeline variant: rotated 4-cluster warp pipeline (gfx950 / CDNA4).

Learned from flash_attn_ikbo_cluster_pipeline (D109588865) and its asymmetric
LDS-ring refinement (D112423091), adapted to 2D jaggedness.

The hot loop over full K/V tiles is eight named logical sub-clusters rotated
across a depth-4 software pipeline, partitioned with `tlx.warp_pipeline_stage`
into 4 clusters (dot1 / mem1 / dot2 / mem2) so one warp group runs a stage ahead
of the other:

    dot_qk  -- Q * K^T MFMA -> qk scores       LRK -- local-read K (LDS -> regs)
    dot_pv  -- P * V   MFMA -> acc              LRV -- local-read V (LDS -> regs)
    VEC1    -- softmax numerator (max + exp2)   ACK -- async-copy K (global -> LDS)
    VEC2    -- softmax denominator + rescale    ACV -- async-copy V (global -> LDS)

LDS ring is asymmetric: K is prefetched 3 tiles ahead (tighter WAR) so it gets
K_DEPTH=3 slots; V is prefetched 2 ahead and stays at V_DEPTH=2. At BLOCK_N=64/
D=128 that is ~80KB, well within the gfx950 160KB budget.

Jaggedness adaptation: the reference pipeline is mask-free and needs
KV % BLOCK_N == 0 and >= 4 tiles. Jagged kv_seq_len is arbitrary, so we run the
deep pipeline over the `n_full = kv_seq_len // BLOCK_N` fully-populated tiles
(only when n_full >= 4), then finish the ragged tail with a boundary-masked step.
When n_full < 4 (very short kv) we fall back to masked steps over all tiles.
===============================================================================
"""


@triton.aggregate
class SoftmaxState:
    """Running softmax accumulator (acc), denominator (l_i), and row-max (m_i),
    with the two VEC sub-cluster primitives as methods (D109588865)."""

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
    def vec1(self, qk, QK_SCALE):
        """VEC1: softmax numerator -- new row-max + exp2 burst. Produces the
        unnormalized probabilities p and the rescale factor alpha, carried to the
        next iteration (consumed by vec2 and dot_pv). Uses plain tl.max (matching
        the base jagged kernel numerics); the full-tile region is unmasked."""
        m_ij = tl.max(qk, 1) * QK_SCALE
        m_new = tl.maximum(self.m_i, m_ij)
        p = tl.math.exp2(qk * QK_SCALE - m_new[:, None])
        alpha = tl.math.exp2(self.m_i - m_new)
        return SoftmaxState(self.acc, self.l_i, m_new), p, alpha

    @triton.jit
    def vec2(self, p, alpha, out_dtype: tl.constexpr):
        """VEC2: softmax denominator + accumulator correction. Op order matches
        the reference: row-sum, acc rescale, denominator update, p->fp16 cast."""
        l_ij = tl.sum(p, 1)
        acc = self.acc * alpha[:, None]
        l_i = self.l_i * alpha + l_ij
        p_cast = p.to(out_dtype)
        return SoftmaxState(acc, l_i, self.m_i), p_cast


@triton.jit  # pragma: no cover
def _jagged_attn_inner_pipelined(
    state,
    q,
    k_ptrs,
    v_ptrs,
    block_start,
    block_end,
    k_buf,
    v_buf,
    stride_kn,
    stride_vn,
    qk_scale,
    BLOCK_N: tl.constexpr,
    K_DEPTH: tl.constexpr,
    V_DEPTH: tl.constexpr,
):
    """Rotated 4-cluster pipeline over FULL K/V tiles [block_start, block_end).
    Requires (block_end - block_start) >= 4 and every tile fully in-bounds
    (mask-free). K/V LDS buffers are owned by the caller and reused. The softmax
    `state` is threaded in and out. Faithful port of D109588865 + D112423091 with
    the causal masking path removed (jagged FA is non-causal)."""
    # -- Prologue: prime the pipeline for output tile block_start ----------
    # Commit order: K0, V0, K1, K2, V1 -> asymmetric ring (K_DEPTH=3, V_DEPTH=2).
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
    state, p_c, alpha_c = state.vec1(qk, qk_scale)  # VEC1[block_start]

    # ACK[2] into slot 2 (K_DEPTH=3 gives WAR slack, no debug_barrier needed).
    tok_k2 = tlx.async_load(
        k_ptrs + (b0 + 2) * BLOCK_N * stride_kn, tlx.local_view(k_buf, 2 % K_DEPTH)
    )
    tlx.async_load_commit_group([tok_k2])  # ACK[2]
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
        # Tile-absolute LDS slots: K tile T -> (T-block_start) % K_DEPTH, likewise
        # V with V_DEPTH. Deeper K ring gives the async prefetch more WAR slack.
        rel = block_n - block_start
        v_rd_slot = rel % V_DEPTH  # LRV[i]
        k_rd_slot = (rel + 2) % K_DEPTH  # LRK[i+2]
        ack_slot = (rel + 3) % K_DEPTH  # ACK[i+3]
        acv_slot = (rel + 2) % V_DEPTH  # ACV[i+2]
        ack_n = (block_n + 3) * BLOCK_N
        acv_n = (block_n + 2) * BLOCK_N

        # cluster 0 DOT1: dot_qk[i+1] then VEC2[i].
        with tlx.warp_pipeline_stage("dot1", priority=0):
            qk = tl.dot(q, kt_dot)  # dot_qk s1 -> qk[i+1]
            state, p_dot = state.vec2(p_c, alpha_c, q.dtype)  # VEC2 s0

        tlx.async_load_wait_group(1)  # V[i] complete (for LRV[i])

        # cluster 1 MEM1: LRV[i] then ACK[i+3].
        with tlx.warp_pipeline_stage("mem1", priority=1):
            v_dot = tlx.local_load(
                tlx.local_view(v_buf, v_rd_slot), relaxed=True
            )  # LRV s0
            tok_k = tlx.async_load(
                k_ptrs + ack_n * stride_kn, tlx.local_view(k_buf, ack_slot)
            )
            tlx.async_load_commit_group([tok_k])  # ACK s3

        # cluster 2 DOT2: dot_pv[i] then VEC1[i+1] (exp2 burst lands after PV).
        with tlx.warp_pipeline_stage("dot2", priority=0):
            acc = tl.dot(p_dot, v_dot, state.acc)  # dot_pv s0
            state = SoftmaxState(acc, state.l_i, state.m_i)
            state, p_c, alpha_c = state.vec1(qk, qk_scale)  # VEC1 s1 -> p[i+1]

        tlx.async_load_wait_group(1)  # K[i+2] complete (for LRK[i+2])

        # cluster 3 MEM2: LRK[i+2] then ACV[i+2].
        with tlx.warp_pipeline_stage("mem2", priority=1):
            kt_dot = tlx.local_load(
                tlx.local_trans(tlx.local_view(k_buf, k_rd_slot)), relaxed=True
            )  # LRK s2
            tok_v = tlx.async_load(
                v_ptrs + acv_n * stride_vn, tlx.local_view(v_buf, acv_slot)
            )
            tlx.async_load_commit_group([tok_v])  # ACV s2

    # -- Drain: last 3 output tiles, no OOB global prefetch ----------------
    nm3 = block_end - 3
    nm2 = block_end - 2
    nm1 = block_end - 1
    v_s_nm3 = (nm3 - block_start) % V_DEPTH
    v_s_nm2 = (nm2 - block_start) % V_DEPTH
    v_s_nm1 = (nm1 - block_start) % V_DEPTH
    k_s_nm1 = (nm1 - block_start) % K_DEPTH

    # output tile n-3 (also issues the final V prefetch, ACV[n-1])
    qk = tl.dot(q, kt_dot)  # dot_qk[n-2]
    tlx.async_load_wait_group(2)  # V[n-3] complete
    v_dot = tlx.local_load(tlx.local_view(v_buf, v_s_nm3), relaxed=True)  # LRV[n-3]
    state, p_dot = state.vec2(p_c, alpha_c, q.dtype)  # VEC2[n-3]
    acc = tl.dot(p_dot, v_dot, state.acc)  # dot_pv[n-3]
    state = SoftmaxState(acc, state.l_i, state.m_i)
    state, p_c, alpha_c = state.vec1(qk, qk_scale)  # VEC1[n-2]
    tl.debug_barrier()  # WAR: LRV[n-3] vs V[n-1] write
    tok_vlast = tlx.async_load(
        v_ptrs + nm1 * BLOCK_N * stride_vn, tlx.local_view(v_buf, v_s_nm1)
    )
    tlx.async_load_commit_group([tok_vlast])  # ACV[n-1]
    tlx.async_load_wait_group(2)  # K[n-1] complete
    kt_dot = tlx.local_load(
        tlx.local_trans(tlx.local_view(k_buf, k_s_nm1)), relaxed=True
    )  # LRK[n-1]

    # output tile n-2
    qk = tl.dot(q, kt_dot)  # dot_qk[n-1]
    tlx.async_load_wait_group(1)  # V[n-2] complete
    v_dot = tlx.local_load(tlx.local_view(v_buf, v_s_nm2), relaxed=True)  # LRV[n-2]
    state, p_dot = state.vec2(p_c, alpha_c, q.dtype)  # VEC2[n-2]
    acc = tl.dot(p_dot, v_dot, state.acc)  # dot_pv[n-2]
    state = SoftmaxState(acc, state.l_i, state.m_i)
    state, p_c, alpha_c = state.vec1(qk, qk_scale)  # VEC1[n-1]

    # output tile n-1 (final; no further dot_qk / prefetch)
    tlx.async_load_wait_group(0)  # V[n-1] complete
    v_dot = tlx.local_load(tlx.local_view(v_buf, v_s_nm1), relaxed=True)  # LRV[n-1]
    state, p_dot = state.vec2(p_c, alpha_c, q.dtype)  # VEC2[n-1]
    acc = tl.dot(p_dot, v_dot, state.acc)  # dot_pv[n-1]
    state = SoftmaxState(acc, state.l_i, state.m_i)

    return state


@triton.jit  # pragma: no cover
def _attn_fwd_jagged_tlx_cluster_pipeline(
    query,
    q_offsets,
    key,
    k_offsets,
    value,
    out,
    ad_to_request_offset,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    qk_scale,
    d_head,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_DEPTH: tl.constexpr,
    V_DEPTH: tl.constexpr,
):
    """Rotated 4-cluster FA forward with 2D jaggedness. Non-causal, m-block on
    grid axis 0 (every tile is equal cost). The deep pipeline runs over the
    fully-populated K/V tiles; a boundary-masked step finishes the ragged tail."""
    pid_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    begin_q = tl.load(q_offsets + off_z)
    end_q = tl.load(q_offsets + off_z + 1)
    q_seq_len = end_q - begin_q

    if pid_m * BLOCK_M >= q_seq_len:
        return

    off_zkv = tl.load(ad_to_request_offset + off_z)
    begin_k = tl.load(k_offsets + off_zkv)
    end_k = tl.load(k_offsets + off_zkv + 1)
    kv_seq_len = end_k - begin_k

    q_head_off = off_h.to(tl.int64) * stride_qh
    kv_head_off = off_h.to(tl.int64) * stride_kh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        query
        + q_head_off
        + (begin_q + offs_m[:, None]) * stride_qm
        + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < q_seq_len,
        other=0.0,
    )

    k_ptrs = (
        key
        + kv_head_off
        + (begin_k + offs_n[:, None]) * stride_kn
        + offs_d[None, :] * stride_kd
    )
    v_ptrs = (
        value
        + kv_head_off
        + (begin_k + offs_n[:, None]) * stride_vn
        + offs_d[None, :] * stride_vd
    )

    state = SoftmaxState.create(BLOCK_M, BLOCK_D)

    k_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), key.dtype.element_ty, K_DEPTH)
    v_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), value.dtype.element_ty, V_DEPTH)

    n_blocks = tl.cdiv(kv_seq_len, BLOCK_N)
    n_full = kv_seq_len // BLOCK_N  # number of fully-populated K/V tiles

    # Deep rotated pipeline over the mask-free full tiles (needs >= 4 tiles).
    if n_full >= 4:
        state = _jagged_attn_inner_pipelined(
            state,
            q,
            k_ptrs,
            v_ptrs,
            0,
            n_full,
            k_buf,
            v_buf,
            stride_kn,
            stride_vn,
            qk_scale,
            BLOCK_N,
            K_DEPTH,
            V_DEPTH,
        )
        masked_start = n_full
    else:
        masked_start = 0

    # Boundary-masked remainder: the ragged tail (and all tiles when n_full < 4).
    # Uses synchronous loads (tiny, not perf-critical) to avoid LDS WAR hazards.
    for blk in tl.range(masked_start, n_blocks, num_stages=0):
        start_n = blk * BLOCK_N
        kn = start_n + offs_n
        kmask = kn[:, None] < kv_seq_len
        k = tl.load(k_ptrs + start_n * stride_kn, mask=kmask, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=kmask, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(kn[None, :] < kv_seq_len, qk, -1.0e10)
        m_ij = tl.maximum(state.m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        alpha = tl.math.exp2(state.m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = state.acc * alpha[:, None]
        l_i = state.l_i * alpha + l_ij
        acc = tl.dot(p.to(v.dtype), v, acc)
        state = SoftmaxState(acc, l_i, m_ij)

    acc = state.acc / state.l_i[:, None]
    o_ptrs = (
        out
        + off_h.to(tl.int64) * stride_oh
        + (begin_q + offs_m[:, None]) * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(
        o_ptrs,
        acc.to(out.dtype.element_ty),
        mask=(offs_m[:, None] < q_seq_len) & (offs_d[None, :] < d_head),
    )


def tlx_jagged_flash_attn_ikbo_cluster_pipeline(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    ad_to_request_mapping: torch.Tensor,
    max_seq_len: int,  # Maximum sequence length of queries
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Rotated 4-cluster warp-pipeline FA forward, jagged IKBO variant (gfx950).

    Same interface/semantics as tlx_jagged_flash_attn_ikbo. Uses a fixed anchor
    tile (BLOCK_M=256, num_warps=8) and the deep warp-pipeline schedule over the
    fully-populated K/V tiles, with a boundary-masked ragged-tail finisher.
    Requires an AMD CDNA4 (gfx950 / MI350) GPU for `tlx.warp_pipeline_stage`.
    """
    d_head = query.shape[-1]
    BLOCK_D = triton.next_power_of_2(d_head)

    sm_scale = scale
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d_head)
    qk_scale = sm_scale / math.log(2.0)

    output = torch.empty_like(query)

    nheads = query.shape[1]
    BATCH = query_offset.size(0) - 1

    # Fixed anchor tile (matches the validated cluster-pipeline config). Each
    # warp owns BLOCK_M // num_warps dot rows, which must be >= the 32-row CDNA4
    # MFMA tile, so num_warps is capped accordingly (8 at BLOCK_M=256).
    MFMA_M = 32
    BLOCK_M = 256
    # Shorter jagged kv (vs the reference's long dense N) amortizes the deep
    # pipeline's 3-tile prologue/drain better with a smaller BLOCK_N (more steady
    # iterations, and fewer heads fall under the <4-tile threshold). Env-tunable.
    BLOCK_N = int(os.environ.get("IKBO_CLUSTER_BLOCK_N", "32"))
    # Asymmetric LDS ring depth. Tile-absolute slot indexing keeps any
    # K_DEPTH>=3 / V_DEPTH>=2 correct (deeper only adds WAR slack for the async
    # prefetch). At D=128/BLOCK_N=32 a slot is only ~8KB, so the 160KB gfx950
    # budget leaves ample room to deepen the rings. Env-tunable.
    K_DEPTH = int(os.environ.get("IKBO_CLUSTER_K_DEPTH", "3"))
    V_DEPTH = int(os.environ.get("IKBO_CLUSTER_V_DEPTH", "2"))
    num_warps = min(8, max(1, BLOCK_M // MFMA_M))
    # Pin occupancy target: 2 waves/EU keeps register pressure in check for this
    # deep-pipeline anchor (matches the non-causal reference). Env-tunable.
    waves_per_eu = int(os.environ.get("IKBO_CLUSTER_WAVES_PER_EU", "2"))

    m_blocks = triton.cdiv(max_seq_len, BLOCK_M)
    # Non-causal: m-block on axis 0 (every tile equal cost), head on axis 1,
    # ad-batch on axis 2.
    grid = (m_blocks, nheads, BATCH)

    _attn_fwd_jagged_tlx_cluster_pipeline[grid](
        query,
        query_offset,
        key,
        key_offset,
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
        d_head=d_head,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        K_DEPTH=K_DEPTH,
        V_DEPTH=V_DEPTH,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    return output


# ═══════════════════════════════════════════════════════════════════════════
# Input generation + jagged reference — inlined from
# ikbo_jagged_flash_attention_bench.py
# ═══════════════════════════════════════════════════════════════════════════


def pytorch_sdpa(quer, key, value):
    return torch.nn.functional.scaled_dot_product_attention(
        quer, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    )


def pytorch_jagged_sdpa(
    query, key, value, query_offset, key_offset, ad_to_user_mapping, B, H, d_head
):
    """
    Correct jagged reference: each ad attends only over its user's real key
    range (no padding), so this is a valid ground truth for the jagged kernel
    (unlike pytorch_padded_sdpa, which attends over the zero-padding).

    query: [sum(q_seq_len), H, d_head] packed over ads
    key/value: [sum(kv_seq_len), H, d_head] packed over users (jagged)
    query_offset: [B + 1] per-ad cumulative query offsets
    key_offset: [Bu + 1] per-user cumulative key offsets
    ad_to_user_mapping: [B] ad batch id -> user batch id
    """
    q_off = query_offset.tolist()
    k_off = key_offset.tolist()
    ad2u = ad_to_user_mapping.tolist()
    output = torch.empty_like(query)

    for i in range(B):
        q_start, q_end = q_off[i], q_off[i + 1]
        u = ad2u[i]
        k_start, k_end = k_off[u], k_off[u + 1]  # real per-user key range only

        # [seq, H, d_head] -> [H, seq, d_head] for SDPA (heads as batch dim)
        q = query[q_start:q_end].permute(1, 0, 2)
        k = key[k_start:k_end].permute(1, 0, 2)
        v = value[k_start:k_end].permute(1, 0, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )  # [H, q_seq_len, d_head]
        output[q_start:q_end] = out.permute(1, 0, 2)  # back to [q_seq_len, H, d_head]
    return output


def pytorch_padded_sdpa(
    query, key, value, ad_to_user_mapping, B, q_seq_len, H, d_head, max_seq_len
):
    query_sdpa = query.view(B, q_seq_len, H, d_head).permute(0, 2, 1, 3)
    key_sdpa = key.view(-1, max_seq_len, H, d_head)
    key_sdpa_broadcast = torch.index_select(
        key_sdpa, dim=0, index=ad_to_user_mapping
    ).permute(0, 2, 1, 3)
    value_sdpa = value.view(-1, max_seq_len, H, d_head)
    value_sdpa_broadcast = torch.index_select(
        value_sdpa, dim=0, index=ad_to_user_mapping
    ).permute(0, 2, 1, 3)
    return pytorch_sdpa(query_sdpa, key_sdpa_broadcast, value_sdpa_broadcast)


def _generate_num_ads_per_user(
    low: int,
    high: int,
    max_threshold: int,
    seed: int = 2,
) -> list[int]:
    """
    Generate list of int, corresponding to number of ads for every
    request.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    res = []
    cum_sum = 0
    while True:
        # Checking num of ads distribution for request.
        # It is roughly conform even distribution.
        cur = random.randint(low, high)
        if cum_sum + cur == max_threshold:
            res.append(cur)
            break
        if cum_sum + cur >= max_threshold:
            res.append(max_threshold - cum_sum)
            break
        cum_sum += cur
        res.append(cur)

    return res


def _generate_variable_kv_seq_len(
    low: int,
    high: int,
    Bu: int,
    alignment: int = 16,
    seed: int = 2,
):
    """
    Generate list of int, corresponding to kv sequence length for every
    request.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    kv_seq_list = []
    k_offset_list = [0]
    total_seq_len = 0
    for _ in range(Bu):
        cur = random.randint(low, high)
        # Applied local padding to the sequence to ensure the inference memory alignment
        cur_aligned = ((cur + alignment - 1) // alignment) * alignment
        kv_seq_list.append(cur_aligned)
        total_seq_len += cur_aligned
        k_offset_list.append(total_seq_len)
    return kv_seq_list, k_offset_list, total_seq_len


def generate_ikbo_jagged_flash_attention_inputs(
    low_num_ads_per_req: int,
    high_num_ads_per_req: int,
    B: int,  # batch size for ads
    H: int = 1,  # number of heads
    d_head: int = 128,  # embedding dimension
    q_seq_len: int = 32,  # Query sequence length
    min_kv_seq_len: int = 200,  # minimum key and value sequence length
    max_kv_seq_len: int = 1000,  # maximum key and value sequence length
    dtype: torch.dtype = torch.float16,
    device=DEVICE,
    debug: bool = False,
    seed: int = 2,
):
    torch.manual_seed(seed)
    # Prepare input user ads mapping
    num_ads_per_user = torch.tensor(
        _generate_num_ads_per_user(
            low=low_num_ads_per_req,
            high=high_num_ads_per_req,
            max_threshold=B,
            seed=seed,
        )
    )
    Bu: int = num_ads_per_user.size(0)
    # Prepare query and key offset list (variable sequence length)
    kv_seq_list, key_offset_list, total_kv_seq_len = _generate_variable_kv_seq_len(
        low=min_kv_seq_len,
        high=max_kv_seq_len,
        Bu=Bu,
        alignment=1,
        seed=seed,
    )
    if debug:
        print(f"Bu is {Bu} and kv seqence length list is {kv_seq_list}", flush=True)
        print(f"kv sequence offset list is {key_offset_list}", flush=True)
    ad_to_user_mapping = (
        torch.repeat_interleave(
            torch.arange(num_ads_per_user.size(0)),
            num_ads_per_user,
        )
        .int()
        .to(device)
    )

    key_list = []
    value_list = []
    for i_usr in range(Bu):
        kv_seq_len = kv_seq_list[i_usr]
        key_cur = torch.randn((kv_seq_len, H, d_head), device=device, dtype=dtype)
        key_list.append(key_cur)
        value_cur = torch.randn((kv_seq_len, H, d_head), device=device, dtype=dtype)
        value_list.append(value_cur)

    query = torch.randn((B * q_seq_len, H, d_head), device=device, dtype=dtype)
    key = torch.cat(key_list, dim=0)
    value = torch.cat(value_list, dim=0)
    query_offset = torch.arange(B + 1, device=device, dtype=torch.int32) * q_seq_len
    key_offset = torch.tensor(key_offset_list, device=device, dtype=torch.int32)

    # Generate padded key and value data to max_kev_seq_len
    key_padded = torch.zeros(
        (Bu * max_kv_seq_len, H, d_head), device=device, dtype=dtype
    )
    value_padded = torch.zeros(
        (Bu * max_kv_seq_len, H, d_head), device=device, dtype=dtype
    )
    for i_usr in range(Bu):
        kv_seq_len = kv_seq_list[i_usr]
        kv_seq_start = key_offset_list[i_usr]
        kv_seq_end = key_offset_list[i_usr + 1]
        key_padded[(i_usr * max_kv_seq_len) : (i_usr * max_kv_seq_len + kv_seq_len)] = (
            key[kv_seq_start:kv_seq_end]
        )
        value_padded[
            (i_usr * max_kv_seq_len) : (i_usr * max_kv_seq_len + kv_seq_len)
        ] = value[kv_seq_start:kv_seq_end]

    return (
        query,
        key,
        value,
        ad_to_user_mapping,
        query_offset,
        key_offset,
        kv_seq_list,
        key_padded,
        value_padded,
    )


def measure_tflops(
    ms, B, H, q_seq_len, kv_seq_list, d_head, ads_to_user_mapping, causal=False
):
    total_flops = 0
    ads_to_user_map = ads_to_user_mapping.tolist()
    for iBatch in range(B):
        usrBatch = ads_to_user_map[iBatch]
        kv_seq_len = kv_seq_list[usrBatch]
        valid_el = (
            kv_seq_len * (kv_seq_len + 1) // 2 if causal else q_seq_len * kv_seq_len
        )
        total_flops += 2 * 2.0 * H * valid_el * d_head
    return total_flops / ms * 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# Kernel registry + calling convention (all jagged kernels share one signature)
# ═══════════════════════════════════════════════════════════════════════════

KERNEL_REGISTRY = {
    "jagged": tlx_jagged_flash_attn_ikbo,
    "jagged_persistent": tlx_jagged_flash_attn_ikbo_persistent,
    "jagged_cluster_pipeline": tlx_jagged_flash_attn_ikbo_cluster_pipeline,
}


def get_kernel(name):
    if name not in KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown kernel: {name!r}. Available: {list(KERNEL_REGISTRY.keys())}"
        )
    return KERNEL_REGISTRY[name]


def make_kernel_call(
    name, query, key, value, query_offset, key_offset, mapping, q_seq_len
):
    """Return a zero-arg `call()` that runs only the kernel (for timing). Output is
    the packed [sum(q_seq_len), H, d_head] layout, same as pytorch_jagged_sdpa."""
    fn = get_kernel(name)

    def call():
        return fn(query, key, value, query_offset, key_offset, mapping, q_seq_len)

    return call


# ═══════════════════════════════════════════════════════════════════════════
# Verification + summary table
# ═══════════════════════════════════════════════════════════════════════════


def verify(name, got, ref, atol=2e-2, rtol=2e-2, log=True):
    diff = (got.float() - ref.float()).abs()
    ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
    if log:
        status = "PASS" if ok else "FAIL"
        print(
            f"  {name:<32} {status}  max={diff.max().item():.6f}  mean={diff.mean().item():.6f}"
        )
    return ok


def print_summary_table(results, providers):
    rows = []
    for key in sorted(results.keys()):
        B, H, D, q_seq_len, min_kv, max_kv = key
        rows.append(
            (
                f"B={B}, H={H}, D={D}, q={q_seq_len}, kv=[{min_kv},{max_kv}]",
                results[key],
            )
        )

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
# Benchmark / correctness drivers
# ═══════════════════════════════════════════════════════════════════════════


def _make_inputs(args, B, H, D, q_seq_len, min_kv, max_kv, dtype):
    random.seed(2)
    torch.manual_seed(2)
    (
        query,
        key,
        value,
        mapping,
        query_offset,
        key_offset,
        kv_seq_list,
        _key_padded,
        _value_padded,
    ) = generate_ikbo_jagged_flash_attention_inputs(
        low_num_ads_per_req=args.low,
        high_num_ads_per_req=args.high,
        B=B,
        H=H,
        d_head=D,
        q_seq_len=q_seq_len,
        min_kv_seq_len=min_kv,
        max_kv_seq_len=max_kv,
        dtype=dtype,
    )
    return query, key, value, mapping, query_offset, key_offset, kv_seq_list


def run_benchmark(args):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    results = {}
    ref_name = "Torch Jagged SDPA"
    providers = ([ref_name] if args.time_ref else []) + list(args.kernel)

    for B in args.b:
        for H in args.hq:
            for D in args.d:
                for q_seq_len in args.nseed:
                    for min_kv, max_kv in zip(args.min_kv, args.max_kv):
                        query, key, value, mapping, q_off, k_off, kv_list = (
                            _make_inputs(
                                args, B, H, D, q_seq_len, min_kv, max_kv, dtype
                            )
                        )
                        ref = pytorch_jagged_sdpa(
                            query, key, value, q_off, k_off, mapping, B, H, D
                        )

                        key_id = (B, H, D, q_seq_len, min_kv, max_kv)
                        results.setdefault(key_id, {})

                        def _tflops(ms):
                            return measure_tflops(
                                ms, B, H, q_seq_len, kv_list, D, mapping, causal=False
                            )

                        if args.time_ref and ref_name not in results[key_id]:
                            ref_fn = lambda: pytorch_jagged_sdpa(  # noqa: E731
                                query, key, value, q_off, k_off, mapping, B, H, D
                            )
                            ms = triton.testing.do_bench(ref_fn, warmup=2, rep=5)
                            results[key_id][ref_name] = {
                                "ms": ms,
                                "tflops": _tflops(ms),
                            }

                        for kernel_name in args.kernel:
                            tag = f"{kernel_name} B={B} H={H} D={D} q={q_seq_len} kv=[{min_kv},{max_kv}]"
                            try:
                                call = make_kernel_call(
                                    kernel_name,
                                    query,
                                    key,
                                    value,
                                    q_off,
                                    k_off,
                                    mapping,
                                    q_seq_len,
                                )
                                out = call()
                                if not verify(
                                    "", out, ref, args.atol, args.rtol, log=False
                                ):
                                    print(f"  {tag:60s} -> SKIPPED (correctness)")
                                    continue
                                ms = triton.testing.do_bench(call, warmup=25, rep=100)
                            except Exception as e:
                                print(f"  {tag:60s} -> SKIPPED ({e})")
                                continue
                            results[key_id][kernel_name] = {
                                "ms": ms,
                                "tflops": _tflops(ms),
                            }

    print_summary_table(results, providers)


def run_correctness(args):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    all_ok = True
    for B in args.b:
        for H in args.hq:
            for D in args.d:
                for q_seq_len in args.nseed:
                    for min_kv, max_kv in zip(args.min_kv, args.max_kv):
                        query, key, value, mapping, q_off, k_off, _kv_list = (
                            _make_inputs(
                                args, B, H, D, q_seq_len, min_kv, max_kv, dtype
                            )
                        )
                        ref = pytorch_jagged_sdpa(
                            query, key, value, q_off, k_off, mapping, B, H, D
                        )
                        print(f"B={B} H={H} D={D} q={q_seq_len} kv=[{min_kv},{max_kv}]")
                        for kernel_name in args.kernel:
                            try:
                                call = make_kernel_call(
                                    kernel_name,
                                    query,
                                    key,
                                    value,
                                    q_off,
                                    k_off,
                                    mapping,
                                    q_seq_len,
                                )
                                ok = verify(
                                    kernel_name, call(), ref, args.atol, args.rtol
                                )
                            except Exception as e:
                                ok = False
                                print(f"  {kernel_name:<32} SKIPPED ({e})")
                            all_ok &= ok
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# Pytest correctness (vs the jagged PyTorch SDPA reference)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("kernel_name", list(KERNEL_REGISTRY))
def test_ikbo_jagged_fa_correctness(
    kernel_name, B=1024, H=1, q_seq_len=256, D=128, min_kv=400, max_kv=2000
):
    """Each jagged IKBO kernel vs the per-ad PyTorch jagged SDPA reference. Shape
    matches ikbo_jagged_flash_attention_test.py's large-batch case."""
    random.seed(2)
    torch.manual_seed(2)
    (
        query,
        key,
        value,
        mapping,
        query_offset,
        key_offset,
        _kv_list,
        _kp,
        _vp,
    ) = generate_ikbo_jagged_flash_attention_inputs(
        low_num_ads_per_req=30,
        high_num_ads_per_req=40,
        B=B,
        H=H,
        d_head=D,
        q_seq_len=q_seq_len,
        min_kv_seq_len=min_kv,
        max_kv_seq_len=max_kv,
    )
    ref = pytorch_jagged_sdpa(
        query, key, value, query_offset, key_offset, mapping, B, H, D
    )
    call = make_kernel_call(
        kernel_name, query, key, value, query_offset, key_offset, mapping, q_seq_len
    )
    assert verify(kernel_name, call(), ref, atol=1e-2, rtol=1e-2), (
        f"correctness failed: {kernel_name}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(prog="AMD TLX IKBO Jagged FA Pipelined")
    p.add_argument(
        "-b", type=int, nargs="+", default=[1024, 2048], help="ads batch sizes"
    )
    p.add_argument("-hq", type=int, nargs="+", default=[1], help="num heads")
    p.add_argument(
        "-nseed", type=int, nargs="+", default=[256], help="query seq length (# seeds)"
    )
    p.add_argument(
        "--min-kv", type=int, nargs="+", default=[400], help="min kv seq length(s)"
    )
    p.add_argument(
        "--max-kv", type=int, nargs="+", default=[2000], help="max kv seq length(s)"
    )
    p.add_argument("-d", type=int, nargs="+", default=[128], help="d_head")
    p.add_argument("--low", type=int, default=30, help="min ads per user")
    p.add_argument("--high", type=int, default=40, help="max ads per user")
    p.add_argument("--dtype", type=str, default="fp16", choices=["bf16", "fp16"])
    p.add_argument("--atol", type=float, default=2e-2, help="allclose atol")
    p.add_argument("--rtol", type=float, default=2e-2, help="allclose rtol")
    p.add_argument(
        "--time-ref",
        action="store_true",
        help="also time the (slow) per-ad PyTorch jagged SDPA reference",
    )
    p.add_argument(
        "--kernel",
        type=str,
        nargs="+",
        default=list(KERNEL_REGISTRY),
        choices=list(KERNEL_REGISTRY),
        help="jagged IKBO kernel variants",
    )
    p.add_argument("--mode", choices=["benchmark", "correctness"], default="benchmark")
    args = p.parse_args()
    assert len(args.min_kv) == len(args.max_kv), (
        "--min-kv and --max-kv must have the same number of values (paired)"
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "correctness":
        run_correctness(args)
    else:
        run_benchmark(args)
