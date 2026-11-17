# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parallel softmax backward schedules for small gfx950 batches.

Shared runtime JIT dispatch preserves specialization and runtime hooks.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from ._backward_pipeline import _tlx_gfx950_cross_attn_v3_ttgir_bwd


@triton.jit
def _small_bwd_impl(Q, K, V, DO, M, OUT, SOQ, SOK, DQ, DK, DV, alpha, H: tl.constexpr, KV_TILES: tl.constexpr,
                    SHARED: tl.constexpr, KV_FIRST: tl.constexpr = False):
    BLOCK: tl.constexpr = 64
    HEAD_DIM: tl.constexpr = 128
    QUERY_TILES: tl.constexpr = 4
    head_batch = tl.program_id(0)
    task = tl.program_id(1)
    if KV_FIRST:
        task = (task + QUERY_TILES) % (QUERY_TILES + KV_TILES)
    batch = head_batch // H
    head = head_batch % H
    d = tl.arange(0, HEAD_DIM)
    q_start = tl.load(SOQ + batch).to(tl.int64)
    q_end = tl.load(SOQ + batch + 1).to(tl.int64)
    k_start = tl.load(SOK + batch).to(tl.int64)
    k_end = tl.load(SOK + batch + 1).to(tl.int64)

    if task < QUERY_TILES:
        if k_end - k_start != 1:
            rows = q_start + task * BLOCK + tl.arange(0, BLOCK)
            q_mask = rows < q_end
            q_offsets = (rows[:, None] * H + head) * HEAD_DIM + d[None, :]
            q = tl.load(Q + q_offsets, mask=q_mask[:, None], other=0)
            do = tl.load(DO + q_offsets, mask=q_mask[:, None], other=0)
            out = tl.load(OUT + q_offsets, mask=q_mask[:, None], other=0)
            m = tl.load(M + rows * H + head, mask=q_mask, other=0)
            delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
            dq = tl.full((BLOCK, HEAD_DIM), 0, tl.float32)
            for tile in range(KV_TILES):
                columns = k_start + tile * BLOCK + tl.arange(0, BLOCK)
                k_mask = columns < k_end
                k_offsets = (columns[:, None] * H + head) * HEAD_DIM + d[None, :]
                k = tl.load(K + k_offsets, mask=k_mask[:, None], other=0)
                if SHARED:
                    v = k
                else:
                    v = tl.load(V + k_offsets, mask=k_mask[:, None], other=0)
                scores = tl.dot(q, tl.trans(k), allow_tf32=False)
                p = tl.exp2(scores * (alpha * 1.44269504) - m[:, None])
                dp = tl.dot(do, tl.trans(v), allow_tf32=False)
                ds = tl.where(q_mask[:, None] & k_mask[None, :], p * (dp - delta[:, None]), 0)
                dq = tl.dot(ds.to(tl.bfloat16), k, dq, allow_tf32=False)
            tl.store(DQ + q_offsets, dq * alpha, mask=q_mask[:, None])
        else:
            rows = q_start + task * BLOCK + tl.arange(0, BLOCK)
            offsets = (rows[:, None] * H + head) * HEAD_DIM + d[None, :]
            tl.store(DQ + offsets, 0.0, mask=(rows < q_end)[:, None])
    else:
        if k_end - k_start != 1:
            columns = k_start + (task - QUERY_TILES) * BLOCK + tl.arange(0, BLOCK)
            k_mask = columns < k_end
            k_offsets = (columns[:, None] * H + head) * HEAD_DIM + d[None, :]
            k = tl.load(K + k_offsets, mask=k_mask[:, None], other=0)
            if SHARED:
                v = k
            else:
                v = tl.load(V + k_offsets, mask=k_mask[:, None], other=0)
            dk = tl.full((BLOCK, HEAD_DIM), 0, tl.float32)
            dv = tl.full((BLOCK, HEAD_DIM), 0, tl.float32)
            for tile in range(QUERY_TILES):
                rows = q_start + tile * BLOCK + tl.arange(0, BLOCK)
                q_mask = rows < q_end
                q_offsets = (rows[:, None] * H + head) * HEAD_DIM + d[None, :]
                q = tl.load(Q + q_offsets, mask=q_mask[:, None], other=0)
                do = tl.load(DO + q_offsets, mask=q_mask[:, None], other=0)
                out = tl.load(OUT + q_offsets, mask=q_mask[:, None], other=0)
                m = tl.load(M + rows * H + head, mask=q_mask, other=0)
                delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
                valid = k_mask[:, None] & q_mask[None, :]
                scores = tl.dot(k, tl.trans(q), allow_tf32=False)
                p = tl.exp2(scores * (alpha * 1.44269504) - m[None, :])
                p = tl.where(valid, p, 0)
                dp = tl.dot(v, tl.trans(do), allow_tf32=False)
                ds = tl.where(valid, p * (dp - delta[None, :]), 0)
                dv = tl.dot(p.to(tl.bfloat16), do, dv, allow_tf32=False)
                dk = tl.dot(ds.to(tl.bfloat16), q, dk, allow_tf32=False)
            dk = dk * alpha
            if SHARED:
                dk += dv
            tl.store(DK + k_offsets, dk, mask=k_mask[:, None])
            if not SHARED:
                tl.store(DV + k_offsets, dv, mask=k_mask[:, None])
        elif task == QUERY_TILES:
            value_gradient = tl.full((HEAD_DIM, ), 0, tl.float32)
            for tile in range(QUERY_TILES):
                rows = q_start + tile * BLOCK + tl.arange(0, BLOCK)
                offsets = (rows[:, None] * H + head) * HEAD_DIM + d[None, :]
                values = tl.load(DO + offsets, mask=(rows < q_end)[:, None], other=0).to(tl.float32)
                value_gradient += tl.sum(values, axis=0)
            offsets = (k_start * H + head) * HEAD_DIM + d
            if SHARED:
                tl.store(DK + offsets, value_gradient)
            else:
                tl.store(DK + offsets, 0.0)
                tl.store(DV + offsets, value_gradient)


@triton.jit
def _small_bwd_shared(Q, K, DO, M, OUT, SOQ, SOK, DQ, DK, alpha, H: tl.constexpr, KV_TILES: tl.constexpr):
    _small_bwd_impl(Q, K, K, DO, M, OUT, SOQ, SOK, DQ, DK, DK, alpha, H, KV_TILES, True)


@triton.jit
def _small_bwd_shared_kv_first(Q, K, DO, M, OUT, SOQ, SOK, DQ, DK, alpha, H: tl.constexpr, KV_TILES: tl.constexpr):
    _small_bwd_impl(Q, K, K, DO, M, OUT, SOQ, SOK, DQ, DK, DK, alpha, H, KV_TILES, True, True)


@triton.jit
def _small_bwd_separate(Q, K, V, DO, M, OUT, SOQ, SOK, DQ, DK, DV, alpha, H: tl.constexpr, KV_TILES: tl.constexpr):
    _small_bwd_impl(Q, K, V, DO, M, OUT, SOQ, SOK, DQ, DK, DV, alpha, H, KV_TILES, False)


_SMALL_OPTIONS = dict(num_warps=4, num_stages=1, matrix_instr_nonkdim=16, waves_per_eu=1)


def small_softmax_backward(q, k, v, dout, m, out, offsets_q, offsets_kv, max_kv, alpha, shared_kv):
    """Compute gradients for contiguous BF16 inputs with Q <= 256 and D = 128."""
    dout = dout.contiguous()
    heads = q.shape[1]
    kv_tiles = triton.cdiv(max_kv, 64)
    grid = ((offsets_q.numel() - 1) * heads, 4 + kv_tiles, 1)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    if shared_kv:
        args = (q, k, dout, m, out, offsets_q, offsets_kv, dq, dk, alpha, heads, kv_tiles)
        kernel = _small_bwd_shared_kv_first if heads == 1 and grid[0] == 64 and max_kv == 128 else _small_bwd_shared
        kernel[grid](*args, **_SMALL_OPTIONS)
        dv = None
    else:
        dv = torch.empty_like(v)
        args = (q, k, v, dout, m, out, offsets_q, offsets_kv, dq, dk, dv, alpha, heads, kv_tiles)
        _small_bwd_separate[grid](*args, **_SMALL_OPTIONS)
    return dq, dk, dv


@triton.jit
def _coarse_bwd(Q, K, DO, SOQ, SOK, DQ, DK, M, OUT, TQ, SQ, SK, SDO, SDQ, SDK, alpha):
    _tlx_gfx950_cross_attn_v3_ttgir_bwd(Q, K, DO, TQ, SOK, SOQ, DQ, DK, SQ, SK, SDO, SDQ, SDK, alpha, M, 0,
                                        PIPELINE_QDO=True, DQ_FINAL=None, OUT=OUT, KV_SPLITS=4)


@triton.jit
def _coarse_bwd_cached(Q, K, DO, SOQ, SOK, DQ, DK, M, OUT, TQ, SQ, SK, SDO, SDQ, SDK, alpha):
    _tlx_gfx950_cross_attn_v3_ttgir_bwd(Q, K, DO, TQ, SOK, SOQ, DQ, DK, SQ, SK, SDO, SDQ, SDK, alpha, M, 0,
                                        PIPELINE_QDO=True, DQ_FINAL=None, OUT=OUT, KV_SPLITS=4, CACHE_DELTA=True)


@triton.jit
def _reduce_partial_dq(PARTIAL, DQ, N, DO, DK, SOQ, SOK):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    qs = tl.load(SOQ + batch).to(tl.int64)
    qe = tl.load(SOQ + batch + 1).to(tl.int64)
    ks = tl.load(SOK + batch).to(tl.int64)
    ke = tl.load(SOK + batch + 1).to(tl.int64)
    offsets = qs * 128 + tile * 2048 + tl.arange(0, 2048)
    mask = offsets < qe * 128
    if ke - ks != 1:
        value = tl.full((2048, ), 0, tl.float32)
        for split in tl.static_range(4):
            value += tl.load(PARTIAL + split * N.to(tl.int64) + offsets, mask, other=0)
        tl.store(DQ + offsets, value, mask)
    else:
        tl.store(DQ + offsets, 0., mask)
        if tile == 0:
            d = tl.arange(0, 128)
            value = tl.full((128, ), 0., tl.float32)
            for start in range(0, qe - qs, 32):
                rows = qs + start + tl.arange(0, 32)
                do = tl.load(DO + rows[:, None] * 128 + d[None, :], rows[:, None] < qe, 0)
                value += tl.sum(do.to(tl.float32), 0)
            tl.store(DK + ks * 128 + d, value)


_COARSE_OPTIONS = dict(_SMALL_OPTIONS, regclass_priority_trumps_globalness=False)


def coarse_softmax_backward(q, k, dout, m, out, offsets_q, offsets_kv, alpha, cache_delta=False):
    """Split shared-K/V sequences into four disjoint blocks of KV tiles."""
    dout = dout.contiguous()
    tokens = q.shape[0]
    partial = torch.empty((4, tokens, 1, 128), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    grid = (offsets_q.numel() - 1, 4, 1)
    args = (q, k, dout, offsets_q, offsets_kv, partial, dk, m, out, tokens, q.stride(0), k.stride(0), dout.stride(0),
            128, 128, alpha)
    if cache_delta:
        kernel = _coarse_bwd_cached
    else:
        kernel = _coarse_bwd
    kernel[grid](*args, **_COARSE_OPTIONS)
    _reduce_partial_dq[(offsets_q.numel() - 1, 16, 1)](partial, dq, q.numel(), dout, dk, offsets_q, offsets_kv)
    return dq, dk, None


# Separate helpers keep the Q16 and KV32 tensor layouts independent.
@triton.jit
def _compact_q_owner_staged(Q, K, DO, M, OUT, DQ, qs, qe, ks, ke, head, task, alpha, H: tl.constexpr,
                            MAX_KV: tl.constexpr, BQ: tl.constexpr, KL: tl.constexpr):
    rows = qs + task * BQ + tl.arange(0, BQ)
    d = tl.arange(0, 128)
    qm = rows < qe
    qo = (rows[:, None] * H + head) * 128 + d[None, :]
    q = tl.load(Q + qo, qm[:, None], other=0)
    do = tl.load(DO + qo, qm[:, None], other=0)
    out = tl.load(OUT + qo, qm[:, None], other=0)
    m = tl.load(M + rows * H + head, qm, other=0)
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), 1)
    # Distribute score columns across all selected warps.
    score_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                     warps_per_cta=([1, 8] if H == 2 and MAX_KV == 1024 else [1, 4]))
    lhs_layout: tl.constexpr = tlx.dot_operand_layout(0, score_layout, k_width=4)
    rhs_layout: tl.constexpr = tlx.dot_operand_layout(1, score_layout, k_width=4)
    q_fragment = tlx.require_layout(q, lhs_layout, pin=False)
    do_fragment = tlx.require_layout(do, lhs_layout, pin=False)
    # A single softmax key has a zero score derivative.
    if H == 2 and MAX_KV == 1024:
        dq = tlx.zeros((BQ, 128), tl.float32, layout=score_layout)
    else:
        dq = tl.full((BQ, 128), 0, tl.float32)
    # One K buffer overlaps refill with register consumers.
    if MAX_KV > 0:
        kb = tlx.local_alloc((KL, 128), tl.bfloat16, 1)
        first_columns = ks + tl.arange(0, KL)
        first_offsets = (first_columns[:, None] * H + head) * 128 + d[None, :]
        first_token = tlx.async_load(K + first_offsets, tlx.local_view(kb, 0), mask=(first_columns < ke)[:, None],
                                     other=0)
        tlx.async_load_commit_group([first_token])
        for tile in range(triton.cdiv(MAX_KV, KL)):
            columns = ks + tile * KL + tl.arange(0, KL)
            km = columns < ke
            # Wait before reading the active buffer.
            wait = tlx.async_load_wait_group(0)
            k = tlx.local_load(tlx.local_view(kb, 0), token=wait)
            # The final iteration starts no copy.
            if tile + 1 < triton.cdiv(MAX_KV, KL):
                next_columns = ks + (tile + 1) * KL + tl.arange(0, KL)
                next_offsets = (next_columns[:, None] * H + head) * 128 + d[None, :]
                next_token = tlx.async_load(K + next_offsets, tlx.local_view(kb, 0), mask=(next_columns < ke)[:, None],
                                            other=0)
                tlx.async_load_commit_group([next_token])
            k_fragment = tlx.require_layout(tl.trans(k), rhs_layout, pin=False)
            zero_score = tlx.zeros((BQ, KL), tl.float32, layout=score_layout)
            scores = tlx.release_layout(tl.dot(q_fragment, k_fragment, zero_score, allow_tf32=False))
            p = tl.where(ke - ks == 1, 1.0, tl.exp2(scores * (alpha * 1.44269504) - m[:, None]))
            dp = tlx.release_layout(tl.dot(do_fragment, k_fragment, zero_score, allow_tf32=False))
            ds = tl.where(qm[:, None] & km[None, :] & (ke - ks > 1), p * (dp - delta[:, None]), 0).to(tl.bfloat16)
            if H == 2 and MAX_KV == 1024:
                ds_fragment = tlx.require_layout(ds, lhs_layout, pin=False)
                k_dq_fragment = tlx.require_layout(k, rhs_layout, pin=False)
                dq = tl.dot(ds_fragment, k_dq_fragment, dq, allow_tf32=False)
            else:
                dq = tl.dot(ds, k, dq, allow_tf32=False)
    if H == 2 and MAX_KV == 1024:
        dq = tlx.release_layout(dq)
    tl.store(DQ + qo, dq * alpha, qm[:, None])


@triton.jit
def _compact_q_owner(Q, K, V, DO, M, OUT, DQ, qs, qe, ks, ke, head, task, alpha, H: tl.constexpr, MAX_KV: tl.constexpr,
                     SHARED: tl.constexpr, BQ: tl.constexpr, KL: tl.constexpr):
    rows = qs + task * BQ + tl.arange(0, BQ)
    d = tl.arange(0, 128)
    qm = rows < qe
    qo = (rows[:, None] * H + head) * 128 + d[None, :]
    q = tl.load(Q + qo, qm[:, None], other=0)
    do = tl.load(DO + qo, qm[:, None], other=0)
    out = tl.load(OUT + qo, qm[:, None], other=0)
    m = tl.load(M + rows * H + head, qm, other=0)
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), 1)
    score_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                     warps_per_cta=[1, 4])
    lhs_layout: tl.constexpr = tlx.dot_operand_layout(0, score_layout, k_width=4)
    rhs_layout: tl.constexpr = tlx.dot_operand_layout(1, score_layout, k_width=4)
    q_fragment = tlx.require_layout(q, lhs_layout, pin=False)
    do_fragment = tlx.require_layout(do, lhs_layout, pin=False)
    # A single softmax key has a zero score derivative.
    dq = tl.full((BQ, 128), 0, tl.float32)
    for tile in range(triton.cdiv(MAX_KV, KL)):
        columns = ks + tile * KL + tl.arange(0, KL)
        km = columns < ke
        ko = (columns[:, None] * H + head) * 128 + d[None, :]
        k = tl.load(K + ko, km[:, None], other=0)
        if not SHARED: v = tl.load(V + ko, km[:, None], other=0)
        k_fragment = tlx.require_layout(tl.trans(k), rhs_layout, pin=False)
        zero_score = tlx.zeros((BQ, KL), tl.float32, layout=score_layout)
        scores = tlx.release_layout(tl.dot(q_fragment, k_fragment, zero_score, allow_tf32=False))
        p = tl.where(ke - ks == 1, 1.0, tl.exp2(scores * (alpha * 1.44269504) - m[:, None]))
        if SHARED:
            v_fragment = k_fragment
        else:
            v_fragment = tlx.require_layout(tl.trans(v), rhs_layout, pin=False)
        dp = tlx.release_layout(tl.dot(do_fragment, v_fragment, zero_score, allow_tf32=False))
        ds = tl.where(qm[:, None] & km[None, :] & (ke - ks > 1), p * (dp - delta[:, None]), 0).to(tl.bfloat16)
        dq = tl.dot(ds, k, dq, allow_tf32=False)
    tl.store(DQ + qo, dq * alpha, qm[:, None])


@triton.jit
def _compact_kv_owner(Q, K, V, DO, M, OUT, DK, DV, qs, qe, ks, ke, head, task, alpha, H: tl.constexpr,
                      SHARED: tl.constexpr, BK: tl.constexpr, QL: tl.constexpr, WIDE: tl.constexpr = False):
    columns = ks + task * BK + tl.arange(0, BK)
    d = tl.arange(0, 128)
    km = columns < ke
    ko = (columns[:, None] * H + head) * 128 + d[None, :]
    k = tl.load(K + ko, km[:, None], other=0)
    if SHARED: v = k
    else: v = tl.load(V + ko, km[:, None], other=0)
    # Distribute the two KV row groups across two warps.
    score_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                     warps_per_cta=([2, 4] if WIDE else [2, 2]))
    lhs_layout: tl.constexpr = tlx.dot_operand_layout(0, score_layout, k_width=4)
    rhs_layout: tl.constexpr = tlx.dot_operand_layout(1, score_layout, k_width=4)
    k_fragment = tlx.require_layout(k, lhs_layout, pin=False)
    v_fragment = tlx.require_layout(v, lhs_layout, pin=False)
    # Preserve dV while clearing the single-key score derivative.
    if WIDE:
        dk = tlx.zeros((BK, 128), tl.float32, layout=score_layout)
    else:
        dk = tl.full((BK, 128), 0, tl.float32)
    if WIDE:
        dv = tlx.zeros((BK, 128), tl.float32, layout=score_layout)
    else:
        dv = tl.full((BK, 128), 0, tl.float32)
    for tile in range(triton.cdiv(256, QL)):
        rows = qs + tile * QL + tl.arange(0, QL)
        qm = rows < qe
        qo = (rows[:, None] * H + head) * 128 + d[None, :]
        q = tl.load(Q + qo, qm[:, None], other=0)
        do = tl.load(DO + qo, qm[:, None], other=0)
        out = tl.load(OUT + qo, qm[:, None], other=0)
        m = tl.load(M + rows * H + head, qm, other=0)
        delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), 1)
        valid = km[:, None] & qm[None, :]
        q_fragment = tlx.require_layout(tl.trans(q), rhs_layout, pin=False)
        do_fragment = tlx.require_layout(tl.trans(do), rhs_layout, pin=False)
        zero_score = tlx.zeros((BK, QL), tl.float32, layout=score_layout)
        scores = tlx.release_layout(tl.dot(k_fragment, q_fragment, zero_score, allow_tf32=False))
        p = tl.where(valid, tl.where(ke - ks == 1, 1.0, tl.exp2(scores * (alpha * 1.44269504) - m[None, :])), 0)
        dp = tlx.release_layout(tl.dot(v_fragment, do_fragment, zero_score, allow_tf32=False))
        ds = tl.where(valid & (ke - ks > 1), p * (dp - delta[None, :]), 0).to(tl.bfloat16)
        if WIDE:
            p_fragment = tlx.require_layout(p.to(tl.bfloat16), lhs_layout, pin=False)
            do_dv_fragment = tlx.require_layout(do, rhs_layout, pin=False)
            dv = tl.dot(p_fragment, do_dv_fragment, dv, allow_tf32=False)
        else:
            dv = tl.dot(p.to(tl.bfloat16), do, dv, allow_tf32=False)
        if WIDE:
            ds_fragment = tlx.require_layout(ds, lhs_layout, pin=False)
            q_dk_fragment = tlx.require_layout(q, rhs_layout, pin=False)
            dk = tl.dot(ds_fragment, q_dk_fragment, dk, allow_tf32=False)
        else:
            dk = tl.dot(ds, q, dk, allow_tf32=False)
    if WIDE:
        dk = tlx.release_layout(dk)
        dv = tlx.release_layout(dv)
    dk = dk * alpha
    if SHARED: dk += dv
    tl.store(DK + ko, dk, km[:, None])
    if not SHARED: tl.store(DV + ko, dv, km[:, None])


@triton.jit
def _compact_bwd_impl(Q, K, V, DO, M, OUT, SOQ, SOK, DQ, DK, DV, alpha, H: tl.constexpr, MAX_KV: tl.constexpr,
                      SHARED: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, QL: tl.constexpr, KL: tl.constexpr):
    bh = tl.program_id(0)
    task = tl.program_id(1)
    batch = bh // H
    head = bh % H
    qs = tl.load(SOQ + batch).to(tl.int64)
    qe = tl.load(SOQ + batch + 1).to(tl.int64)
    ks = tl.load(SOK + batch).to(tl.int64)
    ke = tl.load(SOK + batch + 1).to(tl.int64)
    if task < triton.cdiv(256, BQ):
        if SHARED and MAX_KV > 512:
            _compact_q_owner_staged(Q, K, DO, M, OUT, DQ, qs, qe, ks, ke, head, task, alpha, H, MAX_KV, BQ, KL)
        else:
            _compact_q_owner(Q, K, V, DO, M, OUT, DQ, qs, qe, ks, ke, head, task, alpha, H, MAX_KV, SHARED, BQ, KL)
    else:
        _compact_kv_owner(Q, K, V, DO, M, OUT, DK, DV, qs, qe, ks, ke, head, task - triton.cdiv(256, BQ), alpha, H,
                          SHARED, BK, QL, WIDE=SHARED and H == 2 and MAX_KV == 1024)


@triton.jit
def _compact_bwd_shared(Q, K, DO, M, OUT, SOQ, SOK, DQ, DK, alpha, H: tl.constexpr, MAX_KV: tl.constexpr):
    _compact_bwd_impl(Q, K, K, DO, M, OUT, SOQ, SOK, DQ, DK, DK, alpha, H, MAX_KV, True, 16, 32, 64, 128)


@triton.jit
def _compact_bwd_separate(Q, K, V, DO, M, OUT, SOQ, SOK, DQ, DK, DV, alpha, H: tl.constexpr, MAX_KV: tl.constexpr):
    _compact_bwd_impl(Q, K, V, DO, M, OUT, SOQ, SOK, DQ, DK, DV, alpha, H, MAX_KV, False, 16, 32, 64, 128)


# Two warp groups accumulate separate KV halves in FP32.
@triton.jit
def _compact_q_owner_split(Q, K, DO, M, OUT, DQ, qs, qe, ks, ke, task, alpha):
    rows = qs + task * 16 + tl.arange(0, 16)
    dims = tl.arange(0, 128)
    qm = rows < qe
    qo = rows[:, None] * 128 + dims[None, :]
    q = tl.load(Q + qo, qm[:, None], other=0)
    do = tl.load(DO + qo, qm[:, None], other=0)
    out = tl.load(OUT + qo, qm[:, None], other=0)
    m = tl.load(M + rows, qm, other=0)
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), 1)
    mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                            warps_per_cta=[2, 1, 4])
    lhs: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    rhs: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    q_fragment = tlx.require_layout(tl.broadcast_to(q[None, :, :], (2, 16, 128)), lhs, pin=False)
    do_fragment = tlx.require_layout(tl.broadcast_to(do[None, :, :], (2, 16, 128)), lhs, pin=False)
    dq = tlx.zeros((2, 16, 128), tl.float32, layout=mma)
    groups = tl.arange(0, 2)
    for tile in tl.range(0, 8, loop_unroll_factor=2):
        columns = ks + groups[:, None] * 1024 + tile * 128 + tl.arange(0, 128)[None, :]
        km = columns < ke
        ko = columns[:, :, None] * 128 + dims[None, None, :]
        k = tl.load(K + ko, km[:, :, None], other=0)
        kt_fragment = tlx.require_layout(k.permute(0, 2, 1), rhs, pin=False)
        zero_score = tlx.zeros((2, 16, 128), tl.float32, layout=mma)
        scores = tlx.release_layout(tl.dot(q_fragment, kt_fragment, zero_score, allow_tf32=False))
        p = tl.where(ke - ks == 1, 1.0, tl.exp2(scores * (alpha * 1.44269504) - m[None, :, None]))
        dp = tlx.release_layout(tl.dot(do_fragment, kt_fragment, zero_score, allow_tf32=False))
        ds = tl.where(qm[None, :, None] & km[:, None, :] & (ke - ks > 1), p * (dp - delta[None, :, None]),
                      0).to(tl.bfloat16)
        ds_fragment = tlx.require_layout(ds, lhs, pin=False)
        k_fragment = tlx.require_layout(k, rhs, pin=False)
        dq = tl.dot(ds_fragment, k_fragment, dq, allow_tf32=False)
    result = tl.sum(tlx.release_layout(dq), 0)
    tl.store(DQ + qo, result * alpha, qm[:, None])


@triton.jit
def _compact_bwd_shared_split(Q, K, DO, M, OUT, SOQ, SOK, DQ, DK, alpha, H: tl.constexpr, MAX_KV: tl.constexpr):
    batch = tl.program_id(0)
    task = tl.program_id(1)
    qs = tl.load(SOQ + batch).to(tl.int64)
    qe = tl.load(SOQ + batch + 1).to(tl.int64)
    ks = tl.load(SOK + batch).to(tl.int64)
    ke = tl.load(SOK + batch + 1).to(tl.int64)
    if task < 16:
        _compact_q_owner_split(Q, K, DO, M, OUT, DQ, qs, qe, ks, ke, task, alpha)
    else:
        _compact_kv_owner(Q, K, K, DO, M, OUT, DK, DK, qs, qe, ks, ke, 0, task - 16, alpha, 1, True, 32, 64, WIDE=True)


_COMPACT_SPLIT_OPTIONS = dict(_SMALL_OPTIONS, num_warps=8)

_COMPACT_H2_1024_OPTIONS = dict(_SMALL_OPTIONS, num_warps=8)


def compact_softmax_backward(q, k, v, dout, m, out, offsets_q, offsets_kv, max_kv, alpha, shared_kv):
    dout = dout.contiguous()
    heads = q.shape[1]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = None if shared_kv else torch.empty_like(v)
    grid = ((offsets_q.numel() - 1) * heads, 16 + triton.cdiv(max_kv, 32), 1)
    if shared_kv:
        if heads == 1 and max_kv == 2048 and offsets_q.numel() == 5:
            _compact_bwd_shared_split[grid](q, k, dout, m, out, offsets_q, offsets_kv, dq, dk, alpha, heads, max_kv,
                                            **_COMPACT_SPLIT_OPTIONS)
        elif heads == 2 and max_kv == 1024:
            _compact_bwd_shared[grid](q, k, dout, m, out, offsets_q, offsets_kv, dq, dk, alpha, heads, max_kv,
                                      **_COMPACT_H2_1024_OPTIONS)
        else:
            _compact_bwd_shared[grid](q, k, dout, m, out, offsets_q, offsets_kv, dq, dk, alpha, heads, max_kv,
                                      **_SMALL_OPTIONS)
    else:
        _compact_bwd_separate[grid](q, k, v, dout, m, out, offsets_q, offsets_kv, dq, dk, dv, alpha, heads, max_kv,
                                    **_SMALL_OPTIONS)
    return dq, dk, dv
