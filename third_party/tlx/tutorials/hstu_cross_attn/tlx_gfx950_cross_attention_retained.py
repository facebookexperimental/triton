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
"""Retain FP32 dQ in LDS for shared-KV gfx950 softmax backward.

The caller must enforce the packed H1, BF16, D128, and Q<=256 contract.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _compute_softmax_delta(OUT, DO, DELTA, TQ):
    B: tl.constexpr = 64
    r = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, 128)
    o = tl.load(OUT + r[:, None] * 128 + d[None, :], r[:, None] < TQ, 0).to(tl.float32)
    do = tl.load(DO + r[:, None] * 128 + d[None, :], r[:, None] < TQ, 0).to(tl.float32)
    tl.store(DELTA + r, tl.sum(o * do, 1), r < TQ)


@triton.jit
def _retained_softmax_backward(Q, K, DO, LOGSUMEXP2, DELTA, Q_OFFSETS, KV_OFFSETS, DKV, DQ, alpha,
                               FIXED_Q: tl.constexpr = False):
    BLOCK_Q: tl.constexpr = 16
    BLOCK_KV: tl.constexpr = 256
    z = tl.program_id(0)
    if FIXED_Q:
        qs = z.to(tl.int64) * 256
        qe = qs + 256
    else:
        qs = tl.load(Q_OFFSETS + z)
        qe = tl.load(Q_OFFSETS + z + 1)
    ks = tl.load(KV_OFFSETS + z)
    ke = tl.load(KV_OFFSETS + z + 1)
    dq_layout: tl.constexpr = tlx.swizzled_shared_layout_encoding(4, 1, 8, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0])
    dq_cache = tlx.local_alloc((BLOCK_Q, 128), tl.float32, 256 // BLOCK_Q, layout=dq_layout)
    for ki in range(tl.cdiv(ke - ks, BLOCK_KV)):
        n = ks + ki * BLOCK_KV + tl.arange(0, BLOCK_KV)
        d = tl.arange(0, 128)
        score_mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                      warps_per_cta=[4, 1], tiles_per_warp=[2, 1])
        score_a: tl.constexpr = tlx.dot_operand_layout(0, score_mma, k_width=8)
        score_b: tl.constexpr = tlx.dot_operand_layout(1, score_mma, k_width=8)
        dq_parent: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                      warps_per_cta=[1, 4])
        dq_b: tl.constexpr = tlx.dot_operand_layout(1, dq_parent, k_width=4)
        k = tl.load(K + n[:, None] * 128 + d[None, :], n[:, None] < ke, 0)
        k = tlx.release_layout(tlx.require_layout(k, score_a))
        k_shared: tl.constexpr = tlx.swizzled_shared_layout_encoding(8, 1, 8, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0])
        # Reuse one 32 KiB buffer for the two K halves.
        kbuf = tlx.local_alloc((128, 128), tl.bfloat16, 1, layout=k_shared)
        kv = tlx.local_view(kbuf, 0)
        k_half_load: tl.constexpr = tlx.layout(shape=((16, 4, 4), (4, 8, 2)), stride=((1, 512, 16), (128, 2048, 64)))
        ksl = tlx.require_layout(k, score_a)
        kh0 = tlx.extract_slice(ksl, [128, 128], [0, 0])
        tlx.local_store(kv, kh0)
        kr0 = tlx.require_layout(tlx.local_load(kv, layout=k_half_load), k_half_load)
        kh1 = tlx.extract_slice(ksl, [128, 128], [128, 0])
        tlx.local_store(kv, kh1)
        kr1 = tlx.require_layout(tlx.local_load(kv, layout=k_half_load), k_half_load)
        kq = tl.reshape(tl.permute(tl.join(kr0, kr1), (2, 0, 1)), (256, 128))
        kq = tlx.release_layout(tlx.require_layout(kq, dq_b))
        mfma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 16], transposed=True,
                                                 warps_per_cta=[4, 1])
        al: tl.constexpr = tlx.dot_operand_layout(0, mfma, k_width=8)
        bl: tl.constexpr = tlx.dot_operand_layout(1, mfma, k_width=8)
        acc0 = tlx.zeros((BLOCK_KV, 32), tl.float32, layout=mfma)
        acc1 = tlx.zeros((BLOCK_KV, 32), tl.float32, layout=mfma)
        acc2 = tlx.zeros((BLOCK_KV, 32), tl.float32, layout=mfma)
        acc3 = tlx.zeros((BLOCK_KV, 32), tl.float32, layout=mfma)
        # Match both Q/dO operand layouts without an intermediate register layout.
        qdo_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [(512, 32), (1024, 16)],
            [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [0, 64], [4, 0], [2, 0], [8, 0]], [BLOCK_Q, 128])
        qb = tlx.local_alloc((BLOCK_Q, 128), tl.bfloat16, 1, layout=qdo_layout)
        db = tlx.local_alloc((BLOCK_Q, 128), tl.bfloat16, 1, layout=qdo_layout)
        qv = tlx.local_view(qb, 0)
        dv = tlx.local_view(db, 0)
        first = qs + tl.arange(0, BLOCK_Q)
        qt = tlx.async_load(Q + first[:, None] * 128 + d[None, :], qv, mask=FIXED_Q | (first[:, None] < qe), other=0)
        dt = tlx.async_load(DO + first[:, None] * 128 + d[None, :], dv, mask=FIXED_Q | (first[:, None] < qe), other=0)
        tlx.async_load_commit_group([qt, dt])
        ds_shared: tl.constexpr = tlx.swizzled_shared_layout_encoding(4, 1, 16, [0, 1], [1, 1], [1, 1], [1, 1], [0, 1])
        dsbuf = tlx.local_alloc((BLOCK_KV, BLOCK_Q), tl.bfloat16, 1, layout=ds_shared)
        dsv = tlx.local_view(dsbuf, 0)
        ds_load: tl.constexpr = tlx.layout(shape=((16, 4, 4), (4, 16)),
                                           stride=((1, 4 * BLOCK_Q, 0), (BLOCK_Q, 16 * BLOCK_Q)))
        for start in range(0, qe - qs, BLOCK_Q):
            m = qs + start + tl.arange(0, BLOCK_Q)
            wait = tlx.async_load_wait_group(0)
            q = tlx.require_layout(tlx.local_load(qv, token=wait, layout=bl), bl)
            do = tlx.require_layout(tlx.local_load(dv, token=wait, layout=bl), bl)
            q_t = tlx.require_layout(tlx.local_load(tlx.local_trans(qv), token=wait, layout=score_b), score_b)
            do_t = tlx.require_layout(tlx.local_load(tlx.local_trans(dv), token=wait, layout=score_b), score_b)
            lse = tl.load(LOGSUMEXP2 + m, FIXED_Q | (m < qe), 0)
            delta = tl.load(DELTA + m, FIXED_Q | (m < qe), 0)
            nxt = m + BLOCK_Q
            qt = tlx.async_load(Q + nxt[:, None] * 128 + d[None, :], qv, mask=nxt[:, None] < qe, other=0)
            dt = tlx.async_load(DO + nxt[:, None] * 128 + d[None, :], dv, mask=nxt[:, None] < qe, other=0)
            tlx.async_load_commit_group([qt, dt])
            score = tl.dot(k, q_t, allow_tf32=False)
            valid = (n[:, None] < ke) & (FIXED_Q | (m[None, :] < qe))
            p = tl.where(valid, tl.exp2(score * (alpha * 1.44269502) - lse[None, :]), 0)
            dp = tl.dot(k, do_t, allow_tf32=False)
            ds = tl.where(valid, p * (dp - delta[None, :]), 0).to(tl.bfloat16)
            tlx.local_store(dsv, ds)
            pl = tlx.require_layout(p.to(tl.bfloat16), al, pin=False)
            dsl = tlx.require_layout(ds, al, pin=False)
            ql = tlx.require_layout(q, bl, pin=False)
            dol = tlx.require_layout(do, bl, pin=False)
            q0 = tlx.extract_slice(ql, [BLOCK_Q, 32], [0, 0])
            do0 = tlx.extract_slice(dol, [BLOCK_Q, 32], [0, 0])
            acc0 = tl.dot(pl, do0, acc0, allow_tf32=False)
            dk0 = tl.dot(dsl, q0, tlx.zeros((BLOCK_KV, 32), tl.float32, layout=mfma), allow_tf32=False)
            acc0 += dk0 * alpha
            q1 = tlx.extract_slice(ql, [BLOCK_Q, 32], [0, 32])
            do1 = tlx.extract_slice(dol, [BLOCK_Q, 32], [0, 32])
            acc1 = tl.dot(pl, do1, acc1, allow_tf32=False)
            dk1 = tl.dot(dsl, q1, tlx.zeros((BLOCK_KV, 32), tl.float32, layout=mfma), allow_tf32=False)
            acc1 += dk1 * alpha
            q2 = tlx.extract_slice(ql, [BLOCK_Q, 32], [0, 64])
            do2 = tlx.extract_slice(dol, [BLOCK_Q, 32], [0, 64])
            acc2 = tl.dot(pl, do2, acc2, allow_tf32=False)
            dk2 = tl.dot(dsl, q2, tlx.zeros((BLOCK_KV, 32), tl.float32, layout=mfma), allow_tf32=False)
            acc2 += dk2 * alpha
            dq_mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                       warps_per_cta=[1, 4])
            dq_a: tl.constexpr = tlx.dot_operand_layout(0, dq_mma, k_width=4)
            # Read dS before the final dKV fragment to overlap LDS latency.
            ds_for_q = tlx.require_layout(tlx.local_load(dsv, layout=ds_load), ds_load)
            lhs = tlx.require_layout(tl.trans(ds_for_q), dq_a, pin=False)
            q3 = tlx.extract_slice(ql, [BLOCK_Q, 32], [0, 96])
            do3 = tlx.extract_slice(dol, [BLOCK_Q, 32], [0, 96])
            acc3 = tl.dot(pl, do3, acc3, allow_tf32=False)
            dk3 = tl.dot(dsl, q3, tlx.zeros((BLOCK_KV, 32), tl.float32, layout=mfma), allow_tf32=False)
            acc3 += dk3 * alpha
            rhs = tlx.require_layout(kq, dq_b, pin=False)
            part = tl.dot(lhs, rhs, tlx.zeros((BLOCK_Q, 128), tl.float32, layout=dq_mma), allow_tf32=False) * alpha
            part = tlx.require_layout(part, dq_mma)
            dq_view = tlx.local_view(dq_cache, (start // BLOCK_Q).to(tl.int32))
            if ki != 0:
                prev = tlx.local_load(dq_view)
            else:
                prev = tl.zeros((BLOCK_Q, 128), tl.float32)
            prev = tlx.require_layout(prev, dq_mma)
            total = prev + part
            if (ki + 1) * BLOCK_KV >= ke - ks:
                tl.store(DQ + m[:, None] * 128 + d[None, :], total, FIXED_Q | (m[:, None] < qe))
            else:
                tlx.local_store(dq_view, total)
        tlx.async_load_wait_group(0)
        dc = tl.arange(0, 32)
        tl.store(DKV + n[:, None] * 128 + dc[None, :] + 0, tlx.release_layout(acc0), n[:, None] < ke)
        tl.store(DKV + n[:, None] * 128 + dc[None, :] + 32, tlx.release_layout(acc1), n[:, None] < ke)
        tl.store(DKV + n[:, None] * 128 + dc[None, :] + 64, tlx.release_layout(acc2), n[:, None] < ke)
        tl.store(DKV + n[:, None] * 128 + dc[None, :] + 96, tlx.release_layout(acc3), n[:, None] < ke)
    if ke == ks:
        for start in range(0, qe - qs, BLOCK_Q):
            m = qs + start + tl.arange(0, BLOCK_Q)
            d = tl.arange(0, 128)
            tl.store(DQ + m[:, None] * 128 + d[None, :], 0, FIXED_Q | (m[:, None] < qe))


@triton.jit
def _correct_retained_single_key(DO, Q_OFFSETS, KV_OFFSETS, DQ, DKV):
    z = tl.program_id(0)
    ks = tl.load(KV_OFFSETS + z)
    ke = tl.load(KV_OFFSETS + z + 1)
    if ke - ks == 1:
        qs = tl.load(Q_OFFSETS + z)
        qe = tl.load(Q_OFFSETS + z + 1)
        d = tl.arange(0, 128)
        dkv = tl.zeros((128, ), tl.float32)
        for start in range(0, qe - qs, 32):
            m = qs + start + tl.arange(0, 32)
            do = tl.load(DO + m[:, None] * 128 + d[None, :], m[:, None] < qe, 0).to(tl.float32)
            dkv += tl.sum(do, axis=0)
            tl.store(DQ + m[:, None] * 128 + d[None, :], 0, m[:, None] < qe)
        tl.store(DKV + ks * 128 + d, dkv)


def retained_softmax_backward(q, k, dout, logsumexp2, out, offsets_q, offsets_kv, alpha):
    """Compute packed shared-KV gradients with FP32 recurrence."""
    total_q = q.shape[0]
    batch = offsets_q.numel() - 1
    dout = dout.contiguous()
    delta = torch.empty((total_q, ), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dkv = torch.empty_like(k)
    _compute_softmax_delta[(triton.cdiv(total_q, 64), )](out, dout, delta, total_q)
    # Packed lengths are at most 256, so this equality proves every query length is 256.
    _retained_softmax_backward[(batch, )](q, k, dout, logsumexp2, delta, offsets_q, offsets_kv, dkv, dq, alpha,
                                          FIXED_Q=total_q == batch * 256, num_warps=4, num_stages=1,
                                          matrix_instr_nonkdim=16, waves_per_eu=1,
                                          regclass_priority_trumps_globalness=False)
    # Restore exact one-key gradients after the ordinary kernel completes.
    _correct_retained_single_key[(batch, )](dout, offsets_q, offsets_kv, dq, dkv, num_warps=4)
    return dq, dkv
